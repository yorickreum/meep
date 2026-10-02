#!/usr/bin/env bash
#
# Build Meep's native dependencies for a binary wheel.
#
# Run as cibuildwheel's `before-all` step: it installs the C libraries that are
# packaged for the platform, then builds into $MEEP_DEPS_PREFIX the ones that
# are not: HDF5 (the packaged builds are serial; Meep needs MPI-IO for
# collective output) and the three NanoComp siblings, libctl, harminv and MPB.
#
# Everything is built shared, because auditwheel/delocate vendor whatever ends
# up linked into _meep.so.
#
# On Windows this runs in MSYS2's UCRT64 shell. There the build is serial (no
# MPICH, packaged serial HDF5), and libctl and harminv need the same
# adjustments as in .github/workflows/build-ci-windows.yml.
#
# Nothing here needs Guile: the wheel has no Scheme interface, and libctl only
# needs Guile to generate utils/geom-ctl-io.c, which the release tarball ships
# ready-made. Hence tarballs below rather than Git checkouts.
#
# Environment variables:
#   MEEP_DEPS_PREFIX        install prefix (default /usr/local)
#   MEEP_SKIP_SYSTEM_DEPS   do not touch the system package manager
#   WORKDIR                 where tarballs are unpacked

set -euo pipefail

PREFIX="${MEEP_DEPS_PREFIX:-/usr/local}"
case "$(uname -s)" in
  MINGW* | MSYS*)
    MINGW=1
    PREFIX="$(cygpath -u "${PREFIX}")" # C:/x would split PATH at its colon
    ;;
  *) MINGW= ;;
esac
LIBCTL_VERSION="${LIBCTL_VERSION:-4.7.1}"   # >= 4.7.0 for mesh geometry (configure.ac)
HARMINV_VERSION="${HARMINV_VERSION:-1.4.3}"
MPB_VERSION="${MPB_VERSION:-1.12.0}"
MPICH_VERSION="${MPICH_VERSION:-5.0.1.post1}"
HDF5_VERSION="${HDF5_VERSION:-1.14.6}"
PATCHELF_VERSION="${PATCHELF_VERSION:-0.19.1.0}"

WORKDIR="${WORKDIR:-$(mktemp -d)}"
NPROC="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) )"

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${PREFIX}/bin:${PATH}"

log() { printf '\n=== %s ===\n' "$*"; }

# gfortran is not optional: LAPACK is Fortran and harminv's configure runs
# AC_F77_WRAPPERS. Meep's own workflow never names it only because the runners
# ship it preinstalled.
install_system_deps_linux() {
  if [ -n "${MEEP_SKIP_SYSTEM_DEPS:-}" ]; then
    log "skipping system libraries (MEEP_SKIP_SYSTEM_DEPS set)"
    return
  fi

  if command -v dnf >/dev/null; then
    log "installing system libraries (dnf)"
    dnf -y install epel-release
    # CRB/PowerTools carries the -devel packages for several of these on EL8/9.
    dnf -y config-manager --set-enabled powertools 2>/dev/null \
      || dnf -y config-manager --set-enabled crb 2>/dev/null \
      || true
    dnf -y install \
      autoconf automake libtool pkgconfig swig ccache \
      gcc-gfortran \
      fftw-devel \
      gsl-devel \
      openblas-devel \
      lapack-devel \
      libpng-devel \
      zlib-devel
  elif command -v apt-get >/dev/null; then
    # Not a manylinux target: this branch is for building and testing the wheel
    # machinery on a plain Debian/Ubuntu box (including WSL).
    log "installing system libraries (apt-get)"
    apt-get -y update
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
      build-essential \
      autoconf automake libtool pkg-config swig ccache \
      gfortran \
      libfftw3-dev \
      libgsl-dev \
      liblapack-dev \
      libpng-dev \
      zlib1g-dev \
      python3-dev
  else
    echo "no supported package manager found; set MEEP_SKIP_SYSTEM_DEPS=1 and install by hand" >&2
    exit 1
  fi
}

# The workflow installs these with msys2/setup-msys2 already, which makes this a
# no-op there; it is for a local build. No `pacman -Syu`: updating the MSYS2
# runtime kills the shell running this script.
install_system_deps_mingw() {
  if [ -n "${MEEP_SKIP_SYSTEM_DEPS:-}" ]; then
    log "skipping system libraries (MEEP_SKIP_SYSTEM_DEPS set)"
    return
  fi
  log "installing system libraries (pacman)"
  local p=mingw-w64-ucrt-x86_64
  pacman -S --needed --noconfirm \
    make m4 curl tar autoconf-wrapper autoconf2.72 automake-wrapper automake1.18 libtool \
    $p-gcc $p-gcc-fortran $p-pkgconf $p-swig \
    $p-hdf5 $p-gsl $p-fftw $p-openblas
}

install_system_deps_macos() {
  if [ -n "${MEEP_SKIP_SYSTEM_DEPS:-}" ]; then
    log "skipping system libraries (MEEP_SKIP_SYSTEM_DEPS set)"
    return
  fi
  log "installing system libraries (brew)"
  # `gcc` is what provides gfortran on macOS.
  brew install autoconf automake libtool pkg-config swig ccache gcc fftw gsl libpng
}

# fetch_and_build <repo> <version> [configure args...]
fetch_and_build() {
  local repo="$1" version="$2"; shift 2
  local name="${repo}-${version}"
  local url="https://github.com/NanoComp/${repo}/releases/download/v${version}/${name}.tar.gz"

  log "building ${name}"
  mkdir -p "${WORKDIR}"
  if [ ! -d "${WORKDIR}/${name}" ]; then
    curl -sSL --fail -o "${WORKDIR}/${name}.tar.gz" "${url}"
    tar xzf "${WORKDIR}/${name}.tar.gz" -C "${WORKDIR}"
  fi

  # Without pass_all, libtool on Windows looks for a shared -lmingw32, finds
  # none, and silently builds a static archive instead of a DLL.
  pushd "${WORKDIR}/${name}" >/dev/null
  ./configure --prefix="${PREFIX}" --enable-shared --disable-static \
    ${MINGW:+lt_cv_deplibs_check_method=pass_all} "$@"
  make -j"${NPROC}" ${MAKE_VARS:+"${MAKE_VARS}"}
  make install ${MAKE_VARS:+"${MAKE_VARS}"}
  popd >/dev/null
}

# MPICH comes from its PyPI wheel (mpicc, mpicxx, mpiexec, mpi.h,
# libmpi.so.12), the same runtime end users get from `pip install mpich`:
# libmpi is never vendored into the wheel, so the ABI has to match the
# launcher's. Any interpreter with pip will do, but the manylinux images'
# /usr/bin/python3 has none, hence /opt/python first.
find_pip_python() {
  local candidate
  for candidate in /opt/python/cp31*/bin/python python3 python; do
    candidate=$(command -v "${candidate}" 2>/dev/null || echo "${candidate}")
    if [ -x "${candidate}" ] && "${candidate}" -m pip --version >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "no python with pip found" >&2
  return 1
}

# manylinux_2_28 pins patchelf 0.17.2, which miscomputes segment alignment when
# it has to grow a binary: renaming a SONAME there yields a library the loader
# rejects with "ELF load command address/offset not properly aligned". Fixed
# upstream in 0.18. auditwheel shells out to whatever patchelf is on PATH, so
# installing a current one ahead of the system copy fixes both it and setup.py.
install_patchelf() {
  local py
  py=$(find_pip_python) || exit 1
  log "installing patchelf ${PATCHELF_VERSION} into ${PREFIX} (image ships $(patchelf --version 2>&1))"
  "${py}" -m pip install --quiet --prefix="${PREFIX}" "patchelf==${PATCHELF_VERSION}"
  hash -r
  patchelf --version
}

# The packaged HDF5 is built without MPI, which leaves Meep writing output one
# rank at a time through h5file.cpp's exclusive-access path. --enable-parallel
# gives it H5Pset_fapl_mpio and therefore collective I/O, and needs mpicc, so
# this runs after install_mpich.
install_hdf5() {
  local name="hdf5-${HDF5_VERSION}"
  local url="https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/${name}.tar.gz"

  log "building ${name} (parallel)"
  mkdir -p "${WORKDIR}"
  if [ ! -d "${WORKDIR}/${name}" ]; then
    curl -sSL --fail -o "${WORKDIR}/${name}.tar.gz" "${url}"
    tar xzf "${WORKDIR}/${name}.tar.gz" -C "${WORKDIR}"
  fi

  pushd "${WORKDIR}/${name}" >/dev/null
  # C only: --enable-parallel rules out the C++ bindings, and Meep uses neither
  # those nor Fortran.
  CC="${PREFIX}/bin/mpicc" ./configure \
    --prefix="${PREFIX}" \
    --enable-shared --disable-static \
    --enable-parallel \
    --disable-cxx --disable-fortran \
    --disable-tests --disable-tools
  make -j"${NPROC}"
  make install
  popd >/dev/null

  # A serial HDF5 here would be a silent performance regression, not a build
  # failure, so make it a build failure.
  if ! grep -q "define H5_HAVE_PARALLEL 1" "${PREFIX}/include/H5pubconf.h"; then
    echo "the HDF5 build is not parallel; check ${WORKDIR}/${name}/config.log" >&2
    exit 1
  fi
}

install_mpich() {
  local py
  py=$(find_pip_python) || exit 1

  log "installing MPICH from the PyPI wheel into ${PREFIX} (using ${py})"
  "${py}" -m pip install --quiet --prefix="${PREFIX}" "mpich==${MPICH_VERSION}"

  if [ ! -x "${PREFIX}/bin/mpicc" ]; then
    echo "the mpich wheel did not provide ${PREFIX}/bin/mpicc" >&2
    exit 1
  fi
  "${PREFIX}/bin/mpicc" -show || true
}

main_mingw() {
  install_system_deps_mingw

  # libctlgeom must be a DLL: libmeep and libpympb both link it, and a static
  # copy in each gives duplicate symbols. Its Makefile.am lacks -no-undefined,
  # so supply it on the make command line.
  MAKE_VARS="libctlgeom_la_LDFLAGS=-no-undefined -avoid-version" \
    fetch_and_build libctl "${LIBCTL_VERSION}" --without-guile --without-python

  # harminv lacks -no-undefined too, but only libmeep links it, so a static
  # archive does.
  fetch_and_build harminv "${HARMINV_VERSION}" --disable-shared --enable-static \
    --with-blas=openblas --with-lapack=openblas

  # MPB without HDF5, as in build-ci-windows.yml; Meep does its own HDF5 I/O.
  fetch_and_build mpb "${MPB_VERSION}" --without-libctl --without-hdf5 \
    --with-hermitian-eps

  log "dependency prefix contents"
  ls -1 "${PREFIX}/bin" "${PREFIX}/lib" 2>/dev/null | sort -u | head -40 || true
}

main() {
  case "$(uname -s)" in
    Linux)  install_system_deps_linux ;;
    Darwin) install_system_deps_macos ;;
    MINGW* | MSYS*) main_mingw; return ;;
    *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
  esac

  # Before anything is patched: both setup.py and auditwheel need a good one.
  if [ "$(uname -s)" = Linux ]; then
    install_patchelf
  fi

  # Before the libraries that link them.
  install_mpich
  install_hdf5

  # Meep's Python build needs only libctlgeom from libctl; the Scheme half is
  # gated on --with-scheme in Meep's configure.ac.
  fetch_and_build libctl "${LIBCTL_VERSION}" --without-guile
  fetch_and_build harminv "${HARMINV_VERSION}"

  # --without-libctl drops Guile and the mpb executable and leaves the libmpb
  # C library, which is all Meep links against (AC_CHECK_LIB(mpb, ...)).
  fetch_and_build mpb "${MPB_VERSION}" --without-libctl --with-hermitian-eps LIBS=-ldl

  # `|| true` because there is no lib64 on every platform, and a failing ls
  # under `set -e` would take the whole cibuildwheel before-all step down.
  log "dependency prefix contents"
  ls -1 "${PREFIX}/lib" "${PREFIX}/lib64" 2>/dev/null | sort -u | head -40 || true
}

main "$@"
