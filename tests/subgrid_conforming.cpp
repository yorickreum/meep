/* Stability of the conforming interface rows, measured exactly.

   The interface is stepped by rows that are adjoint by
   construction, so one step conserves energy and its spectral radius, over
   every stored value (owned and ghost), is 1 to round-off -- also with
   subpixel averaging's anisotropic eps at the interface.  A time
   series cannot show that -- the diagonal energy beats, and growth can take
   longer than any horizon to appear -- so the step matrix is built by impulse
   response on small cells and its eigenvalues computed.  Without PML that
   state is closed, so the check is exact.

   With PML the state includes the PML auxiliaries; there a pulse must simply
   leave: a box whose faces lie on the PML's inner edge, PML on every side. */

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <meep.hpp>
#include "config.h"

using namespace meep;

#ifdef HAVE_LAPACK
extern "C" void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                       const int *lda, double *wr, double *wi, double *vl, const int *ldvl,
                       double *vr, const int *ldvr, double *work, const int *lwork, int *info);
#endif

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }

static std::unique_ptr<binary_partition> leaf(int &pid, int r) {
  return std::unique_ptr<binary_partition>(new binary_partition(pid++, r));
}

// refined in [lo, hi] along direction d and the ones after it, up to nd
static std::unique_ptr<binary_partition> region(int d, int nd, const double *lo, const double *hi,
                                                const double *size, int R, int &pid) {
  if (d == nd) return leaf(pid, R);
  std::unique_ptr<binary_partition> inner(region(d + 1, nd, lo, hi, size, R, pid));
  const direction dir = direction(d);
  if (hi[d] < size[d])
    inner.reset(new binary_partition(split_plane{dir, hi[d]}, std::move(inner), leaf(pid, 1)));
  if (lo[d] > 0)
    inner.reset(new binary_partition(split_plane{dir, lo[d]}, leaf(pid, 1), std::move(inner)));
  return inner;
}

#ifdef HAVE_LAPACK
// eps = 12 in a disk across the x = 0.5 face of the 2-D box below
static double disk_eps(const vec &p) {
  const double dx = p.x() - 0.5, dy = p.y() - 0.3;
  return dx * dx + dy * dy < 0.14 * 0.14 ? 12.0 : 1.0;
}

/* The largest |eigenvalue| of one step, over every stored value meep steps:
   E and H, and D and B where they are arrays of their own.  E is derived from
   D within the step, so the map's image is consistent and its spectrum is the
   physical one plus zeros.  With a Bloch phase real and imaginary parts
   couple, so both are poked; otherwise the imaginary one is an uncoupled copy. */
static double step_radius(fields &f, bool bloch = false) {
  struct value {
    int chunk;
    component c;
    int cmp;
    ptrdiff_t idx;
  };
  std::vector<value> all;
  for (int i = 0; i < f.num_chunks; ++i) {
    if (!f.chunks[i]->is_mine()) continue;
    const fields_chunk *fc = f.chunks[i];
    FOR_COMPONENTS(c) {
      if (c >= NUM_FIELD_COMPONENTS || !fc->f[c][0]) continue;
      if (!is_electric(c) && !is_magnetic(c) && !is_D(c) && !is_B(c)) continue;
      const component own = direction_component(is_D(c) ? Ex : Hx, component_direction(c));
      if ((is_D(c) || is_B(c)) && fc->f[c][0] == fc->f[own][0]) continue;
      for (int cmp = 0; cmp < (bloch ? 2 : 1); ++cmp)
        if (fc->f[c][cmp]) LOOP_OVER_VOL(fc->gv, c, n) {
            all.push_back(value{i, c, cmp, n});
          }
    }
  }
  const int N = (int)all.size();
  auto zero_all = [&]() {
    for (int i = 0; i < f.num_chunks; ++i)
      if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
          for (int cmp = 0; cmp < 2; ++cmp)
            if (f.chunks[i]->f[c][cmp])
              memset(f.chunks[i]->f[c][cmp], 0, sizeof(realnum) * f.chunks[i]->gv.ntot());
        }
  };
  std::vector<double> M((size_t)N * N);
  for (int k = 0; k < N; ++k) {
    zero_all();
    const value &v = all[k];
    f.chunks[v.chunk]->f[v.c][v.cmp][v.idx] = 1.0;
    f.step();
    for (int l = 0; l < N; ++l)
      M[(size_t)k * N + l] = f.chunks[all[l].chunk]->f[all[l].c][all[l].cmp][all[l].idx];
  }
  std::vector<double> wr(N), wi(N);
  int lwork = -1, info = 0, one = 1;
  double wq;
  dgeev_("N", "N", &N, M.data(), &N, wr.data(), wi.data(), NULL, &one, NULL, &one, &wq, &lwork,
         &info);
  lwork = (int)wq;
  std::vector<double> work(lwork);
  dgeev_("N", "N", &N, M.data(), &N, wr.data(), wi.data(), NULL, &one, NULL, &one, work.data(),
         &lwork, &info);
  if (info) meep::abort("dgeev failed (%d)", info);
  double rho = 0;
  for (int i = 0; i < N; ++i)
    rho = std::max(rho, std::hypot(wr[i], wi[i]));
  master_printf("    %d stored values\n", N);
  return rho;
}

/* A refined box on a small cell, stepped a few times to allocate every field,
   sources removed; then its step's spectral radius. */
static void spectral(const char *name, int nd, double res, const double *size, const double *lo,
                     const double *hi, int R, bool periodic, component src,
                     double (*eps)(const vec &) = one_eps, double kx = 0) {
  grid_volume gv = nd == 3 ? vol3d(size[0], size[1], size[2], res) : vol2d(size[0], size[1], res);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(region(0, nd, lo, hi, size, R, pid));
  // subpixel averaging makes eps anisotropic at the disk's edge
  structure s(gv, eps, no_pml(), identity(), 0, 0.5, eps != one_eps, DEFAULT_SUBPIXEL_TOL,
              DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  if (periodic) f.use_bloch(nd == 3 ? vec(kx, 0.0, 0.0) : vec(kx, 0.0));
  gaussian_src_time g(1.0, 0.5);
  f.add_point_source(src, g, nd == 3 ? vec(0.21, 0.23, 0.27) : vec(0.21, 0.23));
  for (int k = 0; k < 20; ++k)
    f.step();
  f.remove_sources();
  const double rho = step_radius(f, kx != 0);
  const bool ok = rho < 1 + 1e-9;
  master_printf("  %-44s rho - 1 = %+.2e  %s\n", name, rho - 1, ok ? "ok" : "FAILED");
  failures += !ok;
}
#endif

/* A pulse in a cell with PML on every side must leave, also when the refined
   box's faces lie on the PML's inner edge. */
static void pml_decay() {
  const double size[3] = {2.0, 1.0, 1.0}, lo[3] = {0.75, 0.25, 0.25}, hi[3] = {1.25, 0.75, 0.75};
  grid_volume gv = vol3d(size[0], size[1], size[2], 8);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(region(0, 3, lo, hi, size, 2, pid));
  structure s(gv, one_eps, pml(0.25), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
              DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  gaussian_src_time g(1.0, 0.5);
  f.add_point_source(Ez, g, vec(0.9, 0.45, 0.55));
  double peak = 0;
  while (f.time() < 10)
    f.step();
  peak = f.field_energy();
  while (f.time() < 80)
    f.step();
  const double last = f.field_energy(), ratio = last / peak;
  const bool ok = ratio < 1e-6;
  master_printf("  %-44s energy t=80 / t=10 = %.2e  %s\n", "3-D box on the PML edge, PML all round",
                ratio, ok ? "ok" : "FAILED");
  failures += !ok;
}

/* A lossy Lorentz disk across the box's upper face, where the rows derive E
   and step P themselves: the energy must go.  (The polarization is state the
   step matrix above cannot poke.) */
static double lorentz_sigma(const vec &p) {
  const double dx = p.x() - 0.62, dy = p.y() - 0.375;
  return dx * dx + dy * dy < 0.1 * 0.1 ? 4.0 : 0.0;
}
static void lorentz_decay() {
  const double size[2] = {1.0, 0.5}, lo[2] = {0.5, 0.125}, hi[2] = {0.75, 0.375};
  grid_volume gv = vol2d(size[0], size[1], 16);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(region(0, 2, lo, hi, size, 2, pid));
  structure s(gv, one_eps, no_pml(), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
              DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  simple_material_function sig(lorentz_sigma);
  s.add_susceptibility(sig, E_stuff, lorentzian_susceptibility(1.2, 0.1));
  fields f(&s);
  gaussian_src_time g(1.0, 0.5);
  f.add_point_source(Hz, g, vec(0.3, 0.21));
  while (f.time() < 20)
    f.step();
  const double peak = f.field_energy();
  while (f.time() < 1000)
    f.step();
  const double ratio = f.field_energy() / peak;
  const bool ok = ratio < 1e-8;
  master_printf("  %-44s energy t=1000 / t=20 = %.2e  %s\n",
                "2-D TE, lossy Lorentz disk across the upper face", ratio, ok ? "ok" : "FAILED");
  failures += !ok;
}

/* synchronize_magnetic_fields (energies, H outputs) must half-step the
   interface as step() does: H at t is then exactly the mean of H at t - dt/2
   and t + dt/2, at every stored point.  3-D box with PML. */
static std::vector<double> magnetic(fields &f) {
  std::vector<double> v;
  for (int i = 0; i < f.num_chunks; ++i)
    FOR_COMPONENTS(c) {
      if (c >= NUM_FIELD_COMPONENTS || (!is_magnetic(c) && !is_B(c))) continue;
      for (int cmp = 0; cmp < 2; ++cmp)
        if (const realnum *a = f.chunks[i]->f[c][cmp])
          for (size_t n = 0; n < f.chunks[i]->gv.ntot(); ++n)
            v.push_back(a[n]);
    }
  return v;
}
static void sync_check() {
  const double size[3] = {1, 1, 1}, lo[3] = {0.5, 0.125, 0.25}, hi[3] = {0.75, 0.375, 0.75};
  grid_volume gv = vol3d(1, 1, 1, 8);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(region(0, 3, lo, hi, size, 2, pid));
  structure s(gv, one_eps, pml(0.1), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
              DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  gaussian_src_time g(1.0, 0.5);
  f.add_point_source(Ez, g, vec(0.61, 0.23, 0.47));
  while (f.time() < 3)
    f.step();
  f.remove_sources();
  f.step();
  const std::vector<double> old = magnetic(f);
  f.synchronize_magnetic_fields();
  const std::vector<double> sync = magnetic(f);
  f.restore_magnetic_fields();
  f.step();
  const std::vector<double> now = magnetic(f);
  double err = 0, scale = 0;
  for (size_t n = 0; n < sync.size(); ++n) {
    err = std::max(err, std::fabs(sync[n] - 0.5 * (old[n] + now[n])));
    scale = std::max(scale, std::fabs(sync[n]));
  }
  const bool ok = err <= 1e-12 * scale;
  master_printf("  %-44s |H - step mean| / |H| = %.1e  %s\n", "3-D synchronized H, PML",
                err / scale, ok ? "ok" : "FAILED");
  failures += !ok;
}

/* In vacuum D = E, so the two outputs must agree everywhere, also where the
   centred grid averages over copies of interface dofs. */
static void d_equals_e(int nd) {
  const double size[3] = {1, 0.5, 0.5}, lo[3] = {0.5, 0.125, 0.125}, hi[3] = {0.75, 0.375, 0.375};
  grid_volume gv = nd == 3 ? vol3d(size[0], size[1], size[2], 8) : vol2d(size[0], size[1], 16);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(region(0, nd, lo, hi, size, 2, pid));
  structure s(gv, one_eps, no_pml(), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
              DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  gaussian_src_time g(1.0, 0.5);
  f.add_point_source(nd == 3 ? Ey : Ez, g, nd == 3 ? vec(0.61, 0.23, 0.27) : vec(0.61, 0.23));
  while (f.time() < 5)
    f.step();
  const volume all = f.gv.surroundings();
  size_t dims[3] = {1, 1, 1};
  direction dirs[3];
  const int rank = f.get_array_slice_dimensions(all, dims, dirs, true, false);
  size_t n = 1;
  for (int k = 0; k < rank; ++k)
    n *= dims[k];
  double err = 0, scale = 0;
  for (component c : {Ex, Ey, Ez}) {
    if (!f.gv.has_field(c)) continue;
    realnum *e = f.get_array_slice(all, c),
            *d = f.get_array_slice(all, direction_component(Dx, component_direction(c)));
    for (size_t k = 0; k < n; ++k) {
      err = std::max(err, (double)std::fabs(d[k] - e[k]));
      scale = std::max(scale, (double)std::fabs(e[k]));
    }
    delete[] e;
    delete[] d;
  }
  const bool ok = err <= 1e-6 * scale;
  master_printf("  %-44s |D - E| / |E| = %.1e  %s\n",
                nd == 3 ? "3-D vacuum, D and E outputs agree" : "2-D vacuum, D and E outputs agree",
                err / scale, ok ? "ok" : "FAILED");
  failures += !ok;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  if (count_processors() > 1) {
    master_printf("subgrid_conforming: serial only for now, skipped\n");
    return 77;
  }
  master_printf("conforming interface rows:\n");
#ifdef HAVE_LAPACK
  {
    const double size[2] = {1.0, 0.5}, lo[2] = {0.5, 0.125}, hi[2] = {0.75, 0.375};
    spectral("2-D TM box, r = 2, periodic", 2, 16, size, lo, hi, 2, true, Ez);
    spectral("2-D TE box, r = 3, metal walls", 2, 16, size, lo, hi, 3, false, Hz);
    spectral("2-D TE, anisotropic eps 12 across a face, r = 3", 2, 16, size, lo, hi, 3, true, Hz,
             disk_eps);
    // a strip across the whole period: rows, copies and slaves wrap with a phase
    const double slo[2] = {0, 0.125}, shi[2] = {1.0, 0.375};
    spectral("2-D TE strip across the period, Bloch k = 0.3", 2, 8, size, slo, shi, 2, true, Hz,
             one_eps, 0.3);
  }
  {
    const double size[3] = {1, 1, 1}, lo[3] = {1.0 / 3, 1.0 / 3, 1.0 / 3},
                 hi[3] = {2.0 / 3, 2.0 / 3, 2.0 / 3};
    spectral("3-D box, r = 2, periodic", 3, 3, size, lo, hi, 2, true, Ez);
    spectral("3-D box, r = 3, metal walls", 3, 3, size, lo, hi, 3, false, Ez);
  }
#else
  master_printf("  (no LAPACK: spectral checks skipped)\n");
#endif
  pml_decay();
  lorentz_decay();
  sync_check();
  d_equals_e(2);
  d_equals_e(3);
  if (failures) {
    master_printf("%d conforming check(s) failed\n", failures);
    return 1;
  }
  master_printf("subgrid conforming OK\n");
  return 0;
}
