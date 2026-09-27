/* Point sources at and near a refined box's face, with the conforming rows.

   In vacuum with PML all round the LDOS of a dipole depends on the position
   only through the grid under it, so each placement is compared with the
   uniform grid that is really there: the fine grid inside the box, the coarse
   grid (at the fine dt, which a refined run steps with) outside.  On the face
   Ez lies on coarse samples; Hz, which lives in cells, is placed in the first
   fine ones.  Placing the dipole per chunk counted it one and a half times
   just inside the face; slaved and interface dofs need their own amplitudes. */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }
static std::unique_ptr<binary_partition> leaf(int &pid, int r) {
  return std::unique_ptr<binary_partition>(new binary_partition(pid++, r));
}
// 3 x 3 cell, box [1.5, 2.25] x [1.125, 1.875] refined by 2
static std::unique_ptr<binary_partition> box(int &pid) {
  std::unique_ptr<binary_partition> ybox(
      new binary_partition(split_plane{Y, 1.125}, leaf(pid, 1),
                           std::unique_ptr<binary_partition>(new binary_partition(
                               split_plane{Y, 1.875}, leaf(pid, 2), leaf(pid, 1)))));
  std::unique_ptr<binary_partition> right(
      new binary_partition(split_plane{X, 2.25}, std::move(ybox), leaf(pid, 1)));
  return std::unique_ptr<binary_partition>(
      new binary_partition(split_plane{X, 1.5}, leaf(pid, 1), std::move(right)));
}

static double ldos(double res, bool refined, double courant, component c, const vec &p) {
  grid_volume gv = vol2d(3.0, 3.0, res);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(refined ? box(pid) : nullptr);
  structure s(gv, one_eps, pml(0.75), identity(), bp ? 0 : count_processors(), courant, false,
              DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  gaussian_src_time src(1.0, 0.4);
  f.add_point_source(c, src, p);
  dft_ldos l(1.0, 1.0, 1);
  while (f.time() < 45) {
    f.step();
    l.update(f);
  }
  double *v = l.ldos();
  const double out = v[0];
  delete[] v;
  return out;
}

// a dipole at x = 1.5 + dx against the uniform grid at x = 1.5 + ref_dx
static void check(component c, double dx, bool inside, double ref_dx, double tol,
                  const char *what) {
  const double res = 16, y = 1.5 + 1.0 / 32; // between coarse nodes
  const vec p(1.5 + dx, y), q(1.5 + ref_dx, y);
  const double ref = inside ? ldos(2 * res, false, 0.5, c, q) : ldos(res, false, 0.25, c, q);
  const double got = ldos(res, true, 0.5, c, p), err = got / ref - 1;
  const bool ok = std::fabs(err) < tol;
  master_printf("  %s %-38s %.5f vs %.5f (%s): %+.2e  %s\n", c == Ez ? "Ez" : "Hz", what, got, ref,
                inside ? "fine" : "coarse, fine dt", err, ok ? "ok" : "FAILED");
  failures += !ok;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  master_printf("point sources at a refined box's face, LDOS:\n");
  for (component c : {Ez, Hz}) {
    check(c, -1.0 / 32, false, -1.0 / 32, 1e-2, "half a coarse cell outside");
    if (c == Ez)
      check(c, 0.0, false, 0.0, 1e-2, "on the face");
    else
      check(c, 0.0, true, 1.0 / 64, 1e-2, "on the face");
    check(c, 1.0 / 32, true, 1.0 / 32, 1e-2, "one fine cell inside");
    check(c, 2.0 / 32, true, 2.0 / 32, 1e-2, "two fine cells inside");
  }
  if (failures) {
    master_printf("%d source check(s) failed\n", failures);
    return 1;
  }
  master_printf("subgrid sources OK\n");
  return 0;
}
