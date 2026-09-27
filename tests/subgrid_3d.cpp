/* Refinement in three dimensions.

   A box, not a slab: its twelve edges and eight corners are where the slaved
   samples of the interface meet.  Same measurement as subgrid_convergence:
   energy in a closed lossless cell at a fixed physical time, against a
   uniform high-resolution reference, with the uniform run as the control and
   the ratio swept. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }

static const double SX = 1.0, SY = 0.5, SZ = 0.5;
/* on a coarse cell boundary at every base resolution swept below */
static const double BLO[3] = {0.500, 0.125, 0.125};
static const double BHI[3] = {0.750, 0.375, 0.375};

/* Splits each direction twice, so the refined leaf is a box with edges and
   corners against coarse neighbours on all six faces. */
static std::unique_ptr<binary_partition> box_tree(int d, int R, int &pid) {
  if (d == 3) return std::unique_ptr<binary_partition>(new binary_partition(pid++, R));
  const direction dir = d == 0 ? X : (d == 1 ? Y : Z);
  std::unique_ptr<binary_partition> inner(box_tree(d + 1, R, pid));
  std::unique_ptr<binary_partition> above(new binary_partition(pid++, 1));
  std::unique_ptr<binary_partition> upper(
      new binary_partition(split_plane{dir, BHI[d]}, std::move(inner), std::move(above)));
  std::unique_ptr<binary_partition> below(new binary_partition(pid++, 1));
  return std::unique_ptr<binary_partition>(
      new binary_partition(split_plane{dir, BLO[d]}, std::move(below), std::move(upper)));
}

static double probe(double base_res, int R) {
  grid_volume gv = vol3d(SX, SY, SZ, base_res);
  std::unique_ptr<structure> s;
  if (R == 1) { s.reset(new structure(gv, one_eps, no_pml())); }
  else {
    int pid = 0;
    std::unique_ptr<binary_partition> bp(box_tree(0, R, pid));
    s.reset(new structure(gv, one_eps, no_pml(), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
                          DEFAULT_SUBPIXEL_MAXEVAL, bp.get()));
  }
  fields f(s.get());
  gaussian_src_time src(1.0, 0.5);
  f.add_point_source(Ez, src, vec(0.25, 0.25, 0.25));
  const double t_end = src.last_time() + 1.0;
  while (f.time() < t_end)
    f.step();
  return f.field_energy();
}

static double order_of(const double *err, const double *res, int n) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < n; ++i) {
    const double x = log(1.0 / res[i]), y = log(err[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  const double res[3] = {8.0, 16.0, 32.0};
  const double reference = probe(64.0, 1);
  printf("reference (uniform 3-D res 64): energy = %.8e\n", reference);

  const int ratio[] = {1, 2, 4};
  for (size_t ri = 0; ri < sizeof(ratio) / sizeof(*ratio); ++ri) {
    const int R = ratio[ri];
    double err[3];
    for (int i = 0; i < 3; ++i)
      err[i] = fabs(probe(res[i], R) - reference);
    const double p = order_of(err, res, 3);
    printf("  R=%d  err %.3e %.3e %.3e   order %.2f%s\n", R, err[0], err[1], err[2], p,
           R == 1 ? "   (uniform control)" : "");
    if (R > 1 && p < 1.5) {
      printf("FAILED: 3-D R=%d converges at order %.2f, expected second order\n", R, p);
      ++failures;
    }
  }

  if (failures) {
    printf("%d 3-D convergence check(s) failed\n", failures);
    return 1;
  }
  printf("subgrid 3d OK\n");
  return 0;
}
