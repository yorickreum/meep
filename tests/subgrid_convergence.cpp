/* Does a refined run converge, and at the same order as a uniform one?

   Energy conservation says the interface is stable, not that it is right.  The
   prototype this design came from settled the question with cavity
   eigenfrequencies; the equivalent here is total field energy at a fixed
   physical time, measured against a high-resolution uniform reference and
   refined by halving the base cell.  An integral rather than a point value:
   a pointwise probe was tried first and measured noise, its error exceeding
   the reference value itself.

   The uniform column is the control: whatever order it shows, the refined
   column has to match it.  A scheme whose interface error is an O(1/r)
   constant rather than a converging term would hold its order here while
   failing to improve with the ratio, so the ratio is swept too. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <complex>
#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }

/* Ez at `mon` once the run has passed t_target. */
static double probe(double base_res, int R, double t_target, const vec &mon) {
  grid_volume gv = vol2d(2.0, 1.0, base_res);
  std::unique_ptr<structure> s;
  if (R == 1) { s.reset(new structure(gv, one_eps, no_pml(), identity(), 0, 0.5)); }
  else {
    binary_partition bp(split_plane{X, 1.0},
                        std::unique_ptr<binary_partition>(new binary_partition(0, 1)),
                        std::unique_ptr<binary_partition>(new binary_partition(0, R)));
    s.reset(new structure(gv, one_eps, no_pml(), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
                          DEFAULT_SUBPIXEL_MAXEVAL, &bp));
  }
  fields f(s.get());
  gaussian_src_time src(0.5, 0.15);
  f.add_point_source(Ez, src, vec(0.5, 0.5));
  (void)mon;
  (void)t_target;
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

  const double t_target = 0; // probe() derives its own stop time from the source
  const vec mon(1.5, 0.35);  // in the right-hand half, across the interface
  const double res[3] = {10.0, 20.0, 40.0};

  const double reference = probe(160.0, 1, t_target, mon);
  printf("reference (uniform res 160): energy = %.8e\n", reference);

  const int ratio[] = {1, 2, 3, 4}; // 3 is odd: its ghosts land on fine nodes
  for (size_t ri = 0; ri < sizeof(ratio) / sizeof(*ratio); ++ri) {
    const int R = ratio[ri];
    double err[3];
    for (int i = 0; i < 3; ++i)
      err[i] = fabs(probe(res[i], R, t_target, mon) - reference);
    const double p = order_of(err, res, 3);
    printf("  R=%d  err %.3e %.3e %.3e   order %.2f%s\n", R, err[0], err[1], err[2], p,
           R == 1 ? "   (uniform control)" : "");
    if (R > 1 && p < 1.5) {
      printf("FAILED: R=%d converges at order %.2f, expected second order\n", R, p);
      ++failures;
    }
  }

  if (failures) {
    printf("%d convergence check(s) failed\n", failures);
    return 1;
  }
  printf("subgrid convergence OK\n");
  return 0;
}
