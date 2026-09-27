/* Does a Lorentzian medium sitting across a refinement interface converge?

   Where the rows derive E at a ghost they also step P there, so the
   polarization crosses the interface as the fields do.  Same shape as
   subgrid_convergence: total field energy against a uniform high-resolution
   reference, base cell halved, with the uniform run as the control and the
   ratio swept.  The susceptibility covers the whole cell, so it spans the
   interface rather than stopping at it. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <complex>
#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }
static double all_sigma(const vec &) { return 0.4; }

static double probe(double base_res, int R) {
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
  s->add_susceptibility(all_sigma, E_stuff, lorentzian_susceptibility(0.7, 0.02));

  fields f(s.get());
  gaussian_src_time src(0.5, 0.15);
  f.add_point_source(Ez, src, vec(0.5, 0.5));
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

  const double res[3] = {10.0, 20.0, 40.0};
  const double reference = probe(160.0, 1);
  printf("reference (uniform res 160, lorentzian): energy = %.8e\n", reference);

  const int ratio[] = {1, 2, 3, 4};
  for (size_t ri = 0; ri < sizeof(ratio) / sizeof(*ratio); ++ri) {
    const int R = ratio[ri];
    double err[3];
    for (int i = 0; i < 3; ++i)
      err[i] = fabs(probe(res[i], R) - reference);
    const double p = order_of(err, res, 3);
    printf("  R=%d  err %.3e %.3e %.3e   order %.2f%s\n", R, err[0], err[1], err[2], p,
           R == 1 ? "   (uniform control)" : "");
    if (R > 1 && p < 1.5) {
      printf("FAILED: R=%d converges at order %.2f, expected second order\n", R, p);
      ++failures;
    }
  }

  if (failures) {
    printf("%d dispersive convergence check(s) failed\n", failures);
    return 1;
  }
  printf("subgrid dispersive convergence OK\n");
  return 0;
}
