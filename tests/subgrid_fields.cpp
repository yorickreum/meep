/* Energy conservation across a refinement interface.

   In a closed lossless cell with the source switched off, total field energy
   must stay put up to the bounded beat of E.D and H.B sampled half a step
   apart.  The uniform run is the control: the refined runs must stay within
   ten times its drift. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }

static double run_inner(int R, const char *label) {
  const double a = 20.0;
  grid_volume gv = vol2d(2.0, 1.0, a);

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

  const double t_off = src.last_time() + 1.0;
  while (f.time() < t_off)
    f.step();
  const double e0 = f.field_energy();
  while (f.time() < t_off + 8.0)
    f.step();
  const double e1 = f.field_energy();

  const double drift = (e0 > 0) ? fabs(e1 - e0) / e0 : 0.0;
  printf("  %-20s energy %.6e -> %.6e   drift %.3e\n", label, e0, e1, drift);
  fflush(stdout);
  return drift;
}

/* Meep throws when the fields go non-finite, which a broken interface does;
   report it rather than dying. */
static double run(int R, const char *label) {
  try {
    return run_inner(R, label);
  }
  catch (const std::exception &e) {
    printf("  %-20s blew up: %s\n", label, e.what());
    fflush(stdout);
    return 1e300;
  }
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  const double control = run(1, "uniform (control)");
  const double tol = std::max(control * 10, 1e-5);
  printf("  tolerance from control: %.3e\n", tol);
  fflush(stdout);

  for (int R = 2; R <= 4; R *= 2) {
    char label[32];
    snprintf(label, sizeof(label), "refined R=%d", R);
    if (run(R, label) > tol) {
      printf("FAILED: %s does not conserve energy like the uniform control\n", label);
      ++failures;
    }
  }

  if (failures) {
    printf("%d subgrid field check(s) failed\n", failures);
    return 1;
  }
  printf("subgrid field checks OK\n");
  return 0;
}
