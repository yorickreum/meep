/* What does a refinement interface reflect out of empty space?

   Energy is blind to this: it is conserved whether the wave goes on or comes
   back.  A flux plane between the source and a refined strip sees the wave go
   out and anything come back; differencing against the same run with no
   refinement leaves only what the interface reflected.

   The reflection must fall at second order in the base cell.  The wave
   crosses head-on and is nearly uniform along the interface, so the
   transverse coupling is barely exercised; that wants oblique incidence. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <vector>
#include <meep.hpp>

using namespace meep;

static double one_eps(const vec &) { return 1.0; }

static const double SX = 3.0, SY = 5.0;
static const double Y_SRC = -2.0, Y_MON = -1.6, Y_LO = 0.2, Y_HI = 0.6;
static const double FCEN = 1.0, DF = 0.8;
static const int NFREQ = 21;

/* Flux spectrum at the monitor.  R = 1 leaves the grid uniform; R > 1 refines
   the strip between Y_LO and Y_HI, which the wave has not reached yet when it
   first crosses the monitor. */
static std::vector<double> spectrum(double base_res, int R) {
  grid_volume gv = vol2d(SX, SY, base_res);
  std::unique_ptr<structure> s;
  if (R == 1) { s.reset(new structure(gv, one_eps, pml(0.5))); }
  else {
    // the cell runs from 0, so the split positions are shifted from the
    // physical ones by half the cell size
    binary_partition bp(split_plane{Y, Y_LO + 0.5 * SY},
                        std::unique_ptr<binary_partition>(new binary_partition(0, 1)),
                        std::unique_ptr<binary_partition>(new binary_partition(
                            split_plane{Y, Y_HI + 0.5 * SY},
                            std::unique_ptr<binary_partition>(new binary_partition(1, R)),
                            std::unique_ptr<binary_partition>(new binary_partition(2, 1)))));
    s.reset(new structure(gv, one_eps, pml(0.5), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
                          DEFAULT_SUBPIXEL_MAXEVAL, &bp));
  }

  fields f(s.get());
  gaussian_src_time src(FCEN, DF);
  f.add_point_source(Ez, src, vec(0.5 * SX, Y_SRC + 0.5 * SY));

  volume mon(vec(0.5 * SX - 1.0, Y_MON + 0.5 * SY), vec(0.5 * SX + 1.0, Y_MON + 0.5 * SY));
  dft_flux flux = f.add_dft_flux_plane(mon, FCEN - 0.5 * DF, FCEN + 0.5 * DF, NFREQ);

  const double t_end = src.last_time() + 40.0;
  while (f.time() < t_end)
    f.step();

  double *raw = flux.flux();
  std::vector<double> out(raw, raw + NFREQ);
  delete[] raw;
  return out;
}

/* Reflected amplitude relative to the incident peak. */
static double reflection(double base_res, int R) {
  std::vector<double> inc = spectrum(base_res, 1), got = spectrum(base_res, R);
  double peak = 0, worst = 0;
  for (int i = 0; i < NFREQ; ++i) {
    if (fabs(inc[i]) > peak) peak = fabs(inc[i]);
    if (fabs(got[i] - inc[i]) > worst) worst = fabs(got[i] - inc[i]);
  }
  return worst / peak;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  const double r20 = reflection(20.0, 2), r40 = reflection(40.0, 2);
  const double order = log(r20 / r40) / log(2.0);
  master_printf("  reflection r=2: base 20 %.3e, base 40 %.3e, order %.2f (want 2)\n", r20, r40,
                order);

  int failures = 0;
  if (!(r40 < r20)) {
    master_printf("FAILED: refining the base grid did not reduce the reflection\n");
    ++failures;
  }
  if (!(log(r20 / r40) / log(2.0) > 1.8)) {
    master_printf("FAILED: the reflection falls slower than second order\n");
    ++failures;
  }
  if (!(r40 < 1.0e-3)) {
    master_printf("FAILED: reflection %.3e at base 40 exceeds 1.0e-3\n", r40);
    ++failures;
  }
  if (failures) return 1;
  master_printf("subgrid reflection OK\n");
  return 0;
}
