/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

/* Checks on dispersive_spectral_radius, the von Neumann estimate used to warn
   about unstable dispersive media at setup. */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <meep.hpp>
#include <meepgeom.hpp>
#include <stability.hpp>

using namespace meep;

/* The estimator approaches a marginal mode from above, so it resolves a limit
   to about 1e-5; check_medium_stability uses the same band. */
static const double margin = 1e-3;

static int failures = 0;

static double seed_resolution;

/* Excite the grid-scale mode that first becomes unstable for this update.  Hy
   sits half a cell off the origin, where a cosine of this period samples zero
   exactly and would seed nothing. */
static std::complex<double> nyquist_seed(const vec &p) {
  return std::complex<double>(sin(pi * seed_resolution * p.z()), 0.0);
}

static void check(bool ok, const char *what) {
  master_printf("%-58s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok) ++failures;
}

/* Run the actual 1D field update, rather than just its amplification matrix.
   A periodic Nyquist-mode magnetic field reliably excites the unstable mode. */
static double seeded_field_energy(double resolution, int steps) {
  grid_volume gv = volone(1.0, resolution);
  gv.center_origin();
  structure s(gv, [](const vec &) { return 1.0; });
  s.add_susceptibility([](const vec &) { return 60.0; }, E_stuff,
                       lorentzian_susceptibility(2.0, 0.0));
  fields f(&s);
  f.use_bloch(0.0);
  seed_resolution = resolution;
  f.initialize_field(Hy, nyquist_seed);
  for (int i = 0; i < steps; ++i)
    f.step();
  return f.field_energy();
}

// A medium that is unstable on a coarse grid, dispersive along direction d.
static meep_geom::medium_struct unstable_medium(int d) {
  meep_geom::medium_struct mm;
  meep_geom::susceptibility ss = {}; // vector3 members are not zeroed otherwise
  ss.frequency = 2.0;
  ss.gamma = 0.0;
  ss.drude = false;
  vector3 *sigma = &ss.sigma_diag;
  (d == 0 ? sigma->x : d == 1 ? sigma->y : sigma->z) = 60.0;
  mm.E_susceptibilities.push_back(ss);
  return mm;
}

/* Exercise the material-facing half, which turns a medium_struct and a grid
   into a verdict.  These are the decisions the estimator itself cannot make. */
static void check_medium_analysis() {
  const double res = 24, courant = 0.5, dt = courant / res;

  grid_volume gv3 = vol3d(1.0, 1.0, 1.0, res);
  const meep_geom::medium_struct mx = unstable_medium(0);
  meep_geom::stability_report r = meep_geom::analyze_medium_stability(&mx, gv3, dt, courant);
  check(r.growing && r.electric && r.rho > 1 && r.omega_0 == 2.0,
        "a coarse 3D grid is reported as growing");
  check(r.resolution > res && r.resolution <= meep_geom::stability_report::res_max,
        "the reported resolution is finer than the one in use");

  // At that resolution the medium must come back clean, or the advice is wrong.
  grid_volume gvfix = vol3d(1.0, 1.0, 1.0, r.resolution);
  check(!meep_geom::analyze_medium_stability(&mx, gvfix, courant / r.resolution, courant).growing,
        "the reported resolution is actually stable");

  /* In 1D only Ex is stepped, so dispersion along y or z is not a problem
     there, however unstable the same material would be in 3D. */
  grid_volume gv1 = volone(1.0, res);
  const meep_geom::medium_struct my = unstable_medium(1);
  check(meep_geom::analyze_medium_stability(&mx, gv1, dt, courant).growing,
        "1D still reports growth along the stepped component");
  check(!meep_geom::analyze_medium_stability(&my, gv1, dt, courant).growing,
        "1D ignores a component the grid never steps");

  /* Turning the same material 45 degrees about z spreads its sigma over the
     off-diagonal entries without changing the physics, so the verdict has to
     come out identical. */
  meep_geom::medium_struct rot = unstable_medium(0);
  rot.E_susceptibilities[0].sigma_diag.x = 30.0; // 60 * v v^T, v = (1,1,0)/sqrt(2)
  rot.E_susceptibilities[0].sigma_diag.y = 30.0;
  rot.E_susceptibilities[0].sigma_offdiag.x = 30.0;
  const meep_geom::stability_report rr =
      meep_geom::analyze_medium_stability(&rot, gv3, dt, courant);
  check(rr.growing && fabs(rr.rho - r.rho) < 1e-9 && rr.resolution == r.resolution,
        "a rotated susceptibility gives the same verdict as the diagonal one");

  // Rotated axes mix components, so 1D cannot say anything about them.
  check(!meep_geom::analyze_medium_stability(&rot, gv1, dt, courant).growing,
        "a rotated susceptibility is declined where not all components step");

  // Nor can it, if the instantaneous response does not survive the rotation.
  meep_geom::medium_struct rot_aniso = rot;
  rot_aniso.epsilon_diag.y = 2.0;
  check(!meep_geom::analyze_medium_stability(&rot_aniso, gv3, dt, courant).growing,
        "a rotated susceptibility is declined under anisotropic epsilon");

  /* Two susceptibilities turned to different axes stay coupled in every basis,
     which is the case this reduction genuinely cannot represent. */
  meep_geom::medium_struct clash = rot;
  clash.E_susceptibilities.push_back(unstable_medium(0).E_susceptibilities[0]);
  check(!meep_geom::analyze_medium_stability(&clash, gv3, dt, courant).growing,
        "susceptibilities with different principal axes are declined");

  // A magnetic susceptibility runs the same recurrence and must be caught too.
  meep_geom::medium_struct mag;
  mag.H_susceptibilities = mx.E_susceptibilities;
  r = meep_geom::analyze_medium_stability(&mag, gv3, dt, courant);
  check(r.growing && !r.electric, "a magnetic susceptibility is reported as magnetic");

  /* A cylindrical grid steps r and z, but its angular term makes it stricter
     than the 2D Cartesian grid it would otherwise look like, so it can never
     get away with a coarser resolution. */
  grid_volume gv2 = vol2d(1.0, 1.0, res);
  grid_volume gcyl = volcyl(1.0, 1.0, res);
  const double res2 = meep_geom::analyze_medium_stability(&mx, gv2, dt, courant).resolution;
  const double rescyl = meep_geom::analyze_medium_stability(&mx, gcyl, dt, courant).resolution;
  check(res2 > 0 && rescyl >= res2, "a cylindrical grid is no more forgiving than 2D");

  // Analysing the same medium twice must warn once.
  std::vector<const meep_geom::medium_struct *> seen;
  meep_geom::check_medium_stability_once(&mx, gv3, dt, courant, seen);
  meep_geom::check_medium_stability_once(&mx, gv3, dt, courant, seen);
  const meep_geom::medium_struct copy = unstable_medium(0);
  meep_geom::check_medium_stability_once(&copy, gv3, dt, courant, seen);
  check(seen.size() == 1, "an identical medium is only analysed once");
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);

  /* Without dispersion the estimate has to reproduce the Courant limit,
     dt <= dx/sqrt(ndim), in each dimension. */
  const std::vector<lorentzian_pole> none;
  for (int ndim = 1; ndim <= 3; ++ndim) {
    const double dx = 0.05, dt = dx / sqrt(double(ndim));
    char what[64];
    snprintf(what, sizeof(what), "vacuum brackets the Courant limit in %dD", ndim);
    check(dispersive_spectral_radius(1.0, 1.0, none, 0.99 * dt, dx, ndim) <= 1 + margin &&
              dispersive_spectral_radius(1.0, 1.0, none, 1.01 * dt, dx, ndim) > 1 + margin,
          what);
  }

  /* A Lorentzian strong enough to be unstable on a coarse grid. The estimate
     has to grow with dimension: the discrete Laplacian sums over the axes, so
     a 1D-only symbol reports 2D and 3D setups as safer than they are. */
  std::vector<lorentzian_pole> pole(1);
  pole[0].omega_0 = 2.0;
  pole[0].gamma = 0.0;
  pole[0].sigma = 60.0;
  pole[0].no_omega_0_denominator = false;

  const double dx = 1.0 / 24, dt = 0.5 * dx;
  double rho[4] = {0, 0, 0, 0};
  for (int ndim = 1; ndim <= 3; ++ndim)
    rho[ndim] = dispersive_spectral_radius(1.0, 1.0, pole, dt, dx, ndim);

  check(rho[1] > 1 + margin, "coarse grid is flagged unstable in 1D");
  check(rho[2] > rho[1] * 1.05, "2D estimate exceeds 1D");
  check(rho[3] > rho[2] * 1.05, "3D estimate exceeds 2D");

  /* A cylindrical grid steps two axes and weighs its angular term as a
     fraction of a third, so its Courant limit sits between 2D and 3D. */
  const double cyl_dx = 0.05, cyl_dt = cyl_dx / sqrt(2.6);
  check(dispersive_spectral_radius(1.0, 1.0, none, 0.99 * cyl_dt, cyl_dx, 2.6) <= 1 + margin &&
            dispersive_spectral_radius(1.0, 1.0, none, 1.01 * cyl_dt, cyl_dx, 2.6) > 1 + margin,
        "a cylindrical grid brackets its own Courant limit");
  check(dispersive_spectral_radius(1.0, 1.0, none, cyl_dx / sqrt(2.0), cyl_dx, 2.6) > 1 + margin,
        "the 2D limit is too generous for a cylindrical grid");

  // Refining far enough makes it stable again.
  const double fine_dx = 1.0 / 400;
  check(dispersive_spectral_radius(1.0, 1.0, pole, 0.5 * fine_dx, fine_dx, 3) <= 1 + margin,
        "a fine enough grid is stable in 3D");

  // The dominant-pole report must identify the unstable oscillator, not its order.
  std::vector<lorentzian_pole> two_poles(2);
  two_poles[0] = pole[0];
  two_poles[0].omega_0 = 0.1;
  two_poles[0].sigma = 0.01;
  two_poles[1] = pole[0];
  int worst = -1;
  check(dispersive_spectral_radius(1.0, 1.0, two_poles, dt, dx, 1, &worst) > 1 + margin &&
            worst == 1,
        "the growing pole is identified in a multi-pole medium");

  /* Check the analysis against the field update.  The deliberately coarse
     grid should grow by orders of magnitude, while refinement is bounded. */
  const double coarse_energy = seeded_field_energy(24.0, 8);
  const double fine_energy = seeded_field_energy(400.0, 8);
  check(std::isfinite(coarse_energy) && coarse_energy > 1e4 * fine_energy && fine_energy < 10.0,
        "FDTD growth agrees with the coarse-versus-fine estimate");

  /* mu scales the Courant limit exactly like eps_inf: without dispersion the
     limit is dt <= dx*sqrt(eps_inf*mu), so it must be resolved sharply on
     either side of mu = 0.25 at dt = dx/2. */
  const double vdx = 0.05;
  check(dispersive_spectral_radius(1.0, 0.25, none, 0.49 * vdx, vdx, 1) <= 1 + margin,
        "mu = 1/4 is stable just inside dt = dx/2");
  check(dispersive_spectral_radius(1.0, 0.25, none, 0.51 * vdx, vdx, 1) > 1 + margin,
        "mu = 1/4 is unstable just outside it");

  check_medium_analysis();

  // A perfect metal is not a dispersion problem and must not be warned about.
  check(dispersive_spectral_radius(0.0, 1.0, pole, dt, dx, 1) <= 1 + margin,
        "a non-positive eps_inf raises no warning");

  master_printf("%s\n", failures ? "FAILED" : "all dispersive stability checks passed");
  return failures ? 1 : 0;
}
