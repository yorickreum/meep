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

/* Not installed: the von Neumann estimate is a setup-time diagnostic, not part
   of Meep's interface.  Declared here rather than in stability.cpp so that
   tests/dispersive_stability.cpp can reach it. */

#ifndef MEEP_STABILITY_H
#define MEEP_STABILITY_H

#include <vector>

namespace meep {

/* One Lorentzian or Drude pole of a dispersive medium.  lorentzian_susceptibility
   holds the same three numbers, but its sigma is per grid point and its members
   are protected, so it cannot be passed around by value. */
struct lorentzian_pole {
  double omega_0, gamma, sigma;
  bool no_omega_0_denominator; // true for a Drude term
};

/* Estimated spectral radius of the coupled Yee + ADE timestep for a uniform,
   diagonal medium, maximized over spatial frequency.  The setup diagnostic
   flags growth when this estimate exceeds one.

   naxes weights the Yee symbol, which sums over the grid axes: the worst case
   grows as sqrt(naxes), so a 1D estimate would call unstable 2D and 3D setups
   safe.  Cartesian grids pass their dimension.  Cylindrical grids pass a
   fractional count, because their angular term adds less than a full axis; see
   check_medium_stability.  When worst_pole is non-NULL it receives the index of
   the pole carrying the growing mode, or -1 if there are no poles. */
double dispersive_spectral_radius(double eps_inf, double mu_inf,
                                  const std::vector<lorentzian_pole> &poles, double dt, double dx,
                                  double naxes, int *worst_pole = NULL);

} // namespace meep

#endif /* MEEP_STABILITY_H */
