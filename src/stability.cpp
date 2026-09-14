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

/* Setup-time check that a dispersive material can be timestepped stably on a
   given grid: the von Neumann estimate in namespace meep, and the medium_struct
   plumbing that feeds it in namespace meep_geom. */

#include <cmath>
#include <complex>
#include <vector>

#include "meep.hpp"
#include "stability.hpp"
#include "meepgeom.hpp"

using namespace std;

namespace meep {

/* Diagnose unstable dispersive timesteps (issue #12).

   An earlier version of this test looked at each Lorentzian ODE in
   isolation, asking whether its pole leaves the unit circle.  That is not
   the thing that goes unstable: the Yee update for E and H and the N
   polarization recursions advance as one iteration, and it is the joint
   iteration whose spectral radius has to stay <= 1.  The isolated test is
   also degenerate for a Drude term, whose pole sits exactly on the unit
   circle, so its verdict there was decided by rounding error.

   Substituting a plane wave into the update equations below leaves a
   (2+2N)x(2+2N) amplification matrix acting on

       [ E, H, P_0, P_0_prev, ..., P_{N-1}, P_{N-1}_prev ],

   built from the same coefficients update_P computes.  The spatial
   derivative enters only through the Yee symbol 2*sin(k*dx/2)/dx, so it is
   enough to sweep k*dx over (0, pi].  This estimate models a uniform,
   diagonal medium; naxes carries whatever the grid geometry does to the
   symbol. */

namespace {

// Row-major n x n complex matrix.
typedef std::vector<std::complex<double> > cmatrix;

// Matrix infinity norm (max row sum), which is submultiplicative.
double matrix_norm(const cmatrix &A, int n) {
  double best = 0;
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (int j = 0; j < n; ++j)
      s += std::abs(A[i * n + j]);
    if (s > best) best = s;
  }
  return best;
}

void matrix_square(cmatrix &A, int n) {
  cmatrix B(n * n, std::complex<double>(0, 0));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < n; ++k) {
      const std::complex<double> a = A[i * n + k];
      if (a == std::complex<double>(0, 0)) continue;
      for (int j = 0; j < n; ++j)
        B[i * n + j] += a * A[k * n + j];
    }
  A.swap(B);
}

/* Spectral radius via Gelfand's formula, |A^m|^(1/m) as m -> infinity, with
   m = 2^nsquare reached by repeated squaring.  This needs no eigensolver (Meep
   does not always link LAPACK) and is unbothered by the degenerate and
   complex-conjugate eigenvalues these matrices are full of.  Each squaring is
   renormalized so the entries cannot overflow.

   The normalized powered matrix is also used to estimate which polarization
   contributes most to a growing mode; mode holds that estimate.

   m has to be large: a Drude term contributes a mode at exactly |z| = 1 (its
   undamped DC response), and around a marginal mode the estimate approaches
   rho from above like m^(1/m), which is 1.0068 at m = 2^10 but 1.000013 at
   m = 2^20.  Reading that overshoot as growth is how the previous test came to
   flag every Drude material; the extra squarings are negligible for these
   small setup-time matrices. */
double spectral_radius(cmatrix A, int n, std::vector<double> *mode) {
  const int nsquare = 20;
  double log_rho = 0;
  for (int j = 1; j <= nsquare; ++j) {
    matrix_square(A, n);
    const double nrm = matrix_norm(A, n);
    if (nrm != nrm) return infinity; // NaN: report as unstable, never as safe
    if (!(nrm > 0)) return 0;        // nilpotent, or underflowed to zero
    for (int i = 0; i < n * n; ++i)
      A[i] /= nrm;
    log_rho += log(nrm) / double(1 << j);
  }
  if (mode) {
    int best = 0;
    double best_norm = -1;
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int i = 0; i < n; ++i)
        s += std::abs(A[i * n + j]);
      if (s > best_norm) {
        best_norm = s;
        best = j;
      }
    }
    mode->resize(n);
    for (int i = 0; i < n; ++i)
      (*mode)[i] = best_norm > 0 ? std::abs(A[i * n + best]) / best_norm : 0;
  }
  return exp(log_rho);
}

} // namespace

double dispersive_spectral_radius(double eps_inf, double mu_inf,
                                  const std::vector<lorentzian_pole> &poles, double dt, double dx,
                                  double naxes, int *worst_pole) {
  const int N = int(poles.size()), n = 2 + 2 * N;
  const int nk = 40;
  std::vector<double> a(N), b(N), c(N), mode;
  double worst = 0;

  if (worst_pole) *worst_pole = -1;
  /* A non-positive instantaneous response is not a dispersion problem (a
     perfect metal is spelled that way), and the amplification matrix does not
     exist for it.  Report no growth rather than a warning nobody can act on. */
  if (!(eps_inf > 0) || !(mu_inf > 0)) return 1;
  /* The discrete Laplacian symbol sums over the axes, so with equal spacing the
     largest |k| is sqrt(naxes) times the single-axis value. Sampling that one
     scaled axis reaches the same worst case as sweeping the full k-space. */
  const double kscale = sqrt(naxes < 1 ? 1 : naxes);
  for (int ik = 1; ik <= nk; ++ik) {
    const double q = kscale * dt * (2 * sin(pi * ik / nk / 2) / dx);
    double csum = 0;
    for (int j = 0; j < N; ++j) {
      const double w = 2 * pi * poles[j].omega_0, g = 2 * pi * poles[j].gamma;
      const double w2dt2 = (w * dt) * (w * dt);
      const double gamma1inv = 1 / (1 + g * dt / 2), gamma1 = 1 - g * dt / 2;
      a[j] = gamma1inv * (2 - (poles[j].no_omega_0_denominator ? 0 : w2dt2));
      b[j] = -gamma1inv * gamma1;
      c[j] = gamma1inv * w2dt2 * poles[j].sigma;
      csum += c[j];
    }

    cmatrix M(n * n, std::complex<double>(0, 0));
    /* H is advanced first and E then sees the new H, so the E row picks up the
       H row's 1/mu as well.  The q^2 term must therefore carry 1/(eps*mu) to
       recover the vacuum limit dt <= dx*sqrt(eps*mu).  Dropping 1/mu reports
       every mu < 1 as unstable at any timestep. */
    M[0 * n + 0] = (eps_inf - csum - q * q / mu_inf) / eps_inf;
    M[0 * n + 1] = std::complex<double>(0, -q) / eps_inf;
    M[1 * n + 0] = std::complex<double>(0, -q) / mu_inf;
    M[1 * n + 1] = 1;
    for (int j = 0; j < N; ++j) {
      M[0 * n + (2 + 2 * j)] = (1 - a[j]) / eps_inf;
      M[0 * n + (3 + 2 * j)] = -b[j] / eps_inf;
      M[(2 + 2 * j) * n + 0] = c[j];
      M[(2 + 2 * j) * n + (2 + 2 * j)] = a[j];
      M[(2 + 2 * j) * n + (3 + 2 * j)] = b[j];
      M[(3 + 2 * j) * n + (2 + 2 * j)] = 1;
    }

    const double rho = spectral_radius(M, n, worst_pole ? &mode : NULL);
    if (rho > worst) {
      worst = rho;
      if (worst_pole) { // which polarization carries the growing mode
        double best_amp = -1;
        for (int j = 0; j < N; ++j)
          if (mode[2 + 2 * j] > best_amp) {
            best_amp = mode[2 + 2 * j];
            *worst_pole = j;
          }
      }
    }
  }
  return worst;
}

} // namespace meep

namespace meep_geom {

static double diag_component(vector3 v, int d) {
  if (d == 0) return v.x;
  return d == 1 ? v.y : v.z;
}

/* True if this susceptibility steps by a recurrence the estimate does not
   model at all.  Gyrotropic and multilevel media are genuinely different
   updates; an off-diagonal sigma is not, and is handled below. */
static bool outside_model(const susceptibility &ss) {
  return ss.is_file || ss.saturated_gyrotropy || ss.bias.x != 0.0 || ss.bias.y != 0.0 ||
         ss.bias.z != 0.0 || ss.transitions.size() != 0 || ss.initial_populations.size() != 0;
}

// A susceptibility's sigma as the symm_matrix the rest of meepgeom passes around.
static symm_matrix sigma_tensor(const susceptibility &ss) {
  symm_matrix t;
  t.m00 = ss.sigma_diag.x;
  t.m11 = ss.sigma_diag.y;
  t.m22 = ss.sigma_diag.z;
  t.m01 = ss.sigma_offdiag.x;
  t.m02 = ss.sigma_offdiag.y;
  t.m12 = ss.sigma_offdiag.z;
  return t;
}

static bool is_diagonal(const symm_matrix &t) {
  return t.m01 == 0.0 && t.m02 == 0.0 && t.m12 == 0.0;
}

static void as_rows(const symm_matrix &t, double m[3][3]) {
  m[0][0] = t.m00;
  m[1][1] = t.m11;
  m[2][2] = t.m22;
  m[0][1] = m[1][0] = t.m01;
  m[0][2] = m[2][0] = t.m02;
  m[1][2] = m[2][1] = t.m12;
}

// a^T t b: the tensor's response along one axis, or the coupling between two.
static double project(const symm_matrix &t, const double a[3], const double b[3]) {
  double m[3][3];
  as_rows(t, m);
  double s = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      s += a[i] * m[i][j] * b[j];
  return s;
}

/* Eigenvectors of a symmetric 3x3 by cyclic Jacobi rotations, which converge in
   a few sweeps at this size and need no eigensolver.  axis[i] receives the i-th
   eigenvector. */
static void eigen_axes(const symm_matrix &tensor, double axis[3][3]) {
  double a[3][3], v[3][3];
  as_rows(tensor, a);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      v[i][j] = (i == j) ? 1.0 : 0.0;

  for (int sweep = 0; sweep < 24; ++sweep) {
    double off = 0;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q)
        off += a[p][q] * a[p][q];
    if (off == 0) break;

    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0) continue;
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
        const double c = 1 / sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 3; ++k) { // columns, then rows, then the basis
          const double kp = a[k][p], kq = a[k][q];
          a[k][p] = c * kp - s * kq;
          a[k][q] = s * kp + c * kq;
        }
        for (int k = 0; k < 3; ++k) {
          const double pk = a[p][k], qk = a[q][k];
          a[p][k] = c * pk - s * qk;
          a[q][k] = s * pk + c * qk;
        }
        for (int k = 0; k < 3; ++k) {
          const double kp = v[k][p], kq = v[k][q];
          v[k][p] = c * kp - s * kq;
          v[k][q] = s * kp + c * kq;
        }
      }
  }
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      axis[i][k] = v[k][i];
}

/* Replace axis with a basis in which every susceptibility's sigma is diagonal,
   leaving it alone if they all already are.  A symmetric sigma is diagonal in
   its own eigenbasis, and there each eigenvalue gives back the uncoupled scalar
   problem this file solves.  Two tensors with different principal axes have no
   such basis, and the components stay coupled: return false and let the caller
   decline the medium rather than analyse a coupling it cannot see. */
static bool shared_eigenbasis(const std::vector<susceptibility> &suscs, double axis[3][3]) {
  size_t first = suscs.size();
  for (size_t i = 0; i < suscs.size(); ++i)
    if (!is_diagonal(sigma_tensor(suscs[i]))) {
      first = i;
      break;
    }
  if (first == suscs.size()) return true;

  eigen_axes(sigma_tensor(suscs[first]), axis);

  for (size_t j = 0; j < suscs.size(); ++j) {
    const symm_matrix t = sigma_tensor(suscs[j]);
    double rows[3][3], scale = 1;
    as_rows(t, rows);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        if (fabs(rows[i][k]) > scale) scale = fabs(rows[i][k]);
    for (int i = 0; i < 3; ++i)
      for (int k = i + 1; k < 3; ++k)
        if (fabs(project(t, axis[i], axis[k])) > 1e-9 * scale) return false;
  }
  return true;
}

/* True if the instantaneous response is a scalar.  Rotating into the
   susceptibility's axes leaves such a response unchanged; an anisotropic one
   would have to be rotated too, and in general does not diagonalize there. */
static bool isotropic_response(const medium_struct *mm) {
  return mm->epsilon_diag.x == mm->epsilon_diag.y && mm->epsilon_diag.y == mm->epsilon_diag.z &&
         mm->mu_diag.x == mm->mu_diag.y && mm->mu_diag.y == mm->mu_diag.z &&
         mm->epsilon_offdiag.x.re == 0 && mm->epsilon_offdiag.y.re == 0 &&
         mm->epsilon_offdiag.z.re == 0 && mm->mu_offdiag.x.re == 0 && mm->mu_offdiag.y.re == 0 &&
         mm->mu_offdiag.z.re == 0;
}

// Collect the poles of one medium along one axis of the analysis basis.
static std::vector<meep::lorentzian_pole> medium_poles(const std::vector<susceptibility> &suscs,
                                                       const double axis[3]) {
  std::vector<meep::lorentzian_pole> poles;
  for (size_t i = 0; i < suscs.size(); ++i) {
    const susceptibility &ss = suscs[i];
    meep::lorentzian_pole p;
    p.omega_0 = ss.frequency;
    p.gamma = ss.gamma;
    p.sigma = project(sigma_tensor(ss), axis, axis);
    p.no_omega_0_denominator = ss.drude;
    if (p.sigma != 0.0) poles.push_back(p);
  }
  return poles;
}

/* The material's three diagonal entries, as grid directions.  For cylindrical
   media they are r, phi and z rather than x, y and z. */
static meep::direction diag_direction(meep::ndim dim, int d) {
  if (dim == meep::Dcyl) return d == 0 ? meep::R : (d == 1 ? meep::P : meep::Z);
  return meep::direction(d);
}

/* Decide whether the timestep is too large for a dispersive material to be
   stable.  Unlike a Courant condition this depends on the material, so it
   cannot be checked when the grid is created; see issue #12. */
stability_report analyze_medium_stability(const medium_struct *mm, const meep::grid_volume &gv,
                                          double dt, double Courant) {
  stability_report report;
  report.growing = false;
  report.electric = true;
  report.rho = 1;
  report.omega_0 = 0;
  report.resolution = 0;
  report.low_eps_inf = false;

  const double dx = 1.0 / gv.a;

  /* A cylindrical grid steps r and z like a 2D Cartesian one, and its curl
     carries an extra angular term i*m/r (step_db.cpp).  Meep's default
     zero_fields_near_cylorigin zeroes the fields inside r = |m| cells, so that
     term stays bounded by 1/dx however large m is, against 2/dx for a full Yee
     axis.  It therefore weighs a fixed fraction of an axis whatever m turns out
     to be -- which is just as well, since m belongs to the fields object and
     does not exist yet when the structure is built.

     That symbol alone puts the fraction at 0.25, but it drops the 1/r term in
     (1/r) d(r f)/dr and the r = 0 special cases, and step_db.cpp reports the
     measured vacuum limit as Courant < 0.62 rather than the 0.667 that 2.25
     would give.  Use the measured figure, 1/0.62^2, so that the resolution this
     recommends is not the optimistic one.

     Without the zeroing the angular term grows to 2*|m|/dx and this is far too
     generous -- but so is the vacuum Courant limit, which step_db.cpp puts at
     1/(|m| + 0.5) in that case. */
  const double naxes = meep::number_of_directions(gv.dim) + (gv.dim == meep::Dcyl ? 0.60 : 0.0);

  // Electric and magnetic susceptibilities run through the same ADE recursion.
  for (int field = 0; field < 2; ++field) {
    const bool electric = (field == 0);
    const std::vector<susceptibility> &suscs =
        electric ? mm->E_susceptibilities : mm->H_susceptibilities;
    bool modelled = true;
    for (size_t i = 0; i < suscs.size(); ++i)
      if (outside_model(suscs[i])) modelled = false;
    if (!modelled) continue;

    /* Analyse along the grid axes unless an off-diagonal sigma rotates the
       medium's own axes away from them. */
    double axis[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    bool rotated = false;
    for (size_t i = 0; i < suscs.size() && !rotated; ++i)
      rotated = !is_diagonal(sigma_tensor(suscs[i]));

    if (rotated) {
      /* A rotated axis mixes the field components, so all three have to be
         stepped for the mixture to mean anything, and the instantaneous
         response has to survive the rotation. */
      bool all_stepped = true;
      for (int d = 0; d < 3; ++d)
        if (!gv.has_field(meep::direction_component(electric ? meep::Ex : meep::Hx,
                                                    diag_direction(gv.dim, d))))
          all_stepped = false;
      if (!all_stepped || !isotropic_response(mm)) continue;
      if (!shared_eigenbasis(suscs, axis)) continue;
    }

    for (int d = 0; d < 3; ++d) {
      // Skip components the grid never steps, such as Ey and Ez in 1D.
      const meep::component c =
          meep::direction_component(electric ? meep::Ex : meep::Hx, diag_direction(gv.dim, d));
      if (!rotated && !gv.has_field(c)) continue;

      const std::vector<meep::lorentzian_pole> poles = medium_poles(suscs, axis[d]);
      if (poles.empty()) continue;
      /* The dispersive field sees its own instantaneous response; the other one
         only scales the coupling. Swapping the two covers the magnetic case
         with the same matrix. */
      const double eps_inf = diag_component(electric ? mm->epsilon_diag : mm->mu_diag, d);
      const double mu_inf = diag_component(electric ? mm->mu_diag : mm->epsilon_diag, d);

      /* A Drude term is marginally stable by construction, so only count growth
         that is clearly above the estimator's own resolution.  Real instabilities
         here run from 1.2x to 8x per step, far above this band. */
      const double margin = 1e-3;

      int worst = -1;
      const double rho =
          meep::dispersive_spectral_radius(eps_inf, mu_inf, poles, dt, dx, naxes, &worst);
      if (rho <= 1 + margin || worst < 0) continue;

      report.growing = true;
      report.electric = electric;
      report.rho = rho;
      report.omega_0 = poles[worst].omega_0;
      report.low_eps_inf = eps_inf < 1;

      // Estimated smallest stable resolution at this Courant factor.
      double lo = 1 / dx, hi = lo;
      while (hi <= stability_report::res_max &&
             meep::dispersive_spectral_radius(eps_inf, mu_inf, poles, Courant / hi, 1 / hi, naxes) >
                 1 + margin)
        hi *= 2;

      /* Leave resolution at zero if refining does not help.  The usual cause is
         an instantaneous epsilon below 1, which no timestep can rescue. */
      if (hi > stability_report::res_max) return report;

      for (int i = 0; i < 20 && hi - lo > 0.5; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (meep::dispersive_spectral_radius(eps_inf, mu_inf, poles, Courant / mid, 1 / mid,
                                             naxes) > 1 + margin)
          lo = mid;
        else
          hi = mid;
      }
      report.resolution = ceil(hi);
      return report; // one verdict per material is enough
    }
  }
  return report;
}

/* Skip a medium already analysed in this call.  Compares by content, since the
   same material can arrive as separate medium_struct instances. */
void check_medium_stability_once(const medium_struct *mm, const meep::grid_volume &gv, double dt,
                                 double Courant, std::vector<const medium_struct *> &checked) {
  for (size_t i = 0; i < checked.size(); ++i)
    if (checked[i] == mm || medium_struct_equal(checked[i], mm)) return;
  checked.push_back(mm);

  const stability_report r = analyze_medium_stability(mm, gv, dt, Courant);
  if (!r.growing) return;

  const char *which = r.electric ? "electric" : "magnetic";
  if (r.resolution > 0)
    meep::master_printf_stderr(
        "warning: dispersive %s material is unstable at dt = %g: its pole at frequency %g grows "
        "%.3gx per timestep in that material.  Use resolution %g or finer at Courant %g, or "
        "reduce Courant.\n",
        which, dt, r.omega_0, r.rho, r.resolution, Courant);
  else
    meep::master_printf_stderr(
        "warning: dispersive %s material is unstable at dt = %g and stays unstable at every "
        "resolution up to %g: its pole at frequency %g grows %.3gx per timestep in that "
        "material.%s\n",
        which, dt, double(stability_report::res_max), r.omega_0, r.rho,
        r.low_eps_inf ? "  Its instantaneous epsilon is less than 1, which FDTD cannot"
                        " timestep stably."
                      : "");
}

} // namespace meep_geom
