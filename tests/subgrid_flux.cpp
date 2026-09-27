/* DFT flux through a refined box, with the conforming rows.

   A beam in vacuum, PML all round, crosses a refined box.  The rows conserve
   energy, so the net flux through a closed surface around the box is what it
   is on the uniform grid: ~0 in TE, and in TM the same corner artefact of the
   Ez flux planes.  (Less of the beam leaves at the far side than on the
   uniform grid: the box is a weak phase object, its dispersion being the
   fine grid's; that is scattering, second order, and the balance holds.) */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <meep.hpp>

using namespace meep;

static int failures = 0;
static double one_eps(const vec &) { return 1.0; }
static std::unique_ptr<binary_partition> leaf(int &pid, int r) {
  return std::unique_ptr<binary_partition>(new binary_partition(pid++, r));
}
// 8 x 4 cell, box [3, 5] x [1.5, 2.5] refined by 2
static std::unique_ptr<binary_partition> box(int &pid) {
  std::unique_ptr<binary_partition> ybox(
      new binary_partition(split_plane{Y, 1.5}, leaf(pid, 1),
                           std::unique_ptr<binary_partition>(new binary_partition(
                               split_plane{Y, 2.5}, leaf(pid, 2), leaf(pid, 1)))));
  std::unique_ptr<binary_partition> right(
      new binary_partition(split_plane{X, 5.0}, std::move(ybox), leaf(pid, 1)));
  return std::unique_ptr<binary_partition>(
      new binary_partition(split_plane{X, 3.0}, leaf(pid, 1), std::move(right)));
}

// net flux into [2.5, 5.5] x [1.1, 2.9] over the flux in at the left, at f = 0.8, 1.0, 1.2
static std::vector<double> balance(bool refined, component c) {
  const int nf = 3;
  grid_volume gv = vol2d(8.0, 4.0, 16);
  int pid = 0;
  std::unique_ptr<binary_partition> bp(refined ? box(pid) : nullptr);
  structure s(gv, one_eps, pml(1.0), identity(), bp ? 0 : count_processors(), refined ? 0.5 : 0.25,
              false, DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL, bp.get());
  fields f(&s);
  gaussian_src_time src(1.0, 0.5);
  f.add_volume_source(c, src, volume(vec(1.5, 1.2), vec(1.5, 2.8)));
  const double x0 = 2.5, x1 = 5.5, y0 = 1.1, y1 = 2.9;
  const volume side[4] = {volume(vec(x0, y0), vec(x0, y1)), volume(vec(x1, y0), vec(x1, y1)),
                          volume(vec(x0, y0), vec(x1, y0)), volume(vec(x0, y1), vec(x1, y1))};
  std::vector<dft_flux> fl;
  for (const volume &v : side)
    fl.push_back(f.add_dft_flux_plane(v, 0.8, 1.2, nf));
  while (f.time() < 50)
    f.step();
  std::vector<double *> F;
  for (dft_flux &d : fl)
    F.push_back(d.flux());
  std::vector<double> out;
  for (int j = 0; j < nf; ++j)
    out.push_back((F[0][j] - F[1][j] + F[2][j] - F[3][j]) / F[0][j]);
  for (double *p : F)
    delete[] p;
  return out;
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;
  master_printf("flux balance on a closed box around a refined region:\n");
  for (component c : {Hz, Ez}) {
    // the uniform grid at the fine dt, which a refined run steps with
    const std::vector<double> uniform = balance(false, c), rows = balance(true, c);
    for (size_t j = 0; j < rows.size(); ++j) {
      const double d = rows[j] - uniform[j];
      const bool ok = std::fabs(d) < 1e-3;
      master_printf("  %s f = %.1f: rows %+.2e, uniform %+.2e, difference %+.1e  %s\n",
                    c == Ez ? "Ez" : "Hz", 0.8 + 0.2 * j, rows[j], uniform[j], d,
                    ok ? "ok" : "FAILED");
      failures += !ok;
    }
  }
  if (failures) {
    master_printf("%d flux check(s) failed\n", failures);
    return 1;
  }
  master_printf("subgrid flux OK\n");
  return 0;
}
