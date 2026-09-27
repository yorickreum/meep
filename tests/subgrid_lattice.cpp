#include <stdarg.h>
/* Bookkeeping invariants of the integer lattice when chunks differ in
   resolution.  No fields are involved: grids covering the same region must
   agree where their shared points are, index their own points reversibly, and
   between them own each point exactly once. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <meep.hpp>

using namespace meep;

static int failures = 0;

static void check(bool ok, const char *what, ...) {
  if (!ok) {
    va_list ap;
    va_start(ap, what);
    printf("FAILED: ");
    vprintf(what, ap);
    printf("\n");
    va_end(ap);
    ++failures;
  }
}

/* ivec -> physical must not depend on which grid_volume you ask. */
static void check_shared_locations(const grid_volume &fine, const grid_volume &coarse) {
  const int q = coarse.q();
  ivec p = coarse.little_corner();
  LOOP_OVER_DIRECTIONS(coarse.dim, d) {
    for (int i = 0; i <= coarse.num_direction(d); ++i) {
      ivec pt = p;
      pt.set_direction(d, p.in_direction(d) + i * 2 * q);
      const vec vf = fine[pt], vc = coarse[pt];
      check(fabs(vf.in_direction(d) - vc.in_direction(d)) < 1e-12,
            "location disagrees at lattice %d along %s: fine %g vs coarse %g", pt.in_direction(d),
            direction_name(d), vf.in_direction(d), vc.in_direction(d));
    }
  }
}

/* index() and iloc() must invert each other on every owned point. */
static void check_index_roundtrip(const grid_volume &gv, const char *label) {
  FOR_COMPONENTS(c) {
    if (!gv.has_field(c)) continue;
    LOOP_OVER_VOL(gv, c, idx) {
      IVEC_LOOP_ILOC(gv, here);
      const ptrdiff_t back = gv.index(c, here);
      check(back == idx, "%s: index/iloc roundtrip for %s: idx %td -> %td", label,
            component_name(c), (ptrdiff_t)idx, back);
      if (failures > 5) return;
    }
  }
}

/* Corners must sit on the lattice this grid can actually represent. */
static void check_corner_alignment(const grid_volume &gv, const char *label) {
  const int step = 2 * gv.q();
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    const int lo = gv.little_corner().in_direction(d);
    const int hi = gv.big_corner().in_direction(d);
    check((hi - lo) % step == 0, "%s: extent %d along %s is not a multiple of 2q=%d", label,
          hi - lo, direction_name(d), step);
    check((hi - lo) / step == gv.num_direction(d),
          "%s: extent %d along %s implies %d cells, num_direction says %d", label, hi - lo,
          direction_name(d), (hi - lo) / step, gv.num_direction(d));
  }
}

/* Splitting a grid must partition ownership: no point owned twice, none lost. */
static void check_ownership_partition(const grid_volume &parent, const grid_volume &lo,
                                      const grid_volume &hi, const char *label) {
  FOR_COMPONENTS(c) {
    if (!parent.has_field(c)) continue;
    LOOP_OVER_VOL_OWNED(parent, c, idx) {
      (void)idx;
      IVEC_LOOP_ILOC(parent, here);
      const int n = (lo.owns(here) ? 1 : 0) + (hi.owns(here) ? 1 : 0);
      check(n == 1, "%s: %s at lattice point owned by %d of 2 children", label, component_name(c),
            n);
      if (failures > 5) return;
    }
  }
}

/* Coarse chunk beside a fine one.  Their point sets differ, so the invariant
   is not that every lattice point is owned once -- most belong to neither --
   but that each chunk owns its own points and the neighbour claims none. */
static void check_mixed_adjacency(const grid_volume &lo, const grid_volume &hi, direction d,
                                  const char *label) {
  check(lo.big_corner().in_direction(d) == hi.little_corner().in_direction(d),
        "%s: shared face does not line up: %d vs %d", label, lo.big_corner().in_direction(d),
        hi.little_corner().in_direction(d));

  const double gap = hi.surroundings().in_direction_min(d) - lo.surroundings().in_direction_max(d);
  check(fabs(gap) < 1e-9, "%s: physical regions do not abut, gap %g", label, gap);

  const grid_volume *pair[2] = {&lo, &hi};
  const char *side[2] = {"low", "high"};
  for (int s = 0; s < 2; ++s) {
    const grid_volume &mine = *pair[s], &other = *pair[1 - s];
    FOR_COMPONENTS(c) {
      if (!mine.has_field(c)) continue;
      LOOP_OVER_VOL_OWNED(mine, c, idx) {
        (void)idx;
        IVEC_LOOP_ILOC(mine, here);
        check(mine.owns(here), "%s: %s chunk does not own its own %s point", label, side[s],
              component_name(c));
        check(!other.owns(here), "%s: %s chunk's %s point is also owned by the neighbour", label,
              side[s], component_name(c));
        if (failures > 5) return;
      }
    }
  }
}

/* refine/coarsen must invert; relabeling must move nothing physical. */
static void check_refine_coarsen(double a) {
  char label[64];
  for (int R = 2; R <= 4; R *= 2) {
    snprintf(label, sizeof(label), "refine/coarsen R=%d", R);
    grid_volume base = vol2d(1.0, 1.0, a);
    // A zero origin hides origin bugs: scaling it is then a no-op.
    base.center_origin();
    check(base.little_corner().in_direction(X) != 0,
          "%s: test needs a nonzero origin to be meaningful", label);
    grid_volume relabelled = base.with_lattice_refinement(R);

    check(relabelled.q() == R, "%s: relabelled q is %d", label, relabelled.q());
    check(fabs(relabelled.a - base.a) < 1e-12, "%s: relabelling changed a: %g vs %g", label,
          relabelled.a, base.a);
    check(fabs(relabelled.a_lattice() - base.a * R) < 1e-12,
          "%s: relabelled a_lattice is %g, expected %g", label, relabelled.a_lattice(), base.a * R);
    LOOP_OVER_DIRECTIONS(base.dim, d) {
      check(relabelled.num_direction(d) == base.num_direction(d),
            "%s: relabelling changed the cell count along %s", label, direction_name(d));
    }
    /* the corners must be the same physical points */
    const vec b0 = base[base.little_corner()], r0 = relabelled[relabelled.little_corner()];
    const vec b1 = base[base.big_corner()], r1 = relabelled[relabelled.big_corner()];
    LOOP_OVER_DIRECTIONS(base.dim, d) {
      check(fabs(b0.in_direction(d) - r0.in_direction(d)) < 1e-12,
            "%s: relabelling moved the low corner along %s", label, direction_name(d));
      check(fabs(b1.in_direction(d) - r1.in_direction(d)) < 1e-12,
            "%s: relabelling moved the high corner along %s", label, direction_name(d));
    }

    grid_volume refined = relabelled.refine(R);
    check(refined.q() == 1, "%s: refined q is %d, expected 1", label, refined.q());
    check(fabs(refined.a - base.a * R) < 1e-12, "%s: refined a is %g, expected %g", label,
          refined.a, base.a * R);
    check(fabs(refined.a_lattice() - relabelled.a_lattice()) < 1e-12,
          "%s: refining changed the lattice", label);
    LOOP_OVER_DIRECTIONS(base.dim, d) {
      check(refined.num_direction(d) == base.num_direction(d) * R,
            "%s: refined cell count along %s is %d, expected %d", label, direction_name(d),
            refined.num_direction(d), base.num_direction(d) * R);
    }
    check_corner_alignment(refined, label);
    check_shared_locations(refined, relabelled);

    /* round trip */
    grid_volume back = refined.coarsen(R);
    check(back.q() == relabelled.q() && fabs(back.a - relabelled.a) < 1e-12,
          "%s: coarsen(refine(x)) != x  (q %d vs %d, a %g vs %g)", label, back.q(), relabelled.q(),
          back.a, relabelled.a);
    LOOP_OVER_DIRECTIONS(base.dim, d) {
      check(back.little_corner().in_direction(d) == relabelled.little_corner().in_direction(d),
            "%s: round trip moved the origin along %s", label, direction_name(d));
    }
  }
}

static double one_eps(const vec &) { return 1.0; }

/* A structure whose chunks are not all at one resolution.  Construction runs
   check_chunks(), so a tiling mistake aborts here. */
static void check_mixed_structure(double a) {
  for (int R = 2; R <= 4; R *= 2) {
    char label[64];
    snprintf(label, sizeof(label), "mixed structure R=%d", R);

    binary_partition bp(split_plane{X, 1.0},
                        std::unique_ptr<binary_partition>(new binary_partition(0, 1)),
                        std::unique_ptr<binary_partition>(new binary_partition(0, R)));
    check(bp.max_refinement() == R, "%s: max_refinement is %d", label, bp.max_refinement());

    grid_volume gv = vol2d(2.0, 1.0, a);
    structure s(gv, one_eps, no_pml(), identity(), 0, 0.5, false, DEFAULT_SUBPIXEL_TOL,
                DEFAULT_SUBPIXEL_MAXEVAL, &bp);

    check(fabs(s.a - a) < 1e-12, "%s: structure::a is %g, should stay the base resolution %g",
          label, s.a, a);
    check(fabs(s.dt - 0.5 / (a * R)) < 1e-12, "%s: dt is %g, expected Courant/(a*R) = %g", label,
          s.dt, 0.5 / (a * R));

    int n_base = 0, n_fine = 0;
    for (int i = 0; i < s.num_chunks; ++i) {
      const double ca = s.chunks[i]->gv.a;
      if (fabs(ca - a) < 1e-12)
        ++n_base;
      else if (fabs(ca - a * R) < 1e-12)
        ++n_fine;
      else
        check(false, "%s: chunk %d has unexpected resolution %g", label, i, ca);
      check(fabs(s.chunks[i]->gv.a_lattice() - a * R) < 1e-12,
            "%s: chunk %d disagrees about the lattice", label, i);
    }
    check(n_base > 0 && n_fine > 0, "%s: expected both resolutions, got %d base and %d fine", label,
          n_base, n_fine);
  }
}

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  verbosity = 0;

  const double a = 20.0;

  for (int q = 1; q <= 4; q *= 2) {
    char label[64];

    grid_volume fine = vol2d(1.0, 1.0, a);
    check(fine.q() == 1, "a freshly built grid_volume should have q == 1, got %d", fine.q());

    grid_volume coarse = fine.coarsen(q);
    snprintf(label, sizeof(label), "2d q=%d", q);

    check(coarse.q() == q, "%s: coarsen(%d) gave q=%d", label, q, coarse.q());
    check(fabs(coarse.a - a / q) < 1e-12, "%s: coarsened a is %g, expected %g", label, coarse.a,
          a / q);
    check(fabs(coarse.a_lattice() - a) < 1e-12,
          "%s: a_lattice is %g, expected the finest resolution %g", label, coarse.a_lattice(), a);

    check_corner_alignment(coarse, label);
    check_shared_locations(fine, coarse);
    check_index_roundtrip(coarse, label);

    /* the coarse grid must cover the same physical region as the fine one */
    check(fabs(coarse.xmax() - fine.xmax()) < 1.0 / a, "%s: coarse xmax %g vs fine %g", label,
          coarse.xmax(), fine.xmax());

    if (coarse.num_direction(X) >= 4) {
      grid_volume l = coarse.split_at_fraction(false, coarse.num_direction(X) / 2, X);
      grid_volume h = coarse.split_at_fraction(true, coarse.num_direction(X) / 2, X);
      check(l.q() == q && h.q() == q, "%s: split lost q (%d, %d)", label, l.q(), h.q());
      check_corner_alignment(l, label);
      check_corner_alignment(h, label);
      check_ownership_partition(coarse, l, h, label);
    }
  }

  /* 3d, where the strides and the yee shifts interact */
  for (int q = 1; q <= 2; q *= 2) {
    char label[64];
    snprintf(label, sizeof(label), "3d q=%d", q);
    grid_volume fine = vol3d(0.4, 0.4, 0.4, a);
    grid_volume coarse = fine.coarsen(q);
    check_corner_alignment(coarse, label);
    check_shared_locations(fine, coarse);
    check_index_roundtrip(coarse, label);
  }

  /* coarse beside fine, both orientations, in 2d and 3d */
  for (int r = 2; r <= 4; r *= 2) {
    char label[80];

    grid_volume parent = vol2d(2.0, 1.0, a);
    const int half = parent.num_direction(X) / 2;
    grid_volume fine = parent.split_at_fraction(false, half, X);
    grid_volume coarse = parent.split_at_fraction(true, half, X).coarsen(r);

    snprintf(label, sizeof(label), "2d fine|coarse r=%d", r);
    check(coarse.q() == r, "%s: coarse side has q=%d", label, coarse.q());
    check(fine.q() == 1, "%s: fine side has q=%d", label, fine.q());
    check_corner_alignment(fine, label);
    check_corner_alignment(coarse, label);
    check_mixed_adjacency(fine, coarse, X, label);

    /* and with the coarse side on the low end */
    grid_volume coarse_lo = parent.split_at_fraction(false, half, X).coarsen(r);
    grid_volume fine_hi = parent.split_at_fraction(true, half, X);
    snprintf(label, sizeof(label), "2d coarse|fine r=%d", r);
    check_mixed_adjacency(coarse_lo, fine_hi, X, label);

    /* the two sides must still agree on where shared lattice points are */
    check_shared_locations(fine, coarse);
  }

  {
    grid_volume parent = vol3d(0.8, 0.4, 0.4, a);
    const int half = parent.num_direction(X) / 2;
    grid_volume fine = parent.split_at_fraction(false, half, X);
    grid_volume coarse = parent.split_at_fraction(true, half, X).coarsen(2);
    check_mixed_adjacency(fine, coarse, X, "3d fine|coarse r=2");
  }

  check_refine_coarsen(a);

  check_mixed_structure(a);

  if (failures) {
    printf("%d subgrid lattice invariant(s) failed\n", failures);
    return 1;
  }
  printf("subgrid lattice invariants OK\n");
  return 0;
}
