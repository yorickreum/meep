import unittest
import warnings

import meep as mp


class TestMeshRefinement(unittest.TestCase):
    """The Python side of per-chunk refinement: a leaf resolution becomes an
    integer factor, a region becomes a partition, and everything the user cannot
    have predicted gets said out loud."""

    def test_leaf_resolution_becomes_a_factor(self):
        bp = mp.BinaryPartition(
            split_dir=mp.X,
            split_pos=0.0,
            left=mp.BinaryPartition(proc_id=0),
            right=mp.BinaryPartition(proc_id=1, resolution=60),
        )
        bp._resolve_refinement(20)
        self.assertEqual([l.refine_factor for l in bp._leaves()], [1, 3])

    def test_non_integer_resolution_snaps_up_and_warns(self):
        bp = mp.BinaryPartition(proc_id=0, resolution=51)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bp._resolve_refinement(20)
        self.assertEqual(bp.refine_factor, 3)
        self.assertTrue(any("snapping up" in str(x.message) for x in w))

    def test_coarsening_is_refused(self):
        bp = mp.BinaryPartition(proc_id=0, resolution=5)
        with self.assertRaises(ValueError):
            bp._resolve_refinement(20)

    def test_plain_tree_is_unrefined(self):
        bp = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
        bp._resolve_refinement(20)
        self.assertEqual([l.refine_factor for l in bp._leaves()], [1, 1])

    def test_region_becomes_a_partition(self):
        sim = mp.Simulation(
            cell_size=mp.Vector3(4, 4),
            resolution=10,
            mesh_refinement=[
                mp.Refinement(
                    center=mp.Vector3(1, 0), size=mp.Vector3(1, 1), resolution=30
                )
            ],
        )
        sim.init_sim()
        factors = sorted(l.refine_factor for l in sim.chunk_layout._leaves())
        self.assertEqual(factors, [1, 1, 1, 1, 3])

    def test_two_disjoint_regions(self):
        sim = mp.Simulation(
            cell_size=mp.Vector3(6, 2),
            resolution=10,
            mesh_refinement=[
                mp.Refinement(
                    center=mp.Vector3(-2, 0), size=mp.Vector3(1, 1), resolution=20
                ),
                mp.Refinement(
                    center=mp.Vector3(2, 0), size=mp.Vector3(1, 1), resolution=20
                ),
            ],
        )
        sim.init_sim()
        factors = sorted(l.refine_factor for l in sim.chunk_layout._leaves())
        self.assertEqual(factors[-2:], [2, 2])

    def test_mixed_refinement_resolutions_are_refused(self):
        with self.assertRaises(ValueError):
            mp.Simulation(
                cell_size=mp.Vector3(6, 2),
                resolution=10,
                mesh_refinement=[
                    mp.Refinement(
                        center=mp.Vector3(-2, 0), size=mp.Vector3(1, 1), resolution=20
                    ),
                    mp.Refinement(
                        center=mp.Vector3(2, 0), size=mp.Vector3(1, 1), resolution=40
                    ),
                ],
            )

    def test_overlapping_regions_are_refused(self):
        with self.assertRaises(ValueError):
            mp.Simulation(
                cell_size=mp.Vector3(4, 4),
                resolution=10,
                mesh_refinement=[
                    mp.Refinement(
                        center=mp.Vector3(), size=mp.Vector3(2, 2), resolution=20
                    ),
                    mp.Refinement(
                        center=mp.Vector3(0.5, 0), size=mp.Vector3(1, 1), resolution=40
                    ),
                ],
            )

    def test_chunk_layout_and_mesh_refinement_conflict(self):
        with self.assertRaises(ValueError):
            mp.Simulation(
                cell_size=mp.Vector3(4, 4),
                resolution=10,
                chunk_layout=mp.BinaryPartition(proc_id=0),
                mesh_refinement=[
                    mp.Refinement(
                        center=mp.Vector3(), size=mp.Vector3(1, 1), resolution=20
                    )
                ],
            )

    def test_snapping_counts_from_the_cell_corner(self):
        """27 coarse cells across: the grid lines are not at multiples of the step
        from 0, and a band the full width of the cell must still fit."""
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.7, 2),
            resolution=10,
            mesh_refinement=[
                mp.Refinement(
                    center=mp.Vector3(), size=mp.Vector3(2.7, 0.4), resolution=40
                )
            ],
        )
        lo, hi = mp.simulation._snap_refinement_box(
            sim.mesh_refinement[0], 10, mp.Vector3(-1.35, -1)
        )
        self.assertAlmostEqual(lo.x, -1.35)
        self.assertAlmostEqual(hi.x, 1.35)
        self.assertAlmostEqual(lo.y, -0.2)
        self.assertAlmostEqual(hi.y, 0.2)
        sim.init_sim()

    def test_symmetries_are_refused(self):
        """A refining partition tiles the whole cell, not the symmetry-reduced
        one; with a mirror the energies came out 37% off."""
        with self.assertRaises(ValueError):
            mp.Simulation(
                cell_size=mp.Vector3(4, 4),
                resolution=10,
                symmetries=[mp.Mirror(mp.Y)],
                mesh_refinement=[
                    mp.Refinement(
                        center=mp.Vector3(), size=mp.Vector3(1, 1), resolution=20
                    )
                ],
            )

    def test_region_outside_the_cell_is_refused(self):
        with self.assertRaises(ValueError):
            mp.Simulation(
                cell_size=mp.Vector3(2, 2),
                resolution=10,
                mesh_refinement=[
                    mp.Refinement(
                        center=mp.Vector3(0.8, 0), size=mp.Vector3(1, 1), resolution=20
                    )
                ],
            )

    def test_refined_run_tracks_a_uniform_one(self):
        """The refined answer has to be near the uniform one; this is a smoke test
        for the plumbing, not a convergence claim -- tests/subgrid_convergence.cpp
        makes that one."""

        def energy(mesh_refinement):
            sim = mp.Simulation(
                cell_size=mp.Vector3(4, 2),
                resolution=20,
                mesh_refinement=mesh_refinement,
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.5, fwidth=0.2),
                        component=mp.Ez,
                        center=mp.Vector3(-1, 0),
                    )
                ],
            )
            sim.run(until=8)
            return sim.field_energy_in_box(
                mp.Volume(center=mp.Vector3(), size=sim.cell_size)
            )

        plain = energy(None)
        refined = energy(
            [
                mp.Refinement(
                    center=mp.Vector3(1, 0), size=mp.Vector3(1, 1), resolution=40
                )
            ]
        )
        self.assertGreater(plain, 0)
        self.assertLess(abs(refined - plain) / plain, 0.2)

    def test_closed_refined_cell_conserves_energy(self):
        """Metal walls, no loss, source off: the energy beats (E and synchronized
        H are half a step apart) but must not drift, so it stays within the
        uniform run's beat."""

        def spread(mesh_refinement):
            sim = mp.Simulation(
                cell_size=mp.Vector3(4, 2),
                resolution=10,
                mesh_refinement=mesh_refinement,
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.5, fwidth=0.5),
                        component=mp.Ez,
                        center=mp.Vector3(-1, 0.1),
                    )
                ],
            )
            box = mp.Volume(center=mp.Vector3(), size=sim.cell_size)
            sim.run(until=25)
            e0 = sim.field_energy_in_box(box)
            worst = 0
            for _ in range(20):
                sim.run(until=10)
                worst = max(worst, abs(sim.field_energy_in_box(box) / e0 - 1))
            return worst

        uniform = spread(None)
        refined = spread(
            [
                mp.Refinement(
                    center=mp.Vector3(0.5, 0), size=mp.Vector3(1, 1), resolution=20
                )
            ]
        )
        self.assertLess(refined, 2 * uniform + 1e-6)


if __name__ == "__main__":
    unittest.main()
