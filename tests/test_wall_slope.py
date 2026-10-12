# coding: utf-8
"""Tests for the optional upwind wall slope (wall_slope_model = 1)."""

import tempfile
import unittest
from os.path import abspath, dirname, join

import numpy as np
import sansmic

testdir = dirname(abspath(str(__file__)))


def leach_from_borehole(wall_slope_model, cells=100, days=150):
    """Single ordinary-leach stage from a 1 ft borehole under a blanket."""
    scenario = sansmic.model.Scenario(
        title="wall slope test",
        num_cells=cells,
        cavern_height=600.0,
        floor_depth=4000.0,
        geometry_data={"radii": [0.5] * (cells + 1)},
        geometry_format=sansmic.model.GeometryFormat.RADIUS_LIST,
        units=sansmic.model.Units.FT_IN_BBL,
        insolubles_ratio=0.05,
        advanced=sansmic.model.AdvancedOptions(wall_slope_model=wall_slope_model),
    )
    scenario.stages.append(
        sansmic.model.StageDefinition(
            title="leach",
            simulation_mode=sansmic.model.SimulationMode.ORDINARY,
            solver_timestep=0.1,
            save_frequency=240,
            injection_duration=days * 24,
            rest_duration=0,
            inner_tbg_inside_diam=4.0,
            inner_tbg_outside_diam=4.5,
            outer_csg_inside_diam=8.0,
            outer_csg_outside_diam=8.6,
            brine_injection_sg=1.0,
            brine_injection_depth=3990.0,
            brine_production_depth=3900.0,
            brine_injection_rate=15000.0,
            set_initial_conditions=True,
            set_cavern_sg=1.2,
            brine_interface_depth=3800.0,
            product_injection_rate=0.0,
        )
    )
    return scenario


def run(scenario):
    with tempfile.TemporaryDirectory() as tmp:
        with scenario.new_simulation(join(tmp, "wall"), verbosity=0) as sim:
            sim.run_sim()
    return sim.results


def final_radii(results):
    last = results.df_t_1D.index[-1]
    return results.df_t_z_2D.r.xs(last, level="save_idx").values


class TestWallSlopeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.centered = run(leach_from_borehole(0))
        cls.upwind = run(leach_from_borehole(1))

    def test_default_is_centered(self):
        default = run(leach_from_borehole(None))
        for name in ("df_t_1D", "df_z_1D", "df_t_z_2D"):
            self.assertTrue(
                getattr(default, name).equals(getattr(self.centered, name)), name
            )

    def test_upwind_removes_roof_spike(self):
        # In this case the centered slope leaves the top brine cell much
        # wider than the cell below it, which is itself narrower than its
        # neighbour (a "hat"); the upwind slope keeps the top cells level.
        def top_three(results):
            r = final_radii(results)
            top = np.nonzero(r > 1.0)[0][-1]
            return r[top], r[top - 1], r[top - 2]

        top, below, below2 = top_three(self.centered)
        self.assertGreater(top / below, 1.2)
        self.assertLess(below / below2, 0.95)
        top, below, below2 = top_three(self.upwind)
        self.assertLess(abs(top / below - 1.0), 0.05)
        self.assertLess(abs(below / below2 - 1.0), 0.05)

    def test_volume_changes_little(self):
        v0 = self.centered.df_t_1D.V_cav.iloc[-1]
        v1 = self.upwind.df_t_1D.V_cav.iloc[-1]
        self.assertLess(abs(v1 / v0 - 1.0), 0.02)

    def test_regression_deck_with_upwind_runs(self):
        scenario = sansmic.io.read_scenario(join(testdir, "baseline.dat"))
        scenario.advanced.wall_slope_model = 1
        results = run(scenario)
        self.assertGreater(results.df_t_1D.V_cav.iloc[-1], 0.0)

    def test_invalid_value_rejected(self):
        for value in (2, -1, True):
            with self.assertRaises(ValueError):
                run(leach_from_borehole(value, days=1))

    def test_setting_survives_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            sansmic.io.write_scenario(leach_from_borehole(1), join(tmp, "s.toml"))
            again = sansmic.io.read_scenario(join(tmp, "s.toml"))
        self.assertEqual(again.advanced.wall_slope_model, 1)


if __name__ == "__main__":
    unittest.main()
