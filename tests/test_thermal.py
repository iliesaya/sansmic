# coding: utf-8
"""Tests for the optional temperature model (temperature_model_version > 0)."""

import copy
import math
import tempfile
import unittest
from os.path import abspath, dirname, join

import numpy as np
import sansmic

testdir = dirname(abspath(str(__file__)))
LIB = sansmic.libsansmic
K_ROCK = 5.0 * 0.5778  # Btu/hr-ft-degF, the default rock_conductivity
RHO_CP = 2.16 * 62.428 * 880.0 / 4186.8  # Btu/ft3-degF, default salt
ALPHA = K_ROCK / RHO_CP  # ft2/hr


def baseline(version=0, T_inj=None, **advanced):
    scenario = sansmic.io.read_scenario(join(testdir, "baseline.dat"))
    # This extension deliberately rejects native product/interface phase modes.
    # Upstream tests retain their unchanged multi-mode regression deck.
    scenario.stages = scenario.stages[:1]
    scenario.stages[0].simulation_mode = sansmic.model.SimulationMode.ORDINARY
    scenario.stages[0].brine_production_depth = 3700.0
    scenario.stages[0].set_cavern_sg = 1.15
    scenario.advanced.temperature_model_version = version
    for key, value in advanced.items():
        setattr(scenario.advanced, key, value)
    if T_inj is not None:
        scenario.stages[0].brine_injection_temperature = T_inj
    return scenario


def run(scenario):
    with tempfile.TemporaryDirectory() as tmp:
        with scenario.new_simulation(join(tmp, "thermal"), verbosity=0) as sim:
            sim.run_sim()
    return sim.results.df_t_1D


def potter_sg_sat(T_F):
    """Expected saturated sg: SANSMIC 1.2019 at 75 degF shifted along its
    sg(w) curve by the Potter solubility change."""
    c = [1.0019795678, 0.67459890538, 0.32350531044]
    potter = lambda T: 26.18 + 0.0072 * T + 0.000106 * T * T
    w = 0.2632 * potter((T_F - 32) / 1.8) / potter((75.0 - 32) / 1.8)
    quad = lambda y: c[0] + c[1] * y + c[2] * y * y
    return 1.2019 + quad(w) - quad(0.2632)


class TestTemperatureModelOff(unittest.TestCase):
    def test_default_outputs_are_constant(self):
        df = run(baseline())
        self.assertTrue((df.T_brine == 75.0).all())
        self.assertTrue((df.sg_sat == 1.2019).all())
        for name in ("Q_rock", "H_inj", "H_rock", "H_diss"):
            self.assertTrue((df[name] == 0.0).all(), name)


class TestTemperatureModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = run(baseline())

    def test_isothermal_75F_reproduces_default(self):
        """All temperatures at 75 degF and no heat of solution give the
        default model (to round-off)."""
        for version in (1, 2):
            df = run(baseline(version, T_inj=75.0, heat_of_solution=0.0))
            for name in ("V_cav", "sg_out", "sg_ave", "V_insol", "z_obi", "V_inj"):
                err = np.max(np.abs(df[name] - self.off[name]) / np.abs(self.off[name]))
                self.assertLess(err, 1.0e-6, (version, name))
            self.assertLess(np.max(np.abs(df.T_brine - 75.0)), 1.0e-9)

    def test_uniform_temperature_stays_uniform(self):
        df = run(baseline(2, T_inj=140.0, rock_temperature=140.0, heat_of_solution=0.0))
        self.assertLess(np.max(np.abs(df.T_brine - 140.0)), 1.0e-9)
        self.assertAlmostEqual(df.sg_sat.iloc[-1], 1.2019, places=12)

    def test_heat_of_solution_cools_brine(self):
        df = run(baseline(2, T_inj=100.0, rock_temperature=100.0))
        self.assertLess(df.T_brine.min(), 100.0 - 0.1)
        self.assertGreater(df.T_brine.min(), 90.0)
        self.assertGreater(df.H_diss.iloc[-1], 0.0)
        self.assertGreater(df.H_rock.iloc[-1], 0.0)  # rock warms cooled brine

    def test_caloric_inventory_residual_is_recorded(self):
        df = run(baseline(2, T_inj=86.0, rock_temperature=140.0))
        self.assertTrue(np.isfinite(df.E_balance).all())
        self.assertLess(np.max(np.abs(df.E_balance)), 0.1)

    def test_hot_concentrated_inlet_is_not_silently_diluted(self):
        s = baseline(2, T_inj=140.0, rock_temperature=50.0)
        s.stages[0].brine_injection_sg = 1.204
        with self.assertRaisesRegex(ValueError, "composition was not changed"):
            run(s)
        self.assertEqual(s.stages[0].brine_injection_sg, 1.204)

    def test_evaluated_enthalpy_reconstructs_integral_reference(self):
        # Primary Table 25A: m=0 and 6 at 25 C, evaluated Q J/mol.
        for m, Q in ((0.0, 3858.4), (6.0, 1869.3)):
            w = m * 0.058443 / (1 + m * 0.058443)
            T = 77.0
            sensible = ((1 - w) + w * 880 / 4186.8) * (T - 75)
            expected = sensible + w * Q / 58.443 / 2.326
            self.assertAlmostEqual(LIB.thermal_brine_enthalpy(T, w), expected, places=8)

    def test_layer_inventory_retains_nonlinear_salinity_variation(self):
        masses = [1e5, 2e5]
        fractions = [0.03, 0.26]
        curve = LIB.thermal_energy_curve(masses, fractions)
        for T in (75.0, 86.0, 104.0, 122.0, 140.0):
            exact = sum(
                m * LIB.thermal_brine_enthalpy(T, w) for m, w in zip(masses, fractions)
            )
            self.assertAlmostEqual(LIB.thermal_energy_at(T, curve), exact, delta=1e-8)
        mean = sum(m * w for m, w in zip(masses, fractions)) / sum(masses)
        bulk = sum(masses) * LIB.thermal_brine_enthalpy(86, mean)
        self.assertGreater(abs(bulk - LIB.thermal_energy_at(86, curve)), 1000.0)

    def test_closed_dissolution_uses_enthalpy_not_fixed_sink(self):
        A = 1e4
        M0 = 1e5
        w0 = 0.1
        Ms = 100
        M1 = M0 + Ms
        w1 = (M0 * w0 + Ms) / M1
        Tb = 86.0
        cp = 880 / 4186.8
        rock = LIB.CThermalRock(Tb)
        E0 = M0 * LIB.thermal_brine_enthalpy(Tb, w0) + rock.content(A, RHO_CP)
        T1 = rock.step_enthalpy(
            0.1,
            A,
            0,
            K_ROCK,
            RHO_CP,
            0,
            Tb,
            M0,
            w0,
            M1,
            w1,
            0,
            0,
            Tb,
            Ms,
            0,
            0,
            cp,
            float("nan"),
        )
        E1 = M1 * LIB.thermal_brine_enthalpy(T1, w1) + rock.content(A, RHO_CP)
        incoming_solid = Ms * cp * (rock.T[0] - 75)
        self.assertLess(T1, Tb)
        self.assertLess(abs(E1 - E0 - incoming_solid), 1e-4)
        self.assertLess(abs(rock.energy_residual), 1e-4)

    def test_warm_water_dissolves_more(self):
        for version in (1, 2):
            cool = run(baseline(version, T_inj=77.0, rock_temperature=77.0))
            warm = run(baseline(version, T_inj=113.0, rock_temperature=113.0))
            grown = lambda df: df.V_cav.iloc[-1] - df.V_cav.iloc[0]
            self.assertGreater(grown(warm), grown(cool) * 1.05, version)

    def test_hot_rock_warms_cold_brine(self):
        df = run(
            baseline(
                2,
                T_inj=80.0,
                rock_temperature=140.0,
                initial_brine_temperature=80.0,
                heat_of_solution=0.0,
            )
        )
        self.assertGreater(df.T_brine.iloc[-1], 80.0)
        self.assertLess(df.T_brine.max(), 140.0)
        self.assertGreater(df.H_rock.iloc[-1], 0.0)

    def test_cooling_precipitation_is_rejected_without_inventory_clipping(self):
        # brine saturated at 140 degF, cooled by injecting saturated 50 degF
        # brine (no dilution) into 50 degF rock
        scenario = baseline(
            2,
            T_inj=50.0,
            rock_temperature=50.0,
            initial_brine_temperature=140.0,
            thermal_saturation_model=1,
        )
        scenario.stages[0].set_cavern_sg = potter_sg_sat(140.0)
        scenario.stages[0].brine_injection_sg = potter_sg_sat(50.0)
        with self.assertRaisesRegex(ValueError, "precipitation"):
            run(scenario)

    def test_first_stage_needs_injection_temperature(self):
        with self.assertRaises(ValueError):
            run(baseline(1))

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            run(baseline(3, T_inj=75.0))
        with self.assertRaises(ValueError):
            run(baseline(1, T_inj=75.0, rock_conductivity=-1.0))

    def test_staged_cavern_caloric_ledger_closes_with_explicit_reset_allowance(self):
        s = baseline(2, T_inj=86.0, rock_temperature=140.0)
        first = s.stages[0]
        first.injection_duration = 24
        first.rest_duration = 0
        second = __import__("copy").deepcopy(first)
        second.set_initial_conditions = True
        second.set_cavern_sg = 1.1
        second.brine_interface_depth = 3600.0
        second.brine_injection_temperature = 104.0
        s.stages.append(second)
        df = run(s)
        rhs = df.H_stage + df.H_inj + df.H_rock + df.H_native_correction + df.H_residual
        self.assertLess(float(np.max(np.abs(df.E_brine - rhs))), 1e-8)
        self.assertGreater(abs(df.H_stage.iloc[-1] - df.H_stage.iloc[1]), 0.01)

    def test_settings_survive_toml(self):
        scenario = baseline(
            2, T_inj=95.0, rock_temperature=140.0, heat_of_solution=10.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            sansmic.io.write_scenario(scenario, join(tmp, "s.toml"))
            again = sansmic.io.read_scenario(join(tmp, "s.toml"))
        self.assertEqual(again.advanced.temperature_model_version, 2)
        self.assertEqual(again.advanced.rock_temperature, 140.0)
        self.assertEqual(again.advanced.heat_of_solution, 10.0)
        self.assertEqual(again.stages[0].brine_injection_temperature, 95.0)


class TestThermalRock(unittest.TestCase):
    """Rock conduction grid against analytic solutions."""

    def test_sphere_constant_wall_temperature(self):
        # q = k A dT (1/R + 1/sqrt(pi alpha t)) (Carslaw & Jaeger, sphere)
        for R in (5.0, 50.0):
            A = 4.0 * math.pi * R * R
            rock = LIB.CThermalRock(100.0)
            t, dt = 0.0, 0.01
            for check in (10.0, 100.0, 1000.0, 10000.0):
                while t < check - 1e-9:
                    step = min(dt, check - t)
                    rock.step(
                        step,
                        A,
                        0.0,
                        K_ROCK,
                        RHO_CP,
                        0.0,
                        50.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    t += step
                    dt = min(dt * 1.02, 1.0)
                exact = K_ROCK * A * 50.0 * (1 / R + 1 / math.sqrt(math.pi * ALPHA * t))
                self.assertLess(abs(rock.q_wall / exact - 1.0), 0.005, (R, t))

    def test_receding_wall_steady_flux(self):
        # steady planar wall receding at v: q/A = rho cp v dT
        A = 4.0 * math.pi * 1.0e12
        for v in (0.005, 0.01, 0.02):
            rock = LIB.CThermalRock(100.0)
            for _ in range(40000):
                rock.step(1.0, A, v, K_ROCK, RHO_CP, 0.0, 50.0, -1.0, 0, 0, 0, 0)
            self.assertLess(abs(rock.q_wall / A / (RHO_CP * v * 50.0) - 1.0), 0.005)

    def test_wall_film_coefficient(self):
        # steady receding wall with a film: q/A = dT / (1/h + 1/(rho cp v))
        A, v, h = 4.0 * math.pi * 1.0e12, 0.01, 0.5
        rock = LIB.CThermalRock(100.0)
        for _ in range(40000):
            rock.step(1.0, A, v, K_ROCK, RHO_CP, h, 50.0, -1.0, 0, 0, 0, 0)
        exact = 50.0 / (1.0 / h + 1.0 / (RHO_CP * v))
        self.assertLess(abs(rock.q_wall / A / exact - 1.0), 0.005)

    def test_closed_exchange_conserves_energy(self):
        R = 30.0
        A = 4.0 * math.pi * R * R
        C_b = 4.0 / 3.0 * math.pi * R**3 * 1.2 * 62.428 * 0.79
        rock = LIB.CThermalRock(140.0)
        T_b = 80.0
        e0 = C_b * T_b + rock.content(A, RHO_CP)
        for _ in range(2000):
            T_b = rock.step(1.0, A, 0.0, K_ROCK, RHO_CP, 0.0, T_b, C_b, 0, 0, 0, 0)
        e1 = C_b * T_b + rock.content(A, RHO_CP)
        self.assertLess(abs(e1 - e0) / e0, 1.0e-12)
        self.assertGreater(T_b, 80.0)

    def test_open_dissolving_receding_grid_energy_budget(self):
        # Independent integral budget, including the wall-frame advection,
        # far-boundary heat and change in equivalent-sphere grid volume.
        rock = LIB.CThermalRock(140.0)
        cp = 880 / 4186.8
        Tb = 90.0
        M = 1e6
        w = 0.1
        old_A = 1e4
        xi = np.array(rock.centres())
        faces = [0.0]
        for centre in xi:
            faces.append(2 * centre - faces[-1])
        faces = np.array(faces)
        widths = np.diff(faces)
        for A in (1e4, 1.01e4, 1.02e4):
            old = np.array(rock.T)
            dt = 0.1
            v = 0.01
            solid = 2.16 * 62.428 * A * v * dt
            injected = 100.0
            returned = 100.0
            newM = M + solid + injected - returned
            neww = (M * w + solid + injected * 0.04 - returned * 0.15) / newM
            Tnew = rock.step_enthalpy(
                dt,
                A,
                v,
                K_ROCK,
                RHO_CP,
                0,
                Tb,
                M,
                w,
                newM,
                neww,
                injected,
                0.04,
                86.0,
                solid,
                returned,
                0.15,
                cp,
                float("nan"),
            )
            new = np.array(rock.T)

            def volumes(area):
                R = math.sqrt(area / (4 * math.pi))
                return 4 * math.pi / 3 * np.diff((R + faces) ** 3)

            Vold, Vnew = volumes(old_A), volumes(A)
            R = math.sqrt(A / (4 * math.pi))
            outer = (
                K_ROCK
                * 4
                * math.pi
                * (R + faces[-1]) ** 2
                / (faces[-1] - xi[-1])
                * (140 - new[-1])
            )
            adv = np.sum(RHO_CP * v * Vnew / widths * (np.r_[new[1:], 140.0] - new))
            geometry = np.sum(RHO_CP * (Vnew - Vold) * (old - 75))
            E0 = M * LIB.thermal_brine_enthalpy(Tb, w) + np.sum(
                RHO_CP * Vold * (old - 75)
            )
            E1 = newM * LIB.thermal_brine_enthalpy(Tnew, neww) + np.sum(
                RHO_CP * Vnew * (new - 75)
            )
            boundary = (
                injected * LIB.thermal_brine_enthalpy(86, 0.04)
                - returned * LIB.thermal_brine_enthalpy(Tnew, 0.15)
                + solid * cp * (new[0] - 75)
                + dt * (outer + adv)
                + geometry
            )
            self.assertLess(abs(E1 - E0 - boundary), 1e-6 * max(1, abs(boundary)))
            self.assertLess(abs(rock.energy_residual), 1e-3)
            Tb, M, w, old_A = Tnew, newM, neww, A

    def test_brine_balance(self):
        A, C_b, G_in, Q_src, dt = 1.0e4, 1.0e7, 500.0, -2000.0, 0.5
        rock = LIB.CThermalRock(100.0)
        T_new = rock.step(
            dt, A, 0.0, K_ROCK, RHO_CP, 0.0, 100.0, C_b, G_in, 60.0, 0.0, Q_src
        )
        lhs = C_b * (T_new - 100.0) / dt
        rhs = G_in * (60.0 - T_new) + rock.q_wall + Q_src
        self.assertLess(abs(lhs - rhs), 1.0e-6 * abs(lhs))


class TestPrecipitationCheck(unittest.TestCase):
    """The precipitation check must cover every layer that holds brine in the
    enthalpy inventory, including a partly wetted interface cell."""

    def test_partly_wetted_interface_cell_is_checked(self):
        sg = [1.19, 1.19, 1.2019]  # top cell saturated at 75 degF
        sat_50F = 1.20064  # saturated sg at 50 degF (Potter, scaled)
        self.assertEqual(
            LIB.thermal_first_supersaturated(sg, [1.0, 1.0, 0.3], sat_50F), 2
        )

    def test_model_rejects_supersaturated_interface_cell(self):
        # Cells above the blanket are reset to saturation every step. While
        # injection cools the brine, raising the blanket by one cell in a new
        # stage leaves the newly wetted, partly filled interface cell slightly
        # above the (now lower) saturation, while the brine below is diluted.
        cells, height, floor = 100, 600.0, 4000.0
        dz = height / cells
        scenario = sansmic.model.Scenario(
            title="interface cell",
            num_cells=cells,
            cavern_height=height,
            floor_depth=floor,
            geometry_data={"radii": [20.0] * (cells + 1)},
            geometry_format=sansmic.model.GeometryFormat.RADIUS_LIST,
            units=sansmic.model.Units.FT_IN_BBL,
            insolubles_ratio=0.0,
            advanced=sansmic.model.AdvancedOptions(
                temperature_model_version=2,
                rock_temperature=75.0,
                thermal_saturation_model=1,
            ),
        )
        first = sansmic.model.StageDefinition(
            title="cool",
            simulation_mode=sansmic.model.SimulationMode.ORDINARY,
            solver_timestep=0.1,
            save_frequency=10,
            injection_duration=24.0,
            rest_duration=0,
            inner_tbg_inside_diam=4.0,
            inner_tbg_outside_diam=4.5,
            outer_csg_inside_diam=8.0,
            outer_csg_outside_diam=8.6,
            brine_injection_sg=1.0,
            brine_injection_depth=3990.0,
            brine_production_depth=3900.0,
            brine_injection_rate=20000.0,
            brine_injection_temperature=60.0,
            set_initial_conditions=True,
            set_cavern_sg=1.2,
            brine_interface_depth=floor - 79.3 * dz,  # interface cell 80
            product_injection_rate=0.0,
        )
        second = copy.deepcopy(first)
        second.set_initial_conditions = False
        second.set_cavern_sg = None
        second.brine_interface_depth = floor - 80.3 * dz  # interface cell 81
        scenario.stages.extend([first, second])
        with self.assertRaisesRegex(ValueError, "cell=81 "):
            run(scenario)

    def test_dry_layers_and_undersaturated_brine_pass(self):
        sat_50F = 1.20064
        self.assertEqual(
            LIB.thermal_first_supersaturated(
                [1.19, 1.19, 1.2019], [1.0, 1.0, 0.0], sat_50F
            ),
            -1,
        )
        self.assertEqual(
            LIB.thermal_first_supersaturated(
                [1.19, 1.20, 1.2006], [1.0, 1.0, 1.0], sat_50F
            ),
            -1,
        )


class TestUnresolvedStep(unittest.TestCase):
    def test_step_larger_than_cavity_stops_with_advice(self):
        # 1 ft borehole (about 84 bbl); 45,000 bbl/d injects about 190 bbl per
        # 0.1 h step, so SANSMIC's first step is not mass-consistent.
        cells = 100
        scenario = sansmic.model.Scenario(
            title="unresolved first step",
            num_cells=cells,
            cavern_height=600.0,
            floor_depth=4000.0,
            geometry_data={"radii": [0.5] * (cells + 1)},
            geometry_format=sansmic.model.GeometryFormat.RADIUS_LIST,
            units=sansmic.model.Units.FT_IN_BBL,
            insolubles_ratio=0.05,
            advanced=sansmic.model.AdvancedOptions(
                temperature_model_version=2, rock_temperature=140.0
            ),
        )
        scenario.stages.append(
            sansmic.model.StageDefinition(
                title="leach",
                simulation_mode=sansmic.model.SimulationMode.ORDINARY,
                solver_timestep=0.1,
                save_frequency=10,
                injection_duration=24.0,
                rest_duration=0,
                inner_tbg_inside_diam=4.0,
                inner_tbg_outside_diam=4.5,
                outer_csg_inside_diam=8.0,
                outer_csg_outside_diam=8.6,
                brine_injection_sg=1.0,
                brine_injection_depth=3990.0,
                brine_production_depth=3900.0,
                brine_injection_rate=45000.0,
                brine_injection_temperature=86.0,
                set_initial_conditions=True,
                set_cavern_sg=1.2,
                brine_interface_depth=3800.0,
                product_injection_rate=0.0,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Reduce the timestep"):
            run(scenario)


class TestOpenTankAnalytic(unittest.TestCase):
    """Well-mixed tank with negligible rock exchange, against solutions derived
    by hand (independent of the C++ enthalpy and ledger code)."""

    CP_SALT = 880.0 / 4186.8
    A = 1.0e4
    K_TINY = 1.0e-20  # rock conduction effectively off

    def step(self, rock, dt, T, M, w, M_in, w_in, T_in, M_salt, Q):
        M_out = M_in + M_salt  # constant brine mass
        return rock.step_enthalpy(
            dt,
            self.A,
            0.0,
            self.K_TINY,
            RHO_CP,
            0.0,
            T,
            M,
            w,
            M,
            w,
            M_in,
            w_in,
            T_in,
            M_salt,
            M_out,
            w,
            self.CP_SALT,
            Q,
        )

    def test_flushing_tank(self):
        # Same salinity in and out, no dissolution: M dT/dt = mdot (T_in - T),
        # so T - T_in = (T0 - T_in) exp(-mdot t / M); backward Euler gives
        # (T0 - T_in) (1 + mdot dt / M)^-n exactly.
        M, mdot, w, T0, T_in, dt, n = 1.0e6, 2.0e4, 0.2, 140.0, 80.0, 0.5, 400
        rock = LIB.CThermalRock(T0)
        T = T0
        for _ in range(n):
            T = self.step(rock, dt, T, M, w, mdot * dt, w, T_in, 0.0, 13.0)
        discrete = T_in + (T0 - T_in) * (1.0 + mdot * dt / M) ** (-n)
        exact = T_in + (T0 - T_in) * math.exp(-mdot * dt * n / M)
        self.assertLess(abs(T - discrete), 1.0e-7)
        # first-order time error, bounded by (T0 - T_in) * k dt / 2 here
        self.assertLess(abs(T - exact), (T0 - T_in) * (mdot * dt / M) / 2.0)

    def test_steady_dissolving_tank(self):
        # Fresh water in at T_in, salt dissolving at mdot_s from rock at T_w,
        # constant heat of solution Q (Btu/lb). Steady state at constant mass:
        # w = s / (f + s) and
        # (f + s) [((1 - w) + w cp) (T - 75) + w Q] = f (T_in - 75)
        #                                           + s cp (T_w - 75).
        f, s, Q, T_in, T_w, M, dt = 2.0e4, 6.0e3, 13.0, 80.0, 140.0, 1.0e6, 1.0
        cp = self.CP_SALT
        w = s / (f + s)
        steady = 75.0 + (
            f * (T_in - 75.0) + s * cp * (T_w - 75.0) - (f + s) * w * Q
        ) / ((f + s) * ((1.0 - w) + w * cp))
        rock = LIB.CThermalRock(T_w)
        T = 100.0
        for _ in range(3000):
            T = self.step(rock, dt, T, M, w, f * dt, 0.0, T_in, s * dt, Q)
        self.assertLess(abs(T - steady), 1.0e-6)
        self.assertLess(steady, T_in)  # dissolution cools below the inlet here


if __name__ == "__main__":
    unittest.main()
