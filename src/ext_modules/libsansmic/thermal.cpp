// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file thermal.cpp
 * @brief Optional brine and rock energy balance (temperature model).
 *
 * Enabled with temperature_model_version 1 or 2. With the default, 0, none of
 * these functions run and SANSMIC keeps its fixed 75 degF behaviour.
 *
 * Brine energy
 * ------------
 * All brine below the blanket has one temperature T. Its enthalpy (Btu,
 * relative to water and halite at 75 degF) is the sum over native layers of
 *
 *     H_layer = M [ (1 - w) + w cp_s ] (T - 75) + M w Q(T, m)
 *
 * with M the layer mass, w the NaCl weight fraction, cp_s the halite specific
 * heat (water cp = 1 Btu/lb-degF) and Q the integral heat of solution per lb
 * NaCl at molality m, interpolated in the Clarke & Glew (1985) Table 25A
 * (10-60 degC, atmospheric pressure; caloric_data.hpp). The table ends at the
 * saturation molality of each temperature (6.106-6.341 mol/kg); above 6 mol/kg
 * the last table segment is extended linearly, and states above 6.4 mol/kg are
 * rejected. Layer masses are scaled to SANSMIC's tracked brine mass. Over a
 * step,
 *
 *     H_new(T) - dH_native(T) + M_out h(T, w_out) - H_old - M_in h(T_in, w_in)
 *         - M_salt cp_s (T_wall - 75) - Q_rock dt = 0,
 *
 * solved for T, where dH_native(T) = M_new [h(T, w_new) - h(T, w_conserved)].
 * w_new is the mean salinity of SANSMIC's corrected concentration field and
 * w_conserved the mean salinity implied by the salt and mass ledger
 * (old salt + inflow salt + dissolved salt - outflow salt). dH_native is a
 * bulk-composition allowance for SANSMIC's numerical salt difference, at the
 * mean salinity, unlike the layered inventory H_new. M_out is the mass leaving
 * according to SANSMIC's own mass ledger and w_out the production-cell
 * salinity. Dissolved salt enters at the wall temperature; its chemical heat is
 * in H_new through Q. A constant heat_of_solution replaces Q for diagnostics
 * only.
 *
 * SANSMIC's mass-balance correction (CFAC) adjusts concentrations without
 * conserving a separately reconstructed salt inventory exactly. That numerical
 * salt difference and its enthalpy are reported (M_salt_correction,
 * H_native_correction), not hidden in the brine temperature. Stage restarts
 * that set a new brine state add their inventory change to H_stage. The
 * outputs satisfy E_brine = H_stage + H_inj + H_rock + H_native_correction +
 * H_residual; a small residual after these explicit terms does not show that
 * SANSMIC's salt transport is conservative.
 *
 * Rock
 * ----
 * Transient conduction in the salt on a 1D grid of spherical shells around a
 * sphere with the same wetted wall area, attached to the receding wall, so
 * wall recession carries rock towards the wall (first-order upwind). It is
 * linear in the brine temperature, so it is solved twice per step and
 * combined with the brine equation exactly (ThermalRock::step_inventory).
 *
 * Dissolution rate and saturation
 * -------------------------------
 * The 75 degF recession-rate curve is evaluated at the same relative density
 * excess (sg - 1)/(sg_sat - 1) and multiplied by a temperature factor:
 * version 1, the legacy SANSMIC factor ((T+460)/535)^0.75 exp(0.01 (T-75));
 * version 2, the ratio of the Sandia smooth-sample regression
 * (SAND2013-2398C; measured 24-50 degC and 0-90% saturation, held at the
 * range limits outside). By default (thermal_saturation_model 0) saturation
 * stays at SANSMIC's 75 degF value. With 1 it follows the halite solubility of
 * Potter et al. (1977), scaled to 0.2632 at 75 degF; states that would
 * precipitate salt stop the run with an error, because no solid phase is
 * modelled.
 *
 * Scope and limits
 * ----------------
 * Ordinary leaching with one well only. Not modelled: heat exchange with the
 * blanket and the wellbore, the geothermal gradient along the cavern,
 * temperature differences inside the brine, thermal buoyancy, pressure effects
 * on enthalpy and solubility, and precipitation. Not validated against a
 * measured thermal cavern.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "caloric_data.hpp"
#include "libsansmic.hpp"

namespace {
const double w_sat_75F = 0.2632;  // SANSMIC saturation weight fraction

/// Halite solubility (wt%), Potter et al. 1977 via MOOSE, T in degC.
double potter_wt_pct(double T_C) {
  return 26.18 + 0.0072 * T_C + 0.000106 * T_C * T_C;
}

double F_to_C(double T_F) { return (T_F - 32.0) / 1.8; }

/// Sandia smooth-sample recession regression (mm/h * 0.01 m), T in degC and
/// saturation in percent. Only ratios are used.
double sandia_rate(double T_C, double S) {
  return 1.647e-8 * T_C * T_C + 8.425e-9 * S * S - 4.293e-8 * T_C * S +
         3.064e-6 * T_C - 1.269e-6 * S + 6.185e-5;
}

void check_range(double value, double lo, double hi, const std::string &name) {
  if (!std::isfinite(value) || value < lo || value > hi) {
    throw std::invalid_argument(name + " is outside its allowed range (" +
                                std::to_string(lo) + " to " +
                                std::to_string(hi) + ")");
  }
}
}  // namespace

/**
 * @brief Read and check the temperature model settings, build the rock grid
 * and set the saturation for the initial brine temperature.
 * @param scenario the scenario being configured
 */
void sansmic::Model::thermal_configure(Scenario &scenario) {
  if (tempModelVer == 0) return;
  if (tempModelVer < 0 || tempModelVer > 2) {
    throw std::invalid_argument("temperature_model_version must be 0, 1 or 2");
  }
  check_range(scenario.rock_temperature, 32.0, 300.0, "rock_temperature");
  check_range(scenario.rock_conductivity, 1.0e-3, 10.0, "rock_conductivity");
  check_range(scenario.rock_heat_capacity, 0.05, 1.0, "rock_heat_capacity");
  check_range(scenario.wall_heat_transfer, 0.0, 1.0e4, "wall_heat_transfer");
  if (!std::isnan(scenario.heat_of_solution))
    check_range(scenario.heat_of_solution, -100.0, 100.0, "heat_of_solution");
  T_rock = scenario.rock_temperature;
  T_brine = std::isnan(scenario.initial_brine_temperature)
                ? T_rock
                : scenario.initial_brine_temperature;
  check_range(T_brine, 32.0, 300.0, "initial_brine_temperature");
  T_injStage = T_brine;
  k_rock = scenario.rock_conductivity;
  cp_rock = scenario.rock_heat_capacity;
  h_wall = scenario.wall_heat_transfer;
  dH_soln = scenario.heat_of_solution;
  thermalSaturationModel = scenario.thermal_saturation_model;
  if (thermalSaturationModel != 0 && thermalSaturationModel != 1)
    throw std::invalid_argument("thermal_saturation_model must be 0 or 1");
  if (std::isnan(dH_soln)) {
    check_range(T_rock, 50.0, 140.0, "evaluated rock temperature");
    check_range(T_brine, 50.0, 140.0, "evaluated initial brine temperature");
  }
  if (scenario.coallescing_wells != 1)
    throw std::invalid_argument(
        "thermal extension currently supports one well");
  C_sat0 = scenario.max_brine_sg;
  salt75 = new sansmic::Salt(scenario.max_brine_sg, scenario.solid_density);

  rock = new sansmic::ThermalRock(T_rock);
  thermal_update_saturation();
}

/**
 * @brief Set saturated weight fraction and sg for the current brine
 * temperature. Unsupported inlet or cooling precipitation is rejected.
 * No concentration, salt mass, solvent mass or heat is silently clipped.
 */
void sansmic::Model::thermal_update_saturation(void) {
  if (tempModelVer == 0) return;
  double w = thermalSaturationModel ? thermal_wt_sat(T_brine) : w_sat_75F;
  double C_new = thermalSaturationModel ? thermal_sg_sat(T_brine) : C_sat0;
  salt->set_wt_max(w);
  salt->set_sg_max(C_new);
  C_hat = C_new;
  w_hat = w;
  // Never change an inlet composition to make it acceptable to the engine.
  // Native transport cannot carry a separate crystalline phase.
  if (C_inj > C_hat + 1.0e-10)
    throw std::invalid_argument(
        "thermal inlet exceeds cavern saturation; solid-phase precipitation is "
        "not implemented; inlet composition was not changed");
  if (C_cav.size() == 0 || !thermalSaturationModel) return;
  // Check every layer that holds brine in the enthalpy inventory, including
  // a partly wetted interface cell.
  vector<double> masses, fractions;
  thermal_layer_masses(masses, fractions);
  vector<double> sg(C_cav.begin() + 1, C_cav.begin() + 1 + masses.size());
  int k = thermal_first_supersaturated(sg, masses, C_hat);
  if (k >= 0)
    throw std::invalid_argument(
        "thermal cooling requires precipitation; solid-phase mass, solvent "
        "and heat coupling is not implemented; no clipped result is "
        "reported: cell=" +
        std::to_string(k + 1) + " sg=" + std::to_string(sg[k]) +
        " saturation=" + std::to_string(C_hat) +
        " T=" + std::to_string(T_brine));
}

/**
 * @brief Index of the first layer holding brine (mass > 0) whose sg exceeds
 * the saturated sg, or -1 if none.
 */
int sansmic::thermal_first_supersaturated(const vector<double> &sg,
                                          const vector<double> &masses,
                                          double sg_sat) {
  for (int k = 0; k < (int)sg.size() && k < (int)masses.size(); k++) {
    if (masses[k] > 0.0 && sg[k] > sg_sat + 1.0e-10) return k;
  }
  return -1;
}

/**
 * @brief Saturated weight fraction at temperature T (degF): Potter's
 * solubility change applied to SANSMIC's 0.2632 at 75 degF.
 */
double sansmic::Model::thermal_wt_sat(double T) {
  return w_sat_75F * potter_wt_pct(F_to_C(T)) / potter_wt_pct(F_to_C(75.0));
}

/**
 * @brief Saturated brine sg at temperature T (degF): SANSMIC's sg(w) curve
 * continued to thermal_wt_sat(T); equal to max_brine_sg at 75 degF.
 */
double sansmic::Model::thermal_sg_sat(double T) {
  array<double, 3> c = salt75->get_sg_wt_pct_convert_coeff();
  double w = thermal_wt_sat(T);
  double quad_w = c[0] + c[1] * w + c[2] * w * w;
  double quad_75 = c[0] + c[1] * w_sat_75F + c[2] * w_sat_75F * w_sat_75F;
  return C_sat0 + quad_w - quad_75;
}

/**
 * @brief Temperature factor on the 75 degF recession rate.
 * @param T brine temperature (degF)
 * @param s relative saturation (0-1)
 * @return the multiplier (1 at 75 degF)
 */
double sansmic::Model::thermal_rate_factor(double T, double s) {
  if (tempModelVer == 1) {
    return pow((T + 460.0) / 535.0, 0.75) * exp(0.01 * (T - 75.0));
  }
  double T_C = fmin(fmax(F_to_C(T), 24.0), 50.0);
  double S = 100.0 * fmin(fmax(s, 0.0), 0.9);
  return sandia_rate(T_C, S) / sandia_rate(24.0, S);
}

/**
 * @brief Wall recession rate for brine of sg C; identical to
 * Salt::recession_rate when the temperature model is off.
 * @param sg brine specific gravity
 * @return recession rate (ft/s)
 */
double sansmic::Model::wall_rate(double sg) {
  if (tempModelVer == 0) return salt->recession_rate(sg);
  // Evaluate the 75 degF curve at the same relative density excess,
  // (sg - 1) / (sg_sat - 1). This mapping is smooth; SANSMIC's piecewise
  // sg <-> wt% conversions jump by 7.5e-5 in sg at 10 wt% and would amplify
  // round-off. It is the identity when sg_sat equals max_brine_sg (75 degF).
  double C_sat = salt->get_sg_max();
  double C75 = sg;
  if (C_sat != C_sat0) C75 = 1.0 + (sg - 1.0) * (C_sat0 - 1.0) / (C_sat - 1.0);
  C75 = fmin(C75, C_sat0);
  double s = fmin(fmax(salt->wt_pct(sg) / salt->get_wt_max(), 0.0), 1.0);
  return salt75->recession_rate(C75) * thermal_rate_factor(T_brine, s);
}

/**
 * @brief Advance brine and rock temperatures over the current timestep
 * (implicit), using the salt dissolved during the step.
 */
void sansmic::Model::thermal_step(void) {
  if (tempModelVer == 0) return;
  // salt dissolved this step (same sum as the cavern mass balance)
  double V_salt = 0.0;
  for (int i = 1; i <= obiCellBelow; i++) V_salt += V_saltRemove[i];
  V_salt = fmax(V_salt, 0.0);
  double mdot_s = V_salt * C_wall * water_density_lb_cf / dt;  // lb/hr

  // wetted wall area: brine-filled cell walls plus the floor
  double A = _pi_ * sq(r_cav[1]);
  for (int i = 1; i <= obiCellBelow; i++) {
    A += 2.0 * _pi_ * r_cav[i] * dz / fmax(fabs(cosTheta[i]), 0.05);
  }
  A = fmax(A, 1.0e-6);
  // mean wall recession speed, salt plus insolubles (ft/hr)
  double v_wall = V_salt * (1.0 + f_insol0) / (A * dt);

  double M_b = m_brine * water_density_lb_cf;  // native tracked mass, lb
  double M_in = fmax(Q_in, 0.0) * C_inj * water_density_lb_cf * dt;
  double M_s = mdot_s * dt;
  M_saltDissolvedCum += M_s;
  // The discharge MASS is the native mass ledger's loss, not an imposed
  // volume flow times an assumed density. SANSMIC's CFAC still adjusts fields;
  // this caloric coupling does not independently validate its salt transport.
  double M_out = thermal_old_mass + M_in + M_s - M_b;
  double roundoff = 64.0 * std::numeric_limits<double>::epsilon() *
                    fmax(fmax(thermal_old_mass, M_b), 1.0);
  if (M_out < -roundoff)
    throw std::runtime_error(
        "temperature model: SANSMIC's brine mass ledger implies an inflow "
        "other than injection (mass-balance correction factor " +
        std::to_string(err_cfac) +
        "); reduce the timestep, especially while the cavity is small");
  M_out = fmax(M_out, 0.0);
  double w_new = thermal_mean_fraction();
  double w_in = salt->wt_pct(C_inj);
  double w_out =
      0.5 * (salt->wt_pct(C_tmp[prodCell]) + salt->wt_pct(C_cav[prodCell]));
  thermal_outlet_fraction = w_out;
  // SANSMIC corrects concentration fields with CFAC; it does not exactly close
  // a separately reconstructed species inventory. Account for that numerical
  // salt/water exchange explicitly, rather than heating brine with it or
  // inventing an unphysical outlet composition.
  double w_conserved = (thermal_old_mass * thermal_old_fraction + M_in * w_in +
                        M_s - M_out * w_out) /
                       M_b;
  if (w_conserved < -1e-10 || w_conserved > 0.28)
    throw std::runtime_error(
        "temperature model: the salt ledger gives an impossible mean salinity "
        "(w = " +
        std::to_string(w_conserved) + ", mass-balance correction factor " +
        std::to_string(err_cfac) +
        "); SANSMIC's step is not mass-consistent. Reduce the timestep, "
        "especially while the cavity is small (injection per step should be "
        "well below the cavity volume)");
  w_conserved = fmax(0.0, w_conserved);
  M_saltCorrectionCum += M_b * (w_new - w_conserved);
  double rc = C_wall * water_density_lb_cf * cp_rock;  // Btu/ft3-degF
  vector<double> curve = thermal_profile_curve();
  double T_new = rock->step_inventory(
      dt, A, v_wall, k_rock, rc, h_wall, T_brine, thermal_old_energy, curve,
      M_b, w_new, w_conserved, M_in, w_in, T_injStage, M_s, M_out, w_out,
      cp_rock, dH_soln);
  T_brine = T_new;
  Q_rockRate = rock->q_wall;
  H_rockCum += Q_rockRate * dt;
  H_injCum +=
      M_in * thermal_brine_enthalpy(T_injStage, w_in, cp_rock, dH_soln) -
      M_out * thermal_brine_enthalpy(T_brine, w_out, cp_rock, dH_soln);
  // Chemical inventory/flux demand, not a second energy sink.
  auto chem = [&](double T, double w) {
    return thermal_brine_enthalpy(T, w, cp_rock, dH_soln) -
           ((1.0 - w) + w * cp_rock) * (T - 75.0);
  };
  H_dissCum += thermal_energy_at(T_brine, curve) - curve[0] * (T_brine - 75.0) -
               thermal_old_chemical - M_in * chem(T_injStage, w_in) +
               M_out * chem(T_brine, w_out);
  H_nativeCorrectionCum +=
      M_b * (thermal_brine_enthalpy(T_brine, w_new, cp_rock, dH_soln) -
             thermal_brine_enthalpy(T_brine, w_conserved, cp_rock, dH_soln));
  E_balance = rock->energy_residual;
  H_residualCum += E_balance;
  // Store the saturation at the output temperature and reject phase change
  // before saving a completed step, rather than leaving a one-step lag.
  thermal_update_saturation();
  double w_after = thermal_mean_fraction();
  // Also expose the inherited SG/property reconciliation at the new saturation.
  M_saltCorrectionCum += M_b * (w_after - w_new);
  E_brineCurrent = thermal_energy_at(T_brine, curve);
  if (thermalSaturationModel) {
    double after = thermal_energy_at(T_brine, thermal_profile_curve());
    H_nativeCorrectionCum += after - E_brineCurrent;
    E_brineCurrent = after;
  }
}

/**
 * @brief Mass-weighted NaCl weight fraction of the brine below the blanket,
 * excluding the volume occupied by insolubles.
 */
double sansmic::Model::thermal_mean_fraction(void) {
  vector<double> masses, fractions;
  thermal_layer_masses(masses, fractions);
  double mass = 0.0, dissolved = 0.0;
  for (size_t k = 0; k < masses.size(); k++) {
    mass += masses[k];
    dissolved += masses[k] * fractions[k];
  }
  return mass > 0.0 ? dissolved / mass : 0.0;
}

/**
 * @brief Brine mass (ft3 * sg, unscaled) and NaCl weight fraction of each
 * layer up to the interface cell: the cell volume below the interface minus
 * the volume taken by insolubles, filled from the floor up. Shared by the
 * enthalpy inventory and the precipitation check.
 */
void sansmic::Model::thermal_layer_masses(vector<double> &masses,
                                          vector<double> &fractions) {
  masses.clear();
  fractions.clear();
  double deposits = V_insol;
  for (int i = 1; i <= obiCell && i < (int)r_cav.size(); ++i) {
    double height = (i == obiCell) ? fmax(0.0, h_obi - (i - 1) * dz) : dz;
    double volume = _pi_ * sq(r_cav[i]) * height;
    double covered = fmin(deposits, volume);
    deposits -= covered;
    volume -= covered;
    masses.push_back(volume * C_cav[i]);
    fractions.push_back(salt->wt_pct(C_cav[i]));
  }
}

/**
 * @brief Enthalpy coefficients of the current layered brine (see
 * thermal_energy_curve), with layer masses scaled to the tracked brine mass.
 */
vector<double> sansmic::Model::thermal_profile_curve(void) {
  vector<double> masses, fractions;
  thermal_layer_masses(masses, fractions);
  double total = 0.0;
  for (double mass : masses) total += mass;
  if (total <= 0.0)
    throw std::runtime_error("no positive native brine profile inventory");
  double scale = m_brine * water_density_lb_cf / total;
  for (double &mass : masses) mass *= scale;
  return thermal_energy_curve(masses, fractions, cp_rock, dH_soln);
}

/**
 * @brief Record brine mass, salinity and enthalpy at the start of a step.
 */
void sansmic::Model::thermal_prepare(void) {
  if (tempModelVer == 0) return;
  thermal_old_mass = m_brine * water_density_lb_cf;
  thermal_old_fraction = thermal_mean_fraction();
  thermal_old_energy = E_brineCurrent;
  double capacity = thermal_old_mass * ((1.0 - thermal_old_fraction) +
                                        thermal_old_fraction * cp_rock);
  thermal_old_chemical = thermal_old_energy - capacity * (T_brine - 75.0);
}

/**
 * @brief Rock conduction grid: n cells with geometric spacing from `first`
 * at the wall to a fixed far-field temperature at `depth` (ft). The defaults
 * (0.02 ft to 300 ft, 80 cells) reach beyond the conduction depth of a
 * multi-year leach (about 6 sqrt(alpha t) ~ 250 ft at 4 years for halite).
 * @param T_far undisturbed (initial and far-field) temperature (degF)
 */
sansmic::ThermalRock::ThermalRock(double T_far_, int n, double first,
                                  double depth) {
  if (n < 2 || first <= 0.0 || depth <= first * n) {
    throw std::invalid_argument("invalid rock grid");
  }
  T_far = T_far_;
  q_wall = 0.0;
  double lo = 1.0 + 1.0e-12, hi = 2.0;
  for (int it = 0; it < 200; it++) {
    double g = 0.5 * (lo + hi);
    double total = first * (pow(g, n) - 1.0) / (g - 1.0);
    if (total > depth) {
      hi = g;
    } else {
      lo = g;
    }
  }
  double g = 0.5 * (lo + hi);
  xf = vector<double>(n + 1, 0.0);
  xi = vector<double>(n, 0.0);
  double width = first;
  for (int k = 0; k < n; k++) {
    xf[k + 1] = xf[k] + width;
    xi[k] = 0.5 * (xf[k] + xf[k + 1]);
    width *= g;
  }
  xf[n] = depth;
  xi[n - 1] = 0.5 * (xf[n - 1] + xf[n]);
  T = vector<double>(n, T_far);
}

/**
 * @brief Volume of rock cell k for a sphere of wall area A (ft3).
 */
double sansmic::ThermalRock::cell_volume(int k, double A) {
  double R = sqrt(A / (4.0 * _pi_));
  return A / sq(R) * (pow(R + xf[k + 1], 3) - pow(R + xf[k], 3)) / 3.0;
}

/**
 * @brief Heat content of the rock grid relative to 0 degF (Btu).
 */
double sansmic::ThermalRock::content(double A, double rho_cp) {
  double e = 0.0;
  for (int k = 0; k < (int)T.size(); k++)
    e += rho_cp * cell_volume(k, A) * T[k];
  return e;
}

/**
 * @brief Advance the rock and one well-mixed brine node by an implicit step.
 *
 * Brine:  C_b dT_b/dt = G_in (T_in - T_b) + (G_w + S_salt)(T_1 - T_b) + Q_src
 * Rock:   rho_cp dT/dt = k lap(T) + v dT/dxi (spherical shells, wall frame)
 *
 * @param dt timestep (hr)
 * @param A wetted wall area (ft2); the sphere radius is sqrt(A/4pi)
 * @param v_wall wall recession speed (ft/hr), carries rock towards the wall
 * @param k rock conductivity (Btu/hr-ft-degF)
 * @param rho_cp rock volumetric heat capacity (Btu/ft3-degF)
 * @param h_wall wall film coefficient (Btu/hr-ft2-degF); 0 = perfect contact
 * @param T_b brine temperature at the start of the step (degF)
 * @param C_b brine heat capacity (Btu/degF); negative holds T_b fixed
 * @param G_in inflow capacity rate (Btu/hr-degF)
 * @param T_in inflow temperature (degF)
 * @param S_salt capacity rate of dissolved salt entering at T_1 (Btu/hr-degF)
 * @param Q_src other heat source into the brine (Btu/hr)
 * @return brine temperature at the end of the step (degF)
 */
double sansmic::ThermalRock::step(double dt, double A, double v_wall, double k,
                                  double rho_cp, double h_wall, double T_b,
                                  double C_b, double G_in, double T_in,
                                  double S_salt, double Q_src) {
  int n = (int)T.size();
  double R = sqrt(A / (4.0 * _pi_));
  vector<double> a(n + 1, 0.0), b(n + 1, 0.0), c(n + 1, 0.0), d(n + 1, 0.0);
  double G_cond = k * A / xi[0];
  double G_w =
      (h_wall > 0.0) ? 1.0 / (1.0 / (h_wall * A) + 1.0 / G_cond) : G_cond;
  // brine (unknown 0)
  if (C_b < 0.0) {
    b[0] = 1.0;
    d[0] = T_b;
  } else {
    b[0] = C_b / dt + G_in + G_w + S_salt;
    c[0] = -(G_w + S_salt);
    d[0] = C_b / dt * T_b + G_in * T_in + Q_src;
  }
  // rock cells (unknowns 1..n)
  for (int m = 0; m < n; m++) {
    int j = m + 1;
    double V_m = cell_volume(m, A);
    double dx_e = (m == n - 1) ? (xf[n] - xi[m]) : (xi[m + 1] - xi[m]);
    double G_e = k * A * sq((R + xf[m + 1]) / R) / dx_e;
    double G_west = G_w;
    if (m > 0) G_west = k * A * sq((R + xf[m]) / R) / (xi[m] - xi[m - 1]);
    // upwind face difference rho_cp v A_m (T_east - T_m), with A_m the mean
    // cell area; telescopes to rho_cp v A (T_far - T_1) for a planar wall
    double adv = rho_cp * v_wall * V_m / (xf[m + 1] - xf[m]);
    double C_r = rho_cp * V_m / dt;
    a[j] = -G_west;
    b[j] = C_r + G_west + G_e + adv;
    d[j] = C_r * T[m];
    if (m == n - 1) {
      d[j] += (G_e + adv) * T_far;
    } else {
      c[j] = -(G_e + adv);
    }
  }
  // Thomas algorithm
  for (int j = 1; j <= n; j++) {
    double f = a[j] / b[j - 1];
    b[j] -= f * c[j - 1];
    d[j] -= f * d[j - 1];
  }
  vector<double> x(n + 1, 0.0);
  x[n] = d[n] / b[n];
  for (int j = n - 1; j >= 0; j--) x[j] = (d[j] - c[j] * x[j + 1]) / b[j];
  for (int m = 0; m < n; m++) T[m] = x[m + 1];
  // conduction plus sensible heat of the dissolved salt, both from the rock
  q_wall = (G_w + S_salt) * (T[0] - x[0]);
  return x[0];
}

/** @brief Distances of the rock cell centres from the wall (ft). */
vector<double> sansmic::ThermalRock::centres(void) { return xi; }

/**
 * @brief Specific enthalpy of brine (Btu/lb) relative to water and halite at
 * 75 degF: [(1 - w) + w cp_salt] (T - 75) + w Q(T, m).
 * @param T_F temperature (degF)
 * @param w NaCl weight fraction
 * @param cp_salt halite specific heat (Btu/lb-degF)
 * @param heat_override constant Q (Btu/lb), or NaN for the Clarke & Glew table
 */
double sansmic::thermal_brine_enthalpy(double T_F, double w, double cp_salt,
                                       double heat_override) {
  if (!std::isfinite(T_F) || !std::isfinite(w) || w < 0.0 || w > 0.28)
    throw std::invalid_argument("invalid caloric state");
  double Q = heat_override;
  if (std::isnan(Q)) {
    using namespace sansmic_caloric_data;
    double tc = (T_F - 32.0) / 1.8;
    if (tc < 10.0 - 1e-9 || tc > 60.0 + 1e-9)
      throw std::invalid_argument(
          "evaluated thermal enthalpy is limited to 10-60 degC");
    tc = fmin(fmax(tc, 10.0), 60.0);
    double m = w / ((1.0 - w) * 0.058443);
    if (m > 6.4)
      throw std::invalid_argument("caloric molality exceeds 6.4 mol/kg water");
    int t = 0;
    while (t < 6 && temperatures[t + 1] < tc) ++t;
    auto at = [&](int row) {
      int j = 0;
      while (j < NM - 2 && molalities[row][j + 1] < m) ++j;
      double f = (m - molalities[row][j]) /
                 (molalities[row][j + 1] - molalities[row][j]);
      return heats[row][j] + f * (heats[row][j + 1] - heats[row][j]);
    };
    double f = (tc - temperatures[t]) / (temperatures[t + 1] - temperatures[t]);
    // Integral solution heat J/mol -> Btu/lb NaCl.
    Q = (at(t) + f * (at(t + 1) - at(t))) / 58.443 / 2.326;
  }
  return ((1.0 - w) + w * cp_salt) * (T_F - 75.0) + w * Q;
}

/**
 * @brief step_inventory for a single uniform brine layer (used in tests).
 */
double sansmic::ThermalRock::step_enthalpy(
    double dt, double A, double v_wall, double k, double rho_cp, double h_wall,
    double T_b, double M_old, double w_old, double M_new, double w_new,
    double M_in, double w_in, double T_in, double M_salt, double M_out,
    double w_out, double cp_salt, double heat_override) {
  if (M_old <= 0.0 || !std::isfinite(M_old))
    throw std::invalid_argument("invalid old brine mass");
  vector<double> curve =
      thermal_energy_curve({M_new}, {w_new}, cp_salt, heat_override);
  return step_inventory(
      dt, A, v_wall, k, rho_cp, h_wall, T_b,
      M_old * thermal_brine_enthalpy(T_b, w_old, cp_salt, heat_override), curve,
      M_new, w_new, w_new, M_in, w_in, T_in, M_salt, M_out, w_out, cp_salt,
      heat_override);
}

/**
 * @brief Advance rock and brine one step, closing the brine enthalpy balance
 * written in the file header. The rock response is linear in the brine
 * temperature, so two rock solves (at T_b and T_b + 1) give it exactly; the
 * scalar energy equation is then solved by Newton iteration and the rock
 * profile interpolated to the result.
 * @return brine temperature at the end of the step (degF)
 */
double sansmic::ThermalRock::step_inventory(
    double dt, double A, double v_wall, double k, double rho_cp, double h_wall,
    double T_b, double E_old, const vector<double> &curve, double M_new,
    double w_new, double w_conserved, double M_in, double w_in, double T_in,
    double M_salt, double M_out, double w_out, double cp_salt,
    double heat_override) {
  if (dt <= 0 || M_new <= 0 || M_in < 0 || M_salt < 0 || M_out < 0)
    throw std::invalid_argument("invalid thermal inventory step");
  auto h = [&](double t, double w) {
    return thermal_brine_enthalpy(t, w, cp_salt, heat_override);
  };
  const vector<double> old = T;
  step(dt, A, v_wall, k, rho_cp, h_wall, T_b, -1.0, 0, 0, 0, 0);
  vector<double> at0 = T;
  double q0 = q_wall;
  T = old;
  step(dt, A, v_wall, k, rho_cp, h_wall, T_b + 1.0, -1.0, 0, 0, 0, 0);
  vector<double> at1 = T;
  double dq = q_wall - q0;
  auto wall = [&](double t) { return at0[0] + (t - T_b) * (at1[0] - at0[0]); };
  double energy0 = E_old + M_in * h(T_in, w_in);
  auto residual = [&](double t) {
    double correction = M_new * (h(t, w_new) - h(t, w_conserved));
    return thermal_energy_at(t, curve) - correction + M_out * h(t, w_out) -
           energy0 - M_salt * cp_salt * (wall(t) - 75.0) -
           dt * (q0 + (t - T_b) * dq);
  };
  double t = T_b;
  for (int it = 0; it < 12; ++it) {
    double f = residual(t);
    double scale = fmax(1.0, fabs(energy0) + M_new * 50.0);
    if (fabs(f) < 1e-11 * scale) break;
    double eps = 1e-3;
    double lo = std::isnan(heat_override) ? 50.0 : 32.0;
    double hi = std::isnan(heat_override) ? 140.0 : 300.0;
    double t0 = fmax(lo, t - eps), t1 = fmin(hi, t + eps);
    double df = (residual(t1) - residual(t0)) / (t1 - t0);
    if (!std::isfinite(df) || df <= 0.0)
      throw std::runtime_error("nonpositive caloric heat capacity");
    t = fmin(fmax(t - f / df, lo), hi);
  }
  energy_residual = residual(t);
  if (fabs(energy_residual) > 1e-9 * fmax(1.0, fabs(energy0) + M_new * 50.0)) {
    T = old;
    throw std::runtime_error(
        "caloric energy root failed or left the supported temperature range");
  }
  for (int i = 0; i < (int)T.size(); ++i)
    T[i] = at0[i] + (t - T_b) * (at1[i] - at0[i]);
  q_wall = q0 + (t - T_b) * dq + M_salt * cp_salt * (T[0] - 75.0) / dt;
  return t;
}

/**
 * @brief Brine enthalpy coefficients for layers of given mass and salinity:
 * element 0 is the sensible capacity sum M [(1 - w) + w cp_salt] (Btu/degF);
 * elements 1-8 are sum M w Q at the eight table temperatures (Btu). See
 * thermal_energy_at.
 */
vector<double> sansmic::thermal_energy_curve(vector<double> masses,
                                             vector<double> fractions,
                                             double cp_salt,
                                             double heat_override) {
  using namespace sansmic_caloric_data;
  if (masses.size() != fractions.size() || masses.empty())
    throw std::invalid_argument("invalid caloric profile");
  vector<double> curve(9, 0.0);
  for (int i = 0; i < (int)masses.size(); ++i) {
    double mass = masses[i], w = fractions[i];
    if (!std::isfinite(mass) || mass < 0 || !std::isfinite(w) || w < 0 ||
        w > 0.28)
      throw std::invalid_argument("invalid caloric profile state");
    curve[0] += mass * ((1.0 - w) + w * cp_salt);
    if (!std::isnan(heat_override)) {
      for (int r = 0; r < 8; ++r) curve[r + 1] += mass * w * heat_override;
    } else if (w > 0) {
      double m = w / ((1.0 - w) * 0.058443);
      if (m > 6.4)
        throw std::invalid_argument(
            "caloric molality exceeds 6.4 mol/kg water");
      int j = (int)(std::lower_bound(molalities[0], molalities[0] + NM - 1, m) -
                    molalities[0]) -
              1;
      j = std::max(0, std::min(NM - 2, j));
      for (int r = 0; r < 8; ++r) {
        double f =
            (m - molalities[r][j]) / (molalities[r][j + 1] - molalities[r][j]);
        double Q = heats[r][j] + f * (heats[r][j + 1] - heats[r][j]);
        curve[r + 1] += mass * w * Q / 58.443 / 2.326;
      }
    }
  }
  return curve;
}

/**
 * @brief Brine enthalpy (Btu) at temperature T_F from thermal_energy_curve
 * coefficients, interpolating Q linearly between table temperatures.
 */
double sansmic::thermal_energy_at(double T_F, const vector<double> &curve) {
  using namespace sansmic_caloric_data;
  if (curve.size() != 9)
    throw std::invalid_argument("invalid caloric energy curve");
  double tc = (T_F - 32.0) / 1.8;
  int j = 0;
  while (j < 6 && temperatures[j + 1] < tc) ++j;
  double f = (tc - temperatures[j]) / (temperatures[j + 1] - temperatures[j]);
  return curve[0] * (T_F - 75.0) + curve[j + 1] +
         f * (curve[j + 2] - curve[j + 1]);
}
