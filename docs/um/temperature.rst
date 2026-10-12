Temperature model
=================

SANSMIC keeps the brine at a fixed 75 °F. The optional temperature model
(:confval:`temperature-model-version` 1 or 2) adds a brine and rock energy
balance and a temperature-dependent dissolution rate. With the default, 0,
none of it runs and results are unchanged.

This page describes what the model computes, how its energy bookkeeping works,
how it was checked, and what it does not cover.


Brine enthalpy
--------------

All brine below the blanket has one temperature :math:`T`. Each native layer
(cell) of mass :math:`M` and NaCl weight fraction :math:`w` contributes the
enthalpy, relative to water and halite at 75 °F,

.. math::

   H = M \left[ (1 - w) + w\,c_{p,s} \right] (T - 75) + M\,w\,Q(T, m),

where :math:`c_{p,s}` is the halite specific heat (water: 1 Btu/lb-°F) and
:math:`Q` the integral heat of solution per pound of NaCl at molality
:math:`m`, interpolated in the evaluated table of Clarke and Glew (1985),
Table 25A (10–60 °C, 0–6.4 mol/kg, atmospheric pressure). Summing per layer,
rather than using the mean salinity, keeps the non-linear dependence of
:math:`Q` on concentration. Layer masses are scaled to SANSMIC's tracked brine
mass.

Over a timestep the brine temperature solves

.. math::

   H_{new}(T) + M_{out}\,h(T, w_{out}) - H_{old} - M_{in}\,h(T_{in}, w_{in})
   - M_{salt}\,c_{p,s} (T_{wall} - 75) - Q_{rock}\,\Delta t = 0,

where :math:`h` is the specific enthalpy above. :math:`M_{in}` and
:math:`T_{in}` are the injected mass and
:confval:`brine-injection-temperature`; :math:`M_{out}` is the mass leaving
according to SANSMIC's own mass ledger, at the production-cell salinity
:math:`w_{out}`; :math:`M_{salt}` is the salt dissolved in the step, entering
at the wall temperature. The chemical heat of dissolution is not a separate
sink: it appears through :math:`Q` in :math:`H_{new}`. A constant
:confval:`heat-of-solution` replaces :math:`Q` for diagnostic runs only.


Rock
----

Heat conduction in the salt is solved on a 1D grid of spherical shells
(80 cells, 0.02 ft at the wall to a fixed far-field temperature 300 ft away)
around a sphere with the same wetted wall area as the cavern. The grid moves
with the receding wall, so recession carries rock towards the wall
(first-order upwind). The rock response is linear in the brine temperature;
it is solved twice per step and combined exactly with the scalar brine
equation, which is then solved by Newton iteration.


Dissolution rate and saturation
-------------------------------

The 75 °F recession-rate curve is evaluated at the same relative density
excess :math:`(sg - 1)/(sg_{sat} - 1)` and multiplied by a temperature factor:

* version 1: the legacy SANSMIC factor
  :math:`((T + 460)/535)^{0.75} \exp(0.01 (T - 75))`;
* version 2: the ratio of the Sandia smooth-sample regression
  (SAND2013-2398C), measured at 24–50 °C and 0–90% saturation and held at the
  range limits outside it.

By default (:confval:`thermal-saturation-model` 0) saturation stays at
SANSMIC's 75 °F value. With 1 it follows the halite solubility of Potter et
al. (1977), scaled to 0.2632 at 75 °F. States that would precipitate salt,
including an inlet above saturation, stop the run with an error: no solid
phase is modelled, and concentrations are never clipped.


Energy bookkeeping
------------------

SANSMIC's mass-balance correction factor adjusts concentrations without
exactly conserving a separately reconstructed salt inventory. The temperature
model does not hide that difference in the brine temperature. It reports

* ``M_salt_correction``: the cumulative numerical salt difference (lb);
* ``H_native_correction``: its enthalpy (MMBtu);
* ``H_stage``: inventory changes when a stage sets a new brine state;
* ``H_inj``: inlet minus outlet enthalpy (MMBtu; not heater energy);
* ``H_rock``: heat from the rock, including the sensible heat of dissolved salt;
* ``H_residual``: the accumulated root-solver residual;

and the saved brine enthalpy satisfies

.. math::

   E_{brine} = H_{stage} + H_{inj} + H_{rock} + H_{native\,correction}
   + H_{residual}.

``H_diss`` reports the chemical part of the inventory change; it is a
diagnostic, not a second sink. A small residual after these explicit terms
shows that the energy equation is solved consistently. It does not show that
SANSMIC's salt transport is conservative.


Checks
------

The tests in ``tests/test_thermal.py`` cover:

* temperature model off: outputs constant at 75 °F and unchanged;
* everything at 75 °F with no heat of solution: the default model is
  reproduced;
* the rock grid against analytic solutions: a sphere with fixed wall
  temperature, a steadily receding wall, a wall film coefficient and closed
  energy exchange;
* open-tank solutions derived independently of the code: a flushing tank and
  a steady dissolving tank;
* the enthalpy function at printed table values, layered versus bulk
  inventories, the staged energy ledger, rejected precipitation and inlet
  states, input validation and TOML round trips.


Scope and limits
----------------

* Ordinary leaching with one well only.
* One brine temperature: no temperature differences inside the brine and no
  thermal buoyancy.
* No heat exchange with the blanket or the wellbore; no geothermal gradient
  along the cavern height.
* Atmospheric-pressure calorimetry; no pressure effects on enthalpy or
  solubility.
* No precipitation or solid phase.
* The temperature response of the dissolution rate comes from laboratory
  smooth-sample measurements applied to SANSMIC's empirical 75 °F curve.
* Not validated against a measured thermal cavern.
