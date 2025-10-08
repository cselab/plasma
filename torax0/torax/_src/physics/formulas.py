from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry


def calculate_main_ion_dilution_factor(
    Z_i: array_typing.FloatScalar,
    Z_impurity: array_typing.FloatVector,
    Z_eff: array_typing.FloatVector,
) -> array_typing.FloatVector:
    return (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))


def calculate_pressure(
    core_profiles: state.CoreProfiles,
) -> tuple[cell_variable.CellVariable, ...]:
    pressure_thermal_el = cell_variable.CellVariable(
        value=core_profiles.n_e.value * core_profiles.T_e.value *
        constants.CONSTANTS.keV_to_J,
        dr=core_profiles.n_e.dr,
        right_face_constraint=core_profiles.n_e.right_face_constraint *
        core_profiles.T_e.right_face_constraint * constants.CONSTANTS.keV_to_J,
        right_face_grad_constraint=None,
    )
    pressure_thermal_ion = cell_variable.CellVariable(
        value=core_profiles.T_i.value * constants.CONSTANTS.keV_to_J *
        (core_profiles.n_i.value + core_profiles.n_impurity.value),
        dr=core_profiles.n_i.dr,
        right_face_constraint=core_profiles.T_i.right_face_constraint *
        constants.CONSTANTS.keV_to_J *
        (core_profiles.n_i.right_face_constraint +
         core_profiles.n_impurity.right_face_constraint),
        right_face_grad_constraint=None,
    )
    pressure_thermal_tot = cell_variable.CellVariable(
        value=pressure_thermal_el.value + pressure_thermal_ion.value,
        dr=pressure_thermal_el.dr,
        right_face_constraint=pressure_thermal_el.right_face_constraint +
        pressure_thermal_ion.right_face_constraint,
        right_face_grad_constraint=None,
    )
    return (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    )


def calc_pprime(
    core_profiles: state.CoreProfiles, ) -> array_typing.FloatVector:
    _, _, p_total = calculate_pressure(core_profiles)
    psi = core_profiles.psi.face_value()
    n_e = core_profiles.n_e.face_value()
    n_i = core_profiles.n_i.face_value()
    n_impurity = core_profiles.n_impurity.face_value()
    T_i = core_profiles.T_i.face_value()
    T_e = core_profiles.T_e.face_value()
    dne_drhon = core_profiles.n_e.face_grad()
    dni_drhon = core_profiles.n_i.face_grad()
    dnimp_drhon = core_profiles.n_impurity.face_grad()
    dti_drhon = core_profiles.T_i.face_grad()
    dte_drhon = core_profiles.T_e.face_grad()
    dpsi_drhon = core_profiles.psi.face_grad()
    dptot_drhon = constants.CONSTANTS.keV_to_J * (
        n_e * dte_drhon + n_i * dti_drhon + n_impurity * dti_drhon +
        dne_drhon * T_e + dni_drhon * T_i + dnimp_drhon * T_i)
    p_total_face = p_total.face_value()
    pprime_face_axis = jnp.expand_dims(
        (2 * p_total_face[0] - 5 * p_total_face[1] + 4 * p_total_face[2] -
         p_total_face[3]) / (2 * psi[0] - 5 * psi[1] + 4 * psi[2] - psi[3]),
        axis=0,
    )
    pprime_face = jnp.concatenate(
        [pprime_face_axis, dptot_drhon[1:] / dpsi_drhon[1:]])
    return pprime_face


def calc_FFprime(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatVector:
    mu0 = constants.CONSTANTS.mu_0
    pprime = calc_pprime(core_profiles)
    g3 = geo.g3_face
    jtor_over_R = core_profiles.j_total_face / geo.R_major
    FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
    return FFprime_face


def calculate_stored_thermal_energy(
    p_el: cell_variable.CellVariable,
    p_ion: cell_variable.CellVariable,
    p_tot: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> tuple[array_typing.FloatScalar, ...]:
    wth_el = math_utils.volume_integration(1.5 * p_el.value, geo)
    wth_ion = math_utils.volume_integration(1.5 * p_ion.value, geo)
    wth_tot = math_utils.volume_integration(1.5 * p_tot.value, geo)
    return wth_el, wth_ion, wth_tot


def calculate_greenwald_fraction(
    n_e_avg: array_typing.FloatScalar,
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
    gw_limit = (core_profiles.Ip_profile_face[-1] * 1e-6 /
                (jnp.pi * geo.a_minor**2))
    fgw = n_e_avg / (gw_limit * 1e20)
    return fgw


def calculate_betas(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
    _, _, p_total = calculate_pressure(core_profiles)
    p_total_volume_avg = math_utils.volume_average(p_total.value, geo)
    magnetic_pressure_on_axis = geo.B_0**2 / (2 * constants.CONSTANTS.mu_0)
    beta_tor = p_total_volume_avg / (magnetic_pressure_on_axis +
                                     constants.CONSTANTS.eps)
    beta_pol = (
        4.0 * geo.volume[-1] * p_total_volume_avg /
        (constants.CONSTANTS.mu_0 * core_profiles.Ip_profile_face[-1]**2 *
         geo.R_major + constants.CONSTANTS.eps))
    beta_N = (1e8 * beta_tor *
              (geo.a_minor * geo.B_0 /
               (core_profiles.Ip_profile_face[-1] + constants.CONSTANTS.eps)))
    return beta_tor, beta_pol, beta_N
