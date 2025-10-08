import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.geometry import geometry
def coll_exchange(
    core_profiles: state.CoreProfiles,
    Qei_multiplier: float,
) -> jax.Array:
  log_lambda_ei = calculate_log_lambda_ei(
      core_profiles.T_e.value, core_profiles.n_e.value
  )
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.T_e.value,
      core_profiles.n_e.value,
      log_lambda_ei,
  )
  weighted_Z_eff = _calculate_weighted_Z_eff(core_profiles)
  log_Qei_coef = (
      jnp.log(Qei_multiplier * 1.5 * core_profiles.n_e.value)
      + jnp.log(constants.CONSTANTS.keV_to_J / constants.CONSTANTS.m_amu)
      + jnp.log(2 * constants.CONSTANTS.m_e)
      + jnp.log(weighted_Z_eff)
      - log_tau_e_Z1
  )
  Qei_coef = jnp.exp(log_Qei_coef)
  return Qei_coef
def calc_nu_star(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    collisionality_multiplier: float,
) -> jax.Array:
  log_lambda_ei_face = calculate_log_lambda_ei(
      core_profiles.T_e.face_value(),
      core_profiles.n_e.face_value(),
  )
  log_tau_e_Z1 = _calculate_log_tau_e_Z1(
      core_profiles.T_e.face_value(),
      core_profiles.n_e.face_value(),
      log_lambda_ei_face,
  )
  nu_e = (
      1
      / jnp.exp(log_tau_e_Z1)
      * core_profiles.Z_eff_face
      * collisionality_multiplier
  )
  epsilon = geo.rho_face / geo.R_major
  epsilon = jnp.clip(epsilon, constants.CONSTANTS.eps)
  tau_bounce = (
      core_profiles.q_face
      * geo.R_major
      / (
          epsilon**1.5
          * jnp.sqrt(
              core_profiles.T_e.face_value()
              * constants.CONSTANTS.keV_to_J
              / constants.CONSTANTS.m_e
          )
      )
  )
  tau_bounce = tau_bounce.at[0].set(tau_bounce[1])
  nustar = nu_e * tau_bounce
  return nustar
def fast_ion_fractional_heating_formula(
    birth_energy: float | array_typing.FloatVector,
    T_e: array_typing.FloatVector,
    fast_ion_mass: float,
) -> array_typing.FloatVector:
  critical_energy = 10 * fast_ion_mass * T_e  
  energy_ratio = birth_energy / critical_energy
  x_squared = energy_ratio
  x = jnp.sqrt(x_squared)
  frac_i = (
      2
      * (
          (1 / 6) * jnp.log((1.0 - x + x_squared) / (1.0 + 2.0 * x + x_squared))
          + (jnp.arctan((2.0 * x - 1.0) / jnp.sqrt(3)) + jnp.pi / 6)
          / jnp.sqrt(3)
      )
      / x_squared
  )
  return frac_i
def calculate_log_lambda_ei(
    T_e: jax.Array,
    n_e: jax.Array,
) -> jax.Array:
  T_e_ev = T_e * 1e3
  return 31.3 - 0.5 * jnp.log(n_e) + jnp.log(T_e_ev)
def calculate_log_lambda_ii(
    T_i: jax.Array,
    n_i: jax.Array,
    Z_i: jax.Array,
) -> jax.Array:
  T_i_ev = T_i * 1e3
  return 30.0 - 0.5 * jnp.log(n_i) + 1.5 * jnp.log(T_i_ev) - 3.0 * jnp.log(Z_i)
def _calculate_weighted_Z_eff(
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  return (
      core_profiles.n_i.value * core_profiles.Z_i**2 / core_profiles.A_i
      + core_profiles.n_impurity.value
      * core_profiles.Z_impurity**2
      / core_profiles.A_impurity
  ) / core_profiles.n_e.value
def _calculate_log_tau_e_Z1(
    T_e: jax.Array,
    n_e: jax.Array,
    log_lambda_ei: jax.Array,
) -> jax.Array:
  return (
      jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei))
      - 4 * jnp.log(constants.CONSTANTS.q_e)
      + 0.5 * jnp.log(constants.CONSTANTS.m_e / 2.0)
      + 2 * jnp.log(constants.CONSTANTS.epsilon_0)
      + 1.5 * jnp.log(T_e * constants.CONSTANTS.keV_to_J)
  )
