import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.geometry import geometry as geometry_lib
def calculate_f_trap(
    geo: geometry_lib.Geometry,
) -> array_typing.FloatVectorFace:
  epsilon_effective = (
      0.67
      * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face)
      * geo.epsilon_face
  )
  aa = (1.0 - geo.epsilon_face) / (1.0 + geo.epsilon_face)
  return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
      1.0 + 2.0 * jnp.sqrt(epsilon_effective)
  )
def calculate_L31(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  denom = (
      1.0
      + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - f_trap) * nu_e_star / Z_eff
  )
  ft31 = f_trap / denom
  term_0 = (1 + 1.4 / (Z_eff + 1)) * ft31
  term_1 = -1.9 / (Z_eff + 1) * ft31**2
  term_2 = 0.3 / (Z_eff + 1) * ft31**3
  term_3 = 0.2 / (Z_eff + 1) * ft31**4
  return term_0 + term_1 + term_2 + term_3
def calculate_L32(
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  ft32ee = f_trap / (
      1
      + 0.26 * (1 - f_trap) * jnp.sqrt(nu_e_star)
      + 0.18 * (1 - 0.37 * f_trap) * nu_e_star / jnp.sqrt(Z_eff)
  )
  ft32ei = f_trap / (
      1
      + (1 + 0.6 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.85 * (1 - 0.37 * f_trap) * nu_e_star * (1 + Z_eff)
  )
  F32ee = (
      (0.05 + 0.62 * Z_eff)
      / (Z_eff * (1 + 0.44 * Z_eff))
      * (ft32ee - ft32ee**4)
      + 1
      / (1 + 0.22 * Z_eff)
      * (ft32ee**2 - ft32ee**4 - 1.2 * (ft32ee**3 - ft32ee**4))
      + 1.2 / (1 + 0.5 * Z_eff) * ft32ee**4
  )
  F32ei = (
      -(0.56 + 1.93 * Z_eff)
      / (Z_eff * (1 + 0.44 * Z_eff))
      * (ft32ei - ft32ei**4)
      + 4.95
      / (1 + 2.48 * Z_eff)
      * (ft32ei**2 - ft32ei**4 - 0.55 * (ft32ei**3 - ft32ei**4))
      - 1.2 / (1 + 0.5 * Z_eff) * ft32ei**4
  )
  return F32ee + F32ei
def calculate_nu_e_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_e: array_typing.FloatVectorFace,
    T_e: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    log_lambda_ei: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  return (
      6.921e-18
      * q
      * geo.R_major
      * n_e
      * Z_eff
      * log_lambda_ei
      / (
          ((T_e * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )
def calculate_nu_i_star(
    q: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    n_i: array_typing.FloatVectorFace,
    T_i: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    log_lambda_ii: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorFace:
  return (
      4.9e-18
      * q
      * geo.R_major
      * n_i
      * Z_eff**4
      * log_lambda_ii
      / (
          ((T_i * 1e3) ** 2)
          * (geo.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )
