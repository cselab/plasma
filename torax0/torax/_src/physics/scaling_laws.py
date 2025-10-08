import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry
_trapz = jax.scipy.integrate.trapezoid
def calculate_plh_scaling_factor(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  line_avg_n_e = math_utils.line_average(core_profiles.n_e.value, geo)
  P_LH_hi_dens_D = (
      2.15
      * (line_avg_n_e / 1e20) ** 0.782
      * geo.B_0**0.772
      * geo.a_minor**0.975
      * geo.R_major**0.999
      * 1e6
  )
  A_deuterium = constants.ION_PROPERTIES_DICT['D'].A
  P_LH_hi_dens = P_LH_hi_dens_D * A_deuterium / core_profiles.A_i
  Ip_total = core_profiles.Ip_profile_face[..., -1]
  n_e_min_P_LH = (
      0.7
      * (Ip_total / 1e6) ** 0.34
      * geo.a_minor**-0.95
      * geo.B_0**0.62
      * (geo.R_major / geo.a_minor) ** 0.4
      * 1e19
  )
  P_LH_min_D = (
      0.36
      * (Ip_total / 1e6) ** 0.27
      * geo.B_0**1.25
      * geo.R_major**1.23
      * (geo.R_major / geo.a_minor) ** 0.08
      * 1e6
  )
  P_LH_min = P_LH_min_D * A_deuterium / core_profiles.A_i
  P_LH = jnp.maximum(P_LH_min, P_LH_hi_dens)
  return P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH
def calculate_scaling_law_confinement_time(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    Ploss: jax.Array,
    scaling_law: str,
) -> jax.Array:
  scaling_params = {
      'H89P': {
          'prefactor': 0.038128,
          'Ip_exponent': 0.85,
          'B_exponent': 0.2,
          'line_avg_n_e_exponent': 0.1,
          'Ploss_exponent': -0.5,
          'R_exponent': 1.5,
          'inverse_aspect_ratio_exponent': 0.3,
          'elongation_exponent': 0.5,
          'effective_mass_exponent': 0.50,
          'triangularity_exponent': 0.0,
      },
      'H98': {
          'prefactor': 0.0562,
          'Ip_exponent': 0.93,
          'B_exponent': 0.15,
          'line_avg_n_e_exponent': 0.41,
          'Ploss_exponent': -0.69,
          'R_exponent': 1.97,
          'inverse_aspect_ratio_exponent': 0.58,
          'elongation_exponent': 0.78,
          'effective_mass_exponent': 0.19,
          'triangularity_exponent': 0.0,
      },
      'H97L': {
          'prefactor': 0.023,
          'Ip_exponent': 0.96,
          'B_exponent': 0.03,
          'line_avg_n_e_exponent': 0.4,
          'Ploss_exponent': -0.73,
          'R_exponent': 1.83,
          'inverse_aspect_ratio_exponent': -0.06,
          'elongation_exponent': 0.64,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.0,
      },
      'H20': {
          'prefactor': 0.053,
          'Ip_exponent': 0.98,
          'B_exponent': 0.22,
          'line_avg_n_e_exponent': 0.24,
          'Ploss_exponent': -0.669,
          'R_exponent': 1.71,
          'inverse_aspect_ratio_exponent': 0.35,
          'elongation_exponent': 0.80,
          'effective_mass_exponent': 0.20,
          'triangularity_exponent': 0.36,  
      },
  }
  if scaling_law not in scaling_params:
    raise ValueError(f'Unknown scaling law: {scaling_law}')
  params = scaling_params[scaling_law]
  scaled_Ip = core_profiles.Ip_profile_face[-1] / 1e6  
  scaled_Ploss = Ploss / 1e6  
  B = geo.B_0
  line_avg_n_e = (  
      math_utils.line_average(core_profiles.n_e.value, geo) / 1e19
  )
  R = geo.R_major
  inverse_aspect_ratio = geo.a_minor / geo.R_major
  elongation = geo.area_face[-1] / (jnp.pi * geo.a_minor**2)
  effective_mass = core_profiles.A_i
  triangularity = geo.delta_face[-1]
  tau_scaling = (
      params['prefactor']
      * scaled_Ip ** params['Ip_exponent']
      * B ** params['B_exponent']
      * line_avg_n_e ** params['line_avg_n_e_exponent']
      * scaled_Ploss ** params['Ploss_exponent']
      * R ** params['R_exponent']
      * inverse_aspect_ratio ** params['inverse_aspect_ratio_exponent']
      * elongation ** params['elongation_exponent']
      * effective_mass ** params['effective_mass_exponent']
      * (1 + triangularity) ** params['triangularity_exponent']
  )
  return tau_scaling
