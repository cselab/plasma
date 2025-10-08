import jax
from jax import numpy as jnp
from torax._src import math_utils
from torax._src.geometry import geometry
def exponential_profile(
    geo: geometry.Geometry,
    *,
    decay_start: float,
    width: float,
    total: float,
) -> jax.Array:
  r = geo.rho_norm
  S = jnp.exp(-(decay_start - r) / width)
  C = total / math_utils.volume_integration(S, geo)
  return C * S
def gaussian_profile(
    geo: geometry.Geometry,
    *,
    center: float,
    width: float,
    total: float,
) -> jax.Array:
  r = geo.rho_norm
  S = jnp.exp(-((r - center) ** 2) / (2 * width**2))
  C = total / math_utils.volume_integration(S, geo)
  return C * S
