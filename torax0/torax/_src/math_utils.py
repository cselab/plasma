import enum
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
@enum.unique
class IntegralPreservationQuantity(enum.Enum):
  VOLUME = 'volume'
  SURFACE = 'surface'
  VALUE = 'value'
@array_typing.jaxtyped
def tridiag(
    diag: jt.Shaped[array_typing.Array, 'size'],
    above: jt.Shaped[array_typing.Array, 'size-1'],
    below: jt.Shaped[array_typing.Array, 'size-1'],
) -> jt.Shaped[array_typing.Array, 'size size']:
  return jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)
@jax_utils.jit
@array_typing.jaxtyped
def cell_integration(
    x, geo
):
  if x.shape != geo.rho_norm.shape:
    raise ValueError(
        'For cell_integration, input "x" must have same shape as the cell grid'
        f'Got x.shape={x.shape}, expected {geo.rho_norm.shape}.'
    )
  return jnp.sum(x * geo.drho_norm)
@array_typing.jaxtyped
def area_integration(
    value,
    geo,
):
  return cell_integration(value * geo.spr, geo)
@array_typing.jaxtyped
def volume_integration(
    value,
    geo,
):
  return cell_integration(value * geo.vpr, geo)
@array_typing.jaxtyped
def line_average(
    value,
    geo,
):
  return cell_integration(value, geo)
@array_typing.jaxtyped
def volume_average(
    value,
    geo,
):
  return cell_integration(value * geo.vpr, geo) / geo.volume_face[-1]
