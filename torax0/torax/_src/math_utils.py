# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Math operations.

Math operations that are needed for Torax, but are not specific to plasma
physics or differential equation solvers.
"""

import enum
import functools

import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import geometry


@enum.unique
class IntegralPreservationQuantity(enum.Enum):
  """The quantity to preserve the integral of when converting to face values."""

  # Indicate that the volume integral should be preserved.
  VOLUME = 'volume'
  # Indicate that the surface integral should be preserved.
  SURFACE = 'surface'
  # Indicate that the value integral should be preserved.
  VALUE = 'value'


@array_typing.jaxtyped
def tridiag(
    diag: jt.Shaped[array_typing.Array, 'size'],
    above: jt.Shaped[array_typing.Array, 'size-1'],
    below: jt.Shaped[array_typing.Array, 'size-1'],
) -> jt.Shaped[array_typing.Array, 'size size']:
  """Builds a tridiagonal matrix.

  Args:
    diag: The main diagonal.
    above: The +1 diagonal.
    below: The -1 diagonal.

  Returns:
    The tridiagonal matrix.
  """

  return jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)


@jax_utils.jit
@array_typing.jaxtyped
def cell_integration(
    x: array_typing.FloatVectorCell, geo: geometry.Geometry
) -> array_typing.FloatScalar:
  r"""Integrate a value `x` over the rhon grid.

  Cell variables in TORAX are defined as the average of the face values. This
  method integrates that face value over the rhon grid implicitly using the
  trapezium rule to sum the averaged face values by the face grid spacing.

  Args:
    x: The cell averaged value to integrate.
    geo: The geometry instance.

  Returns:
    Integration over the rhon grid: :math:`\int_0^1 x_{face} d\hat{rho}`
  """
  if x.shape != geo.rho_norm.shape:
    raise ValueError(
        'For cell_integration, input "x" must have same shape as the cell grid'
        f'Got x.shape={x.shape}, expected {geo.rho_norm.shape}.'
    )
  return jnp.sum(x * geo.drho_norm)


@array_typing.jaxtyped
def area_integration(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates integral of value using an area metric."""
  return cell_integration(value * geo.spr, geo)


@array_typing.jaxtyped
def volume_integration(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates integral of value using a volume metric."""
  return cell_integration(value * geo.vpr, geo)


@array_typing.jaxtyped
def line_average(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates line-averaged value from input profile."""
  return cell_integration(value, geo)


@array_typing.jaxtyped
def volume_average(
    value: array_typing.FloatVector,
    geo: geometry.Geometry,
) -> array_typing.FloatScalar:
  """Calculates volume-averaged value from input profile."""
  return cell_integration(value * geo.vpr, geo) / geo.volume_face[-1]
