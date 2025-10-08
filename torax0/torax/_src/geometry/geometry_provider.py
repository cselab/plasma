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

"""GeometryProvider interface and implementations.

NOTE: Time dependent providers currently live in `geometry.py` and match the
protocol defined here.
"""
from collections.abc import Mapping
import dataclasses
from typing import Protocol, Type

import chex
import jax
import numpy as np
from torax._src import interpolated_param
from torax._src import jax_utils
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name


class GeometryProvider(Protocol):
  """Returns the geometry to use during one time step of the simulation.

  A GeometryProvider is any callable (class or function) which takes the
  time of a time step and returns the Geometry for that
  time step. See `SimulationStepFn` for how this callable is used.

  This class is a typing.Protocol, meaning it defines an interface, but any
  function asking for a GeometryProvider as an argument can accept any function
  or class that implements this API without specifically extending this class.

  For instance, the following is an equivalent implementation of the
  ConstantGeometryProvider without actually creating a class, and equally valid.

  .. code-block:: python

    geo = circular_geometry.build_circular_geometry(...)
    constant_geo_provider = lambda t: geo

    def func_expecting_geo_provider(gp: GeometryProvider):
      ... # do something with the provider.

    func_expecting_geo_provider(constant_geo_provider)  # this works.

  NOTE: In order to maintain consistency between the RuntimeParams
  and the geometry,
  `build_runtime_params.get_consistent_runtime_params_and_geometry`
  should be used to get a Geometry and a corresponding
  RuntimeParams.
  """

  def __call__(
      self,
      t: chex.Numeric,
  ) -> geometry.Geometry:
    """Returns the geometry to use during one time step of the simulation.

    The geometry may change from time step to time step, so the sim needs a
    callable to provide which geometry to use for a given time step (this is
    that callable).

    Args:
      t: The time at which the geometry is being requested.

    Returns:
      Geometry of the torus to use for the time step.
    """

  @property
  def torax_mesh(self) -> torax_pydantic.Grid1D:
    """Returns the mesh used by Torax, this is consistent across time."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConstantGeometryProvider(GeometryProvider):
  """Returns the same Geometry for all calls."""
  geo: geometry.Geometry

  def __call__(self, t: chex.Numeric) -> geometry.Geometry:
    # The API includes time as an arg even though it is unused in order
    # to match the API of a GeometryProvider.
    del t  # Ignored.
    return self.geo

  @property
  def torax_mesh(self) -> torax_pydantic.Grid1D:
    return self.geo.torax_mesh

