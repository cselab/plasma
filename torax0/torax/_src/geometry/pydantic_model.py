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

"""Pydantic model for geometry."""

from collections.abc import Callable
from collections.abc import Mapping
import functools
import inspect
import logging
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from imas import ids_toplevel
import pydantic
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name
T = TypeVar('T')

LY_OBJECT_TYPE: TypeAlias = (
    str | Mapping[str, torax_pydantic.NumpyArray | float]
)

TIME_INVARIANT = torax_pydantic.TIME_INVARIANT
class CheaseConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the CHEASE geometry.

  Attributes:
    geometry_type: Always set to 'chease'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
  """

  geometry_type: Annotated[Literal['chease'], TIME_INVARIANT] = 'chease'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols'
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:

    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_chease,
            self.__dict__,
        )
    )


class GeometryConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a single geometry config."""

  config: (
      CheaseConfig
  ) = pydantic.Field(discriminator='geometry_type')


class Geometry(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a geometry.

  This object can be constructed via `Geometry.from_dict(config)`, where
  `config` is a dict described in
  https://torax.readthedocs.io/en/latest/configuration.html#geometry.

  Attributes:
    geometry_type: A `geometry.GeometryType` enum.
    geometry_configs: Either a single `GeometryConfig` or a dict of
      `GeometryConfig` objects, where the keys are times in seconds.
  """

  geometry_type: geometry.GeometryType
  geometry_configs: GeometryConfig | dict[torax_pydantic.Second, GeometryConfig]

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:

    if 'geometry_type' not in data:
      raise ValueError('geometry_type must be set in the input config.')

    geometry_type = data['geometry_type']
    # The geometry type can be an int if loading from JSON.
    if isinstance(geometry_type, geometry.GeometryType | int):
      return data
    # Parse the user config dict.
    elif isinstance(geometry_type, str):
      return _conform_user_data(data)
    else:
      raise ValueError(f'Invalid value for geometry: {geometry_type}')

  @functools.cached_property
  def build_provider(self) -> geometry_provider.GeometryProvider:
    # TODO(b/398191165): Remove this branch once the FBT bundle logic is
    # redesigned.
    if isinstance(self.geometry_configs, dict):
      geometries = {
          time: config.config.build_geometry()
          for time, config in self.geometry_configs.items()
      }
      provider = (
          geometry_provider.TimeDependentGeometryProvider.create_provider
          if self.geometry_type == geometry.GeometryType.CIRCULAR
          else standard_geometry.StandardGeometryProvider.create_provider
      )
    else:
      geometries = self.geometry_configs.config.build_geometry()
      provider = geometry_provider.ConstantGeometryProvider

    return provider(geometries)  # pytype: disable=attribute-error


def _conform_user_data(data: dict[str, Any]) -> dict[str, Any]:
  """Conform the user geometry dict to the pydantic model."""

  if 'LY_bundle_object' in data and 'geometry_configs' in data:
    raise ValueError(
        'Cannot use both `LY_bundle_object` and `geometry_configs` together.'
    )

  data_copy = data.copy()
  # Useful to avoid failing if users mistakenly give the wrong case.
  data_copy['geometry_type'] = data['geometry_type'].lower()
  geometry_type = getattr(geometry.GeometryType, data['geometry_type'].upper())
  constructor_args = {'geometry_type': geometry_type}
  configs_time_dependent = data_copy.pop('geometry_configs', None)

  if configs_time_dependent:
    # geometry config has sequence of standalone geometry files.
    if not isinstance(data['geometry_configs'], dict):
      raise ValueError('geometry_configs must be a dict.')
    constructor_args['geometry_configs'] = {}
    for time, c_time_dependent in configs_time_dependent.items():
      gc = GeometryConfig.from_dict({'config': c_time_dependent | data_copy})
      constructor_args['geometry_configs'][time] = gc
      if x := set(gc.config.time_invariant_fields()).intersection(
          c_time_dependent.keys()
      ):
        raise ValueError(
            'The following parameters cannot be set per geometry_config:'
            f' {", ".join(x)}'
        )
  else:
    constructor_args['geometry_configs'] = {'config': data_copy}

  return constructor_args


def _apply_relevant_kwargs(f: Callable[..., T], kwargs: Mapping[str, Any]) -> T:
  """Apply only the kwargs actually used by the function."""
  relevant_kwargs = [i.name for i in inspect.signature(f).parameters.values()]
  kwargs = {k: kwargs[k] for k in relevant_kwargs}
  return f(**kwargs)
