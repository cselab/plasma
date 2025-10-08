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

"""Classes and functions for defining interpolated parameters."""

import functools
from typing import Any, TypeAlias
import chex
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import interpolated_param
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
import typing_extensions


class TimeVaryingScalar(model_base.BaseModelFrozen):
  time: pydantic_types.NumpyArray1DSorted
  value: pydantic_types.NumpyArray
  is_bool_param: typing_extensions.Annotated[bool, model_base.JAX_STATIC] = (
      False
  )
  interpolation_mode: typing_extensions.Annotated[
      interpolated_param.InterpolationMode, model_base.JAX_STATIC
  ] = interpolated_param.InterpolationMode.PIECEWISE_LINEAR

  def get_value(self, t: chex.Numeric) -> array_typing.Array:
    """Returns the value of this parameter interpolated at time t.

    Args:
      t: An array of times to interpolate at.

    Returns:
      An array of interpolated values.
    """
    return self._get_cached_interpolated_param.get_value(t)

  @pydantic.model_validator(mode='after')
  def _ensure_consistent_arrays(self) -> typing_extensions.Self:
    return self

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(
      cls, data
  ):
    time, value, interpolation_mode, is_bool_param = (
        interpolated_param.convert_input_to_xs_ys(data)
    )
    sort_order = np.argsort(time)
    time = time[sort_order]
    value = value[sort_order]
    return dict(
        time=time,
        value=value,
        interpolation_mode=interpolation_mode,
        is_bool_param=is_bool_param,
    )

  @functools.cached_property
  def _get_cached_interpolated_param(
      self,
  ) -> interpolated_param.InterpolatedVarSingleAxis:
    """Interpolates the input param at time t."""

    return interpolated_param.InterpolatedVarSingleAxis(
        value=(self.time, self.value),
        interpolation_mode=self.interpolation_mode,
        is_bool_param=self.is_bool_param,
    )


def _is_positive(time_varying_scalar: TimeVaryingScalar) -> TimeVaryingScalar:
  if not np.all(time_varying_scalar.value > 0):
    raise ValueError('All values must be positive.')
  return time_varying_scalar


def _interval(
    time_varying_scalar: TimeVaryingScalar,
    lower_bound: float,
    upper_bound: float,
) -> TimeVaryingScalar:
  """Validates that values are in the interval [lower_bound, upper_bound]."""
  if not np.all(
      (time_varying_scalar.value >= lower_bound)
      & (time_varying_scalar.value <= upper_bound)
  ):
    raise ValueError(
        'All values must be less than %f and greater than %f.'
        % (upper_bound, lower_bound)
    )
  return time_varying_scalar


PositiveTimeVaryingScalar: TypeAlias = typing_extensions.Annotated[
    TimeVaryingScalar, pydantic.AfterValidator(_is_positive)
]
UnitIntervalTimeVaryingScalar: TypeAlias = typing_extensions.Annotated[
    TimeVaryingScalar,
    pydantic.AfterValidator(
        functools.partial(_interval, lower_bound=0.0, upper_bound=1.0)
    ),
]
