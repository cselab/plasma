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

"""Pydantic config for Pedestal."""
import abc
from typing import Annotated, Literal

import chex
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_tped_nped
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal. Can be
      time varying.
  """

  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )

  @abc.abstractmethod
  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the pedestal model."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params."""


class SetTpedNped(BasePedestal):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_ped: Ion temperature at the pedestal [keV].
    T_e_ped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal['set_T_ped_n_ped'], torax_pydantic.JAX_STATIC
  ] = 'set_T_ped_n_ped'
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  T_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
    return set_tped_nped.SetTemperatureDensityPedestalModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.RuntimeParams:
    return set_tped_nped.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


PedestalConfig = SetTpedNped
