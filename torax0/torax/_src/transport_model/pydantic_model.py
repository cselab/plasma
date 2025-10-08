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

"""Pydantic config for Transport models."""

import copy
import dataclasses
from typing import Annotated, Any, Literal

import chex
import pydantic
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import qlknn_transport_model

# pylint: disable=invalid-name
class QLKNNTransportModel(pydantic_model_base.TransportBase):
  model_name: Annotated[Literal['qlknn'], torax_pydantic.JAX_STATIC] = 'qlknn'
  model_path: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  qlknn_model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  include_ITG: bool = True
  include_TEM: bool = True
  include_ETG: bool = True
  ITG_flux_ratio_correction: float = 1.0
  ETG_correction_factor: float = 1.0 / 3.0
  clip_inputs: bool = False
  clip_margin: float = 0.95
  collisionality_multiplier: float = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    data['qlknn_model_name'] = data.get('qlknn_model_name', '')
    if 'smoothing_width' not in data:
      data['smoothing_width'] = 0.1
    return data

  def build_transport_model(self) -> qlknn_transport_model.QLKNNTransportModel:
    return qlknn_transport_model.QLKNNTransportModel(
        path=self.model_path, name=self.qlknn_model_name
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> qlknn_transport_model.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return qlknn_transport_model.RuntimeParams(
        include_ITG=self.include_ITG,
        include_TEM=self.include_TEM,
        include_ETG=self.include_ETG,
        ITG_flux_ratio_correction=self.ITG_flux_ratio_correction,
        ETG_correction_factor=self.ETG_correction_factor,
        clip_inputs=self.clip_inputs,
        clip_margin=self.clip_margin,
        collisionality_multiplier=self.collisionality_multiplier,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        **base_kwargs,
    )



CombinedCompatibleTransportModel = QLKNNTransportModel
TransportConfig = CombinedCompatibleTransportModel  # pytype: disable=invalid-annotation
