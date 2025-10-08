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
"""Zeros model for bootstrap current."""
from typing import Annotated, Literal

import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_runtime_params
from torax._src.torax_pydantic import torax_pydantic

class ZerosModel(base.BootstrapCurrentModel):
  pass

class ZerosModelConfig(base.BootstrapCurrentModelConfig):
  model_name: Annotated[Literal['zeros'], torax_pydantic.JAX_STATIC] = 'zeros'
  def build_runtime_params(self):
    return bootstrap_runtime_params.RuntimeParams()
  def build_model(self):
    return ZerosModel()
