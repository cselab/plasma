# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common types for using jaxtyping in TORAX."""

from typing import TypeAlias, TypeVar
import jax
import jaxtyping as jt
import numpy as np
from torax._src import jax_utils
import typeguard

T = TypeVar("T")

Array: TypeAlias = jax.Array | np.ndarray

FloatScalar: TypeAlias = jt.Float[Array | float, ""]
BoolScalar: TypeAlias = jt.Bool[Array | bool, ""]
IntScalar: TypeAlias = jt.Int[Array | int, ""]

FloatVector: TypeAlias = jt.Float[Array, "_"]
BoolVector: TypeAlias = jt.Bool[Array, "_"]
FloatVectorCell: TypeAlias = jt.Float[Array, "rhon"]
FloatVectorCellPlusBoundaries: TypeAlias = jt.Float[Array, "rhon+2"]
FloatMatrixCell: TypeAlias = jt.Float[Array, "rhon rhon"]
FloatVectorFace: TypeAlias = jt.Float[Array, "rhon+1"]

def jaxtyped(fn):
  return fn
