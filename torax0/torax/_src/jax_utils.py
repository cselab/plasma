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
"""Commonly repeated jax expressions."""

import functools
import os
from typing import Any, Callable, ParamSpec, TypeAlias, TypeVar

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np

T = TypeVar('T')
BooleanNumeric: TypeAlias = Any  # A bool, or a Boolean array.
_State = ParamSpec('_State')


@functools.cache
def get_dtype():
    precision = os.getenv('JAX_PRECISION', 'f64')
    return jnp.float64 if precision == 'f64' else jnp.float32


@functools.cache
def get_np_dtype():
    precision = os.getenv('JAX_PRECISION', 'f64')
    return np.float64 if precision == 'f64' else np.float32

def jit(*args, **kwargs):
    return jax.jit(*args, **kwargs)
