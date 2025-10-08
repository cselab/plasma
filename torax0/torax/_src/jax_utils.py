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
def get_dtype() -> type(jnp.float32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return jnp.float64 if precision == 'f64' else jnp.float32


@functools.cache
def get_np_dtype() -> type(np.float32):
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  return np.float64 if precision == 'f64' else np.float32


def env_bool(name: str, default: bool) -> bool:
  """Get a bool from an environment variable.

  Args:
    name: The name of the environment variable.
    default: The default value of the bool.

  Returns:
    value: The value of the bool.
  """
  if name not in os.environ:
    return default
  str_value = os.environ[name]
  if str_value in ['1', 'True', 'true']:
    return True
  if str_value in ['0', 'False', 'false']:
    return False
  raise ValueError(f'Unrecognized boolean string {str_value}.')


# If True, `error_if` functions will raise errors.  Otherwise they are
# pass throughs.
# Default to False, because host_callbacks are incompatible with the
# persistent compilation cache.
_ERRORS_ENABLED: bool = env_bool('TORAX_ERRORS_ENABLED', False)


def error_if(
    var: jax.Array,
    cond: jax.Array,
    msg: str,
) -> jax.Array:
  """Raises error if cond is true, and `errors_enabled` is True.

  This is just a wrapper around `equinox.error_if`, gated by `errors_enabled`.

  Args:
    var: The variable to pass through.
    cond: Boolean array, error if cond is true.
    msg: Message to print on error.

  Returns:
    var: Identity wrapper that must be used for the check to be included.
  """
  if not _ERRORS_ENABLED:
    return var
  return eqx.error_if(var, cond, msg)


def assert_rank(
    inputs: chex.Numeric | jax.stages.ArgInfo,
    rank: int,
) -> None:
  """Wrapper around chex.assert_rank that supports jax.stages.ArgInfo."""
  if isinstance(inputs, jax.stages.ArgInfo):
    chex.assert_rank(inputs.shape, rank)
  else:
    chex.assert_rank(inputs, rank)


def jit(*args, **kwargs) -> Callable[..., Any]:
  """Calls jax.jit if TORAX_COMPILATION_ENABLED is True, otherwise no-op."""
  if env_bool('TORAX_COMPILATION_ENABLED', True):
    return jax.jit(*args, **kwargs)
  return args[0]
