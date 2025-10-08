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
"""Polymorphic JAX/NumPy module."""

import contextlib
import functools
import threading
from typing import Any, Callable, TYPE_CHECKING, TypeVar

from absl import logging as native_logging
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils

# Export all symbols from jax.numpy API for type checkers (including editors).
# pylint: disable=wildcard-import
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
if TYPE_CHECKING:
  from jax.numpy import *
# pylint: enable=wildcard-import
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top


T = TypeVar('T')
BooleanNumeric = Any  # A bool, or a Boolean array.

# Create thread-local storage.
thread_context = threading.local()
def jit(*args, **kwargs):
  func = args[0]
  return func

def py_while(
    cond_fun: Callable[list[T], BooleanNumeric],
    body_fun: Callable[list[T], T],
    init_val: T,
) -> T:
  val = init_val
  while cond_fun(val):
    val = body_fun(val)
  return val


def while_loop(
    cond_fun: Callable[list[T], BooleanNumeric],
    body_fun: Callable[list[T], T],
    init_val: T,
):
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.while_loop(cond_fun, body_fun, init_val)
  else:
    return py_while(cond_fun, body_fun, init_val)


# pylint: disable=g-bare-generic
def py_cond(
    cond_val: bool,
    true_fun: Callable,
    false_fun: Callable,
    *operands,
) -> Any:
  """Pure Python implementation of jax.lax.cond.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future, if we want to expand the scope of the jit
  compilation.

  Args:
    cond_val: The condition.
    true_fun: Function to be called if cond==True.
    false_fun: Function to be called if cond==False.
    *operands: The operands to be passed to the functions.

  Returns:
    The output from either true_fun or false_fun.
  """
  if cond_val:
    return true_fun(*operands)
  else:
    return false_fun(*operands)


def cond(
    cond_val: bool,
    true_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    false_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    *operands,
) -> Any:
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.cond(cond_val, true_fun, false_fun, *operands)
  else:
    return py_cond(cond_val, true_fun, false_fun, *operands)


def py_fori_loop(
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T  # pytype: disable=invalid-annotation
) -> T:
  """Pure Python implementation of jax.lax.fori_loop.

  This gives us a way to write code that could easily be changed to be
  Jax-compatible in the future, if we want to expand the scope of the jit
  compilation.

  Args:
    lower: lower integer of loop
    upper: upper integer of loop. upper<=lower will produce no iterations.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.

  .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
  """
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val


def fori_loop(
    lower: int,
    upper: int,
    body_fun: Callable[..., Any],  # pytype: disable=invalid-annotation
    init_val: Any,
):
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)
  else:
    return py_fori_loop(lower, upper, body_fun, init_val)

def _get_current_lib():
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jnp
  else:
    return np


def __getattr__(name):  # pylint: disable=invalid-name
  current_lib = _get_current_lib()
  return getattr(current_lib, name)

