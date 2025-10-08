import threading
from typing import Any, Callable, TYPE_CHECKING, TypeVar
import jax
import jax.numpy as jnp
import numpy as np
if TYPE_CHECKING:
  from jax.numpy import *
T = TypeVar('T')
BooleanNumeric = Any  
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
def py_cond(
    cond_val: bool,
    true_fun: Callable,
    false_fun: Callable,
    *operands,
) -> Any:
  if cond_val:
    return true_fun(*operands)
  else:
    return false_fun(*operands)
def cond(
    cond_val: bool,
    true_fun: Callable[..., Any],  
    false_fun: Callable[..., Any],  
    *operands,
) -> Any:
  is_jax = getattr(thread_context, 'is_jax', False)
  if is_jax:
    return jax.lax.cond(cond_val, true_fun, false_fun, *operands)
  else:
    return py_cond(cond_val, true_fun, false_fun, *operands)
def py_fori_loop(
    lower: int, upper: int, body_fun: Callable[[int, T], T], init_val: T  
) -> T:
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val
def fori_loop(
    lower: int,
    upper: int,
    body_fun: Callable[..., Any],  
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
def __getattr__(name):  
  current_lib = _get_current_lib()
  return getattr(current_lib, name)
