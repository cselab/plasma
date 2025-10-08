import functools
import os
from typing import Any, ParamSpec, TypeAlias, TypeVar
import jax
from jax import numpy as jnp
import numpy as np
T = TypeVar('T')
BooleanNumeric: TypeAlias = Any  
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
