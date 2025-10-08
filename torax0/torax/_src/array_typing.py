from typing import TypeAlias, TypeVar
import jax
import jaxtyping as jt
import numpy as np
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
