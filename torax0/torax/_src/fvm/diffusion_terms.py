import chex
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src.fvm import cell_variable
def make_diffusion_terms(
    d_face: array_typing.FloatVectorFace, var: cell_variable.CellVariable
) -> tuple[array_typing.FloatMatrixCell, array_typing.FloatVectorCell]:
  denom = var.dr**2
  diag = jnp.asarray(-d_face[1:] - d_face[:-1])
  off = d_face[1:-1]
  vec = jnp.zeros_like(diag)
  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )
  chex.assert_exactly_one_is_none(
      var.left_face_grad_constraint, var.left_face_constraint
  )
  chex.assert_exactly_one_is_none(
      var.right_face_grad_constraint, var.right_face_constraint
  )
  if var.left_face_constraint is not None:
    diag = diag.at[0].set(-2 * d_face[0] - d_face[1])
    vec = vec.at[0].set(2 * d_face[0] * var.left_face_constraint / denom)
  else:
    diag = diag.at[0].set(-d_face[1])
    vec = vec.at[0].set(-d_face[0] * var.left_face_grad_constraint / var.dr)
  if var.right_face_constraint is not None:
    diag = diag.at[-1].set(-2 * d_face[-1] - d_face[-2])
    vec = vec.at[-1].set(2 * d_face[-1] * var.right_face_constraint / denom)
  else:
    diag = diag.at[-1].set(-d_face[-2])
    vec = vec.at[-1].set(d_face[-1] * var.right_face_grad_constraint / var.dr)
  mat = math_utils.tridiag(diag, off, off) / denom
  return mat, vec
