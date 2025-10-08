from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import math_utils


def make_convection_terms(v_face,
                          d_face,
                          var,
                          dirichlet_mode='ghost',
                          neumann_mode='ghost'):
    eps = 1e-20
    is_neg = d_face < 0.0
    nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
    d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))
    half = jnp.array([0.5], dtype=jax_utils.get_dtype())
    ones = jnp.ones_like(v_face[1:-1])
    scale = jnp.concatenate((half, ones, half))
    ratio = scale * var.dr * v_face / d_face
    left_peclet = -ratio[:-1]
    right_peclet = ratio[1:]

    def peclet_to_alpha(p):
        eps = 1e-3
        p = jnp.where(jnp.abs(p) < eps, eps, p)
        alpha_pg10 = (p - 1) / p
        alpha_p0to10 = ((p - 1) + (1 - p / 10)**5) / p
        alpha_pneg10to0 = ((1 + p / 10)**5 - 1) / p
        alpha_plneg10 = -1 / p
        alpha = 0.5 * jnp.ones_like(p)
        alpha = jnp.where(p > 10.0, alpha_pg10, alpha)
        alpha = jnp.where(jnp.logical_and(10.0 >= p, p > eps), alpha_p0to10,
                          alpha)
        alpha = jnp.where(jnp.logical_and(-eps > p, p >= -10), alpha_pneg10to0,
                          alpha)
        alpha = jnp.where(p < -10.0, alpha_plneg10, alpha)
        return alpha

    left_alpha = peclet_to_alpha(left_peclet)
    right_alpha = peclet_to_alpha(right_peclet)
    left_v = v_face[:-1]
    right_v = v_face[1:]
    diag = (left_alpha * left_v - right_alpha * right_v) / var.dr
    above = -(1.0 - right_alpha) * right_v / var.dr
    above = above[:-1]
    below = (1.0 - left_alpha) * left_v / var.dr
    below = below[1:]
    mat = math_utils.tridiag(diag, above, below)
    vec = jnp.zeros_like(diag)
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / var.dr
    vec_value = (-v_face[0] * (1.0 - left_alpha[0]) *
                 var.left_face_grad_constraint)
    mat = mat.at[0, 0].set(mat_value)
    vec = vec.at[0].set(vec_value)
    if var.right_face_constraint is not None:
        mat_value = (v_face[-2] * left_alpha[-1] + v_face[-1] *
                     (1.0 - 2.0 * right_alpha[-1])) / var.dr
        vec_value = (-2.0 * v_face[-1] * (1.0 - right_alpha[-1]) *
                     var.right_face_constraint) / var.dr
    else:
        mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / var.dr
        vec_value = (-v_face[-1] * (1.0 - right_alpha[-1]) *
                     var.right_face_grad_constraint)
    mat = mat.at[-1, -1].set(mat_value)
    vec = vec.at[-1].set(vec_value)
    return mat, vec
