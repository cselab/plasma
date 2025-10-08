import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms
from torax._src.geometry import geometry
_trapz = jax.scipy.integrate.trapezoid
def calc_q_face(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> array_typing.FloatVectorFace:
  inv_iota = jnp.abs(
      (2 * geo.Phi_b * geo.rho_face_norm[1:]) / psi.face_grad()[1:]
  )
  inv_iota0 = jnp.expand_dims(
      jnp.abs((2 * geo.Phi_b * geo.drho_norm) / psi.face_grad()[1]), 0
  )
  q_face = jnp.concatenate([inv_iota0, inv_iota])
  return q_face * geo.q_correction_factor
def calc_j_total(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorFace,
    array_typing.FloatVectorFace,
]:
  Ip_profile_face = (
      psi.face_grad()
      * geo.g2g3_over_rhon_face
      * geo.F_face
      / geo.Phi_b
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu_0)
  )
  Ip_profile = (
      psi.grad()
      * geo.g2g3_over_rhon
      * geo.F
      / geo.Phi_b
      / (16 * jnp.pi**3 * constants.CONSTANTS.mu_0)
  )
  dI_drhon_face = jnp.gradient(Ip_profile_face, geo.rho_face_norm)
  dI_drhon = jnp.gradient(Ip_profile, geo.rho_norm)
  j_total_bulk = dI_drhon[1:] / geo.spr[1:]
  j_total_face_bulk = dI_drhon_face[1:] / geo.spr_face[1:]
  j_total_axis = j_total_bulk[0] - (j_total_bulk[1] - j_total_bulk[0])
  j_total = jnp.concatenate([jnp.array([j_total_axis]), j_total_bulk])
  j_total_face = jnp.concatenate([jnp.array([j_total_axis]), j_total_face_bulk])
  return j_total, j_total_face, Ip_profile_face
def calc_s_face(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )
  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
  s_face = (
      -geo.rho_face_norm
      * jnp.gradient(iota_scaled, geo.rho_face_norm)
      / iota_scaled
  )
  return s_face
def calc_s_rmid(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))
  iota_scaled0 = jnp.expand_dims(
      jnp.abs(psi.face_grad()[1] / geo.drho_norm), axis=0
  )
  iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
  rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5
  s_face = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled
  return s_face
def _calc_bpol2(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  bpol2_bulk = (
      (psi.face_grad()[1:] / (2 * jnp.pi)) ** 2
      * geo.g2_face[1:]
      / geo.vpr_face[1:] ** 2
  )
  bpol2_axis = jnp.array([0.0], dtype=jax_utils.get_dtype())
  bpol2_face = jnp.concatenate([bpol2_axis, bpol2_bulk])
  return bpol2_face
def calc_Wpol(
    geo: geometry.Geometry, psi: cell_variable.CellVariable
) -> jax.Array:
  bpol2 = _calc_bpol2(geo, psi)
  Wpol = _trapz(bpol2 * geo.vpr_face, geo.rho_face_norm) / (
      2 * constants.CONSTANTS.mu_0
  )
  return Wpol
def calc_li3(
    R_major: jax.Array,
    Wpol: jax.Array,
    Ip_total: jax.Array,
) -> jax.Array:
  return 4 * Wpol / (constants.CONSTANTS.mu_0 * Ip_total**2 * R_major)
def calc_q95(
    psi_norm_face: array_typing.FloatVector,
    q_face: array_typing.FloatVector,
) -> array_typing.FloatScalar:
  q95 = jnp.interp(0.95, psi_norm_face, q_face)
  return q95
def calculate_psi_grad_constraint_from_Ip(
    Ip: array_typing.FloatScalar,
    geo: geometry.Geometry,
) -> jax.Array:
  return (
      Ip
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu_0 * geo.Phi_b)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )
def calculate_psidot_from_psi_sources(
    *,
    psi_sources: array_typing.FloatVector,
    sigma: array_typing.FloatVector,
    resistivity_multiplier: float,
    psi: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> jax.Array:
  consts = constants.CONSTANTS
  toc_psi = (
      1.0
      / resistivity_multiplier
      * geo.rho_norm
      * sigma
      * consts.mu_0
      * 16
      * jnp.pi**2
      * geo.Phi_b**2
      / geo.F**2
  )
  d_face_psi = geo.g2g3_over_rhon_face
  v_face_psi = jnp.zeros_like(d_face_psi)
  psi_sources += (
      8.0
      * jnp.pi**2
      * consts.mu_0
      * geo.Phi_b_dot
      * geo.Phi_b
      * geo.rho_norm**2
      * sigma
      / geo.F**2
      * psi.grad()
  )
  diffusion_mat, diffusion_vec = diffusion_terms.make_diffusion_terms(
      d_face_psi, psi
  )
  conv_mat, conv_vec = convection_terms.make_convection_terms(
      v_face_psi, d_face_psi, psi
  )
  c_mat = diffusion_mat + conv_mat
  c = diffusion_vec + conv_vec
  c += psi_sources
  psidot = (jnp.dot(c_mat, psi.value) + c) / toc_psi
  return psidot
