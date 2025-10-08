import abc
import dataclasses
import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
class TransportModel(abc.ABC):
  def __setattr__(self, attr, value):
    if getattr(self, "_frozen", False):
      raise AttributeError("TransportModels are immutable.")
    return super().__setattr__(attr, value)
  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    if not getattr(self, "_frozen", False):
      raise RuntimeError(
          f"Subclass implementation {type(self)} forgot to "
          "freeze at the end of __init__."
      )
    transport_runtime_params = runtime_params.transport
    transport_coeffs = self._call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    )
    transport_coeffs = self._apply_domain_restriction(
        transport_runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )
    transport_coeffs = self._apply_transport_patches(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
    )
    return self._smooth_coeffs(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )
  @abc.abstractmethod
  def _call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    pass
  @abc.abstractmethod
  def __hash__(self) -> int:
    pass
  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass
  def _apply_domain_restriction(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    active_mask = (
        (geo.rho_face_norm > transport_runtime_params.rho_min)
        & (geo.rho_face_norm <= transport_runtime_params.rho_max)
        & (geo.rho_face_norm <= pedestal_model_output.rho_norm_ped_top)
    )
    active_mask = (
        jnp.asarray(active_mask)
        .at[0]
        .set(transport_runtime_params.rho_min == 0)
    )
    chi_face_ion = jnp.where(active_mask, transport_coeffs.chi_face_ion, 0.0)
    chi_face_el = jnp.where(active_mask, transport_coeffs.chi_face_el, 0.0)
    d_face_el = jnp.where(active_mask, transport_coeffs.d_face_el, 0.0)
    v_face_el = jnp.where(active_mask, transport_coeffs.v_face_el, 0.0)
    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
  def _apply_clipping(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    chi_face_ion = jnp.clip(
        transport_coeffs.chi_face_ion,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    chi_face_el = jnp.clip(
        transport_coeffs.chi_face_el,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    d_face_el = jnp.clip(
        transport_coeffs.d_face_el,
        transport_runtime_params.D_e_min,
        transport_runtime_params.D_e_max,
    )
    v_face_el = jnp.clip(
        transport_coeffs.v_face_el,
        transport_runtime_params.V_e_min,
        transport_runtime_params.V_e_max,
    )
    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
  def _apply_transport_patches(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    consts = constants.CONSTANTS
    chi_face_ion = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_i_inner,
        transport_coeffs.chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_e_inner,
        transport_coeffs.chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.D_e_inner,
        transport_coeffs.d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.V_e_inner,
        transport_coeffs.v_face_el,
    )
    chi_face_ion = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    runtime_params.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_i_outer,
        chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    runtime_params.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_e_outer,
        chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    runtime_params.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.D_e_outer,
        d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    runtime_params.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.V_e_outer,
        v_face_el,
    )
    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
  def _smooth_coeffs(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    smoothing_matrix = _build_smoothing_matrix(
        transport_runtime_params,
        runtime_params,
        geo,
        pedestal_model_output,
    )
    def smooth_single_coeff(coeff):
      return jax.lax.cond(
          jnp.all(coeff == 0.0),
          lambda: coeff,
          lambda: jnp.dot(smoothing_matrix, coeff),
      )
    return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)
def _build_smoothing_matrix(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
) -> jax.Array:
  lower_cutoff = 0.01
  consts = constants.CONSTANTS
  kernel = jnp.exp(
      -jnp.log(2)
      * (geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm) ** 2
      / (transport_runtime_params.smoothing_width**2 + consts.eps)
  )
  mask_outer_edge = jax.lax.cond(
      jnp.logical_and(
          jnp.logical_not(runtime_params.pedestal.set_pedestal),
          transport_runtime_params.apply_outer_patch,
      ),
      lambda: transport_runtime_params.rho_outer - consts.eps,
      lambda: pedestal_model_output.rho_norm_ped_top - consts.eps,
  )
  mask_inner_edge = jax.lax.cond(
      transport_runtime_params.apply_inner_patch,
      lambda: transport_runtime_params.rho_inner + consts.eps,
      lambda: 0.0,
  )
  mask = jnp.where(
      jnp.logical_or(
          transport_runtime_params.smooth_everywhere,
          jnp.logical_and(
              geo.rho_face_norm > mask_inner_edge,
              geo.rho_face_norm < mask_outer_edge,
          ),
      ),
      1.0,
      0.0,
  )
  diag_mask = jnp.diag(mask)
  kernel = jnp.dot(diag_mask, kernel)
  num_rows = len(mask)
  mask_mat = jnp.tile(mask, (num_rows, 1))
  kernel *= mask_mat
  zero_row_mask = jnp.all(kernel == 0, axis=1)
  kernel = jnp.where(
      zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]), kernel
  )
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]
  kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]
  return kernel
