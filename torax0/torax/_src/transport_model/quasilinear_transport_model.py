import dataclasses
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib
import typing_extensions
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NormalizedLogarithmicGradients:
  lref_over_lti: array_typing.FloatVectorFace
  lref_over_lte: array_typing.FloatVectorFace
  lref_over_lne: array_typing.FloatVectorFace
  lref_over_lni0: array_typing.FloatVectorFace
  lref_over_lni1: array_typing.FloatVectorFace
  @classmethod
  def from_profiles(
      cls,
      core_profiles: state.CoreProfiles,
      radial_coordinate: jnp.ndarray,
      reference_length: jnp.ndarray,
  ) -> typing_extensions.Self:
    gradients = {}
    for name, profile in {
        "lref_over_lti": core_profiles.T_i,
        "lref_over_lte": core_profiles.T_e,
        "lref_over_lne": core_profiles.n_e,
        "lref_over_lni0": core_profiles.n_i,
        "lref_over_lni1": core_profiles.n_impurity,
    }.items():
      gradients[name] = calculate_normalized_logarithmic_gradient(
          var=profile,
          radial_coordinate=radial_coordinate,
          reference_length=reference_length,
      )
    return cls(**gradients)
def calculate_chiGB(
    reference_temperature: array_typing.Array,
    reference_magnetic_field: chex.Numeric,
    reference_mass: chex.Numeric,
    reference_length: chex.Numeric,
) -> array_typing.Array:
  constants = constants_module.CONSTANTS
  return (
      (reference_mass * constants.m_amu) ** 0.5
      / (reference_magnetic_field * constants.q_e) ** 2
      * (reference_temperature * constants.keV_to_J) ** 1.5
      / reference_length
  )
def calculate_alpha(
    core_profiles: state.CoreProfiles,
    q: array_typing.FloatVectorFace,
    reference_magnetic_field: chex.Numeric,
    normalized_logarithmic_gradients: NormalizedLogarithmicGradients,
) -> array_typing.FloatVectorFace:
  constants = constants_module.CONSTANTS
  factor_0 = (
      2
      * constants.keV_to_J
      / reference_magnetic_field**2
      * constants.mu_0
      * q**2
  )
  alpha = factor_0 * (
      core_profiles.T_e.face_value()
      * core_profiles.n_e.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lte
          + normalized_logarithmic_gradients.lref_over_lne
      )
      + core_profiles.n_i.face_value()
      * core_profiles.T_i.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lti
          + normalized_logarithmic_gradients.lref_over_lni0
      )
      + core_profiles.n_impurity.face_value()
      * core_profiles.T_i.face_value()
      * (
          normalized_logarithmic_gradients.lref_over_lti
          + normalized_logarithmic_gradients.lref_over_lni1
      )
  )
  return alpha
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  DV_effective: bool
  An_min: float
def calculate_normalized_logarithmic_gradient(
    var: cell_variable.CellVariable,
    radial_coordinate: jax.Array,
    reference_length: jax.Array,
) -> jax.Array:
  result = jnp.where(
      jnp.abs(var.face_value()) < constants_module.CONSTANTS.eps,
      constants_module.CONSTANTS.eps,
      -reference_length * var.face_grad(radial_coordinate) / var.face_value(),
  )
  result = jnp.where(
      jnp.abs(result) < constants_module.CONSTANTS.eps,
      constants_module.CONSTANTS.eps,
      result,
  )
  return result
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QuasilinearInputs:
  chiGB: (
      array_typing.FloatVectorFace
  )  
  Rmin: array_typing.FloatScalar  
  Rmaj: array_typing.FloatScalar  
  lref_over_lti: array_typing.FloatVectorFace
  lref_over_lte: array_typing.FloatVectorFace
  lref_over_lne: array_typing.FloatVectorFace
  lref_over_lni0: array_typing.FloatVectorFace
  lref_over_lni1: array_typing.FloatVectorFace
class QuasilinearTransportModel(transport_model_lib.TransportModel):
  def _make_core_transport(
      self,
      qi: jax.Array,
      qe: jax.Array,
      pfe: jax.Array,
      quasilinear_inputs: QuasilinearInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      gradient_reference_length: chex.Numeric,
      gyrobohm_flux_reference_length: chex.Numeric,
  ) -> transport_model_lib.TurbulentTransport:
    constants = constants_module.CONSTANTS
    pfe_SI = (
        pfe
        * core_profiles.n_e.face_value()
        * quasilinear_inputs.chiGB
        / gyrobohm_flux_reference_length
    )
    chi_face_ion = (
        ((gradient_reference_length / gyrobohm_flux_reference_length) * qi)
        / quasilinear_inputs.lref_over_lti
    ) * quasilinear_inputs.chiGB
    chi_face_el = (
        ((gradient_reference_length / gyrobohm_flux_reference_length) * qe)
        / quasilinear_inputs.lref_over_lte
    ) * quasilinear_inputs.chiGB
    def DV_effective_approach() -> tuple[jax.Array, jax.Array]:
      Deff = -pfe_SI / (
          core_profiles.n_e.face_grad() * geo.g1_over_vpr2_face * geo.rho_b
          + constants.eps
      )
      Veff = pfe_SI / (
          core_profiles.n_e.face_value() * geo.g0_over_vpr_face * geo.rho_b
      )
      Deff_mask = (
          ((pfe >= 0) & (quasilinear_inputs.lref_over_lne >= 0))
          | ((pfe < 0) & (quasilinear_inputs.lref_over_lne < 0))
      ) & (abs(quasilinear_inputs.lref_over_lne) >= transport.An_min)
      Veff_mask = jnp.invert(Deff_mask)
      d_face_el = jnp.where(Veff_mask, 0.0, Deff)
      v_face_el = jnp.where(Deff_mask, 0.0, Veff)
      return d_face_el, v_face_el
    def Dscaled_approach():
      d_face_el = chi_face_el
      v_face_el = (
          pfe_SI / core_profiles.n_e.face_value()
          - quasilinear_inputs.lref_over_lne
          * d_face_el
          / gradient_reference_length
          * geo.g1_over_vpr2_face
          * geo.rho_b**2
      ) / (geo.g0_over_vpr_face * geo.rho_b)
      return d_face_el, v_face_el
    d_face_el, v_face_el = jax.lax.cond(
        transport.DV_effective,
        DV_effective_approach,
        Dscaled_approach,
    )
    return transport_model_lib.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
