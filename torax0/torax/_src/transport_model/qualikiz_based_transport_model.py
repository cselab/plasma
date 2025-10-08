import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants as constants_module
from torax._src import state
from torax._src.geometry import geometry
from torax._src.physics import collisions
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_transport_model
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  collisionality_multiplier: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QualikizInputs(quasilinear_transport_model.QuasilinearInputs):
  Z_eff_face: array_typing.FloatVectorFace
  q: array_typing.FloatVectorFace
  smag: array_typing.FloatVectorFace
  x: array_typing.FloatVectorFace
  Ti_Te: array_typing.FloatVectorFace
  log_nu_star_face: array_typing.FloatVectorFace
  normni: array_typing.FloatVectorFace
  alpha: array_typing.FloatVectorFace
  epsilon_lcfs: array_typing.FloatScalar
  @property
  def Ati(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lti
  @property
  def Ate(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lte
  @property
  def Ane(self) -> array_typing.Array:
    return self.lref_over_lne
  @property
  def Ani0(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lni0
  @property
  def Ani1(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lni1
class QualikizBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  def _prepare_qualikiz_inputs(
      self,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> QualikizInputs:
    constants = constants_module.CONSTANTS
    rmid = (geo.R_out - geo.R_in) * 0.5
    rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.T_i.face_value(),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.A_i,
        reference_length=geo.a_minor,
    )
    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=rmid,
        reference_length=geo.R_major,
    )
    q = core_profiles.q_face
    smag = psi_calculations.calc_s_rmid(
        geo,
        core_profiles.psi,
    )
    epsilon_lcfs = rmid_face[-1] / geo.R_major
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)
    Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
    nu_star = collisions.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        collisionality_multiplier=transport.collisionality_multiplier,
    )
    log_nu_star_face = jnp.log10(nu_star)
    alpha = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        q=q,
        reference_magnetic_field=geo.B_0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )
    smag = jnp.where(
        transport.smag_alpha_correction,
        smag - alpha / 2,
        smag,
    )
    smag = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        0.1,
        smag,
    )
    q = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        1,
        q,
    )
    smag = jnp.where(
        jnp.logical_and(
            transport.avoid_big_negative_s,
            smag - alpha < -0.2,
        ),
        alpha - 0.2,
        smag,
    )
    normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value()
    return QualikizInputs(
        Z_eff_face=core_profiles.Z_eff_face,
        lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
        lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
        lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
        lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
        lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
        q=q,
        smag=smag,
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=geo.R_major,
        Rmin=geo.a_minor,
        alpha=alpha,
        epsilon_lcfs=epsilon_lcfs,
    )
