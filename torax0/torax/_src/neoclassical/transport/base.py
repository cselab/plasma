import abc
import dataclasses
import jax
import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions
@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class NeoclassicalTransport:
  chi_neo_i: jax.Array
  chi_neo_e: jax.Array
  D_neo_e: jax.Array
  V_neo_e: jax.Array
  V_neo_ware_e: jax.Array
  @classmethod
  def zeros(cls, geometry: geometry_lib.Geometry) -> 'NeoclassicalTransport':
    return cls(
        chi_neo_i=jnp.zeros_like(geometry.rho_face_norm),
        chi_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        D_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        V_neo_e=jnp.zeros_like(geometry.rho_face_norm),
        V_neo_ware_e=jnp.zeros_like(geometry.rho_face_norm),
    )
class NeoclassicalTransportModel(abc.ABC):
  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> NeoclassicalTransport:
    neoclassical_transport = self._call_implementation(
        runtime_params,
        geometry,
        core_profiles,
    )
    neoclassical_transport = self._apply_clipping(
        runtime_params,
        neoclassical_transport,
    )
    return neoclassical_transport
  def _apply_clipping(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      neoclassical_transport: NeoclassicalTransport,
  ) -> NeoclassicalTransport:
    chi_neo_i = jnp.clip(
        neoclassical_transport.chi_neo_i,
        runtime_params.neoclassical.transport.chi_min,
        runtime_params.neoclassical.transport.chi_max,
    )
    chi_neo_e = jnp.clip(
        neoclassical_transport.chi_neo_e,
        runtime_params.neoclassical.transport.chi_min,
        runtime_params.neoclassical.transport.chi_max,
    )
    D_neo_e = jnp.clip(
        neoclassical_transport.D_neo_e,
        runtime_params.neoclassical.transport.D_e_min,
        runtime_params.neoclassical.transport.D_e_max,
    )
    V_neo_e = jnp.clip(
        neoclassical_transport.V_neo_e,
        runtime_params.neoclassical.transport.V_e_min,
        runtime_params.neoclassical.transport.V_e_max,
    )
    V_neo_ware_e = jnp.clip(
        neoclassical_transport.V_neo_ware_e,
        runtime_params.neoclassical.transport.V_e_min,
        runtime_params.neoclassical.transport.V_e_max,
    )
    return NeoclassicalTransport(
        chi_neo_i=chi_neo_i,
        chi_neo_e=chi_neo_e,
        D_neo_e=D_neo_e,
        V_neo_e=V_neo_e,
        V_neo_ware_e=V_neo_ware_e,
    )
  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> NeoclassicalTransport:
    pass
class NeoclassicalTransportModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  chi_min: torax_pydantic.MeterSquaredPerSecond = 0.0
  chi_max: torax_pydantic.MeterSquaredPerSecond = 100
  D_e_min: torax_pydantic.MeterSquaredPerSecond = 0.0
  D_e_max: torax_pydantic.MeterSquaredPerSecond = 100.0
  V_e_min: torax_pydantic.MeterPerSecond = -50.0
  V_e_max: torax_pydantic.MeterPerSecond = 50.0
  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.chi_min < self.chi_max:
      raise ValueError('chi_min must be less than chi_max.')
    if not self.D_e_min < self.D_e_max:
      raise ValueError('D_e_min must be less than D_e_max.')
    if not self.V_e_min < self.V_e_max:
      raise ValueError('V_e_min must be less than V_e_max.')
    return self
  def build_runtime_params(self) -> transport_runtime_params.RuntimeParams:
    return transport_runtime_params.RuntimeParams(
        chi_min=self.chi_min,
        chi_max=self.chi_max,
        D_e_min=self.D_e_min,
        D_e_max=self.D_e_max,
        V_e_min=self.V_e_min,
        V_e_max=self.V_e_max,
    )
  @abc.abstractmethod
  def build_model(self) -> NeoclassicalTransportModel:
    pass
