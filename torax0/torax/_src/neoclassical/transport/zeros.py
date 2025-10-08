from typing import Annotated, Literal
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import base
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override
class ZerosModel(base.NeoclassicalTransportModel):
  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.NeoclassicalTransport:
    return base.NeoclassicalTransport(
        chi_neo_i=jnp.zeros_like(geometry.rho_face),
        chi_neo_e=jnp.zeros_like(geometry.rho_face),
        D_neo_e=jnp.zeros_like(geometry.rho_face),
        V_neo_e=jnp.zeros_like(geometry.rho_face),
        V_neo_ware_e=jnp.zeros_like(geometry.rho_face),
    )
  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)
  def __hash__(self) -> int:
    return hash(self.__class__)
class ZerosModelConfig(base.NeoclassicalTransportModelConfig):
  model_name: Annotated[Literal['zeros'], torax_pydantic.JAX_STATIC] = 'zeros'
  def build_model(self) -> ZerosModel:
    return ZerosModel()
