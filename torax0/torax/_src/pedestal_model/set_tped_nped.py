import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from typing_extensions import override


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    n_e_ped: array_typing.FloatScalar
    T_i_ped: array_typing.FloatScalar
    T_e_ped: array_typing.FloatScalar
    rho_norm_ped_top: array_typing.FloatScalar
    n_e_ped_is_fGW: array_typing.BoolScalar


class SetTemperatureDensityPedestalModel(pedestal_model.PedestalModel):

    def __init__(self, ):
        super().__init__()
        self._frozen = True

    @override
    def _call_implementation(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> pedestal_model.PedestalModelOutput:
        pedestal_params = runtime_params.pedestal
        assert isinstance(pedestal_params, RuntimeParams)
        nGW = (runtime_params.profile_conditions.Ip / 1e6 /
               (jnp.pi * geo.a_minor**2) * 1e20)
        n_e_ped = jnp.where(
            pedestal_params.n_e_ped_is_fGW,
            pedestal_params.n_e_ped * nGW,
            pedestal_params.n_e_ped,
        )
        return pedestal_model.PedestalModelOutput(
            n_e_ped=n_e_ped,
            T_i_ped=pedestal_params.T_i_ped,
            T_e_ped=pedestal_params.T_e_ped,
            rho_norm_ped_top=pedestal_params.rho_norm_ped_top,
            rho_norm_ped_top_idx=jnp.abs(
                geo.rho_norm - pedestal_params.rho_norm_ped_top).argmin(),
        )

    def __hash__(self) -> int:
        return hash('SetTemperatureDensityPedestalModel')

    def __eq__(self, other) -> bool:
        return isinstance(other, SetTemperatureDensityPedestalModel)
