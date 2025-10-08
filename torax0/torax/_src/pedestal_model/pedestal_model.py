import abc
import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
    rho_norm_ped_top: array_typing.FloatScalar
    rho_norm_ped_top_idx: array_typing.IntScalar
    T_i_ped: array_typing.FloatScalar
    T_e_ped: array_typing.FloatScalar
    n_e_ped: array_typing.FloatScalar


class PedestalModel(abc.ABC):

    def __setattr__(self, attr, value):
        return super().__setattr__(attr, value)

    def __call__(
        self,
        runtime_params,
        geo,
        core_profiles,
    ):
        return jax.lax.cond(
            runtime_params.pedestal.set_pedestal,
            lambda: self._call_implementation(runtime_params, geo,
                                              core_profiles),
            lambda: PedestalModelOutput(
                rho_norm_ped_top=jnp.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
                rho_norm_ped_top_idx=geo.torax_mesh.nx,
            ),
        )

    @abc.abstractmethod
    def _call_implementation(
        self,
        runtime_params,
        geo,
        core_profiles,
    ):
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        ...
