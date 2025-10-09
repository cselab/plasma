import abc
import dataclasses
import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_runtime_params
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
    j_bootstrap: jax.Array
    j_bootstrap_face: jax.Array

    @classmethod
    def zeros(cls, geometry: geometry_lib.Geometry) -> 'BootstrapCurrent':
        return cls(
            j_bootstrap=jnp.zeros_like(geometry.rho_norm),
            j_bootstrap_face=jnp.zeros_like(geometry.rho_face_norm),
        )


class BootstrapCurrentModel(abc.ABC):

    @abc.abstractmethod
    def calculate_bootstrap_current(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geometry: geometry_lib.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> BootstrapCurrent:
        pass


class BootstrapCurrentModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):

    @abc.abstractmethod
    def build_model(self) -> BootstrapCurrentModel:
        pass
