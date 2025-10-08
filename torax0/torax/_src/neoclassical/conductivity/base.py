import abc
import dataclasses
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class Conductivity:
    sigma: array_typing.FloatVectorCell
    sigma_face: array_typing.FloatVectorFace


class ConductivityModel(abc.ABC):

    @abc.abstractmethod
    def calculate_conductivity(
        self,
        geometry: geometry_lib.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> Conductivity:
        pass


class ConductivityModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):

    @abc.abstractmethod
    def build_model(self) -> ConductivityModel:
        pass
