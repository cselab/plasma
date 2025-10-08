import abc
import chex
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.torax_pydantic import torax_pydantic


class TriggerModel(abc.ABC):

    @abc.abstractmethod
    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class TriggerConfig(torax_pydantic.BaseModelFrozen):
    minimum_radius: torax_pydantic.PositiveTimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.05))

    def build_runtime_params(
            self,
            t: chex.Numeric) -> sawtooth_runtime_params.TriggerRuntimeParams:
        return sawtooth_runtime_params.TriggerRuntimeParams(
            minimum_radius=self.minimum_radius.get_value(t), )
