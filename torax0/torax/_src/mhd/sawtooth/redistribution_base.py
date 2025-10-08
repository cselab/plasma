import abc
import chex
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.torax_pydantic import torax_pydantic


class RedistributionModel(abc.ABC):

    @abc.abstractmethod
    def __call__(
        self,
        rho_norm_q1: array_typing.FloatScalar,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
    ) -> state.CoreProfiles:
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass


class RedistributionConfig(torax_pydantic.BaseModelFrozen):
    flattening_factor: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(1.01))

    def build_runtime_params(
        self, t: chex.Numeric
    ) -> sawtooth_runtime_params.RedistributionRuntimeParams:
        return sawtooth_runtime_params.RedistributionRuntimeParams(
            flattening_factor=self.flattening_factor.get_value(t), )
