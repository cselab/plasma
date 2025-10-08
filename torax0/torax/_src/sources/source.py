import abc
import dataclasses
import enum
import typing
from typing import ClassVar, Protocol
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source_profiles


@typing.runtime_checkable
class SourceProfileFunction(Protocol):

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        source_name: str,
        core_profiles: state.CoreProfiles,
        calculated_source_profiles: source_profiles.SourceProfiles | None,
        unused_conductivity: conductivity_base.Conductivity | None,
    ) -> tuple[array_typing.FloatVectorCell, ...]:
        ...


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source(abc.ABC):
    SOURCE_NAME: ClassVar[str] = 'source'
    model_func: SourceProfileFunction | None = None

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
        pass

    def get_value(self, runtime_params, geo, core_profiles,
                  calculated_source_profiles, conductivity):
        source_params = runtime_params.sources[self.source_name]
        mode = source_params.mode
        match mode:
            case runtime_params_lib.Mode.MODEL_BASED:
                return self.model_func(
                    runtime_params,
                    geo,
                    self.source_name,
                    core_profiles,
                    calculated_source_profiles,
                    conductivity,
                )
            case _:
                raise ValueError(f'Unknown mode: {mode}')
