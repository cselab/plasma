import dataclasses
from typing import Annotated, ClassVar, Literal
import chex
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import formulas
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

DEFAULT_MODEL_FUNCTION_NAME: str = 'exponential'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    puff_decay_length: array_typing.FloatScalar
    S_total: array_typing.FloatScalar


def calc_puff_source(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
    source_params = runtime_params.sources[source_name]
    assert isinstance(source_params, RuntimeParams)
    return (formulas.exponential_profile(
        decay_start=1.0,
        width=source_params.puff_decay_length,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(source.Source):
    SOURCE_NAME: ClassVar[str] = 'gas_puff'
    model_func: source.SourceProfileFunction = calc_puff_source

    @property
    def source_name(self) -> str:
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
        return (source.AffectedCoreProfile.NE, )


class GasPuffSourceConfig(base.SourceModelBase):
    model_name: Annotated[Literal['exponential'],
                          torax_pydantic.JAX_STATIC] = ('exponential')
    puff_decay_length: torax_pydantic.TimeVaryingScalar = (
        torax_pydantic.ValidatedDefault(0.05))
    S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
        1e22)
    mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
        runtime_params_lib.Mode.MODEL_BASED)

    @property
    def model_func(self) -> source.SourceProfileFunction:
        return calc_puff_source

    def build_runtime_params(
        self,
        t: chex.Numeric,
    ) -> RuntimeParams:
        return RuntimeParams(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            puff_decay_length=self.puff_decay_length.get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self) -> GasPuffSource:
        return GasPuffSource(model_func=self.model_func)
