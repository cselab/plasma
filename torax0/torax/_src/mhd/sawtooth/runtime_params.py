import dataclasses
import jax
from torax._src import array_typing


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TriggerRuntimeParams:
    minimum_radius: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RedistributionRuntimeParams:
    flattening_factor: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    trigger_params: TriggerRuntimeParams
    redistribution_params: RedistributionRuntimeParams
    crash_step_duration: array_typing.FloatScalar
