import dataclasses
import jax
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    sawtooth: sawtooth_runtime_params.RuntimeParams | None = None
