import dataclasses
import jax
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_current_runtime_params
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    bootstrap_current: bootstrap_current_runtime_params.RuntimeParams
    conductivity: conductivity_runtime_params.RuntimeParams
    transport: transport_runtime_params.RuntimeParams
