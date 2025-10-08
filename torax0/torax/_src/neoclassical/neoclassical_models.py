import dataclasses
import jax
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.transport import base as transport_base


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class NeoclassicalModels:
    conductivity: conductivity_base.ConductivityModel
    bootstrap_current: bootstrap_current_base.BootstrapCurrentModel
    transport: transport_base.NeoclassicalTransportModel

    def __hash__(self) -> int:
        return hash(
            (self.bootstrap_current, self.conductivity, self.transport))

    def __eq__(self, other) -> bool:
        return (isinstance(other, NeoclassicalModels)
                and self.conductivity == other.conductivity
                and self.bootstrap_current == other.bootstrap_current
                and self.transport == other.transport)
