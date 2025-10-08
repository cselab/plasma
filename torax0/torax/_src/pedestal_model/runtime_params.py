import dataclasses
import jax
from torax._src import array_typing
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  set_pedestal: array_typing.BoolScalar
