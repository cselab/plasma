import dataclasses
import enum
import jax
from torax._src import array_typing
from torax._src import interpolated_param

TimeInterpolatedInput = interpolated_param.TimeInterpolatedInput


@enum.unique
class Mode(enum.Enum):
    ZERO = "ZERO"
    MODEL_BASED = "MODEL_BASED"
    PRESCRIBED = "PRESCRIBED"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    prescribed_values: tuple[array_typing.FloatVector, ...]
    mode: Mode = dataclasses.field(metadata={"static": True})
    is_explicit: bool = dataclasses.field(metadata={"static": True})
