import dataclasses
import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    chi_min: float
    chi_max: float
    D_e_min: float
    D_e_max: float
    V_e_min: float
    V_e_max: float
