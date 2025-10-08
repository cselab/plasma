import dataclasses
import jax
from torax._src import array_typing
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  chi_min: float
  chi_max: float
  D_e_min: float
  D_e_max: float
  V_e_min: float
  V_e_max: float
  rho_min: array_typing.FloatScalar
  rho_max: array_typing.FloatScalar
  apply_inner_patch: array_typing.BoolScalar
  D_e_inner: array_typing.FloatScalar
  V_e_inner: array_typing.FloatScalar
  chi_i_inner: array_typing.FloatScalar
  chi_e_inner: array_typing.FloatScalar
  rho_inner: array_typing.FloatScalar
  apply_outer_patch: array_typing.BoolScalar
  D_e_outer: array_typing.FloatScalar
  V_e_outer: array_typing.FloatScalar
  chi_i_outer: array_typing.FloatScalar
  chi_e_outer: array_typing.FloatScalar
  rho_outer: array_typing.FloatScalar
  smoothing_width: float
  smooth_everywhere: bool
