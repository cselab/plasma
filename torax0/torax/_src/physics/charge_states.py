import dataclasses
from typing import Final, Mapping, Sequence
import immutabledict
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
_MAVRIN_Z_COEFFS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'C': np.array([  
            [-7.2007e00, -1.2217e01, -7.3521e00, -1.7632e00, 5.8588e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 6.0000e00],
        ]),
        'N': np.array([  
            [0.0000e00, 3.3818e00, 1.8861e00, 1.5668e-01, 6.9728e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 7.0000e00],
        ]),
        'O': np.array([  
            [0.0000e00, -1.8560e01, -3.8664e01, -2.2093e01, 4.0451e00],
            [-4.3092e00, -4.6261e-01, -3.7050e-02, 8.0180e-02, 7.9878e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 8.0000e00],
        ]),
        'Ne': np.array([  
            [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
            [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
        ]),
        'Ar': np.array([  
            [6.8717e00, -1.1595e01, -4.3776e01, -2.0781e01, 1.3171e01],
            [-4.8830e-02, 1.8455e00, 2.5023e00, 1.1413e00, 1.5986e01],
            [-5.9213e-01, 3.5667e00, -8.0048e00, 7.9986e00, 1.4948e01],
        ]),
        'Kr': np.array([  
            [1.3630e02, 4.6320e02, 5.6890e02, 3.0638e02, 7.7040e01],
            [-1.0279e02, 6.8446e01, 1.5744e01, 1.5186e00, 2.4728e01],
            [-2.4682e00, 1.3215e01, -2.5703e01, 2.3443e01, 2.5368e01],
        ]),
        'Xe': np.array([  
            [5.8178e02, 1.9967e03, 2.5189e03, 1.3973e03, 3.0532e02],
            [8.6824e01, -2.9061e01, -4.8384e01, 1.6271e01, 3.2616e01],
            [4.0756e02, -9.0008e02, 6.6739e02, -1.7259e02, 4.8066e01],
            [-1.0019e01, 7.3261e01, -1.9931e02, 2.4056e02, -5.7527e01],
        ]),
        'W': np.array([  
            [1.6823e01, 3.4582e01, 2.1027e01, 1.6518e01, 2.6703e01],
            [-2.5887e02, -1.0577e01, 2.5532e02, -7.9611e01, 3.6902e01],
            [1.5119e01, -8.4207e01, 1.5985e02, -1.0011e02, 6.3795e01],
        ]),
    })
)
_TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'C': np.array([0.7]),
        'N': np.array([0.7]),
        'O': np.array([0.3, 1.5]),
        'Ne': np.array([0.5, 2.0]),
        'Ar': np.array([0.6, 3.0]),
        'Kr': np.array([0.447, 4.117]),
        'Xe': np.array([0.3, 1.5, 8.0]),
        'W': np.array([1.5, 4.0]),
    })
)
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ChargeStateInfo:
  Z_avg: array_typing.FloatVector
  Z2_avg: array_typing.FloatVector
  Z_per_species: array_typing.FloatVector
  @property
  def Z_mixture(self) -> array_typing.FloatVector:
    return self.Z2_avg / self.Z_avg
def calculate_average_charge_state_single_species(
    T_e: array_typing.FloatVector,
    ion_symbol: str,
) -> array_typing.FloatVector:
  if T_e.ndim == 0:
    raise ValueError(
        'T_e must be a 1D array, but is a scalar. Please provide a 1D array.'
    )
  if ion_symbol not in constants.ION_SYMBOLS:
    raise ValueError(
        f'Invalid ion symbol: {ion_symbol}. Allowed symbols are :'
        f' {constants.ION_SYMBOLS}'
    )
  if ion_symbol not in _MAVRIN_Z_COEFFS:
    return jnp.ones_like(T_e) * constants.ION_PROPERTIES_DICT[ion_symbol].Z
  T_e_allowed_range = (0.1, 100.0)
  T_e = jnp.clip(T_e, *T_e_allowed_range)
  interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], T_e)
  Zavg_coeffs_in_range = jnp.take(
      _MAVRIN_Z_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()
  X = jnp.log10(T_e)
  Zavg = jnp.polyval(Zavg_coeffs_in_range, X)
  return Zavg
def get_average_charge_state(
    ion_symbols: Sequence[str],
    T_e: array_typing.FloatVector,
    fractions: array_typing.FloatVector,
    Z_override: array_typing.FloatScalar | None = None,
) -> ChargeStateInfo:
  if T_e.ndim == 0:
    raise ValueError(
        'T_e must be a 1D array, but is a scalar. Please provide a 1D array.'
    )
  if Z_override is not None:
    override_val = jnp.ones_like(T_e) * Z_override
    return ChargeStateInfo(
        Z_avg=override_val,
        Z2_avg=override_val**2,
        Z_per_species=jnp.stack([override_val for _ in ion_symbols]),
    )
  Z_per_species = jnp.stack([
      calculate_average_charge_state_single_species(T_e, ion_symbol)
      for ion_symbol in ion_symbols
  ])
  fractions = fractions if fractions.ndim == 2 else fractions[:, jnp.newaxis]
  Z_avg = jnp.sum(fractions * Z_per_species, axis=0)
  Z2_avg = jnp.sum(fractions * Z_per_species**2, axis=0)
  return ChargeStateInfo(
      Z_avg=Z_avg,
      Z2_avg=Z2_avg,
      Z_per_species=Z_per_species,
  )
