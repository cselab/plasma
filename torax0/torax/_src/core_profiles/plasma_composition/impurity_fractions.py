from collections.abc import Mapping
import dataclasses
from typing import Annotated, Any, Literal, TypeAlias
import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Final

_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'


def _impurity_before_validator(value: Any) -> Any:
    if isinstance(value, str):
        return {value: 1.0}
    return value


def _impurity_after_validator(
    value: Mapping[str, torax_pydantic.TimeVaryingArray],
) -> Mapping[str, torax_pydantic.TimeVaryingArray]:
    if not value:
        raise ValueError('The species dictionary cannot be empty.')
    first_key = next(iter(value))
    first_tva = value[first_key]
    reference_times = first_tva.value.keys()
    for species_key, tva in value.items():
        if tva.value.keys() != reference_times:
            raise ValueError(
                f'Inconsistent times for key "{species_key}". Expected'
                f' {sorted(list(reference_times))}, got'
                f' {sorted(list(tva.value.keys()))}')
    for t in reference_times:
        reference_rho_norm, _ = first_tva.value[t]
        for species_key, tva in value.items():
            current_rho_norm, _ = tva.value[t]
            if not np.array_equal(current_rho_norm, reference_rho_norm):
                raise ValueError(
                    f'Inconsistent rho_norm for key "{species_key}" at time {t}.'
                )
        values_at_t = [tva.value[t][1] for tva in value.values()]
        reference_shape = values_at_t[0].shape
        for i, v_arr in enumerate(values_at_t):
            if v_arr.shape != reference_shape:
                species_key = list(value.keys())[i]
                raise ValueError(
                    f'Inconsistent value shape for key "{species_key}" at time {t}.'
                    f' Expected {reference_shape}, got {v_arr.shape}.')
        sum_of_values = np.sum(np.stack(values_at_t, axis=0), axis=0)
        if not np.allclose(sum_of_values, 1.0):
            raise ValueError(
                f'Values do not sum to 1 at time {t}. Sum is {sum_of_values}.')
    return value


ImpurityMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.NonNegativeTimeVaryingArray],
    pydantic.BeforeValidator(_impurity_before_validator),
    pydantic.AfterValidator(_impurity_after_validator),
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    fractions: jt.Float[array_typing.Array, 'ion_symbol rhon']
    fractions_face: jt.Float[array_typing.Array, 'ion_symbol rhon+1']
    A_avg: array_typing.FloatVectorCell
    A_avg_face: array_typing.FloatVectorFace
    Z_override: array_typing.FloatScalar | None = None


class ImpurityFractions(torax_pydantic.BaseModelFrozen):
    impurity_mode: Annotated[Literal['fractions'],
                             torax_pydantic.JAX_STATIC] = ('fractions')
    species: ImpurityMapping = torax_pydantic.ValidatedDefault({'Ne': 1.0})
    Z_override: torax_pydantic.TimeVaryingScalar | None = None
    A_override: torax_pydantic.TimeVaryingScalar | None = None

    def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        fractions_face = jnp.array(
            [self.species[ion].get_value(t, grid_type='face') for ion in ions])
        Z_override = None if not self.Z_override else self.Z_override.get_value(
            t)
        if not self.A_override:
            As = jnp.array(
                [constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
            A_avg = jnp.sum(As[..., jnp.newaxis] * fractions, axis=0)
            A_avg_face = jnp.sum(As[..., jnp.newaxis] * fractions_face, axis=0)
        else:
            A_avg = jnp.full_like(fractions[0], self.A_override.get_value(t))
            A_avg_face = jnp.full_like(fractions_face[0],
                                       self.A_override.get_value(t))
        return RuntimeParams(
            fractions=fractions,
            fractions_face=fractions_face,
            A_avg=A_avg,
            A_avg_face=A_avg_face,
            Z_override=Z_override,
        )

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_impurity_data(cls, data):
        if 'legacy' in data:
            del data['legacy']
        return data
