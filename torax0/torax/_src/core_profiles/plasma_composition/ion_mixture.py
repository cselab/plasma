import dataclasses
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_validation_utils
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Final

_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    fractions: array_typing.FloatVector
    A_avg: array_typing.FloatScalar | array_typing.FloatVectorCell
    Z_override: array_typing.FloatScalar | None = None


class IonMixture(torax_pydantic.BaseModelFrozen):
    species: runtime_validation_utils.IonMapping
    Z_override: torax_pydantic.TimeVaryingScalar | None = None
    A_override: torax_pydantic.TimeVaryingScalar | None = None

    def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        Z_override = None if not self.Z_override else self.Z_override.get_value(
            t)
        if not self.A_override:
            As = jnp.array(
                [constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
            A_avg = jnp.sum(As * fractions)
        else:
            A_avg = self.A_override.get_value(t)
        return RuntimeParams(
            fractions=fractions,
            A_avg=A_avg,
            Z_override=Z_override,
        )
