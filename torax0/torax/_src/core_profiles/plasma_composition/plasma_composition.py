import copy
import dataclasses
import functools
from typing import Annotated, Any
import chex
import jax
import pydantic
from torax._src import array_typing
from torax._src.config import runtime_validation_utils
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.core_profiles.plasma_composition import ion_mixture
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Final

_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'
_IMPURITY_MODE_NE_RATIOS: Final[str] = 'n_e_ratios'
_IMPURITY_MODE_NE_RATIOS_ZEFF: Final[str] = 'n_e_ratios_Z_eff'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParamsP:
    main_ion_names: tuple[str,
                          ...] = dataclasses.field(metadata={'static': True})
    impurity_names: tuple[str,
                          ...] = dataclasses.field(metadata={'static': True})
    main_ion: ion_mixture.RuntimeParams
    impurity: (ion_mixture.RuntimeParams)
    Z_eff: array_typing.FloatVectorCell
    Z_eff_face: array_typing.FloatVectorFace


@jax.tree_util.register_pytree_node_class
class PlasmaComposition(torax_pydantic.BaseModelFrozen):
    impurity: Annotated[
        impurity_fractions.ImpurityFractions,
        pydantic.Field(discriminator='impurity_mode'),
    ]
    main_ion: runtime_validation_utils.IonMapping = (
        torax_pydantic.ValidatedDefault({
            'D': 0.5,
            'T': 0.5
        }))
    Z_eff: (runtime_validation_utils.
            TimeVaryingArrayDefinedAtRightBoundaryAndBounded
            ) = torax_pydantic.ValidatedDefault(1.0)
    Z_i_override: torax_pydantic.TimeVaryingScalar | None = None
    A_i_override: torax_pydantic.TimeVaryingScalar | None = None
    Z_impurity_override: torax_pydantic.TimeVaryingScalar | None = None
    A_impurity_override: torax_pydantic.TimeVaryingScalar | None = None

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_impurity_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        configurable_data = copy.deepcopy(data)
        Z_impurity_override = configurable_data.get('Z_impurity_override')
        A_impurity_override = configurable_data.get('A_impurity_override')
        impurity_data = configurable_data['impurity']
        configurable_data['impurity'] = {
            'impurity_mode': _IMPURITY_MODE_FRACTIONS,
            'species': impurity_data,
            'Z_override': Z_impurity_override,
            'A_override': A_impurity_override,
            'legacy': True,
        }
        return configurable_data

    def tree_flatten(self):
        children = (
            self.main_ion,
            self.impurity,
            self.Z_eff,
            self.Z_i_override,
            self.A_i_override,
            self.Z_impurity_override,
            self.A_impurity_override,
            self._main_ion_mixture,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.model_construct(
            main_ion=children[0],
            impurity=children[1],
            Z_eff=children[2],
            Z_i_override=children[3],
            A_i_override=children[4],
            Z_impurity_override=children[5],
            A_impurity_override=children[6],
        )
        obj._main_ion_mixture = children[7]
        return obj

    @functools.cached_property
    def _main_ion_mixture(self):
        return ion_mixture.IonMixture.model_construct(
            species=self.main_ion,
            Z_override=self.Z_i_override,
            A_override=self.A_i_override,
        )

    def get_main_ion_names(self):
        return tuple(self._main_ion_mixture.species.keys())

    def get_impurity_names(self):
        return tuple(self.impurity.species.keys())

    def build_runtime_params(self, t):
        return RuntimeParamsP(
            main_ion_names=self.get_main_ion_names(),
            impurity_names=self.get_impurity_names(),
            main_ion=self._main_ion_mixture.build_runtime_params(t),
            impurity=self.impurity.build_runtime_params(t),
            Z_eff=self.Z_eff.get_value(t),
            Z_eff_face=self.Z_eff.get_value(t, grid_type='face'),
        )
