import copy
from typing import Any
import immutabledict
import pydantic
from torax._src.sources import fusion_heat_source as fusion_heat_source_lib
from torax._src.sources import gas_puff_source as gas_puff_source_lib
from torax._src.sources import generic_current_source as generic_current_source_lib
from torax._src.sources import generic_ion_el_heat_source as generic_ion_el_heat_source_lib
from torax._src.sources import generic_particle_source as generic_particle_source_lib
from torax._src.sources import pellet_source as pellet_source_lib
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_models
from torax._src.torax_pydantic import torax_pydantic


class Sources(torax_pydantic.BaseModelFrozen):
    ei_exchange: qei_source_lib.QeiSourceConfig = torax_pydantic.ValidatedDefault(
        {'mode': 'ZERO'})
    cyclotron_radiation: (None) = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    fusion: fusion_heat_source_lib.FusionHeatSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    gas_puff: gas_puff_source_lib.GasPuffSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    generic_current: generic_current_source_lib.GenericCurrentSourceConfig = (
        torax_pydantic.ValidatedDefault({'mode': 'ZERO'}))
    generic_heat: (generic_ion_el_heat_source_lib.GenericIonElHeatSourceConfig
                   | None) = pydantic.Field(
                       discriminator='model_name',
                       default=None,
                   )
    generic_particle: (generic_particle_source_lib.GenericParticleSourceConfig
                       | None) = pydantic.Field(
                           discriminator='model_name',
                           default=None,
                       )
    impurity_radiation: (None) = pydantic.Field(
        discriminator='model_name',
        default=None,
    )
    pellet: pellet_source_lib.PelletSourceConfig | None = pydantic.Field(
        discriminator='model_name',
        default=None,
    )

    @pydantic.model_validator(mode='before')
    @classmethod
    def _set_default_model_functions(cls, x: dict[str, Any]) -> dict[str, Any]:
        constructor_data = copy.deepcopy(x)
        for k, v in x.items():
            match k:
                case 'gas_puff':
                    if 'model_name' not in v:
                        constructor_data[k][
                            'model_name'] = gas_puff_source_lib.DEFAULT_MODEL_FUNCTION_NAME
                case 'generic_particle':
                    if 'model_name' not in v:
                        constructor_data[k][
                            'model_name'] = generic_particle_source_lib.DEFAULT_MODEL_FUNCTION_NAME
                case 'pellet':
                    if 'model_name' not in v:
                        constructor_data[k][
                            'model_name'] = pellet_source_lib.DEFAULT_MODEL_FUNCTION_NAME
                case 'fusion':
                    if 'model_name' not in v:
                        constructor_data[k][
                            'model_name'] = fusion_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
                case 'generic_heat':
                    if 'model_name' not in v:
                        constructor_data[k][
                            'model_name'] = generic_ion_el_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        return constructor_data

    def build_models(self) -> source_models.SourceModels:
        standard_sources = {}
        for k, v in dict(self).items():
            if k == 'ei_exchange':
                continue
            else:
                if v is not None:
                    source = v.build_source()
                    standard_sources[k] = source
        qei_source_model = self.ei_exchange.build_source()
        return source_models.SourceModels(
            qei_source=qei_source_model,
            standard_sources=immutabledict.immutabledict(standard_sources),
        )

