from collections.abc import Mapping
from typing import Final
from fusion_surrogates.qlknn import qlknn_model
import immutabledict
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src.transport_model import base_qlknn_model
from torax._src.transport_model import qualikiz_based_transport_model

_FLUX_NAME_MAP: Final[Mapping[str, str]] = immutabledict.immutabledict({
    'efiITG':
    'qi_itg',
    'efeITG':
    'qe_itg',
    'pfeITG':
    'pfe_itg',
    'efeTEM':
    'qe_tem',
    'efiTEM':
    'qi_tem',
    'pfeTEM':
    'pfe_tem',
    'efeETG':
    'qe_etg',
})


class QLKNNModelWrapper(base_qlknn_model.BaseQLKNNModel):

    def __init__(
        self,
        path: str,
        name: str = '',
        flux_name_map: Mapping[str, str] | None = None,
    ):
        if flux_name_map is None:
            flux_name_map = _FLUX_NAME_MAP
        self._flux_name_map = flux_name_map
        if path:
            self._model = qlknn_model.QLKNNModel.load_model_from_path(
                path, name)
        elif name:
            self._model = qlknn_model.QLKNNModel.load_model_from_name(name)
        else:
            self._model = qlknn_model.QLKNNModel.load_default_model()
        super().__init__(path=self._model.path, name=self._model.name)

    @property
    def inputs_and_ranges(self) -> base_qlknn_model.InputsAndRanges:
        return self._model.inputs_and_ranges

    def get_model_inputs_from_qualikiz_inputs(
        self, qualikiz_inputs: qualikiz_based_transport_model.QualikizInputs
    ) -> jax.Array:
        input_map = {
            'Ani': lambda x: x.Ani0,
            'LogNuStar': lambda x: x.log_nu_star_face,
        }

        def _get_input(key: str) -> jax.Array:
            return jnp.array(
                input_map.get(key, lambda x: getattr(x, key))(qualikiz_inputs),
                dtype=jax_utils.get_dtype(),
            )

        return jnp.array(
            [_get_input(key) for key in self.inputs_and_ranges.keys()],
            dtype=jax_utils.get_dtype(),
        ).T

    def predict(self, inputs: jax.Array) -> dict[str, jax.Array]:
        model_predictions = self._model.predict(inputs)
        return {
            self._flux_name_map.get(flux_name, flux_name): flux_value
            for flux_name, flux_value in model_predictions.items()
        }
