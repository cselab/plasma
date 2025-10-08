from absl import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from fusion_surrogates.qlknn import qlknn_model
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src import xnp
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms
from torax._src.fvm import enums
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_lib
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical import runtime_params
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.physics import charge_states
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.physics import scaling_laws
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib
from typing import Annotated, Any, Literal
from typing import Annotated, Any, Literal, TypeAlias, TypeVar
from typing import Any
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Callable
from typing import Final
from typing import Final, Mapping, Tuple
from typing import Mapping
from typing import TypeAlias
from typing_extensions import Annotated
import abc
import chex
import copy
import dataclasses
import enum
import functools
import immutabledict
import inspect
import itertools
import jax
import jax.numpy as jnp
import logging
import numpy as np
import pydantic
import typing_extensions
import xarray as xr

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


class QLKNNModelWrapper:

    def __init__(
        self,
        path: str,
        name: str = '',
        flux_name_map: Mapping[str, str] | None = None,
    ):
        self.path = path
        self.name = name
        if flux_name_map is None:
            flux_name_map = _FLUX_NAME_MAP
        self._flux_name_map = flux_name_map
        self._model = qlknn_model.QLKNNModel.load_default_model()

    @property
    def inputs_and_ranges(self):
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams0(qualikiz_based_transport_model.RuntimeParams):
    include_ITG: bool
    include_TEM: bool
    include_ETG: bool
    ITG_flux_ratio_correction: float
    ETG_correction_factor: float
    clip_inputs: bool
    clip_margin: float


_EPSILON_NN: Final[float] = (1 / 3)


@functools.lru_cache(maxsize=1)
def get_model(path: str, name: str):
    return QLKNNModelWrapper(path, name)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QLKNNRuntimeConfigInputs:
    transport: RuntimeParams0
    Ped_top: float
    set_pedestal: bool

    @staticmethod
    def from_runtime_params_slice(
        transport_runtime_params: runtime_params_lib.RuntimeParams,
        runtime_params: runtime_params_slice.RuntimeParams,
        pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
    ) -> 'QLKNNRuntimeConfigInputs':
        assert isinstance(transport_runtime_params, RuntimeParams0)
        return QLKNNRuntimeConfigInputs(
            transport=transport_runtime_params,
            Ped_top=pedestal_model_output.rho_norm_ped_top,
            set_pedestal=runtime_params.pedestal.set_pedestal,
        )


def _filter_model_output(
    model_output: None,
    include_ITG: bool,
    include_TEM: bool,
    include_ETG: bool,
):
    filter_map = {
        'qi_itg': include_ITG,
        'qe_itg': include_ITG,
        'pfe_itg': include_ITG,
        'qe_tem': include_TEM,
        'qi_tem': include_TEM,
        'pfe_tem': include_TEM,
        'qe_etg': include_ETG,
    }

    def filter_flux(flux_name: str, value: jax.Array) -> jax.Array:
        return jax.lax.cond(
            filter_map.get(flux_name, True),
            lambda: value,
            lambda: jnp.zeros_like(value),
        )

    return {k: filter_flux(k, v) for k, v in model_output.items()}


def clip_inputs(feature_scan, clip_margin, inputs_and_ranges):
    for i, key in enumerate(inputs_and_ranges.keys()):
        bounds = inputs_and_ranges[key]
        min_val = bounds.get('min', -jnp.inf)
        max_val = bounds.get('max', jnp.inf)
        min_val += jnp.where(jnp.isfinite(min_val),
                             jnp.abs(min_val) * (1 - clip_margin), 0.0)
        max_val -= jnp.where(jnp.isfinite(max_val),
                             jnp.abs(max_val) * (1 - clip_margin), 0.0)
        feature_scan = feature_scan.at[:, i].set(
            jnp.clip(
                feature_scan[:, i],
                min_val,
                max_val,
            ))
    return feature_scan


class QLKNNTransportModel0(
        qualikiz_based_transport_model.QualikizBasedTransportModel):

    def __init__(
        self,
        path: str,
        name: str,
    ):
        super().__init__()
        self._path = path
        self._name = name
        self._frozen = True

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._name

    def _call_implementation(
        self,
        transport_runtime_params: RuntimeParams0,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
        pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
    ) -> transport_model_lib.TurbulentTransport:
        runtime_config_inputs = QLKNNRuntimeConfigInputs.from_runtime_params_slice(
            transport_runtime_params,
            runtime_params,
            pedestal_model_output,
        )
        return self._combined(runtime_config_inputs, geo, core_profiles)

    def _combined(
        self,
        runtime_config_inputs: QLKNNRuntimeConfigInputs,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> transport_model_lib.TurbulentTransport:
        qualikiz_inputs = self._prepare_qualikiz_inputs(
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
        )
        model = get_model(self.path, self.name)
        qualikiz_inputs = dataclasses.replace(
            qualikiz_inputs,
            x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
        )
        feature_scan = model.get_model_inputs_from_qualikiz_inputs(
            qualikiz_inputs)
        feature_scan = jax.lax.cond(
            runtime_config_inputs.transport.clip_inputs,
            lambda: clip_inputs(
                feature_scan,
                runtime_config_inputs.transport.clip_margin,
                model.inputs_and_ranges,
            ),
            lambda: feature_scan,
        )
        model_output = model.predict(feature_scan)
        model_output = _filter_model_output(
            model_output=model_output,
            include_ITG=runtime_config_inputs.transport.include_ITG,
            include_TEM=runtime_config_inputs.transport.include_TEM,
            include_ETG=runtime_config_inputs.transport.include_ETG,
        )
        qi_itg_squeezed = model_output['qi_itg'].squeeze()
        qi = qi_itg_squeezed + model_output['qi_tem'].squeeze()
        qe = (model_output['qe_itg'].squeeze() *
              runtime_config_inputs.transport.ITG_flux_ratio_correction +
              model_output['qe_tem'].squeeze() +
              model_output['qe_etg'].squeeze() *
              runtime_config_inputs.transport.ETG_correction_factor)
        pfe = model_output['pfe_itg'].squeeze(
        ) + model_output['pfe_tem'].squeeze()
        return self._make_core_transport(
            qi=qi,
            qe=qe,
            pfe=pfe,
            quasilinear_inputs=qualikiz_inputs,
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
            gradient_reference_length=geo.R_major,
            gyrobohm_flux_reference_length=geo.a_minor,
        )

    def __hash__(self) -> int:
        return hash(('QLKNNTransportModel' + self.path + self.name))

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return (isinstance(other, QLKNNTransportModel)
                and self.path == other.path and self.name == other.name)


class QLKNNTransportModel(pydantic_model_base.TransportBase):
    model_name: Annotated[Literal['qlknn'],
                          torax_pydantic.JAX_STATIC] = 'qlknn'
    model_path: Annotated[str, torax_pydantic.JAX_STATIC] = ''
    qlknn_model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
    include_ITG: bool = True
    include_TEM: bool = True
    include_ETG: bool = True
    ITG_flux_ratio_correction: float = 1.0
    ETG_correction_factor: float = 1.0 / 3.0
    clip_inputs: bool = False
    clip_margin: float = 0.95
    collisionality_multiplier: float = 1.0
    avoid_big_negative_s: bool = True
    smag_alpha_correction: bool = True
    q_sawtooth_proxy: bool = True
    DV_effective: bool = False
    An_min: pydantic.PositiveFloat = 0.05

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
        data = copy.deepcopy(data)
        data['qlknn_model_name'] = data.get('qlknn_model_name', '')
        if 'smoothing_width' not in data:
            data['smoothing_width'] = 0.1
        return data

    def build_transport_model(self):
        return QLKNNTransportModel0(path=self.model_path,
                                    name=self.qlknn_model_name)

    def build_runtime_params(self, t: chex.Numeric):
        base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
        return RuntimeParams0(
            include_ITG=self.include_ITG,
            include_TEM=self.include_TEM,
            include_ETG=self.include_ETG,
            ITG_flux_ratio_correction=self.ITG_flux_ratio_correction,
            ETG_correction_factor=self.ETG_correction_factor,
            clip_inputs=self.clip_inputs,
            clip_margin=self.clip_margin,
            collisionality_multiplier=self.collisionality_multiplier,
            avoid_big_negative_s=self.avoid_big_negative_s,
            smag_alpha_correction=self.smag_alpha_correction,
            q_sawtooth_proxy=self.q_sawtooth_proxy,
            DV_effective=self.DV_effective,
            An_min=self.An_min,
            **base_kwargs,
        )


@functools.partial(jax_utils.jit, static_argnums=(0, 1, 2))
def calculate_total_transport_coeffs(
    pedestal_model: pedestal_model_lib.PedestalModel,
    transport_model: transport_model_lib.TransportModel,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> state.CoreTransport:
    pedestal_model_output = pedestal_model(runtime_params, geo, core_profiles)
    turbulent_transport = transport_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        pedestal_model_output=pedestal_model_output,
    )
    neoclassical_transport_coeffs = neoclassical_models.transport(
        runtime_params,
        geo,
        core_profiles,
    )
    return state.CoreTransport(
        **dataclasses.asdict(turbulent_transport),
        **dataclasses.asdict(neoclassical_transport_coeffs),
    )


class FileRestart(torax_pydantic.BaseModelFrozen):
    filename: pydantic.FilePath
    time: torax_pydantic.Second
    do_restart: bool
    stitch: bool


class Neoclassical0(torax_pydantic.BaseModelFrozen):
    bootstrap_current: (bootstrap_current_zeros.ZerosModelConfig
                        | sauter_current.SauterModelConfig) = pydantic.Field(
                            discriminator="model_name")
    conductivity: sauter_conductivity.SauterModelConfig = (
        torax_pydantic.ValidatedDefault(
            sauter_conductivity.SauterModelConfig()))
    transport: (transport_zeros.ZerosModelConfig) = pydantic.Field(
        discriminator="model_name")

    @pydantic.model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        configurable_data = copy.deepcopy(data)
        if "bootstrap_current" not in configurable_data:
            configurable_data["bootstrap_current"] = {"model_name": "zeros"}
        if "transport" not in configurable_data:
            configurable_data["transport"] = {"model_name": "zeros"}
        if "model_name" not in configurable_data["bootstrap_current"]:
            configurable_data["bootstrap_current"]["model_name"] = "sauter"
        return configurable_data

    def build_runtime_params(self) -> runtime_params.RuntimeParams:
        return runtime_params.RuntimeParams(
            bootstrap_current=self.bootstrap_current.build_runtime_params(),
            conductivity=self.conductivity.build_runtime_params(),
            transport=self.transport.build_runtime_params(),
        )

    def build_models(self) -> neoclassical_models.NeoclassicalModels:
        return neoclassical_models.NeoclassicalModels(
            conductivity=self.conductivity.build_model(),
            bootstrap_current=self.bootstrap_current.build_model(),
            transport=self.transport.build_model(),
        )


T = TypeVar('T')
LY_OBJECT_TYPE: TypeAlias = (str
                             | Mapping[str, torax_pydantic.NumpyArray | float])
TIME_INVARIANT = torax_pydantic.TIME_INVARIANT


class CheaseConfig(torax_pydantic.BaseModelFrozen):
    geometry_type: Annotated[Literal['chease'], TIME_INVARIANT] = 'chease'
    n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
    hires_factor: pydantic.PositiveInt = 4
    geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
    Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
    geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols'
    R_major: torax_pydantic.Meter = 6.2
    a_minor: torax_pydantic.Meter = 2.0
    B_0: torax_pydantic.Tesla = 5.3

    @pydantic.model_validator(mode='after')
    def _check_fields(self):
        return self

    def build_geometry(self) -> standard_geometry.StandardGeometry:
        return standard_geometry.build_standard_geometry(
            _apply_relevant_kwargs(
                standard_geometry.StandardGeometryIntermediates.from_chease,
                self.__dict__,
            ))


class GeometryConfig(torax_pydantic.BaseModelFrozen):
    config: (CheaseConfig) = pydantic.Field(discriminator='geometry_type')


class Geometry0(torax_pydantic.BaseModelFrozen):
    geometry_type: geometry.GeometryType
    geometry_configs: GeometryConfig | dict[torax_pydantic.Second,
                                            GeometryConfig]

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(cls, data):
        geometry_type = data['geometry_type']
        return _conform_user_data(data)

    @functools.cached_property
    def build_provider(self):
        geometries = self.geometry_configs.config.build_geometry()
        provider = geometry_provider.ConstantGeometryProvider
        return provider(geometries)


def _conform_user_data(data: dict[str, Any]) -> dict[str, Any]:
    data_copy = data.copy()
    data_copy['geometry_type'] = data['geometry_type'].lower()
    geometry_type = getattr(geometry.GeometryType,
                            data['geometry_type'].upper())
    constructor_args = {'geometry_type': geometry_type}
    configs_time_dependent = data_copy.pop('geometry_configs', None)
    constructor_args['geometry_configs'] = {'config': data_copy}
    return constructor_args


def _apply_relevant_kwargs(f: Callable[..., T], kwargs: Mapping[str,
                                                                Any]) -> T:
    relevant_kwargs = [
        i.name for i in inspect.signature(f).parameters.values()
    ]
    kwargs = {k: kwargs[k] for k in relevant_kwargs}
    return f(**kwargs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SafetyFactorFit:
    rho_q_min: array_typing.FloatScalar
    q_min: array_typing.FloatScalar
    rho_q_3_2_first: array_typing.FloatScalar
    rho_q_2_1_first: array_typing.FloatScalar
    rho_q_3_1_first: array_typing.FloatScalar
    rho_q_3_2_second: array_typing.FloatScalar
    rho_q_2_1_second: array_typing.FloatScalar
    rho_q_3_1_second: array_typing.FloatScalar


def _sliding_window_of_three(flat_array: jax.Array) -> jax.Array:
    window_size = 3
    starts = jnp.arange(len(flat_array) - window_size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(
        flat_array, (start, ), (window_size, )))(starts)


def _fit_polynomial_to_intervals_of_three(
        rho_norm: jax.Array,
        q_face: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    q_face_intervals = _sliding_window_of_three(q_face, )
    rho_norm_intervals = _sliding_window_of_three(rho_norm, )

    @jax.vmap
    def batch_polyfit(q_face_interval: jax.Array,
                      rho_norm_interval: jax.Array) -> jax.Array:
        chex.assert_shape(q_face_interval, (3, ))
        chex.assert_shape(rho_norm_interval, (3, ))
        rho_norm_squared = rho_norm_interval**2
        A = jnp.array([
            [rho_norm_squared[0], rho_norm_interval[0], 1],
            [rho_norm_squared[1], rho_norm_interval[1], 1],
            [rho_norm_squared[2], rho_norm_interval[2], 1],
        ])
        b = jnp.array(
            [q_face_interval[0], q_face_interval[1], q_face_interval[2]])
        coeffs = jnp.linalg.solve(A, b)
        return coeffs

    return (
        batch_polyfit(q_face_intervals, rho_norm_intervals),
        rho_norm_intervals,
        q_face_intervals,
    )


@jax.vmap
def _minimum_location_value_in_interval(
        coeffs: jax.Array, rho_norm_interval: jax.Array,
        q_interval: jax.Array) -> tuple[jax.Array, jax.Array]:
    min_interval, max_interval = rho_norm_interval[0], rho_norm_interval[1]
    q_min_interval, q_max_interval = (
        q_interval[0],
        q_interval[1],
    )
    a, b = coeffs[0], coeffs[1]
    extremum_location = -b / (2 * a)
    extremum_in_interval = jnp.greater(extremum_location,
                                       min_interval) & jnp.less(
                                           extremum_location, max_interval)
    extremum_value = jax.lax.cond(
        extremum_in_interval,
        lambda x: jnp.polyval(coeffs, x),
        lambda x: jnp.inf,
        extremum_location,
    )
    interval_minimum_location, interval_minimum_value = jax.lax.cond(
        jnp.less(q_min_interval, q_max_interval),
        lambda: (min_interval, q_min_interval),
        lambda: (max_interval, q_max_interval),
    )
    overall_minimum_location, overall_minimum_value = jax.lax.cond(
        jnp.less(interval_minimum_value, extremum_value),
        lambda: (interval_minimum_location, interval_minimum_value),
        lambda: (extremum_location, extremum_value),
    )
    return overall_minimum_location, overall_minimum_value


def _find_roots_quadratic(coeffs: jax.Array) -> jax.Array:
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    determinant = b**2 - 4.0 * a * c
    roots_exist = jnp.greater(determinant, 0)
    plus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b + jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    minus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b - jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    return jnp.array([plus_root, minus_root])


@functools.partial(jax.vmap, in_axes=(0, 0, None))
def _root_in_interval(coeffs: jax.Array, interval: jax.Array,
                      q_surface: float) -> jax.Array:
    intercept_coeffs = coeffs - jnp.array([0.0, 0.0, q_surface])
    min_interval, max_interval = interval[0], interval[1]
    root_values = _find_roots_quadratic(intercept_coeffs)
    in_interval = jnp.greater(root_values, min_interval) & jnp.less(
        root_values, max_interval)
    return jnp.where(in_interval, root_values, -jnp.inf)


@jax_utils.jit
def find_min_q_and_q_surface_intercepts(rho_norm, q_face):
    sorted_indices = jnp.argsort(rho_norm)
    rho_norm = rho_norm[sorted_indices]
    q_face = q_face[sorted_indices]
    poly_coeffs, rho_norm_3, q_face_3 = _fit_polynomial_to_intervals_of_three(
        rho_norm, q_face)
    first_rho_norm = jnp.expand_dims(jnp.array([rho_norm[0], rho_norm[2]]),
                                     axis=0)
    first_q_face = jnp.expand_dims(jnp.array([q_face[0], q_face[2]]), axis=0)
    rho_norms = jnp.concat([first_rho_norm, rho_norm_3[1:, 1:]], axis=0)
    q_faces = jnp.concat([first_q_face, q_face_3[1:, 1:]], axis=0)
    rho_q_min_intervals, q_min_intervals = _minimum_location_value_in_interval(
        poly_coeffs, rho_norms, q_faces)
    arg_q_min = jnp.argmin(q_min_intervals)
    rho_q_min = rho_q_min_intervals[arg_q_min]
    q_min = q_min_intervals[arg_q_min]
    rho_q_3_2 = _root_in_interval(poly_coeffs, rho_norms, 1.5).flatten()
    outermost_rho_q_3_2 = rho_q_3_2[jnp.argsort(rho_q_3_2)[-2:]]
    rho_q_2_1 = _root_in_interval(poly_coeffs, rho_norms, 2.0).flatten()
    outermost_rho_q_2_1 = rho_q_2_1[jnp.argsort(rho_q_2_1)[-2:]]
    rho_q_3_1 = _root_in_interval(poly_coeffs, rho_norms, 3.0).flatten()
    outermost_rho_q_3_1 = rho_q_3_1[jnp.argsort(rho_q_3_1)[-2:]]
    return SafetyFactorFit(
        rho_q_min=rho_q_min,
        q_min=q_min,
        rho_q_3_2_first=outermost_rho_q_3_2[0],
        rho_q_2_1_first=outermost_rho_q_2_1[0],
        rho_q_3_1_first=outermost_rho_q_3_1[0],
        rho_q_3_2_second=outermost_rho_q_3_2[1],
        rho_q_2_1_second=outermost_rho_q_2_1[1],
        rho_q_3_1_second=outermost_rho_q_3_1[1],
    )


RADIATION_OUTPUT_NAME = "radiation_impurity_species"
DENSITY_OUTPUT_NAME = "n_impurity_species"
Z_OUTPUT_NAME = "Z_impurity_species"
IMPURITY_DIM = "impurity_symbol"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpuritySpeciesOutput:
    radiation: array_typing.FloatVectorCell
    n_impurity: array_typing.FloatVectorCell
    Z_impurity: array_typing.FloatVectorCell


def calculate_impurity_species_output(sim_state, runtime_params):
    impurity_species_output = {}
    mavrin_active = False
    impurity_fractions = sim_state.core_profiles.impurity_fractions
    impurity_names = runtime_params.plasma_composition.impurity_names
    charge_state_info = charge_states.get_average_charge_state(
        ion_symbols=impurity_names,
        T_e=sim_state.core_profiles.T_e.value,
        fractions=jnp.stack(
            [impurity_fractions[symbol] for symbol in impurity_names]),
        Z_override=runtime_params.plasma_composition.impurity.Z_override,
    )
    for i, symbol in enumerate(impurity_names):
        core_profiles = sim_state.core_profiles
        impurity_density_scaling = (core_profiles.Z_impurity /
                                    charge_state_info.Z_avg)
        n_imp = (impurity_fractions[symbol] * core_profiles.n_impurity.value *
                 impurity_density_scaling)
        Z_imp = charge_state_info.Z_per_species[i]
        radiation = jnp.zeros_like(n_imp)
        impurity_species_output[symbol] = ImpuritySpeciesOutput(
            radiation=radiation, n_impurity=n_imp, Z_impurity=Z_imp)
    return impurity_species_output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
    pressure_thermal_i: cell_variable.CellVariable
    pressure_thermal_e: cell_variable.CellVariable
    pressure_thermal_total: cell_variable.CellVariable
    pprime: array_typing.FloatVector
    W_thermal_i: array_typing.FloatScalar
    W_thermal_e: array_typing.FloatScalar
    W_thermal_total: array_typing.FloatScalar
    tau_E: array_typing.FloatScalar
    H89P: array_typing.FloatScalar
    H98: array_typing.FloatScalar
    H97L: array_typing.FloatScalar
    H20: array_typing.FloatScalar
    FFprime: array_typing.FloatVector
    psi_norm: array_typing.FloatVector
    P_SOL_i: array_typing.FloatScalar
    P_SOL_e: array_typing.FloatScalar
    P_SOL_total: array_typing.FloatScalar
    P_aux_i: array_typing.FloatScalar
    P_aux_e: array_typing.FloatScalar
    P_aux_total: array_typing.FloatScalar
    P_external_injected: array_typing.FloatScalar
    P_external_total: array_typing.FloatScalar
    P_ei_exchange_i: array_typing.FloatScalar
    P_ei_exchange_e: array_typing.FloatScalar
    P_aux_generic_i: array_typing.FloatScalar
    P_aux_generic_e: array_typing.FloatScalar
    P_aux_generic_total: array_typing.FloatScalar
    P_alpha_i: array_typing.FloatScalar
    P_alpha_e: array_typing.FloatScalar
    P_alpha_total: array_typing.FloatScalar
    P_ohmic_e: array_typing.FloatScalar
    P_bremsstrahlung_e: array_typing.FloatScalar
    P_cyclotron_e: array_typing.FloatScalar
    P_ecrh_e: array_typing.FloatScalar
    P_radiation_e: array_typing.FloatScalar
    I_ecrh: array_typing.FloatScalar
    I_aux_generic: array_typing.FloatScalar
    P_fusion: array_typing.FloatScalar
    Q_fusion: array_typing.FloatScalar
    P_icrh_e: array_typing.FloatScalar
    P_icrh_i: array_typing.FloatScalar
    P_icrh_total: array_typing.FloatScalar
    P_LH_high_density: array_typing.FloatScalar
    P_LH_min: array_typing.FloatScalar
    P_LH: array_typing.FloatScalar
    n_e_min_P_LH: array_typing.FloatScalar
    E_fusion: array_typing.FloatScalar
    E_aux_total: array_typing.FloatScalar
    E_ohmic_e: array_typing.FloatScalar
    E_external_injected: array_typing.FloatScalar
    E_external_total: array_typing.FloatScalar
    T_e_volume_avg: array_typing.FloatScalar
    T_i_volume_avg: array_typing.FloatScalar
    n_e_volume_avg: array_typing.FloatScalar
    n_i_volume_avg: array_typing.FloatScalar
    n_e_line_avg: array_typing.FloatScalar
    n_i_line_avg: array_typing.FloatScalar
    fgw_n_e_volume_avg: array_typing.FloatScalar
    fgw_n_e_line_avg: array_typing.FloatScalar
    q95: array_typing.FloatScalar
    W_pol: array_typing.FloatScalar
    li3: array_typing.FloatScalar
    dW_thermal_dt: array_typing.FloatScalar
    rho_q_min: array_typing.FloatScalar
    q_min: array_typing.FloatScalar
    rho_q_3_2_first: array_typing.FloatScalar
    rho_q_3_2_second: array_typing.FloatScalar
    rho_q_2_1_first: array_typing.FloatScalar
    rho_q_2_1_second: array_typing.FloatScalar
    rho_q_3_1_first: array_typing.FloatScalar
    rho_q_3_1_second: array_typing.FloatScalar
    I_bootstrap: array_typing.FloatScalar
    j_external: array_typing.FloatVector
    j_ohmic: array_typing.FloatVector
    S_gas_puff: array_typing.FloatScalar
    S_pellet: array_typing.FloatScalar
    S_generic_particle: array_typing.FloatScalar
    beta_tor: array_typing.FloatScalar
    beta_pol: array_typing.FloatScalar
    beta_N: array_typing.FloatScalar
    S_total: array_typing.FloatScalar
    impurity_species: dict[str, ImpuritySpeciesOutput]

    def check_for_errors(self):
        return state.SimError.NO_ERROR


ION_EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'generic_heat': 'P_aux_generic',
    'fusion': 'P_alpha',
    'icrh': 'P_icrh',
}
EL_HEAT_SOURCE_TRANSFORMATIONS = {
    'ohmic': 'P_ohmic_e',
    'bremsstrahlung': 'P_bremsstrahlung_e',
    'cyclotron_radiation': 'P_cyclotron_e',
    'ecrh': 'P_ecrh_e',
    'impurity_radiation': 'P_radiation_e',
}
EXTERNAL_HEATING_SOURCES = [
    'generic_heat',
    'ecrh',
    'icrh',
]
CURRENT_SOURCE_TRANSFORMATIONS = {
    'generic_current': 'I_aux_generic',
    'ecrh': 'I_ecrh',
}
PARTICLE_SOURCE_TRANSFORMATIONS = {
    'gas_puff': 'S_gas_puff',
    'pellet': 'S_pellet',
    'generic_particle': 'S_generic_particle',
}


def _get_integrated_source_value(
    source_profiles_dict: dict[str, array_typing.FloatVector],
    internal_source_name: str,
    geo: geometry.Geometry,
    integration_fn: Callable[[array_typing.FloatVector, geometry.Geometry],
                             jax.Array],
) -> jax.Array:
    if internal_source_name in source_profiles_dict:
        return integration_fn(source_profiles_dict[internal_source_name], geo)
    else:
        return jnp.array(0.0, dtype=jax_utils.get_dtype())


def _calculate_integrated_sources(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles.SourceProfiles,
    runtime_params: runtime_params_slice.RuntimeParams,
) -> dict[str, jax.Array]:
    integrated = {}
    integrated['P_alpha_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['S_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    qei = core_sources.qei.qei_coef * (core_profiles.T_e.value -
                                       core_profiles.T_i.value)
    integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
    integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']
    integrated['P_SOL_i'] = integrated['P_ei_exchange_i']
    integrated['P_SOL_e'] = integrated['P_ei_exchange_e']
    integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_external_injected'] = jnp.array(0.0,
                                                  dtype=jax_utils.get_dtype())
    for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        is_in_T_i = key in core_sources.T_i
        is_in_T_e = key in core_sources.T_e
        integrated[f'{value}_i'] = _get_integrated_source_value(
            core_sources.T_i, key, geo, math_utils.volume_integration)
        integrated[f'{value}_e'] = _get_integrated_source_value(
            core_sources.T_e, key, geo, math_utils.volume_integration)
        integrated[f'{value}_total'] = (integrated[f'{value}_i'] +
                                        integrated[f'{value}_e'])
        integrated['P_SOL_i'] += integrated[f'{value}_i']
        integrated['P_SOL_e'] += integrated[f'{value}_e']
        if key in EXTERNAL_HEATING_SOURCES:
            integrated['P_aux_i'] += integrated[f'{value}_i']
            integrated['P_aux_e'] += integrated[f'{value}_e']
            source_params = runtime_params.sources.get(key)
            if source_params is not None and hasattr(source_params,
                                                     'absorption_fraction'):
                total_absorbed = integrated[f'{value}_total']
                injected_power = total_absorbed / source_params.absorption_fraction
                integrated['P_external_injected'] += injected_power
            else:
                integrated['P_external_injected'] += integrated[
                    f'{value}_total']
    for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.T_e, key, geo, math_utils.volume_integration)
        integrated['P_SOL_e'] += integrated[f'{value}']
        if key in EXTERNAL_HEATING_SOURCES:
            integrated['P_aux_e'] += integrated[f'{value}']
            integrated['P_external_injected'] += integrated[f'{value}']
    for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.psi, key, geo, math_utils.area_integration)
    for key, value in PARTICLE_SOURCE_TRANSFORMATIONS.items():
        integrated[f'{value}'] = _get_integrated_source_value(
            core_sources.n_e, key, geo, math_utils.volume_integration)
        integrated['S_total'] += integrated[f'{value}']
    integrated['P_SOL_total'] = integrated['P_SOL_i'] + integrated['P_SOL_e']
    integrated['P_aux_total'] = integrated['P_aux_i'] + integrated['P_aux_e']
    integrated['P_fusion'] = 5 * integrated['P_alpha_total']
    integrated['P_external_total'] = (integrated['P_external_injected'] +
                                      integrated['P_ohmic_e'])
    return integrated


@jax_utils.jit
def make_post_processed_outputs(
    sim_state,
    runtime_params: runtime_params_slice.RuntimeParams,
    previous_post_processed_outputs: PostProcessedOutputs | None = None,
) -> PostProcessedOutputs:
    impurity_radiation_outputs = (calculate_impurity_species_output(
        sim_state, runtime_params))
    (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    ) = formulas.calculate_pressure(sim_state.core_profiles)
    pprime_face = formulas.calc_pprime(sim_state.core_profiles)
    W_thermal_el, W_thermal_ion, W_thermal_tot = (
        formulas.calculate_stored_thermal_energy(
            pressure_thermal_el,
            pressure_thermal_ion,
            pressure_thermal_tot,
            sim_state.geometry,
        ))
    FFprime_face = formulas.calc_FFprime(sim_state.core_profiles,
                                         sim_state.geometry)
    psi_face = sim_state.core_profiles.psi.face_value()
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
    integrated_sources = _calculate_integrated_sources(
        sim_state.geometry,
        sim_state.core_profiles,
        sim_state.core_sources,
        runtime_params,
    )
    Q_fusion = (
        integrated_sources['P_fusion'] /
        (integrated_sources['P_external_total'] + constants.CONSTANTS.eps))
    P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
        scaling_laws.calculate_plh_scaling_factor(sim_state.geometry,
                                                  sim_state.core_profiles))
    Ploss = (integrated_sources['P_alpha_total'] +
             integrated_sources['P_aux_total'] +
             integrated_sources['P_ohmic_e'] + constants.CONSTANTS.eps)
    if previous_post_processed_outputs is not None:
        dW_th_dt = (
            W_thermal_tot -
            previous_post_processed_outputs.W_thermal_total) / sim_state.dt
    else:
        dW_th_dt = 0.0
    tauE = W_thermal_tot / Ploss
    tauH89P = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H89P')
    tauH98 = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H98')
    tauH97L = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H97L')
    tauH20 = scaling_laws.calculate_scaling_law_confinement_time(
        sim_state.geometry, sim_state.core_profiles, Ploss, 'H20')
    H89P = tauE / tauH89P
    H98 = tauE / tauH98
    H97L = tauE / tauH97L
    H20 = tauE / tauH20
    if previous_post_processed_outputs is not None:
        E_fusion = (previous_post_processed_outputs.E_fusion + sim_state.dt *
                    (integrated_sources['P_fusion'] +
                     previous_post_processed_outputs.P_fusion) / 2.0)
        E_aux_total = (previous_post_processed_outputs.E_aux_total +
                       sim_state.dt *
                       (integrated_sources['P_aux_total'] +
                        previous_post_processed_outputs.P_aux_total) / 2.0)
        E_ohmic_e = (previous_post_processed_outputs.E_ohmic_e + sim_state.dt *
                     (integrated_sources['P_ohmic_e'] +
                      previous_post_processed_outputs.P_ohmic_e) / 2.0)
        E_external_injected = (
            previous_post_processed_outputs.E_external_injected +
            sim_state.dt *
            (integrated_sources['P_external_injected'] +
             previous_post_processed_outputs.P_external_injected) / 2.0)
        E_external_total = (
            previous_post_processed_outputs.E_external_total + sim_state.dt *
            (integrated_sources['P_external_total'] +
             previous_post_processed_outputs.P_external_total) / 2.0)
    else:
        E_fusion = 0.0
        E_aux_total = 0.0
        E_ohmic_e = 0.0
        E_external_injected = 0.0
        E_external_total = 0.0
    q95 = psi_calculations.calc_q95(psi_norm_face,
                                    sim_state.core_profiles.q_face)
    te_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_e.value, sim_state.geometry)
    ti_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_i.value, sim_state.geometry)
    n_e_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.n_e.value, sim_state.geometry)
    n_i_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.n_i.value, sim_state.geometry)
    n_e_line_avg = math_utils.line_average(sim_state.core_profiles.n_e.value,
                                           sim_state.geometry)
    n_i_line_avg = math_utils.line_average(sim_state.core_profiles.n_i.value,
                                           sim_state.geometry)
    fgw_n_e_volume_avg = formulas.calculate_greenwald_fraction(
        n_e_volume_avg, sim_state.core_profiles, sim_state.geometry)
    fgw_n_e_line_avg = formulas.calculate_greenwald_fraction(
        n_e_line_avg, sim_state.core_profiles, sim_state.geometry)
    Wpol = psi_calculations.calc_Wpol(sim_state.geometry,
                                      sim_state.core_profiles.psi)
    li3 = psi_calculations.calc_li3(
        sim_state.geometry.R_major,
        Wpol,
        sim_state.core_profiles.Ip_profile_face[-1],
    )
    safety_factor_fit_outputs = (find_min_q_and_q_surface_intercepts(
        sim_state.geometry.rho_face_norm,
        sim_state.core_profiles.q_face,
    ))
    I_bootstrap = math_utils.area_integration(
        sim_state.core_sources.bootstrap_current.j_bootstrap,
        sim_state.geometry)
    j_external = sum(sim_state.core_sources.psi.values())
    psi_current = (j_external +
                   sim_state.core_sources.bootstrap_current.j_bootstrap)
    j_ohmic = sim_state.core_profiles.j_total - psi_current
    beta_tor, beta_pol, beta_N = formulas.calculate_betas(
        sim_state.core_profiles, sim_state.geometry)
    return PostProcessedOutputs(
        pressure_thermal_i=pressure_thermal_ion,
        pressure_thermal_e=pressure_thermal_el,
        pressure_thermal_total=pressure_thermal_tot,
        pprime=pprime_face,
        W_thermal_i=W_thermal_ion,
        W_thermal_e=W_thermal_el,
        W_thermal_total=W_thermal_tot,
        tau_E=tauE,
        H89P=H89P,
        H98=H98,
        H97L=H97L,
        H20=H20,
        FFprime=FFprime_face,
        psi_norm=psi_norm_face,
        **integrated_sources,
        Q_fusion=Q_fusion,
        P_LH=P_LH,
        P_LH_min=P_LH_min,
        P_LH_high_density=P_LH_hi_dens,
        n_e_min_P_LH=n_e_min_P_LH,
        E_fusion=E_fusion,
        E_aux_total=E_aux_total,
        E_ohmic_e=E_ohmic_e,
        E_external_injected=E_external_injected,
        E_external_total=E_external_total,
        T_e_volume_avg=te_volume_avg,
        T_i_volume_avg=ti_volume_avg,
        n_e_volume_avg=n_e_volume_avg,
        n_i_volume_avg=n_i_volume_avg,
        n_e_line_avg=n_e_line_avg,
        n_i_line_avg=n_i_line_avg,
        fgw_n_e_volume_avg=fgw_n_e_volume_avg,
        fgw_n_e_line_avg=fgw_n_e_line_avg,
        q95=q95,
        W_pol=Wpol,
        li3=li3,
        dW_thermal_dt=dW_th_dt,
        rho_q_min=safety_factor_fit_outputs.rho_q_min,
        q_min=safety_factor_fit_outputs.q_min,
        rho_q_3_2_first=safety_factor_fit_outputs.rho_q_3_2_first,
        rho_q_2_1_first=safety_factor_fit_outputs.rho_q_2_1_first,
        rho_q_3_1_first=safety_factor_fit_outputs.rho_q_3_1_first,
        rho_q_3_2_second=safety_factor_fit_outputs.rho_q_3_2_second,
        rho_q_2_1_second=safety_factor_fit_outputs.rho_q_2_1_second,
        rho_q_3_1_second=safety_factor_fit_outputs.rho_q_3_1_second,
        I_bootstrap=I_bootstrap,
        j_external=j_external,
        j_ohmic=j_ohmic,
        beta_tor=beta_tor,
        beta_pol=beta_pol,
        beta_N=beta_N,
        impurity_species=impurity_radiation_outputs,
    )


def construct_xarray_for_radiation_output(
    impurity_radiation_outputs: dict[str, ImpuritySpeciesOutput],
    times: jax.Array,
    rho_cell_norm: jax.Array,
    time_coord: str,
    rho_cell_norm_coord: str,
) -> dict[str, xr.DataArray]:
    radiation_data = []
    n_impurity_data = []
    Z_impurity_data = []
    impurity_symbols = []
    xr_dict = {}
    for impurity_symbol in impurity_radiation_outputs:
        radiation_data.append(
            impurity_radiation_outputs[impurity_symbol].radiation)
        n_impurity_data.append(
            impurity_radiation_outputs[impurity_symbol].n_impurity)
        Z_impurity_data.append(
            impurity_radiation_outputs[impurity_symbol].Z_impurity)
        impurity_symbols.append(impurity_symbol)
    radiation_data = np.stack(radiation_data, axis=0)
    n_impurity_data = np.stack(n_impurity_data, axis=0)
    Z_impurity_data = np.stack(Z_impurity_data, axis=0)
    xr_dict[RADIATION_OUTPUT_NAME] = xr.DataArray(
        radiation_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=RADIATION_OUTPUT_NAME,
    )
    xr_dict[DENSITY_OUTPUT_NAME] = xr.DataArray(
        n_impurity_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=DENSITY_OUTPUT_NAME,
    )
    xr_dict[Z_OUTPUT_NAME] = xr.DataArray(
        Z_impurity_data,
        dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
        coords={
            IMPURITY_DIM: impurity_symbols,
            time_coord: times,
            rho_cell_norm_coord: rho_cell_norm,
        },
        name=Z_OUTPUT_NAME,
    )
    return xr_dict


SCALING_FACTORS: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'T_i':
    1.0,
    'T_e':
    1.0,
    'n_e':
    1e20,
    'psi':
    1.0,
})
_trapz = jax.scipy.integrate.trapezoid


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
    n_i: cell_variable.CellVariable
    n_impurity: cell_variable.CellVariable
    impurity_fractions: Mapping[str, array_typing.FloatVectorCell]
    Z_i: array_typing.FloatVectorCell
    Z_i_face: array_typing.FloatVectorFace
    Z_impurity: array_typing.FloatVectorCell
    Z_impurity_face: array_typing.FloatVectorFace
    A_i: array_typing.FloatScalar
    A_impurity: array_typing.FloatVectorCell
    A_impurity_face: array_typing.FloatVectorFace
    Z_eff: array_typing.FloatVectorCell
    Z_eff_face: array_typing.FloatVectorFace


def get_updated_ion_temperature(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
    T_i = cell_variable.CellVariable(
        value=profile_conditions_params.T_i,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=profile_conditions_params.T_i_right_bc,
        dr=geo.drho_norm,
    )
    return T_i


def get_updated_electron_temperature(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
    T_e = cell_variable.CellVariable(
        value=profile_conditions_params.T_e,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=profile_conditions_params.T_e_right_bc,
        dr=geo.drho_norm,
    )
    return T_e


def get_updated_electron_density(
    profile_conditions_params: profile_conditions.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
    nGW = (profile_conditions_params.Ip / 1e6 / (jnp.pi * geo.a_minor**2) *
           1e20)
    n_e_value = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.n_e * nGW,
        profile_conditions_params.n_e,
    )
    n_e_right_bc = jnp.where(
        profile_conditions_params.n_e_right_bc_is_fGW,
        profile_conditions_params.n_e_right_bc * nGW,
        profile_conditions_params.n_e_right_bc,
    )
    face_left = n_e_value[0]
    face_right = n_e_right_bc
    face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]], )
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    target_nbar = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.nbar * nGW,
        profile_conditions_params.nbar,
    )
    nbar_from_n_e_face_inner = (_trapz(n_e_face[:-1], geo.R_out_face[:-1]) /
                                a_minor_out)
    dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]
    C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
        nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out)
    n_e_value = C * n_e_value
    n_e = cell_variable.CellVariable(
        value=n_e_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e_right_bc,
    )
    return n_e


@dataclasses.dataclass(frozen=True)
class _IonProperties:
    A_impurity: array_typing.FloatVectorCell
    A_impurity_face: array_typing.FloatVectorFace
    Z_impurity: array_typing.FloatVectorCell
    Z_impurity_face: array_typing.FloatVectorFace
    Z_eff: array_typing.FloatVectorCell
    dilution_factor: array_typing.FloatVectorCell
    dilution_factor_edge: array_typing.FloatScalar
    impurity_fractions: array_typing.FloatVector


def _get_ion_properties_from_fractions(
    impurity_symbols: tuple[str, ...],
    impurity_params: impurity_fractions.RuntimeParams,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
    Z_eff_from_config: array_typing.FloatVectorCell,
    Z_eff_face_from_config: array_typing.FloatVectorFace,
) -> _IonProperties:
    Z_impurity = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.value,
        fractions=impurity_params.fractions,
        Z_override=impurity_params.Z_override,
    ).Z_mixture
    Z_impurity_face = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
        Z_override=impurity_params.Z_override,
    ).Z_mixture
    Z_eff = Z_eff_from_config
    Z_eff_edge = Z_eff_face_from_config[-1]
    dilution_factor = jnp.where(
        Z_eff == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
    )
    dilution_factor_edge = jnp.where(
        Z_eff_edge == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(Z_i_face[-1],
                                                    Z_impurity_face[-1],
                                                    Z_eff_edge),
    )
    return _IonProperties(
        A_impurity=impurity_params.A_avg,
        A_impurity_face=impurity_params.A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_edge,
        impurity_fractions=impurity_params.fractions,
    )


@jax_utils.jit
def get_updated_ions(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> Ions:
    Z_i = charge_states.get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.value,
        fractions=runtime_params.plasma_composition.main_ion.fractions,
        Z_override=runtime_params.plasma_composition.main_ion.Z_override,
    ).Z_mixture
    Z_i_face = charge_states.get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.face_value(),
        fractions=runtime_params.plasma_composition.main_ion.fractions,
        Z_override=runtime_params.plasma_composition.main_ion.Z_override,
    ).Z_mixture
    impurity_params = runtime_params.plasma_composition.impurity
    match impurity_params:
        case impurity_fractions.RuntimeParams():
            ion_properties = _get_ion_properties_from_fractions(
                runtime_params.plasma_composition.impurity_names,
                impurity_params,
                T_e,
                Z_i,
                Z_i_face,
                runtime_params.plasma_composition.Z_eff,
                runtime_params.plasma_composition.Z_eff_face,
            )
        case _:
            raise ValueError("Unknown impurity mode.")
    n_i = cell_variable.CellVariable(
        value=n_e.value * ion_properties.dilution_factor,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e.right_face_constraint *
        ion_properties.dilution_factor_edge,
    )
    n_impurity_value = jnp.where(
        ion_properties.dilution_factor == 1.0,
        0.0,
        (n_e.value - n_i.value * Z_i) / ion_properties.Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        ion_properties.dilution_factor_edge == 1.0,
        0.0,
        (n_e.right_face_constraint - n_i.right_face_constraint * Z_i_face[-1])
        / ion_properties.Z_impurity_face[-1],
    )
    n_impurity = cell_variable.CellVariable(
        value=n_impurity_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_impurity_right_face_constraint,
    )
    Z_eff_face = _calculate_Z_eff(
        Z_i_face,
        ion_properties.Z_impurity_face,
        n_i.face_value(),
        n_impurity.face_value(),
        n_e.face_value(),
    )
    impurity_fractions_dict = {}
    for i, symbol in enumerate(
            runtime_params.plasma_composition.impurity_names):
        fraction = ion_properties.impurity_fractions[i]
        impurity_fractions_dict[symbol] = fraction
    return Ions(
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=impurity_fractions_dict,
        Z_i=Z_i,
        Z_i_face=Z_i_face,
        Z_impurity=ion_properties.Z_impurity,
        Z_impurity_face=ion_properties.Z_impurity_face,
        A_i=runtime_params.plasma_composition.main_ion.A_avg,
        A_impurity=ion_properties.A_impurity,
        A_impurity_face=ion_properties.A_impurity_face,
        Z_eff=ion_properties.Z_eff,
        Z_eff_face=Z_eff_face,
    )


def _calculate_Z_eff(Z_i, Z_impurity, n_i, n_impurity, n_e):
    return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e


def initial_core_profiles0(runtime_params, geo, source_models,
                           neoclassical_models):
    T_i = get_updated_ion_temperature(runtime_params.profile_conditions, geo)
    T_e = get_updated_electron_temperature(runtime_params.profile_conditions,
                                           geo)
    n_e = get_updated_electron_density(runtime_params.profile_conditions, geo)
    ions = get_updated_ions(runtime_params, geo, n_e, T_e)
    v_loop_lcfs = (
        np.array(runtime_params.profile_conditions.v_loop_lcfs)
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else np.array(0.0, dtype=jax_utils.get_dtype()))
    psidot = cell_variable.CellVariable(
        value=np.zeros_like(geo.rho),
        dr=geo.drho_norm,
    )
    psi = cell_variable.CellVariable(value=np.zeros_like(geo.rho),
                                     dr=geo.drho_norm)
    core_profiles = state.CoreProfiles(
        T_i=T_i,
        T_e=T_e,
        n_e=n_e,
        n_i=ions.n_i,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        A_i=ions.A_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        psi=psi,
        psidot=psidot,
        q_face=np.zeros_like(geo.rho_face),
        s_face=np.zeros_like(geo.rho_face),
        v_loop_lcfs=v_loop_lcfs,
        sigma=np.zeros_like(geo.rho),
        sigma_face=np.zeros_like(geo.rho_face),
        j_total=np.zeros_like(geo.rho),
        j_total_face=np.zeros_like(geo.rho_face),
        Ip_profile_face=np.zeros_like(geo.rho_face),
    )
    return _init_psi_and_psi_derived(
        runtime_params,
        geo,
        core_profiles,
        source_models,
        neoclassical_models,
    )


def _get_initial_psi_mode(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
) -> profile_conditions_lib.InitialPsiMode:
    psi_mode = runtime_params.profile_conditions.initial_psi_mode
    if psi_mode == profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS:
        if runtime_params.profile_conditions.psi is None:
            logging.warning(
                'Falling back to legacy behavior as `profile_conditions.psi` is '
                'None. Future versions of TORAX will require `psi` to be provided '
                'if `initial_psi_mode` is PROFILE_CONDITIONS. Use '
                '`initial_psi_mode` to initialize psi from `j` or `geometry` and '
                'avoid this warning.')
            if (isinstance(geo, standard_geometry.StandardGeometry) and
                    not runtime_params.profile_conditions.initial_psi_from_j):
                psi_mode = profile_conditions_lib.InitialPsiMode.GEOMETRY
            else:
                psi_mode = profile_conditions_lib.InitialPsiMode.J
    return psi_mode


def _init_psi_and_psi_derived(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
    sources_are_calculated = False
    source_profiles = source_profile_builders.build_all_zero_profiles(geo)
    initial_psi_mode = _get_initial_psi_mode(runtime_params, geo)
    match initial_psi_mode:
        case profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS:
            if runtime_params.profile_conditions.psi is None:
                raise ValueError(
                    'psi is None, but initial_psi_mode is PROFILE_CONDITIONS.')
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))
            if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
                right_face_grad_constraint = None
                right_face_constraint = (
                    runtime_params.profile_conditions.psi[-1] +
                    dpsi_drhonorm_edge * geo.drho_norm / 2)
            else:
                right_face_grad_constraint = dpsi_drhonorm_edge
                right_face_constraint = None
            psi = cell_variable.CellVariable(
                value=runtime_params.profile_conditions.psi,
                right_face_grad_constraint=right_face_grad_constraint,
                right_face_constraint=right_face_constraint,
                dr=geo.drho_norm,
            )
        case profile_conditions_lib.InitialPsiMode.GEOMETRY:
            if not isinstance(geo, standard_geometry.StandardGeometry):
                raise ValueError(
                    'GEOMETRY initial_psi_source is only supported for standard'
                    ' geometry.')
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))
            psi = cell_variable.CellVariable(
                value=geo.psi_from_Ip,
                right_face_grad_constraint=None
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else dpsi_drhonorm_edge,
                right_face_constraint=geo.psi_from_Ip_face[-1]
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None,
                dr=geo.drho_norm,
            )
        case profile_conditions_lib.InitialPsiMode.J:
            j_total_hires = _get_j_total_hires_with_no_external_sources(
                runtime_params, geo)
            psi = update_psi_from_j(
                runtime_params.profile_conditions.Ip,
                geo,
                j_total_hires,
                use_v_loop_lcfs_boundary_condition=runtime_params.
                profile_conditions.use_v_loop_lcfs_boundary_condition,
            )
            if not (runtime_params.profile_conditions.
                    initial_j_is_total_current):
                core_profiles_initial = dataclasses.replace(
                    core_profiles,
                    psi=psi,
                    q_face=psi_calculations.calc_q_face(geo, psi),
                    s_face=psi_calculations.calc_s_face(geo, psi),
                )
                psi, source_profiles = _iterate_psi_and_sources(
                    runtime_params=runtime_params,
                    geo=geo,
                    core_profiles=core_profiles_initial,
                    neoclassical_models=neoclassical_models,
                    source_models=source_models,
                    source_profiles=source_profiles,
                    iterations=2,
                )
                sources_are_calculated = True
    core_profiles = _calculate_all_psi_dependent_profiles(
        runtime_params=runtime_params,
        geo=geo,
        psi=psi,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        sources_are_calculated=sources_are_calculated,
    )
    return core_profiles


def _calculate_all_psi_dependent_profiles(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
    core_profiles: state.CoreProfiles,
    source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    sources_are_calculated: bool,
) -> state.CoreProfiles:
    j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
        geo, psi)
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
        j_total=j_total,
        j_total_face=j_total_face,
        Ip_profile_face=Ip_profile_face,
    )
    conductivity = neoclassical_models.conductivity.calculate_conductivity(
        geo,
        core_profiles,
    )
    if not sources_are_calculated:
        source_profiles = _get_bootstrap_and_standard_source_profiles(
            runtime_params,
            geo,
            core_profiles,
            neoclassical_models,
            source_models,
            source_profiles,
        )
    if (not runtime_params.numerics.evolve_current
            and runtime_params.profile_conditions.psidot is not None):
        psidot_value = runtime_params.profile_conditions.psidot
    else:
        psi_sources = source_profiles.total_psi_sources(geo)
        psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
            psi_sources=psi_sources,
            sigma=conductivity.sigma,
            resistivity_multiplier=runtime_params.numerics.
            resistivity_multiplier,
            psi=psi,
            geo=geo,
        )
    v_loop_lcfs = (
        runtime_params.profile_conditions.v_loop_lcfs
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else psidot_value[-1])
    psidot = dataclasses.replace(
        core_profiles.psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psidot=psidot,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    return core_profiles


def _get_bootstrap_and_standard_source_profiles(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    source_models: source_models_lib.SourceModels,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> source_profiles_lib.SourceProfiles:
    source_profile_builders.build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    bootstrap_current = (
        neoclassical_models.bootstrap_current.calculate_bootstrap_current(
            runtime_params, geo, core_profiles))
    source_profiles = dataclasses.replace(source_profiles,
                                          bootstrap_current=bootstrap_current)
    return source_profiles


def core_profiles_to_solver_x_tuple(
    core_profiles: state.CoreProfiles,
    evolving_names: Tuple[str, ...],
) -> Tuple[cell_variable.CellVariable, ...]:
    x_tuple_for_solver_list = []
    for name in evolving_names:
        original_units_cv = getattr(core_profiles, name)
        solver_x_tuple_cv = scale_cell_variable(
            cv=original_units_cv,
            scaling_factor=1 / SCALING_FACTORS[name],
        )
        x_tuple_for_solver_list.append(solver_x_tuple_cv)
    return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(
    x_new: tuple[cell_variable.CellVariable, ...],
    evolving_names: tuple[str, ...],
    core_profiles: state.CoreProfiles,
) -> state.CoreProfiles:
    updated_vars = {}
    for i, var_name in enumerate(evolving_names):
        solver_x_tuple_cv = x_new[i]
        original_units_cv = scale_cell_variable(
            cv=solver_x_tuple_cv,
            scaling_factor=SCALING_FACTORS[var_name],
        )
        updated_vars[var_name] = original_units_cv
    return dataclasses.replace(core_profiles, **updated_vars)


def scale_cell_variable(
    cv: cell_variable.CellVariable,
    scaling_factor: float,
) -> cell_variable.CellVariable:
    operation = lambda x, factor: x * factor if x is not None else None
    scaled_value = operation(cv.value, scaling_factor)
    scaled_left_face_constraint = operation(cv.left_face_constraint,
                                            scaling_factor)
    scaled_left_face_grad_constraint = operation(cv.left_face_grad_constraint,
                                                 scaling_factor)
    scaled_right_face_constraint = operation(cv.right_face_constraint,
                                             scaling_factor)
    scaled_right_face_grad_constraint = operation(
        cv.right_face_grad_constraint, scaling_factor)
    return cell_variable.CellVariable(
        value=scaled_value,
        left_face_constraint=scaled_left_face_constraint,
        left_face_grad_constraint=scaled_left_face_grad_constraint,
        right_face_constraint=scaled_right_face_constraint,
        right_face_grad_constraint=scaled_right_face_grad_constraint,
        dr=cv.dr,
    )


OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...],
                                       ...] | None
AuxiliaryOutput: TypeAlias = Any


def _calculate_psi_value_constraint_from_v_loop(
    dt: array_typing.FloatScalar,
    theta: array_typing.FloatScalar,
    v_loop_lcfs_t: array_typing.FloatScalar,
    v_loop_lcfs_t_plus_dt: array_typing.FloatScalar,
    psi_lcfs_t: array_typing.FloatScalar,
) -> jax.Array:
    theta_weighted_v_loop_lcfs = (
        1 - theta) * v_loop_lcfs_t + theta * v_loop_lcfs_t_plus_dt
    return psi_lcfs_t + theta_weighted_v_loop_lcfs * dt


@jax_utils.jit
def get_prescribed_core_profile_values(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> dict[str, array_typing.FloatVector]:
    if not runtime_params.numerics.evolve_ion_heat:
        T_i = get_updated_ion_temperature(runtime_params.profile_conditions,
                                          geo).value
    else:
        T_i = core_profiles.T_i.value
    if not runtime_params.numerics.evolve_electron_heat:
        T_e_cell_variable = get_updated_electron_temperature(
            runtime_params.profile_conditions, geo)
        T_e = T_e_cell_variable.value
    else:
        T_e_cell_variable = core_profiles.T_e
        T_e = T_e_cell_variable.value
    if not runtime_params.numerics.evolve_density:
        n_e_cell_variable = get_updated_electron_density(
            runtime_params.profile_conditions, geo)
    else:
        n_e_cell_variable = core_profiles.n_e
    ions = get_updated_ions(
        runtime_params,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )
    n_e = n_e_cell_variable.value
    n_i = ions.n_i.value
    n_impurity = ions.n_impurity.value
    impurity_fractions = ions.impurity_fractions
    return {
        'T_i': T_i,
        'T_e': T_e,
        'n_e': n_e,
        'n_i': n_i,
        'n_impurity': n_impurity,
        'impurity_fractions': impurity_fractions,
        'Z_i': ions.Z_i,
        'Z_i_face': ions.Z_i_face,
        'Z_impurity': ions.Z_impurity,
        'Z_impurity_face': ions.Z_impurity_face,
        'A_i': ions.A_i,
        'A_impurity': ions.A_impurity,
        'A_impurity_face': ions.A_impurity_face,
        'Z_eff': ions.Z_eff,
        'Z_eff_face': ions.Z_eff_face,
    }


@functools.partial(jax_utils.jit, static_argnames=['evolving_names'])
def update_core_profiles_during_step(
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
    updated_core_profiles = solver_x_tuple_to_core_profiles(
        x_new, evolving_names, core_profiles)
    ions = get_updated_ions(
        runtime_params,
        geo,
        updated_core_profiles.n_e,
        updated_core_profiles.T_e,
    )
    return dataclasses.replace(
        updated_core_profiles,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        q_face=psi_calculations.calc_q_face(geo, updated_core_profiles.psi),
        s_face=psi_calculations.calc_s_face(geo, updated_core_profiles.psi),
    )


def update_core_and_source_profiles_after_step(
    dt: array_typing.FloatScalar,
    x_new: tuple[cell_variable.CellVariable, ...],
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
    evolving_names: tuple[str, ...],
) -> tuple[state.CoreProfiles, source_profiles_lib.SourceProfiles]:
    updated_core_profiles_t_plus_dt = solver_x_tuple_to_core_profiles(
        x_new, evolving_names, core_profiles_t_plus_dt)
    ions = get_updated_ions(
        runtime_params_t_plus_dt,
        geo,
        updated_core_profiles_t_plus_dt.n_e,
        updated_core_profiles_t_plus_dt.T_e,
    )
    v_loop_lcfs = (runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs
                   if runtime_params_t_plus_dt.profile_conditions.
                   use_v_loop_lcfs_boundary_condition else
                   _update_v_loop_lcfs_from_psi(
                       core_profiles_t.psi,
                       updated_core_profiles_t_plus_dt.psi,
                       dt,
                   ))
    j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
        geo,
        updated_core_profiles_t_plus_dt.psi,
    )
    intermediate_core_profiles = state.CoreProfiles(
        T_i=updated_core_profiles_t_plus_dt.T_i,
        T_e=updated_core_profiles_t_plus_dt.T_e,
        psi=updated_core_profiles_t_plus_dt.psi,
        n_e=updated_core_profiles_t_plus_dt.n_e,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        psidot=core_profiles_t_plus_dt.psidot,
        q_face=psi_calculations.calc_q_face(
            geo, updated_core_profiles_t_plus_dt.psi),
        s_face=psi_calculations.calc_s_face(
            geo, updated_core_profiles_t_plus_dt.psi),
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        v_loop_lcfs=v_loop_lcfs,
        sigma=core_profiles_t_plus_dt.sigma,
        sigma_face=core_profiles_t_plus_dt.sigma_face,
        j_total=j_total,
        j_total_face=j_total_face,
        Ip_profile_face=Ip_profile_face,
    )
    conductivity = neoclassical_models.conductivity.calculate_conductivity(
        geo, intermediate_core_profiles)
    intermediate_core_profiles = dataclasses.replace(
        intermediate_core_profiles,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    total_source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        core_profiles=intermediate_core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    if (not runtime_params_t_plus_dt.numerics.evolve_current and
            runtime_params_t_plus_dt.profile_conditions.psidot is not None):
        psidot_value = (runtime_params_t_plus_dt.profile_conditions.psidot)
    else:
        psi_sources = total_source_profiles.total_psi_sources(geo)
        psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
            psi_sources=psi_sources,
            sigma=intermediate_core_profiles.sigma,
            resistivity_multiplier=runtime_params_t_plus_dt.numerics.
            resistivity_multiplier,
            psi=intermediate_core_profiles.psi,
            geo=geo,
        )
    psidot = dataclasses.replace(
        core_profiles_t_plus_dt.psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        intermediate_core_profiles,
        psidot=psidot,
    )
    return core_profiles_t_plus_dt, total_source_profiles


def compute_boundary_conditions_for_t_plus_dt(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> dict[str, dict[str, jax.Array | None]]:
    profile_conditions_t_plus_dt = (
        runtime_params_t_plus_dt.profile_conditions)
    n_e = get_updated_electron_density(profile_conditions_t_plus_dt,
                                       geo_t_plus_dt)
    n_e_right_bc = n_e.right_face_constraint
    ions_edge = get_updated_ions(
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        dataclasses.replace(
            core_profiles_t.n_e,
            right_face_constraint=profile_conditions_t_plus_dt.n_e_right_bc,
        ),
        dataclasses.replace(
            core_profiles_t.T_e,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
    )
    Z_i_edge = ions_edge.Z_i_face[-1]
    Z_impurity_edge = ions_edge.Z_impurity_face[-1]
    dilution_factor_edge = formulas.calculate_main_ion_dilution_factor(
        Z_i_edge,
        Z_impurity_edge,
        runtime_params_t_plus_dt.plasma_composition.Z_eff_face[-1],
    )
    n_i_bound_right = n_e_right_bc * dilution_factor_edge
    n_impurity_bound_right = (n_e_right_bc -
                              n_i_bound_right * Z_i_edge) / Z_impurity_edge
    return {
        'T_i':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_i_right_bc,
        ),
        'T_e':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
        'n_e':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_e_right_bc),
        ),
        'n_i':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_i_bound_right),
        ),
        'n_impurity':
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_impurity_bound_right),
        ),
        'psi':
        dict(
            right_face_grad_constraint=(
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    Ip=profile_conditions_t_plus_dt.Ip,
                    geo=geo_t_plus_dt,
                ) if not runtime_params_t.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None),
            right_face_constraint=(_calculate_psi_value_constraint_from_v_loop(
                dt=dt,
                v_loop_lcfs_t=runtime_params_t.profile_conditions.v_loop_lcfs,
                v_loop_lcfs_t_plus_dt=profile_conditions_t_plus_dt.v_loop_lcfs,
                psi_lcfs_t=core_profiles_t.psi.right_face_constraint,
                theta=runtime_params_t.solver.theta_implicit,
            ) if runtime_params_t.profile_conditions.
                                   use_v_loop_lcfs_boundary_condition else
                                   None),
        ),
        'Z_i_edge':
        Z_i_edge,
        'Z_impurity_edge':
        Z_impurity_edge,
    }


def provide_core_profiles_t_plus_dt(
    dt: jax.Array,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
    updated_boundary_conditions = compute_boundary_conditions_for_t_plus_dt(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )
    updated_values = get_prescribed_core_profile_values(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo_t_plus_dt,
        core_profiles=core_profiles_t,
    )
    T_i = dataclasses.replace(
        core_profiles_t.T_i,
        value=updated_values['T_i'],
        **updated_boundary_conditions['T_i'],
    )
    T_e = dataclasses.replace(
        core_profiles_t.T_e,
        value=updated_values['T_e'],
        **updated_boundary_conditions['T_e'],
    )
    psi = dataclasses.replace(core_profiles_t.psi,
                              **updated_boundary_conditions['psi'])
    n_e = dataclasses.replace(
        core_profiles_t.n_e,
        value=updated_values['n_e'],
        **updated_boundary_conditions['n_e'],
    )
    n_i = dataclasses.replace(
        core_profiles_t.n_i,
        value=updated_values['n_i'],
        **updated_boundary_conditions['n_i'],
    )
    n_impurity = dataclasses.replace(
        core_profiles_t.n_impurity,
        value=updated_values['n_impurity'],
        **updated_boundary_conditions['n_impurity'],
    )
    Z_i_face = jnp.concatenate([
        updated_values['Z_i_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_i_edge']]),
    ], )
    Z_impurity_face = jnp.concatenate([
        updated_values['Z_impurity_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_impurity_edge']]),
    ], )
    core_profiles_t_plus_dt = dataclasses.replace(
        core_profiles_t,
        T_i=T_i,
        T_e=T_e,
        psi=psi,
        n_e=n_e,
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=updated_values['impurity_fractions'],
        Z_i=updated_values['Z_i'],
        Z_i_face=Z_i_face,
        Z_impurity=updated_values['Z_impurity'],
        Z_impurity_face=Z_impurity_face,
        A_i=updated_values['A_i'],
        A_impurity=updated_values['A_impurity'],
        A_impurity_face=updated_values['A_impurity_face'],
        Z_eff=updated_values['Z_eff'],
        Z_eff_face=updated_values['Z_eff_face'],
    )
    return core_profiles_t_plus_dt


def _update_v_loop_lcfs_from_psi(
    psi_t: cell_variable.CellVariable,
    psi_t_plus_dt: cell_variable.CellVariable,
    dt: array_typing.FloatScalar,
) -> jax.Array:
    psi_lcfs_t = psi_t.face_value()[-1]
    psi_lcfs_t_plus_dt = psi_t_plus_dt.face_value()[-1]
    v_loop_lcfs_t_plus_dt = (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt
    return v_loop_lcfs_t_plus_dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Block1DCoeffs:
    transient_in_cell: tuple[jax.Array, ...]
    transient_out_cell: tuple[jax.Array, ...] | None = None
    d_face: tuple[jax.Array, ...] | None = None
    v_face: tuple[jax.Array, ...] | None = None
    source_mat_cell: OptionalTupleMatrix = None
    source_cell: tuple[jax.Array | None, ...] | None = None
    auxiliary_outputs: AuxiliaryOutput | None = None


Block1DCoeffs: TypeAlias = Block1DCoeffs
AuxiliaryOutput: TypeAlias = AuxiliaryOutput


def cell_variable_tuple_to_vec(
    x_tuple: tuple[cell_variable.CellVariable, ...], ) -> jax.Array:
    x_vec = jnp.concatenate([x.value for x in x_tuple])
    return x_vec


class CoeffsCallback:

    def __init__(self, physics_models, evolving_names):
        self.physics_models = physics_models
        self.evolving_names = evolving_names

    def __hash__(self) -> int:
        return hash((
            self.physics_models,
            self.evolving_names,
        ))

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return (self.physics_models == other.physics_models
                and self.evolving_names == other.evolving_names)

    def __call__(
        self,
        runtime_params: runtime_params_slice.RuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
        x: tuple[cell_variable.CellVariable, ...],
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
        allow_pereverzev: bool = False,
        explicit_call: bool = False,
    ):
        core_profiles = update_core_profiles_during_step(
            x,
            runtime_params,
            geo,
            core_profiles,
            self.evolving_names,
        )
        if allow_pereverzev:
            use_pereverzev = runtime_params.solver.use_pereverzev
        else:
            use_pereverzev = False
        return calc_coeffs(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=self.physics_models,
            evolving_names=self.evolving_names,
            use_pereverzev=use_pereverzev,
            explicit_call=explicit_call,
        )


def _calculate_pereverzev_flux(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    consts = constants.CONSTANTS
    geo_factor = jnp.concatenate(
        [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]])
    chi_face_per_ion = (geo.g1_over_vpr_face * core_profiles.n_i.face_value() *
                        consts.keV_to_J * runtime_params.solver.chi_pereverzev)
    chi_face_per_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                       consts.keV_to_J * runtime_params.solver.chi_pereverzev)
    d_face_per_el = runtime_params.solver.D_pereverzev
    v_face_per_el = (core_profiles.n_e.face_grad() /
                     core_profiles.n_e.face_value() * d_face_per_el *
                     geo_factor)
    chi_face_per_ion = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        chi_face_per_ion,
    )
    chi_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        chi_face_per_el,
    )
    v_heat_face_ion = (core_profiles.T_i.face_grad() /
                       core_profiles.T_i.face_value() * chi_face_per_ion)
    v_heat_face_el = (core_profiles.T_e.face_grad() /
                      core_profiles.T_e.face_value() * chi_face_per_el)
    d_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        d_face_per_el * geo.g1_over_vpr_face,
    )
    v_face_per_el = jnp.where(
        geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
        0.0,
        v_face_per_el * geo.g0_face,
    )
    chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
    chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])
    return (
        chi_face_per_ion,
        chi_face_per_el,
        v_heat_face_ion,
        v_heat_face_el,
        d_face_per_el,
        v_face_per_el,
    )


def calc_coeffs(runtime_params,
                geo,
                core_profiles,
                explicit_source_profiles,
                physics_models,
                evolving_names,
                use_pereverzev=False,
                explicit_call=False):
    if explicit_call and runtime_params.solver.theta_implicit == 1.0:
        return _calc_coeffs_reduced(
            geo,
            core_profiles,
            evolving_names,
        )
    else:
        return _calc_coeffs_full(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            physics_models=physics_models,
            evolving_names=evolving_names,
            use_pereverzev=use_pereverzev,
        )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_full(runtime_params,
                      geo,
                      core_profiles,
                      explicit_source_profiles,
                      physics_models,
                      evolving_names,
                      use_pereverzev=False):
    consts = constants.CONSTANTS
    pedestal_model_output = physics_models.pedestal_model(
        runtime_params, geo, core_profiles)
    mask = (jnp.zeros_like(
        geo.rho,
        dtype=bool).at[pedestal_model_output.rho_norm_ped_top_idx].set(True))
    conductivity = (
        physics_models.neoclassical_models.conductivity.calculate_conductivity(
            geo, core_profiles))
    merged_source_profiles = source_profile_builders.build_source_profiles(
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    source_mat_psi = jnp.zeros_like(geo.rho)
    source_psi = merged_source_profiles.total_psi_sources(geo)
    toc_T_i = 1.5 * geo.vpr**(-2.0 / 3.0) * consts.keV_to_J
    tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
    toc_T_e = 1.5 * geo.vpr**(-2.0 / 3.0) * consts.keV_to_J
    tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
    toc_psi = (1.0 / runtime_params.numerics.resistivity_multiplier *
               geo.rho_norm * conductivity.sigma * consts.mu_0 * 16 *
               jnp.pi**2 * geo.Phi_b**2 / geo.F**2)
    tic_psi = jnp.ones_like(toc_psi)
    toc_dens_el = jnp.ones_like(geo.vpr)
    tic_dens_el = geo.vpr
    turbulent_transport = physics_models.transport_model(
        runtime_params, geo, core_profiles, pedestal_model_output)
    neoclassical_transport = physics_models.neoclassical_models.transport(
        runtime_params, geo, core_profiles)
    chi_face_ion_total = (turbulent_transport.chi_face_ion +
                          neoclassical_transport.chi_neo_i)
    chi_face_el_total = (turbulent_transport.chi_face_el +
                         neoclassical_transport.chi_neo_e)
    d_face_el_total = (turbulent_transport.d_face_el +
                       neoclassical_transport.D_neo_e)
    v_face_el_total = (turbulent_transport.v_face_el +
                       neoclassical_transport.V_neo_e +
                       neoclassical_transport.V_neo_ware_e)
    d_face_psi = geo.g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    full_chi_face_ion = (geo.g1_over_vpr_face *
                         core_profiles.n_i.face_value() * consts.keV_to_J *
                         chi_face_ion_total)
    full_chi_face_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                        consts.keV_to_J * chi_face_el_total)
    full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
    full_v_face_el = geo.g0_face * v_face_el_total
    source_mat_nn = jnp.zeros_like(geo.rho)
    source_n_e = merged_source_profiles.total_sources('n_e', geo)
    source_n_e += (mask * runtime_params.numerics.adaptive_n_source_prefactor *
                   pedestal_model_output.n_e_ped)
    source_mat_nn += -(mask *
                       runtime_params.numerics.adaptive_n_source_prefactor)
    (
        chi_face_per_ion,
        chi_face_per_el,
        v_heat_face_ion,
        v_heat_face_el,
        d_face_per_el,
        v_face_per_el,
    ) = jax.lax.cond(
        use_pereverzev,
        lambda: _calculate_pereverzev_flux(
            runtime_params,
            geo,
            core_profiles,
            pedestal_model_output,
        ),
        lambda: tuple([jnp.zeros_like(geo.rho_face)] * 6),
    )
    full_chi_face_ion += chi_face_per_ion
    full_chi_face_el += chi_face_per_el
    full_d_face_el += d_face_per_el
    full_v_face_el += v_face_per_el
    v_heat_face_ion += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                        geo.rho_face_norm * geo.vpr_face *
                        core_profiles.n_i.face_value() * consts.keV_to_J)
    v_heat_face_el += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face *
                       core_profiles.n_e.face_value() * consts.keV_to_J)
    full_v_face_el += (-1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face)
    source_i = merged_source_profiles.total_sources('T_i', geo)
    source_e = merged_source_profiles.total_sources('T_e', geo)
    qei = merged_source_profiles.qei
    source_mat_ii = qei.implicit_ii * geo.vpr
    source_i += qei.explicit_i * geo.vpr
    source_mat_ee = qei.implicit_ee * geo.vpr
    source_e += qei.explicit_e * geo.vpr
    source_mat_ie = qei.implicit_ie * geo.vpr
    source_mat_ei = qei.implicit_ei * geo.vpr
    source_i += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_i_ped)
    source_e += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_e_ped)
    source_mat_ii -= mask * runtime_params.numerics.adaptive_T_source_prefactor
    source_mat_ee -= mask * runtime_params.numerics.adaptive_T_source_prefactor
    d_vpr53_rhon_n_e_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
        geo.rho_norm,
    )
    d_vpr53_rhon_n_i_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
        geo.rho_norm,
    )
    source_i += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_i_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_i.value *
                 consts.keV_to_J)
    source_e += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_e_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_e.value *
                 consts.keV_to_J)
    d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)
    source_n_e += (1.0 / 2.0 * d_vpr_rhon_drhon * geo.Phi_b_dot / geo.Phi_b *
                   core_profiles.n_e.value)
    source_psi += (8.0 * jnp.pi**2 * consts.mu_0 * geo.Phi_b_dot * geo.Phi_b *
                   geo.rho_norm**2 * conductivity.sigma / geo.F**2 *
                   core_profiles.psi.grad())
    var_to_toc = {
        'T_i': toc_T_i,
        'T_e': toc_T_e,
        'psi': toc_psi,
        'n_e': toc_dens_el,
    }
    var_to_tic = {
        'T_i': tic_T_i,
        'T_e': tic_T_e,
        'psi': tic_psi,
        'n_e': tic_dens_el,
    }
    transient_out_cell = tuple(var_to_toc[var] for var in evolving_names)
    transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)
    var_to_d_face = {
        'T_i': full_chi_face_ion,
        'T_e': full_chi_face_el,
        'psi': d_face_psi,
        'n_e': full_d_face_el,
    }
    d_face = tuple(var_to_d_face[var] for var in evolving_names)
    var_to_v_face = {
        'T_i': v_heat_face_ion,
        'T_e': v_heat_face_el,
        'psi': v_face_psi,
        'n_e': full_v_face_el,
    }
    v_face = tuple(var_to_v_face.get(var) for var in evolving_names)
    d = {
        ('T_i', 'T_i'): source_mat_ii,
        ('T_i', 'T_e'): source_mat_ie,
        ('T_e', 'T_i'): source_mat_ei,
        ('T_e', 'T_e'): source_mat_ee,
        ('n_e', 'n_e'): source_mat_nn,
        ('psi', 'psi'): source_mat_psi,
    }
    source_mat_cell = tuple(
        tuple(d.get((row_block, col_block)) for col_block in evolving_names)
        for row_block in evolving_names)
    var_to_source = {
        'T_i': source_i / SCALING_FACTORS['T_i'],
        'T_e': source_e / SCALING_FACTORS['T_e'],
        'psi': source_psi / SCALING_FACTORS['psi'],
        'n_e': source_n_e / SCALING_FACTORS['n_e'],
    }
    source_cell = tuple(var_to_source.get(var) for var in evolving_names)
    coeffs = Block1DCoeffs(
        transient_out_cell=transient_out_cell,
        transient_in_cell=transient_in_cell,
        d_face=d_face,
        v_face=v_face,
        source_mat_cell=source_mat_cell,
        source_cell=source_cell,
        auxiliary_outputs=(
            merged_source_profiles,
            conductivity,
            state.CoreTransport(**dataclasses.asdict(turbulent_transport),
                                **dataclasses.asdict(neoclassical_transport)),
        ),
    )
    return coeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
):
    tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
    tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
    tic_psi = jnp.ones_like(geo.vpr)
    tic_dens_el = geo.vpr
    var_to_tic = {
        'T_i': tic_T_i,
        'T_e': tic_T_e,
        'psi': tic_psi,
        'n_e': tic_dens_el,
    }
    transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)
    coeffs = Block1DCoeffs(transient_in_cell=transient_in_cell, )
    return coeffs


def calc_c(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs: Block1DCoeffs,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array]:
    d_face = coeffs.d_face
    v_face = coeffs.v_face
    source_mat_cell = coeffs.source_mat_cell
    source_cell = coeffs.source_cell
    num_cells = x[0].value.shape[0]
    num_channels = len(x)
    for _, x_i in enumerate(x):
        if x_i.value.shape != (num_cells, ):
            raise ValueError(
                f'Expected each x channel to have shape ({num_cells},) '
                f'but got {x_i.value.shape}.')
    zero_block = jnp.zeros((num_cells, num_cells))
    zero_row_of_blocks = [zero_block] * num_channels
    zero_vec = jnp.zeros((num_cells))
    zero_block_vec = [zero_vec] * num_channels
    c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
    c = zero_block_vec.copy()
    if d_face is not None:
        for i in range(num_channels):
            (
                diffusion_mat,
                diffusion_vec,
            ) = diffusion_terms.make_diffusion_terms(
                d_face[i],
                x[i],
            )
            c_mat[i][i] += diffusion_mat
            c[i] += diffusion_vec
    if v_face is not None:
        for i in range(num_channels):
            d_face_i = d_face[i] if d_face is not None else None
            d_face_i = jnp.zeros_like(
                v_face[i]) if d_face_i is None else d_face_i
            (
                conv_mat,
                conv_vec,
            ) = convection_terms.make_convection_terms(
                v_face[i],
                d_face_i,
                x[i],
                dirichlet_mode=convection_dirichlet_mode,
                neumann_mode=convection_neumann_mode,
            )
            c_mat[i][i] += conv_mat
            c[i] += conv_vec
    if source_mat_cell is not None:
        for i in range(num_channels):
            for j in range(num_channels):
                source = source_mat_cell[i][j]
                if source is not None:
                    c_mat[i][j] += jnp.diag(source)

    def add(left: jax.Array, right: jax.Array | None):
        if right is not None:
            return left + right
        return left

    if source_cell is not None:
        c = [add(c_i, source_i) for c_i, source_i in zip(c, source_cell)]
    c_mat = jnp.block(c_mat)
    c = jnp.block(c)
    return c_mat, c


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def theta_method_matrix_equation(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old: Block1DCoeffs,
    coeffs_new: Block1DCoeffs,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
):
    x_new_guess_vec = cell_variable_tuple_to_vec(x_new_guess)
    theta_exp = 1.0 - theta_implicit
    tc_in_old = jnp.concatenate(coeffs_old.transient_in_cell)
    tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
    tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)
    eps = 1e-7
    left_transient = jnp.identity(len(x_new_guess_vec))
    right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))
    c_mat_new, c_new = calc_c(
        x_new_guess,
        coeffs_new,
        convection_dirichlet_mode,
        convection_neumann_mode,
    )
    broadcasted = jnp.expand_dims(1 / (tc_out_new * tc_in_new), 1)
    lhs_mat = left_transient - dt * theta_implicit * broadcasted * c_mat_new
    lhs_vec = -theta_implicit * dt * (1 / (tc_out_new * tc_in_new)) * c_new
    if theta_exp > 0.0:
        tc_out_old = jnp.concatenate(coeffs_old.transient_out_cell)
        tc_in_new = jax_utils.error_if(
            tc_in_new,
            jnp.any(jnp.abs(tc_out_old * tc_in_new) < eps),
            msg='|tc_out_old*tc_in_new| unexpectedly < eps',
        )
        c_mat_old, c_old = discrete_system.calc_c(
            x_old,
            coeffs_old,
            convection_dirichlet_mode,
            convection_neumann_mode,
        )
        broadcasted = jnp.expand_dims(1 / (tc_out_old * tc_in_new), 1)
        rhs_mat = right_transient + dt * theta_exp * broadcasted * c_mat_old
        rhs_vec = dt * theta_exp * (1 / (tc_out_old * tc_in_new)) * c_old
    else:
        rhs_mat = right_transient
        rhs_vec = jnp.zeros_like(x_new_guess_vec)
    return lhs_mat, lhs_vec, rhs_mat, rhs_vec


MIN_DELTA: Final[float] = 1e-7


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'convection_dirichlet_mode',
        'convection_neumann_mode',
        'theta_implicit',
    ],
)
def implicit_solve_block(
    dt: jax.Array,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    coeffs_old,
    coeffs_new,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[cell_variable.CellVariable, ...]:
    x_old_vec = cell_variable_tuple_to_vec(x_old)
    lhs_mat, lhs_vec, rhs_mat, rhs_vec = (theta_method_matrix_equation(
        dt=dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        coeffs_old=coeffs_old,
        coeffs_new=coeffs_new,
        theta_implicit=theta_implicit,
        convection_dirichlet_mode=convection_dirichlet_mode,
        convection_neumann_mode=convection_neumann_mode,
    ))
    rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec - lhs_vec
    x_new = jnp.linalg.solve(lhs_mat, rhs)
    x_new = jnp.split(x_new, len(x_old))
    out = [
        dataclasses.replace(var, value=value)
        for var, value in zip(x_new_guess, x_new)
    ]
    out = tuple(out)
    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    theta_implicit: float = dataclasses.field(metadata={'static': True})
    use_predictor_corrector: bool = dataclasses.field(
        metadata={'static': True})
    n_corrector_steps: int = dataclasses.field(metadata={'static': True})
    convection_dirichlet_mode: str = dataclasses.field(
        metadata={'static': True})
    convection_neumann_mode: str = dataclasses.field(metadata={'static': True})
    use_pereverzev: bool = dataclasses.field(metadata={'static': True})
    chi_pereverzev: float
    D_pereverzev: float


class Solver(abc.ABC):

    def __init__(self, physics_models):
        self.physics_models = physics_models

    def __hash__(self) -> int:
        return hash(self.physics_models)

    def __eq__(self, other: typing_extensions.Self) -> bool:
        return self.physics_models == other.physics_models

    @functools.partial(
        jax_utils.jit,
        static_argnames=[
            'self',
        ],
    )
    def __call__(
        self,
        t: jax.Array,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: source_profiles.SourceProfiles,
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        if runtime_params_t.numerics.evolving_names:
            (
                x_new,
                solver_numeric_output,
            ) = self._x_new(
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t=geo_t,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=core_profiles_t,
                core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles,
                evolving_names=runtime_params_t.numerics.evolving_names,
            )
        else:
            x_new = tuple()
            solver_numeric_output = state.SolverNumericOutputs()
        return (
            x_new,
            solver_numeric_output,
        )

    def _x_new(
        self,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: source_profiles.SourceProfiles,
        evolving_names: tuple[str, ...],
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        raise NotImplementedError(
            f'{type(self)} must implement `_x_new` or '
            'implement a different `__call__` that does not'
            ' need `_x_new`.')


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'coeffs_callback',
    ],
)
def predictor_corrector_method(
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    coeffs_exp,
    explicit_source_profiles: source_profiles.SourceProfiles,
    coeffs_callback: CoeffsCallback,
) -> tuple[cell_variable.CellVariable, ...]:
    solver_params = runtime_params_t_plus_dt.solver

    def loop_body(i, x_new_guess):
        coeffs_new = coeffs_callback(
            runtime_params_t_plus_dt,
            geo_t_plus_dt,
            core_profiles_t_plus_dt,
            x_new_guess,
            explicit_source_profiles=explicit_source_profiles,
            allow_pereverzev=True,
        )
        return implicit_solve_block(
            dt=dt,
            x_old=x_old,
            x_new_guess=x_new_guess,
            coeffs_old=coeffs_exp,
            coeffs_new=coeffs_new,
            theta_implicit=solver_params.theta_implicit,
            convection_dirichlet_mode=(
                solver_params.convection_dirichlet_mode),
            convection_neumann_mode=(solver_params.convection_neumann_mode),
        )

    if solver_params.use_predictor_corrector:
        x_new = xnp.fori_loop(
            0,
            runtime_params_t_plus_dt.solver.n_corrector_steps + 1,
            loop_body,
            x_new_guess,
        )
    else:
        x_new = loop_body(0, x_new_guess)
    return x_new


class LinearThetaMethod0(Solver):

    @functools.partial(
        jax_utils.jit,
        static_argnames=[
            'self',
            'evolving_names',
        ],
    )
    def _x_new(
        self,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        core_profiles_t: state.CoreProfiles,
        core_profiles_t_plus_dt: state.CoreProfiles,
        explicit_source_profiles: source_profiles.SourceProfiles,
        evolving_names: tuple[str, ...],
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        x_old = core_profiles_to_solver_x_tuple(core_profiles_t,
                                                evolving_names)
        x_new_guess = core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt,
                                                      evolving_names)
        coeffs_callback = CoeffsCallback(
            physics_models=self.physics_models,
            evolving_names=evolving_names,
        )
        coeffs_exp = coeffs_callback(
            runtime_params_t,
            geo_t,
            core_profiles_t,
            x_old,
            explicit_source_profiles=explicit_source_profiles,
            allow_pereverzev=True,
            explicit_call=True,
        )
        x_new = predictor_corrector_method(
            dt=dt,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=x_old,
            x_new_guess=x_new_guess,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            coeffs_exp=coeffs_exp,
            coeffs_callback=coeffs_callback,
            explicit_source_profiles=explicit_source_profiles,
        )
        if runtime_params_t_plus_dt.solver.use_predictor_corrector:
            inner_solver_iterations = (
                1 + runtime_params_t_plus_dt.solver.n_corrector_steps)
        else:
            inner_solver_iterations = 1
        solver_numeric_outputs = state.SolverNumericOutputs(
            inner_solver_iterations=inner_solver_iterations,
            outer_solver_iterations=1,
            solver_error_state=0,
        )
        return (
            x_new,
            solver_numeric_outputs,
        )


class BaseSolver(torax_pydantic.BaseModelFrozen, abc.ABC):
    theta_implicit: Annotated[torax_pydantic.UnitInterval,
                              torax_pydantic.JAX_STATIC] = 1.0
    use_predictor_corrector: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    n_corrector_steps: Annotated[pydantic.PositiveInt,
                                 torax_pydantic.JAX_STATIC] = 10
    convection_dirichlet_mode: Annotated[Literal['ghost', 'direct',
                                                 'semi-implicit'],
                                         torax_pydantic.JAX_STATIC] = 'ghost'
    convection_neumann_mode: Annotated[Literal['ghost', 'semi-implicit'],
                                       torax_pydantic.JAX_STATIC] = 'ghost'
    use_pereverzev: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    chi_pereverzev: pydantic.PositiveFloat = 30.0
    D_pereverzev: pydantic.NonNegativeFloat = 15.0

    @property
    @abc.abstractmethod
    def build_runtime_params(self):
        pass

    @abc.abstractmethod
    def build_solver(self, physics_models):
        pass


class LinearThetaMethod(BaseSolver):
    solver_type: Annotated[Literal['linear'],
                           torax_pydantic.JAX_STATIC] = ('linear')

    @pydantic.model_validator(mode='before')
    @classmethod
    def scrub_log_iterations(cls, x: dict[str, Any]) -> dict[str, Any]:
        if 'log_iterations' in x:
            del x['log_iterations']
        return x

    @functools.cached_property
    def build_runtime_params(self):
        return RuntimeParams(
            theta_implicit=self.theta_implicit,
            convection_dirichlet_mode=self.convection_dirichlet_mode,
            convection_neumann_mode=self.convection_neumann_mode,
            use_pereverzev=self.use_pereverzev,
            use_predictor_corrector=self.use_predictor_corrector,
            chi_pereverzev=self.chi_pereverzev,
            D_pereverzev=self.D_pereverzev,
            n_corrector_steps=self.n_corrector_steps,
        )

    def build_solver(self, physics_models):
        return LinearThetaMethod0(physics_models=physics_models, )


class NewtonRaphsonThetaMethod(BaseSolver):
    solver_type: Annotated[Literal['newton_raphson'],
                           torax_pydantic.JAX_STATIC] = 'newton_raphson'
    log_iterations: Annotated[bool, torax_pydantic.JAX_STATIC] = False
    initial_guess_mode: Annotated[
        enums.InitialGuessMode,
        torax_pydantic.JAX_STATIC] = enums.InitialGuessMode.LINEAR
    n_max_iterations: pydantic.NonNegativeInt = 30
    residual_tol: float = 1e-5
    residual_coarse_tol: float = 1e-2
    delta_reduction_factor: float = 0.5
    tau_min: float = 0.01


SolverConfig = (LinearThetaMethod | NewtonRaphsonThetaMethod)


class g:
    pass


def not_done(t, t_final):
    return t < (t_final - g.tolerance)


def next_dt(t, runtime_params, geo, core_profiles, core_transport):
    chi_max = core_transport.chi_max(geo)
    basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max
    dt = jnp.minimum(
        runtime_params.numerics.chi_timestep_prefactor * basic_dt,
        runtime_params.numerics.max_dt,
    )
    crosses_t_final = (t < runtime_params.numerics.t_final) * (
        t + dt > runtime_params.numerics.t_final)
    dt = jax.lax.select(
        jnp.logical_and(
            runtime_params.numerics.exact_t_final,
            crosses_t_final,
        ),
        runtime_params.numerics.t_final - t,
        dt,
    )
    return dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
    sources: Any
    numerics: Any
    profile_conditions: Any
    plasma_composition: Any
    transport_model: Any
    solver: Any
    pedestal: Any
    neoclassical: Any

    @classmethod
    def from_config(cls, config):
        return cls(
            sources=config.sources,
            numerics=config.numerics,
            profile_conditions=config.profile_conditions,
            plasma_composition=config.plasma_composition,
            transport_model=config.transport,
            solver=config.solver,
            pedestal=config.pedestal,
            neoclassical=config.neoclassical,
        )

    @jax_utils.jit
    def __call__(
        self,
        t: chex.Numeric,
    ) -> runtime_params_slice.RuntimeParams:
        return runtime_params_slice.RuntimeParams(
            transport=self.transport_model.build_runtime_params(t),
            solver=self.solver.build_runtime_params,
            sources={
                source_name: source_config.build_runtime_params(t)
                for source_name, source_config in dict(self.sources).items()
                if source_config is not None
            },
            plasma_composition=self.plasma_composition.build_runtime_params(t),
            profile_conditions=self.profile_conditions.build_runtime_params(t),
            numerics=self.numerics.build_runtime_params(t),
            neoclassical=self.neoclassical.build_runtime_params(),
            pedestal=self.pedestal.build_runtime_params(t),
        )


def get_consistent_runtime_params_and_geometry(*, t, runtime_params_provider,
                                               geometry_provider):
    geo = geometry_provider(t)
    runtime_params = runtime_params_provider(t=t)
    return runtime_params_slice.make_ip_consistent(runtime_params, geo)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsModels:
    source_models: source_models_lib.SourceModels = dataclasses.field(
        metadata=dict(static=True))
    transport_model: transport_model_lib.TransportModel = dataclasses.field(
        metadata=dict(static=True))
    pedestal_model: pedestal_model_lib.PedestalModel = dataclasses.field(
        metadata=dict(static=True))
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels = (
        dataclasses.field(metadata=dict(static=True)))


TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'
JAX_STATIC: Final[str] = '_pydantic_jax_static_field'
StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"
T_E = "T_e"
T_I = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
MAGNETIC_SHEAR = "magnetic_shear"
N_IMPURITY = "n_impurity"
Z_IMPURITY = "Z_impurity"
Z_EFF = "Z_eff"
SIGMA_PARALLEL = "sigma_parallel"
V_LOOP_LCFS = "v_loop_lcfs"
J_TOTAL = "j_total"
IP_PROFILE = "Ip_profile"
IP = "Ip"
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"
CHI_TURB_I = "chi_turb_i"
CHI_TURB_E = "chi_turb_e"
D_TURB_E = "D_turb_e"
V_TURB_E = "V_turb_e"
CHI_NEO_I = "chi_neo_i"
CHI_NEO_E = "chi_neo_e"
D_NEO_E = "D_neo_e"
V_NEO_E = "V_neo_e"
V_NEO_WARE_E = "V_neo_ware_e"
CHI_BOHM_E = "chi_bohm_e"
CHI_GYROBOHM_E = "chi_gyrobohm_e"
CHI_BOHM_I = "chi_bohm_i"
CHI_GYROBOHM_I = "chi_gyrobohm_i"
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_NORM = "rho_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"
Q_FUSION = "Q_fusion"
SIM_ERROR = "sim_error"
OUTER_SOLVER_ITERATIONS = "outer_solver_iterations"
INNER_SOLVER_ITERATIONS = "inner_solver_iterations"
SAWTOOTH_CRASH = "sawtooth_crash"
EXCLUDED_GEOMETRY_NAMES = frozenset({
    RHO_FACE,
    RHO_CELL,
    RHO_CELL_NORM,
    RHO_FACE_NORM,
    "rho",
    "rho_norm",
    "q_correction_factor",
})


def _extend_cell_grid_to_boundaries(
    cell_var: array_typing.FloatVectorCell,
    face_var: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorCellPlusBoundaries:
    left_value = np.expand_dims(face_var[:, 0], axis=-1)
    right_value = np.expand_dims(face_var[:, -1], axis=-1)
    return np.concatenate([left_value, cell_var, right_value], axis=-1)


class StateHistory:

    def __init__(self, state_history, post_processed_outputs_history,
                 sim_error, torax_config):
        state_history[0].core_profiles = dataclasses.replace(
            state_history[0].core_profiles,
            v_loop_lcfs=state_history[1].core_profiles.v_loop_lcfs,
        )
        self._sim_error = sim_error
        self._torax_config = torax_config
        self._post_processed_outputs = post_processed_outputs_history
        self._solver_numeric_outputs = [
            state.solver_numeric_outputs for state in state_history
        ]
        self._core_profiles = [state.core_profiles for state in state_history]
        self._core_sources = [state.core_sources for state in state_history]
        self._transport = [state.core_transport for state in state_history]
        self._geometries = [state.geometry for state in state_history]
        self._stacked_geometry = geometry_lib.stack_geometries(self.geometries)
        stack = lambda *ys: np.stack(ys)
        self._stacked_core_profiles: state.CoreProfiles = jax.tree_util.tree_map(
            stack, *self._core_profiles)
        self._stacked_core_sources: source_profiles_lib.SourceProfiles = (
            jax.tree_util.tree_map(stack, *self._core_sources))
        self._stacked_core_transport: state.CoreTransport = jax.tree_util.tree_map(
            stack, *self._transport)
        self._stacked_post_processed_outputs: (
            PostProcessedOutputs) = jax.tree_util.tree_map(
                stack, *post_processed_outputs_history)
        self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
            jax.tree_util.tree_map(stack, *self._solver_numeric_outputs))
        self._times = np.array([state.t for state in state_history])
        self._rho_cell_norm = state_history[0].geometry.rho_norm
        self._rho_face_norm = state_history[0].geometry.rho_face_norm
        self._rho_norm = np.concatenate([[0.0], self.rho_cell_norm, [1.0]])

    @property
    def torax_config(self):
        return self._torax_config

    @property
    def sim_error(self) -> state.SimError:
        return self._sim_error

    @property
    def times(self) -> array_typing.Array:
        return self._times

    @property
    def rho_cell_norm(self) -> array_typing.FloatVectorCell:
        return self._rho_cell_norm

    @property
    def rho_face_norm(self) -> array_typing.FloatVectorFace:
        return self._rho_face_norm

    @property
    def rho_norm(self) -> array_typing.FloatVectorCellPlusBoundaries:
        return self._rho_norm

    @property
    def geometries(self) -> Sequence[geometry_lib.Geometry]:
        return self._geometries

    def simulation_output_to_xr(self) -> xr.DataTree:
        time = xr.DataArray(self.times, dims=[TIME], name=TIME)
        rho_face_norm = xr.DataArray(self.rho_face_norm,
                                     dims=[RHO_FACE_NORM],
                                     name=RHO_FACE_NORM)
        rho_cell_norm = xr.DataArray(self.rho_cell_norm,
                                     dims=[RHO_CELL_NORM],
                                     name=RHO_CELL_NORM)
        rho_norm = xr.DataArray(
            self.rho_norm,
            dims=[RHO_NORM],
            name=RHO_NORM,
        )
        coords = {
            TIME: time,
            RHO_FACE_NORM: rho_face_norm,
            RHO_CELL_NORM: rho_cell_norm,
            RHO_NORM: rho_norm,
        }
        all_dicts = [
            self._save_core_profiles(),
            self._save_core_transport(),
            self._save_core_sources(),
            self._save_post_processed_outputs(),
            self._save_geometry(),
        ]
        flat_dict = {}
        for key, value in itertools.chain(*(d.items() for d in all_dicts)):
            if key not in flat_dict:
                flat_dict[key] = value
            else:
                raise ValueError(f"Duplicate key: {key}")
        numerics_dict = {
            SIM_ERROR:
            self.sim_error.value,
            SAWTOOTH_CRASH:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.sawtooth_crash,
                dims=[TIME],
                name=SAWTOOTH_CRASH,
            ),
            OUTER_SOLVER_ITERATIONS:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.outer_solver_iterations,
                dims=[TIME],
                name=OUTER_SOLVER_ITERATIONS,
            ),
            INNER_SOLVER_ITERATIONS:
            xr.DataArray(
                self._stacked_solver_numeric_outputs.inner_solver_iterations,
                dims=[TIME],
                name=INNER_SOLVER_ITERATIONS,
            ),
        }
        numerics = xr.Dataset(numerics_dict)
        profiles_dict = {
            k: v
            for k, v in flat_dict.items()
            if v is not None and v.values.ndim > 1
        }
        profiles = xr.Dataset(profiles_dict)
        scalars_dict = {
            k: v
            for k, v in flat_dict.items()
            if v is not None and v.values.ndim in [0, 1]
        }
        scalars = xr.Dataset(scalars_dict)
        data_tree = xr.DataTree(
            children={
                NUMERICS: xr.DataTree(dataset=numerics),
                PROFILES: xr.DataTree(dataset=profiles),
                SCALARS: xr.DataTree(dataset=scalars),
            },
            dataset=xr.Dataset(
                data_vars=None,
                coords=coords,
            ),
        )
        return data_tree

    def _pack_into_data_array(self, name, data):
        if data is None:
            return None
        is_face_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_face_norm),
        )
        is_cell_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_cell_norm),
        )
        is_cell_plus_boundaries_var = lambda x: x.ndim == 2 and x.shape == (
            len(self.times),
            len(self.rho_norm),
        )
        is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times), )
        is_constant = lambda x: x.ndim == 0
        match data:
            case data if is_face_var(data):
                dims = [TIME, RHO_FACE_NORM]
            case data if is_cell_var(data):
                dims = [TIME, RHO_CELL_NORM]
            case data if is_scalar(data):
                dims = [TIME]
            case data if is_constant(data):
                dims = []
            case data if is_cell_plus_boundaries_var(data):
                dims = [TIME, RHO_NORM]
            case _:
                logging.warning(
                    "Unsupported data shape for %s: %s. Skipping persisting.",
                    name,
                    data.shape,
                )
                return None
        return xr.DataArray(data, dims=dims, name=name)

    def _save_core_profiles(self, ) -> dict[str, xr.DataArray | None]:
        xr_dict = {}
        stacked_core_profiles = self._stacked_core_profiles
        output_name_map = {
            "psidot": V_LOOP,
            "sigma": SIGMA_PARALLEL,
            "Ip_profile_face": IP_PROFILE,
            "q_face": Q,
            "s_face": MAGNETIC_SHEAR,
        }
        core_profile_field_names = {
            f.name
            for f in dataclasses.fields(stacked_core_profiles)
        }
        for field in dataclasses.fields(stacked_core_profiles):
            attr_name = field.name
            if attr_name == "impurity_fractions":
                continue
            attr_value = getattr(stacked_core_profiles, attr_name)
            output_key = output_name_map.get(attr_name, attr_name)
            if attr_name.endswith("_face") and (attr_name.removesuffix("_face")
                                                in core_profile_field_names):
                continue
            if attr_name == "A_impurity":
                is_constant = np.all(attr_value == attr_value[..., 0:1],
                                     axis=-1)
                if np.all(is_constant):
                    data_to_save = attr_value[..., 0]
                else:
                    face_value = getattr(stacked_core_profiles,
                                         "A_impurity_face")
                    data_to_save = _extend_cell_grid_to_boundaries(
                        attr_value, face_value)
                xr_dict[output_key] = self._pack_into_data_array(
                    output_key, data_to_save)
                continue
            if hasattr(attr_value, "cell_plus_boundaries"):
                data_to_save = attr_value.cell_plus_boundaries()
            else:
                face_attr_name = f"{attr_name}_face"
                if face_attr_name in core_profile_field_names:
                    face_value = getattr(stacked_core_profiles, face_attr_name)
                    data_to_save = _extend_cell_grid_to_boundaries(
                        attr_value, face_value)
                else:
                    data_to_save = attr_value
            xr_dict[output_key] = self._pack_into_data_array(
                output_key, data_to_save)
        Ip_data = stacked_core_profiles.Ip_profile_face[..., -1]
        xr_dict[IP] = self._pack_into_data_array(IP, Ip_data)
        return xr_dict

    def _save_core_transport(self, ) -> dict[str, xr.DataArray | None]:
        xr_dict = {}
        core_transport = self._stacked_core_transport
        xr_dict[CHI_TURB_I] = core_transport.chi_face_ion
        xr_dict[CHI_TURB_E] = core_transport.chi_face_el
        xr_dict[D_TURB_E] = core_transport.d_face_el
        xr_dict[V_TURB_E] = core_transport.v_face_el
        xr_dict[CHI_NEO_I] = core_transport.chi_neo_i
        xr_dict[CHI_NEO_E] = core_transport.chi_neo_e
        xr_dict[D_NEO_E] = core_transport.D_neo_e
        xr_dict[V_NEO_E] = core_transport.V_neo_e
        xr_dict[V_NEO_WARE_E] = core_transport.V_neo_ware_e
        xr_dict = {
            name: self._pack_into_data_array(
                name,
                data,
            )
            for name, data in xr_dict.items()
        }
        return xr_dict

    def _save_core_sources(self, ) -> dict[str, xr.DataArray | None]:
        xr_dict = {}
        xr_dict[qei_source_lib.QeiSource.SOURCE_NAME] = (
            self._stacked_core_sources.qei.qei_coef *
            (self._stacked_core_profiles.T_e.value -
             self._stacked_core_profiles.T_i.value))
        xr_dict[J_BOOTSTRAP] = _extend_cell_grid_to_boundaries(
            self._stacked_core_sources.bootstrap_current.j_bootstrap,
            self._stacked_core_sources.bootstrap_current.j_bootstrap_face,
        )
        for profile in self._stacked_core_sources.T_i:
            if profile == "fusion":
                xr_dict["p_alpha_i"] = self._stacked_core_sources.T_i[profile]
            else:
                xr_dict[f"p_{profile}_i"] = self._stacked_core_sources.T_i[
                    profile]
        for profile in self._stacked_core_sources.T_e:
            if profile == "fusion":
                xr_dict["p_alpha_e"] = self._stacked_core_sources.T_e[profile]
            else:
                xr_dict[f"p_{profile}_e"] = self._stacked_core_sources.T_e[
                    profile]
        for profile in self._stacked_core_sources.psi:
            xr_dict[f"j_{profile}"] = self._stacked_core_sources.psi[profile]
        for profile in self._stacked_core_sources.n_e:
            xr_dict[f"s_{profile}"] = self._stacked_core_sources.n_e[profile]
        xr_dict = {
            name: self._pack_into_data_array(name, data)
            for name, data in xr_dict.items()
        }
        return xr_dict

    def _save_post_processed_outputs(self, ) -> dict[str, xr.DataArray | None]:
        xr_dict = {}
        for field in dataclasses.fields(self._stacked_post_processed_outputs):
            attr_name = field.name
            if attr_name == "impurity_species":
                continue
            attr_value = getattr(self._stacked_post_processed_outputs,
                                 attr_name)
            if hasattr(attr_value, "cell_plus_boundaries"):
                data_to_save = attr_value.cell_plus_boundaries()
            else:
                data_to_save = attr_value
            xr_dict[attr_name] = self._pack_into_data_array(
                attr_name, data_to_save)
        if self._stacked_post_processed_outputs.impurity_species:
            radiation_outputs = (construct_xarray_for_radiation_output(
                self._stacked_post_processed_outputs.impurity_species,
                self.times,
                self.rho_cell_norm,
                TIME,
                RHO_CELL_NORM,
            ))
            for key, value in radiation_outputs.items():
                xr_dict[key] = value
        return xr_dict

    def _save_geometry(self, ) -> dict[str, xr.DataArray]:
        xr_dict = {}
        geometry_attributes = dataclasses.asdict(self._stacked_geometry)
        for field_name, data in geometry_attributes.items():
            if ("hires" in field_name or
                (field_name.endswith("_face")
                 and field_name.removesuffix("_face") in geometry_attributes)
                    or field_name == "geometry_type"
                    or field_name == "Ip_from_parameters"
                    or field_name == "j_total"
                    or not isinstance(data, array_typing.Array)):
                continue
            if f"{field_name}_face" in geometry_attributes:
                data = _extend_cell_grid_to_boundaries(
                    data, geometry_attributes[f"{field_name}_face"])
            if field_name.endswith("_face"):
                field_name = field_name.removesuffix("_face")
            if field_name == "Ip_profile":
                field_name = "Ip_profile_from_geo"
            if field_name == "psi":
                field_name = "psi_from_geo"
            if field_name == "_z_magnetic_axis":
                field_name = "z_magnetic_axis"
            data_array = self._pack_into_data_array(
                field_name,
                data,
            )
            if data_array is not None:
                xr_dict[field_name] = data_array
        geometry_properties = inspect.getmembers(type(self._stacked_geometry))
        property_names = set([name for name, _ in geometry_properties])
        for name, value in geometry_properties:
            if (name.endswith("_face")
                    and name.removesuffix("_face") in property_names):
                continue
            if name in EXCLUDED_GEOMETRY_NAMES:
                continue
            if isinstance(value, property):
                property_data = value.fget(self._stacked_geometry)
                if f"{name}_face" in property_names:
                    face_data = getattr(self._stacked_geometry, f"{name}_face")
                    property_data = _extend_cell_grid_to_boundaries(
                        property_data, face_data)
                data_array = self._pack_into_data_array(name, property_data)
                if data_array is not None:
                    if name.endswith("_face"):
                        name = name.removesuffix("_face")
                    xr_dict[name] = data_array
        return xr_dict


@enum.unique
class GeometryType(enum.IntEnum):
    CIRCULAR = 0
    CHEASE = 1
    FBT = 2
    EQDSK = 3
    IMAS = 4


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Geometry:
    geometry_type: GeometryType
    torax_mesh: Any
    Phi: array_typing.Array
    Phi_face: array_typing.Array
    R_major: array_typing.FloatScalar
    a_minor: array_typing.FloatScalar
    B_0: array_typing.FloatScalar
    volume: array_typing.Array
    volume_face: array_typing.Array
    area: array_typing.Array
    area_face: array_typing.Array
    vpr: array_typing.Array
    vpr_face: array_typing.Array
    spr: array_typing.Array
    spr_face: array_typing.Array
    delta_face: array_typing.Array
    elongation: array_typing.Array
    elongation_face: array_typing.Array
    g0: array_typing.Array
    g0_face: array_typing.Array
    g1: array_typing.Array
    g1_face: array_typing.Array
    g2: array_typing.Array
    g2_face: array_typing.Array
    g3: array_typing.Array
    g3_face: array_typing.Array
    gm4: array_typing.Array
    gm4_face: array_typing.Array
    gm5: array_typing.Array
    gm5_face: array_typing.Array
    g2g3_over_rhon: array_typing.Array
    g2g3_over_rhon_face: array_typing.Array
    g2g3_over_rhon_hires: array_typing.Array
    F: array_typing.Array
    F_face: array_typing.Array
    F_hires: array_typing.Array
    R_in: array_typing.Array
    R_in_face: array_typing.Array
    R_out: array_typing.Array
    R_out_face: array_typing.Array
    spr_hires: array_typing.Array
    rho_hires_norm: array_typing.Array
    rho_hires: array_typing.Array
    Phi_b_dot: array_typing.FloatScalar
    _z_magnetic_axis: array_typing.FloatScalar | None


def update_geometries_with_Phibdot(
    *,
    dt: chex.Numeric,
    geo_t: Geometry,
    geo_t_plus_dt: Geometry,
) -> tuple[Geometry, Geometry]:
    Phibdot = (geo_t_plus_dt.Phi_b - geo_t.Phi_b) / dt
    geo_t = dataclasses.replace(geo_t, Phi_b_dot=Phibdot)
    geo_t_plus_dt = dataclasses.replace(geo_t_plus_dt, Phi_b_dot=Phibdot)
    return geo_t, geo_t_plus_dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ToraxSimState:
    t: array_typing.FloatScalar
    dt: array_typing.FloatScalar
    core_profiles: state.CoreProfiles
    core_transport: state.CoreTransport
    core_sources: source_profiles.SourceProfiles
    geometry: Any
    solver_numeric_outputs: state.SolverNumericOutputs

    def check_for_errors(self) -> state.SimError:
        return state.SimError.NO_ERROR

    def has_nan(self) -> bool:
        return any([np.any(np.isnan(x)) for x in jax.tree.leaves(self)])


def _get_initial_state(runtime_params, geo, step_fn):
    physics_models = g.solver.physics_models
    initial_core_profiles = initial_core_profiles0(
        runtime_params,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    initial_core_sources = source_profile_builders.get_all_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        conductivity=conductivity_base.Conductivity(
            sigma=initial_core_profiles.sigma,
            sigma_face=initial_core_profiles.sigma_face,
        ),
    )
    transport_coeffs = (calculate_total_transport_coeffs(
        physics_models.pedestal_model,
        physics_models.transport_model,
        physics_models.neoclassical_models,
        runtime_params,
        geo,
        initial_core_profiles,
    ))
    return ToraxSimState(
        t=np.array(runtime_params.numerics.t_initial),
        dt=np.zeros(()),
        core_profiles=initial_core_profiles,
        core_sources=initial_core_sources,
        core_transport=transport_coeffs,
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=0,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
        ),
        geometry=geo,
    )


def check_for_errors(numerics, output_state, post_processed_outputs):
    return post_processed_outputs.check_for_errors()


class SimulationStepFn:

    def __init__(self, runtime_params_provider, geometry_provider):
        self._geometry_provider = geometry_provider
        self._runtime_params_provider = runtime_params_provider

    @xnp.jit
    def __call__(
        self,
        input_state,
        previous_post_processed_outputs,
    ):
        runtime_params_t, geo_t = (get_consistent_runtime_params_and_geometry(
            t=input_state.t,
            runtime_params_provider=self._runtime_params_provider,
            geometry_provider=self._geometry_provider,
        ))
        explicit_source_profiles = source_profile_builders.build_source_profiles(
            runtime_params=runtime_params_t,
            geo=geo_t,
            core_profiles=input_state.core_profiles,
            source_models=g.solver.physics_models.source_models,
            neoclassical_models=g.solver.physics_models.neoclassical_models,
            explicit=True,
        )

        def _step():
            return self._adaptive_step(
                runtime_params_t,
                geo_t,
                explicit_source_profiles,
                input_state,
                previous_post_processed_outputs,
            )

        output_state, post_processed_outputs = _step()
        return output_state, post_processed_outputs

    def _adaptive_step(
        self,
        runtime_params_t,
        geo_t,
        explicit_source_profiles,
        input_state,
        previous_post_processed_outputs,
    ):
        evolving_names = runtime_params_t.numerics.evolving_names
        initial_dt = next_dt(
            input_state.t,
            runtime_params_t,
            geo_t,
            input_state.core_profiles,
            input_state.core_transport,
        )

        def cond_fun(inputs):
            next_dt, output = inputs
            solver_outputs = output[2]
            is_nan_next_dt = xnp.isnan(next_dt)
            solver_did_not_converge = solver_outputs.solver_error_state == 1
            if runtime_params_t.numerics.exact_t_final:
                at_exact_t_final = xnp.allclose(
                    input_state.t + next_dt,
                    runtime_params_t.numerics.t_final,
                )
            else:
                at_exact_t_final = xnp.array(False)
            next_dt_too_small = next_dt < runtime_params_t.numerics.min_dt
            take_another_step = xnp.cond(
                solver_did_not_converge,
                lambda: xnp.cond(at_exact_t_final, lambda: True, lambda:
                                 ~next_dt_too_small),
                lambda: False,
            )
            return take_another_step & ~is_nan_next_dt

        def body_fun(inputs):
            dt, output = inputs
            old_solver_outputs = output[2]
            runtime_params_t_plus_dt, geo_t_with_phibdot, geo_t_plus_dt = (
                _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
                    input_state.t,
                    dt,
                    self._runtime_params_provider,
                    geo_t,
                    self._geometry_provider,
                ))
            core_profiles_t_plus_dt = provide_core_profiles_t_plus_dt(
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=input_state.core_profiles,
            )
            x_new, solver_numeric_outputs = g.solver(
                t=input_state.t,
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t=geo_t_with_phibdot,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=input_state.core_profiles,
                core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles,
            )
            solver_numeric_outputs = state.SolverNumericOutputs(
                solver_error_state=solver_numeric_outputs.solver_error_state,
                outer_solver_iterations=old_solver_outputs.
                outer_solver_iterations + 1,
                inner_solver_iterations=old_solver_outputs.
                inner_solver_iterations +
                solver_numeric_outputs.inner_solver_iterations,
                sawtooth_crash=solver_numeric_outputs.sawtooth_crash,
            )
            next_dt = dt / runtime_params_t_plus_dt.numerics.dt_reduction_factor
            return next_dt, (
                x_new,
                dt,
                solver_numeric_outputs,
                runtime_params_t_plus_dt,
                geo_t_plus_dt,
                core_profiles_t_plus_dt,
            )

        _, result = xnp.while_loop(
            cond_fun,
            body_fun,
            (
                initial_dt,
                (
                    core_profiles_to_solver_x_tuple(input_state.core_profiles,
                                                    evolving_names),
                    initial_dt,
                    state.SolverNumericOutputs(
                        solver_error_state=1,
                        outer_solver_iterations=0,
                        inner_solver_iterations=0,
                        sawtooth_crash=False,
                    ),
                    runtime_params_t,
                    geo_t,
                    input_state.core_profiles,
                ),
            ),
        )
        output_state, post_processed_outputs = _finalize_outputs(
            t=input_state.t,
            dt=result[1],
            x_new=result[0],
            solver_numeric_outputs=result[2],
            runtime_params_t_plus_dt=result[3],
            geometry_t_plus_dt=result[4],
            core_profiles_t=input_state.core_profiles,
            core_profiles_t_plus_dt=result[5],
            explicit_source_profiles=explicit_source_profiles,
            physics_models=g.solver.physics_models,
            evolving_names=evolving_names,
            input_post_processed_outputs=previous_post_processed_outputs,
        )
        return output_state, post_processed_outputs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _finalize_outputs(t, dt, x_new, solver_numeric_outputs, geometry_t_plus_dt,
                      runtime_params_t_plus_dt, core_profiles_t,
                      core_profiles_t_plus_dt, explicit_source_profiles,
                      physics_models, evolving_names,
                      input_post_processed_outputs):
    final_core_profiles, final_source_profiles = (
        update_core_and_source_profiles_after_step(
            dt=dt,
            x_new=x_new,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo=geometry_t_plus_dt,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
            source_models=physics_models.source_models,
            neoclassical_models=physics_models.neoclassical_models,
            evolving_names=evolving_names,
        ))
    final_total_transport = (calculate_total_transport_coeffs(
        physics_models.pedestal_model,
        physics_models.transport_model,
        physics_models.neoclassical_models,
        runtime_params_t_plus_dt,
        geometry_t_plus_dt,
        final_core_profiles,
    ))
    output_state = ToraxSimState(
        t=t + dt,
        dt=dt,
        core_profiles=final_core_profiles,
        core_sources=final_source_profiles,
        core_transport=final_total_transport,
        geometry=geometry_t_plus_dt,
        solver_numeric_outputs=solver_numeric_outputs,
    )
    post_processed_outputs = make_post_processed_outputs(
        sim_state=output_state,
        runtime_params=runtime_params_t_plus_dt,
        previous_post_processed_outputs=input_post_processed_outputs,
    )
    return output_state, post_processed_outputs


def _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
    t,
    dt,
    runtime_params_provider,
    geo_t,
    geometry_provider,
):
    runtime_params_t_plus_dt, geo_t_plus_dt = (
        get_consistent_runtime_params_and_geometry(
            t=t + dt,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geometry_provider,
        ))
    if runtime_params_t_plus_dt.numerics.calcphibdot:
        geo_t, geo_t_plus_dt = update_geometries_with_Phibdot(
            dt=dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
        )
    return (
        runtime_params_t_plus_dt,
        geo_t,
        geo_t_plus_dt,
    )


class ToraxConfig(model_base.BaseModelFrozen):
    profile_conditions: profile_conditions_lib.ProfileConditions
    numerics: numerics_lib.Numerics
    plasma_composition: plasma_composition_lib.PlasmaComposition
    geometry: Geometry0
    sources: sources_pydantic_model.Sources
    neoclassical: Neoclassical0 = Neoclassical0()
    solver: SolverConfig = pydantic.Field(discriminator='solver_type')
    transport: QLKNNTransportModel = pydantic.Field(discriminator='model_name')
    pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
        discriminator='model_name')


CONFIG = {
    'plasma_composition': {
        'main_ion': {
            'D': 0.5,
            'T': 0.5
        },
        'impurity': 'Ne',
        'Z_eff': 1.6,
    },
    'profile_conditions': {
        'Ip': 10.5e6,
        'T_i': {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        'T_i_right_bc': 0.2,
        'T_e': {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        'T_e_right_bc': 0.2,
        'n_e_right_bc': 0.25e20,
        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,
        'nbar': 0.8,
        'n_e': {
            0: {
                0.0: 1.5,
                1.0: 1.0
            }
        },
    },
    'numerics': {
        't_final': 5,
        'resistivity_multiplier': 200,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        'chi_timestep_prefactor': 50,
        'dt_reduction_factor': 3,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'sources': {
        'generic_current': {
            'fraction_of_total_current': 0.46,
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },
        'generic_particle': {
            'S_total': 2.05e20,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'gas_puff': {
            'puff_decay_length': 0.3,
            'S_total': 6.0e21,
        },
        'pellet': {
            'S_total': 0.0e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        'generic_heat': {
            'gaussian_location': 0.12741589640723575,
            'gaussian_width': 0.07280908366127758,
            'P_total': 51.0e6,
            'electron_heat_fraction': 0.68,
        },
        'fusion': {},
        'ei_exchange': {
            'Qei_multiplier': 1.0,
        },
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': 4.5,
        'T_e_ped': 4.5,
        'n_e_ped': 0.62e20,
        'rho_norm_ped_top': 0.9,
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'D_e_inner': 0.25,
        'V_e_inner': 0.0,
        'chi_i_inner': 1.0,
        'chi_e_inner': 1.0,
        'rho_inner': 0.2,
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.9,
        'chi_min': 0.05,
        'chi_max': 100,
        'D_e_min': 0.05,
        'DV_effective': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'avoid_big_negative_s': True,
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'n_corrector_steps': 1,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },
}
g.tolerance = 1e-7
torax_config = ToraxConfig.from_dict(CONFIG)
mesh = torax_config.geometry.build_provider.torax_mesh
interpolated_param_2d.set_grid(torax_config, mesh, mode='relaxed')
geometry_provider = torax_config.geometry.build_provider
g.physics_models = PhysicsModels(
    pedestal_model=torax_config.pedestal.build_pedestal_model(),
    source_models=torax_config.sources.build_models(),
    transport_model=torax_config.transport.build_transport_model(),
    neoclassical_models=torax_config.neoclassical.build_models(),
)
g.solver = torax_config.solver.build_solver(physics_models=g.physics_models)
runtime_params_provider = (RuntimeParamsProvider.from_config(torax_config))
step_fn = SimulationStepFn(
    geometry_provider=geometry_provider,
    runtime_params_provider=runtime_params_provider,
)
runtime_params_for_init, geo_for_init = (
    get_consistent_runtime_params_and_geometry(
        t=torax_config.numerics.t_initial,
        runtime_params_provider=runtime_params_provider,
        geometry_provider=geometry_provider,
    ))
initial_state = _get_initial_state(
    runtime_params=runtime_params_for_init,
    geo=geo_for_init,
    step_fn=step_fn,
)
post_processed_outputs = make_post_processed_outputs(initial_state,
                                                     runtime_params_for_init)
initial_post_processed_outputs = post_processed_outputs
current_state = initial_state
state_history = [current_state]
post_processing_history = [initial_post_processed_outputs]
sim_error = state.SimError.NO_ERROR
initial_runtime_params = runtime_params_provider(initial_state.t)
while not_done(current_state.t, runtime_params_provider.numerics.t_final):
    current_state, post_processed_outputs = step_fn(
        current_state,
        post_processing_history[-1],
    )
    sim_error = check_for_errors(
        runtime_params_provider.numerics,
        current_state,
        post_processed_outputs,
    )
    state_history.append(current_state)
    post_processing_history.append(post_processed_outputs)
state_history = StateHistory(
    state_history=state_history,
    post_processed_outputs_history=post_processing_history,
    sim_error=sim_error,
    torax_config=torax_config,
)
data_tree = state_history.simulation_output_to_xr()
data_tree.to_netcdf("run.nc")
print(data_tree)
