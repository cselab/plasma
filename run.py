import chex
import pydantic
import dataclasses
import jax
from torax._src.mhd.sawtooth import sawtooth_models as sawtooth_models_lib
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.mhd.sawtooth import pydantic_model as sawtooth_pydantic_model
from torax._src.torax_pydantic import torax_pydantic
from absl import logging
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
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
import pydantic
from torax._src.torax_pydantic import torax_pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
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


class FileRestart(torax_pydantic.BaseModelFrozen):
    filename: pydantic.FilePath
    time: torax_pydantic.Second
    do_restart: bool
    stitch: bool


class Neoclassical0(torax_pydantic.BaseModelFrozen):
    """Config for neoclassical models."""

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
        # Set zero models if model not in config dict.
        if "bootstrap_current" not in configurable_data:
            configurable_data["bootstrap_current"] = {"model_name": "zeros"}
        if "transport" not in configurable_data:
            configurable_data["transport"] = {"model_name": "zeros"}
        # Set default model names.
        if "model_name" not in configurable_data["bootstrap_current"]:
            configurable_data["bootstrap_current"]["model_name"] = "sauter"
        if "model_name" not in configurable_data["transport"]:
            configurable_data["transport"]["model_name"] = "angioni_sauter"

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MHDModels:
    """Container for instantiated MHD model objects."""

    sawtooth_models: sawtooth_models_lib.SawtoothModels | None = None

    def __eq__(self, other: 'MHDModels') -> bool:
        return self.sawtooth_models == other.sawtooth_models

    def __hash__(self) -> int:
        return hash((self.sawtooth_models, ))


class MHD(torax_pydantic.BaseModelFrozen):
    """Config for MHD models.

  Attributes:
    sawtooth: Config for sawtooth models.
  """

    sawtooth: sawtooth_pydantic_model.SawtoothConfig | None = pydantic.Field(
        default=None)

    def build_mhd_models(self):
        """Builds and returns a container with instantiated MHD model objects."""
        sawtooth_model = (None if self.sawtooth is None else
                          self.sawtooth.build_models())
        return MHDModels(sawtooth_models=sawtooth_model)

    def build_runtime_params(
            self, t: chex.Numeric) -> mhd_runtime_params.RuntimeParams:
        """Builds and returns a container with runtime params for MHD models."""

        return mhd_runtime_params.RuntimeParams(
            **{
                mhd_model_name: mhd_model_config.build_runtime_params(t)
                for mhd_model_name, mhd_model_config in self.__dict__.items()
                if mhd_model_config is not None
            })


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsModels:
    """A container for all physics models."""

    source_models: source_models_lib.SourceModels = dataclasses.field(
        metadata=dict(static=True))
    transport_model: transport_model_lib.TransportModel = dataclasses.field(
        metadata=dict(static=True))
    pedestal_model: pedestal_model_lib.PedestalModel = dataclasses.field(
        metadata=dict(static=True))
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels = (
        dataclasses.field(metadata=dict(static=True)))
    mhd_models: MHDModels = dataclasses.field(metadata=dict(static=True))


# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name
T = TypeVar('T')

LY_OBJECT_TYPE: TypeAlias = (str
                             | Mapping[str, torax_pydantic.NumpyArray | float])

TIME_INVARIANT = torax_pydantic.TIME_INVARIANT


class CheaseConfig(torax_pydantic.BaseModelFrozen):
    """Pydantic model for the CHEASE geometry.

  Attributes:
    geometry_type: Always set to 'chease'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
  """

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
    def _check_fields(self) -> typing_extensions.Self:
        if not self.R_major >= self.a_minor:
            raise ValueError('a_minor must be less than or equal to R_major.')
        return self

    def build_geometry(self) -> standard_geometry.StandardGeometry:

        return standard_geometry.build_standard_geometry(
            _apply_relevant_kwargs(
                standard_geometry.StandardGeometryIntermediates.from_chease,
                self.__dict__,
            ))


class GeometryConfig(torax_pydantic.BaseModelFrozen):
    """Pydantic model for a single geometry config."""

    config: (CheaseConfig) = pydantic.Field(discriminator='geometry_type')


class Geometry0(torax_pydantic.BaseModelFrozen):
    """Pydantic model for a geometry.

  This object can be constructed via `Geometry.from_dict(config)`, where
  `config` is a dict described in
  https://torax.readthedocs.io/en/latest/configuration.html#geometry.

  Attributes:
    geometry_type: A `geometry.GeometryType` enum.
    geometry_configs: Either a single `GeometryConfig` or a dict of
      `GeometryConfig` objects, where the keys are times in seconds.
  """

    geometry_type: geometry.GeometryType
    geometry_configs: GeometryConfig | dict[torax_pydantic.Second,
                                            GeometryConfig]

    @pydantic.model_validator(mode='before')
    @classmethod
    def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:

        if 'geometry_type' not in data:
            raise ValueError('geometry_type must be set in the input config.')

        geometry_type = data['geometry_type']
        # The geometry type can be an int if loading from JSON.
        if isinstance(geometry_type, geometry.GeometryType | int):
            return data
        # Parse the user config dict.
        elif isinstance(geometry_type, str):
            return _conform_user_data(data)
        else:
            raise ValueError(f'Invalid value for geometry: {geometry_type}')

    @functools.cached_property
    def build_provider(self) -> geometry_provider.GeometryProvider:
        # TODO(b/398191165): Remove this branch once the FBT bundle logic is
        # redesigned.
        if isinstance(self.geometry_configs, dict):
            geometries = {
                time: config.config.build_geometry()
                for time, config in self.geometry_configs.items()
            }
            provider = (
                geometry_provider.TimeDependentGeometryProvider.create_provider
                if self.geometry_type == geometry.GeometryType.CIRCULAR else
                standard_geometry.StandardGeometryProvider.create_provider)
        else:
            geometries = self.geometry_configs.config.build_geometry()
            provider = geometry_provider.ConstantGeometryProvider

        return provider(geometries)  # pytype: disable=attribute-error


def _conform_user_data(data: dict[str, Any]) -> dict[str, Any]:
    """Conform the user geometry dict to the pydantic model."""

    if 'LY_bundle_object' in data and 'geometry_configs' in data:
        raise ValueError(
            'Cannot use both `LY_bundle_object` and `geometry_configs` together.'
        )

    data_copy = data.copy()
    # Useful to avoid failing if users mistakenly give the wrong case.
    data_copy['geometry_type'] = data['geometry_type'].lower()
    geometry_type = getattr(geometry.GeometryType,
                            data['geometry_type'].upper())
    constructor_args = {'geometry_type': geometry_type}
    configs_time_dependent = data_copy.pop('geometry_configs', None)
    constructor_args['geometry_configs'] = {'config': data_copy}
    return constructor_args


def _apply_relevant_kwargs(f: Callable[..., T], kwargs: Mapping[str,
                                                                Any]) -> T:
    """Apply only the kwargs actually used by the function."""
    relevant_kwargs = [
        i.name for i in inspect.signature(f).parameters.values()
    ]
    kwargs = {k: kwargs[k] for k in relevant_kwargs}
    return f(**kwargs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SafetyFactorFit:
    """Collection of outputs calculated after each simulation step.

  Attributes:
    rho_q_min: rho_norm at the minimum q.
    q_min: Minimum q value.
    rho_q_3_2_first: First outermost rho_norm value that intercepts the
      q=3/2 plane.
    rho_q_2_1_first: First outermost rho_norm value that intercepts the q=2/1
      plane.
    rho_q_3_1_first: First outermost rho_norm value that intercepts the q=3/1
      plane.
    rho_q_3_2_second: Second outermost rho_norm value that intercepts the
      q=3/2 plane.
    rho_q_2_1_second: Second outermost rho_norm value that intercepts the q=2/1
      plane.
    rho_q_3_1_second: Second outermost rho_norm value that intercepts the q=3/1
      plane.
  """

    rho_q_min: array_typing.FloatScalar
    q_min: array_typing.FloatScalar
    rho_q_3_2_first: array_typing.FloatScalar
    rho_q_2_1_first: array_typing.FloatScalar
    rho_q_3_1_first: array_typing.FloatScalar
    rho_q_3_2_second: array_typing.FloatScalar
    rho_q_2_1_second: array_typing.FloatScalar
    rho_q_3_1_second: array_typing.FloatScalar


def _sliding_window_of_three(flat_array: jax.Array) -> jax.Array:
    """Sliding window equivalent to numpy.lib.stride_tricks.sliding_window."""
    window_size = 3
    starts = jnp.arange(len(flat_array) - window_size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(
        flat_array, (start, ), (window_size, )))(starts)


def _fit_polynomial_to_intervals_of_three(
        rho_norm: jax.Array,
        q_face: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit polynomial to every set of three points in given arrays."""
    q_face_intervals = _sliding_window_of_three(q_face, )
    rho_norm_intervals = _sliding_window_of_three(rho_norm, )

    @jax.vmap
    def batch_polyfit(q_face_interval: jax.Array,
                      rho_norm_interval: jax.Array) -> jax.Array:
        """Solve Ax=b to get coefficients for a quadratic polynomial."""
        chex.assert_shape(q_face_interval, (3, ))
        chex.assert_shape(rho_norm_interval, (3, ))
        rho_norm_squared = rho_norm_interval**2
        A = jnp.array([  # pylint: disable=invalid-name
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
    """Returns the minimum value and location of the fit quadratic in the interval."""
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
    """Finds the roots of a quadratic equation."""
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
    """Finds roots of quadratic(coeffs)=q_surface in the interval."""
    intercept_coeffs = coeffs - jnp.array([0.0, 0.0, q_surface])
    min_interval, max_interval = interval[0], interval[1]
    root_values = _find_roots_quadratic(intercept_coeffs)
    in_interval = jnp.greater(root_values, min_interval) & jnp.less(
        root_values, max_interval)
    return jnp.where(in_interval, root_values, -jnp.inf)


@jax_utils.jit
def find_min_q_and_q_surface_intercepts(rho_norm: jax.Array,
                                        q_face: jax.Array) -> SafetyFactorFit:
    """Finds the minimum q and the q surface intercepts.

  This method divides the input arrays into groups of three points, fits a
  quadratic to each interval containing three points. It then uses the fit
  quadratics to find the minimum q value as well as any intercepts of the q=3/2,
  q=2/1, and q=3/1 planes.

  As the quadratic fits could have overlapping definitions where the groups
  overlap, we choose the quadratic for the domain [rho_norm[i-1], rho_norm[i]]
  defined using the points (i, i-1, i-2).
  (There is one exception which is the first group for which the domain for
  points in [rho_norm[0], rho_norm[2]] are defined using points (0, 1, 2))

  Args:
    rho_norm: Array of rho_norm values on face grid.
    q_face: Array of q values on face grid.

  Returns:
    rho_q_min: rho_norm at the minimum q.
    q_min: minimum q value.
    outermost_rho_q_3_2: Array of two outermost rho_norm values that intercept
    the q=3/2 plane. If fewer than two intercepts are found, entries will be set
    to -inf.
    outermost_rho_q_2_1: Array of 2 outermost rho_norm values that intercept the
    q=2/1 plane. If fewer than two intercepts are found, entries will be set to
    -inf.
    outermost_rho_q_3_1: Array of 2 outermost rho_norm values that intercept the
    q=3/1 plane. If fewer than two intercepts are found, entries will be set to
    -inf.
  """
    if len(q_face) != len(rho_norm):
        raise ValueError(
            f'Input arrays must have the same length. {len(q_face)} !='
            f' {len(rho_norm)}')
    if len(q_face) < 4:
        raise ValueError('Input arrays must have at least four points.')
    # Sort in case input is not sorted.
    sorted_indices = jnp.argsort(rho_norm)
    rho_norm = rho_norm[sorted_indices]
    q_face = q_face[sorted_indices]
    # Fit polynomial to every set of three points in given arrays.
    poly_coeffs, rho_norm_3, q_face_3 = _fit_polynomial_to_intervals_of_three(
        rho_norm, q_face)
    # To bridge across the entire span of rho_norm, and avoid overlapping
    # interval domains, we define the first group from points (0, 2), and all
    # subsequent groups from (i+1, i+2), from i=1 until i=len(rho_norm)-1.
    first_rho_norm = jnp.expand_dims(jnp.array([rho_norm[0], rho_norm[2]]),
                                     axis=0)
    first_q_face = jnp.expand_dims(jnp.array([q_face[0], q_face[2]]), axis=0)
    rho_norms = jnp.concat([first_rho_norm, rho_norm_3[1:, 1:]], axis=0)
    q_faces = jnp.concat([first_q_face, q_face_3[1:, 1:]], axis=0)

    # Find the minimum q value and its location for each interval.
    rho_q_min_intervals, q_min_intervals = _minimum_location_value_in_interval(
        poly_coeffs, rho_norms, q_faces)
    # Find the minimum q value and its location out of all intervals.
    arg_q_min = jnp.argmin(q_min_intervals)
    rho_q_min = rho_q_min_intervals[arg_q_min]
    q_min = q_min_intervals[arg_q_min]

    # Find the outermost rho_norm values that intercept the q=3/2, q=2/1, and
    # q=3/1 planes. If none are found, fill from the left with -jnp.inf.
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


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpuritySpeciesOutput:
    """Impurity radiation output for a single species.

  Attributes:
    radiation: The impurity radiation for the species [Wm^-3].
    n_impurity: The impurity density [m^-3].
    Z_impurity: The impurity average charge state.
  """

    radiation: array_typing.FloatVectorCell
    n_impurity: array_typing.FloatVectorCell
    Z_impurity: array_typing.FloatVectorCell


def calculate_impurity_species_output(sim_state, runtime_params):
    impurity_species_output = {}
    mavrin_active = True
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
        if mavrin_active:
            lz = impurity_radiation_mavrin_fit.calculate_impurity_radiation_single_species(
                core_profiles.T_e.value, symbol)
            radiation = n_imp * core_profiles.n_e.value * lz
        else:
            radiation = jnp.zeros_like(n_imp)
        impurity_species_output[symbol] = ImpuritySpeciesOutput(
            radiation=radiation, n_impurity=n_imp, Z_impurity=Z_imp)

    return impurity_species_output


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
    """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
    pressure_thermal_i: Ion thermal pressure [Pa]
    pressure_thermal_e: Electron thermal pressure [Pa]
    pressure_thermal_total: Total thermal pressure [Pa]
    pprime: Derivative of total pressure with respect to poloidal flux on the
      face grid [Pa/Wb]
    W_thermal_i: Ion thermal stored energy [J]
    W_thermal_e: Electron thermal stored energy [J]
    W_thermal_total: Total thermal stored energy [J]
    tau_E: Thermal energy confinement time [s]
    H89P: L-mode confinement quality factor with respect to the ITER89P scaling
      law derived from the ITER L-mode confinement database
    H98: H-mode confinement quality factor with respect to the ITER98y2 scaling
      law derived from the ITER H-mode confinement database
    H97L: L-mode confinement quality factor with respect to the ITER97L scaling
      law derived from the ITER L-mode confinement database
    H20: H-mode confinement quality factor with respect to the ITER20 scaling
      law derived from the updated (2020) ITER H-mode confinement database
    FFprime: FF' on the face grid, where F is the toroidal flux function
    psi_norm: Normalized poloidal flux on the face grid [Wb]
    P_SOL_i: Total ion heating power exiting the plasma with all sources:
      auxiliary heating + ion-electron exchange + fusion [W]
    P_SOL_e: Total electron heating power exiting the plasma with all sources
      and sinks: auxiliary heating + ion-electron exchange + Ohmic + fusion +
      radiation sinks [W]
    P_SOL_total: Total heating power exiting the plasma with all sources and
      sinks
    P_aux_i: Total auxiliary ion heating power [W]
    P_aux_e: Total auxiliary electron heating power [W]
    P_aux_total: Total auxiliary heating power [W]
    P_external_injected: Total external injected power before absorption [W]
    P_external_total: Total external power before absorption (
        P_external_injected + P_ohmic_e) [W]
    P_ei_exchange_i: Electron-ion heat exchange power to ions [W]
    P_ei_exchange_e: Electron-ion heat exchange power to electrons [W]
    P_aux_generic_i: Total generic_ion_el_heat_source power to ions [W]
    P_aux_generic_e: Total generic_ion_el_heat_source power to electrons [W]
    P_aux_generic_total: Total generic_ion_el_heat power [W]
    P_alpha_i: Total fusion power to ions [W]
    P_alpha_e: Total fusion power to electrons [W]
    P_alpha_total: Total fusion power to plasma [W]
    P_ohmic_e: Ohmic heating power to electrons [W]
    P_bremsstrahlung_e: Bremsstrahlung electron heat sink [W]
    P_cyclotron_e: Cyclotron radiation electron heat sink [W]
    P_ecrh_e: Total electron cyclotron source power [W]
    P_radiation_e: Impurity radiation heat sink [W]
    P_fusion: Generated fusion power (5*P_alpha_total) [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_aux_generic: Total generic source current [A]
    Q_fusion: Fusion power gain (P_fusion / P_external_total) [dimensionless]
    P_icrh_e: Ion cyclotron resonance heating to electrons [W]
    P_icrh_i: Ion cyclotron resonance heating to ions [W]
    P_icrh_total: Total ion cyclotron resonance heating power [W]
    P_LH_high_density: H-mode transition power for high density branch [W]
    P_LH_min: Minimum H-mode transition power for at n_e_min_P_LH [W]
    P_LH: H-mode transition power from maximum of P_LH_high_density and P_LH_min
      [W]
    n_e_min_P_LH: Density corresponding to the P_LH_min [m^-3]
    E_fusion: Total cumulative fusion energy [J]
    E_aux_total: Total auxiliary heating energy absorbed by the plasma (
      time integral of P_aux_total) [J].
    E_ohmic_e: Total Ohmic heating energy to electrons (time integral of
      P_ohmic_e) [J].
    E_external_injected: Total external injected energy before absorption (
      time integral of P_external_injected) [J].
    E_external_total: Total external energy absorbed by the plasma (
      time integral of P_external_total) [J].
    T_e_volume_avg: Volume average electron temperature [keV]
    T_i_volume_avg: Volume average ion temperature [keV]
    n_e_volume_avg: Volume average electron density [m^-3]
    n_e_volume_avg: Volume average electron density [m^-3]
    n_i_volume_avg: Volume average main ion density [m^-3]
    n_e_line_avg: Line averaged electron density [m^-3]
    n_i_line_avg: Line averaged main ion density [m^-3]
    fgw_n_e_volume_avg: Greenwald fraction from volume-averaged electron density
      [dimensionless]
    fgw_n_e_line_avg: Greenwald fraction from line-averaged electron density
      [dimensionless]
    q95: q at 95% of the normalized poloidal flux
    W_pol: Total magnetic energy [J]
    li3: Normalized plasma internal inductance, ITER convention [dimensionless]
    dW_thermal_dt: Time derivative of the total stored thermal energy [W]
    q_min: Minimum q value
    rho_q_min: rho_norm at the minimum q
    rho_q_3_2_first: First outermost rho_norm value that intercepts the q=3/2
      plane. If no intercept is found, set to -inf.
    rho_q_2_1_first: First outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_first: First outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_2_second: Second outermost rho_norm value that intercepts the q=3/2
      plane. If no intercept is found, set to -inf.
    rho_q_2_1_second: Second outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_second: Second outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    I_bootstrap: Total bootstrap current [A].
    j_external: Total current density from psi sources which are external to the
      plasma (aka not bootstrap) [A m^-2]
    j_ohmic: Ohmic current density [A/m^2]
    S_gas_puff: Integrated gas puff source [s^-1]
    S_pellet: Integrated pellet source [s^-1]
    S_generic_particle: Integrated generic particle source [s^-1]
    S_total: Total integrated particle sources [s^-1]
    beta_tor: Volume-averaged toroidal plasma beta (thermal) [dimensionless]
    beta_pol: Volume-averaged poloidal plasma beta (thermal) [dimensionless]
    beta_N: Normalized toroidal plasma beta (thermal) [dimensionless].
    impurity_species: Dictionary of outputs for each impurity species.
  """

    pressure_thermal_i: cell_variable.CellVariable
    pressure_thermal_e: cell_variable.CellVariable
    pressure_thermal_total: cell_variable.CellVariable
    pprime: array_typing.FloatVector
    # pylint: disable=invalid-name
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
    # Integrated heat sources
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

    # pylint: enable=invalid-name

    def check_for_errors(self):
        return state.SimError.NO_ERROR


# TODO(b/376010694): use the various SOURCE_NAMES for the keys.
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
    """Integrates a source profile if it exists, otherwise returns 0.0."""
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
    """Calculates total integrated internal and external source power and current.

  Args:
    geo: Magnetic geometry
    core_profiles: Kinetic profiles such as temperature and density
    core_sources: Internal and external sources
    runtime_params: Runtime parameters slice for the current time step

  Returns:
    Dictionary with integrated quantities for all existing sources.
    See PostProcessedOutputs for the full list of keys.

    Output dict used to update the `PostProcessedOutputs` object in the
    output `ToraxSimState`. Sources that don't exist do not have integrated
    quantities included in the returned dict. The corresponding
    `PostProcessedOutputs` attributes remain at their initialized zero values.
  """
    integrated = {}

    # Initialize total alpha power to zero. Needed for Q calculation.
    integrated['P_alpha_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

    # Initialize total particle sources to zero.
    integrated['S_total'] = jnp.array(0.0, dtype=jax_utils.get_dtype())

    # electron-ion heat exchange always exists, and is not in
    # core_sources.profiles, so we calculate it here.
    qei = core_sources.qei.qei_coef * (core_profiles.T_e.value -
                                       core_profiles.T_i.value)
    integrated['P_ei_exchange_i'] = math_utils.volume_integration(qei, geo)
    integrated['P_ei_exchange_e'] = -integrated['P_ei_exchange_i']

    # Initialize total electron and ion powers
    # TODO(b/380848256): P_sol is now correct for stationary state. However,
    # for generality need to add dWth/dt to the equation (time dependence of
    # stored energy).
    integrated['P_SOL_i'] = integrated['P_ei_exchange_i']
    integrated['P_SOL_e'] = integrated['P_ei_exchange_e']
    integrated['P_aux_i'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_aux_e'] = jnp.array(0.0, dtype=jax_utils.get_dtype())
    integrated['P_external_injected'] = jnp.array(0.0,
                                                  dtype=jax_utils.get_dtype())

    # Calculate integrated sources with convenient names, transformed from
    # TORAX internal names.
    for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        is_in_T_i = key in core_sources.T_i
        is_in_T_e = key in core_sources.T_e
        if is_in_T_i != is_in_T_e:
            raise ValueError(
                f"Source '{key}' is expected to be defined for both ion and electron"
                ' channels (in core_sources.T_i and core_sources.T_e respectively).'
                f' Found in T_i: {is_in_T_i}, Found in T_e: {is_in_T_e}.')
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

            # Track injected power for heating sources that have absorption_fraction
            # These are only for sources like ICRH or NBI that are
            # ion_el_heat_sources.
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
        if key in core_sources.T_e and key in core_sources.T_i:
            raise ValueError(
                f"Source '{key}' was expected to only be an electron heat source (in"
                ' core_sources.T_e), but it was also found in ion heat sources'
                ' (core_sources.T_i).')
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
    """Calculates post-processed outputs based on the latest state.

  Args:
    sim_state: The state to add outputs to.
    runtime_params: Runtime parameters slice for the current time step, needed
      for calculating integrated power.
    previous_post_processed_outputs: The previous outputs, used to calculate
      cumulative quantities. If None, then cumulative quantities are set at the
      initialized values in sim_state itself. This is used for the first time
      step of a the simulation. The initialized values are zero for a clean
      simulation, or the last value of the previous simulation for a restarted
      simulation.

  Returns:
    post_processed_outputs: The post_processed_outputs for the given state.
  """
    # TODO(b/444380540): Remove once aux outputs from sources are exposed.
    impurity_radiation_outputs = (calculate_impurity_species_output(
        sim_state, runtime_params))

    (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    ) = formulas.calculate_pressure(sim_state.core_profiles)
    pprime_face = formulas.calc_pprime(sim_state.core_profiles)
    # pylint: disable=invalid-name
    W_thermal_el, W_thermal_ion, W_thermal_tot = (
        formulas.calculate_stored_thermal_energy(
            pressure_thermal_el,
            pressure_thermal_ion,
            pressure_thermal_tot,
            sim_state.geometry,
        ))
    FFprime_face = formulas.calc_FFprime(sim_state.core_profiles,
                                         sim_state.geometry)
    # Calculate normalized poloidal flux.
    psi_face = sim_state.core_profiles.psi.face_value()
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
    integrated_sources = _calculate_integrated_sources(
        sim_state.geometry,
        sim_state.core_profiles,
        sim_state.core_sources,
        runtime_params,
    )
    # Calculate fusion gain with a zero division guard.
    Q_fusion = (
        integrated_sources['P_fusion'] /
        (integrated_sources['P_external_total'] + constants.CONSTANTS.eps))

    P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH = (
        scaling_laws.calculate_plh_scaling_factor(sim_state.geometry,
                                                  sim_state.core_profiles))

    # Thermal energy confinement time is the stored energy divided by the total
    # input power into the plasma.

    # Ploss term here does not include the reduction of radiated power. Most
    # analysis of confinement times from databases have not included this term.
    # Therefore highly radiative scenarios can lead to skewed results.

    Ploss = (
        integrated_sources['P_alpha_total'] +
        integrated_sources['P_aux_total'] + integrated_sources['P_ohmic_e'] +
        constants.CONSTANTS.eps  # Division guard.
    )

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

    # Calculate total external (injected) and fusion (generated) energies based on
    # interval average.
    if previous_post_processed_outputs is not None:
        # Factor 5 due to including neutron energy: E_fusion = 5.0 * E_alpha
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
        # Used during initiailization. Note for restarted simulations this should
        # be overwritten by the previous_post_processed_outputs.
        E_fusion = 0.0
        E_aux_total = 0.0
        E_ohmic_e = 0.0
        E_external_injected = 0.0
        E_external_total = 0.0

    # Calculate q at 95% of the normalized poloidal flux
    q95 = psi_calculations.calc_q95(psi_norm_face,
                                    sim_state.core_profiles.q_face)

    # Calculate te and ti volume average [keV]
    te_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_e.value, sim_state.geometry)
    ti_volume_avg = math_utils.volume_average(
        sim_state.core_profiles.T_i.value, sim_state.geometry)

    # Calculate n_e and n_i (main ion) volume and line averages in m^-3
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
    """Constructs an xarray dictionary for impurity radiation output."""

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


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
    """Helper container for holding ion attributes."""

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
    """Gets initial and/or prescribed ion temperature profiles."""
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
    """Gets initial and/or prescribed electron temperature profiles."""
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
    """Gets initial and/or prescribed electron density profiles."""

    # Greenwald density in m^-3.
    # Ip in MA. a_minor in m.
    nGW = (
        profile_conditions_params.Ip / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2) * 1e20)
    n_e_value = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.n_e * nGW,
        profile_conditions_params.n_e,
    )
    # Calculate n_e_right_bc.
    n_e_right_bc = jnp.where(
        profile_conditions_params.n_e_right_bc_is_fGW,
        profile_conditions_params.n_e_right_bc * nGW,
        profile_conditions_params.n_e_right_bc,
    )

    if profile_conditions_params.normalize_n_e_to_nbar:
        face_left = n_e_value[
            0]  # Zero gradient boundary condition at left face.
        face_right = n_e_right_bc
        face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
        n_e_face = jnp.concatenate(
            [face_left[None], face_inner, face_right[None]], )
        # Find normalization factor such that desired line-averaged n_e is set.
        # Line-averaged electron density (nbar) is poorly defined. In general, the
        # definition is machine-dependent and even shot-dependent since it depends
        # on the usage of a specific interferometry chord. Furthermore, even if we
        # knew the specific chord used, its calculation would depend on magnetic
        # geometry information beyond what is available in StandardGeometry.
        # In lieu of a better solution, we use line-averaged electron density
        # defined on the outer midplane.
        a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
        # find target nbar in absolute units
        target_nbar = jnp.where(
            profile_conditions_params.n_e_nbar_is_fGW,
            profile_conditions_params.nbar * nGW,
            profile_conditions_params.nbar,
        )
        if not profile_conditions_params.n_e_right_bc_is_absolute:
            # In this case, n_e_right_bc is taken from n_e and we also normalize it.
            C = target_nbar / (_trapz(n_e_face, geo.R_out_face) / a_minor_out)
            n_e_right_bc = C * n_e_right_bc
        else:
            # If n_e_right_bc is absolute, subtract off contribution from outer
            # face to get C we need to multiply the inner values with.
            nbar_from_n_e_face_inner = (
                _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / a_minor_out)

            dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

            C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
                nbar_from_n_e_face_inner +
                0.5 * n_e_face[-2] * dr_edge / a_minor_out)
    else:
        C = 1

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
    """Helper container for holding ion calculation outputs."""

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
    """Calculates ion properties when impurity content is defined by fractions."""

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


def _get_ion_properties_from_n_e_ratios(
    impurity_symbols: tuple[str, ...],
    impurity_params,
    T_e: cell_variable.CellVariable,
    Z_i: array_typing.FloatVectorCell,
    Z_i_face: array_typing.FloatVectorFace,
) -> _IonProperties:
    """Calculates ion properties when impurity content is defined by n_e ratios."""
    average_charge_state = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.value,
        fractions=impurity_params.fractions,
        Z_override=impurity_params.Z_override,
    )
    average_charge_state_face = charge_states.get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
        Z_override=impurity_params.Z_override,
    )
    Z_impurity = average_charge_state.Z_mixture
    Z_impurity_face = average_charge_state_face.Z_mixture
    dilution_factor = (1 - jnp.sum(
        average_charge_state.Z_per_species * impurity_params.n_e_ratios,
        axis=0,
    ) / Z_i)
    dilution_factor_edge = (1 - jnp.sum(
        average_charge_state_face.Z_per_species[:, -1] *
        impurity_params.n_e_ratios_face[:, -1], ) / Z_i_face[-1])
    Z_eff = dilution_factor * Z_i**2 + jnp.sum(
        average_charge_state.Z_per_species**2 * impurity_params.n_e_ratios,
        axis=0,
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


# jitted since also used outside the solver
@jax_utils.jit
def get_updated_ions(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    n_e: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
) -> Ions:
    """Updated ion density, charge state, and mass based on state and config.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Z_eff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Z_eff = (Z_i**2 * n_i + Z_impurity**2 * n_impurity)/n_e  ;
  n_impurity*Z_impurity + n_i*Z_i = n_e

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry of the tokamak.
    n_e: Electron density profile [m^-3].
    T_e: Electron temperature profile [keV].

  Returns:
    Ion container with the following attributes:
      n_i: Ion density profile [m^-3].
      n_impurity: Impurity density profile [m^-3].
      Z_i: Average charge state of main ion on cell grid [dimensionless].
        Typically just the average of the atomic numbers since these are
        normally low Z ions and can be assumed to be fully ionized.
      Z_i_face: Average charge state of main ion on face grid [dimensionless].
      Z_impurity: Average charge state of impurities on cell grid
      [dimensionless].
      Z_impurity_face: Average charge state of impurities on face grid
      [dimensionless].
      A_i: Average atomic number of main ion [amu].
      A_impurity: Average atomic number of impurities on cell grid [amu].
      A_impurity_face: Average atomic number of impurities on face grid [amu].
  """

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

        case electron_density_ratios.RuntimeParams():
            ion_properties = _get_ion_properties_from_n_e_ratios(
                runtime_params.plasma_composition.impurity_names,
                impurity_params,
                T_e,
                Z_i,
                Z_i_face,
            )
        case electron_density_ratios_zeff.RuntimeParams():
            ion_properties = _get_ion_properties_from_n_e_ratios_Z_eff(
                impurity_params,
                T_e,
                Z_i,
                Z_i_face,
                runtime_params.plasma_composition.Z_eff,
                runtime_params.plasma_composition.Z_eff_face,
            )
        case _:
            # Not expected to be reached but needed to avoid linter errors.
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

    # Z_eff from plasma composition is imposed and can be passed to CoreProfiles.
    # However, we must recalculate Z_eff_face from the updated densities and
    # charge states since linearly interpolated Z_eff (which is what plasma
    # composition Z_eff_face is) would not be physically consistent.
    Z_eff_face = _calculate_Z_eff(
        Z_i_face,
        ion_properties.Z_impurity_face,
        n_i.face_value(),
        n_impurity.face_value(),
        n_e.face_value(),
    )

    # Convert array of fractions to a mapping from symbol to fraction profile.
    # Ensure that output is always a full radial profile for consistency across
    # all impurity modes.
    impurity_fractions_dict = {}
    for i, symbol in enumerate(
            runtime_params.plasma_composition.impurity_names):
        fraction = ion_properties.impurity_fractions[i]
        if fraction.ndim == 0:
            impurity_fractions_dict[symbol] = jnp.full_like(
                n_e.value, fraction)
        else:
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


def _calculate_Z_eff(
    Z_i: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
    n_i: array_typing.FloatVector,
    n_impurity: array_typing.FloatVector,
    n_e: array_typing.FloatVector,
) -> array_typing.FloatVector:
    """Calculates Z_eff based on single effective impurity and main_ion."""
    return (Z_i**2 * n_i + Z_impurity**2 * n_impurity) / n_e


def initial_core_profiles0(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
    """Calculates the initial core profiles.

  Args:
    runtime_params: Runtime parameters at t=t_initial.
    geo: Torus geometry at t=t_initial.
    source_models: All models for TORAX sources/sinks.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Initial core profiles.
  """
    T_i = get_updated_ion_temperature(runtime_params.profile_conditions, geo)
    T_e = get_updated_electron_temperature(runtime_params.profile_conditions,
                                           geo)
    n_e = get_updated_electron_density(runtime_params.profile_conditions, geo)
    ions = get_updated_ions(runtime_params, geo, n_e, T_e)

    # Set v_loop_lcfs. Two branches:
    # 1. Set the v_loop_lcfs from profile_conditions if using the v_loop BC option
    # 2. Initialize v_loop_lcfs to 0 if using the Ip boundary condition for psi.
    # In case 2, v_loop_lcfs will be updated every timestep based on the psi_lcfs
    # values across the time interval. Since there is is one more time value than
    # time intervals, the v_loop_lcfs time-series is underconstrained. Therefore,
    # we set v_loop_lcfs[0] to v_loop_lcfs[1] when creating the outputs.
    v_loop_lcfs = (
        np.array(runtime_params.profile_conditions.v_loop_lcfs)
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else np.array(0.0, dtype=jax_utils.get_dtype()))

    # Initialise psi and derived quantities to zero before they are calculated.
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
    """Returns the initial psi mode based on the runtime parameters.

  This allows us to support the legacy behavior of initial_psi_from_j, which
  is only available when using the standard geometry and initial psi is not
  provided. Moving forward the initial_psi_mode setting in the profile
  conditions should be preferred.

  Args:
    runtime_params: Runtime parameters.
    geo: Torus geometry.

  Returns:
    How to calculate the initial psi value.
  """
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
    """Initialises psi and currents in core profiles.

  There are three modes of doing this that are supported:
    1. Retrieving psi from the profile conditions.
    2. Retrieving psi from the standard geometry input.
    3. Calculating j according to the nu formula and then
    calculating psi from that. As we are calculating j using a guess for psi,
    this method is iterated to converge to the true psi.

  Args:
    runtime_params: Runtime parameters.
    geo: Torus geometry.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Refined core profiles.
  """

    # Flag to track if sources have been calculated during psi initialization.
    sources_are_calculated = False

    # Initialize psi source profiles and bootstrap current to all zeros.
    source_profiles = source_profile_builders.build_all_zero_profiles(geo)

    initial_psi_mode = _get_initial_psi_mode(runtime_params, geo)

    match initial_psi_mode:
    # Case 1: retrieving psi from the profile conditions, using the prescribed
    # profile and Ip
        case profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS:
            if runtime_params.profile_conditions.psi is None:
                raise ValueError(
                    'psi is None, but initial_psi_mode is PROFILE_CONDITIONS.')
            # Calculate the dpsi/drho necessary to achieve the given Ip
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))

            # Set the BCs to ensure the correct Ip
            if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition:
                # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
                # to achieve the desired Ip
                right_face_grad_constraint = None
                right_face_constraint = (
                    runtime_params.profile_conditions.psi[-1] +
                    dpsi_drhonorm_edge * geo.drho_norm / 2)
            else:
                # Use the dpsi/drho calculated above as the right face gradient
                # constraint
                right_face_grad_constraint = dpsi_drhonorm_edge
                right_face_constraint = None

            psi = cell_variable.CellVariable(
                value=runtime_params.profile_conditions.psi,
                right_face_grad_constraint=right_face_grad_constraint,
                right_face_constraint=right_face_constraint,
                dr=geo.drho_norm,
            )

        # Case 2: retrieving psi from the standard geometry input.
        case profile_conditions_lib.InitialPsiMode.GEOMETRY:
            if not isinstance(geo, standard_geometry.StandardGeometry):
                raise ValueError(
                    'GEOMETRY initial_psi_source is only supported for standard'
                    ' geometry.')
            # psi is already provided from a numerical equilibrium, so no need to
            # first calculate currents.
            dpsi_drhonorm_edge = (
                psi_calculations.calculate_psi_grad_constraint_from_Ip(
                    runtime_params.profile_conditions.Ip,
                    geo,
                ))
            # Use the psi from the equilibrium as the right face constraint
            # This has already been made consistent with the desired Ip
            # by make_ip_consistent
            psi = cell_variable.CellVariable(
                value=geo.psi_from_Ip,  # Use psi from equilibrium
                right_face_grad_constraint=None
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else dpsi_drhonorm_edge,
                right_face_constraint=geo.psi_from_Ip_face[-1]
                if runtime_params.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None,
                dr=geo.drho_norm,
            )

        # Case 3: calculating j according to nu formula and psi from j.
        case profile_conditions_lib.InitialPsiMode.J:
            # calculate j and psi from the nu formula
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
                # In this branch we require non-inductive currents to determine j_total.
                # The nu formula only provides the Ohmic component of the current.
                # However calculating non-inductive currents requires a non-zero psi.
                # We thus iterate between psi and source calculations, using j_total
                # and psi calculated purely with the nu formula as an initial guess

                # Initialize iterations
                core_profiles_initial = dataclasses.replace(
                    core_profiles,
                    psi=psi,
                    q_face=psi_calculations.calc_q_face(geo, psi),
                    s_face=psi_calculations.calc_s_face(geo, psi),
                )

                # TODO(b/440385263): add tunable iteration number or convergence
                # criteria, and modify python for loop to jax fori loop for the general
                # case.

                # Iterate with non-inductive current source calculations. Stop after 2.
                psi, source_profiles = _iterate_psi_and_sources(
                    runtime_params=runtime_params,
                    geo=geo,
                    core_profiles=core_profiles_initial,
                    neoclassical_models=neoclassical_models,
                    source_models=source_models,
                    source_profiles=source_profiles,
                    iterations=2,
                )

                # Mark that sources have been calculated to avoid redundant work.
                sources_are_calculated = True

    # Conclude with completing core_profiles with all psi-dependent profiles.
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
    """Supplements core profiles with all other profiles that depend on psi."""
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
    # Calculate conductivity once we have a consistent set of core profiles
    conductivity = neoclassical_models.conductivity.calculate_conductivity(
        geo,
        core_profiles,
    )

    # Calculate sources if they have not already been calculated.
    if not sources_are_calculated:
        source_profiles = _get_bootstrap_and_standard_source_profiles(
            runtime_params,
            geo,
            core_profiles,
            neoclassical_models,
            source_models,
            source_profiles,
        )

    # psidot calculated here with phibdot=0 in geo, since this is initial
    # conditions and we don't yet have information on geo_t_plus_dt for the
    # phibdot calculation.

    if (not runtime_params.numerics.evolve_current
            and runtime_params.profile_conditions.psidot is not None):
        # If psidot is prescribed and psi does not evolve, use prescribed value
        psidot_value = runtime_params.profile_conditions.psidot
    else:
        # Otherwise, calculate psidot from psi sources.
        psi_sources = source_profiles.total_psi_sources(geo)
        psidot_value = psi_calculations.calculate_psidot_from_psi_sources(
            psi_sources=psi_sources,
            sigma=conductivity.sigma,
            resistivity_multiplier=runtime_params.numerics.
            resistivity_multiplier,
            psi=psi,
            geo=geo,
        )

    # psidot boundary condition. If v_loop_lcfs is not prescribed then we set it
    # to the last calculated psidot for the initialisation since we have no
    # other information.
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
    """Calculates bootstrap current and updates source profiles."""
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
    """Converts evolving parts of CoreProfiles to the 'x' tuple for the solver.

  State variables in the solver are scaled for solver numerical conditioning.
  i.e., the solver methods find the zero of a residual, minimizes a loss, and/or
  invert a linear system with respect to the state vector x, which is a
  concetenated vector of the x_tuple values. It is important that the solution
  state vector elements are of similar order of magnitude such that e.g. scalars
  related to residual or loss minimizations have similar contributions from the
  various state vector components.

  Args:
    core_profiles: The input CoreProfiles object.
    evolving_names: Tuple of strings naming the variables to be evolved by the
      solver.

  Returns:
    A tuple of CellVariable objects, one for each name in evolving_names,
    with density values appropriately scaled for the solver.
  """
    x_tuple_for_solver_list = []

    for name in evolving_names:
        original_units_cv = getattr(core_profiles, name)
        # Scale for solver (divide by scaling factor)
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
    """Gets updated cell variables for evolving state variables in core_profiles.

  If a variable is in `evolving_names`, its new value is taken from `x_new`.
  Otherwise, the existing value from `core_profiles` is kept.
  State variables in the solver may be scaled for solver numerical conditioning,
  and must be scaled back to their original units before being written to
  `core_profiles`.

  Args:
    x_new: The new values of the evolving variables.
    evolving_names: The names of the evolving variables.
    core_profiles: The current set of core plasma profiles.

  Returns:
    An updated CoreProfiles object with the new values.
  """
    updated_vars = {}

    for i, var_name in enumerate(evolving_names):
        solver_x_tuple_cv = x_new[i]
        # Unscale from solver (multiply by scaling factor)
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
    """Scales or unscales a CellVariable's relevant fields.

  Args:
    cv: The CellVariable to scale.
    scaling_factor: The factor to scale values and boundary conditions by.

  Returns:
    A new CellVariable with scaled or unscaled values.
  """
    operation = lambda x, factor: x * factor if x is not None else None

    scaled_value = operation(cv.value, scaling_factor)

    # Only scale constraints if they are not None
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


# An optional argument, consisting of a 2D matrix of nested tuples, with each
# leaf being either None or a JAX Array. Used to define block matrices.
# examples:
#
# ((a, b), (c, d)) where a, b, c, d are each jax.Array
#
# ((a, None), (None, d)) : represents a diagonal block matrix
OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...],
                                       ...] | None

# Alias for better readability.
AuxiliaryOutput: TypeAlias = Any

# pylint: disable=invalid-name


def _calculate_psi_value_constraint_from_v_loop(
    dt: array_typing.FloatScalar,
    theta: array_typing.FloatScalar,
    v_loop_lcfs_t: array_typing.FloatScalar,
    v_loop_lcfs_t_plus_dt: array_typing.FloatScalar,
    psi_lcfs_t: array_typing.FloatScalar,
) -> jax.Array:
    """Calculates the value constraint on the poloidal flux for the next time step from loop voltage."""
    theta_weighted_v_loop_lcfs = (
        1 - theta) * v_loop_lcfs_t + theta * v_loop_lcfs_t_plus_dt
    return psi_lcfs_t + theta_weighted_v_loop_lcfs * dt


@jax_utils.jit
def get_prescribed_core_profile_values(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> dict[str, array_typing.FloatVector]:
    """Updates core profiles which are not being evolved by PDE.

  Uses same functions as for profile initialization.

  Args:
    runtime_params: Runtime parameters at t=t_initial.
    geo: Torus geometry.
    core_profiles: Core profiles dataclass to be updated

  Returns:
    Updated core profiles values on the cell grid.
  """
    # If profiles are not evolved, they can still potential be time-evolving,
    # depending on the runtime params. If so, they are updated below.
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
    """Returns the new core profiles after updating the evolving variables.

  Intended for use during iterative solves in the step function. Only updates
  the core profiles which are being evolved by the PDE and directly derivable
  quantities like q_face, s_face. core_profile calculations which require
  sources are not updated.

  Args:
    x_new: The new values of the evolving variables.
    runtime_params: The runtime params slice.
    geo: Magnetic geometry.
    core_profiles: The old set of core plasma profiles.
    evolving_names: The names of the evolving variables.
  """

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
    """Returns a core profiles and source profiles after the solver has finished.

  Updates the evolved variables and derived variables like q_face, psidot, etc.

  Args:
    dt: The size of the last timestep.
    x_new: The new values of the evolving variables.
    runtime_params_t_plus_dt: The runtime params slice at t=t_plus_dt.
    geo: Magnetic geometry.
    core_profiles_t: The old set of core plasma profiles.
    core_profiles_t_plus_dt: The partially new set of core plasma profiles. On
      input into this function, all prescribed profiles and used boundary
      conditions are already set. But evolving values are not.
    explicit_source_profiles: The explicit source profiles.
    source_models: The source models.
    neoclassical_models: The neoclassical models.
    evolving_names: The names of the evolving variables.

  Returns:
    A tuple of the new core profiles and the source profiles.
  """

    updated_core_profiles_t_plus_dt = solver_x_tuple_to_core_profiles(
        x_new, evolving_names, core_profiles_t_plus_dt)

    ions = get_updated_ions(
        runtime_params_t_plus_dt,
        geo,
        updated_core_profiles_t_plus_dt.n_e,
        updated_core_profiles_t_plus_dt.T_e,
    )

    v_loop_lcfs = (
        runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs  # pylint: disable=g-long-ternary
        if runtime_params_t_plus_dt.profile_conditions.
        use_v_loop_lcfs_boundary_condition else _update_v_loop_lcfs_from_psi(
            core_profiles_t.psi,
            updated_core_profiles_t_plus_dt.psi,
            dt,
        ))

    j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
        geo,
        updated_core_profiles_t_plus_dt.psi,
    )

    # A wholly new core profiles object is defined as a guard against neglecting
    # to update one of the attributes if doing dataclasses.replace
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
        sigma=core_profiles_t_plus_dt.sigma,  # Not yet updated
        sigma_face=core_profiles_t_plus_dt.sigma_face,  # Not yet updated
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

    # build_source_profiles calculates the union with explicit + implicit
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
        # If psidot is prescribed and current does not evolve, use prescribed value
        psidot_value = (runtime_params_t_plus_dt.profile_conditions.psidot)
    else:
        # Otherwise, calculate psidot from psi sources.
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
    """Computes boundary conditions for the next timestep and returns updates to State.

  Args:
    dt: Size of the next timestep
    runtime_params_t: Runtime parameters for the current timestep. Will not be
      used if runtime_params_t.profile_conditions.v_loop_lcfs is
      None, i.e. if the dirichlet psi boundary condition based on Ip is used
    runtime_params_t_plus_dt: Runtime parameters for the next timestep
    geo_t_plus_dt: Geometry object for the next timestep
    core_profiles_t: Core profiles at the current timestep. Will not be used if
      runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs is
      None, i.e. if the dirichlet psi boundary condition based on Ip is used

  Returns:
    Mapping from State attribute names to dictionaries updating attributes of
    each CellVariable in the state. This dict can in theory recursively replace
    values in a State object.
  """
    profile_conditions_t_plus_dt = (
        runtime_params_t_plus_dt.profile_conditions)
    # TODO(b/390143606): Separate out the boundary condition calculation from the
    # core profile calculation.
    n_e = get_updated_electron_density(profile_conditions_t_plus_dt,
                                       geo_t_plus_dt)
    n_e_right_bc = n_e.right_face_constraint

    # Used for edge calculations and input arguments have correct edge BCs.
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
                psi_calculations.calculate_psi_grad_constraint_from_Ip(  # pylint: disable=g-long-ternary
                    Ip=profile_conditions_t_plus_dt.Ip,
                    geo=geo_t_plus_dt,
                ) if not runtime_params_t.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None),
            right_face_constraint=(
                _calculate_psi_value_constraint_from_v_loop(  # pylint: disable=g-long-ternary
                    dt=dt,
                    v_loop_lcfs_t=runtime_params_t.profile_conditions.
                    v_loop_lcfs,
                    v_loop_lcfs_t_plus_dt=profile_conditions_t_plus_dt.
                    v_loop_lcfs,
                    psi_lcfs_t=core_profiles_t.psi.right_face_constraint,
                    theta=runtime_params_t.solver.theta_implicit,
                ) if runtime_params_t.profile_conditions.
                use_v_loop_lcfs_boundary_condition else None),
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
    """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
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

    # pylint: disable=invalid-name
    # Update Z_face with boundary condition Z, needed for cases where T_e
    # is evolving and updated_prescribed_core_profiles is a no-op.
    Z_i_face = jnp.concatenate([
        updated_values['Z_i_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_i_edge']]),
    ], )
    Z_impurity_face = jnp.concatenate([
        updated_values['Z_impurity_face'][:-1],
        jnp.array([updated_boundary_conditions['Z_impurity_edge']]),
    ], )
    # pylint: enable=invalid-name
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


# TODO(b/406173731): Find robust solution for underdetermination and solve this
# for general theta_implicit values.
def _update_v_loop_lcfs_from_psi(
    psi_t: cell_variable.CellVariable,
    psi_t_plus_dt: cell_variable.CellVariable,
    dt: array_typing.FloatScalar,
) -> jax.Array:
    """Updates the v_loop_lcfs for the next timestep.

  For the Ip boundary condition case, the v_loop_lcfs formula is in principle
  calculated from:

  (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt =
    v_loop_lcfs_t_plus_dt*theta_implicit - v_loop_lcfs_t*(1-theta_implicit)

  However this set of equations is underdetermined. We thus restrict this
  calculation assuming a fully implicit system, i.e. theta_implicit=1, which is
  the TORAX default. Be cautious when interpreting the results of this function
  when theta_implicit != 1 (non-standard usage).

  Args:
    psi_t: The psi CellVariable at the beginning of the timestep interval.
    psi_t_plus_dt: The updated psi CellVariable for the next timestep.
    dt: The size of the last timestep.

  Returns:
    The updated v_loop_lcfs for the next timestep.
  """
    psi_lcfs_t = psi_t.face_value()[-1]
    psi_lcfs_t_plus_dt = psi_t_plus_dt.face_value()[-1]
    v_loop_lcfs_t_plus_dt = (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt
    return v_loop_lcfs_t_plus_dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Block1DCoeffs:
    # pyformat: disable  # pyformat removes line breaks needed for readability
    """The coefficients of coupled 1D fluid dynamics PDEs.

  The differential equation is:
  transient_out_coeff partial x transient_in_coeff / partial t = F
  where F =
  divergence(diffusion_coeff * grad(x))
  - divergence(convection_coeff * x)
  + source_mat_coeffs * u
  + sources.

  source_mat_coeffs exists for specific classes of sources where this
  decomposition is valid, allowing x to be treated implicitly in linear solvers,
  even if source_mat_coeffs contains state-dependent terms

  This class captures a snapshot of the coefficients of the equation at one
  instant in time, discretized spatially across a mesh.

  This class imposes the following structure on the problem:
  - It assumes the variables are arranged on a 1-D, evenly spaced grid.
  - It assumes the x variable is broken up into "channels," so the resulting
  matrix equation has one block per channel.

  Attributes:
    transient_out_cell: Tuple with one entry per channel, transient_out_cell[i]
      gives the transient coefficients outside the time derivative for channel i
      on the cell grid.
    transient_in_cell: Tuple with one entry per channel, transient_in_cell[i]
      gives the transient coefficients inside the time derivative for channel i
      on the cell grid.
    d_face: Tuple, with d_face[i] containing diffusion term coefficients for
      channel i on the face grid.
    v_face: Tuple, with v_face[i] containing convection term coefficients for
      channel i on the face grid.
    source_mat_cell: 2-D matrix of Tuples, with source_mat_cell[i][j] adding to
      block-row i a term of the form source_cell[j] * u[channel j]. Depending on
      the source runtime_params, may be constant values for a timestep, or
      updated iteratively with new states in a nonlinear solver
    source_cell: Additional source terms on the cell grid for each channel.
      Depending on the source runtime_params, may be constant values for a
      timestep, or updated iteratively with new states in a nonlinear solver
    auxiliary_outputs: Optional extra output which can include auxiliary state
      or information useful for inspecting the computation inside the callback
      which calculated these coeffs.
  """
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
    """Converts a tuple of CellVariables to a flat array.

  Args:
    x_tuple: A tuple of CellVariables.

  Returns:
    A flat array of evolving state variables.
  """
    x_vec = jnp.concatenate([x.value for x in x_tuple])
    return x_vec


# pylint: disable=invalid-name
class CoeffsCallback:
    """Calculates Block1DCoeffs for a state."""

    def __init__(
        self,
        physics_models: PhysicsModels,
        evolving_names: tuple[str, ...],
    ):
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
        # Checks if reduced calc_coeffs for explicit terms when theta_implicit=1
        # should be called
        explicit_call: bool = False,
    ):
        """Returns coefficients given a state x.

    Used to calculate the coefficients for the implicit or explicit components
    of the PDE system.

    Args:
      runtime_params: Runtime configuration parameters. These values are
        potentially time-dependent and should correspond to the time step of the
        state x.
      geo: The geometry of the system at this time step.
      core_profiles: The core profiles of the system at this time step.
      x: The state with cell-grid values of the evolving variables.
      explicit_source_profiles: Precomputed explicit source profiles. These
        profiles were configured to always depend on state and parameters at
        time t during the solver step. They can thus be inputs, since they are
        not recalculated at time t+plus_dt with updated state during the solver
        iterations. For sources that are implicit, their explicit profiles are
        set to all zeros.
      allow_pereverzev: If True, then the coeffs are being called within a
        linear solver. Thus could be either the use_predictor_corrector solver
        or as part of calculating the initial guess for the nonlinear solver. In
        either case, we allow the inclusion of Pereverzev-Corrigan terms which
        aim to stabilize the linear solver when being used with highly nonlinear
        (stiff) transport coefficients. The nonlinear solver solves the system
        more rigorously and Pereverzev-Corrigan terms are not needed.
      explicit_call: If True, then if theta_implicit=1, only a reduced
        Block1DCoeffs is calculated since most explicit coefficients will not be
        used.

    Returns:
      coeffs: The diffusion, convection, etc. coefficients for this state.
    """

        # Update core_profiles with the subset of new values of evolving variables
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
    """Adds Pereverzev-Corrigan flux to diffusion terms."""

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

    # remove Pereverzev flux from boundary region if pedestal model is on
    # (for PDE stability)
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

    # set heat convection terms to zero out Pereverzev-Corrigan heat diffusion
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


def calc_coeffs(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
):
    """Calculates Block1DCoeffs for the time step described by `core_profiles`.

  Args:
    runtime_params: General input parameters that can change from time step to
      time step or simulation run to run, and do so without triggering a
      recompile.
    geo: Geometry describing the torus.
    core_profiles: Core plasma profiles for this time step during this iteration
      of the solver. Depending on the type of solver being used, this may or may
      not be equal to the original plasma profiles at the beginning of the time
      step.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the core profiles or depend on the
      original core profiles at the start of the time step, not the "live"
      updating core profiles. For sources that are implicit, their explicit
      profiles are set to all zeros.
    physics_models: The physics models to use for the simulation.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms
    explicit_call: If True, indicates that calc_coeffs is being called for the
      explicit component of the PDE. Then calculates a reduced Block1DCoeffs if
      theta_implicit=1. This saves computation for the default fully implicit
      implementation.

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

    # If we are fully implicit and we are making a call for calc_coeffs for the
    # explicit components of the PDE, only return a cheaper reduced Block1DCoeffs
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
def _calc_coeffs_full(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
):
    """See `calc_coeffs` for details."""

    consts = constants.CONSTANTS

    pedestal_model_output = physics_models.pedestal_model(
        runtime_params, geo, core_profiles)

    # Boolean mask for enforcing internal temperature boundary conditions to
    # model the pedestal.
    # If rho_norm_ped_top_idx is outside of bounds of the mesh, the pedestal is
    # not present and the mask is all False. This is what is used in the case that
    # set_pedestal is False.
    mask = (jnp.zeros_like(
        geo.rho,
        dtype=bool).at[pedestal_model_output.rho_norm_ped_top_idx].set(True))

    conductivity = (
        physics_models.neoclassical_models.conductivity.calculate_conductivity(
            geo, core_profiles))

    # Calculate the implicit source profiles and combines with the explicit
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

    # psi source terms. Source matrix is zero for all psi sources
    source_mat_psi = jnp.zeros_like(geo.rho)

    # fill source vector based on both original and updated core profiles
    source_psi = merged_source_profiles.total_psi_sources(geo)

    # Transient term coefficient vector (has radial dependence through r, n)
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

    # Diffusion term coefficients
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
    # No poloidal flux convection term
    v_face_psi = jnp.zeros_like(d_face_psi)

    # entire coefficient preceding dT/dr in heat transport equations
    full_chi_face_ion = (geo.g1_over_vpr_face *
                         core_profiles.n_i.face_value() * consts.keV_to_J *
                         chi_face_ion_total)
    full_chi_face_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                        consts.keV_to_J * chi_face_el_total)

    # entire coefficient preceding dne/dr in particle equation
    full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
    full_v_face_el = geo.g0_face * v_face_el_total

    # density source terms. Initialize source matrix to zero
    source_mat_nn = jnp.zeros_like(geo.rho)

    # density source vector based both on original and updated core profiles
    source_n_e = merged_source_profiles.total_sources('n_e', geo)

    source_n_e += (mask * runtime_params.numerics.adaptive_n_source_prefactor *
                   pedestal_model_output.n_e_ped)
    source_mat_nn += -(mask *
                       runtime_params.numerics.adaptive_n_source_prefactor)

    # Pereverzev-Corrigan correction for heat and particle transport
    # (deals with stiff nonlinearity of transport coefficients)
    # TODO(b/311653933) this forces us to include value 0
    # convection terms in discrete system, slowing compilation down by ~10%.
    # See if can improve with a different pattern.
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

    # Add Phi_b_dot terms to heat transport convection
    v_heat_face_ion += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                        geo.rho_face_norm * geo.vpr_face *
                        core_profiles.n_i.face_value() * consts.keV_to_J)

    v_heat_face_el += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face *
                       core_profiles.n_e.face_value() * consts.keV_to_J)

    # Add Phi_b_dot terms to particle transport convection
    full_v_face_el += (-1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face)

    # Fill heat transport equation sources. Initialize source matrices to zero

    source_i = merged_source_profiles.total_sources('T_i', geo)
    source_e = merged_source_profiles.total_sources('T_e', geo)

    # Add the Qei effects.
    qei = merged_source_profiles.qei
    source_mat_ii = qei.implicit_ii * geo.vpr
    source_i += qei.explicit_i * geo.vpr
    source_mat_ee = qei.implicit_ee * geo.vpr
    source_e += qei.explicit_e * geo.vpr
    source_mat_ie = qei.implicit_ie * geo.vpr
    source_mat_ei = qei.implicit_ei * geo.vpr

    # Pedestal
    source_i += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_i_ped)
    source_e += (mask * runtime_params.numerics.adaptive_T_source_prefactor *
                 pedestal_model_output.T_e_ped)

    source_mat_ii -= mask * runtime_params.numerics.adaptive_T_source_prefactor

    source_mat_ee -= mask * runtime_params.numerics.adaptive_T_source_prefactor

    # Add effective Phi_b_dot heat source terms

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

    # Add effective Phi_b_dot particle source terms
    source_n_e += (1.0 / 2.0 * d_vpr_rhon_drhon * geo.Phi_b_dot / geo.Phi_b *
                   core_profiles.n_e.value)

    # Add effective Phi_b_dot poloidal flux source term
    source_psi += (8.0 * jnp.pi**2 * consts.mu_0 * geo.Phi_b_dot * geo.Phi_b *
                   geo.rho_norm**2 * conductivity.sigma / geo.F**2 *
                   core_profiles.psi.grad())

    # Build arguments to solver based on which variables are evolving
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

    # d maps (row var, col var) to the coefficient for that block of the matrix
    # (Can't use a descriptive name or the nested comprehension to build the
    # matrix gets too long)
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

    # var_to_source ends up as a vector in the constructed PDE. Therefore any
    # scalings from CoreProfiles state variables to x must be applied here too.
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
    """Calculates only the transient_in_cell terms in Block1DCoeffs."""

    # Only transient_in_cell is used for explicit terms if theta_implicit=1
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
    """Calculate C and c such that F = C x + c.

  See docstrings for `Block1DCoeff` and `implicit_solve_block` for
  more detail.

  Args:
    x: Tuple containing CellVariables for each channel. This function uses only
      their shape and their boundary conditions, not their values.
    coeffs: Coefficients defining the differential equation.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    c_mat: matrix C, such that F = C x + c
    c: the vector c
  """

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

    # Make a matrix C and vector c that will accumulate contributions from
    # diffusion, convection, and source terms.
    # C and c are both block structured, with one block per channel.
    c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
    c = zero_block_vec.copy()

    # Add diffusion terms
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

    # Add convection terms
    if v_face is not None:
        for i in range(num_channels):
            # Resolve diffusion to zeros if it is not specified
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

    # Add implicit source terms
    if source_mat_cell is not None:
        for i in range(num_channels):
            for j in range(num_channels):
                source = source_mat_cell[i][j]
                if source is not None:
                    c_mat[i][j] += jnp.diag(source)

    # Add explicit source terms
    def add(left: jax.Array, right: jax.Array | None):
        """Addition with adding None treated as no-op."""
        if right is not None:
            return left + right
        return left

    if source_cell is not None:
        c = [add(c_i, source_i) for c_i, source_i in zip(c, source_cell)]

    # Form block structure
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


# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
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

    # Create updated CellVariable instances based on state_plus_dt which has
    # updated boundary conditions and prescribed profiles.
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
    """Input params for the solver which can be used as compiled args."""
    theta_implicit: float = dataclasses.field(metadata={'static': True})
    use_predictor_corrector: bool = dataclasses.field(
        metadata={'static': True})
    n_corrector_steps: int = dataclasses.field(metadata={'static': True})
    convection_dirichlet_mode: str = dataclasses.field(
        metadata={'static': True})
    convection_neumann_mode: str = dataclasses.field(metadata={'static': True})
    use_pereverzev: bool = dataclasses.field(metadata={'static': True})
    chi_pereverzev: float
    D_pereverzev: float  # pylint: disable=invalid-name


class Solver(abc.ABC):
    """Solves for a single time steps update to State.

  Attributes:
    physics_models: Physics models.
  """

    def __init__(
        self,
        physics_models: PhysicsModels,
    ):
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

    def loop_body(i, x_new_guess):  # pylint: disable=unused-argument
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
    """Time step update using theta method, linearized on coefficients at t."""

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
        """See Solver._x_new docstring."""

        x_old = core_profiles_to_solver_x_tuple(core_profiles_t,
                                                evolving_names)
        x_new_guess = core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt,
                                                      evolving_names)

        coeffs_callback = CoeffsCallback(
            physics_models=self.physics_models,
            evolving_names=evolving_names,
        )

        # Compute the explicit coeffs based on the core profiles at time t and all
        # runtime parameters at time t.
        coeffs_exp = coeffs_callback(
            runtime_params_t,
            geo_t,
            core_profiles_t,
            x_old,
            explicit_source_profiles=explicit_source_profiles,
            allow_pereverzev=True,
            explicit_call=True,
        )

        # Calculate x_new with the predictor corrector method. Reverts to a
        # standard linear solve if
        # runtime_params_slice.predictor_corrector=False.
        # init_val is the initialization for the predictor_corrector loop.
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
            solver_error_state=0,  # linear method always works
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
        """Builds runtime params from the config."""

    @abc.abstractmethod
    def build_solver(
        self,
        physics_models: PhysicsModels,
    ):
        """Builds a solver from the config."""


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

    def build_solver(
        self,
        physics_models: PhysicsModels,
    ):
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
    mhd: Any
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
            mhd=config.mhd,
            neoclassical=config.neoclassical,
        )

    @jax_utils.jit
    def __call__(
        self,
        t: chex.Numeric,
    ) -> runtime_params_slice.RuntimeParams:
        """Returns a runtime_params_slice.RuntimeParams to use during time t of the sim."""
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
            mhd=self.mhd.build_runtime_params(t),
        )


def get_consistent_runtime_params_and_geometry(*, t, runtime_params_provider,
                                               geometry_provider):
    geo = geometry_provider(t)
    runtime_params = runtime_params_provider(t=t)
    return runtime_params_slice.make_ip_consistent(runtime_params, geo)


TIME_INVARIANT = model_base.TIME_INVARIANT
JAX_STATIC = model_base.JAX_STATIC
BaseModelFrozen = model_base.BaseModelFrozen
TimeVaryingScalar = interpolated_param_1d.TimeVaryingScalar
TimeVaryingArray = interpolated_param_2d.TimeVaryingArray
NonNegativeTimeVaryingArray = interpolated_param_2d.NonNegativeTimeVaryingArray
PositiveTimeVaryingScalar = interpolated_param_1d.PositiveTimeVaryingScalar
UnitIntervalTimeVaryingScalar = (
    interpolated_param_1d.UnitIntervalTimeVaryingScalar)
PositiveTimeVaryingArray = interpolated_param_2d.PositiveTimeVaryingArray
ValidatedDefault = functools.partial(pydantic.Field, validate_default=True)


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
    mhd_models: MHDModels = dataclasses.field(metadata=dict(static=True))


TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'
JAX_STATIC: Final[str] = '_pydantic_jax_static_field'

StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]

# pylint: disable=invalid-name

# Dataset names.
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"

# Core profiles.
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

# Calculated or derived currents.
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"

# Core transport.
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

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_NORM = "rho_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Post processed outputs
Q_FUSION = "Q_fusion"

# Numerics.
# Simulation error state.
SIM_ERROR = "sim_error"
OUTER_SOLVER_ITERATIONS = "outer_solver_iterations"
INNER_SOLVER_ITERATIONS = "inner_solver_iterations"
# Boolean array indicating whether the state corresponds to a
# post-sawtooth-crash state.
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
        if (not torax_config.restart and not torax_config.profile_conditions.
                use_v_loop_lcfs_boundary_condition
                and len(state_history) >= 2):
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
            if v is not None and v.values.ndim > 1  # pytype: disable=attribute-error
        }
        profiles = xr.Dataset(profiles_dict)
        scalars_dict = {
            k: v
            for k, v in flat_dict.items()
            if v is not None and v.values.ndim in [0, 1]  # pytype: disable=attribute-error
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

        if (self.torax_config.restart is not None
                and self.torax_config.restart.stitch):
            data_tree = stitch_state_files(self.torax_config.restart,
                                           data_tree)

        return data_tree

    def _pack_into_data_array(
        self,
        name: str,
        data: jax.Array | None,
    ) -> xr.DataArray | None:
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
                    data.shape,  # pytype: disable=attribute-error
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
                else:  # cell array with no face counterpart, or a scalar value
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
            # Remap to avoid outputting _face suffix in output.
            if field_name.endswith("_face"):
                field_name = field_name.removesuffix("_face")
            if field_name == "Ip_profile":
                # Ip_profile exists in core profiles so rename to avoid duplicate.
                field_name = "Ip_profile_from_geo"
            if field_name == "psi":
                # Psi also exists in core profiles so rename to avoid duplicate.
                field_name = "psi_from_geo"
            if field_name == "_z_magnetic_axis":
                # This logic only reached if not None. Avoid leading underscore in name.
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
            # Skip over saving any variables that are named *_face.
            if (name.endswith("_face")
                    and name.removesuffix("_face") in property_names):
                continue
            if name in EXCLUDED_GEOMETRY_NAMES:
                continue
            if isinstance(value, property):
                property_data = value.fget(self._stacked_geometry)
                # Check if there is a corresponding face variable for this property.
                # If so, extend the data to the cell+boundaries grid.
                if f"{name}_face" in property_names:
                    face_data = getattr(self._stacked_geometry, f"{name}_face")
                    property_data = _extend_cell_grid_to_boundaries(
                        property_data, face_data)
                data_array = self._pack_into_data_array(name, property_data)
                if data_array is not None:
                    # Remap to avoid outputting _face suffix in output. Done only for
                    # _face variables with no corresponding non-face variable.
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

    transport_coeffs = (
        transport_coefficients_builder.calculate_total_transport_coeffs(
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


def check_for_errors(
    numerics: numerics_lib.Numerics,
    output_state,
    post_processed_outputs: PostProcessedOutputs,
):
    if numerics.adaptive_dt:
        if output_state.solver_numeric_outputs.solver_error_state == 1:
            if output_state.dt / numerics.dt_reduction_factor < numerics.min_dt:
                return state.SimError.REACHED_MIN_DT
    state_error = output_state.check_for_errors()
    if state_error != state.SimError.NO_ERROR:
        return state_error
    else:
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
    final_total_transport = (
        transport_coefficients_builder.calculate_total_transport_coeffs(
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


class ToraxConfig(BaseModelFrozen):
    profile_conditions: profile_conditions_lib.ProfileConditions
    numerics: numerics_lib.Numerics
    plasma_composition: plasma_composition_lib.PlasmaComposition
    geometry: Geometry0
    sources: sources_pydantic_model.Sources
    neoclassical: Neoclassical0 = (
        Neoclassical0()  # pylint: disable=missing-kwoa
    )
    solver: SolverConfig = pydantic.Field(discriminator='solver_type')
    transport: transport_model_pydantic_model.TransportConfig = pydantic.Field(
        discriminator='model_name')
    pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
        discriminator='model_name')
    mhd: MHD = MHD()
    restart: FileRestart | None = pydantic.Field(default=None)

    def build_physics_models(self):
        return PhysicsModels(
            pedestal_model=self.pedestal.build_pedestal_model(),
            source_models=self.sources.build_models(),
            transport_model=self.transport.build_transport_model(),
            neoclassical_models=self.neoclassical.build_models(),
            mhd_models=self.mhd.build_mhd_models(),
        )

    @pydantic.model_validator(mode='before')
    @classmethod
    def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        configurable_data = copy.deepcopy(data)
        return configurable_data

    @pydantic.model_validator(mode='after')
    def _check_fields(self) -> typing_extensions.Self:
        using_nonlinear_transport_model = self.transport.model_name in [
            'qualikiz',
            'qlknn',
            'CGM',
        ]
        using_linear_solver = isinstance(self.solver, LinearThetaMethod)

        # pylint: disable=g-long-ternary
        # pylint: disable=attribute-error
        initial_guess_mode_is_linear = (False if using_linear_solver else
                                        self.solver.initial_guess_mode
                                        == enums.InitialGuessMode.LINEAR)
        return self

    @pydantic.computed_field
    @property
    def torax_version(self) -> str:
        return version.TORAX_VERSION


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
g.physics_models = torax_config.build_physics_models()
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
