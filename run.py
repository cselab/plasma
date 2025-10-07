from absl import logging
from collections.abc import Sequence
from collections.abc import Set
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import updaters
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src import version
from torax._src import xnp
from torax._src.mhd import base as mhd_model_lib
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.physics import formulas
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.solver import solver as solver_lib
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import pydantic_model as transport_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_model as transport_model_lib
from typing_extensions import Annotated
from typing_extensions import Self
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Any, Mapping
from typing import TypeAlias
import chex
import copy
import dataclasses
import enum
import functools
import inspect
import itertools
import jax
import jax.numpy as jnp
import logging
import numpy as np
import os
import pydantic
import time
import torax
import treelib
import typing_extensions
import xarray as xr

import dataclasses

import jax
import abc
import functools

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import runtime_params as time_runtime_params
import jax
from jax import numpy as jnp
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.time_step_calculator import time_step_calculator


class g:
    pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
    tolerance: float


class TimeStepCalculator(abc.ABC):

    def not_done(self, t, t_final, time_calculator_params):
        return t < (t_final - time_calculator_params.tolerance)

    @functools.partial(
        jax_utils.jit,
        static_argnames=['self'],
    )
    def next_dt(self, t, runtime_params, geo, core_profiles, core_transport):
        dt = self._next_dt(
            runtime_params,
            geo,
            core_profiles,
            core_transport,
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

    @abc.abstractmethod
    def _next_dt(self, runtime_params, geo, core_profiles, core_transport):
        """Returns the next time step duration."""

    @abc.abstractmethod
    def __eq__(self, other):
        """Equality for the TimeStepCalculator, needed for JAX."""

    @abc.abstractmethod
    def __hash__(self):
        """Hash for the TimeStepCalculator, needed for JAX."""


class ChiTimeStepCalculator(TimeStepCalculator):

    def _next_dt(self, runtime_params, geo, core_profiles, core_transport):
        chi_max = core_transport.chi_max(geo)

        basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max

        dt = jnp.minimum(
            runtime_params.numerics.chi_timestep_prefactor * basic_dt,
            runtime_params.numerics.max_dt,
        )

        return dt

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
    sources: sources_pydantic_model.Sources
    numerics: numerics_lib.Numerics
    profile_conditions: profile_conditions_lib.ProfileConditions
    plasma_composition: plasma_composition_lib.PlasmaComposition
    transport_model: transport_pydantic_model.TransportConfig
    solver: solver_pydantic_model.SolverConfig
    pedestal: pedestal_pydantic_model.PedestalConfig
    mhd: mhd_pydantic_model.MHD
    neoclassical: neoclassical_pydantic_model.Neoclassical
    time_step_calculator: TimeStepCalculator

    @classmethod
    def from_config(cls, config) -> typing_extensions.Self:
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
            time_step_calculator=config.time_step_calculator,
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
            time_step_calculator=self.time_step_calculator.
            build_runtime_params(),
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
NonNegativeTimeVaryingScalar = (
    interpolated_param_1d.NonNegativeTimeVaryingScalar)
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
    mhd_models: mhd_model_lib.MHDModels = dataclasses.field(metadata=dict(
        static=True))


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
            post_processing.PostProcessedOutputs) = jax.tree_util.tree_map(
                stack, *post_processed_outputs_history)
        self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
            jax.tree_util.tree_map(stack, *self._solver_numeric_outputs))
        self._times = np.array([state.t for state in state_history])
        chex.assert_rank(self.times, 1)
        # The rho grid does not change in time so we can just take the first one.
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
                attrs={"config": self.torax_config.model_dump_json()},
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
            radiation_outputs = (
                impurity_radiation.construct_xarray_for_radiation_output(
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
    physics_models = step_fn.solver.physics_models
    initial_core_profiles = initialization.initial_core_profiles(
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
    post_processed_outputs: post_processing.PostProcessedOutputs,
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

    def __init__(self, solver, time_step_calculator, runtime_params_provider,
                 geometry_provider):
        self._solver = solver
        self._time_step_calculator = time_step_calculator
        self._geometry_provider = geometry_provider
        self._runtime_params_provider = runtime_params_provider

    @property
    def solver(self):
        return self._solver

    @property
    def time_step_calculator(self):
        return self._time_step_calculator

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
            source_models=self._solver.physics_models.source_models,
            neoclassical_models=self._solver.physics_models.
            neoclassical_models,
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
        initial_dt = self.time_step_calculator.next_dt(
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

            core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
                dt=dt,
                runtime_params_t=runtime_params_t,
                runtime_params_t_plus_dt=runtime_params_t_plus_dt,
                geo_t_plus_dt=geo_t_plus_dt,
                core_profiles_t=input_state.core_profiles,
            )
            x_new, solver_numeric_outputs = self._solver(
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
                    convertors.core_profiles_to_solver_x_tuple(
                        input_state.core_profiles, evolving_names),
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
            physics_models=self._solver.physics_models,
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
        updaters.update_core_and_source_profiles_after_step(
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
    post_processed_outputs = post_processing.make_post_processed_outputs(
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
    geometry: geometry_pydantic_model.Geometry
    sources: sources_pydantic_model.Sources
    neoclassical: neoclassical_pydantic_model.Neoclassical = (
        neoclassical_pydantic_model.Neoclassical()  # pylint: disable=missing-kwoa
    )
    solver: solver_pydantic_model.SolverConfig = pydantic.Field(
        discriminator='solver_type')
    transport: transport_model_pydantic_model.TransportConfig = pydantic.Field(
        discriminator='model_name')
    pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
        discriminator='model_name')
    mhd: mhd_pydantic_model.MHD = mhd_pydantic_model.MHD()
    time_step_calculator: (
        time_step_calculator_pydantic_model.TimeStepCalculator
    ) = time_step_calculator_pydantic_model.TimeStepCalculator()
    restart: file_restart_pydantic_model.FileRestart | None = pydantic.Field(
        default=None)

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
        using_linear_solver = isinstance(
            self.solver, solver_pydantic_model.LinearThetaMethod)

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
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
g.ts = ChiTimeStepCalculator()
torax_config = ToraxConfig.from_dict(CONFIG)
mesh = torax_config.geometry.build_provider.torax_mesh
interpolated_param_2d.set_grid(torax_config, mesh, mode='relaxed')

geometry_provider = torax_config.geometry.build_provider
physics_models = torax_config.build_physics_models()
solver = torax_config.solver.build_solver(physics_models=physics_models, )
runtime_params_provider = (RuntimeParamsProvider.from_config(torax_config))
step_fn = SimulationStepFn(
    solver=solver,
    time_step_calculator=ChiTimeStepCalculator(),
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
post_processed_outputs = post_processing.make_post_processed_outputs(
    initial_state, runtime_params_for_init)

initial_post_processed_outputs = post_processed_outputs
current_state = initial_state
state_history = [current_state]
post_processing_history = [initial_post_processed_outputs]
sim_error = state.SimError.NO_ERROR
initial_runtime_params = runtime_params_provider(initial_state.t)
time_step_calculator_params = initial_runtime_params.time_step_calculator
while step_fn.time_step_calculator.not_done(
        current_state.t,
        runtime_params_provider.numerics.t_final,
        time_step_calculator_params,
):
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
