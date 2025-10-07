from absl import logging
from collections.abc import Sequence
from collections.abc import Set
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src import version
from torax._src import xnp
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import updaters
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.mhd import base as mhd_model_lib
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import output
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
from torax._src.time_step_calculator import time_step_calculator as ts
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import pydantic_types
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_model as transport_model_lib
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Any, Mapping
from typing import TypeAlias
from typing_extensions import Annotated
from typing_extensions import Self
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
import tqdm
import treelib
import typing_extensions
import xarray as xr

TIME_INVARIANT = model_base.TIME_INVARIANT
JAX_STATIC = model_base.JAX_STATIC
BaseModelFrozen = model_base.BaseModelFrozen

TimeVaryingScalar = interpolated_param_1d.TimeVaryingScalar
TimeVaryingArray = interpolated_param_2d.TimeVaryingArray
NonNegativeTimeVaryingArray = interpolated_param_2d.NonNegativeTimeVaryingArray
PositiveTimeVaryingScalar = interpolated_param_1d.PositiveTimeVaryingScalar
NonNegativeTimeVaryingScalar = (
    interpolated_param_1d.NonNegativeTimeVaryingScalar
)
UnitIntervalTimeVaryingScalar = (
    interpolated_param_1d.UnitIntervalTimeVaryingScalar
)
PositiveTimeVaryingArray = interpolated_param_2d.PositiveTimeVaryingArray

ValidatedDefault = functools.partial(pydantic.Field, validate_default=True)

Grid1D = interpolated_param_2d.Grid1D
set_grid = interpolated_param_2d.set_grid

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsModels:
  """A container for all physics models."""

  source_models: source_models_lib.SourceModels = dataclasses.field(
      metadata=dict(static=True)
  )
  transport_model: transport_model_lib.TransportModel = dataclasses.field(
      metadata=dict(static=True)
  )
  pedestal_model: pedestal_model_lib.PedestalModel = dataclasses.field(
      metadata=dict(static=True)
  )
  neoclassical_models: neoclassical_models_lib.NeoclassicalModels = (
      dataclasses.field(metadata=dict(static=True))
  )
  mhd_models: mhd_model_lib.MHDModels = dataclasses.field(
      metadata=dict(static=True)
  )

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

# ToraxConfig.
CONFIG = "config"

# Excluded coordinates from geometry since they are at the top DataTree level.
# Exclude q_correction_factor as it is not an interesting quantity to save.
# TODO(b/338033916): consolidate on either rho or rho_cell naming for cell grid
EXCLUDED_GEOMETRY_NAMES = frozenset({
    RHO_FACE,
    RHO_CELL,
    RHO_CELL_NORM,
    RHO_FACE_NORM,
    "rho",
    "rho_norm",
    "q_correction_factor",
})

@enum.unique
class GeometryType(enum.IntEnum):
    """Integer enum for geometry type.

  This type can be used within JAX expressions to access the geometry type
  without having to call isinstance.
  """

    CIRCULAR = 0
    CHEASE = 1
    FBT = 2
    EQDSK = 3
    IMAS = 4


# pylint: disable=invalid-name


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
    """Update Phibdot in the geometry dataclasses used in the time interval.

  Phibdot is used in calc_coeffs to calculate terms related to time-dependent
  geometry. It should be set to be the same for geo_t and geo_t_plus_dt for
  each given time interval. This means that geo_t_plus_dt.Phibdot will not
  necessarily be the same as the geo_t.Phibdot at the next time step.

  Args:
    dt: Time step duration.
    geo_t: The geometry of the torus during this time step of the simulation.
    geo_t_plus_dt: The geometry of the torus during the next time step of the
      simulation.

  Returns:
    Tuple containing:
      - The geometry of the torus during this time step of the simulation.
      - The geometry of the torus during the next time step of the simulation.
  """
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

def get_initial_state_and_post_processed_outputs(
    t,
    runtime_params_provider,
    geometry_provider,
    step_fn,
):
    runtime_params_for_init, geo_for_init = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=t,
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
    return initial_state, post_processed_outputs


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


def run_loop(
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    initial_state,
    initial_post_processed_outputs: post_processing.PostProcessedOutputs,
    step_fn,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
):
    running_main_loop_start_time = time.time()
    wall_clock_step_times = []
    current_state = initial_state
    state_history = [current_state]
    post_processing_history = [initial_post_processed_outputs]
    sim_error = state.SimError.NO_ERROR
    initial_runtime_params = runtime_params_provider(initial_state.t)
    time_step_calculator_params = initial_runtime_params.time_step_calculator
    with tqdm.tqdm(
            total=
            100,  # This makes it so that the progress bar measures a percentage
            desc='Simulating',
            disable=not progress_bar,
            leave=True,
    ) as pbar:
        while step_fn.time_step_calculator.not_done(
                current_state.t,
                runtime_params_provider.numerics.t_final,
                time_step_calculator_params,
        ):
            step_start_time = time.time()
            if log_timestep_info:
                _log_timestep(current_state)
            current_state, post_processed_outputs = step_fn(
                current_state,
                post_processing_history[-1],
            )
            sim_error = check_for_errors(
                runtime_params_provider.numerics,
                current_state,
                post_processed_outputs,
            )
            wall_clock_step_times.append(time.time() - step_start_time)
            if sim_error != state.SimError.NO_ERROR:
                sim_error.log_error()
                break
            else:
                state_history.append(current_state)
                post_processing_history.append(post_processed_outputs)
                progress_ratio = (
                    float(current_state.t) -
                    runtime_params_provider.numerics.t_initial) / (
                        runtime_params_provider.numerics.t_final -
                        runtime_params_provider.numerics.t_initial)
                pbar.n = int(progress_ratio * pbar.total)
                pbar.set_description(f'Simulating (t={current_state.t:.5f})')
                pbar.refresh()
    if log_timestep_info and sim_error == state.SimError.NO_ERROR:
        _log_timestep(current_state)
    std_devs = 2  # Check if the first step is more than 2 std devs longer.
    if wall_clock_step_times and wall_clock_step_times[0] > (
            np.mean(wall_clock_step_times) +
            std_devs * np.std(wall_clock_step_times)):
        long_first_step = True
        logging.info(
            'The first step took more than %.1f std devs longer than other steps. '
            'It likely was tracing and compiling the step_fn. It took %.2fs '
            'of wall clock time.',
            std_devs,
            wall_clock_step_times[0],
        )
    else:
        long_first_step = False

    wall_clock_time_elapsed = time.time() - running_main_loop_start_time
    simulation_time = state_history[-1].t - state_history[0].t
    if long_first_step:
        # Don't include the long first step in the total time logged.
        wall_clock_time_elapsed -= wall_clock_step_times[0]
    logging.info(
        'Simulated %.2fs of physics in %.2fs of wall clock time.',
        simulation_time,
        wall_clock_time_elapsed,
    )
    return tuple(state_history), tuple(post_processing_history), sim_error


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


@jax.tree_util.register_pytree_node_class
class SimulationStepFn:

    def __init__(self, solver, time_step_calculator, runtime_params_provider,
                 geometry_provider):
        self._solver = solver
        if self._solver.physics_models.mhd_models.sawtooth_models is not None:
            self._sawtooth_solver = sawtooth_solver_lib.SawtoothSolver(
                physics_models=self._solver.physics_models, )
        else:
            self._sawtooth_solver = None
        self._time_step_calculator = time_step_calculator
        self._geometry_provider = geometry_provider
        self._runtime_params_provider = runtime_params_provider

    @property
    def runtime_params_provider(
        self, ) -> build_runtime_params.RuntimeParamsProvider:
        return self._runtime_params_provider

    def tree_flatten(self):
        children = (
            self._runtime_params_provider,
            self._geometry_provider,
        )
        aux_data = (
            self._solver,
            self._time_step_calculator,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            solver=aux_data[0],
            time_step_calculator=aux_data[1],
            runtime_params_provider=children[0],
            geometry_provider=children[1],
        )

    @property
    def geometry_provider(self):
        return self._geometry_provider

    @property
    def solver(self):
        return self._solver

    @property
    def time_step_calculator(self) -> ts.TimeStepCalculator:
        return self._time_step_calculator

    @xnp.jit
    def __call__(
        self,
        input_state,
        previous_post_processed_outputs,
    ):
        runtime_params_t, geo_t = (
            build_runtime_params.get_consistent_runtime_params_and_geometry(
                t=input_state.t,
                runtime_params_provider=self._runtime_params_provider,
                geometry_provider=self._geometry_provider,
            ))

        # This only computes sources set to explicit in the
        # SourceConfig. All implicit sources will have their profiles
        # set to 0.
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

            # Check for NaN in the next dt to avoid a recursive loop.
            is_nan_next_dt = xnp.isnan(next_dt)

            # If the solver did not converge we need to make a new step.
            solver_did_not_converge = solver_outputs.solver_error_state == 1

            # If t + dt  is exactly the final time we may need a smaller step than
            # min_dt to exactly reach the final time.
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
                # If the solver did not converge then we check if we are at the exact
                # final time and should take a smaller step. If not we also check if
                # the next dt is too small, if so we should end the step.
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
            # The solver returned state is still "intermediate" since the CoreProfiles
            # need to be updated by the evolved CellVariables in x_new
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
                        # The solver has not converged yet as we have not performed
                        # any steps yet.
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
        build_runtime_params.get_consistent_runtime_params_and_geometry(
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
        if (isinstance(configurable_data['pedestal'], dict)
                and 'model_name' not in configurable_data['pedestal']):
            configurable_data['pedestal']['model_name'] = 'no_pedestal'
        if (isinstance(configurable_data['transport'], dict)
                and 'model_name' not in configurable_data['transport']):
            configurable_data['transport']['model_name'] = 'constant'
        if (isinstance(configurable_data['solver'], dict)
                and 'solver_type' not in configurable_data['solver']):
            configurable_data['solver']['solver_type'] = 'linear'
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

    @pydantic.model_validator(mode='after')
    def _check_psidot_and_evolve_current(self) -> typing_extensions.Self:
        return self

    @pydantic.model_validator(mode='after')
    def _set_grid(self) -> Self:
        mesh = self.geometry.build_provider.torax_mesh
        set_grid(self, mesh, mode='relaxed')
        return self

    @pydantic.computed_field
    @property
    def torax_version(self) -> str:
        return version.TORAX_VERSION

    @pydantic.model_validator(mode='before')
    @classmethod
    def _remove_version_field(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'torax_version' in data:
                data = {k: v for k, v in data.items() if k != 'torax_version'}
        return data


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
torax_config = ToraxConfig.from_dict(CONFIG)
geometry_provider = torax_config.geometry.build_provider
physics_models = torax_config.build_physics_models()
solver = torax_config.solver.build_solver(physics_models=physics_models, )
runtime_params_provider = (
    build_runtime_params.RuntimeParamsProvider.from_config(torax_config))
step_fn = SimulationStepFn(
    solver=solver,
    time_step_calculator=torax_config.time_step_calculator.
    time_step_calculator,
    geometry_provider=geometry_provider,
    runtime_params_provider=runtime_params_provider,
)
initial_state, post_processed_outputs = (
    get_initial_state_and_post_processed_outputs(
        t=torax_config.numerics.t_initial,
        runtime_params_provider=runtime_params_provider,
        geometry_provider=geometry_provider,
        step_fn=step_fn,
    ))

state_history, post_processed_outputs_history, sim_error = run_loop(
    runtime_params_provider=runtime_params_provider,
    initial_state=initial_state,
    initial_post_processed_outputs=post_processed_outputs,
    step_fn=step_fn,
    log_timestep_info=False,
    progress_bar=False,
)

state_history = output.StateHistory(
    state_history=state_history,
    post_processed_outputs_history=post_processed_outputs_history,
    sim_error=sim_error,
    torax_config=torax_config,
)

data_tree = state_history.simulation_output_to_xr()
data_tree.to_netcdf("run.nc")
