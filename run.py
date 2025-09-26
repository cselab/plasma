from absl import logging
from collections.abc import Set
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import updaters
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src import jax_utils
from torax._src import physics_models
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import version
from torax._src import xnp
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.mhd.sawtooth import sawtooth_solver as sawtooth_solver_lib
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.physics import formulas
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.solver import solver as solver_lib
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.time_step_calculator import time_step_calculator as ts
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
from typing_extensions import Self
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Any, Mapping
import copy
import dataclasses
import functools
import inspect
import jax
import logging
import numpy as np
import pydantic
import time
import torax
import tqdm
import treelib
import typing_extensions

TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'
JAX_STATIC: Final[str] = '_pydantic_jax_static_field'

StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]


def run_loop(
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    initial_state: sim_state.ToraxSimState,
    initial_post_processed_outputs: post_processing.PostProcessedOutputs,
    step_fn: step_function.SimulationStepFn,
    log_timestep_info: bool = False,
    progress_bar: bool = True,
) -> tuple[
        tuple[sim_state.ToraxSimState, ...],
        tuple[post_processing.PostProcessedOutputs, ...],
        state.SimError,
]:
    """Runs the simulation loop.

  Iterates over the step function until the time_step_calculator tells us we are
  done or the simulation hits an error state.

  Performs logging and updates the progress bar if requested.

  Args:
    runtime_params_provider: Provides a RuntimeParams to use as input for each
      time step.
    initial_state: The starting state of the simulation. This includes both the
      state variables which the solver.Solver will evolve (like ion temp, psi,
      etc.) as well as other states that need to be be tracked, like time.
    initial_post_processed_outputs: The post-processed outputs at the start of
      the simulation. This is used to calculate cumulative quantities.
    step_fn: Callable which takes in ToraxSimState and outputs the ToraxSimState
      after one timestep. Note that step_fn determines dt (how long the timestep
      is). The state_history that run_simulation() outputs comes from these
      ToraxSimState objects.
    log_timestep_info: If True, logs basic timestep info, like time, dt, on
      every step.
    progress_bar: If True, displays a progress bar.

  Returns:
    A tuple of:
      - the simulation history, consisting of a tuple of ToraxSimState objects,
        one for each time step. There are N+1 objects returned, where N is the
        number of simulation steps taken. The first object in the tuple is for
        the initial state. If the sim error state is 1, then a trunctated
        simulation history is returned up until the last valid timestep.
      - the post-processed outputs history, consisting of a tuple of
        PostProcessedOutputs objects, one for each time step. There are N+1
        objects returned, where N is the number of simulation steps taken. The
        first object in the tuple is for the initial state. If the sim error
        state is 1, then a trunctated simulation history is returned up until
        the last valid timestep.
      - The sim error state.
  """

    # Provide logging information on precision setting
    if jax.config.read('jax_enable_x64'):
        logging.info('Precision is set at float64')
    else:
        logging.info('Precision is set at float32')

    logging.info('Starting simulation.')
    # Python while loop implementation.
    # Not efficient for grad, jit of grad.
    # Uses time_step_calculator.not_done to decide when to stop.
    # Note: can't use a jax while loop due to appending to history.

    running_main_loop_start_time = time.time()
    wall_clock_step_times = []

    current_state = initial_state
    state_history = [current_state]
    post_processing_history = [initial_post_processed_outputs]

    # Set the sim_error to NO_ERROR. If we encounter an error, we will set it to
    # the appropriate error code.
    sim_error = state.SimError.NO_ERROR

    # Some of the runtime params are not time-dependent, so we can get them once
    # before the loop.
    initial_runtime_params = runtime_params_provider(initial_state.t)
    time_step_calculator_params = initial_runtime_params.time_step_calculator

    with tqdm.tqdm(
            total=
            100,  # This makes it so that the progress bar measures a percentage
            desc='Simulating',
            disable=not progress_bar,
            leave=True,
    ) as pbar:
        # Advance the simulation until the time_step_calculator tells us we are done
        while step_fn.time_step_calculator.not_done(
                current_state.t,
                runtime_params_provider.numerics.t_final,
                time_step_calculator_params,
        ):
            # Measure how long in wall clock time each simulation step takes.
            step_start_time = time.time()
            if log_timestep_info:
                _log_timestep(current_state)

            current_state, post_processed_outputs = step_fn(
                current_state,
                post_processing_history[-1],
            )
            sim_error = step_function.check_for_errors(
                runtime_params_provider.numerics,
                current_state,
                post_processed_outputs,
            )

            wall_clock_step_times.append(time.time() - step_start_time)

            # Checks if sim_state is valid. If not, exit simulation early.
            # We don't raise an Exception because we want to return the truncated
            # simulation history to the user for inspection.
            if sim_error != state.SimError.NO_ERROR:
                sim_error.log_error()
                break
            else:
                state_history.append(current_state)
                post_processing_history.append(post_processed_outputs)
                # Calculate progress ratio and update pbar.n
                progress_ratio = (
                    float(current_state.t) -
                    runtime_params_provider.numerics.t_initial) / (
                        runtime_params_provider.numerics.t_final -
                        runtime_params_provider.numerics.t_initial)
                pbar.n = int(progress_ratio * pbar.total)
                pbar.set_description(f'Simulating (t={current_state.t:.5f})')
                pbar.refresh()

    # Log final timestep
    if log_timestep_info and sim_error == state.SimError.NO_ERROR:
        # The "sim_state" here has been updated by the loop above.
        _log_timestep(current_state)

    # If the first step of the simulation was very long, call it out. It might
    # have to do with tracing the jitted step_fn.
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
    output_state: sim_state.ToraxSimState,
    post_processed_outputs: post_processing.PostProcessedOutputs,
) -> state.SimError:
    """Checks for errors in the simulation state."""
    if numerics.adaptive_dt:
        if output_state.solver_numeric_outputs.solver_error_state == 1:
            # Only check for min dt if the solver did not converge. Else we may have
            # converged at a dt > min_dt just before we reach min_dt.
            if output_state.dt / numerics.dt_reduction_factor < numerics.min_dt:
                return state.SimError.REACHED_MIN_DT
    state_error = output_state.check_for_errors()
    if state_error != state.SimError.NO_ERROR:
        return state_error
    else:
        return post_processed_outputs.check_for_errors()


@jax.tree_util.register_pytree_node_class
class SimulationStepFn:
    """Advances the TORAX simulation one time step.

  Unlike the Solver class, which updates certain parts of the state, a
  SimulationStepFn takes in the ToraxSimState and outputs the updated
  ToraxSimState, which contains not only the CoreProfiles but also extra
  simulation state useful for stepping as well as extra outputs useful for
  inspection inside the main run loop in `run_simulation()`. It wraps calls to
  Solver with useful features to increase robustness for convergence, like
  dt-backtracking.
  """

    def __init__(
        self,
        solver: solver_lib.Solver,
        time_step_calculator: ts.TimeStepCalculator,
        runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
        geometry_provider: geometry_provider_lib.GeometryProvider,
    ):
        """Initializes the SimulationStepFn.

    Args:
      solver: Evolves the core profiles.
      time_step_calculator: Calculates the dt for each time step.
      runtime_params_provider: Object that returns a set of runtime parameters
        which may change from time step to time step or simulation run to run.
      geometry_provider: Provides the magnetic geometry for each time step based
        on the ToraxSimState at the start of the time step. The geometry may
        change from time step to time step, so the sim needs a function to
        provide which geometry to use for a given time step. A GeometryProvider
        is any callable (class or function) which takes the ToraxSimState at the
        start of a time step and returns the Geometry for that time step. For
        most use cases, only the time will be relevant from the ToraxSimState
        (in order to support time-dependent geometries).
    """
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
    def geometry_provider(self) -> geometry_provider_lib.GeometryProvider:
        return self._geometry_provider

    @property
    def solver(self) -> solver_lib.Solver:
        return self._solver

    @property
    def time_step_calculator(self) -> ts.TimeStepCalculator:
        return self._time_step_calculator

    @xnp.jit
    def __call__(
        self,
        input_state: sim_state.ToraxSimState,
        previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    ) -> tuple[
            sim_state.ToraxSimState,
            post_processing.PostProcessedOutputs,
    ]:
        """Advances the simulation state one time step.

      If a sawtooth model is provided, it will be checked to see if a sawtooth
    should trigger. If it does, the sawtooth model will be applied and instead
    of a full PDE solve, the step_fn will return early with a state following
    sawtooth redistribution, at a t+dt set by the sawtooth model.

    Args:
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      previous_post_processed_outputs: Post-processed outputs from the previous
        time step.

    Returns:
      ToraxSimState containing:
        - the core profiles at the end of the time step.
        - time and time step calculator state info.
        - core_sources and core_transport at the end of the time step.
        - solver_numeric_outputs. This contains the number of iterations
          performed in the solver and the error state. The error states are:
            0 if solver converged with fine tolerance for this step
            1 if solver did not converge for this step (was above coarse tol)
            2 if solver converged within coarse tolerance. Allowed to pass with
              a warning. Occasional error=2 has low impact on final sim state.
      PostProcessedOutputs containing:
        - post-processed outputs at the end of the time step.
        - cumulative quantities.
      SimError indicating if an error has occurred during simulation.
    """
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
            """Take either the adaptive or fixed step, depending on the config."""
            if runtime_params_t.numerics.adaptive_dt:
                return self._adaptive_step(
                    runtime_params_t,
                    geo_t,
                    explicit_source_profiles,
                    input_state,
                    previous_post_processed_outputs,
                )
            else:
                return self._fixed_step(
                    runtime_params_t,
                    geo_t,
                    explicit_source_profiles,
                    input_state,
                    previous_post_processed_outputs,
                )

        # If a sawtooth model is provided, and there was no previous
        # sawtooth crash, it will be checked to see if a sawtooth
        # should trigger. If it does, the sawtooth model will be applied and instead
        # of a full PDE solve, the step_fn will return early with a state following
        # sawtooth redistribution, at a t+dt set by the sawtooth model
        # configuration.
        if self._sawtooth_solver is not None:
            output_state, post_processed_outputs = xnp.cond(
                input_state.solver_numeric_outputs.sawtooth_crash,
                lambda *args: (input_state, previous_post_processed_outputs),
                self._sawtooth_step,
                runtime_params_t,
                geo_t,
                explicit_source_profiles,
                input_state,
                previous_post_processed_outputs,
            )

            output_state, post_processed_outputs = xnp.cond(
                # If the current state is a sawtooth and the previous state was not,
                # then we triggered a sawtooth crash and exit early.
                output_state.solver_numeric_outputs.sawtooth_crash
                & ~input_state.solver_numeric_outputs.sawtooth_crash,
                lambda: (output_state, post_processed_outputs),
                _step,
            )
        else:
            # If no sawtooth model is provided, take a normal step.
            output_state, post_processed_outputs = _step()

        return output_state, post_processed_outputs

    def _sawtooth_step(
        self,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
        input_state: sim_state.ToraxSimState,
        previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    ) -> tuple[
            sim_state.ToraxSimState,
            post_processing.PostProcessedOutputs,
    ]:
        """Performs a simulation step if a sawtooth crash is triggered."""
        assert runtime_params_t.mhd.sawtooth is not None
        dt_crash = runtime_params_t.mhd.sawtooth.crash_step_duration

        runtime_params_t_plus_crash_dt, geo_t, geo_t_plus_crash_dt = (
            _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
                input_state.t,
                dt_crash,
                self._runtime_params_provider,
                geo_t,
                self._geometry_provider,
            ))

        # If no sawtooth crash is triggered, output_state and
        # post_processed_outputs will be the same as the input state and
        # previous_post_processed_outputs.
        output_state, post_processed_outputs = _sawtooth_step(
            sawtooth_solver=self._sawtooth_solver,
            runtime_params_t=runtime_params_t,
            runtime_params_t_plus_crash_dt=runtime_params_t_plus_crash_dt,
            geo_t=geo_t,
            geo_t_plus_crash_dt=geo_t_plus_crash_dt,
            explicit_source_profiles=explicit_source_profiles,
            input_state=input_state,
            input_post_processed_outputs=previous_post_processed_outputs,
        )
        return output_state, post_processed_outputs

    def step(
        self,
        dt: jax.Array,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        geo_t_plus_dt: geometry.Geometry,
        input_state: sim_state.ToraxSimState,
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        """Performs a simulation step with given dt.

    Solver may fail to converge in which case _adaptive_step() can be used to
    try smaller time step durations.

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters at time t.
      runtime_params_t_plus_dt: Runtime parameters at time t + dt.
      geo_t: The geometry of the torus during this time step of the simulation.
      geo_t_plus_dt: The geometry of the torus during the next time step of the
        simulation.
      input_state: State at the start of the time step, including the core
        profiles which are being evolved.
      explicit_source_profiles: Explicit source profiles computed based on the
        core profiles at the start of the time step.

    Returns:
      tuple:
        tuple of CellVariables corresponding to the evolved state variables
        SolverNumericOutputs containing error state and other solver-specific
        outputs.
    """

        core_profiles_t = input_state.core_profiles

        # Construct the CoreProfiles object for time t+dt with evolving boundary
        # conditions and time-dependent prescribed profiles not directly solved by
        # PDE system.
        core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
            dt=dt,
            runtime_params_t=runtime_params_t,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=core_profiles_t,
        )

        # Initial trial for solver. If did not converge (can happen for nonlinear
        # step with large dt) we apply the adaptive time step routine if requested.
        return self._solver(
            t=input_state.t,
            dt=dt,
            runtime_params_t=runtime_params_t,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
        )

    def _adaptive_step(
        self,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
        input_state: sim_state.ToraxSimState,
        previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    ) -> tuple[
            sim_state.ToraxSimState,
            post_processing.PostProcessedOutputs,
    ]:
        """Performs a (possibly) adaptive simulation step."""
        evolving_names = runtime_params_t.numerics.evolving_names

        initial_dt = self.time_step_calculator.next_dt(
            input_state.t,
            runtime_params_t,
            geo_t,
            input_state.core_profiles,
            input_state.core_transport,
        )

        input_type = jax.Array
        output_type = tuple[
            tuple[cell_variable.CellVariable, ...],
            jax.Array,  # dt
            state.SolverNumericOutputs,
            runtime_params_slice.RuntimeParams,
            geometry.Geometry,
            state.CoreProfiles,
        ]

        def cond_fun(inputs: tuple[input_type, output_type]):
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

        def body_fun(inputs: tuple[input_type, output_type]):
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
def _finalize_outputs(
    t: jax.Array,
    dt: jax.Array,
    x_new: tuple[cell_variable.CellVariable, ...],
    solver_numeric_outputs: state.SolverNumericOutputs,
    geometry_t_plus_dt: geometry.Geometry,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
    """Returns the final state and post-processed outputs."""
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

    output_state = sim_state.ToraxSimState(
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


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'sawtooth_solver',
    ],
)
def _sawtooth_step(
    *,
    sawtooth_solver: sawtooth_solver_lib.SawtoothSolver | None,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_crash_dt: runtime_params_slice.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_crash_dt: geometry.Geometry,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    input_state: sim_state.ToraxSimState,
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
    """Checks for and handles a sawtooth crash.

  If a sawtooth model is provided and a crash is triggered, this method
  computes the post-crash state and returns it. Otherwise, returns the input
  state and post-processed outputs unchanged.

  Consecutive sawtooth crashes are not allowed since standard PDE steps
  may then not take place. Therefore if the input state has sawtooth_crash set
  to True, then no crash is triggered.

  Args:
    sawtooth_solver: Sawtooth model which carries out sawtooth step..
    runtime_params_t: Runtime params at time t.
    runtime_params_t_plus_crash_dt: Runtime params at time t + crash_dt.
    geo_t: Geometry at time t.
    geo_t_plus_crash_dt: Geometry at time t + crash_dt.
    explicit_source_profiles: Explicit source profiles at time t.
    input_state: State at the start of the time step.
    input_post_processed_outputs: Post-processed outputs from the previous step.

  Returns:
    Returns a tuple (output_state, post_processed_outputs).
  """

    # Asserts needed for linter.
    assert runtime_params_t.mhd.sawtooth is not None
    assert sawtooth_solver is not None
    dt_crash = runtime_params_t.mhd.sawtooth.crash_step_duration

    # Prepare core_profiles_t_plus_crash_dt with new boundary conditions
    # and prescribed profiles if present.
    core_profiles_t_plus_crash_dt = updaters.provide_core_profiles_t_plus_dt(
        dt=dt_crash,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_crash_dt,
        geo_t_plus_dt=geo_t_plus_crash_dt,
        core_profiles_t=input_state.core_profiles,
    )

    (
        x_candidate,
        solver_numeric_outputs,
    ) = sawtooth_solver(
        t=input_state.t,
        dt=dt_crash,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_crash_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_crash_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_crash_dt,
        explicit_source_profiles=explicit_source_profiles,
    )

def _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
    t: jax.Array,
    dt: jax.Array,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geo_t: geometry.Geometry,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[
        runtime_params_slice.RuntimeParams,
        geometry.Geometry,
        geometry.Geometry,
]:
    """Returns the geos including Phibdot, and runtime params at t + dt.

  Args:
    t: Time at which the simulation is currently at.
    dt: Time step duration.
    runtime_params_provider: Object that returns a set of runtime parameters
      which may change from time step to time step or simulation run to run.
    geo_t: The geometry of the torus during this time step of the simulation.
    geometry_provider: Provides the magnetic geometry for each time step based
      on the ToraxSimState at the start of the time step.

  Returns:
    Tuple containing:
      - The runtime params at time t + dt.
      - The geometry of the torus during this time step of the simulation.
      - The geometry of the torus during the next time step of the simulation.
  """
    runtime_params_t_plus_dt, geo_t_plus_dt = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=t + dt,
            runtime_params_provider=runtime_params_provider,
            geometry_provider=geometry_provider,
        ))
    if runtime_params_t_plus_dt.numerics.calcphibdot:
        geo_t, geo_t_plus_dt = geometry.update_geometries_with_Phibdot(
            dt=dt,
            geo_t=geo_t,
            geo_t_plus_dt=geo_t_plus_dt,
        )

    return (
        runtime_params_t_plus_dt,
        geo_t,
        geo_t_plus_dt,
    )


class ToraxConfig(torax_pydantic.BaseModelFrozen):
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
        return physics_models.PhysicsModels(
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

    def update_fields(self, x: Mapping[str, Any]):
        old_mesh = self.geometry.build_provider.torax_mesh
        self._update_fields(x)
        new_mesh = self.geometry.build_provider.torax_mesh
        if old_mesh != new_mesh:
            for model in self.submodels:
                model.clear_cached_properties()
            torax_pydantic.set_grid(self, new_mesh, mode='force')
        else:
            torax_pydantic.set_grid(self, new_mesh, mode='relaxed')

    @pydantic.model_validator(mode='after')
    def _set_grid(self) -> Self:
        mesh = self.geometry.build_provider.torax_mesh
        torax_pydantic.set_grid(self, mesh, mode='relaxed')
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
    initial_state_lib.get_initial_state_and_post_processed_outputs(
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
