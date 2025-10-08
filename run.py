from absl import logging
from collections.abc import Sequence
from collections.abc import Set
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import state as state_module
from torax._src import version
from torax._src import xnp
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import updaters
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_lib
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.mhd import base as mhd_model_lib
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.physics import formulas
from torax._src.solver import runtime_params
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_base
from torax._src.torax_pydantic import pydantic_types
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import pydantic_model as transport_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_model as transport_model_lib
from typing import Annotated, Any, Literal
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Final
from typing import TypeAlias
from typing_extensions import Annotated
from typing_extensions import Self
import abc
import chex
import copy
import dataclasses
import enum
import functools
import inspect
import itertools
import jax
import jax.numpy as jnp
import jaxopt
import logging
import numpy as np
import os
import pydantic
import time
import torax
import treelib
import typing_extensions
import xarray as xr

Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs
AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput

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
    if x_i.value.shape != (num_cells,):
      raise ValueError(
          f'Expected each x channel to have shape ({num_cells},) '
          f'but got {x_i.value.shape}.'
      )

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
      d_face_i = jnp.zeros_like(v_face[i]) if d_face_i is None else d_face_i

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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns the left-hand and right-hand sides of the theta method equation.

  The theta method solves a differential equation

    tc_out partial (tc_in x) / partial t = F

  where `tc` is the transient coefficient, with `tc_out`
  being outside the partial derivative and `tc_in` inside it.

  We rearrange this to

    partial tc_in x / partial t = F / tc_out

  The theta method calculates one discrete time step by solving:

    | (tc_in_new x_new - tc_in_old x_old) / dt =
    | theta_implicit F_new / tc_out_new + theta_exp F_old / tc_out_old

  The equation is on the cell grid where `tc` is never zero. Therefore
  it's safe to multiply equation by `dt/tc_in_new` and scale the residual to
  `x`, which has O(1) values and thus the residual is scaled appropriately.

  We thus rearrange to:

    | x_new - tc_in_old/tc_in_new x_old =
    | dt theta_implicit F_new / (tc_out_new tc_in_new) +
    | dt theta_exp F_old / (tc_out_old tc_in_new)

  Rearranging we obtain

    | x_new - dt theta_implicit F_new / (tc_out_new tc_in_new) =
    | tc_in_old/tc_in_new x_old + dt theta_exp F_old / (tc_out_old tc_in_new)

  We now substitute in `F = Cu + c`:

    | (I - dt theta_implicit diag(1/(tc_out_new tc_in_new)) C_new) x_new
    | - dt theta_implicit diag(1/(tc_out_new tc_in_new)) c_new
    | =
    | (diag(tc_in_old/tc_in_new)
    | + dt theta_exp diag(1/(tc_out_old tc_in_new)) C_old) x_old
    | + dt theta_exp diag(1/(tc_out_old tc_in_new)) c_old

  Args:
    dt: Time step duration.
    x_old: The starting x defined as a tuple of CellVariables.
    x_new_guess: Current guess of x_new defined as a tuple of CellVariables.
    coeffs_old: The coefficients calculated at x_old.
    coeffs_new: The coefficients calculated at x_new.
    theta_implicit: Coefficient on implicit term of theta method.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    For the equation A x_new + a_vec = B x_old + b_vec. This function returns
     - left-hand side matrix, A
     - left-hand side vector, a
     - right-hand side matrix B
     - right-hand side vector, b
  """

  x_new_guess_vec = fvm_conversions.cell_variable_tuple_to_vec(x_new_guess)

  theta_exp = 1.0 - theta_implicit

  tc_in_old = jnp.concatenate(coeffs_old.transient_in_cell)
  tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
  tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)
  chex.assert_rank(tc_in_old, 1)
  chex.assert_rank(tc_out_new, 1)
  chex.assert_rank(tc_in_new, 1)

  eps = 1e-7
  # adding sanity checks for values in denominators
  # TODO(b/326577625) remove abs in checks once x_new range is restricted
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(jnp.abs(tc_in_new) < eps),
      msg='|tc_in_new| unexpectedly < eps',
  )
  tc_in_new = jax_utils.error_if(
      tc_in_new,
      jnp.any(jnp.abs(tc_out_new * tc_in_new) < eps),
      msg='|tc_out_new*tc_in_new| unexpectedly < eps',
  )

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


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def jaxopt_solver(
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    init_x_new_vec: jax.Array,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
    maxiter: int,
    tol: float,
) -> tuple[jax.Array, float, int]:
  """Advances jaxopt solver by one timestep.

  Args:
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object for time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    init_x_new_vec: Flattened array of initial guess of x_new for all evolving
      core profiles.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    maxiter: maximum number of iterations of jaxopt solver.
    tol: tolerance for jaxopt solver convergence.

  Returns:
    x_new_vec: Flattened evolving profile array after jaxopt evolution.
    final_loss: loss after jaxopt evolution
    num_iterations: number of iterations ran in jaxopt
  """

  loss = functools.partial(
      theta_method_block_loss,
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )
  solver = jaxopt.LBFGS(fun=loss, maxiter=maxiter, tol=tol)
  solver_output = solver.run(init_x_new_vec)
  x_new_vec = solver_output.params
  final_loss = loss(x_new_vec)
  num_iterations = solver_output.state.iter_num

  return x_new_vec, final_loss, num_iterations

# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
        'coeffs_callback',
        'initial_guess_mode',
        'log_iterations',
    ],
)
def newton_raphson_solve_block(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state_module.CoreProfiles,
    core_profiles_t_plus_dt: state_module.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: float,
    coarse_tol: float,
    delta_reduction_factor: float,
    tau_min: float,
    log_iterations: bool = False,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state_module.SolverNumericOutputs,
]:
  # pyformat: disable  # pyformat removes line breaks needed for reability
  """Runs one time step of a Newton-Raphson based root-finding on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  This solver uses iterative root finding on the linearized residual
  between two sides of the equation describing a theta method update.

  The linearized residual for a trial x_new is:

  R(x_old) + jacobian(R(x_old))*(x_new - x_old)

  Setting delta = x_new - x_old, we solve the linear system:

  A*x_new = b, with A = jacobian(R(x_old)), b = A*x_old - R(x_old)

  Each successive iteration sets x_new = x_old - delta, until the residual
  or delta is under a tolerance (tol).
  If either the delta step leads to an unphysical state, represented by NaNs in
  the residual, or if the residual doesn't shrink following the delta step,
  then delta is successively reduced by a delta_reduction_factor.
  If tau = delta_now / delta_original is below a tolerance, then the iterations
  stop. If residual > tol then the function exits with an error flag, producing
  either a warning or recalculation with a lower dt.

  Args:
    dt: Discrete time step.
    runtime_params_t: Runtime parameters for time t.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t: Geometry at time t.
    geo_t_plus_dt: Geometry at time t + dt.
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    core_profiles_t: Core plasma profiles which contain all available prescribed
      quantities at the start of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      core_profiles. Repeatedly called by the iterative optimizer.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    initial_guess_mode: chooses the initial_guess for the iterative method,
      either x_old or linear step. When taking the linear step, it is also
      recommended to use Pereverzev-Corrigan terms if the transport coefficients
      are stiff, e.g. from QLKNN. This can be set by setting use_pereverzev =
      True in the solver config.
    maxiter: Quit iterating after this many iterations reached.
    tol: Quit iterating after the average absolute value of the residual is <=
      tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum delta/delta_original allowed before the newton raphson
      routine resets at a lower timestep.
    log_iterations: If true, output diagnostic information from within iteration
      loop.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    solver_numeric_outputs: state_module.SolverNumericOutputs. Iteration and
      error info. For the error, 0 signifies residual < tol at exit, 1 signifies
      residual > tol, steps became small.
  """
  # pyformat: enable

  coeffs_old = coeffs_callback(
      runtime_params_t,
      geo_t,
      core_profiles_t,
      x_old,
      explicit_source_profiles=explicit_source_profiles,
      explicit_call=True,
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models
      # (e.g. qlknn) are used.
      coeffs_exp_linear = coeffs_callback(
          runtime_params_t,
          geo_t,
          core_profiles_t,
          x_old,
          explicit_source_profiles=explicit_source_profiles,
          allow_pereverzev=True,
          explicit_call=True,
      )

      # See linear_theta_method.py for comments on the predictor_corrector API
      x_new_guess = convertors.core_profiles_to_solver_x_tuple(
          core_profiles_t_plus_dt, evolving_names
      )
      init_x_new = predictor_corrector_method.predictor_corrector_method(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
          explicit_source_profiles=explicit_source_profiles,
      )
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(init_x_new)
    case enums.InitialGuessMode.X_OLD:
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
    case _:
      raise ValueError(
          f'Unknown option for first guess in iterations: {initial_guess_mode}'
      )
  # Create a residual() function with only one argument: x_new.
  # The other arguments (dt, x_old, etc.) are fixed.
  # Note that core_profiles_t_plus_dt only contains the known quantities at
  # t_plus_dt, e.g. boundary conditions and prescribed profiles.
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      physics_models=physics_models,
      explicit_source_profiles=explicit_source_profiles,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )

  # TODO(b/438673871): Set use_jax_custom_root=True once the compile time
  # regression is resolved.
  x_root, metadata = jax_root_finding.root_newton_raphson(
      fun=residual_fun,
      x0=init_x_new_vec,
      maxiter=maxiter,
      tol=tol,
      coarse_tol=coarse_tol,
      delta_reduction_factor=delta_reduction_factor,
      tau_min=tau_min,
      log_iterations=log_iterations,
      use_jax_custom_root=False,
  )

  # Create updated CellVariable instances based on state_plus_dt which has
  # updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_root, core_profiles_t_plus_dt, evolving_names
  )
  solver_numeric_outputs = state_module.SolverNumericOutputs(
      inner_solver_iterations=metadata.iterations,
      solver_error_state=metadata.error,
      outer_solver_iterations=1,
  )

  return x_new, solver_numeric_outputs

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
    coeffs_old: block_1d_coeffs.Block1DCoeffs,
    coeffs_new: block_1d_coeffs.Block1DCoeffs,
    theta_implicit: float = 1.0,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[cell_variable.CellVariable, ...]:

  x_old_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)

  lhs_mat, lhs_vec, rhs_mat, rhs_vec = (
      theta_method_matrix_equation(
          dt=dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          coeffs_old=coeffs_old,
          coeffs_new=coeffs_new,
          theta_implicit=theta_implicit,
          convection_dirichlet_mode=convection_dirichlet_mode,
          convection_neumann_mode=convection_neumann_mode,
      )
  )

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
  use_predictor_corrector: bool = dataclasses.field(metadata={'static': True})
  n_corrector_steps: int = dataclasses.field(metadata={'static': True})
  convection_dirichlet_mode: str = dataclasses.field(metadata={'static': True})
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
      physics_models: physics_models_lib.PhysicsModels,
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
        ' need `_x_new`.'
    )


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
    coeffs_exp: block_1d_coeffs.Block1DCoeffs,
    explicit_source_profiles: source_profiles.SourceProfiles,
    coeffs_callback: calc_coeffs.CoeffsCallback,
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
        convection_dirichlet_mode=(solver_params.convection_dirichlet_mode),
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

        x_old = convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t, evolving_names)
        x_new_guess = convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t_plus_dt, evolving_names)

        coeffs_callback = calc_coeffs.CoeffsCallback(
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
        physics_models: physics_models_lib.PhysicsModels,
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
        physics_models: physics_models_lib.PhysicsModels,
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
    physics_models = g.solver.physics_models
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

            core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
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
    solver: SolverConfig = pydantic.Field(discriminator='solver_type')
    transport: transport_model_pydantic_model.TransportConfig = pydantic.Field(
        discriminator='model_name')
    pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
        discriminator='model_name')
    mhd: mhd_pydantic_model.MHD = mhd_pydantic_model.MHD()
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
post_processed_outputs = post_processing.make_post_processed_outputs(
    initial_state, runtime_params_for_init)

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
