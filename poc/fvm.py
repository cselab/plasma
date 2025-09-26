# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Block1DCoeffs dataclass.

This is the key interface between the `fvm` module, which is abstracted to the
level of a coupled 1D fluid dynamics PDE, and the rest of `torax`, which
includes
calculations specific to plasma physics to provide these coefficients.
"""

import dataclasses
from typing import Any, TypeAlias

import jax

# An optional argument, consisting of a 2D matrix of nested tuples, with each
# leaf being either None or a JAX Array. Used to define block matrices.
# examples:
#
# ((a, b), (c, d)) where a, b, c, d are each jax.Array
#
# ((a, None), (None, d)) : represents a diagonal block matrix
OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...], ...] | None


# Alias for better readability.
AuxiliaryOutput: TypeAlias = Any


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
  """
  transient_in_cell: tuple[jax.Array, ...]
  transient_out_cell: tuple[jax.Array, ...] | None = None
  d_face: tuple[jax.Array, ...] | None = None
  v_face: tuple[jax.Array, ...] | None = None
  source_mat_cell: OptionalTupleMatrix = None
  source_cell: tuple[jax.Array | None, ...] | None = None
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calculates Block1DCoeffs for a time step."""
import functools

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
import typing_extensions


# pylint: disable=invalid-name
class CoeffsCallback:
  """Calculates Block1DCoeffs for a state."""

  def __init__(
      self,
      physics_models: physics_models_lib.PhysicsModels,
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
    return (
        self.physics_models == other.physics_models
        and self.evolving_names == other.evolving_names
    )

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
  ) -> block_1d_coeffs.Block1DCoeffs:
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
    core_profiles = updaters.update_core_profiles_during_step(
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
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  chi_face_per_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  chi_face_per_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  d_face_per_el = runtime_params.solver.D_pereverzev
  v_face_per_el = (
      core_profiles.n_e.face_grad()
      / core_profiles.n_e.face_value()
      * d_face_per_el
      * geo_factor
  )

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
  v_heat_face_ion = (
      core_profiles.T_i.face_grad()
      / core_profiles.T_i.face_value()
      * chi_face_per_ion
  )
  v_heat_face_el = (
      core_profiles.T_e.face_grad()
      / core_profiles.T_e.face_value()
      * chi_face_per_el
  )

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
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
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
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """See `calc_coeffs` for details."""

  consts = constants.CONSTANTS

  pedestal_model_output = physics_models.pedestal_model(
      runtime_params, geo, core_profiles
  )

  # Boolean mask for enforcing internal temperature boundary conditions to
  # model the pedestal.
  # If rho_norm_ped_top_idx is outside of bounds of the mesh, the pedestal is
  # not present and the mask is all False. This is what is used in the case that
  # set_pedestal is False.
  mask = (
      jnp.zeros_like(geo.rho, dtype=bool)
      .at[pedestal_model_output.rho_norm_ped_top_idx]
      .set(True)
  )

  conductivity = (
      physics_models.neoclassical_models.conductivity.calculate_conductivity(
          geo, core_profiles
      )
  )

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
  toc_T_i = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  toc_T_e = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  toc_psi = (
      1.0
      / runtime_params.numerics.resistivity_multiplier
      * geo.rho_norm
      * conductivity.sigma
      * consts.mu_0
      * 16
      * jnp.pi**2
      * geo.Phi_b**2
      / geo.F**2
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  # Diffusion term coefficients
  turbulent_transport = physics_models.transport_model(
      runtime_params, geo, core_profiles, pedestal_model_output
  )
  neoclassical_transport = physics_models.neoclassical_models.transport(
      runtime_params, geo, core_profiles
  )

  chi_face_ion_total = (
      turbulent_transport.chi_face_ion + neoclassical_transport.chi_neo_i
  )
  chi_face_el_total = (
      turbulent_transport.chi_face_el + neoclassical_transport.chi_neo_e
  )
  d_face_el_total = (
      turbulent_transport.d_face_el + neoclassical_transport.D_neo_e
  )
  v_face_el_total = (
      turbulent_transport.v_face_el
      + neoclassical_transport.V_neo_e
      + neoclassical_transport.V_neo_ware_e
  )
  d_face_psi = geo.g2g3_over_rhon_face
  # No poloidal flux convection term
  v_face_psi = jnp.zeros_like(d_face_psi)

  # entire coefficient preceding dT/dr in heat transport equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * chi_face_ion_total
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * chi_face_el_total
  )

  # entire coefficient preceding dne/dr in particle equation
  full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
  full_v_face_el = geo.g0_face * v_face_el_total

  # density source terms. Initialize source matrix to zero
  source_mat_nn = jnp.zeros_like(geo.rho)

  # density source vector based both on original and updated core profiles
  source_n_e = merged_source_profiles.total_sources('n_e', geo)

  source_n_e += (
      mask
      * runtime_params.numerics.adaptive_n_source_prefactor
      * pedestal_model_output.n_e_ped
  )
  source_mat_nn += -(mask * runtime_params.numerics.adaptive_n_source_prefactor)

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
  v_heat_face_ion += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
  )

  v_heat_face_el += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
  )

  # Add Phi_b_dot terms to particle transport convection
  full_v_face_el += (
      -1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b * geo.rho_face_norm * geo.vpr_face
  )

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
  source_i += (
      mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_i_ped
  )
  source_e += (
      mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_e_ped
  )

  source_mat_ii -= mask * runtime_params.numerics.adaptive_T_source_prefactor

  source_mat_ee -= mask * runtime_params.numerics.adaptive_T_source_prefactor

  # Add effective Phi_b_dot heat source terms

  d_vpr53_rhon_n_e_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
      geo.rho_norm,
  )
  d_vpr53_rhon_n_i_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
      geo.rho_norm,
  )

  source_i += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_i_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_i.value
      * consts.keV_to_J
  )

  source_e += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_e_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_e.value
      * consts.keV_to_J
  )

  d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)

  # Add effective Phi_b_dot particle source terms
  source_n_e += (
      1.0
      / 2.0
      * d_vpr_rhon_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.n_e.value
  )

  # Add effective Phi_b_dot poloidal flux source term
  source_psi += (
      8.0
      * jnp.pi**2
      * consts.mu_0
      * geo.Phi_b_dot
      * geo.Phi_b
      * geo.rho_norm**2
      * conductivity.sigma
      / geo.F**2
      * core_profiles.psi.grad()
  )

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
      for row_block in evolving_names
  )

  # var_to_source ends up as a vector in the constructed PDE. Therefore any
  # scalings from CoreProfiles state variables to x must be applied here too.
  var_to_source = {
      'T_i': source_i / convertors.SCALING_FACTORS['T_i'],
      'T_e': source_e / convertors.SCALING_FACTORS['T_e'],
      'psi': source_psi / convertors.SCALING_FACTORS['psi'],
      'n_e': source_n_e / convertors.SCALING_FACTORS['n_e'],
  }
  source_cell = tuple(var_to_source.get(var) for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_out_cell=transient_out_cell,
      transient_in_cell=transient_in_cell,
      d_face=d_face,
      v_face=v_face,
      source_mat_cell=source_mat_cell,
      source_cell=source_cell,
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
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates only the transient_in_cell terms in Block1DCoeffs."""

  # Only transient_in_cell is used for explicit terms if theta_implicit=1
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  tic_psi = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  var_to_tic = {
      'T_i': tic_T_i,
      'T_e': tic_T_e,
      'psi': tic_psi,
      'n_e': tic_dens_el,
  }
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_in_cell=transient_in_cell,
  )
  return coeffs
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The CellVariable class.

A jax_utils.jax_dataclass used to represent variables on meshes for the 1D fvm
solver.
Naming conventions and API are similar to those developed in the FiPy fvm solver
[https://www.ctcms.nist.gov/fipy/]
"""
import dataclasses

import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
import typing_extensions


def _zero() -> array_typing.FloatScalar:
  """Returns a scalar zero as a jax Array."""
  return jnp.zeros(())


@chex.dataclass(frozen=True)
class CellVariable:
  """A variable representing values of the cells along the radius.

  Attributes:
    value: A jax.Array containing the value of this variable at each cell.
    dr: Distance between cell centers.
    left_face_constraint: An optional jax scalar specifying the value of the
      leftmost face. Defaults to None, signifying no constraint. The user can
      modify this field at any time, but when face_grad is called exactly one of
      left_face_constraint and left_face_grad_constraint must be None.
    left_face_grad_constraint: An optional jax scalar specifying the (otherwise
      underdetermined) value of the leftmost face. See left_face_constraint.
    right_face_constraint: Analogous to left_face_constraint but for the right
      face, see left_face_constraint.
    right_face_grad_constraint: A jax scalar specifying the undetermined value
      of the gradient on the rightmost face variable.
  """

  # t* means match 0 or more leading time dimensions.
  value: jt.Float[chex.Array, 't* cell']
  dr: jt.Float[chex.Array, 't*']
  left_face_constraint: jt.Float[chex.Array, 't*'] | None = None
  right_face_constraint: jt.Float[chex.Array, 't*'] | None = None
  left_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
      dataclasses.field(default_factory=_zero)
  )
  right_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
      dataclasses.field(default_factory=_zero)
  )
  # Can't make the above default values be jax zeros because that would be a
  # call to jax before absl.app.run

  def __post_init__(self):
    """Check that the CellVariable is valid.

    How is `sanity_check` different from `__post_init__`?
    - `sanity_check` is exposed to the client directly, so the client can
    explicitly check sanity without violating privacy conventions. This is
    useful for checking objects that were created e.g. using jax tree
    transformations.
    - `sanity_check` is guaranteed not to change the object, while
    `__post_init__` could in principle make changes.
    """
    # Automatically check dtypes of all numeric fields

    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      name = field.name
      if isinstance(value, jax.Array):
        if value.dtype != jnp.float64 and jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float64, got dtype {value.dtype} for `{name}`'
          )
        if value.dtype != jnp.float32 and not jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float32, got dtype {value.dtype} for `{name}`'
          )
    left_and = (
        self.left_face_constraint is not None
        and self.left_face_grad_constraint is not None
    )
    left_or = (
        self.left_face_constraint is not None
        or self.left_face_grad_constraint is not None
    )
    if left_and or not left_or:
      raise ValueError(
          'Exactly one of left_face_constraint and '
          'left_face_grad_constraint must be set.'
      )
    right_and = (
        self.right_face_constraint is not None
        and self.right_face_grad_constraint is not None
    )
    right_or = (
        self.right_face_constraint is not None
        or self.right_face_grad_constraint is not None
    )
    if right_and or not right_or:
      raise ValueError(
          'Exactly one of right_face_constraint and '
          'right_face_grad_constraint must be set.'
      )

  def _assert_unbatched(self):
    if len(self.value.shape) != 1:
      raise AssertionError(
          'CellVariable must be unbatched, but has `value` shape '
          f'{self.value.shape}. Consider using vmap to batch the function call.'
      )
    if self.dr.shape:
      raise AssertionError(
          'CellVariable must be unbatched, but has `dr` shape '
          f'{self.dr.shape}. Consider using vmap to batch the function call.'
      )

  def face_grad(
      self, x: jt.Float[chex.Array, 'cell'] | None = None
  ) -> jt.Float[chex.Array, 'face']:
    """Returns the gradient of this value with respect to the faces.

    Implemented using forward differencing of cells. Leftmost and rightmost
    gradient entries are determined by user specify constraints, see
    CellVariable class docstring.

    Args:
      x: (optional) coordinates over which differentiation is carried out

    Returns:
      A jax.Array of shape (num_faces,) containing the gradient.
    """
    self._assert_unbatched()
    if x is None:
      forward_difference = jnp.diff(self.value) / self.dr
    else:
      forward_difference = jnp.diff(self.value) / jnp.diff(x)

    def constrained_grad(
        face: jax.Array | None,
        grad: jax.Array | None,
        cell: jax.Array,
        right: bool,
    ) -> jax.Array:
      """Calculates the constrained gradient entry for an outer face."""

      if face is not None:
        if grad is not None:
          raise ValueError(
              'Cannot constraint both the value and gradient of '
              'a face variable.'
          )
        if x is None:
          dx = self.dr
        else:
          dx = x[-1] - x[-2] if right else x[1] - x[0]
        sign = -1 if right else 1
        return sign * (cell - face) / (0.5 * dx)
      else:
        if grad is None:
          raise ValueError('Must specify one of value or gradient.')
        return grad

    left_grad = constrained_grad(
        self.left_face_constraint,
        self.left_face_grad_constraint,
        self.value[0],
        right=False,
    )
    right_grad = constrained_grad(
        self.right_face_constraint,
        self.right_face_grad_constraint,
        self.value[-1],
        right=True,
    )

    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])

  def _left_face_value(self) -> jt.Float[chex.Array, '#t']:
    """Calculates the value of the leftmost face."""
    if self.left_face_constraint is not None:
      value = self.left_face_constraint
      # Boundary value has one fewer dim than cell value, expand to concat with.
      value = jnp.expand_dims(value, axis=-1)
    else:
      # When there is no constraint, leftmost face equals
      # leftmost cell
      value = self.value[..., 0:1]
    return value

  def _right_face_value(self) -> jt.Float[chex.Array, '#t']:
    """Calculates the value of the rightmost face."""
    if self.right_face_constraint is not None:
      value = self.right_face_constraint
      # Boundary value has one fewer dim than cell value, expand to concat with.
      value = jnp.expand_dims(value, axis=-1)
    else:
      # Maintain right_face consistent with right_face_grad_constraint
      value = (
          self.value[..., -1:]
          + jnp.expand_dims(self.right_face_grad_constraint, axis=-1)
          * jnp.expand_dims(self.dr, axis=-1)
          / 2
      )
    return value

  def face_value(self) -> jt.Float[jax.Array, 't* face']:
    """Calculates values of this variable on the face grid."""
    inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
    return jnp.concatenate(
        [self._left_face_value(), inner, self._right_face_value()], axis=-1
    )

  def grad(self) -> jt.Float[jax.Array, 't* face']:
    """Returns the gradient of this variable wrt cell centers."""
    face = self.face_value()
    return jnp.diff(face) / jnp.expand_dims(self.dr, axis=-1)

  def __str__(self) -> str:
    output_string = f'CellVariable(value={self.value}'
    if self.left_face_constraint is not None:
      output_string += f', left_face_constraint={self.left_face_constraint}'
    if self.right_face_constraint is not None:
      output_string += f', right_face_constraint={self.right_face_constraint}'
    if self.left_face_grad_constraint is not None:
      output_string += (
          f', left_face_grad_constraint={self.left_face_grad_constraint}'
      )
    if self.right_face_grad_constraint is not None:
      output_string += (
          f', right_face_grad_constraint={self.right_face_grad_constraint}'
      )
    output_string += ')'
    return output_string

  def cell_plus_boundaries(self) -> jt.Float[jax.Array, 't* cell+2']:
    """Returns the value of this variable plus left and right boundaries."""
    right_value = self._right_face_value()
    left_value = self._left_face_value()
    return jnp.concatenate(
        [left_value, self.value, right_value],
        axis=-1,
    )

  def __eq__(self, other: typing_extensions.Self) -> bool:
    try:
      chex.assert_trees_all_equal(self, other)
      return True
    except AssertionError:
      return False
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `make_convection_terms` function.

Builds the convection terms of the discrete matrix equation.
"""

import chex
import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.fvm import cell_variable


def make_convection_terms(
    v_face: jax.Array,
    d_face: jax.Array,
    var: cell_variable.CellVariable,
    dirichlet_mode: str = 'ghost',
    neumann_mode: str = 'ghost',
) -> tuple[jax.Array, jax.Array]:
  """Makes the terms of the matrix equation derived from the convection term.

  The convection term of the differential equation is of the form
  - (partial / partial r) v u

  Args:
    v_face: Convection coefficient on faces.
    d_face: Diffusion coefficient on faces. The relative strength of convection
      to diffusion is used to weight the contribution of neighboring cells when
      calculating face values of u.
    var: CellVariable to define mesh and boundary conditions.
    dirichlet_mode: The strategy to use to handle Dirichlet boundary conditions.
      The default is 'ghost', which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells 'direct' -> Boundary face values are read directly from constraints
      'semi-implicit' -> Matches FiPy. Boundary face values are alpha weighted
      with the constraint value specifying the value of the "other" cell:
      x_{boundary_face} = alpha x_{last_cell} + (1 - alpha) BC
    neumann_mode: Which strategy to use to handle Neumann boundary conditions.
      The default is `ghost`, which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells. 'semi-implicit' -> Matches FiPy. Boundary face values are alpha
      weighted, with the (1 - alpha) weight applied to the external face value
      rather than to a ghost cell.

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Alpha weighting calculated using power law scheme described in
  # https://www.ctcms.nist.gov/fipy/documentation/numerical/scheme.html

  # Avoid divide by zero
  eps = 1e-20
  is_neg = d_face < 0.0
  nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
  d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))

  # FiPy uses half mesh width at the boundaries
  half = jnp.array([0.5], dtype=jax_utils.get_dtype())
  ones = jnp.ones_like(v_face[1:-1])
  scale = jnp.concatenate((half, ones, half))

  ratio = scale * var.dr * v_face / d_face

  # left_peclet[i] gives the Péclet number of cell i's left face
  left_peclet = -ratio[:-1]
  right_peclet = ratio[1:]

  def peclet_to_alpha(p):
    eps = 1e-3
    p = jnp.where(jnp.abs(p) < eps, eps, p)

    alpha_pg10 = (p - 1) / p
    alpha_p0to10 = ((p - 1) + (1 - p / 10) ** 5) / p
    # FiPy doc has a typo on the next line, where we use a + the doc has a
    # -, which is clearly a mistake since it makes the function
    # discontinuous and negative
    alpha_pneg10to0 = ((1 + p / 10) ** 5 - 1) / p
    alpha_plneg10 = -1 / p

    alpha = 0.5 * jnp.ones_like(p)
    alpha = jnp.where(p > 10.0, alpha_pg10, alpha)
    alpha = jnp.where(jnp.logical_and(10.0 >= p, p > eps), alpha_p0to10, alpha)
    alpha = jnp.where(
        jnp.logical_and(-eps > p, p >= -10), alpha_pneg10to0, alpha
    )
    alpha = jnp.where(p < -10.0, alpha_plneg10, alpha)

    return alpha

  left_alpha = peclet_to_alpha(left_peclet)
  right_alpha = peclet_to_alpha(right_peclet)

  left_v = v_face[:-1]
  right_v = v_face[1:]

  diag = (left_alpha * left_v - right_alpha * right_v) / var.dr
  above = -(1.0 - right_alpha) * right_v / var.dr
  above = above[:-1]
  below = (1.0 - left_alpha) * left_v / var.dr
  below = below[1:]
  mat = math_utils.tridiag(diag, above, below)

  vec = jnp.zeros_like(diag)

  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )

  # Boundary rows need to be special-cased.
  #
  # Check that the boundary conditions are well-posed.
  # These checks are redundant with CellVariable.__post_init__, but including
  # them here for readability because they're in important part of the logic
  # of this function.
  chex.assert_exactly_one_is_none(
      var.left_face_grad_constraint, var.left_face_constraint
  )

  chex.assert_exactly_one_is_none(
      var.right_face_grad_constraint, var.right_face_constraint
  )

  if var.left_face_constraint is not None:
    # Dirichlet condition at leftmost face
    if dirichlet_mode == 'ghost':
      mat_value = (
          v_face[0] * (2.0 * left_alpha[0] - 1.0) - v_face[1] * right_alpha[0]
      ) / var.dr
      vec_value = (
          2.0 * v_face[0] * (1.0 - left_alpha[0]) * var.left_face_constraint
      ) / var.dr
    elif dirichlet_mode == 'direct':
      vec_value = v_face[0] * var.left_face_constraint / var.dr
      mat_value = -v_face[1] * right_alpha[0]
    elif dirichlet_mode == 'semi-implicit':
      vec_value = (
          v_face[0] * (1.0 - left_alpha[0]) * var.left_face_constraint
      ) / var.dr
      mat_value = mat[0, 0]
      print('left vec_value: ', vec_value)
    else:
      raise ValueError(dirichlet_mode)
  else:
    # Gradient boundary condition at leftmost face
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / var.dr
    vec_value = (
        -v_face[0] * (1.0 - left_alpha[0]) * var.left_face_grad_constraint
    )
    if neumann_mode == 'ghost':
      pass  # no adjustment needed
    elif neumann_mode == 'semi-implicit':
      vec_value /= 2.0
    else:
      raise ValueError(neumann_mode)

  mat = mat.at[0, 0].set(mat_value)
  vec = vec.at[0].set(vec_value)

  if var.right_face_constraint is not None:
    # Dirichlet condition at rightmost face
    if dirichlet_mode == 'ghost':
      mat_value = (
          v_face[-2] * left_alpha[-1]
          + v_face[-1] * (1.0 - 2.0 * right_alpha[-1])
      ) / var.dr
      vec_value = (
          -2.0
          * v_face[-1]
          * (1.0 - right_alpha[-1])
          * var.right_face_constraint
      ) / var.dr
    elif dirichlet_mode == 'direct':
      mat_value = v_face[-2] * left_alpha[-1] / var.dr
      vec_value = -v_face[-1] * var.right_face_constraint / var.dr
    elif dirichlet_mode == 'semi-implicit':
      mat_value = mat[-1, -1]
      vec_value = (
          -(v_face[-1] * (1.0 - right_alpha[-1]) * var.right_face_constraint)
          / var.dr
      )
    else:
      raise ValueError(dirichlet_mode)
  else:
    # Gradient boundary condition at rightmost face
    mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / var.dr
    vec_value = (
        -v_face[-1] * (1.0 - right_alpha[-1]) * var.right_face_grad_constraint
    )
    if neumann_mode == 'ghost':
      pass  # no adjustment needed
    elif neumann_mode == 'semi-implicit':
      vec_value /= 2.0
    else:
      raise ValueError(neumann_mode)

  mat = mat.at[-1, -1].set(mat_value)
  vec = vec.at[-1].set(vec_value)

  return mat, vec
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `make_diffusion_terms` function.

Builds the diffusion terms of the discrete matrix equation.
"""

import chex
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src.fvm import cell_variable


def make_diffusion_terms(
    d_face: array_typing.FloatVectorFace, var: cell_variable.CellVariable
) -> tuple[array_typing.FloatMatrixCell, array_typing.FloatVectorCell]:
  """Makes the terms of the matrix equation derived from the diffusion term.

  The diffusion term is of the form
  (partial / partial x) D partial x / partial x

  Args:
    d_face: Diffusivity coefficient on faces.
    var: CellVariable (to define geometry and boundary conditions)

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """

  # Start by using the formula for the interior rows everywhere
  denom = var.dr**2
  diag = jnp.asarray(-d_face[1:] - d_face[:-1])

  off = d_face[1:-1]
  vec = jnp.zeros_like(diag)

  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )

  # Boundary rows need to be special-cased.
  #
  # Check that the boundary conditions are well-posed.
  # These checks are redundant with CellVariable.__post_init__, but including
  # them here for readability because they're in important part of the logic
  # of this function.
  chex.assert_exactly_one_is_none(
      var.left_face_grad_constraint, var.left_face_constraint
  )
  chex.assert_exactly_one_is_none(
      var.right_face_grad_constraint, var.right_face_constraint
  )

  if var.left_face_constraint is not None:
    # Left face Dirichlet condition
    diag = diag.at[0].set(-2 * d_face[0] - d_face[1])
    vec = vec.at[0].set(2 * d_face[0] * var.left_face_constraint / denom)
  else:
    # Left face gradient condition
    diag = diag.at[0].set(-d_face[1])
    vec = vec.at[0].set(-d_face[0] * var.left_face_grad_constraint / var.dr)
  if var.right_face_constraint is not None:
    # Right face Dirichlet condition
    diag = diag.at[-1].set(-2 * d_face[-1] - d_face[-2])
    vec = vec.at[-1].set(2 * d_face[-1] * var.right_face_constraint / denom)
  else:
    # Right face gradient constraint
    diag = diag.at[-1].set(-d_face[-2])
    vec = vec.at[-1].set(d_face[-1] * var.right_face_grad_constraint / var.dr)

  # Build the matrix
  mat = math_utils.tridiag(diag, off, off) / denom
  return mat, vec
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality for building discrete linear systems.

This file is expected to be used mostly internally by `fvm` itself.

The functionality here is for constructing a description of one discrete
time step of a PDE in terms of a linear equation. In practice, the
actual expressive power of the resulting Jax expression may still be
nonlinear because the coefficients of this linear equation are Jax
expressions, not just numeric values, so nonlinear solvers like
newton_raphson_solve_block can capture nonlinear dynamics even when
each step is expressed using a matrix multiply.
"""
from typing import TypeAlias

import jax
from jax import numpy as jnp
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms

AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


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
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The `enums` module.

Enums shared through the `fvm` package.
"""
import enum


@enum.unique
class InitialGuessMode(enum.StrEnum):
  """Modes for initial guess of x_new for iterative solvers."""

  # Initialize x_new = x_old
  X_OLD = 'x_old'

  # Use the linear solver to guess x_new
  LINEAR = 'linear'
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversions utilities for fvm objects."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import state
from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable


def cell_variable_tuple_to_vec(
    x_tuple: tuple[cell_variable.CellVariable, ...],
) -> jax.Array:
  """Converts a tuple of CellVariables to a flat array.

  Args:
    x_tuple: A tuple of CellVariables.

  Returns:
    A flat array of evolving state variables.
  """
  x_vec = jnp.concatenate([x.value for x in x_tuple])
  return x_vec


def vec_to_cell_variable_tuple(
    x_vec: jax.Array,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> tuple[cell_variable.CellVariable, ...]:
  """Converts a flat array of core profile state vars to CellVariable tuple.

  Requires an input CoreProfiles to provide boundary condition information.

  Args:
    x_vec: A flat array of evolving core profile state variables. The order of
      the variables in the array must match the order of the evolving_names.
    core_profiles: CoreProfiles containing all CellVariables with appropriate
      boundary conditions.
    evolving_names: The names of the evolving cell variables.

  Returns:
    A tuple of updated CellVariables.
  """
  x_split = jnp.split(x_vec, len(evolving_names))

  # First scale the core profiles to match the scaling in x_split, then
  # update the values in the scaled core profiles with new values from x_split.
  scaled_evolving_cp_list = [
      convertors.scale_cell_variable(
          getattr(core_profiles, name),
          scaling_factor=1 / convertors.SCALING_FACTORS[name],
      )
      for name in evolving_names
  ]

  x_out = [
      dataclasses.replace(
          scaled_evolving_cp,
          value=value,
      )
      for scaled_evolving_cp, value in zip(scaled_evolving_cp_list, x_split)
  ]

  return tuple(x_out)
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `implicit_solve_block` function.

See function docstring for details.
"""
import dataclasses
import functools

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss


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
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Runs one time step of an implicit solver on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  Args:
    dt: Discrete time step.
    x_old: Tuple containing CellVariables for each channel with their values at
    x_new_guess: Tuple containing initial guess for x_new.
    coeffs_old: Coefficients defining the equation, computed for time t.
    coeffs_new: Coefficients defining the equation, computed for time t+dt.
    theta_implicit: Coefficient in [0, 1] determining which solution method to
      use. We solve transient_coeff (x_new - x_old) / dt = theta_implicit
      F(t_new) + (1 - theta_implicit) F(t_old). Three values of theta_implicit
      correspond to named solution methods: theta_implicit = 1: Backward Euler
      implicit method (default). theta_implicit = 0.5: Crank-Nicolson.
      theta_implicit = 0: Forward Euler explicit method
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
  """
  # pyformat: enable

  # In the linear case, we can use the same matrix formulation from the
  # nonlinear case but instead use linalg.solve to directly solve
  # residual, where the implicit coefficients are calculated with
  # an approximation of x_new, e.g. x_old for a single-step linear solve,
  # or from Picard iterations with predictor-corrector.
  # See residual_and_loss.theta_method_matrix_equation for a complete
  # description of how the equation is set up.

  x_old_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)

  lhs_mat, lhs_vec, rhs_mat, rhs_vec = (
      residual_and_loss.theta_method_matrix_equation(
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
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finite volume method.

This module contains functionality related to solving differential equations
built using the finite volume method. This module is meant to be somewhat
problem-independent, with other modules providing the coefficients on relatively
generic differential equations.
"""
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX root finding functions."""
import dataclasses
import functools
from typing import Callable, Final

import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils

# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RootMetadata:
  iterations: jax.Array
  residual: jax.Array
  last_tau: jax.Array
  error: jax.Array


def root_newton_raphson(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.Array | np.ndarray,
    *,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    log_iterations: bool = False,
    use_jax_custom_root: bool = True,
) -> tuple[jax.Array, RootMetadata]:
  """A differentiable Newton-Raphson root finder.

  A similar API to scipy.optimize.root.

  Args:
    fun: The function to find the root of.
    x0: The initial guess of the location of the root.
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
    use_jax_custom_root: If true, use jax.lax.custom_root to allow for
      differentiable solving. This can increase compile times even when no
      derivatives are requested.

  Returns:
    A tuple `(x_root, RootMetadata(...))`.
  """

  def _newton_raphson(f, x):
    residual_fun = f
    init_x_new_vec = x
    jacobian_fun = jax.jacfwd(f)
    # initialize state dict being passed around Newton-Raphson iterations
    residual_vec_init_x_new = residual_fun(init_x_new_vec)
    initial_state = {
        'x': init_x_new_vec,
        # jax.lax.custom_root is broken with aux outputs of integer type. Use
        # float for the iterations https://github.com/jax-ml/jax/issues/24295.
        'iterations': jnp.array(0, dtype=jax_utils.get_dtype()),
        'residual': residual_vec_init_x_new,
        'last_tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
    }

    # carry out iterations.
    cond_fun = functools.partial(
        _cond, tol=tol, tau_min=tau_min, maxiter=maxiter
    )
    body_fun = functools.partial(
        _body,
        jacobian_fun=jacobian_fun,
        residual_fun=residual_fun,
        log_iterations=log_iterations,
        delta_reduction_factor=delta_reduction_factor,
    )
    output_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    x_out = output_state.pop('x')
    return x_out, output_state

  # jax.lax.custom_root allows for differentiating through the solver,
  # efficiently. As the solver has a jax.lax.while_loop, it cannot be
  # reverse-mode differentiated. But even if we could, this would be highly
  # inefficient. This uses the implicit function theorem to differentiate
  # through the solver with only needing the result of the solver,
  # rather than the entire solver computational graph.
  # See also this discussion:
  # https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations

  def back(g, y):
    return jnp.linalg.solve(jax.jacfwd(g)(y), y)

  if use_jax_custom_root:
    x_out, metadata = jax.lax.custom_root(
        f=fun,
        initial_guess=x0,
        solve=_newton_raphson,
        tangent_solve=back,
        has_aux=True,
    )
  else:
    x_out, metadata = _newton_raphson(fun, x0)

  # Tell the caller whether or not x_new successfully reduces the residual below
  # the tolerance by providing an extra output, error.
  # error = 0: residual converged within fine tolerance (tol)
  # error = 1: not converged. Possibly backtrack to smaller dt and retry
  # error = 2: residual not strictly converged but is still within reasonable
  # tolerance (coarse_tol). Can occur when solver exits early due to small steps
  # in solution vicinity. Proceed but provide a warning to user.
  error = _error_cond(
      residual=metadata['residual'], coarse_tol=coarse_tol, tol=tol
  )
  # Workaround for https://github.com/google/jax/issues/24295: cast iterations
  # to the correct int dtype.
  metadata['iterations'] = metadata['iterations'].astype(
      jax_utils.get_int_dtype()
  )
  return x_out, RootMetadata(**metadata, error=error)


def _error_cond(residual: jax.Array, coarse_tol: float, tol: float):
  return jax.lax.cond(
      _residual_scalar(residual) < tol,
      lambda: 0,  # Called when True
      lambda: jax.lax.cond(  # Called when False
          _residual_scalar(residual) < coarse_tol,
          lambda: 2,  # Called when True
          lambda: 1,  # Called when False
      ),
  )


def _residual_scalar(x):
  return jnp.mean(jnp.abs(x))


def _cond(
    state: dict[str, jax.Array],
    tau_min: float,
    maxiter: int,
    tol: float,
) -> bool:
  """Check if exit condition reached for Newton-Raphson iterations."""
  iteration = state['iterations'][...]
  return jnp.bool_(
      jnp.logical_and(
          jnp.logical_and(
              _residual_scalar(state['residual']) > tol, iteration < maxiter
          ),
          state['last_tau'] > tau_min,
      )
  )


def _body(
    input_state: dict[str, jax.Array],
    jacobian_fun: Callable[[jax.Array], jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""
  dtype = input_state['x'].dtype
  a_mat = jacobian_fun(input_state['x'])
  rhs = -input_state['residual']
  # delta = x_new - x_old
  # tau = delta/delta0, where delta0 is the delta that sets the linearized
  # residual to zero. tau < 1 when needed such that x_new meets
  # conditions of reduced residual and valid state quantities.
  # If tau < taumin while residual > tol, then the routine exits with an
  # error flag, leading to either a warning or recalculation at lower dt
  initial_delta_state = {
      'x': input_state['x'],
      'delta': jnp.linalg.solve(a_mat, rhs),
      'residual_old': input_state['residual'],
      'residual_new': input_state['residual'],
      'tau': jnp.array(1.0, dtype=dtype),
  }
  output_delta_state = _compute_output_delta_state(
      initial_delta_state, residual_fun, delta_reduction_factor
  )

  output_state = {
      'x': input_state['x'] + output_delta_state['delta'],
      'residual': output_delta_state['residual_new'],
      'iterations': jnp.array(input_state['iterations'][...], dtype=dtype) + 1,
      'last_tau': output_delta_state['tau'],
  }
  if log_iterations:
    jax.debug.print(
        'Iteration: {iteration:d}. Residual: {residual:.16f}. tau = {tau:.6f}',
        iteration=output_state['iterations'].astype(jax_utils.get_int_dtype()),
        residual=_residual_scalar(output_state['residual']),
        tau=output_delta_state['tau'],
    )

  return output_state


def _compute_output_delta_state(
    initial_state: dict[str, jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    delta_reduction_factor: float,
):
  """Updates output delta state."""
  delta_body_fun = functools.partial(
      _delta_body,
      delta_reduction_factor=delta_reduction_factor,
  )
  delta_cond_fun = functools.partial(
      _delta_cond,
      residual_fun=residual_fun,
  )
  output_delta_state = jax.lax.while_loop(
      delta_cond_fun, delta_body_fun, initial_state
  )

  x_new = output_delta_state['x'] + output_delta_state['delta']
  residual_vec_x_new = residual_fun(x_new)
  output_delta_state |= dict(
      residual_new=residual_vec_x_new,
  )
  return output_delta_state


def _delta_cond(
    delta_state: dict[str, jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
) -> bool:
  """Check if delta obtained from Newton step is valid.

  Args:
    delta_state: see `delta_body`.
    residual_fun: Residual function.

  Returns:
    True if the new value of `x` causes any NaNs or has increased the residual
    relative to the old value of `x`.
  """
  x_old = delta_state['x']
  x_new = x_old + delta_state['delta']
  residual_vec_x_old = delta_state['residual_old']
  residual_scalar_x_old = _residual_scalar(residual_vec_x_old)
  # Avoid sanity checking inside residual, since we directly
  # afterwards check sanity on the output (NaN checking)
  # TODO(b/312453092) consider instead sanity-checking x_new
  with jax_utils.enable_errors(False):
    residual_vec_x_new = residual_fun(x_new)
    residual_scalar_x_new = _residual_scalar(residual_vec_x_new)
    delta_state['residual_new'] = residual_vec_x_new
  return jnp.bool_(
      jnp.logical_and(
          jnp.max(delta_state['delta']) > MIN_DELTA,
          jnp.logical_or(
              residual_scalar_x_old < residual_scalar_x_new,
              jnp.isnan(residual_scalar_x_new),
          ),
      ),
  )


def _delta_body(
    input_delta_state: dict[str, jax.Array],
    delta_reduction_factor: float,
) -> dict[str, jax.Array]:
  """Reduces step size for this Newton iteration."""
  return input_delta_state | dict(
      delta=input_delta_state['delta'] * delta_reduction_factor,
      tau=jnp.array(input_delta_state['tau'][...], dtype=jax_utils.get_dtype())
      * delta_reduction_factor,
  )
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `newton_raphson_solve_block` function.

See function docstring for details.
"""

import functools
from typing import Final
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state as state_module
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import jax_root_finding
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.solver import predictor_corrector_method
from torax._src.sources import source_profiles

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
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The `optimizer_solve_block` function.

See function docstring for details.
"""
import functools
from typing import TypeAlias

import jax
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.solver import predictor_corrector_method
from torax._src.sources import source_profiles

AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'coeffs_callback',
        'evolving_names',
        'initial_guess_mode',
    ],
)
def optimizer_solve_block(
    dt: jax.Array,
    runtime_params_t: runtime_params_slice.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: float,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state.SolverNumericOutputs,
]:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Runs one time step of an optimization-based solver on the equation defined by `coeffs`.

  This solver is relatively generic in that it models diffusion, convection,
  etc. abstractly. The caller must do the problem-specific physics calculations
  to obtain the coefficients for a particular problem.

  This solver uses iterative optimization to minimize the norm of the residual
  between two sides of the equation describing a theta method update.

  Args:
    dt: Discrete time step.
    runtime_params_t: Runtime params for time t (the start time of the step).
      These runtime params can change from step to step without triggering a
      recompilation.
    runtime_params_t_plus_dt: Runtime params for time t + dt.
    geo_t: Geometry object used to initialize auxiliary outputs at time t.
    geo_t_plus_dt: Geometry object used to initialize auxiliary outputs at time
      t + dt.
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
    initial_guess_mode: Chooses the initial_guess for the iterative method,
      either x_old or linear step. When taking the linear step, it is also
      recommended to use Pereverzev-Corrigan terms if the transport use
      pereverzev terms for linear solver. Is only applied in the nonlinear
      solver for the optional initial guess from the linear solver.
    maxiter: See docstring of `jaxopt.LBFGS`.
    tol: See docstring of `jaxopt.LBFGS`.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step
    solver_numeric_outputs: SolverNumericOutputs. Info about iterations and
      errors
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
    # corrector method if use_predictor_corrector=True in the solver runtime
    # params
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models (e.g. qlknn)
      # are used.
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

  solver_numeric_outputs = state.SolverNumericOutputs()

  # Advance jaxopt_solver by one timestep
  (
      x_new_vec,
      final_loss,
      solver_numeric_outputs.inner_solver_iterations,
  ) = residual_and_loss.jaxopt_solver(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      init_x_new_vec=init_x_new_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      maxiter=maxiter,
      tol=tol,
  )

  # Create updated CellVariable instances based on core_profiles_t_plus_dt which
  # has updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_vec, core_profiles_t_plus_dt, evolving_names
  )

  # Tell the caller whether or not x_new successfully reduces the loss below
  # the tolerance by providing an extra output, error.
  solver_numeric_outputs.solver_error_state = jax.lax.cond(
      final_loss > tol,
      lambda: 1,  # Called when True
      lambda: 0,  # Called when False
  )

  return x_new, solver_numeric_outputs
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Residual functions and loss functions.

Residual functions define a full differential equation and give a vector
measuring (left hand side) - (right hand side). Loss functions collapse
these to scalar functions, for example using mean squared error.
Residual functions are for use with e.g. the Newton-Raphson method
while loss functions can be minimized using any optimization method.
"""

import functools
from typing import TypeAlias

import chex
import jax
from jax import numpy as jnp
import jaxopt
from torax._src import jax_utils
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import discrete_system
from torax._src.fvm import fvm_conversions
from torax._src.geometry import geometry
from torax._src.sources import source_profiles

Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


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

  c_mat_new, c_new = discrete_system.calc_c(
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
def theta_method_block_residual(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
) -> jax.Array:
  """Residual of theta-method equation for core profiles at next time-step.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: The geometry at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.

  Returns:
    residual: Vector residual between LHS and RHS of the theta method equation.
  """
  x_old_vec = jnp.concatenate([var.value for var in x_old])
  # Prepare core_profiles_t_plus_dt for calc_coeffs. Explanation:
  # 1. The original (before iterative solving) core_profiles_t_plus_dt contained
  #    updated boundary conditions and prescribed profiles.
  # 2. Before calling calc_coeffs, we need to update the evolving subset of the
  #    core_profiles_t_plus_dt CellVariables with the current x_new_guess.
  # 3. Ion and impurity density and charge states are also updated here, since
  #    they are state dependent (on n_e and T_e).
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_guess_vec, core_profiles_t_plus_dt, evolving_names
  )
  core_profiles_t_plus_dt = updaters.update_core_profiles_during_step(
      x_new_guess,
      runtime_params_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      evolving_names=evolving_names,
      use_pereverzev=False,
  )

  solver_params = runtime_params_t_plus_dt.solver
  lhs_mat, lhs_vec, rhs_mat, rhs_vec = theta_method_matrix_equation(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_implicit=solver_params.theta_implicit,
      convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
      convection_neumann_mode=solver_params.convection_neumann_mode,
  )

  lhs = jnp.dot(lhs_mat, x_new_guess_vec) + lhs_vec
  rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec

  residual = lhs - rhs
  return residual


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def theta_method_block_loss(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    coeffs_old: Block1DCoeffs,
    evolving_names: tuple[str, ...],
) -> jax.Array:
  """Loss for the optimizer method of nonlinear solution.

  Args:
    x_new_guess_vec: Flattened array of current guess of x_new for all evolving
      core profiles.
    dt: Time step duration.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t_plus_dt: geometry object at time t + dt.
    x_old: The starting x defined as a tuple of CellVariables.
    core_profiles_t_plus_dt: Core plasma profiles which contain all available
      prescribed quantities at the end of the time step. This includes evolving
      boundary conditions and prescribed time-dependent profiles that are not
      being evolved by the PDE system.
    explicit_source_profiles: pre-calculated sources implemented as explicit
      sources in the PDE
    physics_models: Physics models used for the calculations.
    coeffs_old: The coefficients calculated at x_old.
    evolving_names: The names of variables within the core profiles that should
      evolve.

  Returns:
    loss: mean squared loss of theta method residual.
  """

  residual = theta_method_block_residual(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      x_new_guess_vec=x_new_guess_vec,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      physics_models=physics_models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
  )
  loss = jnp.mean(jnp.square(residual))
  return loss


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
