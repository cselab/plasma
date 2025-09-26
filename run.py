import torax
from torax._src.config import build_runtime_params
from torax._src.orchestration import step_function
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_loop
from torax._src.output_tools import output
import copy
import logging
from typing import Any, Mapping

import pydantic
from torax._src import physics_models
from torax._src import version
from torax._src.config import numerics as numerics_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import enums
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
import typing_extensions
from typing_extensions import Self


class ToraxConfig(torax_pydantic.BaseModelFrozen):
  """Base config class for Torax.

  Attributes:
    profile_conditions: Config for the profile conditions.
    numerics: Config for the numerics.
    plasma_composition: Config for the plasma composition.
    geometry: Config for the geometry.
    pedestal: Config for the pedestal model. If an empty dictionary is passed
      in, the pedestal model will be set to `no_pedestal`.
    sources: Config for the sources.
    neoclassical: Config for the neoclassical models.
    solver: Config for the solver. If an empty dictionary is passed in, the
      solver model will be set to `linear`.
    transport: Config for the transport model. If an empty dictionary is passed
      in, the transport model will be set to `constant`.
    mhd: Optional config for mhd models. If None, no MHD models are used.
    time_step_calculator: Optional config for the time step calculator. If not
      provided the default chi time step calculator is used.
    restart: Optional config for file restart. If None, no file restart is
      performed.
  """

  profile_conditions: profile_conditions_lib.ProfileConditions
  numerics: numerics_lib.Numerics
  plasma_composition: plasma_composition_lib.PlasmaComposition
  geometry: geometry_pydantic_model.Geometry
  sources: sources_pydantic_model.Sources
  neoclassical: neoclassical_pydantic_model.Neoclassical = (
      neoclassical_pydantic_model.Neoclassical()  # pylint: disable=missing-kwoa
  )
  solver: solver_pydantic_model.SolverConfig = pydantic.Field(
      discriminator='solver_type'
  )
  transport: transport_model_pydantic_model.TransportConfig = pydantic.Field(
      discriminator='model_name'
  )
  pedestal: pedestal_pydantic_model.PedestalConfig = pydantic.Field(
      discriminator='model_name'
  )
  mhd: mhd_pydantic_model.MHD = mhd_pydantic_model.MHD()
  time_step_calculator: (
      time_step_calculator_pydantic_model.TimeStepCalculator
  ) = time_step_calculator_pydantic_model.TimeStepCalculator()
  restart: file_restart_pydantic_model.FileRestart | None = pydantic.Field(
      default=None
  )

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
    if (
        isinstance(configurable_data['pedestal'], dict)
        and 'model_name' not in configurable_data['pedestal']
    ):
      configurable_data['pedestal']['model_name'] = 'no_pedestal'
    if (
        isinstance(configurable_data['transport'], dict)
        and 'model_name' not in configurable_data['transport']
    ):
      configurable_data['transport']['model_name'] = 'constant'
    if (
        isinstance(configurable_data['solver'], dict)
        and 'solver_type' not in configurable_data['solver']
    ):
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
        self.solver, solver_pydantic_model.LinearThetaMethod
    )

    # pylint: disable=g-long-ternary
    # pylint: disable=attribute-error
    initial_guess_mode_is_linear = (
        False
        if using_linear_solver
        else self.solver.initial_guess_mode == enums.InitialGuessMode.LINEAR
    )

    if (
        using_nonlinear_transport_model
        and (using_linear_solver or initial_guess_mode_is_linear)
        and not self.solver.use_pereverzev
    ):
      logging.warning("""
          use_pereverzev=False in a configuration where setting
          use_pereverzev=True is recommended.

          A nonlinear transport model is used. However, a linear solver is also
          being used, either directly, or to provide an initial guess for a
          nonlinear solver.

          With this configuration, it is strongly recommended to set
          use_pereverzev=True to avoid numerical instability in the solver.
          """)
    return self

  @pydantic.model_validator(mode='after')
  def _check_psidot_and_evolve_current(self) -> typing_extensions.Self:
    """Warns if psidot is provided but evolve_current is True."""
    if (
        self.profile_conditions.psidot is not None
        and self.numerics.evolve_current
    ):
      logging.warning("""
          profile_conditions.psidot input is ignored as numerics.evolve_current
          is True.

          Prescribed psidot is only applied when current diffusion is off.
          """)
    return self

  def update_fields(self, x: Mapping[str, Any]):
    """Safely update fields in the config.

    This works with Frozen models.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these ancestral models will be
    re-validated.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """

    old_mesh = self.geometry.build_provider.torax_mesh
    self._update_fields(x)
    new_mesh = self.geometry.build_provider.torax_mesh

    if old_mesh != new_mesh:
      # The grid has changed, e.g. due to a new n_rho.
      # Clear the cached properties of all submodels and update the grid.
      for model in self.submodels:
        model.clear_cached_properties()
      torax_pydantic.set_grid(self, new_mesh, mode='force')
    else:
      # Update the grid on any new models which are added and have not had their
      # grid set yet.
      torax_pydantic.set_grid(self, new_mesh, mode='relaxed')

  @pydantic.model_validator(mode='after')
  def _set_grid(self) -> Self:
    # Interpolated `TimeVaryingArray` objects require a mesh, only available
    # once the geometry provider is built. This could be done in the before
    # validator, but is harder than setting it after construction.
    mesh = self.geometry.build_provider.torax_mesh
    # Note that the grid could already be set, eg. if the config is serialized
    # and deserialized. In this case, we do not want to overwrite it nor fail
    # when trying to set it, which is why mode='relaxed'.
    torax_pydantic.set_grid(self, mesh, mode='relaxed')
    return self

  # This is primarily used for serialization, so the importer can check which
  # version of Torax was used to generate the serialized config.
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
step_fn = step_function.SimulationStepFn(
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

state_history, post_processed_outputs_history, sim_error = run_loop.run_loop(
    runtime_params_provider=runtime_params_provider,
    initial_state=initial_state,
    initial_post_processed_outputs=post_processed_outputs,
    step_fn=step_fn,
    log_timestep_info=True,
    progress_bar=True,
)

state_history = output.StateHistory(
    state_history=state_history,
    post_processed_outputs_history=post_processed_outputs_history,
    sim_error=sim_error,
    torax_config=torax_config,
)

data_tree = state_history.simulation_output_to_xr()
