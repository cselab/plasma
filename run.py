from absl import logging
from absl import logging
from collections.abc import Sequence
from collections.abc import Set
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import updaters
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.geometry import geometry
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_lib
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src import array_typing
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import physics_models
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src import state
from torax._src import version
from torax._src import xnp
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.mhd.sawtooth import sawtooth_solver as sawtooth_solver_lib
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.physics import formulas
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.solver import solver as solver_lib
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import qei_source as qei_source_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.time_step_calculator import time_step_calculator as ts
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model as transport_model_pydantic_model
from torax._src.transport_model import transport_coefficients_builder
from typing_extensions import Self
from typing import Any, Final, Mapping, Sequence, TypeAlias
from typing import Any, Mapping
import chex
import copy
import dataclasses
import dataclasses
import functools
import inspect
import itertools
import jax
import jax
import logging
import numpy as np
import numpy as np
import os
import pydantic
import time
import torax
import tqdm
import treelib
import typing_extensions
import xarray as xr

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

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves core_profiles which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which may evolve on each simulation step, such
  as sources and transport.

  This class includes both core_profiles and these additional elements.

  Attributes:
    t: time coordinate.
    dt: timestep interval.
    core_profiles: Core plasma profiles at time t.
    core_transport: Core plasma transport coefficients computed at time t.
    core_sources: Profiles for all sources/sinks. These are the profiles that
      are used to calculate the coefficients for the t+dt time step. For the
      explicit sources, these are calculated at the start of the time step, so
      are the values at time t. For the implicit sources, these are the most
      recent guess for time t+dt. The profiles here are the merged version of
      the explicit and implicit profiles.
    geometry: Geometry at this time step used for the simulation.
    solver_numeric_outputs: Numerical quantities related to the solver.
  """

  t: array_typing.FloatScalar
  dt: array_typing.FloatScalar
  core_profiles: state.CoreProfiles
  core_transport: state.CoreTransport
  core_sources: source_profiles.SourceProfiles
  geometry: geometry.Geometry
  solver_numeric_outputs: state.SolverNumericOutputs

  def check_for_errors(self) -> state.SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.negative_temperature_or_density():
      logging.info("Unphysical negative values detected in core profiles:\n")
      _log_negative_profile_names(self.core_profiles)
      return state.SimError.NEGATIVE_CORE_PROFILES
    if self.has_nan():
      logging.info("NaNs detected in ToraxSimState:\n")
      _log_nans(self)
      return state.SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return state.SimError.QUASINEUTRALITY_BROKEN
    else:
      return state.SimError.NO_ERROR

  def has_nan(self) -> bool:
    return any([np.any(np.isnan(x)) for x in jax.tree.leaves(self)])


def _log_nans(
    inputs: ToraxSimState,
) -> None:
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  nan_count = 0
  for path, value in path_vals:
    if np.any(np.isnan(value)):
      logging.info("Found NaNs in sim_state%s", jax.tree_util.keystr(path))
      nan_count += 1
  if nan_count >= 10:
    logging.info("""\nA common cause of widespread NaNs is negative densities or
        temperatures evolving during the solver step. This often arises through
        physical reasons like radiation collapse, or unphysical configuration
        such as impurity densities incompatible with physical quasineutrality.
        Check the output file for near-zero temperatures or densities at the
        last valid step.""")


def _log_negative_profile_names(inputs: state.CoreProfiles) -> None:
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  for path, value in path_vals:
    if np.any(np.less(value, 0.0)):
      logging.info("Found negative value in %s", jax.tree_util.keystr(path))

def safe_load_dataset(filepath: str) -> xr.DataTree:
  with open(filepath, "rb") as f:
    with xr.open_datatree(f) as dt_open:
      data_tree = dt_open.compute()
  return data_tree


def load_state_file(
    filepath: str,
) -> xr.DataTree:
  """Loads a state file from a filepath."""
  if os.path.exists(filepath):
    data_tree = safe_load_dataset(filepath)
    logging.info("Loading state file %s", filepath)
    return data_tree
  else:
    raise ValueError(f"File {filepath} does not exist.")


def concat_datatrees(
    tree1: xr.DataTree,
    tree2: xr.DataTree,
) -> xr.DataTree:
  """Concats two xr.DataTrees along the time dimension.

  For any duplicate time steps, the values from the first dataset are kept.

  Args:
    tree1: The first xr.DataTree to concatenate.
    tree2: The second xr.DataTree to concatenate.

  Returns:
    A xr.DataTree containing the concatenation of the two datasets.
  """

  def _concat_datasets(
      previous_ds: xr.Dataset,
      ds: xr.Dataset,
  ) -> xr.Dataset:
    """Concats two xr.Datasets."""
    # Do a minimal concat to avoid concatting any non time indexed vars.
    ds = xr.concat([previous_ds, ds], dim=TIME, data_vars="minimal")
    # Drop any duplicate time steps. Using "first" imposes
    # keeping the restart state from the earlier dataset. In the case of TORAX
    # restarts this contains more complete information e.g. transport and post
    # processed outputs.
    ds = ds.drop_duplicates(dim=TIME, keep="first")
    return ds

  return xr.map_over_datasets(_concat_datasets, tree1, tree2)


def _extend_cell_grid_to_boundaries(
    cell_var: array_typing.FloatVectorCell,
    face_var: array_typing.FloatVectorFace,
) -> array_typing.FloatVectorCellPlusBoundaries:
  """Merge face+cell grids into single [left_face, cells, right_face] grid."""

  left_value = np.expand_dims(face_var[:, 0], axis=-1)
  right_value = np.expand_dims(face_var[:, -1], axis=-1)

  return np.concatenate([left_value, cell_var, right_value], axis=-1)


def stitch_state_files(
    file_restart: file_restart_pydantic_model.FileRestart, datatree: xr.DataTree
) -> xr.DataTree:
  """Stitch a datatree to the end of a previous state file.

  Args:
    file_restart: Contains information on a file this sim was restarted from.
    datatree: The xr.DataTree to stitch to the end of the previous state file.

  Returns:
    A xr.DataTree containing the stitched dataset.
  """
  previous_datatree = load_state_file(file_restart.filename)
  # Reduce previous_ds to all times before the first time step in this
  # sim output. We use ds.time[0] instead of file_restart.time because
  # we are uncertain if file_restart.time is the exact time of the
  # first time step in this sim output (it takes the nearest time).
  previous_datatree = previous_datatree.sel(time=slice(None, datatree.time[0]))
  return concat_datatrees(previous_datatree, datatree)


class StateHistory:
  """A history of the state of the simulation and its error state."""

  def __init__(
      self,
      state_history,
      post_processed_outputs_history: tuple[
          post_processing.PostProcessedOutputs, ...
      ],
      sim_error: state.SimError,
      torax_config: model_config.ToraxConfig,
  ):
    if (
        not torax_config.restart
        and not torax_config.profile_conditions.use_v_loop_lcfs_boundary_condition
        and len(state_history) >= 2
    ):
      # For the Ip BC case, set v_loop_lcfs[0] to the same value as
      # v_loop_lcfs[1] due the v_loop_lcfs timeseries being
      # underconstrained
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
        stack, *self._core_profiles
    )
    self._stacked_core_sources: source_profiles_lib.SourceProfiles = (
        jax.tree_util.tree_map(stack, *self._core_sources)
    )
    self._stacked_core_transport: state.CoreTransport = jax.tree_util.tree_map(
        stack, *self._transport
    )
    self._stacked_post_processed_outputs: (
        post_processing.PostProcessedOutputs
    ) = jax.tree_util.tree_map(stack, *post_processed_outputs_history)
    self._stacked_solver_numeric_outputs: state.SolverNumericOutputs = (
        jax.tree_util.tree_map(stack, *self._solver_numeric_outputs)
    )
    self._times = np.array([state.t for state in state_history])
    chex.assert_rank(self.times, 1)
    # The rho grid does not change in time so we can just take the first one.
    self._rho_cell_norm = state_history[0].geometry.rho_norm
    self._rho_face_norm = state_history[0].geometry.rho_face_norm
    self._rho_norm = np.concatenate([[0.0], self.rho_cell_norm, [1.0]])

  @property
  def torax_config(self) -> model_config.ToraxConfig:
    """Returns the ToraxConfig used to run the simulation."""
    return self._torax_config

  @property
  def sim_error(self) -> state.SimError:
    """Returns the simulation error state."""
    return self._sim_error

  @property
  def times(self) -> array_typing.Array:
    """Returns the time of the simulation."""
    return self._times

  @property
  def rho_cell_norm(self) -> array_typing.FloatVectorCell:
    """Returns the normalized toroidal coordinate on the cell grid."""
    return self._rho_cell_norm

  @property
  def rho_face_norm(self) -> array_typing.FloatVectorFace:
    """Returns the normalized toroidal coordinate on the face grid."""
    return self._rho_face_norm

  @property
  def rho_norm(self) -> array_typing.FloatVectorCellPlusBoundaries:
    """Returns the rho on the cell grid with the left and right face boundaries."""
    return self._rho_norm

  @property
  def geometries(self) -> Sequence[geometry_lib.Geometry]:
    """Returns the geometries of the simulation."""
    return self._geometries

  @property
  def core_profiles(self) -> Sequence[state.CoreProfiles]:
    """Returns the core profiles."""
    return self._core_profiles

  @property
  def source_profiles(self) -> Sequence[source_profiles_lib.SourceProfiles]:
    """Returns the source profiles for the simulation."""
    return self._core_sources

  @property
  def core_transport(self) -> Sequence[state.CoreTransport]:
    """Returns the core transport for the simulation."""
    return self._transport

  @property
  def solver_numeric_outputs(self) -> Sequence[state.SolverNumericOutputs]:
    """Returns the solver numeric outputs."""
    return self._solver_numeric_outputs

  @property
  def post_processed_outputs(
      self,
  ) -> Sequence[post_processing.PostProcessedOutputs]:
    """Returns the post processed outputs for the simulation."""
    return self._post_processed_outputs

  def simulation_output_to_xr(self) -> xr.DataTree:
    """Build an xr.DataTree of the simulation output.

    Returns:
      A xr.DataTree containing a single top level xr.Dataset and four child
      datasets. The top level dataset contains the following variables:
        - time: The time of the simulation.
        - rho_norm: The normalized toroidal coordinate on the cell grid with the
            left and right face boundaries added.
        - rho_face_norm: The normalized toroidal coordinate on the face grid.
        - rho_cell_norm: The normalized toroidal coordinate on the cell grid.
        - config: The ToraxConfig used to run the simulation serialized to JSON.
      The child datasets contain the following variables:
        - numerics: Contains data variables for numeric quantities to do with
            the simulation.
        - profiles: Contains data variables for 1D profiles.
        - scalars: Contains data variables for scalars.
    """
    # Cleanup structure by excluding QeiInfo from core_sources altogether.
    # Add attribute to dataset variables with explanation of contents + units.

    # Get coordinate variables for dimensions ("time", "rho_face", "rho_cell")
    time = xr.DataArray(self.times, dims=[TIME], name=TIME)
    rho_face_norm = xr.DataArray(
        self.rho_face_norm, dims=[RHO_FACE_NORM], name=RHO_FACE_NORM
    )
    rho_cell_norm = xr.DataArray(
        self.rho_cell_norm, dims=[RHO_CELL_NORM], name=RHO_CELL_NORM
    )
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

    # Update dict with flattened StateHistory dataclass containers
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
        SIM_ERROR: self.sim_error.value,
        SAWTOOTH_CRASH: xr.DataArray(
            self._stacked_solver_numeric_outputs.sawtooth_crash,
            dims=[TIME],
            name=SAWTOOTH_CRASH,
        ),
        OUTER_SOLVER_ITERATIONS: xr.DataArray(
            self._stacked_solver_numeric_outputs.outer_solver_iterations,
            dims=[TIME],
            name=OUTER_SOLVER_ITERATIONS,
        ),
        INNER_SOLVER_ITERATIONS: xr.DataArray(
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
            attrs={CONFIG: self.torax_config.model_dump_json()},
        ),
    )

    if (
        self.torax_config.restart is not None
        and self.torax_config.restart.stitch
    ):
      data_tree = stitch_state_files(self.torax_config.restart, data_tree)

    return data_tree

  def _pack_into_data_array(
      self,
      name: str,
      data: jax.Array | None,
  ) -> xr.DataArray | None:
    """Packs the data into an xr.DataArray."""
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
    is_scalar = lambda x: x.ndim == 1 and x.shape == (len(self.times),)
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

  def _save_core_profiles(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the stacked core profiles to a dictionary of xr.DataArrays."""
    xr_dict = {}
    stacked_core_profiles = self._stacked_core_profiles

    # Map from CoreProfiles attribute name to the desired output name.
    # Needed for attributes that are not 1:1 with the output name.
    # Other attributes will use the same name as in CoreProfiles
    output_name_map = {
        "psidot": V_LOOP,
        "sigma": SIGMA_PARALLEL,
        "Ip_profile_face": IP_PROFILE,
        "q_face": Q,
        "s_face": MAGNETIC_SHEAR,
    }

    core_profile_field_names = {
        f.name for f in dataclasses.fields(stacked_core_profiles)
    }

    for field in dataclasses.fields(stacked_core_profiles):
      attr_name = field.name

      # Skip impurity_fractions since we have not yet converged on the public
      # API for individual impurity density extensions.
      if attr_name == "impurity_fractions":
        continue

      attr_value = getattr(stacked_core_profiles, attr_name)

      output_key = output_name_map.get(attr_name, attr_name)

      # Skip _face attributes if their cell counterpart exists;
      # they are handled when the cell attribute is processed.
      if attr_name.endswith("_face") and (
          attr_name.removesuffix("_face") in core_profile_field_names
      ):
        continue

      # Special handling for A_impurity for backward compatibility with V1
      # API for default 'fractions' impurity mode where A_impurity was a scalar.
      # TODO(b/434175938): Remove this once we move to V2
      if attr_name == "A_impurity":
        # Check if A_impurity is constant across the radial dimension for all
        # time steps. Need slicing (not indexing) to avoid a broadcasting error.
        is_constant = np.all(attr_value == attr_value[..., 0:1], axis=-1)
        if np.all(is_constant):
          # Save as a scalar time-series. Take the value at the first point.
          data_to_save = attr_value[..., 0]
        else:
          # Save as a profile.
          face_value = getattr(stacked_core_profiles, "A_impurity_face")
          data_to_save = _extend_cell_grid_to_boundaries(attr_value, face_value)
        xr_dict[output_key] = self._pack_into_data_array(
            output_key, data_to_save
        )
        continue

      if hasattr(attr_value, "cell_plus_boundaries"):
        # Handles stacked CellVariable-like objects.
        data_to_save = attr_value.cell_plus_boundaries()
      else:
        face_attr_name = f"{attr_name}_face"
        if face_attr_name in core_profile_field_names:
          # Combine cell and edge face values.
          face_value = getattr(stacked_core_profiles, face_attr_name)
          data_to_save = _extend_cell_grid_to_boundaries(attr_value, face_value)
        else:  # cell array with no face counterpart, or a scalar value
          data_to_save = attr_value

      xr_dict[output_key] = self._pack_into_data_array(output_key, data_to_save)

    # Handle derived quantities
    Ip_data = stacked_core_profiles.Ip_profile_face[..., -1]
    xr_dict[IP] = self._pack_into_data_array(IP, Ip_data)

    return xr_dict

  def _save_core_transport(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core transport to a dict."""
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

    # Save optional BohmGyroBohm attributes if present.
    core_transport = self._stacked_core_transport
    if (
        core_transport.chi_face_el_bohm is not None
        or core_transport.chi_face_el_gyrobohm is not None
        or core_transport.chi_face_ion_bohm is not None
        or core_transport.chi_face_ion_gyrobohm is not None
    ):
      xr_dict[CHI_BOHM_E] = core_transport.chi_face_el_bohm
      xr_dict[CHI_GYROBOHM_E] = core_transport.chi_face_el_gyrobohm
      xr_dict[CHI_BOHM_I] = core_transport.chi_face_ion_bohm
      xr_dict[CHI_GYROBOHM_I] = core_transport.chi_face_ion_gyrobohm

    xr_dict = {
        name: self._pack_into_data_array(
            name,
            data,
        )
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_core_sources(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the core sources to a dict."""
    xr_dict = {}

    xr_dict[qei_source_lib.QeiSource.SOURCE_NAME] = (
        self._stacked_core_sources.qei.qei_coef
        * (
            self._stacked_core_profiles.T_e.value
            - self._stacked_core_profiles.T_i.value
        )
    )

    xr_dict[J_BOOTSTRAP] = _extend_cell_grid_to_boundaries(
        self._stacked_core_sources.bootstrap_current.j_bootstrap,
        self._stacked_core_sources.bootstrap_current.j_bootstrap_face,
    )

    # Add source profiles with suffixes indicating which profile they affect.
    for profile in self._stacked_core_sources.T_i:
      if profile == "fusion":
        xr_dict["p_alpha_i"] = self._stacked_core_sources.T_i[profile]
      else:
        xr_dict[f"p_{profile}_i"] = self._stacked_core_sources.T_i[profile]
    for profile in self._stacked_core_sources.T_e:
      if profile == "fusion":
        xr_dict["p_alpha_e"] = self._stacked_core_sources.T_e[profile]
      else:
        xr_dict[f"p_{profile}_e"] = self._stacked_core_sources.T_e[profile]
    for profile in self._stacked_core_sources.psi:
      xr_dict[f"j_{profile}"] = self._stacked_core_sources.psi[profile]
    for profile in self._stacked_core_sources.n_e:
      xr_dict[f"s_{profile}"] = self._stacked_core_sources.n_e[profile]

    xr_dict = {
        name: self._pack_into_data_array(name, data)
        for name, data in xr_dict.items()
    }

    return xr_dict

  def _save_post_processed_outputs(
      self,
  ) -> dict[str, xr.DataArray | None]:
    """Saves the post processed outputs to a dict."""
    xr_dict = {}
    for field in dataclasses.fields(self._stacked_post_processed_outputs):
      attr_name = field.name

      # The impurity_radiation is structured differently and handled separately.
      if attr_name == "impurity_species":
        continue

      attr_value = getattr(self._stacked_post_processed_outputs, attr_name)
      if hasattr(attr_value, "cell_plus_boundaries"):
        # Handles stacked CellVariable-like objects.
        data_to_save = attr_value.cell_plus_boundaries()
      else:
        data_to_save = attr_value
      xr_dict[attr_name] = self._pack_into_data_array(attr_name, data_to_save)

    if self._stacked_post_processed_outputs.impurity_species:
      radiation_outputs = (
          impurity_radiation.construct_xarray_for_radiation_output(
              self._stacked_post_processed_outputs.impurity_species,
              self.times,
              self.rho_cell_norm,
              TIME,
              RHO_CELL_NORM,
          )
      )
      for key, value in radiation_outputs.items():
        xr_dict[key] = value

    return xr_dict

  def _save_geometry(
      self,
  ) -> dict[str, xr.DataArray]:
    """Save geometry to a dict. We skip over hires and non-array quantities."""
    xr_dict = {}
    geometry_attributes = dataclasses.asdict(self._stacked_geometry)

    # Get the variables from dataclass fields.
    for field_name, data in geometry_attributes.items():
      if (
          "hires" in field_name
          or (
              field_name.endswith("_face")
              and field_name.removesuffix("_face") in geometry_attributes
          )
          or field_name == "geometry_type"
          or field_name == "Ip_from_parameters"
          or field_name == "j_total"
          or not isinstance(data, array_typing.Array)
      ):
        continue
      if f"{field_name}_face" in geometry_attributes:
        data = _extend_cell_grid_to_boundaries(
            data, geometry_attributes[f"{field_name}_face"]
        )
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

    # Get variables from property methods
    geometry_properties = inspect.getmembers(type(self._stacked_geometry))
    property_names = set([name for name, _ in geometry_properties])

    for name, value in geometry_properties:
      # Skip over saving any variables that are named *_face.
      if (
          name.endswith("_face")
          and name.removesuffix("_face") in property_names
      ):
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
              property_data, face_data
          )
        data_array = self._pack_into_data_array(name, property_data)
        if data_array is not None:
          # Remap to avoid outputting _face suffix in output. Done only for
          # _face variables with no corresponding non-face variable.
          if name.endswith("_face"):
            name = name.removesuffix("_face")
          xr_dict[name] = data_array

    return xr_dict

def get_initial_state_and_post_processed_outputs(
    t: float,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    step_fn,
):
  """Returns the initial state and post processed outputs."""
  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
      )
  )
  initial_state = _get_initial_state(
      runtime_params=runtime_params_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  post_processed_outputs = post_processing.make_post_processed_outputs(
      initial_state, runtime_params_for_init
  )
  return initial_state, post_processed_outputs


def _get_initial_state(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    step_fn,
):
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
      )
  )

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


def get_initial_state_and_post_processed_outputs_from_file(
    t_initial: float,
    file_restart: file_restart_pydantic_model.FileRestart,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    step_fn,
):
  data_tree = output.load_state_file(file_restart.filename)
  # Find the closest time in the given dataset.
  data_tree = data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  profiles_dataset = data_tree.children[output.PROFILES].dataset
  profiles_dataset = profiles_dataset.squeeze()
  if t_restart != t_initial:
    logging.warning(
        'Requested restart time %f not exactly available in state file %s.'
        ' Restarting from closest available time %f instead.',
        file_restart.time,
        file_restart.filename,
        t_restart,
    )

  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t_initial,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
      )
  )
  runtime_params_for_init, geo_for_init = (
      _override_initial_runtime_params_from_file(
          runtime_params_for_init,
          geo_for_init,
          t_restart,
          profiles_dataset,
      )
  )
  initial_state = _get_initial_state(
      runtime_params=runtime_params_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  scalars_dataset = data_tree.children[output.SCALARS].dataset
  scalars_dataset = scalars_dataset.squeeze()
  post_processed_outputs = post_processing.make_post_processed_outputs(
      initial_state,
      runtime_params_for_init,
  )
  post_processed_outputs = dataclasses.replace(
      post_processed_outputs,
      E_fusion=scalars_dataset.data_vars['E_fusion'].to_numpy(),
      E_aux_total=scalars_dataset.data_vars['E_aux_total'].to_numpy(),
      E_ohmic_e=scalars_dataset.data_vars['E_ohmic_e'].to_numpy(),
      E_external_injected=scalars_dataset.data_vars[
          'E_external_injected'].to_numpy(),
      E_external_total=scalars_dataset.data_vars['E_external_total'].to_numpy(),
  )
  core_profiles = dataclasses.replace(
      initial_state.core_profiles,
      v_loop_lcfs=scalars_dataset.v_loop_lcfs.values,
  )
  numerics_dataset = data_tree.children[output.NUMERICS].dataset
  numerics_dataset = numerics_dataset.squeeze()
  sawtooth_crash = bool(numerics_dataset[output.SAWTOOTH_CRASH])
  outer_solver_iterations = int(
      numerics_dataset[output.OUTER_SOLVER_ITERATIONS]
  )
  inner_solver_iterations = int(
      numerics_dataset[output.INNER_SOLVER_ITERATIONS]
  )
  return (
      dataclasses.replace(
          initial_state,
          core_profiles=core_profiles,
          solver_numeric_outputs=state.SolverNumericOutputs(
              sawtooth_crash=sawtooth_crash,
              outer_solver_iterations=outer_solver_iterations,
              inner_solver_iterations=inner_solver_iterations,
          ),
      ),
      post_processed_outputs,
  )


def _override_initial_runtime_params_from_file(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    t_restart: float,
    profiles_ds: xr.Dataset,
) -> tuple[runtime_params_slice.RuntimeParams, geometry.Geometry]:
  """Override parts of runtime params slice from state in a file."""
  # pylint: disable=invalid-name
  runtime_params.numerics.t_initial = t_restart
  runtime_params.profile_conditions.Ip = profiles_ds.data_vars[
      output.IP_PROFILE
  ].to_numpy()[-1]
  runtime_params.profile_conditions.T_e = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_e_right_bc = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_i = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_i_right_bc = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  # Density in output is in m^-3.
  runtime_params.profile_conditions.n_e = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.n_e_right_bc = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  runtime_params.profile_conditions.psi = (
      profiles_ds.data_vars[output.PSI]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  # When loading from file we want ne not to have transformations.
  # Both ne and the boundary condition are given in absolute values (not fGW).
  # Additionally we want to avoid normalizing to nbar.
  runtime_params.profile_conditions.n_e_right_bc_is_fGW = False
  runtime_params.profile_conditions.n_e_nbar_is_fGW = False
  runtime_params.profile_conditions.normalize_n_e_to_nbar = False
  runtime_params.profile_conditions.n_e_right_bc_is_absolute = True

  return runtime_params, geo

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
    def __init__(
        self,
        solver: solver_lib.Solver,
        time_step_calculator: ts.TimeStepCalculator,
        runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
        geometry_provider: geometry_provider_lib.GeometryProvider,
    ):
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

    def _sawtooth_step(
        self,
        runtime_params_t: runtime_params_slice.RuntimeParams,
        geo_t: geometry.Geometry,
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
        input_state,
        previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    ):
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
        input_state,
        explicit_source_profiles: source_profiles_lib.SourceProfiles,
    ) -> tuple[
            tuple[cell_variable.CellVariable, ...],
            state.SolverNumericOutputs,
    ]:
        core_profiles_t = input_state.core_profiles
        core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
            dt=dt,
            runtime_params_t=runtime_params_t,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=core_profiles_t,
        )
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
        input_state,
        previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    ):
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
):
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
    input_state,
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
):
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
