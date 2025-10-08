import dataclasses
import enum
from typing import Mapping
from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
import typing_extensions
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CoreProfiles:
  T_i: cell_variable.CellVariable
  T_e: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  psidot: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
  impurity_fractions: Mapping[str, array_typing.FloatVector]
  q_face: array_typing.FloatVectorFace
  s_face: array_typing.FloatVectorFace
  v_loop_lcfs: array_typing.FloatScalar
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  A_i: array_typing.FloatScalar
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace
  sigma: array_typing.FloatVectorCell
  sigma_face: array_typing.FloatVectorFace
  j_total: array_typing.FloatVectorCell
  j_total_face: array_typing.FloatVectorFace
  Ip_profile_face: array_typing.FloatVectorFace
  def quasineutrality_satisfied(self) -> bool:
    return jnp.allclose(
        self.n_i.value * self.Z_i + self.n_impurity.value * self.Z_impurity,
        self.n_e.value,
    ).item()
  def negative_temperature_or_density(self) -> jax.Array:
    profiles_to_check = (
        self.T_i,
        self.T_e,
        self.n_e,
        self.n_i,
        self.n_impurity,
        self.impurity_fractions,
    )
    return np.any(
        np.array([
            np.any(np.less(x, -constants.CONSTANTS.eps))
            for x in jax.tree.leaves(profiles_to_check)
        ])
    )
  def __str__(self) -> str:
    return f"""
      CoreProfiles(
        T_i={self.T_i},
        T_e={self.T_e},
        psi={self.psi},
        n_e={self.n_e},
        n_i={self.n_i},
        n_impurity={self.n_impurity},
        impurity_fractions={self.impurity_fractions},
      )
    """
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CoreTransport:
  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
  chi_neo_i: jax.Array | None = None
  chi_neo_e: jax.Array | None = None
  D_neo_e: jax.Array | None = None
  V_neo_e: jax.Array | None = None
  V_neo_ware_e: jax.Array | None = None
  def __post_init__(self):
    template = self.chi_face_el
    if self.chi_neo_i is None:
      self.chi_neo_i = jnp.zeros_like(template)
    if self.chi_neo_e is None:
      self.chi_neo_e = jnp.zeros_like(template)
    if self.D_neo_e is None:
      self.D_neo_e = jnp.zeros_like(template)
    if self.V_neo_e is None:
      self.V_neo_e = jnp.zeros_like(template)
    if self.V_neo_ware_e is None:
      self.V_neo_ware_e = jnp.zeros_like(template)
  def chi_max(
      self,
      geo: geometry.Geometry,
  ) -> jax.Array:
    return jnp.maximum(
        jnp.max((self.chi_face_ion + self.chi_neo_i) * geo.g1_over_vpr2_face),
        jnp.max((self.chi_face_el + self.chi_neo_e) * geo.g1_over_vpr2_face),
    )
  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    shape = geo.rho_face.shape
    return cls(
        chi_face_ion=jnp.zeros(shape),
        chi_face_el=jnp.zeros(shape),
        d_face_el=jnp.zeros(shape),
        v_face_el=jnp.zeros(shape),
        chi_face_el_bohm=jnp.zeros(shape),
        chi_face_el_gyrobohm=jnp.zeros(shape),
        chi_face_ion_bohm=jnp.zeros(shape),
        chi_face_ion_gyrobohm=jnp.zeros(shape),
        chi_neo_i=jnp.zeros(shape),
        chi_neo_e=jnp.zeros(shape),
        D_neo_e=jnp.zeros(shape),
        V_neo_e=jnp.zeros(shape),
        V_neo_ware_e=jnp.zeros(shape),
    )
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SolverNumericOutputs:
  outer_solver_iterations: array_typing.IntScalar = 0
  solver_error_state: array_typing.IntScalar = 0
  inner_solver_iterations: array_typing.IntScalar = 0
  sawtooth_crash: array_typing.BoolScalar = False
@enum.unique
class SimError(enum.Enum):
  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2
  NEGATIVE_CORE_PROFILES = 3
  REACHED_MIN_DT = 4
  def log_error(self):
    match self:
      case SimError.NEGATIVE_CORE_PROFILES:
        logging.error("""
            Simulation stopped due to negative values in core profiles.
            """)
      case SimError.NAN_DETECTED:
        logging.error("""
            Simulation stopped due to NaNs in state and/or post processed outputs.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.QUASINEUTRALITY_BROKEN:
        logging.error("""
            Simulation stopped due to quasineutrality being violated.
            Possible cause is bad handling of impurity species.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.REACHED_MIN_DT:
        logging.error("""
            Simulation stopped because the adaptive time step became too small.
            A common cause of vanishing timesteps is due to the nonlinear solver
            tending to negative densities or temperatures. This often arises
            through physical reasons like radiation collapse, or unphysical
            configuration such as impurity densities incompatible with physical
            quasineutrality. Check the output file for near-zero temperatures or
            densities at the last valid step.
            """)
      case SimError.NO_ERROR:
        pass
      case _:
        raise ValueError(f"Unknown SimError: {self}")
