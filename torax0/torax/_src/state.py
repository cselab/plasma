import dataclasses
from typing import Mapping
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.fvm import cell_variable


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

    def chi_max(
        self,
        geo,
    ):
        return jnp.maximum(
            jnp.max(
                self.chi_face_ion * geo.g1_over_vpr2_face),
            jnp.max(
                self.chi_face_el * geo.g1_over_vpr2_face),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SolverNumericOutputs:
    outer_solver_iterations: array_typing.IntScalar = 0
    solver_error_state: array_typing.IntScalar = 0
    inner_solver_iterations: array_typing.IntScalar = 0
    sawtooth_crash: array_typing.BoolScalar = False

