from collections.abc import Sequence
import dataclasses
import enum
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic


def face_to_cell(
    face: array_typing.FloatVectorFace, ) -> array_typing.FloatVectorCell:
    return 0.5 * (face[:-1] + face[1:])


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
    torax_mesh: torax_pydantic.Grid1D
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

    def __eq__(self, other: 'Geometry') -> bool:
        try:
            chex.assert_trees_all_equal(self, other)
        except AssertionError:
            return False
        return True

    @property
    def q_correction_factor(self) -> chex.Numeric:
        return jnp.where(
            self.geometry_type == GeometryType.CIRCULAR.value,
            1.25,
            1,
        )

    @property
    def rho_norm(self) -> array_typing.Array:
        return self.torax_mesh.cell_centers

    @property
    def rho_face_norm(self) -> array_typing.Array:
        return self.torax_mesh.face_centers

    @property
    def drho_norm(self) -> array_typing.Array:
        return jnp.array(self.torax_mesh.dx)

    @property
    def rho_face(self) -> array_typing.Array:
        return self.rho_face_norm * jnp.expand_dims(self.rho_b, axis=-1)

    @property
    def rho(self) -> array_typing.Array:
        return self.rho_norm * jnp.expand_dims(self.rho_b, axis=-1)

    @property
    def r_mid(self) -> array_typing.Array:
        return (self.R_out - self.R_in) / 2

    @property
    def r_mid_face(self) -> array_typing.Array:
        return (self.R_out_face - self.R_in_face) / 2

    @property
    def epsilon(self) -> array_typing.Array:
        return (self.R_out - self.R_in) / (self.R_out + self.R_in)

    @property
    def epsilon_face(self) -> array_typing.Array:
        return (self.R_out_face - self.R_in_face) / (self.R_out_face +
                                                     self.R_in_face)

    @property
    def drho(self) -> array_typing.Array:
        return self.drho_norm * self.rho_b

    @property
    def rho_b(self) -> array_typing.FloatScalar:
        return jnp.sqrt(self.Phi_b / np.pi / self.B_0)

    @property
    def Phi_b(self) -> array_typing.FloatScalar:
        return self.Phi_face[..., -1]

    @property
    def g1_over_vpr(self) -> array_typing.Array:
        return self.g1 / self.vpr

    @property
    def g1_over_vpr2(self) -> array_typing.Array:
        return self.g1 / self.vpr**2

    @property
    def g0_over_vpr_face(self) -> jax.Array:
        bulk = self.g0_face[..., 1:] / self.vpr_face[..., 1:]
        first_element = jnp.ones_like(self.rho_b) / self.rho_b
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    @property
    def g1_over_vpr_face(self) -> jax.Array:
        bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:]
        first_element = jnp.zeros_like(self.rho_b)
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    @property
    def g1_over_vpr2_face(self) -> jax.Array:
        bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:]**2
        first_element = jnp.ones_like(self.rho_b) / self.rho_b**2
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    @property
    def gm9(self) -> jax.Array:
        return 2 * jnp.pi * self.spr / self.vpr

    @property
    def gm9_face(self) -> jax.Array:
        bulk = 2 * jnp.pi * self.spr_face[..., 1:] / self.vpr_face[..., 1:]
        first_element = 1 / self.R_major
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    def z_magnetic_axis(self) -> chex.Numeric:
        z_magnetic_axis = self._z_magnetic_axis
        if z_magnetic_axis is not None:
            return z_magnetic_axis
        else:
            raise ValueError('Geometry does not have a z magnetic axis.')


def stack_geometries(geometries: Sequence[Geometry]) -> Geometry:
    if not geometries:
        raise ValueError('No geometries provided.')
    first_geo = geometries[0]
    torax_mesh = first_geo.torax_mesh
    geometry_type = first_geo.geometry_type
    for geometry in geometries[1:]:
        if geometry.torax_mesh != torax_mesh:
            raise ValueError('All geometries must have the same mesh.')
        if geometry.geometry_type != geometry_type:
            raise ValueError(
                'All geometries must have the same geometry type.')
    stacked_data = {}
    for field in dataclasses.fields(first_geo):
        field_name = field.name
        field_value = getattr(first_geo, field_name)
        if isinstance(field_value,
                      (array_typing.Array, array_typing.FloatScalar)):
            field_values = [getattr(geo, field_name) for geo in geometries]
            stacked_data[field_name] = np.stack(field_values)
        else:
            stacked_data[field_name] = field_value
    return first_geo.__class__(**stacked_data)


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
