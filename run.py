from collections.abc import Sequence, Set
from fusion_surrogates.qlknn import qlknn_model
from jax import numpy as jnp
from typing import Annotated, Any, Final, Literal, Mapping, Sequence, TypeAlias, TypeVar
from typing_extensions import override
import chex
import copy
import dataclasses
import enum
import functools
import immutabledict
import inspect
import itertools
import jax
import jaxtyping as jt
import logging
import numpy as np
import os
import pydantic
import scipy
import threading
import typing
import typing_extensions
import matplotlib.pyplot as plt


class g:
    pass


g.evolving_names = "T_i", "T_e", "psi", "n_e"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") +
    " --xla_backend_extra_options=xla_cpu_flatten_after_fusion")
jax.config.update("jax_enable_x64", True)
T = TypeVar("T")
Array: TypeAlias = jax.Array | np.ndarray
FloatScalar: TypeAlias = jt.Float[Array | float, ""]
JAX_STATIC: Final[str] = "_pydantic_jax_static_field"
_interp_fn = jax.jit(jnp.interp)


@jax.tree_util.register_pytree_node_class
class InterpolatedVarSingleAxis:

    def __init__(self, value, interpolation_mode):
        self._value = value
        self.xs, self.ys = value

    def tree_flatten(self):
        static_params = {
            "interpolation_mode": None,
        }
        return (self._value, static_params)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children, **aux_data)

    def get_value(self, x):
        x_shape = getattr(x, "shape", ())
        is_jax = isinstance(x, jax.Array)
        interp = _interp_fn if is_jax else np.interp
        full = jnp.full if is_jax else np.full
        match self.ys.ndim:
            case 1:
                if self.ys.size == 1:
                    if x_shape == ():
                        return self.ys[0]
                    else:
                        return full(x_shape, self.ys[0], dtype=self.ys.dtype)
                else:
                    return interp(x, self.xs, self.ys)
            case 2:
                return self.ys[0]


@jax.tree_util.register_pytree_node_class
class InterpolatedVarTimeRho:

    def __init__(self, values, rho_norm):
        sorted_indices = np.array(sorted(values.keys()))
        rho_norm_interpolated_values = np.stack(
            [
                InterpolatedVarSingleAxis(values[t], None).get_value(rho_norm)
                for t in sorted_indices
            ],
            axis=0,
        )
        self._time_interpolated_var = InterpolatedVarSingleAxis(
            value=(sorted_indices, rho_norm_interpolated_values),
            interpolation_mode=None,
        )

    def tree_flatten(self):
        children = (self._time_interpolated_var, )
        aux_data = (None, None)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(InterpolatedVarTimeRho)
        obj._time_interpolated_var = children[0]
        return obj

    def get_value(self, x):
        return self._time_interpolated_var.get_value(x)


class BaseModelFrozen(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    def __new__(cls, *unused_args, **unused_kwargs):
        try:
            registered_cls = jax.tree_util.register_pytree_node_class(cls)
        except ValueError:
            registered_cls = cls
        return super().__new__(registered_cls)

    @classmethod
    @functools.cache
    def _jit_dynamic_kwarg_names(cls):
        return tuple(name for name in cls.model_fields.keys()
                     if JAX_STATIC not in cls.model_fields[name].metadata)

    @classmethod
    @functools.cache
    def _jit_static_kwarg_names(cls):
        return tuple(name for name in cls.model_fields.keys()
                     if JAX_STATIC in cls.model_fields[name].metadata)

    def tree_flatten(self):
        static_names = self._jit_static_kwarg_names()
        dynamic_names = self._jit_dynamic_kwarg_names()
        static_children = {name: getattr(self, name) for name in static_names}
        dynamic_children = [getattr(self, name) for name in dynamic_names]
        return (dynamic_children, static_children)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamic_kwargs = {
            name: value
            for name, value in zip(
                cls._jit_dynamic_kwarg_names(), children, strict=True)
        }
        return cls.model_construct(**(dynamic_kwargs | aux_data))

    @classmethod
    def from_dict(cls, cfg):
        return cls.model_validate(cfg)


ValueType: TypeAlias = Any


class TimeVaryingArray(BaseModelFrozen):
    value: ValueType
    grid: Any = None

    def tree_flatten(self):
        children = (
            self.value,
            self._get_cached_interpolated_param_cell,
            self._get_cached_interpolated_param_face,
            self._get_cached_interpolated_param_face_right,
        )
        aux_data = (self.grid, )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.model_construct(
            value=children[0],
            grid=aux_data[0],
        )
        obj._get_cached_interpolated_param_cell = children[1]
        obj._get_cached_interpolated_param_face = children[2]
        obj._get_cached_interpolated_param_face_right = children[3]
        return obj

    @functools.cached_property
    def right_boundary_conditions_defined(self):
        return False

    def get_value(self, t, grid_type="cell"):
        match grid_type:
            case "cell":
                return self._get_cached_interpolated_param_cell.get_value(t)
            case "face":
                return self._get_cached_interpolated_param_face.get_value(t)

    @pydantic.field_validator("value", mode="after")
    @classmethod
    def _valid_value(cls, value):
        value = dict(sorted(value.items()))
        return value

    @pydantic.model_validator(mode="before")
    @classmethod
    def _conform_data(cls, data):
        if isinstance(data, dict):
            data.pop("_get_cached_interpolated_param_cell_centers", None)
            data.pop("_get_cached_interpolated_param_face_centers", None)
            data.pop("_get_cached_interpolated_param_face_right_centers", None)
        if isinstance(data, (float, int)):
            data = {0.0: {0.0: data}}
        value = {}
        for t, v in data.items():
            x = np.array(list(v.keys()), dtype=np.float64)
            y = np.array(list(v.values()), dtype=np.float64)
            value[t] = (x, y)
        return dict(value=value, )

    @functools.cached_property
    def _get_cached_interpolated_param_cell(self, ):
        return InterpolatedVarTimeRho(
            self.value,
            rho_norm=g.cell_centers,
        )

    @functools.cached_property
    def _get_cached_interpolated_param_face(self):
        return InterpolatedVarTimeRho(
            self.value,
            rho_norm=g.face_centers,
        )

    @functools.cached_property
    def _get_cached_interpolated_param_face_right(self):
        return InterpolatedVarTimeRho(
            self.value,
            rho_norm=g.face_centers[-1],
        )


class TimeVaryingScalar(BaseModelFrozen):
    time: Any
    value: Any
    interpolation_mode: Any

    def get_value(self, t):
        return self._get_cached_interpolated_param.get_value(t)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _conform_data(cls, data):
        time = np.array([0.0], dtype=np.float64)
        value = np.array([data], dtype=np.float64)
        sort_order = np.argsort(time)
        time = time[sort_order]
        value = value[sort_order]
        return dict(
            time=time,
            value=value,
            interpolation_mode=None,
        )

    @functools.cached_property
    def _get_cached_interpolated_param(self):
        return InterpolatedVarSingleAxis(
            value=(self.time, self.value),
            interpolation_mode=self.interpolation_mode,
        )


def _interval(time_varying_scalar, lower_bound, upper_bound):
    return time_varying_scalar


UnitIntervalTimeVaryingScalar: TypeAlias = typing_extensions.Annotated[
    TimeVaryingScalar,
    pydantic.AfterValidator(
        functools.partial(_interval, lower_bound=0.0, upper_bound=1.0)),
]
UnitInterval: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]
OpenUnitInterval: TypeAlias = Annotated[float, pydantic.Field(gt=0.0, lt=1.0)]
ValidatedDefault = functools.partial(pydantic.Field, validate_default=True)
BooleanNumeric = Any
thread_context = threading.local()


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def cond(cond_val, true_fun, false_fun, *operands):
    if cond_val:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


_TOLERANCE: Final[float] = 1e-6


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IonProperties:
    symbol: Any
    name: Any
    A: Any
    Z: Any


g.keV_to_J = 1e3 * 1.602176634e-19
g.eV_to_J = 1.602176634e-19
g.m_amu = 1.6605390666e-27
g.q_e = 1.602176634e-19
g.m_e = 9.1093837e-31
g.epsilon_0 = 8.85418782e-12
g.mu_0 = 4 * jnp.pi * 1e-7
g.k_B = 1.380649e-23
g.eps = 1e-7
ION_PROPERTIES: Final[tuple[IonProperties, ...]] = (
    IonProperties(symbol="H", name="Hydrogen", A=1.008, Z=1.0),
    IonProperties(symbol="D", name="Deuterium", A=2.0141, Z=1.0),
    IonProperties(symbol="T", name="Tritium", A=3.0160, Z=1.0),
    IonProperties(symbol="He3", name="Helium-3", A=3.0160, Z=2.0),
    IonProperties(symbol="He4", name="Helium-4", A=4.0026, Z=2.0),
    IonProperties(symbol="Li", name="Lithium", A=5.3917, Z=3.0),
    IonProperties(symbol="Be", name="Beryllium", A=9.0122, Z=4.0),
    IonProperties(symbol="C", name="Carbon", A=12.011, Z=6.0),
    IonProperties(symbol="N", name="Nitrogen", A=14.007, Z=7.0),
    IonProperties(symbol="O", name="Oxygen", A=15.999, Z=8.0),
    IonProperties(symbol="Ne", name="Neon", A=20.180, Z=10.0),
    IonProperties(symbol="Ar", name="Argon", A=39.95, Z=18.0),
    IonProperties(symbol="Kr", name="Krypton", A=83.798, Z=36.0),
    IonProperties(symbol="Xe", name="Xenon", A=131.29, Z=54.0),
    IonProperties(symbol="W", name="Tungsten", A=183.84, Z=74.0),
)
ION_PROPERTIES_DICT: Final[Mapping[
    str, IonProperties]] = immutabledict.immutabledict(
        {v.symbol: v
         for v in ION_PROPERTIES})
ION_SYMBOLS = frozenset(ION_PROPERTIES_DICT.keys())


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsSlice:
    plasma_composition: Any
    profile_conditions: Any
    sources: Any
    transport: Any


IonMapping: TypeAlias = Mapping[str, TimeVaryingScalar]


@jax.jit
def cell_integration(x):
    return jnp.sum(x * g.geo.drho_norm)


def area_integration(value, geo):
    return cell_integration(value * geo.spr)


def volume_integration(value, geo):
    return cell_integration(value * geo.vpr)


def line_average(value):
    return cell_integration(value)


def volume_average(value, geo):
    return cell_integration(value * geo.vpr) / geo.volume_face[-1]


@chex.dataclass(frozen=True)
class CellVariable:
    value: Any
    dr: Any
    left_face_constraint: Any = None
    right_face_constraint: Any = None
    left_face_grad_constraint: Any = dataclasses.field(
        default_factory=lambda: jnp.zeros(()))
    right_face_grad_constraint: Any = dataclasses.field(
        default_factory=lambda: jnp.zeros(()))

    def face_grad(self, x=None):
        if x is None:
            forward_difference = jnp.diff(self.value) / self.dr
        else:
            forward_difference = jnp.diff(self.value) / jnp.diff(x)

        def constrained_grad(face, grad, cell, right):
            if face is not None:
                if x is None:
                    dx = self.dr
                else:
                    dx = x[-1] - x[-2] if right else x[1] - x[0]
                sign = -1 if right else 1
                return sign * (cell - face) / (0.5 * dx)
            else:
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

    def _left_face_value(self):
        value = self.value[..., 0:1]
        return value

    def _right_face_value(self):
        if self.right_face_constraint is not None:
            value = self.right_face_constraint
            value = jnp.expand_dims(value, axis=-1)
        else:
            value = (
                self.value[..., -1:] +
                jnp.expand_dims(self.right_face_grad_constraint, axis=-1) *
                jnp.expand_dims(self.dr, axis=-1) / 2)
        return value

    def face_value(self):
        inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
        return jnp.concatenate(
            [self._left_face_value(), inner,
             self._right_face_value()],
            axis=-1)

    def grad(self):
        face = self.face_value()
        return jnp.diff(face) / jnp.expand_dims(self.dr, axis=-1)

    def cell_plus_boundaries(self):
        right_value = self._right_face_value()
        left_value = self._left_face_value()
        return jnp.concatenate(
            [left_value, self.value, right_value],
            axis=-1,
        )


def make_convection_terms(v_face, d_face, var):
    eps = 1e-20
    is_neg = d_face < 0.0
    nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
    d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))
    half = jnp.array([0.5], dtype=jnp.float64)
    ones = jnp.ones_like(v_face[1:-1])
    scale = jnp.concatenate((half, ones, half))
    ratio = scale * var.dr * v_face / d_face
    left_peclet = -ratio[:-1]
    right_peclet = ratio[1:]

    def peclet_to_alpha(p):
        eps = 1e-3
        p = jnp.where(jnp.abs(p) < eps, eps, p)
        alpha_pg10 = (p - 1) / p
        alpha_p0to10 = ((p - 1) + (1 - p / 10)**5) / p
        alpha_pneg10to0 = ((1 + p / 10)**5 - 1) / p
        alpha_plneg10 = -1 / p
        alpha = 0.5 * jnp.ones_like(p)
        alpha = jnp.where(p > 10.0, alpha_pg10, alpha)
        alpha = jnp.where(jnp.logical_and(10.0 >= p, p > eps), alpha_p0to10,
                          alpha)
        alpha = jnp.where(jnp.logical_and(-eps > p, p >= -10), alpha_pneg10to0,
                          alpha)
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
    mat = jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)
    vec = jnp.zeros_like(diag)
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / var.dr
    vec_value = -v_face[0] * (1.0 -
                              left_alpha[0]) * var.left_face_grad_constraint
    mat = mat.at[0, 0].set(mat_value)
    vec = vec.at[0].set(vec_value)
    if var.right_face_constraint is not None:
        mat_value = (v_face[-2] * left_alpha[-1] + v_face[-1] *
                     (1.0 - 2.0 * right_alpha[-1])) / var.dr
        vec_value = (-2.0 * v_face[-1] * (1.0 - right_alpha[-1]) *
                     var.right_face_constraint) / var.dr
    else:
        mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / var.dr
        vec_value = (-v_face[-1] * (1.0 - right_alpha[-1]) *
                     var.right_face_grad_constraint)
    mat = mat.at[-1, -1].set(mat_value)
    vec = vec.at[-1].set(vec_value)
    return mat, vec


def make_diffusion_terms(d_face, var):
    denom = var.dr**2
    diag = jnp.asarray(-d_face[1:] - d_face[:-1])
    off = d_face[1:-1]
    vec = jnp.zeros_like(diag)
    diag = diag.at[0].set(-d_face[1])
    vec = vec.at[0].set(-d_face[0] * var.left_face_grad_constraint / var.dr)
    if var.right_face_constraint is not None:
        diag = diag.at[-1].set(-2 * d_face[-1] - d_face[-2])
        vec = vec.at[-1].set(2 * d_face[-1] * var.right_face_constraint /
                             denom)
    else:
        diag = diag.at[-1].set(-d_face[-2])
        vec = vec.at[-1].set(d_face[-1] * var.right_face_grad_constraint /
                             var.dr)
    mat = (jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)) / denom
    return mat, vec


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CoreProfiles:
    T_i: Any
    T_e: Any
    psi: Any
    psidot: Any
    n_e: Any
    n_i: Any
    n_impurity: Any
    impurity_fractions: Any
    q_face: Any
    s_face: Any
    v_loop_lcfs: Any
    Z_i: Any
    Z_i_face: Any
    A_i: Any
    Z_impurity: Any
    Z_impurity_face: Any
    A_impurity: Any
    A_impurity_face: Any
    Z_eff: Any
    Z_eff_face: Any
    sigma: Any
    sigma_face: Any
    j_total: Any
    j_total_face: Any
    Ip_profile_face: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CoreTransport:
    chi_face_ion: Any
    chi_face_el: Any
    d_face_el: Any
    v_face_el: Any
    chi_face_el_bohm: Any = None
    chi_face_el_gyrobohm: Any = None
    chi_face_ion_bohm: Any = None
    chi_face_ion_gyrobohm: Any = None
    chi_neo_i: Any = None
    chi_neo_e: Any = None
    D_neo_e: Any = None
    V_neo_e: Any = None
    V_neo_ware_e: Any = None

    def chi_max(self, geo):
        return jnp.maximum(
            jnp.max(self.chi_face_ion * geo.g1_over_vpr2_face),
            jnp.max(self.chi_face_el * geo.g1_over_vpr2_face),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SolverNumericOutputs:
    solver_error_state: Any = 0


def face_to_cell(face):
    return 0.5 * (face[:-1] + face[1:])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometry:
    Phi: Any
    Phi_face: Any
    R_major: Any
    a_minor: Any
    B_0: Any
    volume: Any
    volume_face: Any
    area: Any
    area_face: Any
    vpr: Any
    vpr_face: Any
    spr: Any
    spr_face: Any
    delta_face: Any
    elongation: Any
    elongation_face: Any
    g0: Any
    g0_face: Any
    g1: Any
    g1_face: Any
    g2: Any
    g2_face: Any
    g3: Any
    g3_face: Any
    gm4: Any
    gm4_face: Any
    gm5: Any
    gm5_face: Any
    g2g3_over_rhon: Any
    g2g3_over_rhon_face: Any
    g2g3_over_rhon_hires: Any
    F: Any
    F_face: Any
    F_hires: Any
    R_in: Any
    R_in_face: Any
    R_out: Any
    R_out_face: Any
    spr_hires: Any
    rho_hires_norm: Any
    rho_hires: Any
    Phi_b_dot: Any
    _z_magnetic_axis: Any
    Ip_profile_face: Any
    psi: Any
    psi_from_Ip: Any
    psi_from_Ip_face: Any
    j_total: Any
    j_total_face: Any
    delta_upper_face: Any
    delta_lower_face: Any

    @property
    def q_correction_factor(self):
        return jnp.where(
            False,
            1.25,
            1,
        )

    @property
    def rho_norm(self):
        return g.cell_centers

    @property
    def rho_face_norm(self):
        return g.face_centers

    @property
    def drho_norm(self):
        return jnp.array(g.dx)

    @property
    def rho_face(self):
        return self.rho_face_norm * jnp.expand_dims(self.rho_b, axis=-1)

    @property
    def rho(self):
        return self.rho_norm * jnp.expand_dims(self.rho_b, axis=-1)

    @property
    def epsilon_face(self):
        return (self.R_out_face - self.R_in_face) / (self.R_out_face +
                                                     self.R_in_face)

    @property
    def rho_b(self):
        return jnp.sqrt(self.Phi_b / np.pi / g.B_0)

    @property
    def Phi_b(self):
        return self.Phi_face[..., -1]

    @property
    def g0_over_vpr_face(self):
        bulk = self.g0_face[..., 1:] / self.vpr_face[..., 1:]
        first_element = jnp.ones_like(self.rho_b) / self.rho_b
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    @property
    def g1_over_vpr_face(self):
        bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:]
        first_element = jnp.zeros_like(self.rho_b)
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)

    @property
    def g1_over_vpr2_face(self):
        bulk = self.g1_face[..., 1:] / self.vpr_face[..., 1:]**2
        first_element = jnp.ones_like(self.rho_b) / self.rho_b**2
        return jnp.concatenate([jnp.expand_dims(first_element, axis=-1), bulk],
                               axis=-1)


def calculate_scaling_law_confinement_time(geo, core_profiles, Ploss,
                                           scaling_law):
    scaling_params = {
        "H89P": {
            "prefactor": 0.038128,
            "Ip_exponent": 0.85,
            "B_exponent": 0.2,
            "line_avg_n_e_exponent": 0.1,
            "Ploss_exponent": -0.5,
            "R_exponent": 1.5,
            "inverse_aspect_ratio_exponent": 0.3,
            "elongation_exponent": 0.5,
            "effective_mass_exponent": 0.50,
            "triangularity_exponent": 0.0,
        },
        "H98": {
            "prefactor": 0.0562,
            "Ip_exponent": 0.93,
            "B_exponent": 0.15,
            "line_avg_n_e_exponent": 0.41,
            "Ploss_exponent": -0.69,
            "R_exponent": 1.97,
            "inverse_aspect_ratio_exponent": 0.58,
            "elongation_exponent": 0.78,
            "effective_mass_exponent": 0.19,
            "triangularity_exponent": 0.0,
        },
        "H97L": {
            "prefactor": 0.023,
            "Ip_exponent": 0.96,
            "B_exponent": 0.03,
            "line_avg_n_e_exponent": 0.4,
            "Ploss_exponent": -0.73,
            "R_exponent": 1.83,
            "inverse_aspect_ratio_exponent": -0.06,
            "elongation_exponent": 0.64,
            "effective_mass_exponent": 0.20,
            "triangularity_exponent": 0.0,
        },
        "H20": {
            "prefactor": 0.053,
            "Ip_exponent": 0.98,
            "B_exponent": 0.22,
            "line_avg_n_e_exponent": 0.24,
            "Ploss_exponent": -0.669,
            "R_exponent": 1.71,
            "inverse_aspect_ratio_exponent": 0.35,
            "elongation_exponent": 0.80,
            "effective_mass_exponent": 0.20,
            "triangularity_exponent": 0.36,
        },
    }
    params = scaling_params[scaling_law]
    scaled_Ip = core_profiles.Ip_profile_face[-1] / 1e6
    scaled_Ploss = Ploss / 1e6
    B = geo.B_0
    line_avg_n_e = line_average(core_profiles.n_e.value) / 1e19
    R = g.R_major
    inverse_aspect_ratio = geo.a_minor / g.R_major
    elongation = geo.area_face[-1] / (jnp.pi * geo.a_minor**2)
    effective_mass = core_profiles.A_i
    triangularity = geo.delta_face[-1]
    tau_scaling = (
        params["prefactor"] * scaled_Ip**params["Ip_exponent"] *
        B**params["B_exponent"] *
        line_avg_n_e**params["line_avg_n_e_exponent"] *
        scaled_Ploss**params["Ploss_exponent"] * R**params["R_exponent"] *
        inverse_aspect_ratio**params["inverse_aspect_ratio_exponent"] *
        elongation**params["elongation_exponent"] *
        effective_mass**params["effective_mass_exponent"] *
        (1 + triangularity)**params["triangularity_exponent"])
    return tau_scaling


def calc_q_face(geo, psi):
    inv_iota = jnp.abs(
        (2 * geo.Phi_b * geo.rho_face_norm[1:]) / psi.face_grad()[1:])
    inv_iota0 = jnp.expand_dims(
        jnp.abs((2 * geo.Phi_b * geo.drho_norm) / psi.face_grad()[1]), 0)
    q_face = jnp.concatenate([inv_iota0, inv_iota])
    return q_face * geo.q_correction_factor


def calc_j_total(geo, psi):
    Ip_profile_face = (psi.face_grad() * geo.g2g3_over_rhon_face * geo.F_face /
                       geo.Phi_b / (16 * jnp.pi**3 * g.mu_0))
    Ip_profile = (psi.grad() * geo.g2g3_over_rhon * geo.F / geo.Phi_b /
                  (16 * jnp.pi**3 * g.mu_0))
    dI_drhon_face = jnp.gradient(Ip_profile_face, geo.rho_face_norm)
    dI_drhon = jnp.gradient(Ip_profile, geo.rho_norm)
    j_total_bulk = dI_drhon[1:] / geo.spr[1:]
    j_total_face_bulk = dI_drhon_face[1:] / geo.spr_face[1:]
    j_total_axis = j_total_bulk[0] - (j_total_bulk[1] - j_total_bulk[0])
    j_total = jnp.concatenate([jnp.array([j_total_axis]), j_total_bulk])
    j_total_face = jnp.concatenate(
        [jnp.array([j_total_axis]), j_total_face_bulk])
    return j_total, j_total_face, Ip_profile_face


def calc_s_face(geo, psi):
    iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))
    iota_scaled0 = jnp.expand_dims(jnp.abs(psi.face_grad()[1] / geo.drho_norm),
                                   axis=0)
    iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
    s_face = (-geo.rho_face_norm *
              jnp.gradient(iota_scaled, geo.rho_face_norm) / iota_scaled)
    return s_face


def calculate_psi_grad_constraint_from_Ip(Ip, geo):
    return (Ip * (16 * jnp.pi**3 * g.mu_0 * geo.Phi_b) /
            (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1]))


def calculate_psidot_from_psi_sources(*, psi_sources, sigma, psi, geo):
    toc_psi = (1.0 / g.resistivity_multiplier * geo.rho_norm * sigma * g.mu_0 *
               16 * jnp.pi**2 * geo.Phi_b**2 / geo.F**2)
    d_face_psi = geo.g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    psi_sources += (8.0 * jnp.pi**2 * g.mu_0 * geo.Phi_b_dot * geo.Phi_b *
                    geo.rho_norm**2 * sigma / geo.F**2 * psi.grad())
    diffusion_mat, diffusion_vec = make_diffusion_terms(d_face_psi, psi)
    conv_mat, conv_vec = make_convection_terms(v_face_psi, d_face_psi, psi)
    c_mat = diffusion_mat + conv_mat
    c = diffusion_vec + conv_vec
    c += psi_sources
    psidot = (jnp.dot(c_mat, psi.value) + c) / toc_psi
    return psidot


def calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff):
    return (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))


def calculate_pressure(core_profiles):
    pressure_thermal_el = CellVariable(
        value=core_profiles.n_e.value * core_profiles.T_e.value * g.keV_to_J,
        dr=core_profiles.n_e.dr,
        right_face_constraint=core_profiles.n_e.right_face_constraint *
        core_profiles.T_e.right_face_constraint * g.keV_to_J,
        right_face_grad_constraint=None,
    )
    pressure_thermal_ion = CellVariable(
        value=core_profiles.T_i.value * g.keV_to_J *
        (core_profiles.n_i.value + core_profiles.n_impurity.value),
        dr=core_profiles.n_i.dr,
        right_face_constraint=core_profiles.T_i.right_face_constraint *
        g.keV_to_J * (core_profiles.n_i.right_face_constraint +
                      core_profiles.n_impurity.right_face_constraint),
        right_face_grad_constraint=None,
    )
    pressure_thermal_tot = CellVariable(
        value=pressure_thermal_el.value + pressure_thermal_ion.value,
        dr=pressure_thermal_el.dr,
        right_face_constraint=pressure_thermal_el.right_face_constraint +
        pressure_thermal_ion.right_face_constraint,
        right_face_grad_constraint=None,
    )
    return (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    )


def calc_pprime(core_profiles):
    _, _, p_total = calculate_pressure(core_profiles)
    psi = core_profiles.psi.face_value()
    n_e = core_profiles.n_e.face_value()
    n_i = core_profiles.n_i.face_value()
    n_impurity = core_profiles.n_impurity.face_value()
    T_i = core_profiles.T_i.face_value()
    T_e = core_profiles.T_e.face_value()
    dne_drhon = core_profiles.n_e.face_grad()
    dni_drhon = core_profiles.n_i.face_grad()
    dnimp_drhon = core_profiles.n_impurity.face_grad()
    dti_drhon = core_profiles.T_i.face_grad()
    dte_drhon = core_profiles.T_e.face_grad()
    dpsi_drhon = core_profiles.psi.face_grad()
    dptot_drhon = g.keV_to_J * (n_e * dte_drhon + n_i * dti_drhon +
                                n_impurity * dti_drhon + dne_drhon * T_e +
                                dni_drhon * T_i + dnimp_drhon * T_i)
    p_total_face = p_total.face_value()
    pprime_face_axis = jnp.expand_dims(
        (2 * p_total_face[0] - 5 * p_total_face[1] + 4 * p_total_face[2] -
         p_total_face[3]) / (2 * psi[0] - 5 * psi[1] + 4 * psi[2] - psi[3]),
        axis=0,
    )
    pprime_face = jnp.concatenate(
        [pprime_face_axis, dptot_drhon[1:] / dpsi_drhon[1:]])
    return pprime_face


def calculate_greenwald_fraction(n_e_avg, core_profiles, geo):
    gw_limit = core_profiles.Ip_profile_face[-1] * 1e-6 / (jnp.pi *
                                                           geo.a_minor**2)
    fgw = n_e_avg / (gw_limit * 1e20)
    return fgw


def calculate_log_lambda_ei(T_e, n_e):
    T_e_ev = T_e * 1e3
    return 31.3 - 0.5 * jnp.log(n_e) + jnp.log(T_e_ev)


def _calculate_log_tau_e_Z1(T_e, n_e, log_lambda_ei):
    return (jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei)) -
            4 * jnp.log(g.q_e) + 0.5 * jnp.log(g.m_e / 2.0) +
            2 * jnp.log(g.epsilon_0) + 1.5 * jnp.log(T_e * g.keV_to_J))


_MAVRIN_Z_COEFFS = immutabledict.immutabledict({
    "C":
    np.array([
        [-7.2007e00, -1.2217e01, -7.3521e00, -1.7632e00, 5.8588e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 6.0000e00],
    ]),
    "N":
    np.array([
        [0.0000e00, 3.3818e00, 1.8861e00, 1.5668e-01, 6.9728e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 7.0000e00],
    ]),
    "O":
    np.array([
        [0.0000e00, -1.8560e01, -3.8664e01, -2.2093e01, 4.0451e00],
        [-4.3092e00, -4.6261e-01, -3.7050e-02, 8.0180e-02, 7.9878e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 8.0000e00],
    ]),
    "Ne":
    np.array([
        [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
        [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
    ]),
    "Ar":
    np.array([
        [6.8717e00, -1.1595e01, -4.3776e01, -2.0781e01, 1.3171e01],
        [-4.8830e-02, 1.8455e00, 2.5023e00, 1.1413e00, 1.5986e01],
        [-5.9213e-01, 3.5667e00, -8.0048e00, 7.9986e00, 1.4948e01],
    ]),
    "Kr":
    np.array([
        [1.3630e02, 4.6320e02, 5.6890e02, 3.0638e02, 7.7040e01],
        [-1.0279e02, 6.8446e01, 1.5744e01, 1.5186e00, 2.4728e01],
        [-2.4682e00, 1.3215e01, -2.5703e01, 2.3443e01, 2.5368e01],
    ]),
    "Xe":
    np.array([
        [5.8178e02, 1.9967e03, 2.5189e03, 1.3973e03, 3.0532e02],
        [8.6824e01, -2.9061e01, -4.8384e01, 1.6271e01, 3.2616e01],
        [4.0756e02, -9.0008e02, 6.6739e02, -1.7259e02, 4.8066e01],
        [-1.0019e01, 7.3261e01, -1.9931e02, 2.4056e02, -5.7527e01],
    ]),
    "W":
    np.array([
        [1.6823e01, 3.4582e01, 2.1027e01, 1.6518e01, 2.6703e01],
        [-2.5887e02, -1.0577e01, 2.5532e02, -7.9611e01, 3.6902e01],
        [1.5119e01, -8.4207e01, 1.5985e02, -1.0011e02, 6.3795e01],
    ]),
})
_TEMPERATURE_INTERVALS = immutabledict.immutabledict({
    "C":
    np.array([0.7]),
    "N":
    np.array([0.7]),
    "O":
    np.array([0.3, 1.5]),
    "Ne":
    np.array([0.5, 2.0]),
    "Ar":
    np.array([0.6, 3.0]),
    "Kr":
    np.array([0.447, 4.117]),
    "Xe":
    np.array([0.3, 1.5, 8.0]),
    "W":
    np.array([1.5, 4.0]),
})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ChargeStateInfo:
    Z_avg: Any
    Z2_avg: Any
    Z_per_species: Any

    @property
    def Z_mixture(self):
        return self.Z2_avg / self.Z_avg


def calculate_average_charge_state_single_species(T_e, ion_symbol):
    if ion_symbol not in _MAVRIN_Z_COEFFS:
        return jnp.ones_like(T_e) * ION_PROPERTIES_DICT[ion_symbol].Z
    T_e_allowed_range = (0.1, 100.0)
    T_e = jnp.clip(T_e, *T_e_allowed_range)
    interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol],
                                        T_e)
    Zavg_coeffs_in_range = jnp.take(_MAVRIN_Z_COEFFS[ion_symbol],
                                    interval_indices,
                                    axis=0).transpose()
    X = jnp.log10(T_e)
    Zavg = jnp.polyval(Zavg_coeffs_in_range, X)
    return Zavg


def get_average_charge_state(ion_symbols, T_e, fractions):
    Z_per_species = jnp.stack([
        calculate_average_charge_state_single_species(T_e, ion_symbol)
        for ion_symbol in ion_symbols
    ])
    fractions = fractions if fractions.ndim == 2 else fractions[:, jnp.newaxis]
    Z_avg = jnp.sum(fractions * Z_per_species, axis=0)
    Z2_avg = jnp.sum(fractions * Z_per_species**2, axis=0)
    return ChargeStateInfo(
        Z_avg=Z_avg,
        Z2_avg=Z2_avg,
        Z_per_species=Z_per_species,
    )


def calculate_f_trap(geo):
    epsilon_effective = (
        0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) *
        geo.epsilon_face)
    aa = (1.0 - geo.epsilon_face) / (1.0 + geo.epsilon_face)
    return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
        1.0 + 2.0 * jnp.sqrt(epsilon_effective))


def calculate_L31(f_trap, nu_e_star, Z_eff):
    denom = (1.0 + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star) + 0.5 *
             (1.0 - f_trap) * nu_e_star / Z_eff)
    ft31 = f_trap / denom
    term_0 = (1 + 1.4 / (Z_eff + 1)) * ft31
    term_1 = -1.9 / (Z_eff + 1) * ft31**2
    term_2 = 0.3 / (Z_eff + 1) * ft31**3
    term_3 = 0.2 / (Z_eff + 1) * ft31**4
    return term_0 + term_1 + term_2 + term_3


def calculate_L32(f_trap, nu_e_star, Z_eff):
    ft32ee = f_trap / (1 + 0.26 * (1 - f_trap) * jnp.sqrt(nu_e_star) + 0.18 *
                       (1 - 0.37 * f_trap) * nu_e_star / jnp.sqrt(Z_eff))
    ft32ei = f_trap / (1 + (1 + 0.6 * f_trap) * jnp.sqrt(nu_e_star) + 0.85 *
                       (1 - 0.37 * f_trap) * nu_e_star * (1 + Z_eff))
    F32ee = ((0.05 + 0.62 * Z_eff) / (Z_eff * (1 + 0.44 * Z_eff)) *
             (ft32ee - ft32ee**4) + 1 / (1 + 0.22 * Z_eff) *
             (ft32ee**2 - ft32ee**4 - 1.2 *
              (ft32ee**3 - ft32ee**4)) + 1.2 / (1 + 0.5 * Z_eff) * ft32ee**4)
    F32ei = (-(0.56 + 1.93 * Z_eff) / (Z_eff * (1 + 0.44 * Z_eff)) *
             (ft32ei - ft32ei**4) + 4.95 / (1 + 2.48 * Z_eff) *
             (ft32ei**2 - ft32ei**4 - 0.55 *
              (ft32ei**3 - ft32ei**4)) - 1.2 / (1 + 0.5 * Z_eff) * ft32ei**4)
    return F32ee + F32ei


def calculate_nu_e_star(q, geo, n_e, T_e, Z_eff, log_lambda_ei):
    return (6.921e-18 * q * g.R_major * n_e * Z_eff * log_lambda_ei /
            (((T_e * 1e3)**2) * (geo.epsilon_face + g.eps)**1.5))


def calculate_nu_i_star(q, geo, n_i, T_i, Z_eff, log_lambda_ii):
    return (4.9e-18 * q * g.R_major * n_i * Z_eff**4 * log_lambda_ii /
            (((T_i * 1e3)**2) * (geo.epsilon_face + g.eps)**1.5))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class Conductivity:
    sigma: Any
    sigma_face: Any


@jax.jit
def _calculate_conductivity0(*, Z_eff_face, n_e, T_e, q_face, geo):
    f_trap = calculate_f_trap(geo)
    NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
    log_lambda_ei = calculate_log_lambda_ei(T_e.face_value(), n_e.face_value())
    sigsptz = (1.9012e04 * (T_e.face_value() * 1e3)**1.5 / Z_eff_face / NZ /
               log_lambda_ei)
    nu_e_star_face = calculate_nu_e_star(
        q=q_face,
        geo=geo,
        n_e=n_e.face_value(),
        T_e=T_e.face_value(),
        Z_eff=Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    ft33 = f_trap / (1.0 +
                     (0.55 - 0.1 * f_trap) * jnp.sqrt(nu_e_star_face) + 0.45 *
                     (1.0 - f_trap) * nu_e_star_face / (Z_eff_face**1.5))
    signeo_face = 1.0 - ft33 * (1.0 + 0.36 / Z_eff_face - ft33 *
                                (0.59 / Z_eff_face - 0.23 / Z_eff_face * ft33))
    sigma_face = sigsptz * signeo_face
    sigmaneo_cell = face_to_cell(sigma_face)
    return Conductivity(
        sigma=sigmaneo_cell,
        sigma_face=sigma_face,
    )


def calculate_conductivity(geometry, core_profiles):
    result = _calculate_conductivity0(
        Z_eff_face=core_profiles.Z_eff_face,
        n_e=core_profiles.n_e,
        T_e=core_profiles.T_e,
        q_face=core_profiles.q_face,
        geo=geometry,
    )
    return Conductivity(sigma=result.sigma, sigma_face=result.sigma_face)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
    j_bootstrap: jax.Array
    j_bootstrap_face: jax.Array

    @classmethod
    def zeros(cls, geometry):
        return cls(
            j_bootstrap=jnp.zeros_like(geometry.rho_norm),
            j_bootstrap_face=jnp.zeros_like(geometry.rho_face_norm),
        )


class SauterModel:

    def calculate_bootstrap_current(self, geometry, core_profiles):
        result = _calculate_bootstrap_current(
            Z_eff_face=core_profiles.Z_eff_face,
            Z_i_face=core_profiles.Z_i_face,
            n_e=core_profiles.n_e,
            n_i=core_profiles.n_i,
            T_e=core_profiles.T_e,
            T_i=core_profiles.T_i,
            psi=core_profiles.psi,
            q_face=core_profiles.q_face,
            geo=geometry,
        )
        return BootstrapCurrent(
            j_bootstrap=result.j_bootstrap,
            j_bootstrap_face=result.j_bootstrap_face,
        )


class SauterModelConfig(BaseModelFrozen):
    model_name: Annotated[Literal["sauter"], JAX_STATIC] = "sauter"
    bootstrap_multiplier: float = 1.0

    def build_model(self):
        return SauterModel()


@jax.jit
def _calculate_bootstrap_current(*, Z_eff_face, Z_i_face, n_e, n_i, T_e, T_i,
                                 psi, q_face, geo):
    f_trap = calculate_f_trap(geo)
    log_lambda_ei = calculate_log_lambda_ei(T_e.face_value(), n_e.face_value())
    T_i_ev = T_i.face_value() * 1e3
    log_lambda_ii = (30.0 - 0.5 * jnp.log(n_i.face_value()) +
                     1.5 * jnp.log(T_i_ev) - 3.0 * jnp.log(Z_i_face))
    nu_e_star = calculate_nu_e_star(
        q=q_face,
        geo=geo,
        n_e=n_e.face_value(),
        T_e=T_e.face_value(),
        Z_eff=Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    nu_i_star = calculate_nu_i_star(
        q=q_face,
        geo=geo,
        n_i=n_i.face_value(),
        T_i=T_i.face_value(),
        Z_eff=Z_eff_face,
        log_lambda_ii=log_lambda_ii,
    )
    bootstrap_multiplier = 1.0
    L31 = calculate_L31(f_trap, nu_e_star, Z_eff_face)
    L32 = calculate_L32(f_trap, nu_e_star, Z_eff_face)
    L34 = _calculate_L34(f_trap, nu_e_star, Z_eff_face)
    alpha = _calculate_alpha(f_trap, nu_i_star)
    prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0
    pe = n_e.face_value() * T_e.face_value() * 1e3 * 1.6e-19
    pi = n_i.face_value() * T_i.face_value() * 1e3 * 1.6e-19
    dpsi_drnorm = psi.face_grad()
    dlnne_drnorm = n_e.face_grad() / n_e.face_value()
    dlnni_drnorm = n_i.face_grad() / n_i.face_value()
    dlnte_drnorm = T_e.face_grad() / T_e.face_value()
    dlnti_drnorm = T_i.face_grad() / T_i.face_value()
    global_coeff = prefactor[1:] / dpsi_drnorm[1:]
    global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])
    necoeff = L31 * pe
    nicoeff = L31 * pi
    tecoeff = (L31 + L32) * pe
    ticoeff = (L31 + alpha * L34) * pi
    j_bootstrap_face = global_coeff * (
        necoeff * dlnne_drnorm + nicoeff * dlnni_drnorm +
        tecoeff * dlnte_drnorm + ticoeff * dlnti_drnorm)
    j_bootstrap = face_to_cell(j_bootstrap_face)
    return BootstrapCurrent(
        j_bootstrap=j_bootstrap,
        j_bootstrap_face=j_bootstrap_face,
    )


def _calculate_L34(f_trap, nu_e_star, Z_eff):
    ft34 = f_trap / (1.0 + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star) + 0.5 *
                     (1.0 - 0.5 * f_trap) * nu_e_star / Z_eff)
    return ((1 + 1.4 / (Z_eff + 1)) * ft34 - 1.9 / (Z_eff + 1) * ft34**2 +
            0.3 / (Z_eff + 1) * ft34**3 + 0.2 / (Z_eff + 1) * ft34**4)


def _calculate_alpha(f_trap, nu_i_star):
    alpha0 = -1.17 * (1 - f_trap) / (1 - 0.22 * f_trap - 0.19 * f_trap**2)
    alpha = ((alpha0 + 0.25 * (1 - f_trap**2) * jnp.sqrt(nu_i_star)) /
             (1 + 0.5 * jnp.sqrt(nu_i_star)) + 0.315 * nu_i_star**2 *
             f_trap**6) / (1 + 0.15 * nu_i_star**2 * f_trap**6)
    return alpha


def exponential_profile(geo, *, decay_start, width, total):
    r = geo.rho_norm
    S = jnp.exp(-(decay_start - r) / width)
    C = total / volume_integration(S, geo)
    return C * S


def gaussian_profile(geo, *, center, width, total):
    r = geo.rho_norm
    S = jnp.exp(-((r - center)**2) / (2 * width**2))
    C = total / volume_integration(S, geo)
    return C * S


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QeiInfo:
    qei_coef: jax.Array
    implicit_ii: jax.Array
    explicit_i: jax.Array
    implicit_ee: jax.Array
    explicit_e: jax.Array
    implicit_ie: jax.Array
    implicit_ei: jax.Array

    @classmethod
    def zeros(cls, geo):
        return QeiInfo(
            qei_coef=jnp.zeros_like(geo.rho),
            implicit_ii=jnp.zeros_like(geo.rho),
            explicit_i=jnp.zeros_like(geo.rho),
            implicit_ee=jnp.zeros_like(geo.rho),
            explicit_e=jnp.zeros_like(geo.rho),
            implicit_ie=jnp.zeros_like(geo.rho),
            implicit_ei=jnp.zeros_like(geo.rho),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceProfiles:
    bootstrap_current: BootstrapCurrent
    qei: QeiInfo
    T_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    T_i: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    n_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    psi: dict[str, jax.Array] = dataclasses.field(default_factory=dict)

    def total_psi_sources(self, geo):
        total = self.bootstrap_current.j_bootstrap
        total += sum(self.psi.values())
        mu0 = g.mu_0
        prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B_0 * mu0 * geo.Phi_b / geo.F**2
        return -total * prefactor

    def total_sources(self, source_type, geo):
        source = getattr(self, source_type)
        total = sum(source.values())
        return total * geo.vpr


@enum.unique
class Mode(enum.Enum):
    ZERO = "ZERO"
    MODEL_BASED = "MODEL_BASED"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsSrc:
    prescribed_values: Any
    mode: Mode = dataclasses.field(metadata={"static": True})
    is_explicit: bool = dataclasses.field(metadata={"static": True})


class SourceModelBase(BaseModelFrozen):
    mode: Annotated[Mode, JAX_STATIC] = Mode.ZERO
    is_explicit: Annotated[bool, JAX_STATIC] = False
    prescribed_values: tuple[TimeVaryingArray, ...] = ValidatedDefault(({
        0: {
            0: 0,
            1: 0
        }
    }, ))


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Source:
    SOURCE_NAME: typing.ClassVar[str] = "source"
    model_func: Any = None

    def get_value(
        self,
        runtime_params,
        geo,
        core_profiles,
        calculated_source_profiles,
        conductivity,
    ):
        return self.model_func(
            runtime_params,
            geo,
            self.source_name,
            core_profiles,
            calculated_source_profiles,
            conductivity,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsGcS(RuntimeParamsSrc):
    I_generic: Any
    fraction_of_total_current: Any
    gaussian_width: Any
    gaussian_location: Any
    use_absolute_current: Any


def calculate_generic_current(
    runtime_params,
    geo,
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    source_params = runtime_params.sources[source_name]
    I_generic = _calculate_I_generic(
        runtime_params,
        source_params,
    )
    generic_current_form = jnp.exp(
        -((geo.rho_norm - source_params.gaussian_location)**2) /
        (2 * source_params.gaussian_width**2))
    Cext = I_generic / area_integration(generic_current_form, geo)
    generic_current_profile = Cext * generic_current_form
    return (generic_current_profile, )


def _calculate_I_generic(runtime_params, source_params):
    return jnp.where(
        source_params.use_absolute_current,
        source_params.I_generic,
        (runtime_params.profile_conditions.Ip *
         source_params.fraction_of_total_current),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericCurrentSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "generic_current"
    model_func: Any = calculate_generic_current

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.PSI, )


class GenericCurrentSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["gaussian"], JAX_STATIC] = "gaussian"
    I_generic: TimeVaryingScalar = ValidatedDefault(3.0e6)
    fraction_of_total_current: UnitIntervalTimeVaryingScalar = ValidatedDefault(
        0.2)
    gaussian_width: TimeVaryingScalar = ValidatedDefault(0.05)
    gaussian_location: UnitIntervalTimeVaryingScalar = ValidatedDefault(0.4)
    use_absolute_current: bool = False
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return calculate_generic_current

    def build_runtime_params(self, t):
        return RuntimeParamsGcS(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            I_generic=self.I_generic.get_value(t),
            fraction_of_total_current=self.fraction_of_total_current.get_value(
                t),
            gaussian_width=self.gaussian_width.get_value(t),
            gaussian_location=self.gaussian_location.get_value(t),
            use_absolute_current=self.use_absolute_current,
        )

    def build_source(self):
        return GenericCurrentSource(model_func=self.model_func)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsGeIO(RuntimeParamsSrc):
    gaussian_width: Any
    gaussian_location: Any
    P_total: Any
    electron_heat_fraction: Any
    absorption_fraction: Any


def calc_generic_heat_source(
    geo,
    gaussian_location,
    gaussian_width,
    P_total,
    electron_heat_fraction,
    absorption_fraction,
):
    absorbed_power = P_total * absorption_fraction
    profile = gaussian_profile(geo,
                               center=gaussian_location,
                               width=gaussian_width,
                               total=absorbed_power)
    source_ion = profile * (1 - electron_heat_fraction)
    source_el = profile * electron_heat_fraction
    return source_ion, source_el


def default_formula(
    runtime_params: RuntimeParamsSlice,
    geo: Any,
    source_name: str,
    unused_core_profiles: CoreProfiles,
    unused_calculated_source_profiles: SourceProfiles | None,
    unused_conductivity: Conductivity | None,
):
    source_params = runtime_params.sources[source_name]
    ion, el = calc_generic_heat_source(
        geo,
        source_params.gaussian_location,
        source_params.gaussian_width,
        source_params.P_total,
        source_params.electron_heat_fraction,
        source_params.absorption_fraction,
    )
    return (ion, el)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericIonElectronHeatSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "generic_heat"
    model_func: Any = default_formula

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (
            AffectedCoreProfile.TEMP_ION,
            AffectedCoreProfile.TEMP_EL,
        )


class GenericIonElHeatSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["gaussian"], JAX_STATIC] = "gaussian"
    gaussian_width: TimeVaryingScalar = ValidatedDefault(0.25)
    gaussian_location: TimeVaryingScalar = ValidatedDefault(0.0)
    P_total: TimeVaryingScalar = ValidatedDefault(120e6)
    electron_heat_fraction: TimeVaryingScalar = ValidatedDefault(0.66666)
    absorption_fraction: TimeVaryingScalar = ValidatedDefault(1.0)
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return default_formula

    def build_runtime_params(self, t):
        return RuntimeParamsGeIO(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            gaussian_width=self.gaussian_width.get_value(t),
            gaussian_location=self.gaussian_location.get_value(t),
            P_total=self.P_total.get_value(t),
            electron_heat_fraction=self.electron_heat_fraction.get_value(t),
            absorption_fraction=self.absorption_fraction.get_value(t),
        )

    def build_source(self):
        return GenericIonElectronHeatSource(model_func=self.model_func)


def calc_generic_particle_source(
    runtime_params,
    geo,
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    source_params = runtime_params.sources[source_name]
    return (gaussian_profile(
        center=source_params.deposition_location,
        width=source_params.particle_width,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericParticleSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "generic_particle"
    model_func: Any = calc_generic_particle_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPaSo(RuntimeParamsSrc):
    particle_width: Any
    deposition_location: Any
    S_total: Any


class GenericParticleSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["gaussian"], JAX_STATIC] = "gaussian"
    particle_width: TimeVaryingScalar = ValidatedDefault(0.25)
    deposition_location: TimeVaryingScalar = ValidatedDefault(0.0)
    S_total: TimeVaryingScalar = ValidatedDefault(1e22)
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return calc_generic_particle_source

    def build_runtime_params(self, t):
        return RuntimeParamsPaSo(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            particle_width=self.particle_width.get_value(t),
            deposition_location=self.deposition_location.get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self):
        return GenericParticleSource(model_func=self.model_func)


def calc_pellet_source(
    runtime_params,
    geo,
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    source_params = runtime_params.sources[source_name]
    return (gaussian_profile(
        center=source_params.pellet_deposition_location,
        width=source_params.pellet_width,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PelletSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "pellet"
    model_func: Any = calc_pellet_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPE(RuntimeParamsSrc):
    pellet_width: Any
    pellet_deposition_location: Any
    S_total: Any


class PelletSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["gaussian"], JAX_STATIC] = "gaussian"
    pellet_width: TimeVaryingScalar = ValidatedDefault(0.1)
    pellet_deposition_location: TimeVaryingScalar = ValidatedDefault(0.85)
    S_total: TimeVaryingScalar = ValidatedDefault(2e22)
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return calc_pellet_source

    def build_runtime_params(self, t):
        return RuntimeParamsPE(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            pellet_width=self.pellet_width.get_value(t),
            pellet_deposition_location=self.pellet_deposition_location.
            get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self):
        return PelletSource(model_func=self.model_func)


def calc_fusion(geo, core_profiles, runtime_params):
    product = 1.0
    for fraction, symbol in zip(
            runtime_params.plasma_composition.main_ion.fractions,
            runtime_params.plasma_composition.main_ion_names,
    ):
        if symbol == "D" or symbol == "T":
            product *= fraction
            DT_fraction_product = product
            t_face = core_profiles.T_i.face_value()
            Efus = 17.6 * 1e3 * g.keV_to_J
            mrc2 = 1124656
            BG = 34.3827
            C1 = 1.17302e-9
            C2 = 1.51361e-2
            C3 = 7.51886e-2
            C4 = 4.60643e-3
            C5 = 1.35e-2
            C6 = -1.0675e-4
            C7 = 1.366e-5
            theta = t_face / (1.0 - (t_face * (C2 + t_face *
                                               (C4 + t_face * C6))) /
                              (1.0 + t_face * (C3 + t_face *
                                               (C5 + t_face * C7))))
            xi = (BG**2 / (4 * theta))**(1 / 3)
            logsigmav = (jnp.log(C1 * theta) +
                         0.5 * jnp.log(xi / (mrc2 * t_face**3)) - 3 * xi -
                         jnp.log(1e6))
            logPfus = (jnp.log(DT_fraction_product * Efus) +
                       2 * jnp.log(core_profiles.n_i.face_value()) + logsigmav)
            Pfus_face = jnp.exp(logPfus)
            Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])
            P_total = (jax.scipy.integrate.trapezoid(Pfus_face * geo.vpr_face,
                                                     geo.rho_face_norm) / 1e6)
            alpha_fraction = 3.5 / 17.6
            birth_energy = 3520
            alpha_mass = 4.002602
            critical_energy = 10 * alpha_mass * core_profiles.T_e.value
            energy_ratio = birth_energy / critical_energy
            x_squared = energy_ratio
            x = jnp.sqrt(x_squared)
            frac_i = (2 * ((1 / 6) * jnp.log(
                (1.0 - x + x_squared) /
                (1.0 + 2.0 * x + x_squared)) + (jnp.arctan(
                    (2.0 * x - 1.0) / jnp.sqrt(3)) + jnp.pi / 6) / jnp.sqrt(3))
                      / x_squared)
            frac_e = 1.0 - frac_i
            Pfus_i = Pfus_cell * frac_i * alpha_fraction
            Pfus_e = Pfus_cell * frac_e * alpha_fraction
    return P_total, Pfus_i, Pfus_e


def fusion_heat_model_func(
    runtime_params,
    geo,
    unused_source_name,
    core_profiles,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    _, Pfus_i, Pfus_e = calc_fusion(geo, core_profiles, runtime_params)
    return (Pfus_i, Pfus_e)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FusionHeatSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "fusion"
    model_func: Any = fusion_heat_model_func

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (
            AffectedCoreProfile.TEMP_ION,
            AffectedCoreProfile.TEMP_EL,
        )


class FusionHeatSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["bosch_hale"], JAX_STATIC] = "bosch_hale"
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return fusion_heat_model_func

    def build_runtime_params(self, t):
        return RuntimeParamsSrc(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
        )

    def build_source(self):
        return FusionHeatSource(model_func=self.model_func)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsPS(RuntimeParamsSrc):
    puff_decay_length: Any
    S_total: Any


def calc_puff_source(
    runtime_params,
    geo,
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    source_params = runtime_params.sources[source_name]
    return (exponential_profile(
        decay_start=1.0,
        width=source_params.puff_decay_length,
        total=source_params.S_total,
        geo=geo,
    ), )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "gas_puff"
    model_func: Any = calc_puff_source

    @property
    def source_name(self):
        return self.SOURCE_NAME

    @property
    def affected_core_profiles(self):
        return (AffectedCoreProfile.NE, )


class GasPuffSourceConfig(SourceModelBase):
    model_name: Annotated[Literal["exponential"], JAX_STATIC] = "exponential"
    puff_decay_length: TimeVaryingScalar = ValidatedDefault(0.05)
    S_total: TimeVaryingScalar = ValidatedDefault(1e22)
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return calc_puff_source

    def build_runtime_params(self, t):
        return RuntimeParamsPS(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
            puff_decay_length=self.puff_decay_length.get_value(t),
            S_total=self.S_total.get_value(t),
        )

    def build_source(self):
        return GasPuffSource(model_func=self.model_func)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class QeiSource(Source):
    SOURCE_NAME: typing.ClassVar[str] = "ei_exchange"

    @property
    def source_name(self):
        return self.SOURCE_NAME

    def get_qei(self, runtime_params, geo, core_profiles):
        return jax.lax.cond(
            runtime_params.sources[self.source_name].mode == Mode.MODEL_BASED,
            lambda: _model_based_qei(
                runtime_params,
                geo,
                core_profiles,
            ),
            lambda: QeiInfo.zeros(geo),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceModels:
    qei_source: QeiSource
    standard_sources: immutabledict.immutabledict[str, Source]

    @functools.cached_property
    def psi_sources(self):
        return immutabledict.immutabledict({
            name: source
            for name, source in self.standard_sources.items()
            if AffectedCoreProfile.PSI in source.affected_core_profiles
        })


def _model_based_qei(runtime_params, geo, core_profiles):
    zeros = jnp.zeros_like(geo.rho_norm)
    log_lambda_ei = calculate_log_lambda_ei(core_profiles.T_e.value,
                                            core_profiles.n_e.value)
    log_tau_e_Z1 = _calculate_log_tau_e_Z1(
        core_profiles.T_e.value,
        core_profiles.n_e.value,
        log_lambda_ei,
    )
    weighted_Z_eff = (
        core_profiles.n_i.value * core_profiles.Z_i**2 / core_profiles.A_i +
        core_profiles.n_impurity.value * core_profiles.Z_impurity**2 /
        core_profiles.A_impurity) / core_profiles.n_e.value
    log_Qei_coef = (jnp.log(g.Qei_multiplier * 1.5 * core_profiles.n_e.value) +
                    jnp.log(g.keV_to_J / g.m_amu) + jnp.log(2 * g.m_e) +
                    jnp.log(weighted_Z_eff) - log_tau_e_Z1)
    qei_coef = jnp.exp(log_Qei_coef)
    implicit_ii = -qei_coef
    implicit_ee = -qei_coef
    explicit_i = zeros
    explicit_e = zeros
    implicit_ie = qei_coef
    implicit_ei = qei_coef
    return QeiInfo(
        qei_coef=qei_coef,
        implicit_ii=implicit_ii,
        explicit_i=explicit_i,
        implicit_ee=implicit_ee,
        explicit_e=explicit_e,
        implicit_ie=implicit_ie,
        implicit_ei=implicit_ei,
    )


class QeiSourceConfig(SourceModelBase):
    mode: Annotated[Mode, JAX_STATIC] = Mode.MODEL_BASED

    @property
    def model_func(self):
        return None

    def build_runtime_params(self, t):
        return RuntimeParamsSrc(
            prescribed_values=tuple(
                [v.get_value(t) for v in self.prescribed_values]),
            mode=self.mode,
            is_explicit=self.is_explicit,
        )

    def build_source(self):
        return QeiSource(model_func=self.model_func)


class Sources(BaseModelFrozen):
    ei_exchange: QeiSourceConfig = ValidatedDefault({"mode": "ZERO"})
    cyclotron_radiation: None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    fusion: FusionHeatSourceConfig | None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    gas_puff: GasPuffSourceConfig | None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    generic_current: GenericCurrentSourceConfig = ValidatedDefault(
        {"mode": "ZERO"})
    generic_heat: GenericIonElHeatSourceConfig | None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    generic_particle: GenericParticleSourceConfig | None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    impurity_radiation: None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )
    pellet: PelletSourceConfig | None = pydantic.Field(
        discriminator="model_name",
        default=None,
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _set_default_model_functions(cls, x):
        constructor_data = copy.deepcopy(x)
        for k, v in x.items():
            match k:
                case "gas_puff":
                    if "model_name" not in v:
                        constructor_data[k]["model_name"] = "exponential"
                case "generic_particle":
                    if "model_name" not in v:
                        constructor_data[k]["model_name"] = "gaussian"
                case "pellet":
                    if "model_name" not in v:
                        constructor_data[k]["model_name"] = "gaussian"
                case "fusion":
                    if "model_name" not in v:
                        constructor_data[k]["model_name"] = "bosch_hale"
                case "generic_heat":
                    if "model_name" not in v:
                        constructor_data[k]["model_name"] = "gaussian"
        return constructor_data

    def build_models(self):
        standard_sources = {}
        for k, v in dict(self).items():
            if k == "ei_exchange":
                continue
            else:
                if v is not None:
                    source = v.build_source()
                    standard_sources[k] = source
                    qei_source_model = self.ei_exchange.build_source()
        return SourceModels(
            qei_source=qei_source_model,
            standard_sources=immutabledict.immutabledict(standard_sources),
        )


@jax.jit
def build_source_profiles0(runtime_params,
                           geo,
                           core_profiles,
                           explicit_source_profiles=None,
                           conductivity=None):
    qei = QeiInfo.zeros(geo)
    bootstrap_current = BootstrapCurrent.zeros(geo)
    profiles = SourceProfiles(
        bootstrap_current=bootstrap_current,
        qei=qei,
        T_e=explicit_source_profiles.T_e if explicit_source_profiles else {},
        T_i=explicit_source_profiles.T_i if explicit_source_profiles else {},
        n_e=explicit_source_profiles.n_e if explicit_source_profiles else {},
        psi=explicit_source_profiles.psi if explicit_source_profiles else {},
    )
    build_standard_source_profiles(
        calculated_source_profiles=profiles,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=True,
        conductivity=conductivity,
    )
    return profiles


@functools.partial(
    jax.jit,
    static_argnames=[
        "explicit",
    ],
)
def build_source_profiles1(
    runtime_params,
    geo,
    core_profiles,
    explicit,
    explicit_source_profiles=None,
    conductivity=None,
):
    qei = g.source_models.qei_source.get_qei(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    bootstrap_current = g.bootstrap_current.calculate_bootstrap_current(
        geo, core_profiles)
    profiles = SourceProfiles(
        bootstrap_current=bootstrap_current,
        qei=qei,
        T_e=explicit_source_profiles.T_e if explicit_source_profiles else {},
        T_i=explicit_source_profiles.T_i if explicit_source_profiles else {},
        n_e=explicit_source_profiles.n_e if explicit_source_profiles else {},
        psi=explicit_source_profiles.psi if explicit_source_profiles else {},
    )
    build_standard_source_profiles(
        calculated_source_profiles=profiles,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=explicit,
        conductivity=conductivity,
    )
    return profiles


def build_standard_source_profiles(
    *,
    calculated_source_profiles,
    runtime_params,
    geo,
    core_profiles,
    explicit=True,
    conductivity=None,
    calculate_anyway=False,
    psi_only=False,
):

    def calculate_source(source_name, source):
        source_params = runtime_params.sources[source_name]
        if (explicit == source_params.is_explicit) | calculate_anyway:
            value = source.get_value(
                runtime_params,
                geo,
                core_profiles,
                calculated_source_profiles,
                conductivity,
            )
            for profile, affected_core_profile in zip(
                    value, source.affected_core_profiles, strict=True):
                match affected_core_profile:
                    case AffectedCoreProfile.PSI:
                        calculated_source_profiles.psi[source_name] = profile
                    case AffectedCoreProfile.NE:
                        calculated_source_profiles.n_e[source_name] = profile
                    case AffectedCoreProfile.TEMP_ION:
                        calculated_source_profiles.T_i[source_name] = profile
                    case AffectedCoreProfile.TEMP_EL:
                        calculated_source_profiles.T_e[source_name] = profile

    for source_name, source in g.source_models.psi_sources.items():
        calculate_source(source_name, source)
    if psi_only:
        return
    for source_name, source in g.source_models.standard_sources.items():
        if source_name not in g.source_models.psi_sources:
            calculate_source(source_name, source)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
    rho_norm_ped_top_idx: Any


class SetTemperatureDensityPedestalModel:

    def __call__(self, runtime_params, geo, core_profiles):
        return jax.lax.cond(
            True,
            lambda: self._call_implementation(runtime_params, geo,
                                              core_profiles),
            lambda: PedestalModelOutput(rho_norm_ped_top_idx=g.n_rho, ),
        )

    def _call_implementation(self, runtime_params, geo, core_profiles):
        return PedestalModelOutput(
            rho_norm_ped_top_idx=jnp.abs(geo.rho_norm -
                                         g.rho_norm_ped_top).argmin(), )


class PedestalConfig(BaseModelFrozen):
    rho_norm_ped_top: TimeVaryingScalar = 0.91

    def build_pedestal_model(self):
        return SetTemperatureDensityPedestalModel()


_IMPURITY_MODE_FRACTIONS: Final[str] = "fractions"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsIM:
    fractions: Any
    A_avg: Any


class IonMixture(BaseModelFrozen):
    species: IonMapping

    def build_runtime_params(self, t):
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        As = jnp.array([ION_PROPERTIES_DICT[ion].A for ion in ions])
        A_avg = jnp.sum(As * fractions)
        return RuntimeParamsIM(fractions=fractions, A_avg=A_avg)


def _impurity_before_validator(value):
    return {value: 1.0}


ImpurityMapping: TypeAlias = Annotated[
    Mapping[str, TimeVaryingArray],
    pydantic.BeforeValidator(_impurity_before_validator),
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsIF:
    fractions: Any
    fractions_face: Any
    A_avg: Any
    A_avg_face: Any


class ImpurityFractions(BaseModelFrozen):
    impurity_mode: Annotated[Literal["fractions"], JAX_STATIC] = "fractions"
    species: ImpurityMapping = ValidatedDefault({"Ne": 1.0})

    def build_runtime_params(self, t):
        ions = self.species.keys()
        fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
        fractions_face = jnp.array(
            [self.species[ion].get_value(t, grid_type="face") for ion in ions])
        As = jnp.array([ION_PROPERTIES_DICT[ion].A for ion in ions])
        A_avg = jnp.sum(As[..., jnp.newaxis] * fractions, axis=0)
        A_avg_face = jnp.sum(As[..., jnp.newaxis] * fractions_face, axis=0)
        return RuntimeParamsIF(
            fractions=fractions,
            fractions_face=fractions_face,
            A_avg=A_avg,
            A_avg_face=A_avg_face,
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _conform_impurity_data(cls, data):
        if "legacy" in data:
            del data["legacy"]
        return data


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParamsPC:
    Ip: Any
    v_loop_lcfs: Any
    T_i_right_bc: Any
    T_e_right_bc: Any
    T_e: Any
    T_i: Any
    psi: Any
    psidot: Any
    n_e: Any
    nbar: Any
    n_e_right_bc: Any
    use_v_loop_lcfs_boundary_condition: Any = dataclasses.field(
        metadata={"static": True})


class ProfileConditions(BaseModelFrozen):
    Ip: TimeVaryingScalar = ValidatedDefault(15e6)
    use_v_loop_lcfs_boundary_condition: Annotated[bool, JAX_STATIC] = False
    v_loop_lcfs: TimeVaryingScalar = ValidatedDefault(0.0)
    T_i_right_bc: TimeVaryingScalar | None = None
    T_e_right_bc: TimeVaryingScalar | None = None
    T_i: TimeVaryingArray = ValidatedDefault({0: {0: 15.0, 1: 1.0}})
    T_e: TimeVaryingArray = ValidatedDefault({0: {0: 15.0, 1: 1.0}})
    psi: TimeVaryingArray | None = None
    psidot: TimeVaryingArray | None = None
    n_e: TimeVaryingArray = ValidatedDefault({0: {0: 1.2e20, 1: 0.8e20}})
    nbar: TimeVaryingScalar = ValidatedDefault(0.85e20)
    n_e_right_bc: TimeVaryingScalar | None = None

    def build_runtime_params(self, t):
        runtime_params = {
            x.name: getattr(self, x.name)
            for x in dataclasses.fields(RuntimeParamsPC)
        }

        def _get_value(x):
            if isinstance(x, (TimeVaryingScalar, TimeVaryingArray)):
                return x.get_value(t)
            else:
                return x

        runtime_params = {k: _get_value(v) for k, v in runtime_params.items()}
        return RuntimeParamsPC(**runtime_params)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParamsP:
    main_ion_names: tuple[str,
                          ...] = dataclasses.field(metadata={"static": True})
    impurity_names: tuple[str,
                          ...] = dataclasses.field(metadata={"static": True})
    main_ion: Any
    impurity: Any
    Z_eff: Any
    Z_eff_face: Any


@jax.tree_util.register_pytree_node_class
class PlasmaComposition(BaseModelFrozen):
    impurity: Annotated[
        ImpurityFractions,
        pydantic.Field(discriminator="impurity_mode"),
    ]
    main_ion: IonMapping = ValidatedDefault({"D": 0.5, "T": 0.5})
    Z_eff: TimeVaryingArray = ValidatedDefault(1.0)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _conform_impurity_data(cls, data):
        configurable_data = copy.deepcopy(data)
        impurity_data = configurable_data["impurity"]
        configurable_data["impurity"] = {
            "impurity_mode": _IMPURITY_MODE_FRACTIONS,
            "species": impurity_data,
            "legacy": True,
        }
        return configurable_data

    def tree_flatten(self):
        children = (
            self.main_ion,
            self.impurity,
            self.Z_eff,
            self._main_ion_mixture,
        )
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.model_construct(
            main_ion=children[0],
            impurity=children[1],
            Z_eff=children[2],
        )
        obj._main_ion_mixture = children[3]
        return obj

    @functools.cached_property
    def _main_ion_mixture(self):
        return IonMixture.model_construct(species=self.main_ion, )

    def get_main_ion_names(self):
        return tuple(self._main_ion_mixture.species.keys())

    def get_impurity_names(self):
        return tuple(self.impurity.species.keys())

    def build_runtime_params(self, t):
        return RuntimeParamsP(
            main_ion_names=self.get_main_ion_names(),
            impurity_names=self.get_impurity_names(),
            main_ion=self._main_ion_mixture.build_runtime_params(t),
            impurity=self.impurity.build_runtime_params(t),
            Z_eff=self.Z_eff.get_value(t),
            Z_eff_face=self.Z_eff.get_value(t, grid_type="face"),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
    chi_face_ion: jax.Array
    chi_face_el: jax.Array
    d_face_el: jax.Array
    v_face_el: jax.Array
    chi_face_el_bohm: jax.Array | None = None
    chi_face_el_gyrobohm: jax.Array | None = None
    chi_face_ion_bohm: jax.Array | None = None
    chi_face_ion_gyrobohm: jax.Array | None = None


def _build_smoothing_matrix(transport_runtime_params, runtime_params, geo,
                            pedestal_model_output):
    lower_cutoff = 0.01
    kernel = jnp.exp(
        -jnp.log(2) *
        (geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm)**2 /
        (transport_runtime_params.smoothing_width**2 + g.eps))
    mask_outer_edge = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(True),
            True,
        ),
        lambda: g.rho_outer - g.eps,
        lambda: g.rho_norm_ped_top - g.eps,
    )
    mask_inner_edge = jax.lax.cond(
        True,
        lambda: g.rho_inner + g.eps,
        lambda: 0.0,
    )
    mask = jnp.where(
        jnp.logical_or(
            False,
            jnp.logical_and(
                geo.rho_face_norm > mask_inner_edge,
                geo.rho_face_norm < mask_outer_edge,
            ),
        ),
        1.0,
        0.0,
    )
    diag_mask = jnp.diag(mask)
    kernel = jnp.dot(diag_mask, kernel)
    num_rows = len(mask)
    mask_mat = jnp.tile(mask, (num_rows, 1))
    kernel *= mask_mat
    zero_row_mask = jnp.all(kernel == 0, axis=1)
    kernel = jnp.where(zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]),
                       kernel)
    row_sums = jnp.sum(kernel, axis=1)
    kernel /= row_sums[:, jnp.newaxis]
    kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)
    row_sums = jnp.sum(kernel, axis=1)
    kernel /= row_sums[:, jnp.newaxis]
    return kernel


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NormalizedLogarithmicGradients:
    lref_over_lti: Any
    lref_over_lte: Any
    lref_over_lne: Any
    lref_over_lni0: Any
    lref_over_lni1: Any

    @classmethod
    def from_profiles(cls, core_profiles, radial_coordinate, reference_length):
        gradients = {}
        for name, profile in {
                "lref_over_lti": core_profiles.T_i,
                "lref_over_lte": core_profiles.T_e,
                "lref_over_lne": core_profiles.n_e,
                "lref_over_lni0": core_profiles.n_i,
                "lref_over_lni1": core_profiles.n_impurity,
        }.items():
            gradients[name] = calculate_normalized_logarithmic_gradient(
                var=profile,
                radial_coordinate=radial_coordinate,
                reference_length=reference_length,
            )
        return cls(**gradients)


def calculate_chiGB(reference_temperature, reference_magnetic_field,
                    reference_mass, reference_length):
    return ((reference_mass * g.m_amu)**0.5 /
            (reference_magnetic_field * g.q_e)**2 *
            (reference_temperature * g.keV_to_J)**1.5 / reference_length)


def calculate_alpha(core_profiles, q, reference_magnetic_field,
                    normalized_logarithmic_gradients):
    factor_0 = 2 * g.keV_to_J / reference_magnetic_field**2 * g.mu_0 * q**2
    alpha = factor_0 * (
        core_profiles.T_e.face_value() * core_profiles.n_e.face_value() *
        (normalized_logarithmic_gradients.lref_over_lte +
         normalized_logarithmic_gradients.lref_over_lne) +
        core_profiles.n_i.face_value() * core_profiles.T_i.face_value() *
        (normalized_logarithmic_gradients.lref_over_lti +
         normalized_logarithmic_gradients.lref_over_lni0) +
        core_profiles.n_impurity.face_value() *
        core_profiles.T_i.face_value() *
        (normalized_logarithmic_gradients.lref_over_lti +
         normalized_logarithmic_gradients.lref_over_lni1))
    return alpha


def calculate_normalized_logarithmic_gradient(var, radial_coordinate,
                                              reference_length):
    result = jnp.where(
        jnp.abs(var.face_value()) < g.eps,
        g.eps,
        -reference_length * var.face_grad(radial_coordinate) /
        var.face_value(),
    )
    result = jnp.where(
        jnp.abs(result) < g.eps,
        g.eps,
        result,
    )
    return result


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QualikizInputs:
    chiGB: Any
    Rmin: Any
    Rmaj: Any
    lref_over_lti: Any
    lref_over_lte: Any
    lref_over_lne: Any
    lref_over_lni0: Any
    lref_over_lni1: Any
    Z_eff_face: Any
    q: Any
    smag: Any
    x: Any
    Ti_Te: Any
    log_nu_star_face: Any
    normni: Any
    alpha: Any
    epsilon_lcfs: Any

    @property
    def Ati(self):
        return self.lref_over_lti

    @property
    def Ate(self):
        return self.lref_over_lte

    @property
    def Ane(self):
        return self.lref_over_lne

    @property
    def Ani0(self):
        return self.lref_over_lni0


_FLUX_NAME_MAP: Final[Mapping[str, str]] = immutabledict.immutabledict({
    "efiITG":
    "qi_itg",
    "efeITG":
    "qe_itg",
    "pfeITG":
    "pfe_itg",
    "efeTEM":
    "qe_tem",
    "efiTEM":
    "qi_tem",
    "pfeTEM":
    "pfe_tem",
    "efeETG":
    "qe_etg",
})


def get_model_inputs_from_qualikiz_inputs(qualikiz_inputs):
    input_map = {
        "Ani": lambda x: x.Ani0,
        "LogNuStar": lambda x: x.log_nu_star_face,
    }

    def _get_input(key):
        return jnp.array(
            input_map.get(key, lambda x: getattr(x, key))(qualikiz_inputs),
            dtype=jnp.float64,
        )

    return jnp.array(
        [_get_input(key) for key in g.model.inputs_and_ranges.keys()],
        dtype=jnp.float64,
    ).T


def predict(inputs):
    model_predictions = g.model.predict(inputs)
    return {
        _FLUX_NAME_MAP.get(flux_name, flux_name): flux_value
        for flux_name, flux_value in model_predictions.items()
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams0:
    D_e_max: Any
    V_e_min: Any
    V_e_max: Any
    rho_min: Any
    rho_max: Any
    smoothing_width: Any
    collisionality_multiplier: float
    smag_alpha_correction: bool
    q_sawtooth_proxy: bool
    ETG_correction_factor: float
    clip_inputs: bool
    clip_margin: float


_EPSILON_NN: Final[float] = 1 / 3


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QLKNNRuntimeConfigInputs:
    transport: RuntimeParams0
    Ped_top: float

    @staticmethod
    def from_runtime_params_slice(transport_runtime_params, runtime_params,
                                  pedestal_model_output):
        return QLKNNRuntimeConfigInputs(
            transport=transport_runtime_params,
            Ped_top=g.rho_norm_ped_top,
        )


def clip_inputs(feature_scan, clip_margin, inputs_and_ranges):
    for i, key in enumerate(inputs_and_ranges.keys()):
        bounds = inputs_and_ranges[key]
        min_val = bounds.get("min", -jnp.inf)
        max_val = bounds.get("max", jnp.inf)
        min_val += jnp.where(jnp.isfinite(min_val),
                             jnp.abs(min_val) * (1 - clip_margin), 0.0)
        max_val -= jnp.where(jnp.isfinite(max_val),
                             jnp.abs(max_val) * (1 - clip_margin), 0.0)
        feature_scan = feature_scan.at[:, i].set(
            jnp.clip(
                feature_scan[:, i],
                min_val,
                max_val,
            ))
    return feature_scan


class QLKNNTransportModel0:

    def __init__(self, ):
        super().__init__()
        self._frozen = True

    def __setattr__(self, attr, value):
        return super().__setattr__(attr, value)

    def __call__(self, runtime_params, geo, core_profiles,
                 pedestal_model_output):
        transport_runtime_params = runtime_params.transport
        transport_coeffs = self._call_implementation(
            transport_runtime_params,
            runtime_params,
            geo,
            core_profiles,
            pedestal_model_output,
        )
        transport_coeffs = self._apply_domain_restriction(
            transport_runtime_params,
            geo,
            transport_coeffs,
            pedestal_model_output,
        )
        transport_coeffs = self._apply_clipping(
            transport_runtime_params,
            transport_coeffs,
        )
        transport_coeffs = self._apply_transport_patches(
            transport_runtime_params,
            runtime_params,
            geo,
            transport_coeffs,
        )
        return self._smooth_coeffs(
            transport_runtime_params,
            runtime_params,
            geo,
            transport_coeffs,
            pedestal_model_output,
        )

    def _apply_domain_restriction(self, transport_runtime_params, geo,
                                  transport_coeffs, pedestal_model_output):
        active_mask = ((geo.rho_face_norm > transport_runtime_params.rho_min)
                       &
                       (geo.rho_face_norm <= transport_runtime_params.rho_max)
                       & (geo.rho_face_norm <= g.rho_norm_ped_top))
        active_mask = (jnp.asarray(active_mask).at[0].set(
            transport_runtime_params.rho_min == 0))
        chi_face_ion = jnp.where(active_mask, transport_coeffs.chi_face_ion,
                                 0.0)
        chi_face_el = jnp.where(active_mask, transport_coeffs.chi_face_el, 0.0)
        d_face_el = jnp.where(active_mask, transport_coeffs.d_face_el, 0.0)
        v_face_el = jnp.where(active_mask, transport_coeffs.v_face_el, 0.0)
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _apply_clipping(self, transport_runtime_params, transport_coeffs):
        chi_face_ion = jnp.clip(
            transport_coeffs.chi_face_ion,
            g.chi_min,
            g.chi_max,
        )
        chi_face_el = jnp.clip(
            transport_coeffs.chi_face_el,
            g.chi_min,
            g.chi_max,
        )
        d_face_el = jnp.clip(
            transport_coeffs.d_face_el,
            g.D_e_min,
            transport_runtime_params.D_e_max,
        )
        v_face_el = jnp.clip(
            transport_coeffs.v_face_el,
            transport_runtime_params.V_e_min,
            transport_runtime_params.V_e_max,
        )
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _apply_transport_patches(self, transport_runtime_params,
                                 runtime_params, geo, transport_coeffs):
        chi_face_ion = jnp.where(
            jnp.logical_and(
                True,
                geo.rho_face_norm < g.rho_inner + g.eps,
            ),
            g.chi_i_inner,
            transport_coeffs.chi_face_ion,
        )
        chi_face_el = jnp.where(
            jnp.logical_and(
                True,
                geo.rho_face_norm < g.rho_inner + g.eps,
            ),
            g.chi_e_inner,
            transport_coeffs.chi_face_el,
        )
        d_face_el = jnp.where(
            jnp.logical_and(
                True,
                geo.rho_face_norm < g.rho_inner + g.eps,
            ),
            g.D_e_inner,
            transport_coeffs.d_face_el,
        )
        v_face_el = jnp.where(
            jnp.logical_and(
                True,
                geo.rho_face_norm < g.rho_inner + g.eps,
            ),
            g.V_e_inner,
            transport_coeffs.v_face_el,
        )
        chi_face_ion = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    True,
                    jnp.logical_not(True),
                ),
                geo.rho_face_norm > g.rho_outer - g.eps,
            ),
            g.chi_i_outer,
            chi_face_ion,
        )
        chi_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    True,
                    jnp.logical_not(True),
                ),
                geo.rho_face_norm > g.rho_outer - g.eps,
            ),
            g.chi_e_outer,
            chi_face_el,
        )
        d_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    True,
                    jnp.logical_not(True),
                ),
                geo.rho_face_norm > g.rho_outer - g.eps,
            ),
            g.D_e_outer,
            d_face_el,
        )
        v_face_el = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    True,
                    jnp.logical_not(True),
                ),
                geo.rho_face_norm > g.rho_outer - g.eps,
            ),
            g.V_e_outer,
            v_face_el,
        )
        return dataclasses.replace(
            transport_coeffs,
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _smooth_coeffs(
        self,
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    ):
        smoothing_matrix = _build_smoothing_matrix(
            transport_runtime_params,
            runtime_params,
            geo,
            pedestal_model_output,
        )

        def smooth_single_coeff(coeff):
            return jax.lax.cond(
                jnp.all(coeff == 0.0),
                lambda: coeff,
                lambda: jnp.dot(smoothing_matrix, coeff),
            )

        return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)

    def _make_core_transport(
        self,
        qi,
        qe,
        pfe,
        quasilinear_inputs,
        transport,
        geo,
        core_profiles,
        gradient_reference_length,
        gyrobohm_flux_reference_length,
    ):
        pfe_SI = (pfe * core_profiles.n_e.face_value() *
                  quasilinear_inputs.chiGB / gyrobohm_flux_reference_length)
        chi_face_ion = (
            ((gradient_reference_length / gyrobohm_flux_reference_length) * qi)
            / quasilinear_inputs.lref_over_lti) * quasilinear_inputs.chiGB
        chi_face_el = (
            ((gradient_reference_length / gyrobohm_flux_reference_length) * qe)
            / quasilinear_inputs.lref_over_lte) * quasilinear_inputs.chiGB

        def DV_effective_approach():
            Deff = -pfe_SI / (core_profiles.n_e.face_grad() *
                              geo.g1_over_vpr2_face * geo.rho_b + g.eps)
            Veff = pfe_SI / (core_profiles.n_e.face_value() *
                             geo.g0_over_vpr_face * geo.rho_b)
            Deff_mask = (((pfe >= 0) & (quasilinear_inputs.lref_over_lne >= 0))
                         | ((pfe < 0) &
                            (quasilinear_inputs.lref_over_lne < 0))) & (abs(
                                quasilinear_inputs.lref_over_lne) >= g.An_min)
            Veff_mask = jnp.invert(Deff_mask)
            d_face_el = jnp.where(Veff_mask, 0.0, Deff)
            v_face_el = jnp.where(Deff_mask, 0.0, Veff)
            return d_face_el, v_face_el

        def Dscaled_approach():
            d_face_el = chi_face_el
            v_face_el = (pfe_SI / core_profiles.n_e.face_value() -
                         quasilinear_inputs.lref_over_lne * d_face_el /
                         gradient_reference_length * geo.g1_over_vpr2_face *
                         geo.rho_b**2) / (geo.g0_over_vpr_face * geo.rho_b)
            return d_face_el, v_face_el

        d_face_el, v_face_el = jax.lax.cond(
            True,
            DV_effective_approach,
            Dscaled_approach,
        )
        return TurbulentTransport(
            chi_face_ion=chi_face_ion,
            chi_face_el=chi_face_el,
            d_face_el=d_face_el,
            v_face_el=v_face_el,
        )

    def _prepare_qualikiz_inputs(self, transport, geo, core_profiles):
        rmid = (geo.R_out - geo.R_in) * 0.5
        rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5
        chiGB = calculate_chiGB(
            reference_temperature=core_profiles.T_i.face_value(),
            reference_magnetic_field=geo.B_0,
            reference_mass=core_profiles.A_i,
            reference_length=geo.a_minor,
        )
        normalized_logarithmic_gradients = NormalizedLogarithmicGradients.from_profiles(
            core_profiles=core_profiles,
            radial_coordinate=rmid,
            reference_length=g.R_major,
        )
        q = core_profiles.q_face
        iota_scaled = jnp.abs(
            (core_profiles.psi.face_grad()[1:] / geo.rho_face_norm[1:]))
        iota_scaled0 = jnp.expand_dims(jnp.abs(
            core_profiles.psi.face_grad()[1] / geo.drho_norm),
                                       axis=0)
        iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
        rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5
        smag = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled
        epsilon_lcfs = rmid_face[-1] / g.R_major
        x = rmid_face / rmid_face[-1]
        x = jnp.where(jnp.abs(x) < g.eps, g.eps, x)
        Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
        log_lambda_ei_face = calculate_log_lambda_ei(
            core_profiles.T_e.face_value(),
            core_profiles.n_e.face_value(),
        )
        log_tau_e_Z1 = _calculate_log_tau_e_Z1(
            core_profiles.T_e.face_value(),
            core_profiles.n_e.face_value(),
            log_lambda_ei_face,
        )
        nu_e = (1 / jnp.exp(log_tau_e_Z1) * core_profiles.Z_eff_face *
                transport.collisionality_multiplier)
        epsilon = geo.rho_face / g.R_major
        epsilon = jnp.clip(epsilon, g.eps)
        tau_bounce = (
            core_profiles.q_face * g.R_major /
            (epsilon**1.5 *
             jnp.sqrt(core_profiles.T_e.face_value() * g.keV_to_J / g.m_e)))
        tau_bounce = tau_bounce.at[0].set(tau_bounce[1])
        nu_star = nu_e * tau_bounce
        log_nu_star_face = jnp.log10(nu_star)
        alpha = calculate_alpha(
            core_profiles=core_profiles,
            q=q,
            reference_magnetic_field=geo.B_0,
            normalized_logarithmic_gradients=normalized_logarithmic_gradients,
        )
        smag = jnp.where(
            transport.smag_alpha_correction,
            smag - alpha / 2,
            smag,
        )
        smag = jnp.where(
            jnp.logical_and(
                transport.q_sawtooth_proxy,
                q < 1,
            ),
            0.1,
            smag,
        )
        q = jnp.where(
            jnp.logical_and(
                transport.q_sawtooth_proxy,
                q < 1,
            ),
            1,
            q,
        )
        smag = jnp.where(
            jnp.logical_and(
                True,
                smag - alpha < -0.2,
            ),
            alpha - 0.2,
            smag,
        )
        normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value(
        )
        return QualikizInputs(
            Z_eff_face=core_profiles.Z_eff_face,
            lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
            lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
            lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
            lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
            lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
            q=q,
            smag=smag,
            x=x,
            Ti_Te=Ti_Te,
            log_nu_star_face=log_nu_star_face,
            normni=normni,
            chiGB=chiGB,
            Rmaj=g.R_major,
            Rmin=geo.a_minor,
            alpha=alpha,
            epsilon_lcfs=epsilon_lcfs,
        )

    def _call_implementation(
        self,
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    ):
        runtime_config_inputs = QLKNNRuntimeConfigInputs.from_runtime_params_slice(
            transport_runtime_params,
            runtime_params,
            pedestal_model_output,
        )
        return self._combined(runtime_config_inputs, geo, core_profiles)

    def _combined(self, runtime_config_inputs, geo, core_profiles):
        qualikiz_inputs = self._prepare_qualikiz_inputs(
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
        )
        qualikiz_inputs = dataclasses.replace(
            qualikiz_inputs,
            x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
        )
        feature_scan = get_model_inputs_from_qualikiz_inputs(qualikiz_inputs)
        feature_scan = jax.lax.cond(
            runtime_config_inputs.transport.clip_inputs,
            lambda: clip_inputs(
                feature_scan,
                runtime_config_inputs.transport.clip_margin,
                g.model.inputs_and_ranges,
            ),
            lambda: feature_scan,
        )
        model_output = predict(feature_scan)
        qi_itg_squeezed = model_output["qi_itg"].squeeze()
        qi = qi_itg_squeezed + model_output["qi_tem"].squeeze()
        qe = (model_output["qe_itg"].squeeze() * g.ITG_flux_ratio_correction +
              model_output["qe_tem"].squeeze() +
              model_output["qe_etg"].squeeze() *
              runtime_config_inputs.transport.ETG_correction_factor)
        pfe = model_output["pfe_itg"].squeeze(
        ) + model_output["pfe_tem"].squeeze()
        return self._make_core_transport(
            qi=qi,
            qe=qe,
            pfe=pfe,
            quasilinear_inputs=qualikiz_inputs,
            transport=runtime_config_inputs.transport,
            geo=geo,
            core_profiles=core_profiles,
            gradient_reference_length=g.R_major,
            gyrobohm_flux_reference_length=geo.a_minor,
        )


class QLKNNTransportModel(BaseModelFrozen):
    model_name: Annotated[Literal["qlknn"], JAX_STATIC] = "qlknn"
    model_path: Annotated[str, JAX_STATIC] = ""
    ETG_correction_factor: float = 1.0 / 3.0
    clip_inputs: bool = False
    clip_margin: float = 0.95
    collisionality_multiplier: float = 1.0
    smag_alpha_correction: bool = True
    q_sawtooth_proxy: bool = True
    D_e_max: Any = 100.0
    V_e_min: Any = -50.0
    V_e_max: Any = 50.0
    smoothing_width: pydantic.NonNegativeFloat = 0.1
    rho_min: UnitIntervalTimeVaryingScalar = ValidatedDefault(0.0)
    rho_max: UnitIntervalTimeVaryingScalar = ValidatedDefault(1.0)

    def build_runtime_params(self, t):
        return RuntimeParams0(
            D_e_max=self.D_e_max,
            V_e_min=self.V_e_min,
            V_e_max=self.V_e_max,
            rho_min=self.rho_min.get_value(t),
            rho_max=self.rho_max.get_value(t),
            smoothing_width=self.smoothing_width,
            ETG_correction_factor=self.ETG_correction_factor,
            clip_inputs=self.clip_inputs,
            clip_margin=self.clip_margin,
            collisionality_multiplier=self.collisionality_multiplier,
            smag_alpha_correction=self.smag_alpha_correction,
            q_sawtooth_proxy=self.q_sawtooth_proxy,
        )


@jax.jit
def calculate_total_transport_coeffs(runtime_params, geo, core_profiles):
    pedestal_model_output = g.pedestal_model(runtime_params, geo,
                                             core_profiles)
    turbulent_transport = g.transport_model(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        pedestal_model_output=pedestal_model_output,
    )
    return CoreTransport(**dataclasses.asdict(turbulent_transport))


g.rho_smoothing_limit = 0.1


def _smooth_savgol(data, idx_limit, polyorder):
    window_length = 5
    smoothed_data = scipy.signal.savgol_filter(data,
                                               window_length,
                                               polyorder,
                                               mode="nearest")
    return np.concatenate(
        [np.array([data[0]]), smoothed_data[1:idx_limit], data[idx_limit:]])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SafetyFactorFit:
    rho_q_min: Any
    q_min: Any
    rho_q_3_2_first: Any
    rho_q_2_1_first: Any
    rho_q_3_1_first: Any
    rho_q_3_2_second: Any
    rho_q_2_1_second: Any
    rho_q_3_1_second: Any


def _sliding_window_of_three(flat_array):
    window_size = 3
    starts = jnp.arange(len(flat_array) - window_size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(
        flat_array, (start, ), (window_size, )))(starts)


def _fit_polynomial_to_intervals_of_three(rho_norm: jax.Array,
                                          q_face: jax.Array):
    q_face_intervals = _sliding_window_of_three(q_face, )
    rho_norm_intervals = _sliding_window_of_three(rho_norm, )

    @jax.vmap
    def batch_polyfit(q_face_interval: jax.Array,
                      rho_norm_interval: jax.Array):
        rho_norm_squared = rho_norm_interval**2
        A = jnp.array([
            [rho_norm_squared[0], rho_norm_interval[0], 1],
            [rho_norm_squared[1], rho_norm_interval[1], 1],
            [rho_norm_squared[2], rho_norm_interval[2], 1],
        ])
        b = jnp.array(
            [q_face_interval[0], q_face_interval[1], q_face_interval[2]])
        coeffs = jnp.linalg.solve(A, b)
        return coeffs

    return (
        batch_polyfit(q_face_intervals, rho_norm_intervals),
        rho_norm_intervals,
        q_face_intervals,
    )


@jax.vmap
def _minimum_location_value_in_interval(coeffs: jax.Array,
                                        rho_norm_interval: jax.Array,
                                        q_interval: jax.Array):
    min_interval, max_interval = rho_norm_interval[0], rho_norm_interval[1]
    q_min_interval, q_max_interval = (
        q_interval[0],
        q_interval[1],
    )
    a, b = coeffs[0], coeffs[1]
    extremum_location = -b / (2 * a)
    extremum_in_interval = jnp.greater(extremum_location,
                                       min_interval) & jnp.less(
                                           extremum_location, max_interval)
    extremum_value = jax.lax.cond(
        extremum_in_interval,
        lambda x: jnp.polyval(coeffs, x),
        lambda x: jnp.inf,
        extremum_location,
    )
    interval_minimum_location, interval_minimum_value = jax.lax.cond(
        jnp.less(q_min_interval, q_max_interval),
        lambda: (min_interval, q_min_interval),
        lambda: (max_interval, q_max_interval),
    )
    overall_minimum_location, overall_minimum_value = jax.lax.cond(
        jnp.less(interval_minimum_value, extremum_value),
        lambda: (interval_minimum_location, interval_minimum_value),
        lambda: (extremum_location, extremum_value),
    )
    return overall_minimum_location, overall_minimum_value


def _find_roots_quadratic(coeffs):
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    determinant = b**2 - 4.0 * a * c
    roots_exist = jnp.greater(determinant, 0)
    plus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b + jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    minus_root = jax.lax.cond(
        roots_exist,
        lambda: (-b - jnp.sqrt(determinant)) / (2.0 * a),
        lambda: -jnp.inf,
    )
    return jnp.array([plus_root, minus_root])


@functools.partial(jax.vmap, in_axes=(0, 0, None))
def _root_in_interval(coeffs: jax.Array, interval: jax.Array,
                      q_surface: float):
    intercept_coeffs = coeffs - jnp.array([0.0, 0.0, q_surface])
    min_interval, max_interval = interval[0], interval[1]
    root_values = _find_roots_quadratic(intercept_coeffs)
    in_interval = jnp.greater(root_values, min_interval) & jnp.less(
        root_values, max_interval)
    return jnp.where(in_interval, root_values, -jnp.inf)


@jax.jit
def find_min_q_and_q_surface_intercepts(rho_norm, q_face):
    sorted_indices = jnp.argsort(rho_norm)
    rho_norm = rho_norm[sorted_indices]
    q_face = q_face[sorted_indices]
    poly_coeffs, rho_norm_3, q_face_3 = _fit_polynomial_to_intervals_of_three(
        rho_norm, q_face)
    first_rho_norm = jnp.expand_dims(jnp.array([rho_norm[0], rho_norm[2]]),
                                     axis=0)
    first_q_face = jnp.expand_dims(jnp.array([q_face[0], q_face[2]]), axis=0)
    rho_norms = jnp.concat([first_rho_norm, rho_norm_3[1:, 1:]], axis=0)
    q_faces = jnp.concat([first_q_face, q_face_3[1:, 1:]], axis=0)
    rho_q_min_intervals, q_min_intervals = _minimum_location_value_in_interval(
        poly_coeffs, rho_norms, q_faces)
    arg_q_min = jnp.argmin(q_min_intervals)
    rho_q_min = rho_q_min_intervals[arg_q_min]
    q_min = q_min_intervals[arg_q_min]
    rho_q_3_2 = _root_in_interval(poly_coeffs, rho_norms, 1.5).flatten()
    outermost_rho_q_3_2 = rho_q_3_2[jnp.argsort(rho_q_3_2)[-2:]]
    rho_q_2_1 = _root_in_interval(poly_coeffs, rho_norms, 2.0).flatten()
    outermost_rho_q_2_1 = rho_q_2_1[jnp.argsort(rho_q_2_1)[-2:]]
    rho_q_3_1 = _root_in_interval(poly_coeffs, rho_norms, 3.0).flatten()
    outermost_rho_q_3_1 = rho_q_3_1[jnp.argsort(rho_q_3_1)[-2:]]
    return SafetyFactorFit(
        rho_q_min=rho_q_min,
        q_min=q_min,
        rho_q_3_2_first=outermost_rho_q_3_2[0],
        rho_q_2_1_first=outermost_rho_q_2_1[0],
        rho_q_3_1_first=outermost_rho_q_3_1[0],
        rho_q_3_2_second=outermost_rho_q_3_2[1],
        rho_q_2_1_second=outermost_rho_q_2_1[1],
        rho_q_3_1_second=outermost_rho_q_3_1[1],
    )


RADIATION_OUTPUT_NAME = "radiation_impurity_species"
DENSITY_OUTPUT_NAME = "n_impurity_species"
Z_OUTPUT_NAME = "Z_impurity_species"
IMPURITY_DIM = "impurity_symbol"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpuritySpeciesOutput:
    radiation: Any
    n_impurity: Any
    Z_impurity: Any


def calculate_impurity_species_output(sim_state, runtime_params):
    impurity_species_output = {}
    impurity_fractions = sim_state.core_profiles.impurity_fractions
    impurity_names = runtime_params.plasma_composition.impurity_names
    charge_state_info = get_average_charge_state(
        ion_symbols=impurity_names,
        T_e=sim_state.core_profiles.T_e.value,
        fractions=jnp.stack(
            [impurity_fractions[symbol] for symbol in impurity_names]),
    )
    for i, symbol in enumerate(impurity_names):
        core_profiles = sim_state.core_profiles
        impurity_density_scaling = core_profiles.Z_impurity / charge_state_info.Z_avg
        n_imp = (impurity_fractions[symbol] * core_profiles.n_impurity.value *
                 impurity_density_scaling)
        Z_imp = charge_state_info.Z_per_species[i]
        radiation = jnp.zeros_like(n_imp)
        impurity_species_output[symbol] = ImpuritySpeciesOutput(
            radiation=radiation, n_impurity=n_imp, Z_impurity=Z_imp)
    return impurity_species_output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
    pressure_thermal_i: Any
    pressure_thermal_e: Any
    pressure_thermal_total: Any
    pprime: Any
    W_thermal_i: Any
    W_thermal_e: Any
    W_thermal_total: Any
    tau_E: Any
    H89P: Any
    H98: Any
    H97L: Any
    H20: Any
    FFprime: Any
    psi_norm: Any
    P_SOL_i: Any
    P_SOL_e: Any
    P_SOL_total: Any
    P_aux_i: Any
    P_aux_e: Any
    P_aux_total: Any
    P_external_injected: Any
    P_external_total: Any
    P_ei_exchange_i: Any
    P_ei_exchange_e: Any
    P_aux_generic_i: Any
    P_aux_generic_e: Any
    P_aux_generic_total: Any
    P_alpha_i: Any
    P_alpha_e: Any
    P_alpha_total: Any
    P_ohmic_e: Any
    P_bremsstrahlung_e: Any
    P_cyclotron_e: Any
    P_ecrh_e: Any
    P_radiation_e: Any
    I_ecrh: Any
    I_aux_generic: Any
    P_fusion: Any
    Q_fusion: Any
    P_icrh_e: Any
    P_icrh_i: Any
    P_icrh_total: Any
    P_LH_high_density: Any
    P_LH_min: Any
    P_LH: Any
    n_e_min_P_LH: Any
    E_fusion: Any
    E_aux_total: Any
    E_ohmic_e: Any
    E_external_injected: Any
    E_external_total: Any
    T_e_volume_avg: Any
    T_i_volume_avg: Any
    n_e_volume_avg: Any
    n_i_volume_avg: Any
    n_e_line_avg: Any
    n_i_line_avg: Any
    fgw_n_e_volume_avg: Any
    fgw_n_e_line_avg: Any
    q95: Any
    W_pol: Any
    li3: Any
    dW_thermal_dt: Any
    rho_q_min: Any
    q_min: Any
    rho_q_3_2_first: Any
    rho_q_3_2_second: Any
    rho_q_2_1_first: Any
    rho_q_2_1_second: Any
    rho_q_3_1_first: Any
    rho_q_3_1_second: Any
    I_bootstrap: Any
    j_external: Any
    j_ohmic: Any
    S_gas_puff: Any
    S_pellet: Any
    S_generic_particle: Any
    beta_tor: Any
    beta_pol: Any
    beta_N: Any
    S_total: Any
    impurity_species: Any


ION_EL_HEAT_SOURCE_TRANSFORMATIONS = {
    "generic_heat": "P_aux_generic",
    "fusion": "P_alpha",
    "icrh": "P_icrh",
}
EL_HEAT_SOURCE_TRANSFORMATIONS = {
    "ohmic": "P_ohmic_e",
    "bremsstrahlung": "P_bremsstrahlung_e",
    "cyclotron_radiation": "P_cyclotron_e",
    "ecrh": "P_ecrh_e",
    "impurity_radiation": "P_radiation_e",
}
EXTERNAL_HEATING_SOURCES = [
    "generic_heat",
    "ecrh",
    "icrh",
]
CURRENT_SOURCE_TRANSFORMATIONS = {
    "generic_current": "I_aux_generic",
    "ecrh": "I_ecrh",
}
PARTICLE_SOURCE_TRANSFORMATIONS = {
    "gas_puff": "S_gas_puff",
    "pellet": "S_pellet",
    "generic_particle": "S_generic_particle",
}


def _get_integrated_source_value(source_profiles_dict, internal_source_name,
                                 geo, integration_fn):
    if internal_source_name in source_profiles_dict:
        return integration_fn(source_profiles_dict[internal_source_name], geo)
    else:
        return jnp.array(0.0, dtype=jnp.float64)


@jax.jit
def make_post_processed_outputs(
    sim_state,
    runtime_params: RuntimeParamsSlice,
    previous_post_processed_outputs: PostProcessedOutputs | None = None,
):
    impurity_radiation_outputs = calculate_impurity_species_output(
        sim_state, runtime_params)
    (
        pressure_thermal_el,
        pressure_thermal_ion,
        pressure_thermal_tot,
    ) = calculate_pressure(sim_state.core_profiles)
    pprime_face = calc_pprime(sim_state.core_profiles)
    W_thermal_el = volume_integration(1.5 * pressure_thermal_el.value,
                                      sim_state.geometry)
    W_thermal_ion = volume_integration(1.5 * pressure_thermal_ion.value,
                                       sim_state.geometry)
    W_thermal_tot = volume_integration(1.5 * pressure_thermal_tot.value,
                                       sim_state.geometry)
    mu0 = g.mu_0
    pprime = calc_pprime(sim_state.core_profiles)
    g3 = sim_state.geometry.g3_face
    jtor_over_R = sim_state.core_profiles.j_total_face / g.R_major
    FFprime_face = -(jtor_over_R / (2 * jnp.pi) + pprime) * mu0 / g3
    psi_face = sim_state.core_profiles.psi.face_value()
    psi_norm_face = (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])
    geo = sim_state.geometry
    core_profiles = sim_state.core_profiles
    core_sources = sim_state.core_sources
    integrated = {}
    integrated["P_alpha_total"] = jnp.array(0.0, dtype=jnp.float64)
    integrated["S_total"] = jnp.array(0.0, dtype=jnp.float64)
    qei = core_sources.qei.qei_coef * (core_profiles.T_e.value -
                                       core_profiles.T_i.value)
    integrated["P_ei_exchange_i"] = volume_integration(qei, geo)
    integrated["P_ei_exchange_e"] = -integrated["P_ei_exchange_i"]
    integrated["P_SOL_i"] = integrated["P_ei_exchange_i"]
    integrated["P_SOL_e"] = integrated["P_ei_exchange_e"]
    integrated["P_aux_i"] = jnp.array(0.0, dtype=jnp.float64)
    integrated["P_aux_e"] = jnp.array(0.0, dtype=jnp.float64)
    integrated["P_external_injected"] = jnp.array(0.0, dtype=jnp.float64)
    for key, value in ION_EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        integrated[f"{value}_i"] = _get_integrated_source_value(
            core_sources.T_i, key, geo, volume_integration)
        integrated[f"{value}_e"] = _get_integrated_source_value(
            core_sources.T_e, key, geo, volume_integration)
        integrated[f"{value}_total"] = (integrated[f"{value}_i"] +
                                        integrated[f"{value}_e"])
        integrated["P_SOL_i"] += integrated[f"{value}_i"]
        integrated["P_SOL_e"] += integrated[f"{value}_e"]
        if key in EXTERNAL_HEATING_SOURCES:
            integrated["P_aux_i"] += integrated[f"{value}_i"]
            integrated["P_aux_e"] += integrated[f"{value}_e"]
            source_params = runtime_params.sources.get(key)
            if source_params is not None and hasattr(source_params,
                                                     "absorption_fraction"):
                total_absorbed = integrated[f"{value}_total"]
                injected_power = total_absorbed / source_params.absorption_fraction
                integrated["P_external_injected"] += injected_power
            else:
                integrated["P_external_injected"] += integrated[
                    f"{value}_total"]
    for key, value in EL_HEAT_SOURCE_TRANSFORMATIONS.items():
        integrated[f"{value}"] = _get_integrated_source_value(
            core_sources.T_e, key, geo, volume_integration)
        integrated["P_SOL_e"] += integrated[f"{value}"]
        if key in EXTERNAL_HEATING_SOURCES:
            integrated["P_aux_e"] += integrated[f"{value}"]
            integrated["P_external_injected"] += integrated[f"{value}"]
    for key, value in CURRENT_SOURCE_TRANSFORMATIONS.items():
        integrated[f"{value}"] = _get_integrated_source_value(
            core_sources.psi, key, geo, area_integration)
    for key, value in PARTICLE_SOURCE_TRANSFORMATIONS.items():
        integrated[f"{value}"] = _get_integrated_source_value(
            core_sources.n_e, key, geo, volume_integration)
        integrated["S_total"] += integrated[f"{value}"]
    integrated["P_SOL_total"] = integrated["P_SOL_i"] + integrated["P_SOL_e"]
    integrated["P_aux_total"] = integrated["P_aux_i"] + integrated["P_aux_e"]
    integrated["P_fusion"] = 5 * integrated["P_alpha_total"]
    integrated["P_external_total"] = (integrated["P_external_injected"] +
                                      integrated["P_ohmic_e"])
    integrated_sources = integrated
    Q_fusion = integrated_sources["P_fusion"] / (
        integrated_sources["P_external_total"] + g.eps)
    line_avg_n_e = line_average(sim_state.core_profiles.n_e.value)
    P_LH_hi_dens_D = (2.15 * (line_avg_n_e / 1e20)**0.782 *
                      sim_state.geometry.B_0**0.772 *
                      sim_state.geometry.a_minor**0.975 * g.R_major**0.999 *
                      1e6)
    A_deuterium = ION_PROPERTIES_DICT["D"].A
    P_LH_hi_dens = P_LH_hi_dens_D * A_deuterium / sim_state.core_profiles.A_i
    Ip_total = sim_state.core_profiles.Ip_profile_face[..., -1]
    n_e_min_P_LH = (0.7 * (Ip_total / 1e6)**0.34 *
                    sim_state.geometry.a_minor**-0.95 *
                    sim_state.geometry.B_0**0.62 *
                    (g.R_major / sim_state.geometry.a_minor)**0.4 * 1e19)
    P_LH_min_D = (0.36 * (Ip_total / 1e6)**0.27 *
                  sim_state.geometry.B_0**1.25 * g.R_major**1.23 *
                  (g.R_major / sim_state.geometry.a_minor)**0.08 * 1e6)
    P_LH_min = P_LH_min_D * A_deuterium / sim_state.core_profiles.A_i
    P_LH = jnp.maximum(P_LH_min, P_LH_hi_dens)
    Ploss = (integrated_sources["P_alpha_total"] +
             integrated_sources["P_aux_total"] +
             integrated_sources["P_ohmic_e"] + g.eps)
    if previous_post_processed_outputs is not None:
        dW_th_dt = (
            W_thermal_tot -
            previous_post_processed_outputs.W_thermal_total) / sim_state.dt
    else:
        dW_th_dt = 0.0
    tauE = W_thermal_tot / Ploss
    tauH89P = calculate_scaling_law_confinement_time(sim_state.geometry,
                                                     sim_state.core_profiles,
                                                     Ploss, "H89P")
    tauH98 = calculate_scaling_law_confinement_time(sim_state.geometry,
                                                    sim_state.core_profiles,
                                                    Ploss, "H98")
    tauH97L = calculate_scaling_law_confinement_time(sim_state.geometry,
                                                     sim_state.core_profiles,
                                                     Ploss, "H97L")
    tauH20 = calculate_scaling_law_confinement_time(sim_state.geometry,
                                                    sim_state.core_profiles,
                                                    Ploss, "H20")
    H89P = tauE / tauH89P
    H98 = tauE / tauH98
    H97L = tauE / tauH97L
    H20 = tauE / tauH20
    if previous_post_processed_outputs is not None:
        E_fusion = (previous_post_processed_outputs.E_fusion + sim_state.dt *
                    (integrated_sources["P_fusion"] +
                     previous_post_processed_outputs.P_fusion) / 2.0)
        E_aux_total = (previous_post_processed_outputs.E_aux_total +
                       sim_state.dt *
                       (integrated_sources["P_aux_total"] +
                        previous_post_processed_outputs.P_aux_total) / 2.0)
        E_ohmic_e = (previous_post_processed_outputs.E_ohmic_e + sim_state.dt *
                     (integrated_sources["P_ohmic_e"] +
                      previous_post_processed_outputs.P_ohmic_e) / 2.0)
        E_external_injected = (
            previous_post_processed_outputs.E_external_injected +
            sim_state.dt *
            (integrated_sources["P_external_injected"] +
             previous_post_processed_outputs.P_external_injected) / 2.0)
        E_external_total = (
            previous_post_processed_outputs.E_external_total + sim_state.dt *
            (integrated_sources["P_external_total"] +
             previous_post_processed_outputs.P_external_total) / 2.0)
    else:
        E_fusion = 0.0
        E_aux_total = 0.0
        E_ohmic_e = 0.0
        E_external_injected = 0.0
        E_external_total = 0.0
    q95 = jnp.interp(0.95, psi_norm_face, sim_state.core_profiles.q_face)
    te_volume_avg = volume_average(sim_state.core_profiles.T_e.value,
                                   sim_state.geometry)
    ti_volume_avg = volume_average(sim_state.core_profiles.T_i.value,
                                   sim_state.geometry)
    n_e_volume_avg = volume_average(sim_state.core_profiles.n_e.value,
                                    sim_state.geometry)
    n_i_volume_avg = volume_average(sim_state.core_profiles.n_i.value,
                                    sim_state.geometry)
    n_e_line_avg = line_average(sim_state.core_profiles.n_e.value)
    n_i_line_avg = line_average(sim_state.core_profiles.n_i.value)
    fgw_n_e_volume_avg = calculate_greenwald_fraction(n_e_volume_avg,
                                                      sim_state.core_profiles,
                                                      sim_state.geometry)
    fgw_n_e_line_avg = calculate_greenwald_fraction(n_e_line_avg,
                                                    sim_state.core_profiles,
                                                    sim_state.geometry)
    bpol2_bulk = ((sim_state.core_profiles.psi.face_grad()[1:] /
                   (2 * jnp.pi))**2 * sim_state.geometry.g2_face[1:] /
                  sim_state.geometry.vpr_face[1:]**2)
    bpol2_axis = jnp.array([0.0], dtype=jnp.float64)
    bpol2 = jnp.concatenate([bpol2_axis, bpol2_bulk])
    Wpol = jax.scipy.integrate.trapezoid(
        bpol2 * sim_state.geometry.vpr_face,
        sim_state.geometry.rho_face_norm) / (2 * g.mu_0)
    li3 = (4 * Wpol /
           (g.mu_0 * sim_state.core_profiles.Ip_profile_face[-1]**2 *
            sim_state.geometry.R_major))
    safety_factor_fit_outputs = find_min_q_and_q_surface_intercepts(
        sim_state.geometry.rho_face_norm,
        sim_state.core_profiles.q_face,
    )
    I_bootstrap = area_integration(
        sim_state.core_sources.bootstrap_current.j_bootstrap,
        sim_state.geometry)
    j_external = sum(sim_state.core_sources.psi.values())
    psi_current = j_external + sim_state.core_sources.bootstrap_current.j_bootstrap
    j_ohmic = sim_state.core_profiles.j_total - psi_current
    _, _, p_total = calculate_pressure(sim_state.core_profiles)
    p_total_volume_avg = volume_average(p_total.value, sim_state.geometry)
    magnetic_pressure_on_axis = sim_state.geometry.B_0**2 / (2 * g.mu_0)
    beta_tor = p_total_volume_avg / (magnetic_pressure_on_axis + g.eps)
    beta_pol = (
        4.0 * sim_state.geometry.volume[-1] * p_total_volume_avg /
        (g.mu_0 * sim_state.core_profiles.Ip_profile_face[-1]**2 * g.R_major +
         g.eps))
    beta_N = (1e8 * beta_tor *
              (sim_state.geometry.a_minor * sim_state.geometry.B_0 /
               (sim_state.core_profiles.Ip_profile_face[-1] + g.eps)))
    return PostProcessedOutputs(
        pressure_thermal_i=pressure_thermal_ion,
        pressure_thermal_e=pressure_thermal_el,
        pressure_thermal_total=pressure_thermal_tot,
        pprime=pprime_face,
        W_thermal_i=W_thermal_ion,
        W_thermal_e=W_thermal_el,
        W_thermal_total=W_thermal_tot,
        tau_E=tauE,
        H89P=H89P,
        H98=H98,
        H97L=H97L,
        H20=H20,
        FFprime=FFprime_face,
        psi_norm=psi_norm_face,
        **integrated_sources,
        Q_fusion=Q_fusion,
        P_LH=P_LH,
        P_LH_min=P_LH_min,
        P_LH_high_density=P_LH_hi_dens,
        n_e_min_P_LH=n_e_min_P_LH,
        E_fusion=E_fusion,
        E_aux_total=E_aux_total,
        E_ohmic_e=E_ohmic_e,
        E_external_injected=E_external_injected,
        E_external_total=E_external_total,
        T_e_volume_avg=te_volume_avg,
        T_i_volume_avg=ti_volume_avg,
        n_e_volume_avg=n_e_volume_avg,
        n_i_volume_avg=n_i_volume_avg,
        n_e_line_avg=n_e_line_avg,
        n_i_line_avg=n_i_line_avg,
        fgw_n_e_volume_avg=fgw_n_e_volume_avg,
        fgw_n_e_line_avg=fgw_n_e_line_avg,
        q95=q95,
        W_pol=Wpol,
        li3=li3,
        dW_thermal_dt=dW_th_dt,
        rho_q_min=safety_factor_fit_outputs.rho_q_min,
        q_min=safety_factor_fit_outputs.q_min,
        rho_q_3_2_first=safety_factor_fit_outputs.rho_q_3_2_first,
        rho_q_2_1_first=safety_factor_fit_outputs.rho_q_2_1_first,
        rho_q_3_1_first=safety_factor_fit_outputs.rho_q_3_1_first,
        rho_q_3_2_second=safety_factor_fit_outputs.rho_q_3_2_second,
        rho_q_2_1_second=safety_factor_fit_outputs.rho_q_2_1_second,
        rho_q_3_1_second=safety_factor_fit_outputs.rho_q_3_1_second,
        I_bootstrap=I_bootstrap,
        j_external=j_external,
        j_ohmic=j_ohmic,
        beta_tor=beta_tor,
        beta_pol=beta_pol,
        beta_N=beta_N,
        impurity_species=impurity_radiation_outputs,
    )


SCALING_FACTORS: Final[Mapping[str, float]] = immutabledict.immutabledict({
    "T_i":
    1.0,
    "T_e":
    1.0,
    "n_e":
    1e20,
    "psi":
    1.0,
})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Ions:
    n_i: Any
    n_impurity: Any
    impurity_fractions: Any
    Z_i: Any
    Z_i_face: Any
    Z_impurity: Any
    Z_impurity_face: Any
    A_i: Any
    A_impurity: Any
    A_impurity_face: Any
    Z_eff: Any
    Z_eff_face: Any


def get_updated_electron_temperature(profile_conditions_params, geo):
    T_e = CellVariable(
        value=profile_conditions_params.T_e,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=profile_conditions_params.T_e_right_bc,
        dr=geo.drho_norm,
    )
    return T_e


def get_updated_electron_density(profile_conditions_params, geo):
    nGW = profile_conditions_params.Ip / 1e6 / (jnp.pi * geo.a_minor**2) * 1e20
    n_e_value = jnp.where(
        True,
        profile_conditions_params.n_e * nGW,
        profile_conditions_params.n_e,
    )
    n_e_right_bc = jnp.where(
        False,
        profile_conditions_params.n_e_right_bc * nGW,
        profile_conditions_params.n_e_right_bc,
    )
    face_left = n_e_value[0]
    face_right = n_e_right_bc
    face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]], )
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    target_nbar = jnp.where(
        True,
        profile_conditions_params.nbar * nGW,
        profile_conditions_params.nbar,
    )
    nbar_from_n_e_face_inner = (
        jax.scipy.integrate.trapezoid(n_e_face[:-1], geo.R_out_face[:-1]) /
        a_minor_out)
    dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]
    C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
        nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out)
    n_e_value = C * n_e_value
    n_e = CellVariable(
        value=n_e_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e_right_bc,
    )
    return n_e


@dataclasses.dataclass(frozen=True)
class _IonProperties:
    A_impurity: Any
    A_impurity_face: Any
    Z_impurity: Any
    Z_impurity_face: Any
    Z_eff: Any
    dilution_factor: Any
    dilution_factor_edge: Any
    impurity_fractions: Any


def _get_ion_properties_from_fractions(
    impurity_symbols,
    impurity_params,
    T_e,
    Z_i,
    Z_i_face,
    Z_eff_from_config,
    Z_eff_face_from_config,
):
    Z_impurity = get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.value,
        fractions=impurity_params.fractions,
    ).Z_mixture
    Z_impurity_face = get_average_charge_state(
        ion_symbols=impurity_symbols,
        T_e=T_e.face_value(),
        fractions=impurity_params.fractions_face,
    ).Z_mixture
    Z_eff = Z_eff_from_config
    Z_eff_edge = Z_eff_face_from_config[-1]
    dilution_factor = jnp.where(
        Z_eff == 1.0,
        1.0,
        calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
    )
    dilution_factor_edge = jnp.where(
        Z_eff_edge == 1.0,
        1.0,
        calculate_main_ion_dilution_factor(Z_i_face[-1], Z_impurity_face[-1],
                                           Z_eff_edge),
    )
    return _IonProperties(
        A_impurity=impurity_params.A_avg,
        A_impurity_face=impurity_params.A_avg_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        Z_eff=Z_eff,
        dilution_factor=dilution_factor,
        dilution_factor_edge=dilution_factor_edge,
        impurity_fractions=impurity_params.fractions,
    )


@jax.jit
def get_updated_ions(runtime_params, geo, n_e, T_e):
    Z_i = get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.value,
        fractions=runtime_params.plasma_composition.main_ion.fractions,
    ).Z_mixture
    Z_i_face = get_average_charge_state(
        ion_symbols=runtime_params.plasma_composition.main_ion_names,
        T_e=T_e.face_value(),
        fractions=runtime_params.plasma_composition.main_ion.fractions,
    ).Z_mixture
    impurity_params = runtime_params.plasma_composition.impurity
    match impurity_params:
        case RuntimeParamsIF():
            ion_properties = _get_ion_properties_from_fractions(
                runtime_params.plasma_composition.impurity_names,
                impurity_params,
                T_e,
                Z_i,
                Z_i_face,
                runtime_params.plasma_composition.Z_eff,
                runtime_params.plasma_composition.Z_eff_face,
            )
    n_i = CellVariable(
        value=n_e.value * ion_properties.dilution_factor,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_e.right_face_constraint *
        ion_properties.dilution_factor_edge,
    )
    n_impurity_value = jnp.where(
        ion_properties.dilution_factor == 1.0,
        0.0,
        (n_e.value - n_i.value * Z_i) / ion_properties.Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        ion_properties.dilution_factor_edge == 1.0,
        0.0,
        (n_e.right_face_constraint - n_i.right_face_constraint * Z_i_face[-1])
        / ion_properties.Z_impurity_face[-1],
    )
    n_impurity = CellVariable(
        value=n_impurity_value,
        dr=geo.drho_norm,
        right_face_grad_constraint=None,
        right_face_constraint=n_impurity_right_face_constraint,
    )
    Z_eff_face = (Z_i_face**2 * n_i.face_value() +
                  ion_properties.Z_impurity_face**2 *
                  n_impurity.face_value()) / n_e.face_value()
    impurity_fractions_dict = {}
    for i, symbol in enumerate(
            runtime_params.plasma_composition.impurity_names):
        fraction = ion_properties.impurity_fractions[i]
        impurity_fractions_dict[symbol] = fraction
    return Ions(
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=impurity_fractions_dict,
        Z_i=Z_i,
        Z_i_face=Z_i_face,
        Z_impurity=ion_properties.Z_impurity,
        Z_impurity_face=ion_properties.Z_impurity_face,
        A_i=runtime_params.plasma_composition.main_ion.A_avg,
        A_impurity=ion_properties.A_impurity,
        A_impurity_face=ion_properties.A_impurity_face,
        Z_eff=ion_properties.Z_eff,
        Z_eff_face=Z_eff_face,
    )


def core_profiles_to_solver_x_tuple(core_profiles, ):
    x_tuple_for_solver_list = []
    for name in g.evolving_names:
        original_units_cv = getattr(core_profiles, name)
        solver_x_tuple_cv = scale_cell_variable(
            cv=original_units_cv,
            scaling_factor=1 / SCALING_FACTORS[name],
        )
        x_tuple_for_solver_list.append(solver_x_tuple_cv)
    return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(x_new, core_profiles):
    updated_vars = {}
    for i, var_name in enumerate(g.evolving_names):
        solver_x_tuple_cv = x_new[i]
        original_units_cv = scale_cell_variable(
            cv=solver_x_tuple_cv,
            scaling_factor=SCALING_FACTORS[var_name],
        )
        updated_vars[var_name] = original_units_cv
    return dataclasses.replace(core_profiles, **updated_vars)


def scale_cell_variable(cv, scaling_factor):
    operation = lambda x, factor: x * factor if x is not None else None
    scaled_value = operation(cv.value, scaling_factor)
    scaled_left_face_constraint = operation(cv.left_face_constraint,
                                            scaling_factor)
    scaled_left_face_grad_constraint = operation(cv.left_face_grad_constraint,
                                                 scaling_factor)
    scaled_right_face_constraint = operation(cv.right_face_constraint,
                                             scaling_factor)
    scaled_right_face_grad_constraint = operation(
        cv.right_face_grad_constraint, scaling_factor)
    return CellVariable(
        value=scaled_value,
        left_face_constraint=scaled_left_face_constraint,
        left_face_grad_constraint=scaled_left_face_grad_constraint,
        right_face_constraint=scaled_right_face_constraint,
        right_face_grad_constraint=scaled_right_face_grad_constraint,
        dr=cv.dr,
    )


OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...],
                                       ...] | None
AuxiliaryOutput: TypeAlias = Any


@jax.jit
def get_prescribed_core_profile_values(runtime_params, geo, core_profiles):
    T_i = core_profiles.T_i.value
    T_e_cell_variable = core_profiles.T_e
    T_e = T_e_cell_variable.value
    n_e_cell_variable = core_profiles.n_e
    ions = get_updated_ions(
        runtime_params,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )
    n_e = n_e_cell_variable.value
    n_i = ions.n_i.value
    n_impurity = ions.n_impurity.value
    impurity_fractions = ions.impurity_fractions
    return {
        "T_i": T_i,
        "T_e": T_e,
        "n_e": n_e,
        "n_i": n_i,
        "n_impurity": n_impurity,
        "impurity_fractions": impurity_fractions,
        "Z_i": ions.Z_i,
        "Z_i_face": ions.Z_i_face,
        "Z_impurity": ions.Z_impurity,
        "Z_impurity_face": ions.Z_impurity_face,
        "A_i": ions.A_i,
        "A_impurity": ions.A_impurity,
        "A_impurity_face": ions.A_impurity_face,
        "Z_eff": ions.Z_eff,
        "Z_eff_face": ions.Z_eff_face,
    }


@jax.jit
def update_core_profiles_during_step(x_new, runtime_params, geo,
                                     core_profiles):
    updated_core_profiles = solver_x_tuple_to_core_profiles(
        x_new, core_profiles)
    ions = get_updated_ions(
        runtime_params,
        geo,
        updated_core_profiles.n_e,
        updated_core_profiles.T_e,
    )
    return dataclasses.replace(
        updated_core_profiles,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        q_face=calc_q_face(geo, updated_core_profiles.psi),
        s_face=calc_s_face(geo, updated_core_profiles.psi),
    )


def update_core_and_source_profiles_after_step(
    dt,
    x_new,
    runtime_params_t_plus_dt,
    geo,
    core_profiles_t,
    core_profiles_t_plus_dt,
    explicit_source_profiles,
):
    updated_core_profiles_t_plus_dt = solver_x_tuple_to_core_profiles(
        x_new, core_profiles_t_plus_dt)
    ions = get_updated_ions(
        runtime_params_t_plus_dt,
        geo,
        updated_core_profiles_t_plus_dt.n_e,
        updated_core_profiles_t_plus_dt.T_e,
    )
    v_loop_lcfs = (runtime_params_t_plus_dt.profile_conditions.v_loop_lcfs
                   if runtime_params_t_plus_dt.profile_conditions.
                   use_v_loop_lcfs_boundary_condition else
                   (updated_core_profiles_t_plus_dt.psi.face_value()[-1] -
                    core_profiles_t.psi.face_value()[-1]) / dt)
    j_total, j_total_face, Ip_profile_face = calc_j_total(
        geo,
        updated_core_profiles_t_plus_dt.psi,
    )
    intermediate_core_profiles = CoreProfiles(
        T_i=updated_core_profiles_t_plus_dt.T_i,
        T_e=updated_core_profiles_t_plus_dt.T_e,
        psi=updated_core_profiles_t_plus_dt.psi,
        n_e=updated_core_profiles_t_plus_dt.n_e,
        n_i=ions.n_i,
        n_impurity=ions.n_impurity,
        impurity_fractions=ions.impurity_fractions,
        Z_i=ions.Z_i,
        Z_i_face=ions.Z_i_face,
        Z_impurity=ions.Z_impurity,
        Z_impurity_face=ions.Z_impurity_face,
        psidot=core_profiles_t_plus_dt.psidot,
        q_face=calc_q_face(geo, updated_core_profiles_t_plus_dt.psi),
        s_face=calc_s_face(geo, updated_core_profiles_t_plus_dt.psi),
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        v_loop_lcfs=v_loop_lcfs,
        sigma=core_profiles_t_plus_dt.sigma,
        sigma_face=core_profiles_t_plus_dt.sigma_face,
        j_total=j_total,
        j_total_face=j_total_face,
        Ip_profile_face=Ip_profile_face,
    )
    conductivity = calculate_conductivity(geo, intermediate_core_profiles)
    intermediate_core_profiles = dataclasses.replace(
        intermediate_core_profiles,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    total_source_profiles = build_source_profiles1(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo,
        core_profiles=intermediate_core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    psi_sources = total_source_profiles.total_psi_sources(geo)
    psidot_value = calculate_psidot_from_psi_sources(
        psi_sources=psi_sources,
        sigma=intermediate_core_profiles.sigma,
        psi=intermediate_core_profiles.psi,
        geo=geo,
    )
    psidot = dataclasses.replace(
        core_profiles_t_plus_dt.psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        intermediate_core_profiles,
        psidot=psidot,
    )
    return core_profiles_t_plus_dt, total_source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Block1DCoeffs:
    transient_in_cell: tuple[jax.Array, ...]
    transient_out_cell: tuple[jax.Array, ...] | None = None
    d_face: tuple[jax.Array, ...] | None = None
    v_face: tuple[jax.Array, ...] | None = None
    source_mat_cell: OptionalTupleMatrix = None
    source_cell: tuple[jax.Array | None, ...] | None = None
    auxiliary_outputs: AuxiliaryOutput | None = None


def cell_variable_tuple_to_vec(x_tuple):
    return jnp.concatenate([x.value for x in x_tuple])


def coeffs_callback(runtime_params,
                    geo,
                    core_profiles,
                    x,
                    explicit_source_profiles,
                    explicit_call=False):
    core_profiles = update_core_profiles_during_step(x, runtime_params, geo,
                                                     core_profiles)
    return calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        explicit_call=explicit_call,
    )


def _calculate_pereverzev_flux(geo, core_profiles, pedestal_model_output):
    geo_factor = jnp.concatenate(
        [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]])
    chi_face_per_ion = (geo.g1_over_vpr_face * core_profiles.n_i.face_value() *
                        g.keV_to_J * g.chi_pereverzev)
    chi_face_per_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                       g.keV_to_J * g.chi_pereverzev)
    d_face_per_el = g.D_pereverzev
    v_face_per_el = (core_profiles.n_e.face_grad() /
                     core_profiles.n_e.face_value() * d_face_per_el *
                     geo_factor)
    chi_face_per_ion = jnp.where(
        geo.rho_face_norm > g.rho_norm_ped_top,
        0.0,
        chi_face_per_ion,
    )
    chi_face_per_el = jnp.where(
        geo.rho_face_norm > g.rho_norm_ped_top,
        0.0,
        chi_face_per_el,
    )
    v_heat_face_ion = (core_profiles.T_i.face_grad() /
                       core_profiles.T_i.face_value() * chi_face_per_ion)
    v_heat_face_el = (core_profiles.T_e.face_grad() /
                      core_profiles.T_e.face_value() * chi_face_per_el)
    d_face_per_el = jnp.where(
        geo.rho_face_norm > g.rho_norm_ped_top,
        0.0,
        d_face_per_el * geo.g1_over_vpr_face,
    )
    v_face_per_el = jnp.where(
        geo.rho_face_norm > g.rho_norm_ped_top,
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


def calc_coeffs(runtime_params,
                geo,
                core_profiles,
                explicit_source_profiles,
                explicit_call=False):
    if explicit_call and g.theta_implicit == 1.0:
        tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
        tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
        tic_psi = jnp.ones_like(geo.vpr)
        tic_dens_el = geo.vpr
        var_to_tic = {
            "T_i": tic_T_i,
            "T_e": tic_T_e,
            "psi": tic_psi,
            "n_e": tic_dens_el,
        }
        transient_in_cell = tuple(var_to_tic[var] for var in g.evolving_names)
        coeffs = Block1DCoeffs(transient_in_cell=transient_in_cell, )
        return coeffs
    else:
        return _calc_coeffs_full(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
        )


@jax.jit
def _calc_coeffs_full(runtime_params, geo, core_profiles,
                      explicit_source_profiles):
    pedestal_model_output = g.pedestal_model(runtime_params, geo,
                                             core_profiles)
    mask = (jnp.zeros_like(
        geo.rho,
        dtype=bool).at[pedestal_model_output.rho_norm_ped_top_idx].set(True))
    conductivity = calculate_conductivity(geo, core_profiles)
    merged_source_profiles = build_source_profiles1(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    source_mat_psi = jnp.zeros_like(geo.rho)
    source_psi = merged_source_profiles.total_psi_sources(geo)
    toc_T_i = 1.5 * geo.vpr**(-2.0 / 3.0) * g.keV_to_J
    tic_T_i = core_profiles.n_i.value * geo.vpr**(5.0 / 3.0)
    toc_T_e = 1.5 * geo.vpr**(-2.0 / 3.0) * g.keV_to_J
    tic_T_e = core_profiles.n_e.value * geo.vpr**(5.0 / 3.0)
    toc_psi = (1.0 / g.resistivity_multiplier * geo.rho_norm *
               conductivity.sigma * g.mu_0 * 16 * jnp.pi**2 * geo.Phi_b**2 /
               geo.F**2)
    tic_psi = jnp.ones_like(toc_psi)
    toc_dens_el = jnp.ones_like(geo.vpr)
    tic_dens_el = geo.vpr
    turbulent_transport = g.transport_model(runtime_params, geo, core_profiles,
                                            pedestal_model_output)
    chi_face_ion_total = turbulent_transport.chi_face_ion
    chi_face_el_total = turbulent_transport.chi_face_el
    d_face_el_total = turbulent_transport.d_face_el
    v_face_el_total = turbulent_transport.v_face_el
    d_face_psi = geo.g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    full_chi_face_ion = (geo.g1_over_vpr_face *
                         core_profiles.n_i.face_value() * g.keV_to_J *
                         chi_face_ion_total)
    full_chi_face_el = (geo.g1_over_vpr_face * core_profiles.n_e.face_value() *
                        g.keV_to_J * chi_face_el_total)
    full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
    full_v_face_el = geo.g0_face * v_face_el_total
    source_mat_nn = jnp.zeros_like(geo.rho)
    source_n_e = merged_source_profiles.total_sources("n_e", geo)
    source_n_e += mask * g.adaptive_n_source_prefactor * g.n_e_ped
    source_mat_nn += -(mask * g.adaptive_n_source_prefactor)
    (
        chi_face_per_ion,
        chi_face_per_el,
        v_heat_face_ion,
        v_heat_face_el,
        d_face_per_el,
        v_face_per_el,
    ) = jax.lax.cond(
        True,
        lambda: _calculate_pereverzev_flux(
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
    v_heat_face_ion += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                        geo.rho_face_norm * geo.vpr_face *
                        core_profiles.n_i.face_value() * g.keV_to_J)
    v_heat_face_el += (-3.0 / 4.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face *
                       core_profiles.n_e.face_value() * g.keV_to_J)
    full_v_face_el += (-1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b *
                       geo.rho_face_norm * geo.vpr_face)
    source_i = merged_source_profiles.total_sources("T_i", geo)
    source_e = merged_source_profiles.total_sources("T_e", geo)
    qei = merged_source_profiles.qei
    source_mat_ii = qei.implicit_ii * geo.vpr
    source_i += qei.explicit_i * geo.vpr
    source_mat_ee = qei.implicit_ee * geo.vpr
    source_e += qei.explicit_e * geo.vpr
    source_mat_ie = qei.implicit_ie * geo.vpr
    source_mat_ei = qei.implicit_ei * geo.vpr
    source_i += mask * g.adaptive_T_source_prefactor * g.T_i_ped
    source_e += mask * g.adaptive_T_source_prefactor * g.T_e_ped
    source_mat_ii -= mask * g.adaptive_T_source_prefactor
    source_mat_ee -= mask * g.adaptive_T_source_prefactor
    d_vpr53_rhon_n_e_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
        geo.rho_norm,
    )
    d_vpr53_rhon_n_i_drhon = jnp.gradient(
        geo.vpr**(5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
        geo.rho_norm,
    )
    source_i += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_i_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_i.value *
                 g.keV_to_J)
    source_e += (3.0 / 4.0 * geo.vpr**(-2.0 / 3.0) * d_vpr53_rhon_n_e_drhon *
                 geo.Phi_b_dot / geo.Phi_b * core_profiles.T_e.value *
                 g.keV_to_J)
    d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)
    source_n_e += (1.0 / 2.0 * d_vpr_rhon_drhon * geo.Phi_b_dot / geo.Phi_b *
                   core_profiles.n_e.value)
    source_psi += (8.0 * jnp.pi**2 * g.mu_0 * geo.Phi_b_dot * geo.Phi_b *
                   geo.rho_norm**2 * conductivity.sigma / geo.F**2 *
                   core_profiles.psi.grad())
    var_to_toc = {
        "T_i": toc_T_i,
        "T_e": toc_T_e,
        "psi": toc_psi,
        "n_e": toc_dens_el,
    }
    var_to_tic = {
        "T_i": tic_T_i,
        "T_e": tic_T_e,
        "psi": tic_psi,
        "n_e": tic_dens_el,
    }
    transient_out_cell = tuple(var_to_toc[var] for var in g.evolving_names)
    transient_in_cell = tuple(var_to_tic[var] for var in g.evolving_names)
    var_to_d_face = {
        "T_i": full_chi_face_ion,
        "T_e": full_chi_face_el,
        "psi": d_face_psi,
        "n_e": full_d_face_el,
    }
    d_face = tuple(var_to_d_face[var] for var in g.evolving_names)
    var_to_v_face = {
        "T_i": v_heat_face_ion,
        "T_e": v_heat_face_el,
        "psi": v_face_psi,
        "n_e": full_v_face_el,
    }
    v_face = tuple(var_to_v_face.get(var) for var in g.evolving_names)
    d = {
        ("T_i", "T_i"): source_mat_ii,
        ("T_i", "T_e"): source_mat_ie,
        ("T_e", "T_i"): source_mat_ei,
        ("T_e", "T_e"): source_mat_ee,
        ("n_e", "n_e"): source_mat_nn,
        ("psi", "psi"): source_mat_psi,
    }
    source_mat_cell = tuple(
        tuple(d.get((row_block, col_block)) for col_block in g.evolving_names)
        for row_block in g.evolving_names)
    var_to_source = {
        "T_i": source_i / SCALING_FACTORS["T_i"],
        "T_e": source_e / SCALING_FACTORS["T_e"],
        "psi": source_psi / SCALING_FACTORS["psi"],
        "n_e": source_n_e / SCALING_FACTORS["n_e"],
    }
    source_cell = tuple(var_to_source.get(var) for var in g.evolving_names)
    coeffs = Block1DCoeffs(
        transient_out_cell=transient_out_cell,
        transient_in_cell=transient_in_cell,
        d_face=d_face,
        v_face=v_face,
        source_mat_cell=source_mat_cell,
        source_cell=source_cell,
        auxiliary_outputs=(
            merged_source_profiles,
            conductivity,
            CoreTransport(**dataclasses.asdict(turbulent_transport)),
        ),
    )
    return coeffs


MIN_DELTA: Final[float] = 1e-7


@jax.jit
def solver_x_new(
    dt,
    runtime_params_t,
    runtime_params_t_plus_dt,
    geo_t,
    geo_t_plus_dt,
    core_profiles_t,
    core_profiles_t_plus_dt,
    explicit_source_profiles,
):
    x_old = core_profiles_to_solver_x_tuple(core_profiles_t)
    x_new_guess = core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt)
    coeffs_exp = coeffs_callback(
        runtime_params_t,
        geo_t,
        core_profiles_t,
        x_old,
        explicit_source_profiles=explicit_source_profiles,
        explicit_call=True,
    )

    def loop_body(i, x_new_guess):
        coeffs_new = coeffs_callback(
            runtime_params_t_plus_dt,
            geo_t_plus_dt,
            core_profiles_t_plus_dt,
            x_new_guess,
            explicit_source_profiles=explicit_source_profiles,
        )
        x_old_vec = cell_variable_tuple_to_vec(x_old)
        x_new_guess_vec = cell_variable_tuple_to_vec(x_new_guess)
        theta_exp = 1.0 - g.theta_implicit
        tc_in_old = jnp.concatenate(coeffs_exp.transient_in_cell)
        tc_out_new = jnp.concatenate(coeffs_new.transient_out_cell)
        tc_in_new = jnp.concatenate(coeffs_new.transient_in_cell)
        left_transient = jnp.identity(len(x_new_guess_vec))
        right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))
        x = x_new_guess
        coeffs = coeffs_new
        d_face = coeffs.d_face
        v_face = coeffs.v_face
        source_mat_cell = coeffs.source_mat_cell
        source_cell = coeffs.source_cell
        num_cells = x[0].value.shape[0]
        num_channels = len(x)
        zero_block = jnp.zeros((num_cells, num_cells))
        zero_row_of_blocks = [zero_block] * num_channels
        zero_vec = jnp.zeros((num_cells))
        zero_block_vec = [zero_vec] * num_channels
        c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
        c = zero_block_vec.copy()
        if d_face is not None:
            for i in range(num_channels):
                (
                    diffusion_mat,
                    diffusion_vec,
                ) = make_diffusion_terms(
                    d_face[i],
                    x[i],
                )
                c_mat[i][i] += diffusion_mat
                c[i] += diffusion_vec
        if v_face is not None:
            for i in range(num_channels):
                d_face_i = d_face[i] if d_face is not None else None
                d_face_i = jnp.zeros_like(
                    v_face[i]) if d_face_i is None else d_face_i
                (
                    conv_mat,
                    conv_vec,
                ) = make_convection_terms(
                    v_face[i],
                    d_face_i,
                    x[i],
                )
                c_mat[i][i] += conv_mat
                c[i] += conv_vec
        if source_mat_cell is not None:
            for i in range(num_channels):
                for j in range(num_channels):
                    source = source_mat_cell[i][j]
                    if source is not None:
                        c_mat[i][j] += jnp.diag(source)
        if source_cell is not None:
            c = [(c_i + source_i) for c_i, source_i in zip(c, source_cell)]
        c_mat_new = jnp.block(c_mat)
        c_new = jnp.block(c)
        broadcasted = jnp.expand_dims(1 / (tc_out_new * tc_in_new), 1)
        lhs_mat = left_transient - dt * g.theta_implicit * broadcasted * c_mat_new
        lhs_vec = -g.theta_implicit * dt * (1 /
                                            (tc_out_new * tc_in_new)) * c_new
        if theta_exp > 0.0:
            assert False
        rhs_mat = right_transient
        rhs_vec = jnp.zeros_like(x_new_guess_vec)
        rhs = jnp.dot(rhs_mat, x_old_vec) + rhs_vec - lhs_vec
        x_new = jnp.linalg.solve(lhs_mat, rhs)
        x_new = jnp.split(x_new, len(x_old))
        out = [
            dataclasses.replace(var, value=value)
            for var, value in zip(x_new_guess, x_new)
        ]
        out = tuple(out)
        return out

    x_new = fori_loop(
        0,
        g.n_corrector_steps + 1,
        loop_body,
        x_new_guess,
    )
    solver_numeric_outputs = SolverNumericOutputs(solver_error_state=0, )
    return (
        x_new,
        solver_numeric_outputs,
    )


def not_done(t, t_final):
    return t < (t_final - g.tolerance)


def next_dt(t, runtime_params, geo, core_transport):
    chi_max = core_transport.chi_max(geo)
    basic_dt = (3.0 / 4.0) * (geo.drho_norm**2) / chi_max
    dt = jnp.minimum(
        g.chi_timestep_prefactor * basic_dt,
        g.max_dt,
    )
    crosses_t_final = (t < g.t_final) * (t + dt > g.t_final)
    dt = jax.lax.select(
        jnp.logical_and(
            True,
            crosses_t_final,
        ),
        g.t_final - t,
        dt,
    )
    return dt


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
    sources: Any
    profile_conditions: Any
    plasma_composition: Any
    transport_model: Any

    @classmethod
    def from_config(cls):
        return cls(
            sources=g.torax_config.sources,
            profile_conditions=g.torax_config.profile_conditions,
            plasma_composition=g.torax_config.plasma_composition,
            transport_model=g.torax_config.transport,
        )

    @jax.jit
    def __call__(self, t):
        return RuntimeParamsSlice(
            transport=self.transport_model.build_runtime_params(t),
            sources={
                source_name: source_config.build_runtime_params(t)
                for source_name, source_config in dict(self.sources).items()
                if source_config is not None
            },
            plasma_composition=self.plasma_composition.build_runtime_params(t),
            profile_conditions=self.profile_conditions.build_runtime_params(t),
        )


def get_consistent_runtime_params_and_geometry(*, t):
    runtime_params = g.runtime_params_provider(t=t)
    param_Ip = runtime_params.profile_conditions.Ip
    Ip_scale_factor = param_Ip / g.geo.Ip_profile_face[-1]
    geo = dataclasses.replace(
        g.geo,
        Ip_profile_face=g.geo.Ip_profile_face * Ip_scale_factor,
        psi_from_Ip=g.geo.psi_from_Ip * Ip_scale_factor,
        psi_from_Ip_face=g.geo.psi_from_Ip_face * Ip_scale_factor,
        j_total=g.geo.j_total * Ip_scale_factor,
        j_total_face=g.geo.j_total_face * Ip_scale_factor,
    )
    return runtime_params, geo


T_E = "T_e"
T_I = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
V_LOOP_LCFS = "v_loop_lcfs"
J_TOTAL = "j_total"
IP_PROFILE = "Ip_profile"
IP = "Ip"
J_OHMIC = "j_ohmic"
J_EXTERNAL = "j_external"
J_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"
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
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"
Q_FUSION = "Q_fusion"
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
    t: Any
    dt: Any
    core_profiles: Any
    core_transport: Any
    core_sources: Any
    geometry: Any
    solver_numeric_outputs: Any


def _finalize_outputs(
    t,
    dt,
    x_new,
    solver_numeric_outputs,
    geometry_t_plus_dt,
    runtime_params_t_plus_dt,
    core_profiles_t,
    core_profiles_t_plus_dt,
    explicit_source_profiles,
    input_post_processed_outputs,
):
    final_core_profiles, final_source_profiles = (
        update_core_and_source_profiles_after_step(
            dt=dt,
            x_new=x_new,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo=geometry_t_plus_dt,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            explicit_source_profiles=explicit_source_profiles,
        ))
    final_total_transport = calculate_total_transport_coeffs(
        runtime_params_t_plus_dt,
        geometry_t_plus_dt,
        final_core_profiles,
    )
    output_state = ToraxSimState(
        t=t + dt,
        dt=dt,
        core_profiles=final_core_profiles,
        core_sources=final_source_profiles,
        core_transport=final_total_transport,
        geometry=geometry_t_plus_dt,
        solver_numeric_outputs=solver_numeric_outputs,
    )
    post_processed_outputs = make_post_processed_outputs(
        sim_state=output_state,
        runtime_params=runtime_params_t_plus_dt,
        previous_post_processed_outputs=input_post_processed_outputs,
    )
    return output_state, post_processed_outputs


def _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(t, dt, geo_t):
    runtime_params_t_plus_dt, geo_t_plus_dt = (
        get_consistent_runtime_params_and_geometry(t=t + dt))
    Phibdot = (geo_t_plus_dt.Phi_b - geo_t.Phi_b) / dt
    geo_t = dataclasses.replace(geo_t, Phi_b_dot=Phibdot)
    geo_t_plus_dt = dataclasses.replace(geo_t_plus_dt, Phi_b_dot=Phibdot)
    return (runtime_params_t_plus_dt, geo_t, geo_t_plus_dt)


class ToraxConfig(BaseModelFrozen):
    profile_conditions: ProfileConditions
    plasma_composition: PlasmaComposition
    sources: Sources
    transport: QLKNNTransportModel


def body_fun(inputs):
    dt, output = inputs
    runtime_params_t_plus_dt, geo_t_with_phibdot, geo_t_plus_dt = (
        _get_geo_and_runtime_params_at_t_plus_dt_and_phibdot(
            current_state.t,
            dt,
            geo_t,
        ))
    core_profiles_t = current_state.core_profiles
    profile_conditions_t_plus_dt = runtime_params_t_plus_dt.profile_conditions
    n_e = get_updated_electron_density(profile_conditions_t_plus_dt,
                                       geo_t_plus_dt)
    n_e_right_bc = n_e.right_face_constraint
    ions_edge = get_updated_ions(
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        dataclasses.replace(
            core_profiles_t.n_e,
            right_face_constraint=profile_conditions_t_plus_dt.n_e_right_bc,
        ),
        dataclasses.replace(
            core_profiles_t.T_e,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
    )
    Z_i_edge = ions_edge.Z_i_face[-1]
    Z_impurity_edge = ions_edge.Z_impurity_face[-1]
    dilution_factor_edge = calculate_main_ion_dilution_factor(
        Z_i_edge,
        Z_impurity_edge,
        runtime_params_t_plus_dt.plasma_composition.Z_eff_face[-1],
    )
    n_i_bound_right = n_e_right_bc * dilution_factor_edge
    n_impurity_bound_right = (n_e_right_bc -
                              n_i_bound_right * Z_i_edge) / Z_impurity_edge
    updated_boundary_conditions = {
        "T_i":
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_i_right_bc,
        ),
        "T_e":
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
        ),
        "n_e":
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_e_right_bc),
        ),
        "n_i":
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_i_bound_right),
        ),
        "n_impurity":
        dict(
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(n_impurity_bound_right),
        ),
        "psi":
        dict(
            right_face_grad_constraint=(calculate_psi_grad_constraint_from_Ip(
                Ip=profile_conditions_t_plus_dt.Ip,
                geo=geo_t_plus_dt,
            ) if not runtime_params_t.profile_conditions.
                                        use_v_loop_lcfs_boundary_condition else
                                        None),
            right_face_constraint=(_calculate_psi_value_constraint_from_v_loop(
                dt=dt,
                v_loop_lcfs_t=runtime_params_t.profile_conditions.v_loop_lcfs,
                v_loop_lcfs_t_plus_dt=profile_conditions_t_plus_dt.v_loop_lcfs,
                psi_lcfs_t=core_profiles_t.psi.right_face_constraint,
            ) if runtime_params_t.profile_conditions.
                                   use_v_loop_lcfs_boundary_condition else
                                   None),
        ),
        "Z_i_edge":
        Z_i_edge,
        "Z_impurity_edge":
        Z_impurity_edge,
    }
    updated_values = get_prescribed_core_profile_values(
        runtime_params=runtime_params_t_plus_dt,
        geo=geo_t_plus_dt,
        core_profiles=core_profiles_t,
    )
    T_i = dataclasses.replace(
        core_profiles_t.T_i,
        value=updated_values["T_i"],
        **updated_boundary_conditions["T_i"],
    )
    T_e = dataclasses.replace(
        core_profiles_t.T_e,
        value=updated_values["T_e"],
        **updated_boundary_conditions["T_e"],
    )
    psi = dataclasses.replace(core_profiles_t.psi,
                              **updated_boundary_conditions["psi"])
    n_e = dataclasses.replace(
        core_profiles_t.n_e,
        value=updated_values["n_e"],
        **updated_boundary_conditions["n_e"],
    )
    n_i = dataclasses.replace(
        core_profiles_t.n_i,
        value=updated_values["n_i"],
        **updated_boundary_conditions["n_i"],
    )
    n_impurity = dataclasses.replace(
        core_profiles_t.n_impurity,
        value=updated_values["n_impurity"],
        **updated_boundary_conditions["n_impurity"],
    )
    Z_i_face = jnp.concatenate([
        updated_values["Z_i_face"][:-1],
        jnp.array([updated_boundary_conditions["Z_i_edge"]]),
    ], )
    Z_impurity_face = jnp.concatenate([
        updated_values["Z_impurity_face"][:-1],
        jnp.array([updated_boundary_conditions["Z_impurity_edge"]]),
    ], )
    core_profiles_t_plus_dt = dataclasses.replace(
        core_profiles_t,
        T_i=T_i,
        T_e=T_e,
        psi=psi,
        n_e=n_e,
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=updated_values["impurity_fractions"],
        Z_i=updated_values["Z_i"],
        Z_i_face=Z_i_face,
        Z_impurity=updated_values["Z_impurity"],
        Z_impurity_face=Z_impurity_face,
        A_i=updated_values["A_i"],
        A_impurity=updated_values["A_impurity"],
        A_impurity_face=updated_values["A_impurity_face"],
        Z_eff=updated_values["Z_eff"],
        Z_eff_face=updated_values["Z_eff_face"],
    )
    x_new, solver_numeric_outputs = solver_x_new(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t_with_phibdot,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=current_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
    )
    solver_numeric_outputs = SolverNumericOutputs(
        solver_error_state=solver_numeric_outputs.solver_error_state, )
    next_dt = dt / g.dt_reduction_factor
    return next_dt, (
        x_new,
        dt,
        solver_numeric_outputs,
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        core_profiles_t_plus_dt,
    )


def cond_fun(inputs):
    next_dt, output = inputs
    solver_outputs = output[2]
    is_nan_next_dt = jnp.isnan(next_dt)
    solver_did_not_converge = solver_outputs.solver_error_state == 1
    at_exact_t_final = jnp.allclose(
        current_state.t + next_dt,
        g.t_final,
    )
    next_dt_too_small = next_dt < g.min_dt
    take_another_step = cond(
        solver_did_not_converge,
        lambda: cond(at_exact_t_final, lambda: True, lambda: ~next_dt_too_small
                     ),
        lambda: False,
    )
    return take_another_step & ~is_nan_next_dt


CONFIG = {
    "plasma_composition": {
        "main_ion": {
            "D": 0.5,
            "T": 0.5
        },
        "impurity": "Ne",
        "Z_eff": 1.6,
    },
    "profile_conditions": {
        "Ip": 10.5e6,
        "T_i": {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        "T_i_right_bc": 0.2,
        "T_e": {
            0.0: {
                0.0: 15.0,
                1.0: 0.2
            }
        },
        "T_e_right_bc": 0.2,
        "n_e_right_bc": 0.25e20,
        "nbar": 0.8,
        "n_e": {
            0: {
                0.0: 1.5,
                1.0: 1.0
            }
        },
    },
    "sources": {
        "generic_current": {
            "fraction_of_total_current": 0.46,
            "gaussian_width": 0.075,
            "gaussian_location": 0.36,
        },
        "generic_particle": {
            "S_total": 2.05e20,
            "deposition_location": 0.3,
            "particle_width": 0.25,
        },
        "gas_puff": {
            "puff_decay_length": 0.3,
            "S_total": 6.0e21,
        },
        "pellet": {
            "S_total": 0.0e22,
            "pellet_width": 0.1,
            "pellet_deposition_location": 0.85,
        },
        "generic_heat": {
            "gaussian_location": 0.12741589640723575,
            "gaussian_width": 0.07280908366127758,
            "P_total": 51.0e6,
            "electron_heat_fraction": 0.68,
        },
        "fusion": {},
        "ei_exchange": {},
    },
    "transport": {},
}
g.model = qlknn_model.QLKNNModel.load_default_model()
g.R_major = 6.2
g.a_minor = 2.0
g.B_0 = 5.3
g.tolerance = 1e-7
g.n_corrector_steps = 1
g.torax_config = ToraxConfig.from_dict(CONFIG)
g.chi_pereverzev = 30
g.D_pereverzev = 15
g.theta_implicit = 1.0
g.t_final = 5
g.resistivity_multiplier = 200
g.max_dt = 0.5
g.min_dt = 1e-8
g.chi_timestep_prefactor = 50
g.dt_reduction_factor = 3
g.adaptive_T_source_prefactor = 2.0e10
g.adaptive_n_source_prefactor = 2.0e8
g.t_initial = 0.0
g.ITG_flux_ratio_correction = 1
# {'n_rho': 25, 'a_minor': 2.0, 'B_0': 5.3}
g.n_rho = 25
g.dx = 1 / g.n_rho
g.face_centers = np.linspace(0, g.n_rho * g.dx, g.n_rho + 1)
g.cell_centers = np.linspace(g.dx * 0.5, (g.n_rho - 0.5) * g.dx, g.n_rho)
g.hires_factor = 4
g.Qei_multiplier = 1.0
g.rho_norm_ped_top = 0.9
g.n_e_ped = 0.62e20
g.T_i_ped = 4.5
g.T_e_ped = 4.5
g.rho_norm_ped_top = 0.91
# transport
g.D_e_inner = 0.25
g.V_e_inner = 0.0
g.chi_i_inner = 1.0
g.chi_e_inner = 1.0
g.rho_inner = 0.2
g.D_e_outer = 0.1
g.V_e_outer = 0.0
g.chi_i_outer = 2.0
g.chi_e_outer = 2.0
g.rho_outer = 0.9
g.chi_min = 0.05
g.chi_max = 100
g.D_e_min = 0.05
g.An_min = 0.05
# Inline StandardGeometryIntermediates functionality
file_path = os.path.join("geo", "ITER_hybrid_citrin_equil_cheasedata.mat2cols")
with open(file_path, "r") as file:
    chease_data = {}
    var_labels = file.readline().strip().split()[1:]
    for var_label in var_labels:
        chease_data[var_label] = []
    for line in file:
        values = line.strip().split()
        for var_label, value in zip(var_labels, values):
            chease_data[var_label].append(float(value))
chease_data = {
    var_label: np.asarray(chease_data[var_label])
    for var_label in chease_data
}
psiunnormfactor = g.R_major**2 * g.B_0
psi = chease_data["PSIchease=psi/2pi"] * psiunnormfactor * 2 * np.pi
Ip_chease = chease_data["Ipprofile"] / g.mu_0 * g.R_major * g.B_0
Phi = (chease_data["RHO_TOR=sqrt(Phi/pi/B0)"] * g.R_major)**2 * g.B_0 * np.pi
R_in_chease = chease_data["R_INBOARD"] * g.R_major
R_out_chease = chease_data["R_OUTBOARD"] * g.R_major
F_chease = chease_data["T=RBphi"] * g.R_major * g.B_0
int_dl_over_Bp = (chease_data["Int(Rdlp/|grad(psi)|)=Int(Jdchi)"] * g.R_major /
                  g.B_0)
flux_surf_avg_1_over_R = chease_data["<1/R>profile"] / g.R_major
flux_surf_avg_1_over_R2 = chease_data["<1/R**2>"] / g.R_major**2
flux_surf_avg_Bp2 = chease_data["<Bp**2>"] * g.B_0**2
flux_surf_avg_RBp = chease_data["<|grad(psi)|>"] * psiunnormfactor / g.R_major
flux_surf_avg_R2Bp2 = (chease_data["<|grad(psi)|**2>"] * psiunnormfactor**2 /
                       g.R_major**2)
flux_surf_avg_B2 = chease_data["<B**2>"] * g.B_0**2
flux_surf_avg_1_over_B2 = chease_data["<1/B**2>"] / g.B_0**2
rhon = np.sqrt(Phi / Phi[-1])
vpr = 4 * np.pi * Phi[-1] * rhon / (F_chease * flux_surf_avg_1_over_R2)

# Apply smoothing (post_init functionality)
assert not flux_surf_avg_Bp2[-1] < 1e-10
idx_limit = np.argmin(np.abs(rhon - g.rho_smoothing_limit))
flux_surf_avg_Bp2[:] = _smooth_savgol(flux_surf_avg_Bp2, idx_limit, 2)
flux_surf_avg_R2Bp2[:] = _smooth_savgol(flux_surf_avg_R2Bp2, idx_limit, 2)
flux_surf_avg_RBp[:] = _smooth_savgol(flux_surf_avg_RBp, idx_limit, 1)
vpr[:] = _smooth_savgol(vpr, idx_limit, 1)

# Continue with geometry calculations
rho_intermediate = np.sqrt(Phi / (np.pi * g.B_0))
rho_norm_intermediate = rho_intermediate / rho_intermediate[-1]
C1 = int_dl_over_Bp
C0 = flux_surf_avg_RBp * C1
C2 = flux_surf_avg_1_over_R2 * C1
C3 = flux_surf_avg_Bp2 * C1
C4 = flux_surf_avg_R2Bp2 * C1
g0 = C0 * 2 * np.pi
g1 = C1 * C4 * 4 * np.pi**2
g2 = C1 * C3 * 4 * np.pi**2
g3 = C2[1:] / C1[1:]
g3 = np.concatenate((np.array([1 / R_in_chease[0]**2]), g3))
g2g3_over_rhon = g2[1:] * g3[1:] / rho_norm_intermediate[1:]
g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))
dpsidrhon = (Ip_chease[1:] * (16 * g.mu_0 * np.pi**3 * Phi[-1]) /
             (g2g3_over_rhon[1:] * F_chease[1:]))
dpsidrhon = np.concatenate((np.zeros(1), dpsidrhon))
psi_from_Ip = scipy.integrate.cumulative_trapezoid(y=dpsidrhon,
                                                   x=rho_norm_intermediate,
                                                   initial=0.0)
psi_from_Ip += psi[0]
psi_from_Ip[-1] = psi_from_Ip[-2] + (
    16 * g.mu_0 * np.pi**3 * Phi[-1]) * Ip_chease[-1] / (
        g2g3_over_rhon[-1] * F_chease[-1]) * (rho_norm_intermediate[-1] -
                                              rho_norm_intermediate[-2])
spr = vpr * flux_surf_avg_1_over_R / (2 * np.pi)
volume_intermediate = scipy.integrate.cumulative_trapezoid(
    y=vpr, x=rho_norm_intermediate, initial=0.0)
area_intermediate = scipy.integrate.cumulative_trapezoid(
    y=spr, x=rho_norm_intermediate, initial=0.0)
dI_tot_drhon = np.gradient(Ip_chease, rho_norm_intermediate)
j_total_face_bulk = dI_tot_drhon[1:] / spr[1:]
j_total_face_axis = j_total_face_bulk[0]
j_total = np.concatenate([np.array([j_total_face_axis]), j_total_face_bulk])
rho_b = rho_intermediate[-1]
rho_face_norm = g.face_centers
rho_norm = g.cell_centers
rho_hires_norm = np.linspace(0, 1, g.n_rho * g.hires_factor)
rho_hires = rho_hires_norm * rho_b
rhon_interpolation_func = lambda x, y: np.interp(x, rho_norm_intermediate, y)
vpr_face = rhon_interpolation_func(rho_face_norm, vpr)
vpr = rhon_interpolation_func(rho_norm, vpr)
spr_face = rhon_interpolation_func(rho_face_norm, spr)
spr_cell = rhon_interpolation_func(rho_norm, spr)
spr_hires = rhon_interpolation_func(rho_hires_norm, spr)
delta_upper_face = rhon_interpolation_func(rho_face_norm,
                                           chease_data["delta_upper"])
delta_lower_face = rhon_interpolation_func(rho_face_norm,
                                           chease_data["delta_bottom"])
delta_face = 0.5 * (delta_upper_face + delta_lower_face)
elongation = rhon_interpolation_func(rho_norm, chease_data["elongation"])
elongation_face = rhon_interpolation_func(rho_face_norm,
                                          chease_data["elongation"])
Phi_face = rhon_interpolation_func(rho_face_norm, Phi)
Phi = rhon_interpolation_func(rho_norm, Phi)
F_face = rhon_interpolation_func(rho_face_norm, F_chease)
F = rhon_interpolation_func(rho_norm, F_chease)
F_hires = rhon_interpolation_func(rho_hires_norm, F_chease)
psi = rhon_interpolation_func(rho_norm, psi)
psi_from_Ip_face = rhon_interpolation_func(rho_face_norm, psi_from_Ip)
psi_from_Ip = rhon_interpolation_func(rho_norm, psi_from_Ip)
j_total_face = rhon_interpolation_func(rho_face_norm, j_total)
j_total = rhon_interpolation_func(rho_norm, j_total)
Ip_profile_face = rhon_interpolation_func(rho_face_norm, Ip_chease)
Rin_face = rhon_interpolation_func(rho_face_norm, R_in_chease)
Rin = rhon_interpolation_func(rho_norm, R_in_chease)
Rout_face = rhon_interpolation_func(rho_face_norm, R_out_chease)
Rout = rhon_interpolation_func(rho_norm, R_out_chease)
g0_face = rhon_interpolation_func(rho_face_norm, g0)
g0 = rhon_interpolation_func(rho_norm, g0)
g1_face = rhon_interpolation_func(rho_face_norm, g1)
g1 = rhon_interpolation_func(rho_norm, g1)
g2_face = rhon_interpolation_func(rho_face_norm, g2)
g2 = rhon_interpolation_func(rho_norm, g2)
g3_face = rhon_interpolation_func(rho_face_norm, g3)
g3 = rhon_interpolation_func(rho_norm, g3)
g2g3_over_rhon_face = rhon_interpolation_func(rho_face_norm, g2g3_over_rhon)
g2g3_over_rhon_hires = rhon_interpolation_func(rho_hires_norm, g2g3_over_rhon)
g2g3_over_rhon = rhon_interpolation_func(rho_norm, g2g3_over_rhon)
gm4 = rhon_interpolation_func(rho_norm, flux_surf_avg_1_over_B2)
gm4_face = rhon_interpolation_func(rho_face_norm, flux_surf_avg_1_over_B2)
gm5 = rhon_interpolation_func(rho_norm, flux_surf_avg_B2)
gm5_face = rhon_interpolation_func(rho_face_norm, flux_surf_avg_B2)
volume_face = rhon_interpolation_func(rho_face_norm, volume_intermediate)
volume = rhon_interpolation_func(rho_norm, volume_intermediate)
area_face = rhon_interpolation_func(rho_face_norm, area_intermediate)
area = rhon_interpolation_func(rho_norm, area_intermediate)
g.geo = StandardGeometry(
    Phi=Phi,
    Phi_face=Phi_face,
    R_major=g.R_major,
    a_minor=g.a_minor,
    B_0=g.B_0,
    volume=volume,
    volume_face=volume_face,
    area=area,
    area_face=area_face,
    vpr=vpr,
    vpr_face=vpr_face,
    spr=spr_cell,
    spr_face=spr_face,
    delta_face=delta_face,
    g0=g0,
    g0_face=g0_face,
    g1=g1,
    g1_face=g1_face,
    g2=g2,
    g2_face=g2_face,
    g3=g3,
    g3_face=g3_face,
    g2g3_over_rhon=g2g3_over_rhon,
    g2g3_over_rhon_face=g2g3_over_rhon_face,
    g2g3_over_rhon_hires=g2g3_over_rhon_hires,
    gm4=gm4,
    gm4_face=gm4_face,
    gm5=gm5,
    gm5_face=gm5_face,
    F=F,
    F_face=F_face,
    F_hires=F_hires,
    R_in=Rin,
    R_in_face=Rin_face,
    R_out=Rout,
    R_out_face=Rout_face,
    Ip_profile_face=Ip_profile_face,
    psi=psi,
    psi_from_Ip=psi_from_Ip,
    psi_from_Ip_face=psi_from_Ip_face,
    j_total=j_total,
    j_total_face=j_total_face,
    delta_upper_face=delta_upper_face,
    delta_lower_face=delta_lower_face,
    elongation=elongation,
    elongation_face=elongation_face,
    spr_hires=spr_hires,
    rho_hires_norm=rho_hires_norm,
    rho_hires=rho_hires,
    Phi_b_dot=np.asarray(0.0),
    _z_magnetic_axis=None,
)
g.pedestal_model = PedestalConfig().build_pedestal_model()
g.source_models = g.torax_config.sources.build_models()
g.transport_model = QLKNNTransportModel0()
g.bootstrap_current = SauterModelConfig().build_model()
g.runtime_params_provider = RuntimeParamsProvider.from_config()
runtime_params_for_init, geo_for_init = get_consistent_runtime_params_and_geometry(
    t=g.t_initial, )
runtime_params = runtime_params_for_init
geo = geo_for_init
T_i = CellVariable(
    value=runtime_params.profile_conditions.T_i,
    left_face_grad_constraint=jnp.zeros(()),
    right_face_grad_constraint=None,
    right_face_constraint=runtime_params.profile_conditions.T_i_right_bc,
    dr=geo.drho_norm,
)
T_e = get_updated_electron_temperature(runtime_params.profile_conditions, geo)
n_e = get_updated_electron_density(runtime_params.profile_conditions, geo)
ions = get_updated_ions(runtime_params, geo, n_e, T_e)
v_loop_lcfs = (
    np.array(runtime_params.profile_conditions.v_loop_lcfs)
    if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
    else np.array(0.0, dtype=jnp.float64))
psidot = CellVariable(
    value=np.zeros_like(geo.rho),
    dr=geo.drho_norm,
)
psi = CellVariable(value=np.zeros_like(geo.rho), dr=geo.drho_norm)
core_profiles = CoreProfiles(
    T_i=T_i,
    T_e=T_e,
    n_e=n_e,
    n_i=ions.n_i,
    Z_i=ions.Z_i,
    Z_i_face=ions.Z_i_face,
    A_i=ions.A_i,
    n_impurity=ions.n_impurity,
    impurity_fractions=ions.impurity_fractions,
    Z_impurity=ions.Z_impurity,
    Z_impurity_face=ions.Z_impurity_face,
    A_impurity=ions.A_impurity,
    A_impurity_face=ions.A_impurity_face,
    Z_eff=ions.Z_eff,
    Z_eff_face=ions.Z_eff_face,
    psi=psi,
    psidot=psidot,
    q_face=np.zeros_like(geo.rho_face),
    s_face=np.zeros_like(geo.rho_face),
    v_loop_lcfs=v_loop_lcfs,
    sigma=np.zeros_like(geo.rho),
    sigma_face=np.zeros_like(geo.rho_face),
    j_total=np.zeros_like(geo.rho),
    j_total_face=np.zeros_like(geo.rho_face),
    Ip_profile_face=np.zeros_like(geo.rho_face),
)
sources_are_calculated = False
source_profiles = SourceProfiles(bootstrap_current=BootstrapCurrent.zeros(geo),
                                 qei=QeiInfo.zeros(geo))
dpsi_drhonorm_edge = calculate_psi_grad_constraint_from_Ip(
    runtime_params.profile_conditions.Ip,
    geo,
)
assert not runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
psi = CellVariable(
    value=geo.psi_from_Ip,
    right_face_grad_constraint=(
        None
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else dpsi_drhonorm_edge),
    right_face_constraint=(
        geo.psi_from_Ip_face[-1]
        if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
        else None),
    dr=geo.drho_norm,
)
j_total, j_total_face, Ip_profile_face = calc_j_total(geo, psi)
core_profiles = dataclasses.replace(
    core_profiles,
    psi=psi,
    q_face=calc_q_face(geo, psi),
    s_face=calc_s_face(geo, psi),
    j_total=j_total,
    j_total_face=j_total_face,
    Ip_profile_face=Ip_profile_face,
)
conductivity = calculate_conductivity(
    geo,
    core_profiles,
)
if not sources_are_calculated:
    build_standard_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    bootstrap_current = g.bootstrap_current.calculate_bootstrap_current(
        geo, core_profiles)
    source_profiles = dataclasses.replace(source_profiles,
                                          bootstrap_current=bootstrap_current)
psi_sources = source_profiles.total_psi_sources(geo)
psidot_value = calculate_psidot_from_psi_sources(psi_sources=psi_sources,
                                                 sigma=conductivity.sigma,
                                                 psi=psi,
                                                 geo=geo)
v_loop_lcfs = (
    runtime_params.profile_conditions.v_loop_lcfs
    if runtime_params.profile_conditions.use_v_loop_lcfs_boundary_condition
    else psidot_value[-1])
psidot = dataclasses.replace(
    core_profiles.psidot,
    value=psidot_value,
    right_face_constraint=v_loop_lcfs,
    right_face_grad_constraint=None,
)
initial_core_profiles = dataclasses.replace(
    core_profiles,
    psidot=psidot,
    sigma=conductivity.sigma,
    sigma_face=conductivity.sigma_face,
)
conductivity = Conductivity(sigma=initial_core_profiles.sigma,
                            sigma_face=initial_core_profiles.sigma_face)
core_profiles = initial_core_profiles
explicit_source_profiles = build_source_profiles0(
    runtime_params=runtime_params,
    geo=geo,
    core_profiles=core_profiles,
)
initial_core_sources = build_source_profiles1(
    runtime_params=runtime_params,
    geo=geo,
    core_profiles=core_profiles,
    explicit=False,
    explicit_source_profiles=explicit_source_profiles,
    conductivity=conductivity,
)
transport_coeffs = calculate_total_transport_coeffs(
    runtime_params,
    geo,
    initial_core_profiles,
)
current_state = ToraxSimState(
    t=np.array(g.t_initial),
    dt=np.zeros(()),
    core_profiles=initial_core_profiles,
    core_sources=initial_core_sources,
    core_transport=transport_coeffs,
    solver_numeric_outputs=SolverNumericOutputs(solver_error_state=0, ),
    geometry=geo,
)
post_processed_outputs = make_post_processed_outputs(current_state,
                                                     runtime_params_for_init)
initial_post_processed_outputs = post_processed_outputs
state_history = [current_state]
post_processing_history = [initial_post_processed_outputs]
initial_runtime_params = g.runtime_params_provider(current_state.t)
while not_done(current_state.t, g.t_final):
    previous_post_processed_outputs = post_processing_history[-1]
    runtime_params_t, geo_t = get_consistent_runtime_params_and_geometry(
        t=current_state.t)
    explicit_source_profiles = build_source_profiles0(
        runtime_params=runtime_params_t,
        geo=geo_t,
        core_profiles=current_state.core_profiles,
    )
    initial_dt = next_dt(current_state.t, runtime_params_t, geo_t,
                         current_state.core_transport)
    _, result = while_loop(
        cond_fun,
        body_fun,
        (
            initial_dt,
            (
                core_profiles_to_solver_x_tuple(current_state.core_profiles),
                initial_dt,
                SolverNumericOutputs(solver_error_state=1, ),
                runtime_params_t,
                geo_t,
                current_state.core_profiles,
            ),
        ),
    )
    output_state, post_processed_outputs = _finalize_outputs(
        t=current_state.t,
        dt=result[1],
        x_new=result[0],
        solver_numeric_outputs=result[2],
        runtime_params_t_plus_dt=result[3],
        geometry_t_plus_dt=result[4],
        core_profiles_t=current_state.core_profiles,
        core_profiles_t_plus_dt=result[5],
        explicit_source_profiles=explicit_source_profiles,
        input_post_processed_outputs=previous_post_processed_outputs,
    )
    current_state = output_state
    state_history.append(current_state)
    post_processing_history.append(post_processed_outputs)
t = np.array([state.t for state in state_history])
rho = np.concatenate([[0.0], state_history[0].geometry.rho_norm, [1.0]])
(nt, ) = np.shape(t)
evolving_data = {}
for var_name in g.evolving_names:
    var_data = []
    for state in state_history:
        var_cell = getattr(state.core_profiles, var_name)
        if hasattr(var_cell, "cell_plus_boundaries"):
            data = var_cell.cell_plus_boundaries()
        else:
            data = var_cell.value
        var_data.append(data)
    evolving_data[var_name] = np.stack(var_data)
with open("run.raw", "wb") as f:
    t.tofile(f)
    rho.tofile(f)
    for key in g.evolving_names:
        evolving_data[key].tofile(f)
for key in g.evolving_names:
    var = evolving_data[key]
    lo = np.min(var).item()
    hi = np.max(var).item()
    for i, idx in enumerate([0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]):
        plt.title(f"time: {t[idx]:8.3e}")
        plt.axis([None, None, lo, hi])
        plt.plot(rho, var[idx], "o-")
        plt.savefig(f"{key}.{i:04d}.png")
        plt.close()
