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
import logging
import numpy as np
import os
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
JAX_STATIC: Final[str] = "_pydantic_jax_static_field"
_interp_fn = jax.jit(jnp.interp)
thread_context = threading.local()
_TOLERANCE: Final[float] = 1e-6
g.keV_to_J = 1e3 * 1.602176634e-19
g.eV_to_J = 1.602176634e-19
g.m_amu = 1.6605390666e-27
g.q_e = 1.602176634e-19
g.m_e = 9.1093837e-31
g.epsilon_0 = 8.85418782e-12
g.mu_0 = 4 * jnp.pi * 1e-7
g.k_B = 1.380649e-23
g.eps = 1e-7
g.sym = "D", "T", "Ne"
g.z = dict(zip(g.sym, [1.0, 1.0, 10.0]))
g.A = dict(zip(g.sym, [2.0141, 3.0160, 20.180]))


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

    def chi_max(self):
        return jnp.maximum(
            jnp.max(self.chi_face_ion * g.geo_g1_over_vpr2_face),
            jnp.max(self.chi_face_el * g.geo_g1_over_vpr2_face),
        )


def calculate_psidot_from_psi_sources(*, psi_sources, sigma, psi):
    toc_psi = (1.0 / g.resistivity_multiplier * g.cell_centers * sigma *
               g.mu_0 * 16 * jnp.pi**2 * g.geo_Phi_b**2 / g.geo_F**2)
    d_face_psi = g.geo_g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    diffusion_mat, diffusion_vec = make_diffusion_terms(d_face_psi, psi)
    conv_mat, conv_vec = make_convection_terms(v_face_psi, d_face_psi, psi)
    c_mat = diffusion_mat + conv_mat
    c = diffusion_vec + conv_vec
    c += psi_sources
    psidot = (jnp.dot(c_mat, psi.value) + c) / toc_psi
    return psidot


def _calculate_log_tau_e_Z1(T_e, n_e, log_lambda_ei):
    return (jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei)) -
            4 * jnp.log(g.q_e) + 0.5 * jnp.log(g.m_e / 2.0) +
            2 * jnp.log(g.epsilon_0) + 1.5 * jnp.log(T_e * g.keV_to_J))


_MAVRIN_Z_COEFFS = immutabledict.immutabledict({
    "Ne":
    np.array([
        [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
        [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
    ]),
})
_TEMPERATURE_INTERVALS = immutabledict.immutabledict({
    "Ne":
    np.array([0.5, 2.0]),
})


def calculate_average_charge_state_single_species(T_e, ion_symbol):
    if ion_symbol not in _MAVRIN_Z_COEFFS:
        return jnp.ones_like(T_e) * g.z[ion_symbol]
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
    return Z_avg, Z2_avg, Z_per_species


def calculate_f_trap():
    epsilon_effective = (
        0.67 * (1.0 - 1.4 * jnp.abs(g.geo_delta_face) * g.geo_delta_face) *
        g.geo_epsilon_face)
    aa = (1.0 - g.geo_epsilon_face) / (1.0 + g.geo_epsilon_face)
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


def calculate_nu_e_star(q, n_e, T_e, Z_eff, log_lambda_ei):
    return (6.921e-18 * q * g.R_major * n_e * Z_eff * log_lambda_ei /
            (((T_e * 1e3)**2) * (g.geo_epsilon_face + g.eps)**1.5))


def calculate_nu_i_star(q, n_i, T_i, Z_eff, log_lambda_ii):
    return (4.9e-18 * q * g.R_major * n_i * Z_eff**4 * log_lambda_ii /
            (((T_i * 1e3)**2) * (g.geo_epsilon_face + g.eps)**1.5))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class Conductivity:
    sigma: Any
    sigma_face: Any


def calculate_conductivity(core_profiles):
    # Inlined _calculate_conductivity0
    f_trap = calculate_f_trap()
    NZ = 0.58 + 0.74 / (0.76 + core_profiles.Z_eff_face)
    log_lambda_ei = 31.3 - 0.5 * jnp.log(
        core_profiles.n_e.face_value()) + jnp.log(
            core_profiles.T_e.face_value() * 1e3)
    sigsptz = (1.9012e04 * (core_profiles.T_e.face_value() * 1e3)**1.5 /
               core_profiles.Z_eff_face / NZ / log_lambda_ei)
    nu_e_star_face = calculate_nu_e_star(
        q=core_profiles.q_face,
        n_e=core_profiles.n_e.face_value(),
        T_e=core_profiles.T_e.face_value(),
        Z_eff=core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    ft33 = f_trap / (1.0 +
                     (0.55 - 0.1 * f_trap) * jnp.sqrt(nu_e_star_face) + 0.45 *
                     (1.0 - f_trap) * nu_e_star_face /
                     (core_profiles.Z_eff_face**1.5))
    signeo_face = 1.0 - ft33 * (1.0 + 0.36 / core_profiles.Z_eff_face - ft33 *
                                (0.59 / core_profiles.Z_eff_face -
                                 0.23 / core_profiles.Z_eff_face * ft33))
    sigma_face = sigsptz * signeo_face
    sigmaneo_cell = 0.5 * (sigma_face[:-1] + sigma_face[1:])
    return Conductivity(
        sigma=sigmaneo_cell,
        sigma_face=sigma_face,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
    j_bootstrap: jax.Array
    j_bootstrap_face: jax.Array

    @classmethod
    def zeros(cls):
        return cls(
            j_bootstrap=jnp.zeros_like(g.cell_centers),
            j_bootstrap_face=jnp.zeros_like(g.face_centers),
        )


@jax.jit
def _calculate_bootstrap_current(*, Z_eff_face, Z_i_face, n_e, n_i, T_e, T_i,
                                 psi, q_face):
    f_trap = calculate_f_trap()
    log_lambda_ei = 31.3 - 0.5 * jnp.log(n_e.face_value()) + jnp.log(
        T_e.face_value() * 1e3)
    T_i_ev = T_i.face_value() * 1e3
    log_lambda_ii = (30.0 - 0.5 * jnp.log(n_i.face_value()) +
                     1.5 * jnp.log(T_i_ev) - 3.0 * jnp.log(Z_i_face))
    nu_e_star = calculate_nu_e_star(
        q=q_face,
        n_e=n_e.face_value(),
        T_e=T_e.face_value(),
        Z_eff=Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    nu_i_star = calculate_nu_i_star(
        q=q_face,
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
    prefactor = -g.geo_F_face * bootstrap_multiplier * 2 * jnp.pi / g.geo_B_0
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
    j_bootstrap = 0.5 * (j_bootstrap_face[:-1] + j_bootstrap_face[1:])
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


def exponential_profile(*, decay_start, width, total):
    r = g.cell_centers
    S = jnp.exp(-(decay_start - r) / width)
    C = total / jnp.sum(S * g.geo_vpr * jnp.array(g.dx))
    return C * S


def gaussian_profile(*, center, width, total):
    r = g.cell_centers
    S = jnp.exp(-((r - center)**2) / (2 * width**2))
    C = total / jnp.sum(S * g.geo_vpr * jnp.array(g.dx))
    return C * S


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QeiInfo:
    implicit_ii: jax.Array
    implicit_ee: jax.Array
    implicit_ie: jax.Array
    implicit_ei: jax.Array

    @classmethod
    def zeros(cls):
        return QeiInfo(
            implicit_ii=jnp.zeros_like(g.geo_rho),
            implicit_ee=jnp.zeros_like(g.geo_rho),
            implicit_ie=jnp.zeros_like(g.geo_rho),
            implicit_ei=jnp.zeros_like(g.geo_rho),
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

    def total_psi_sources(self):
        total = self.bootstrap_current.j_bootstrap
        total += sum(self.psi.values())
        mu0 = g.mu_0
        prefactor = 8 * g.geo_vpr * jnp.pi**2 * g.geo_B_0 * mu0 * g.geo_Phi_b / g.geo_F**2
        return -total * prefactor

    def total_sources(self, source_type):
        source = getattr(self, source_type)
        total = sum(source.values())
        return total * g.geo_vpr


@enum.unique
class Mode(enum.Enum):
    ZERO = "ZERO"
    MODEL_BASED = "MODEL_BASED"


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
    PSI = 1
    NE = 2
    TEMP_ION = 3
    TEMP_EL = 4


class SourceHandler(typing.NamedTuple):
    affects: tuple[AffectedCoreProfile, ...]
    eval_fn: typing.Callable


def calculate_generic_current(
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    I_generic = g.Ip * g.generic_current_fraction
    generic_current_form = jnp.exp(
        -((g.cell_centers - g.generic_current_location)**2) /
        (2 * g.generic_current_width**2))
    Cext = I_generic / jnp.sum(
        generic_current_form * g.geo_spr * jnp.array(g.dx))
    generic_current_profile = Cext * generic_current_form
    return (generic_current_profile, )


def default_formula(source_name, unused_core_profiles,
                    unused_calculated_source_profiles, unused_conductivity):
    absorbed_power = g.generic_heat_P_total * 1.0  # absorption_fraction is always 1.0
    profile = gaussian_profile(center=g.generic_heat_location,
                               width=g.generic_heat_width,
                               total=absorbed_power)
    ion = profile * (1 - g.generic_heat_electron_fraction)
    el = profile * g.generic_heat_electron_fraction
    return (ion, el)


def calc_generic_particle_source(
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    return (gaussian_profile(
        center=g.generic_particle_location,
        width=g.generic_particle_width,
        total=g.generic_particle_S_total,
    ), )


def calc_pellet_source(
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    return (gaussian_profile(
        center=g.pellet_location,
        width=g.pellet_width,
        total=g.pellet_S_total,
    ), )


def fusion_heat_model_func(
    unused_source_name,
    core_profiles,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    # Inlined calc_fusion
    product = 1.0
    for fraction, symbol in zip(g.main_ion_fractions, g.main_ion_names):
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
    return (Pfus_i, Pfus_e)


def calc_puff_source(
    source_name,
    unused_state,
    unused_calculated_source_profiles,
    unused_conductivity,
):
    return (exponential_profile(
        decay_start=1.0,
        width=g.gas_puff_decay_length,
        total=g.gas_puff_S_total,
    ), )


@jax.jit
def build_source_profiles0(core_profiles,
                           explicit_source_profiles=None,
                           conductivity=None):
    qei = QeiInfo.zeros()
    bootstrap_current = BootstrapCurrent.zeros()
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
        core_profiles=core_profiles,
        explicit=True,
        conductivity=conductivity,
    )
    return profiles


@jax.jit
def build_source_profiles1(core_profiles,
                           explicit_source_profiles=None,
                           conductivity=None):
    # Inlined get_qei
    zeros = jnp.zeros_like(g.cell_centers)
    log_lambda_ei = 31.3 - 0.5 * jnp.log(core_profiles.n_e.value) + jnp.log(
        core_profiles.T_e.value * 1e3)
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
    qei = QeiInfo(
        implicit_ii=-qei_coef,
        implicit_ee=-qei_coef,
        implicit_ie=qei_coef,
        implicit_ei=qei_coef,
    )
    result = _calculate_bootstrap_current(
        Z_eff_face=core_profiles.Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
    )
    bootstrap_current = BootstrapCurrent(
        j_bootstrap=result.j_bootstrap,
        j_bootstrap_face=result.j_bootstrap_face,
    )
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
        core_profiles=core_profiles,
        explicit=False,
        conductivity=conductivity,
    )
    return profiles


def build_standard_source_profiles(*,
                                   calculated_source_profiles,
                                   core_profiles,
                                   explicit=True,
                                   conductivity=None,
                                   calculate_anyway=False,
                                   psi_only=False):

    def calculate_source(source_name):
        handler = g.source_registry[source_name]
        # All sources have is_explicit=False, so check: (explicit == False) | calculate_anyway
        if (not explicit) | calculate_anyway:
            value = handler.eval_fn(
                source_name,
                core_profiles,
                calculated_source_profiles,
                conductivity,
            )
            for profile, affected_core_profile in zip(value,
                                                      handler.affects,
                                                      strict=True):
                match affected_core_profile:
                    case AffectedCoreProfile.PSI:
                        calculated_source_profiles.psi[source_name] = profile
                    case AffectedCoreProfile.NE:
                        calculated_source_profiles.n_e[source_name] = profile
                    case AffectedCoreProfile.TEMP_ION:
                        calculated_source_profiles.T_i[source_name] = profile
                    case AffectedCoreProfile.TEMP_EL:
                        calculated_source_profiles.T_e[source_name] = profile

    # Calculate PSI sources first
    for source_name in g.psi_source_names:
        calculate_source(source_name)
    if psi_only:
        return
    # Calculate non-PSI sources
    for source_name in g.source_registry.keys():
        if source_name not in g.psi_source_names:
            calculate_source(source_name)


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
_EPSILON_NN: Final[float] = 1 / 3


def calculate_transport_coeffs(core_profiles, rho_norm_ped_top_idx):
    # Inlined call_qlknn_implementation and prepare_qualikiz_inputs
    rmid = (g.geo_R_out - g.geo_R_in) * 0.5
    rmid_face = (g.geo_R_out_face - g.geo_R_in_face) * 0.5
    chiGB = ((core_profiles.A_i * g.m_amu)**0.5 / (g.geo_B_0 * g.q_e)**2 *
             (core_profiles.T_i.face_value() * g.keV_to_J)**1.5 /
             g.geo_a_minor)
    # Inlined NormalizedLogarithmicGradients.from_profiles
    lref_over_lti_result = jnp.where(
        jnp.abs(core_profiles.T_i.face_value()) < g.eps,
        g.eps,
        -g.R_major * core_profiles.T_i.face_grad(rmid) /
        core_profiles.T_i.face_value(),
    )
    lref_over_lti = jnp.where(
        jnp.abs(lref_over_lti_result) < g.eps, g.eps, lref_over_lti_result)
    lref_over_lte_result = jnp.where(
        jnp.abs(core_profiles.T_e.face_value()) < g.eps,
        g.eps,
        -g.R_major * core_profiles.T_e.face_grad(rmid) /
        core_profiles.T_e.face_value(),
    )
    lref_over_lte = jnp.where(
        jnp.abs(lref_over_lte_result) < g.eps, g.eps, lref_over_lte_result)
    lref_over_lne_result = jnp.where(
        jnp.abs(core_profiles.n_e.face_value()) < g.eps,
        g.eps,
        -g.R_major * core_profiles.n_e.face_grad(rmid) /
        core_profiles.n_e.face_value(),
    )
    lref_over_lne = jnp.where(
        jnp.abs(lref_over_lne_result) < g.eps, g.eps, lref_over_lne_result)
    lref_over_lni0_result = jnp.where(
        jnp.abs(core_profiles.n_i.face_value()) < g.eps,
        g.eps,
        -g.R_major * core_profiles.n_i.face_grad(rmid) /
        core_profiles.n_i.face_value(),
    )
    lref_over_lni0 = jnp.where(
        jnp.abs(lref_over_lni0_result) < g.eps, g.eps, lref_over_lni0_result)
    lref_over_lni1_result = jnp.where(
        jnp.abs(core_profiles.n_impurity.face_value()) < g.eps,
        g.eps,
        -g.R_major * core_profiles.n_impurity.face_grad(rmid) /
        core_profiles.n_impurity.face_value(),
    )
    lref_over_lni1 = jnp.where(
        jnp.abs(lref_over_lni1_result) < g.eps, g.eps, lref_over_lni1_result)
    q = core_profiles.q_face
    iota_scaled = jnp.abs(
        (core_profiles.psi.face_grad()[1:] / g.face_centers[1:]))
    iota_scaled0 = jnp.expand_dims(jnp.abs(core_profiles.psi.face_grad()[1] /
                                           jnp.array(g.dx)),
                                   axis=0)
    iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
    rmid_face = (g.geo_R_out_face - g.geo_R_in_face) * 0.5
    smag = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled
    epsilon_lcfs = rmid_face[-1] / g.R_major
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < g.eps, g.eps, x)
    Ti_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
    log_lambda_ei_face = (31.3 -
                          0.5 * jnp.log(core_profiles.n_e.face_value()) +
                          jnp.log(core_profiles.T_e.face_value() * 1e3))
    log_tau_e_Z1 = _calculate_log_tau_e_Z1(
        core_profiles.T_e.face_value(),
        core_profiles.n_e.face_value(),
        log_lambda_ei_face,
    )
    nu_e = (1 / jnp.exp(log_tau_e_Z1) * core_profiles.Z_eff_face *
            g.collisionality_multiplier)
    epsilon = g.geo_rho_face / g.R_major
    epsilon = jnp.clip(epsilon, g.eps)
    tau_bounce = (core_profiles.q_face * g.R_major / (epsilon**1.5 * jnp.sqrt(
        core_profiles.T_e.face_value() * g.keV_to_J / g.m_e)))
    tau_bounce = tau_bounce.at[0].set(tau_bounce[1])
    nu_star = nu_e * tau_bounce
    log_nu_star_face = jnp.log10(nu_star)
    factor_0 = 2 * g.keV_to_J / g.geo_B_0**2 * g.mu_0 * q**2
    alpha = factor_0 * (
        core_profiles.T_e.face_value() * core_profiles.n_e.face_value() *
        (lref_over_lte + lref_over_lne) +
        core_profiles.n_i.face_value() * core_profiles.T_i.face_value() *
        (lref_over_lti + lref_over_lni0) +
        core_profiles.n_impurity.face_value() *
        core_profiles.T_i.face_value() * (lref_over_lti + lref_over_lni1))
    # smag_alpha_correction is always True, so always apply correction
    smag = smag - alpha / 2
    # q_sawtooth_proxy is always True, so simplify to just q < 1 condition
    smag = jnp.where(q < 1, 0.1, smag)
    q = jnp.where(q < 1, 1, q)
    smag = jnp.where(
        smag - alpha < -0.2,
        alpha - 0.2,
        smag,
    )
    normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value()
    qualikiz_inputs = QualikizInputs(
        Z_eff_face=core_profiles.Z_eff_face,
        lref_over_lti=lref_over_lti,
        lref_over_lte=lref_over_lte,
        lref_over_lne=lref_over_lne,
        lref_over_lni0=lref_over_lni0,
        lref_over_lni1=lref_over_lni1,
        q=q,
        smag=smag,
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=g.R_major,
        Rmin=g.geo_a_minor,
        alpha=alpha,
        epsilon_lcfs=epsilon_lcfs,
    )
    qualikiz_inputs = dataclasses.replace(
        qualikiz_inputs,
        x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / _EPSILON_NN,
    )
    # Inlined get_model_inputs_from_qualikiz_inputs
    # Map model input names to QualikizInputs attributes
    input_map = {
        "Ati": lambda x: x.lref_over_lti,
        "Ate": lambda x: x.lref_over_lte,
        "Ane": lambda x: x.lref_over_lne,
        "Ani": lambda x: x.lref_over_lni0,
        "LogNuStar": lambda x: x.log_nu_star_face,
    }

    def _get_input(key):
        return jnp.array(
            input_map.get(key, lambda x: getattr(x, key))(qualikiz_inputs),
            dtype=jnp.float64,
        )

    feature_scan = jnp.array(
        [_get_input(key) for key in g.model.inputs_and_ranges.keys()],
        dtype=jnp.float64,
    ).T
    model_predictions = g.model.predict(feature_scan)
    model_output = {
        _FLUX_NAME_MAP.get(flux_name, flux_name): flux_value
        for flux_name, flux_value in model_predictions.items()
    }
    qi_itg_squeezed = model_output["qi_itg"].squeeze()
    qi = qi_itg_squeezed + model_output["qi_tem"].squeeze()
    qe = (model_output["qe_itg"].squeeze() * g.ITG_flux_ratio_correction +
          model_output["qe_tem"].squeeze() +
          model_output["qe_etg"].squeeze() * g.ETG_correction_factor)
    pfe = model_output["pfe_itg"].squeeze() + model_output["pfe_tem"].squeeze()
    # Inline make_core_transport
    gradient_reference_length = g.R_major
    gyrobohm_flux_reference_length = g.geo_a_minor
    pfe_SI = (pfe * core_profiles.n_e.face_value() * qualikiz_inputs.chiGB /
              gyrobohm_flux_reference_length)
    chi_face_ion = ((
        (gradient_reference_length / gyrobohm_flux_reference_length) * qi) /
                    qualikiz_inputs.lref_over_lti) * qualikiz_inputs.chiGB
    chi_face_el = ((
        (gradient_reference_length / gyrobohm_flux_reference_length) * qe) /
                   qualikiz_inputs.lref_over_lte) * qualikiz_inputs.chiGB
    # DV_effective_approach (only branch used)
    Deff = -pfe_SI / (core_profiles.n_e.face_grad() * g.geo_g1_over_vpr2_face *
                      g.geo_rho_b + g.eps)
    Veff = pfe_SI / (core_profiles.n_e.face_value() * g.geo_g0_over_vpr_face *
                     g.geo_rho_b)
    Deff_mask = (((pfe >= 0) & (qualikiz_inputs.lref_over_lne >= 0))
                 | ((pfe < 0) & (qualikiz_inputs.lref_over_lne < 0))) & (abs(
                     qualikiz_inputs.lref_over_lne) >= g.An_min)
    Veff_mask = jnp.invert(Deff_mask)
    d_face_el = jnp.where(Veff_mask, 0.0, Deff)
    v_face_el = jnp.where(Deff_mask, 0.0, Veff)
    transport_coeffs = TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
    # Inlined apply_domain_restriction_transport
    active_mask = ((g.face_centers > g.transport_rho_min)
                   & (g.face_centers <= g.transport_rho_max)
                   & (g.face_centers <= g.rho_norm_ped_top))
    active_mask = (jnp.asarray(active_mask).at[0].set(
        g.transport_rho_min == 0))
    chi_face_ion = jnp.where(active_mask, transport_coeffs.chi_face_ion, 0.0)
    chi_face_el = jnp.where(active_mask, transport_coeffs.chi_face_el, 0.0)
    d_face_el = jnp.where(active_mask, transport_coeffs.d_face_el, 0.0)
    v_face_el = jnp.where(active_mask, transport_coeffs.v_face_el, 0.0)
    transport_coeffs = dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
    # Inlined apply_clipping_transport
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
        g.D_e_max,
    )
    v_face_el = jnp.clip(
        transport_coeffs.v_face_el,
        g.V_e_min,
        g.V_e_max,
    )
    transport_coeffs = dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
    # Inlined apply_transport_patches
    chi_face_ion = jnp.where(
        g.face_centers < g.rho_inner + g.eps,
        g.chi_i_inner,
        transport_coeffs.chi_face_ion,
    )
    chi_face_el = jnp.where(
        g.face_centers < g.rho_inner + g.eps,
        g.chi_e_inner,
        transport_coeffs.chi_face_el,
    )
    d_face_el = jnp.where(
        g.face_centers < g.rho_inner + g.eps,
        g.D_e_inner,
        transport_coeffs.d_face_el,
    )
    v_face_el = jnp.where(
        g.face_centers < g.rho_inner + g.eps,
        g.V_e_inner,
        transport_coeffs.v_face_el,
    )
    transport_coeffs = dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
    # Inlined smooth_coeffs_transport
    lower_cutoff = 0.01
    kernel = jnp.exp(-jnp.log(2) *
                     (g.face_centers[:, jnp.newaxis] - g.face_centers)**2 /
                     (g.smoothing_width**2 + g.eps))
    mask_outer_edge = g.rho_norm_ped_top - g.eps
    mask_inner_edge = g.rho_inner + g.eps
    mask = jnp.where(
        jnp.logical_and(
            g.face_centers > mask_inner_edge,
            g.face_centers < mask_outer_edge,
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
    smoothing_matrix = kernel

    def smooth_single_coeff(coeff):
        return jax.lax.cond(
            jnp.all(coeff == 0.0),
            lambda: coeff,
            lambda: jnp.dot(smoothing_matrix, coeff),
        )

    return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)


@jax.jit
def calculate_total_transport_coeffs(core_profiles):
    rho_norm_ped_top_idx = jnp.abs(g.cell_centers -
                                   g.rho_norm_ped_top).argmin()
    turbulent_transport = calculate_transport_coeffs(
        core_profiles=core_profiles,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
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


def get_updated_electron_density():
    nGW = g.Ip / 1e6 / (jnp.pi * g.geo_a_minor**2) * 1e20
    n_e_value = g.n_e * nGW
    n_e_right_bc = g.n_e_right_bc
    face_left = n_e_value[0]
    face_right = n_e_right_bc
    face_inner = (n_e_value[..., :-1] + n_e_value[..., 1:]) / 2.0
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]], )
    a_minor_out = g.geo_R_out_face[-1] - g.geo_R_out_face[0]
    target_nbar = jnp.where(
        True,
        g.nbar * nGW,
        g.nbar,
    )
    nbar_from_n_e_face_inner = (
        jax.scipy.integrate.trapezoid(n_e_face[:-1], g.geo_R_out_face[:-1]) /
        a_minor_out)
    dr_edge = g.geo_R_out_face[-1] - g.geo_R_out_face[-2]
    C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
        nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out)
    n_e_value = C * n_e_value
    n_e = CellVariable(
        value=n_e_value,
        dr=jnp.array(g.dx),
        right_face_grad_constraint=None,
        right_face_constraint=n_e_right_bc,
    )
    return n_e


@jax.jit
def get_updated_ions(n_e, T_e):
    Z_i_avg, Z_i_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.main_ion_names,
        T_e=T_e.value,
        fractions=g.main_ion_fractions,
    )
    Z_i = Z_i_Z2_avg / Z_i_avg
    Z_i_face_avg, Z_i_face_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.main_ion_names,
        T_e=T_e.face_value(),
        fractions=g.main_ion_fractions,
    )
    Z_i_face = Z_i_face_Z2_avg / Z_i_face_avg
    Z_impurity_avg, Z_impurity_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.impurity_names,
        T_e=T_e.value,
        fractions=g.impurity_fractions,
    )
    Z_impurity = Z_impurity_Z2_avg / Z_impurity_avg
    Z_impurity_face_avg, Z_impurity_face_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.impurity_names,
        T_e=T_e.face_value(),
        fractions=g.impurity_fractions_face,
    )
    Z_impurity_face = Z_impurity_face_Z2_avg / Z_impurity_face_avg
    Z_eff = g.Z_eff
    Z_eff_edge = g.Z_eff
    dilution_factor = jnp.where(
        Z_eff == 1.0,
        1.0,
        (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i)),
    )
    dilution_factor_edge = jnp.where(
        Z_eff_edge == 1.0,
        1.0,
        (Z_impurity_face[-1] - Z_eff_edge) /
        (Z_i_face[-1] * (Z_impurity_face[-1] - Z_i_face[-1])),
    )
    n_i = CellVariable(
        value=n_e.value * dilution_factor,
        dr=jnp.array(g.dx),
        right_face_grad_constraint=None,
        right_face_constraint=n_e.right_face_constraint * dilution_factor_edge,
    )
    n_impurity_value = jnp.where(
        dilution_factor == 1.0,
        0.0,
        (n_e.value - n_i.value * Z_i) / Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        dilution_factor_edge == 1.0,
        0.0,
        (n_e.right_face_constraint - n_i.right_face_constraint * Z_i_face[-1])
        / Z_impurity_face[-1],
    )
    n_impurity = CellVariable(
        value=n_impurity_value,
        dr=jnp.array(g.dx),
        right_face_grad_constraint=None,
        right_face_constraint=n_impurity_right_face_constraint,
    )
    Z_eff_face = (Z_i_face**2 * n_i.face_value() + Z_impurity_face**2 *
                  n_impurity.face_value()) / n_e.face_value()
    impurity_fractions_dict = {}
    for i, symbol in enumerate(g.impurity_names):
        fraction = g.impurity_fractions[i]
        impurity_fractions_dict[symbol] = fraction
    return Ions(
        n_i=n_i,
        n_impurity=n_impurity,
        impurity_fractions=impurity_fractions_dict,
        Z_i=Z_i,
        Z_i_face=Z_i_face,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
        A_i=g.main_ion_A_avg,
        A_impurity=g.impurity_A_avg,
        A_impurity_face=g.impurity_A_avg_face,
        Z_eff=Z_eff,
        Z_eff_face=Z_eff_face,
    )


def core_profiles_to_solver_x_tuple(core_profiles, ):
    x_tuple_for_solver_list = []
    for name in g.evolving_names:
        original_units_cv = getattr(core_profiles, name)
        # Inlined scale_cell_variable
        scaling_factor = 1 / SCALING_FACTORS[name]
        operation = lambda x, factor: x * factor if x is not None else None
        solver_x_tuple_cv = CellVariable(
            value=operation(original_units_cv.value, scaling_factor),
            left_face_constraint=operation(
                original_units_cv.left_face_constraint, scaling_factor),
            left_face_grad_constraint=operation(
                original_units_cv.left_face_grad_constraint, scaling_factor),
            right_face_constraint=operation(
                original_units_cv.right_face_constraint, scaling_factor),
            right_face_grad_constraint=operation(
                original_units_cv.right_face_grad_constraint, scaling_factor),
            dr=original_units_cv.dr,
        )
        x_tuple_for_solver_list.append(solver_x_tuple_cv)
    return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_core_profiles(x_new, core_profiles):
    updated_vars = {}
    for i, var_name in enumerate(g.evolving_names):
        solver_x_tuple_cv = x_new[i]
        # Inlined scale_cell_variable
        scaling_factor = SCALING_FACTORS[var_name]
        operation = lambda x, factor: x * factor if x is not None else None
        original_units_cv = CellVariable(
            value=operation(solver_x_tuple_cv.value, scaling_factor),
            left_face_constraint=operation(
                solver_x_tuple_cv.left_face_constraint, scaling_factor),
            left_face_grad_constraint=operation(
                solver_x_tuple_cv.left_face_grad_constraint, scaling_factor),
            right_face_constraint=operation(
                solver_x_tuple_cv.right_face_constraint, scaling_factor),
            right_face_grad_constraint=operation(
                solver_x_tuple_cv.right_face_grad_constraint, scaling_factor),
            dr=solver_x_tuple_cv.dr,
        )
        updated_vars[var_name] = original_units_cv
    return dataclasses.replace(core_profiles, **updated_vars)


OptionalTupleMatrix: TypeAlias = tuple[tuple[jax.Array | None, ...],
                                       ...] | None
AuxiliaryOutput: TypeAlias = Any


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


def coeffs_callback(core_profiles,
                    x,
                    explicit_source_profiles,
                    explicit_call=False):
    # Inlined update_core_profiles_during_step
    updated_core_profiles = solver_x_tuple_to_core_profiles(x, core_profiles)
    ions = get_updated_ions(
        updated_core_profiles.n_e,
        updated_core_profiles.T_e,
    )
    core_profiles = dataclasses.replace(
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
        q_face=jnp.concatenate([
            jnp.expand_dims(
                jnp.abs((2 * g.geo_Phi_b * jnp.array(g.dx)) /
                        updated_core_profiles.psi.face_grad()[1]), 0),
            jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) /
                    updated_core_profiles.psi.face_grad()[1:])
        ]) * g.geo_q_correction_factor,
    )
    # Inlined calc_coeffs
    if explicit_call and g.theta_implicit == 1.0:
        tic_T_i = core_profiles.n_i.value * g.geo_vpr**(5.0 / 3.0)
        tic_T_e = core_profiles.n_e.value * g.geo_vpr**(5.0 / 3.0)
        tic_psi = jnp.ones_like(g.geo_vpr)
        tic_dens_el = g.geo_vpr
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
        # Inlined _calc_coeffs_full
        rho_norm_ped_top_idx = jnp.abs(g.cell_centers -
                                       g.rho_norm_ped_top).argmin()
        mask = (jnp.zeros_like(g.geo_rho,
                               dtype=bool).at[rho_norm_ped_top_idx].set(True))
        conductivity = calculate_conductivity(core_profiles)
        merged_source_profiles = build_source_profiles1(
            core_profiles=core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            conductivity=conductivity,
        )
        source_mat_psi = jnp.zeros_like(g.geo_rho)
        source_psi = merged_source_profiles.total_psi_sources()
        toc_T_i = g.toc_temperature_factor
        tic_T_i = core_profiles.n_i.value * g.geo_vpr**(5.0 / 3.0)
        toc_T_e = g.toc_temperature_factor
        tic_T_e = core_profiles.n_e.value * g.geo_vpr**(5.0 / 3.0)
        toc_psi = (1.0 / g.resistivity_multiplier * g.cell_centers *
                   conductivity.sigma * g.mu0_pi16sq_Phib_sq_over_F_sq)
        tic_psi = jnp.ones_like(toc_psi)
        toc_dens_el = jnp.ones_like(g.geo_vpr)
        tic_dens_el = g.geo_vpr
        turbulent_transport = calculate_transport_coeffs(core_profiles,
                                                         rho_norm_ped_top_idx)
        chi_face_ion_total = turbulent_transport.chi_face_ion
        chi_face_el_total = turbulent_transport.chi_face_el
        d_face_el_total = turbulent_transport.d_face_el
        v_face_el_total = turbulent_transport.v_face_el
        d_face_psi = g.geo_g2g3_over_rhon_face
        v_face_psi = jnp.zeros_like(d_face_psi)
        full_chi_face_ion = (g.geo_g1_over_vpr_face *
                             core_profiles.n_i.face_value() * g.keV_to_J *
                             chi_face_ion_total)
        full_chi_face_el = (g.geo_g1_over_vpr_face *
                            core_profiles.n_e.face_value() * g.keV_to_J *
                            chi_face_el_total)
        full_d_face_el = g.geo_g1_over_vpr_face * d_face_el_total
        full_v_face_el = g.geo_g0_face * v_face_el_total
        source_mat_nn = jnp.zeros_like(g.geo_rho)
        source_n_e = merged_source_profiles.total_sources("n_e")
        source_n_e += mask * g.adaptive_n_source_prefactor * g.n_e_ped
        source_mat_nn += -(mask * g.adaptive_n_source_prefactor)
        # Inlined _calculate_pereverzev_flux
        geo_factor = jnp.concatenate(
            [jnp.ones(1), g.geo_g1_over_vpr_face[1:] / g.geo_g0_face[1:]])
        chi_face_per_ion = (g.geo_g1_over_vpr_face *
                            core_profiles.n_i.face_value() * g.keV_to_J *
                            g.chi_pereverzev)
        chi_face_per_el = (g.geo_g1_over_vpr_face *
                           core_profiles.n_e.face_value() * g.keV_to_J *
                           g.chi_pereverzev)
        d_face_per_el = g.D_pereverzev
        v_face_per_el = (core_profiles.n_e.face_grad() /
                         core_profiles.n_e.face_value() * d_face_per_el *
                         geo_factor)
        chi_face_per_ion = jnp.where(
            g.face_centers > g.rho_norm_ped_top,
            0.0,
            chi_face_per_ion,
        )
        chi_face_per_el = jnp.where(
            g.face_centers > g.rho_norm_ped_top,
            0.0,
            chi_face_per_el,
        )
        v_heat_face_ion = (core_profiles.T_i.face_grad() /
                           core_profiles.T_i.face_value() * chi_face_per_ion)
        v_heat_face_el = (core_profiles.T_e.face_grad() /
                          core_profiles.T_e.face_value() * chi_face_per_el)
        d_face_per_el = jnp.where(
            g.face_centers > g.rho_norm_ped_top,
            0.0,
            d_face_per_el * g.geo_g1_over_vpr_face,
        )
        v_face_per_el = jnp.where(
            g.face_centers > g.rho_norm_ped_top,
            0.0,
            v_face_per_el * g.geo_g0_face,
        )
        chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
        chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])
        full_chi_face_ion += chi_face_per_ion
        full_chi_face_el += chi_face_per_el
        full_d_face_el += d_face_per_el
        full_v_face_el += v_face_per_el
        source_i = merged_source_profiles.total_sources("T_i")
        source_e = merged_source_profiles.total_sources("T_e")
        qei = merged_source_profiles.qei
        source_mat_ii = qei.implicit_ii * g.geo_vpr
        source_mat_ee = qei.implicit_ee * g.geo_vpr
        source_mat_ie = qei.implicit_ie * g.geo_vpr
        source_mat_ei = qei.implicit_ei * g.geo_vpr
        source_i += mask * g.adaptive_T_source_prefactor * g.T_i_ped
        source_e += mask * g.adaptive_T_source_prefactor * g.T_e_ped
        source_mat_ii -= mask * g.adaptive_T_source_prefactor
        source_mat_ee -= mask * g.adaptive_T_source_prefactor
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
g.generic_current_fraction = 0.46
g.generic_current_width = 0.075
g.generic_current_location = 0.36
g.generic_particle_S_total = 2.05e20
g.generic_particle_location = 0.3
g.generic_particle_width = 0.25
g.gas_puff_decay_length = 0.3
g.gas_puff_S_total = 6.0e21
g.pellet_S_total = 0.0e22
g.pellet_width = 0.1
g.pellet_location = 0.85
g.generic_heat_location = 0.12741589640723575
g.generic_heat_width = 0.07280908366127758
g.generic_heat_P_total = 51.0e6
g.generic_heat_electron_fraction = 0.68
g.model = qlknn_model.QLKNNModel.load_default_model()
g.R_major = 6.2
g.a_minor = 2.0
g.B_0 = 5.3
g.tolerance = 1e-7
g.n_corrector_steps = 1
g.Z_eff = 1.6
g.impurity_names = ('Ne', )
g.main_ion_names = 'D', 'T'
# Pre-compute impurity parameters (constant values for 'Ne')
g.impurity_fractions = jnp.array([1.0])
g.impurity_fractions_face = jnp.array([1.0])
g.impurity_A_avg = g.A['Ne']
g.impurity_A_avg_face = g.A['Ne']
# Pre-compute main ion parameters (constant values for D:0.5, T:0.5)
g.main_ion_fractions = jnp.array([0.5, 0.5])
g.main_ion_A_avg = 0.5 * g.A['D'] + 0.5 * g.A['T']
g.t_initial = 0.0
g.n_rho = 25
g.dx = 1 / g.n_rho
g.face_centers = np.linspace(0, g.n_rho * g.dx, g.n_rho + 1)
g.cell_centers = np.linspace(g.dx * 0.5, (g.n_rho - 0.5) * g.dx, g.n_rho)
g.Ip = 10.5e6
g.T_i_right_bc = jnp.array(0.2)
g.T_e_right_bc = jnp.array(0.2)
g.n_e_right_bc = jnp.array(0.25e20)
g.nbar = 0.8
# Pre-compute constant profile values for t=0.0
g.T_i_profile_dict = {0.0: 15.0, 1.0: 0.2}
g.T_e_profile_dict = {0.0: 15.0, 1.0: 0.2}
g.n_e_profile_dict = {0.0: 1.5, 1.0: 1.0}
# Convert to arrays at t=0.0
g.T_i_profile_x = np.array(list(g.T_i_profile_dict.keys()))
g.T_i_profile_y = np.array(list(g.T_i_profile_dict.values()))
g.T_e_profile_x = np.array(list(g.T_e_profile_dict.keys()))
g.T_e_profile_y = np.array(list(g.T_e_profile_dict.values()))
g.n_e_profile_x = np.array(list(g.n_e_profile_dict.keys()))
g.n_e_profile_y = np.array(list(g.n_e_profile_dict.values()))
# Pre-compute profile values (constant, not time-varying)
g.T_i = jnp.interp(g.cell_centers, g.T_i_profile_x, g.T_i_profile_y)
g.T_e = jnp.interp(g.cell_centers, g.T_e_profile_x, g.T_e_profile_y)
g.n_e = jnp.interp(g.cell_centers, g.n_e_profile_x, g.n_e_profile_y)
g.psi = None
g.psidot = None
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
g.ITG_flux_ratio_correction = 1
g.hires_factor = 4
g.Qei_multiplier = 1.0
g.rho_norm_ped_top = 0.9
g.n_e_ped = 0.62e20
g.T_i_ped = 4.5
g.T_e_ped = 4.5
g.rho_norm_ped_top = 0.91
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
g.D_e_max = 100.0
g.V_e_min = -50.0
g.V_e_max = 50.0
g.An_min = 0.05
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
assert not flux_surf_avg_Bp2[-1] < 1e-10
idx_limit = np.argmin(np.abs(rhon - g.rho_smoothing_limit))
flux_surf_avg_Bp2[:] = _smooth_savgol(flux_surf_avg_Bp2, idx_limit, 2)
flux_surf_avg_R2Bp2[:] = _smooth_savgol(flux_surf_avg_R2Bp2, idx_limit, 2)
flux_surf_avg_RBp[:] = _smooth_savgol(flux_surf_avg_RBp, idx_limit, 1)
vpr[:] = _smooth_savgol(vpr, idx_limit, 1)
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
g.geo_Phi = Phi
g.geo_Phi_face = Phi_face
g.geo_R_major = g.R_major
g.geo_a_minor = g.a_minor
g.geo_B_0 = g.B_0
g.geo_volume = volume
g.geo_volume_face = volume_face
g.geo_area = area
g.geo_area_face = area_face
g.geo_vpr = vpr
g.geo_vpr_face = vpr_face
g.geo_spr = spr_cell
g.geo_spr_face = spr_face
g.geo_delta_face = delta_face
g.geo_elongation = elongation
g.geo_elongation_face = elongation_face
g.geo_g0 = g0
g.geo_g0_face = g0_face
g.geo_g1 = g1
g.geo_g1_face = g1_face
g.geo_g2 = g2
g.geo_g2_face = g2_face
g.geo_g3 = g3
g.geo_g3_face = g3_face
g.geo_g2g3_over_rhon = g2g3_over_rhon
g.geo_g2g3_over_rhon_face = g2g3_over_rhon_face
g.geo_g2g3_over_rhon_hires = g2g3_over_rhon_hires
g.geo_gm4 = gm4
g.geo_gm4_face = gm4_face
g.geo_gm5 = gm5
g.geo_gm5_face = gm5_face
g.geo_F = F
g.geo_F_face = F_face
g.geo_F_hires = F_hires
g.geo_R_in = Rin
g.geo_R_in_face = Rin_face
g.geo_R_out = Rout
g.geo_R_out_face = Rout_face
g.geo_Ip_profile_face_base = Ip_profile_face
g.geo_psi = psi
g.geo_psi_from_Ip_base = psi_from_Ip
g.geo_psi_from_Ip_face_base = psi_from_Ip_face
g.geo_delta_upper_face = delta_upper_face
g.geo_delta_lower_face = delta_lower_face
g.geo_spr_hires = spr_hires
g.geo_rho_hires_norm = rho_hires_norm
g.geo_rho_hires = rho_hires
g.geo_q_correction_factor = 1
Phi_b = g.geo_Phi_face[..., -1]
g.geo_Phi_b = Phi_b
g.geo_rho_b = jnp.sqrt(Phi_b / np.pi / g.B_0)
g.geo_rho_face = g.face_centers * jnp.expand_dims(g.geo_rho_b, axis=-1)
g.geo_rho = g.cell_centers * jnp.expand_dims(g.geo_rho_b, axis=-1)
g.geo_epsilon_face = (g.geo_R_out_face - g.geo_R_in_face) / (g.geo_R_out_face +
                                                             g.geo_R_in_face)
bulk = g.geo_g0_face[..., 1:] / g.geo_vpr_face[..., 1:]
first_element = jnp.ones_like(g.geo_rho_b) / g.geo_rho_b
g.geo_g0_over_vpr_face = jnp.concatenate(
    [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1)
bulk = g.geo_g1_face[..., 1:] / g.geo_vpr_face[..., 1:]
first_element = jnp.zeros_like(g.geo_rho_b)
g.geo_g1_over_vpr_face = jnp.concatenate(
    [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1)
bulk = g.geo_g1_face[..., 1:] / g.geo_vpr_face[..., 1:]**2
first_element = jnp.ones_like(g.geo_rho_b) / g.geo_rho_b**2
g.geo_g1_over_vpr2_face = jnp.concatenate(
    [jnp.expand_dims(first_element, axis=-1), bulk], axis=-1)
# Pre-compute frequently used mathematical constants
g.pi_16_squared = 16 * jnp.pi**2
g.pi_16_cubed = 16 * jnp.pi**3
g.toc_temperature_factor = 1.5 * g.geo_vpr**(-2.0 / 3.0) * g.keV_to_J
g.mu0_pi16sq_Phib_sq_over_F_sq = g.mu_0 * g.pi_16_squared * g.geo_Phi_b**2 / g.geo_F**2
g.pi16cubed_mu0_Phib = g.pi_16_cubed * g.mu_0 * g.geo_Phi_b
# Simplified source registry using SourceHandler - replaces entire Source class hierarchy
g.source_registry = {
    "generic_current":
    SourceHandler(
        affects=(AffectedCoreProfile.PSI, ),
        eval_fn=calculate_generic_current,
    ),
    "generic_heat":
    SourceHandler(
        affects=(AffectedCoreProfile.TEMP_ION, AffectedCoreProfile.TEMP_EL),
        eval_fn=default_formula,
    ),
    "generic_particle":
    SourceHandler(
        affects=(AffectedCoreProfile.NE, ),
        eval_fn=calc_generic_particle_source,
    ),
    "pellet":
    SourceHandler(
        affects=(AffectedCoreProfile.NE, ),
        eval_fn=calc_pellet_source,
    ),
    "gas_puff":
    SourceHandler(
        affects=(AffectedCoreProfile.NE, ),
        eval_fn=calc_puff_source,
    ),
    "fusion":
    SourceHandler(
        affects=(AffectedCoreProfile.TEMP_ION, AffectedCoreProfile.TEMP_EL),
        eval_fn=fusion_heat_model_func,
    ),
}
g.psi_source_names = {"generic_current"}
g.ETG_correction_factor = 1.0 / 3.0
g.collisionality_multiplier = 1.0
g.smoothing_width = 0.1
# Pre-compute transport parameters (constant values)
g.transport_rho_min = 0.0
g.transport_rho_max = 1.0
# Pre-compute source modes (constant values)
g.qei_mode = "ZERO"  # ei_exchange mode
Ip_scale_factor = g.Ip / g.geo_Ip_profile_face_base[-1]
T_i = CellVariable(
    value=g.T_i,
    left_face_grad_constraint=jnp.zeros(()),
    right_face_grad_constraint=None,
    right_face_constraint=g.T_i_right_bc,
    dr=jnp.array(g.dx),
)
T_e = CellVariable(
    value=g.T_e,
    left_face_grad_constraint=jnp.zeros(()),
    right_face_grad_constraint=None,
    right_face_constraint=g.T_e_right_bc,
    dr=jnp.array(g.dx),
)
n_e = get_updated_electron_density()
ions = get_updated_ions(n_e, T_e)
v_loop_lcfs = np.array(
    0.0,
    dtype=jnp.float64)  # use_v_loop_lcfs_boundary_condition is always False
psidot = CellVariable(
    value=np.zeros_like(g.geo_rho),
    dr=jnp.array(g.dx),
)
psi = CellVariable(value=np.zeros_like(g.geo_rho), dr=jnp.array(g.dx))
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
    q_face=np.zeros_like(g.geo_rho_face),
    v_loop_lcfs=v_loop_lcfs,
    sigma=np.zeros_like(g.geo_rho),
    sigma_face=np.zeros_like(g.geo_rho_face),
)
source_profiles = SourceProfiles(bootstrap_current=BootstrapCurrent.zeros(),
                                 qei=QeiInfo.zeros())
dpsi_drhonorm_edge = (g.Ip * g.pi16cubed_mu0_Phib /
                      (g.geo_g2g3_over_rhon_face[-1] * g.geo_F_face[-1]))
# Compute scaled psi values using the Ip scale factor
geo_psi_from_Ip_scaled = g.geo_psi_from_Ip_base * Ip_scale_factor
geo_psi_from_Ip_face_scaled = g.geo_psi_from_Ip_face_base * Ip_scale_factor
psi = CellVariable(
    value=geo_psi_from_Ip_scaled,
    right_face_grad_constraint=dpsi_drhonorm_edge,
    right_face_constraint=None,
    dr=jnp.array(g.dx),
)
core_profiles = dataclasses.replace(
    core_profiles,
    psi=psi,
    q_face=jnp.concatenate([
        jnp.expand_dims(
            jnp.abs(
                (2 * g.geo_Phi_b * jnp.array(g.dx)) / psi.face_grad()[1]), 0),
        jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) / psi.face_grad()[1:])
    ]) * g.geo_q_correction_factor,
)
conductivity = calculate_conductivity(core_profiles, )
build_standard_source_profiles(
    core_profiles=core_profiles,
    psi_only=True,
    calculate_anyway=True,
    calculated_source_profiles=source_profiles,
)
result = _calculate_bootstrap_current(
    Z_eff_face=core_profiles.Z_eff_face,
    Z_i_face=core_profiles.Z_i_face,
    n_e=core_profiles.n_e,
    n_i=core_profiles.n_i,
    T_e=core_profiles.T_e,
    T_i=core_profiles.T_i,
    psi=core_profiles.psi,
    q_face=core_profiles.q_face,
)
bootstrap_current = BootstrapCurrent(
    j_bootstrap=result.j_bootstrap,
    j_bootstrap_face=result.j_bootstrap_face,
)
source_profiles = dataclasses.replace(source_profiles,
                                      bootstrap_current=bootstrap_current)
psi_sources = source_profiles.total_psi_sources()
psidot_value = calculate_psidot_from_psi_sources(psi_sources=psi_sources,
                                                 sigma=conductivity.sigma,
                                                 psi=psi)
v_loop_lcfs = psidot_value[-1]
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
    core_profiles=core_profiles, )
initial_core_sources = build_source_profiles1(
    core_profiles=core_profiles,
    explicit_source_profiles=explicit_source_profiles,
    conductivity=conductivity,
)
core_transport = calculate_total_transport_coeffs(initial_core_profiles, )
current_t = np.array(g.t_initial)
current_core_profiles = initial_core_profiles
state_history = [(current_t, current_core_profiles)]
while current_t < (g.t_final - g.tolerance):
    explicit_source_profiles = build_source_profiles0(
        core_profiles=current_core_profiles, )
    # Inlined next_dt
    chi_max = core_transport.chi_max()
    basic_dt = (3.0 / 4.0) * (jnp.array(g.dx)**2) / chi_max
    initial_dt = jnp.minimum(
        g.chi_timestep_prefactor * basic_dt,
        g.max_dt,
    )
    crosses_t_final = (current_t < g.t_final) * (current_t + initial_dt
                                                 > g.t_final)
    # Simplified: logical_and with True is redundant
    initial_dt = jax.lax.select(
        crosses_t_final,
        g.t_final - current_t,
        initial_dt,
    )
    loop_dt = initial_dt
    loop_output = (
        core_profiles_to_solver_x_tuple(current_core_profiles),
        initial_dt,
        1,
        current_core_profiles,
    )

    # Inlined cond_fun
    def should_continue(loop_dt, loop_output):
        solver_outputs = loop_output[2]
        is_nan_next_dt = jnp.isnan(loop_dt)
        solver_did_not_converge = solver_outputs == 1
        at_exact_t_final = jnp.allclose(
            current_t + loop_dt,
            g.t_final,
        )
        next_dt_too_small = loop_dt < g.min_dt
        if solver_did_not_converge:
            if at_exact_t_final:
                take_another_step = True
            else:
                take_another_step = ~next_dt_too_small
        else:
            take_another_step = False
        return take_another_step & ~is_nan_next_dt

    while should_continue(loop_dt, loop_output):
        dt = loop_dt
        output = loop_output
        core_profiles_t = current_core_profiles
        n_e = get_updated_electron_density()
        n_e_right_bc = n_e.right_face_constraint
        ions_edge = get_updated_ions(
            dataclasses.replace(
                core_profiles_t.n_e,
                right_face_constraint=g.n_e_right_bc,
            ),
            dataclasses.replace(
                core_profiles_t.T_e,
                right_face_constraint=g.T_e_right_bc,
            ),
        )
        Z_i_edge = ions_edge.Z_i_face[-1]
        Z_impurity_edge = ions_edge.Z_impurity_face[-1]
        Z_eff_edge = g.Z_eff
        dilution_factor_edge = (Z_impurity_edge -
                                Z_eff_edge) / (Z_i_edge *
                                               (Z_impurity_edge - Z_i_edge))
        n_i_bound_right = n_e_right_bc * dilution_factor_edge
        n_impurity_bound_right = (n_e_right_bc -
                                  n_i_bound_right * Z_i_edge) / Z_impurity_edge
        updated_boundary_conditions = {
            "T_i":
            dict(
                left_face_grad_constraint=jnp.zeros(()),
                right_face_grad_constraint=None,
                right_face_constraint=g.T_i_right_bc,
            ),
            "T_e":
            dict(
                left_face_grad_constraint=jnp.zeros(()),
                right_face_grad_constraint=None,
                right_face_constraint=g.T_e_right_bc,
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
                right_face_grad_constraint=(
                    g.Ip * (16 * jnp.pi**3 * g.mu_0 * g.geo_Phi_b) /
                    (g.geo_g2g3_over_rhon_face[-1] * g.geo_F_face[-1])),
                right_face_constraint=None,
            ),
            "Z_i_edge":
            Z_i_edge,
            "Z_impurity_edge":
            Z_impurity_edge,
        }
        # Inlined get_prescribed_core_profile_values
        T_i_value = core_profiles_t.T_i.value
        T_e_cell_variable = core_profiles_t.T_e
        T_e_value = T_e_cell_variable.value
        n_e_cell_variable = core_profiles_t.n_e
        ions = get_updated_ions(
            n_e_cell_variable,
            T_e_cell_variable,
        )
        n_e_value = n_e_cell_variable.value
        n_i_value = ions.n_i.value
        n_impurity_value = ions.n_impurity.value
        impurity_fractions_value = ions.impurity_fractions
        Z_i_value = ions.Z_i
        Z_i_face_value = ions.Z_i_face
        Z_impurity_value = ions.Z_impurity
        Z_impurity_face_value = ions.Z_impurity_face
        A_i_value = ions.A_i
        A_impurity_value = ions.A_impurity
        A_impurity_face_value = ions.A_impurity_face
        Z_eff_value = ions.Z_eff
        Z_eff_face_value = ions.Z_eff_face
        T_i = dataclasses.replace(
            core_profiles_t.T_i,
            value=T_i_value,
            **updated_boundary_conditions["T_i"],
        )
        T_e = dataclasses.replace(
            core_profiles_t.T_e,
            value=T_e_value,
            **updated_boundary_conditions["T_e"],
        )
        psi = dataclasses.replace(core_profiles_t.psi,
                                  **updated_boundary_conditions["psi"])
        n_e = dataclasses.replace(
            core_profiles_t.n_e,
            value=n_e_value,
            **updated_boundary_conditions["n_e"],
        )
        n_i = dataclasses.replace(
            core_profiles_t.n_i,
            value=n_i_value,
            **updated_boundary_conditions["n_i"],
        )
        n_impurity = dataclasses.replace(
            core_profiles_t.n_impurity,
            value=n_impurity_value,
            **updated_boundary_conditions["n_impurity"],
        )
        Z_i_face = jnp.concatenate([
            Z_i_face_value[:-1],
            jnp.array([updated_boundary_conditions["Z_i_edge"]]),
        ], )
        Z_impurity_face = jnp.concatenate([
            Z_impurity_face_value[:-1],
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
            impurity_fractions=impurity_fractions_value,
            Z_i=Z_i_value,
            Z_i_face=Z_i_face,
            Z_impurity=Z_impurity_value,
            Z_impurity_face=Z_impurity_face,
            A_i=A_i_value,
            A_impurity=A_impurity_value,
            A_impurity_face=A_impurity_face_value,
            Z_eff=Z_eff_value,
            Z_eff_face=Z_eff_face_value,
        )
        x_old = core_profiles_to_solver_x_tuple(current_core_profiles)
        x_new_guess = core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt)
        coeffs_exp = coeffs_callback(
            current_core_profiles,
            x_old,
            explicit_source_profiles=explicit_source_profiles,
            explicit_call=True,
        )

        @jax.jit
        def solver_loop_body(i, x_new_guess):
            coeffs_new = coeffs_callback(
                core_profiles_t_plus_dt,
                x_new_guess,
                explicit_source_profiles=explicit_source_profiles,
            )
            x_old_vec = jnp.concatenate([x.value for x in x_old])
            x_new_guess_vec = jnp.concatenate([x.value for x in x_new_guess])
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
            lhs_vec = -g.theta_implicit * dt * (
                1 / (tc_out_new * tc_in_new)) * c_new
            assert theta_exp <= 0.0
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

        x_new = x_new_guess
        for i in range(0, g.n_corrector_steps + 1):
            x_new = solver_loop_body(i, x_new)
        solver_numeric_outputs = 0
        reduced_dt = dt / g.dt_reduction_factor
        loop_dt = reduced_dt
        loop_output = (
            x_new,
            dt,
            solver_numeric_outputs,
            core_profiles_t_plus_dt,
        )
    result = loop_output
    updated_core_profiles_t_plus_dt = solver_x_tuple_to_core_profiles(
        result[0], result[3])
    ions = get_updated_ions(
        updated_core_profiles_t_plus_dt.n_e,
        updated_core_profiles_t_plus_dt.T_e,
    )
    v_loop_lcfs = ((updated_core_profiles_t_plus_dt.psi.face_value()[-1] -
                    current_core_profiles.psi.face_value()[-1]) / result[1])
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
        psidot=result[3].psidot,
        q_face=jnp.concatenate([
            jnp.expand_dims(
                jnp.abs(
                    (2 * g.geo_Phi_b * jnp.array(g.dx)) /
                    updated_core_profiles_t_plus_dt.psi.face_grad()[1]), 0),
            jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) /
                    updated_core_profiles_t_plus_dt.psi.face_grad()[1:])
        ]) * g.geo_q_correction_factor,
        A_i=ions.A_i,
        A_impurity=ions.A_impurity,
        A_impurity_face=ions.A_impurity_face,
        Z_eff=ions.Z_eff,
        Z_eff_face=ions.Z_eff_face,
        v_loop_lcfs=v_loop_lcfs,
        sigma=result[3].sigma,
        sigma_face=result[3].sigma_face,
    )
    conductivity = calculate_conductivity(intermediate_core_profiles)
    intermediate_core_profiles = dataclasses.replace(
        intermediate_core_profiles,
        sigma=conductivity.sigma,
        sigma_face=conductivity.sigma_face,
    )
    final_source_profiles = build_source_profiles1(
        core_profiles=intermediate_core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        conductivity=conductivity,
    )
    psi_sources = final_source_profiles.total_psi_sources()
    psidot_value = calculate_psidot_from_psi_sources(
        psi_sources=psi_sources,
        sigma=intermediate_core_profiles.sigma,
        psi=intermediate_core_profiles.psi,
    )
    psidot = dataclasses.replace(
        result[3].psidot,
        value=psidot_value,
        right_face_constraint=v_loop_lcfs,
        right_face_grad_constraint=None,
    )
    final_core_profiles = dataclasses.replace(
        intermediate_core_profiles,
        psidot=psidot,
    )
    core_transport = calculate_total_transport_coeffs(final_core_profiles, )
    current_t = current_t + result[1]
    current_core_profiles = final_core_profiles
    state_history.append((current_t, current_core_profiles))
t = np.array([state_t for state_t, _ in state_history])
rho = np.concatenate([[0.0], np.asarray(g.cell_centers), [1.0]])
(nt, ) = np.shape(t)
evolving_data = {}
for var_name in g.evolving_names:
    var_data = []
    for state_t, state_core_profiles in state_history:
        var_cell = getattr(state_core_profiles, var_name)
        data = var_cell.cell_plus_boundaries()
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
