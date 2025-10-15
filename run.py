from fusion_surrogates.qlknn import qlknn_model
from jax import numpy as jnp
from typing import Any, Final, Mapping, TypeAlias, TypeVar
import dataclasses
import enum
import immutabledict
import jax
import numpy as np
import os
import scipy
import threading
import typing
import matplotlib.pyplot as plt


class g:
    pass


g.evolving_names = "T_i", "T_e", "psi", "n_e"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") +
    " --xla_backend_extra_options=xla_cpu_flatten_after_fusion")
jax.config.update("jax_enable_x64", True)
T = TypeVar("T")
thread_context = threading.local()
g.interp_fn = jax.jit(jnp.interp)
g.TOLERANCE = 1e-6
g.keV_to_J = 1e3 * 1.602176634e-19
g.eV_to_J = 1.602176634e-19
g.m_amu = 1.6605390666e-27
g.q_e = 1.602176634e-19
g.m_e = 9.1093837e-31
g.epsilon_0 = 8.85418782e-12
g.mu_0 = 4 * jnp.pi * 1e-7
g.k_B = 1.380649e-23
g.eps = 1e-7
g.EPS_CONVECTION = 1e-20
g.EPS_PECLET = 1e-3
g.SAVGOL_WINDOW_LENGTH = 5
g.T_E_ALLOWED_RANGE = (0.1, 100.0)
g.sym = "D", "T", "Ne"
g.z = dict(zip(g.sym, [1.0, 1.0, 10.0]))
g.A = dict(zip(g.sym, [2.0141, 3.0160, 20.180]))


def make_bc(left_face_constraint=None,
            right_face_constraint=None,
            left_face_grad_constraint=None,
            right_face_grad_constraint=None):
    return {
        "left_face_constraint":
        left_face_constraint,
        "right_face_constraint":
        right_face_constraint,
        "left_face_grad_constraint":
        (left_face_grad_constraint
         if left_face_grad_constraint is not None else jnp.zeros(())),
        "right_face_grad_constraint":
        (right_face_grad_constraint
         if right_face_grad_constraint is not None else jnp.zeros(())),
    }


def compute_face_grad(value,
                      dr,
                      left_face_constraint,
                      right_face_constraint,
                      left_face_grad_constraint,
                      right_face_grad_constraint,
                      x=None):
    if x is None:
        forward_difference = jnp.diff(value) / dr
    else:
        forward_difference = jnp.diff(value) / jnp.diff(x)

    def constrained_grad(face, grad, cell, right):
        if face is None:
            return grad
        dx = dr if x is None else (x[-1] - x[-2] if right else x[1] - x[0])
        sign = -1 if right else 1
        return sign * (cell - face) / (0.5 * dx)

    left_grad = constrained_grad(
        left_face_constraint,
        left_face_grad_constraint,
        value[0],
        right=False,
    )
    right_grad = constrained_grad(
        right_face_constraint,
        right_face_grad_constraint,
        value[-1],
        right=True,
    )
    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])


def compute_left_face_value(value):
    return value[..., 0:1]


def compute_right_face_value(value, dr, right_face_constraint,
                             right_face_grad_constraint):
    if right_face_constraint is not None:
        face_value = right_face_constraint
        face_value = jnp.expand_dims(face_value, axis=-1)
    else:
        face_value = (value[..., -1:] +
                      jnp.expand_dims(right_face_grad_constraint, axis=-1) *
                      jnp.expand_dims(dr, axis=-1) / 2)
    return face_value


def compute_face_value(value, dr, right_face_constraint,
                       right_face_grad_constraint):
    left_face = compute_left_face_value(value)
    inner = (value[..., :-1] + value[..., 1:]) / 2.0
    right_face = compute_right_face_value(value, dr, right_face_constraint,
                                          right_face_grad_constraint)
    return jnp.concatenate([left_face, inner, right_face], axis=-1)


def compute_cell_plus_boundaries(value, dr, right_face_constraint,
                                 right_face_grad_constraint):
    left_value = compute_left_face_value(value)
    right_value = compute_right_face_value(value, dr, right_face_constraint,
                                           right_face_grad_constraint)
    return jnp.concatenate([left_value, value, right_value], axis=-1)


def compute_face_grad_bc(value, dr, bc, x=None):
    return compute_face_grad(
        value,
        dr,
        bc["left_face_constraint"],
        bc["right_face_constraint"],
        bc["left_face_grad_constraint"],
        bc["right_face_grad_constraint"],
        x=x,
    )


def compute_face_value_bc(value, dr, bc):
    return compute_face_value(value, dr, bc["right_face_constraint"],
                              bc["right_face_grad_constraint"])


def compute_cell_plus_boundaries_bc(value, dr, bc):
    return compute_cell_plus_boundaries(value, dr, bc["right_face_constraint"],
                                        bc["right_face_grad_constraint"])


def make_convection_terms(v_face, d_face, dr, bc):
    eps = g.EPS_CONVECTION
    is_neg = d_face < 0.0
    nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
    d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))
    half = jnp.array([0.5], dtype=jnp.float64)
    ones = jnp.ones_like(v_face[1:-1])
    scale = jnp.concatenate((half, ones, half))
    ratio = scale * dr * v_face / d_face
    left_peclet = -ratio[:-1]
    right_peclet = ratio[1:]

    def peclet_to_alpha(p):
        eps = g.EPS_PECLET
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
    diag = (left_alpha * left_v - right_alpha * right_v) / dr
    above = -(1.0 - right_alpha) * right_v / dr
    above = above[:-1]
    below = (1.0 - left_alpha) * left_v / dr
    below = below[1:]
    mat = jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)
    vec = jnp.zeros_like(diag)
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / dr
    vec_value = -v_face[0] * (1.0 -
                              left_alpha[0]) * bc["left_face_grad_constraint"]
    mat = mat.at[0, 0].set(mat_value)
    vec = vec.at[0].set(vec_value)
    if bc["right_face_constraint"] is not None:
        mat_value = (v_face[-2] * left_alpha[-1] + v_face[-1] *
                     (1.0 - 2.0 * right_alpha[-1])) / dr
        vec_value = (-2.0 * v_face[-1] * (1.0 - right_alpha[-1]) *
                     bc["right_face_constraint"]) / dr
    else:
        mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / dr
        vec_value = (-v_face[-1] * (1.0 - right_alpha[-1]) *
                     bc["right_face_grad_constraint"])
    mat = mat.at[-1, -1].set(mat_value)
    vec = vec.at[-1].set(vec_value)
    return mat, vec


def make_diffusion_terms(d_face, dr, bc):
    denom = dr**2
    diag = jnp.asarray(-d_face[1:] - d_face[:-1])
    off = d_face[1:-1]
    vec = jnp.zeros_like(diag)
    diag = diag.at[0].set(-d_face[1])
    vec = vec.at[0].set(-d_face[0] * bc["left_face_grad_constraint"] / dr)
    if bc["right_face_constraint"] is not None:
        diag = diag.at[-1].set(-2 * d_face[-1] - d_face[-2])
        vec = vec.at[-1].set(2 * d_face[-1] * bc["right_face_constraint"] /
                             denom)
    else:
        diag = diag.at[-1].set(-d_face[-2])
        vec = vec.at[-1].set(d_face[-1] * bc["right_face_grad_constraint"] /
                             dr)
    mat = (jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)) / denom
    return mat, vec


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CoreProfiles:
    T_i: Any
    T_e: Any
    n_i: Any
    n_i_bc: dict


def calculate_psidot_from_psi_sources(*, psi_sources, sigma, psi, psi_bc):
    toc_psi = (1.0 / g.resistivity_multiplier * g.cell_centers * sigma *
               g.mu_0 * 16 * jnp.pi**2 * g.geo_Phi_b**2 / g.geo_F**2)
    d_face_psi = g.geo_g2g3_over_rhon_face
    v_face_psi = jnp.zeros_like(d_face_psi)
    diffusion_mat, diffusion_vec = make_diffusion_terms(
        d_face_psi, jnp.array(g.dx), psi_bc)
    conv_mat, conv_vec = make_convection_terms(v_face_psi, d_face_psi,
                                               jnp.array(g.dx), psi_bc)
    c_mat = diffusion_mat + conv_mat
    c = diffusion_vec + conv_vec
    c += psi_sources
    psidot = (jnp.dot(c_mat, psi) + c) / toc_psi
    return psidot


def calculate_log_lambda_ei(n_e, T_e_keV):
    return 31.3 - 0.5 * jnp.log(n_e) + jnp.log(T_e_keV * 1e3)


def _calculate_log_tau_e_Z1(T_e, n_e, log_lambda_ei):
    return (jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei)) -
            4 * jnp.log(g.q_e) + 0.5 * jnp.log(g.m_e / 2.0) +
            2 * jnp.log(g.epsilon_0) + 1.5 * jnp.log(T_e * g.keV_to_J))


g.MAVRIN_Z_COEFFS = immutabledict.immutabledict({
    "Ne":
    np.array([
        [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
        [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
    ]),
})
g.TEMPERATURE_INTERVALS = immutabledict.immutabledict({
    "Ne":
    np.array([0.5, 2.0]),
})


def calculate_average_charge_state_single_species(T_e, ion_symbol):
    if ion_symbol not in g.MAVRIN_Z_COEFFS:
        return jnp.ones_like(T_e) * g.z[ion_symbol]
    T_e = jnp.clip(T_e, *g.T_E_ALLOWED_RANGE)
    interval_indices = jnp.searchsorted(g.TEMPERATURE_INTERVALS[ion_symbol],
                                        T_e)
    Zavg_coeffs_in_range = jnp.take(g.MAVRIN_Z_COEFFS[ion_symbol],
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


def calculate_conductivity(n_e, T_e, Z_eff_face, q_face):
    n_e_face = compute_face_value_bc(n_e, jnp.array(g.dx), g.n_e_bc)
    T_e_face = compute_face_value_bc(T_e, jnp.array(g.dx), g.T_e_bc)
    f_trap = calculate_f_trap()
    NZ = 0.58 + 0.74 / (0.76 + Z_eff_face)
    log_lambda_ei = calculate_log_lambda_ei(n_e_face, T_e_face)
    sigsptz = 1.9012e04 * (T_e_face *
                           1e3)**1.5 / Z_eff_face / NZ / log_lambda_ei
    nu_e_star_face = calculate_nu_e_star(
        q=q_face,
        n_e=n_e_face,
        T_e=T_e_face,
        Z_eff=Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    ft33 = f_trap / (1.0 +
                     (0.55 - 0.1 * f_trap) * jnp.sqrt(nu_e_star_face) + 0.45 *
                     (1.0 - f_trap) * nu_e_star_face / (Z_eff_face**1.5))
    signeo_face = 1.0 - ft33 * (1.0 + 0.36 / Z_eff_face - ft33 *
                                (0.59 / Z_eff_face - 0.23 / Z_eff_face * ft33))
    sigma_face = sigsptz * signeo_face
    sigma = 0.5 * (sigma_face[:-1] + sigma_face[1:])
    return (sigma, sigma_face)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True)
class BootstrapCurrent:
    j_bootstrap: Any
    j_bootstrap_face: Any

    @classmethod
    def zeros(cls):
        return cls(
            j_bootstrap=jnp.zeros_like(g.cell_centers),
            j_bootstrap_face=jnp.zeros_like(g.face_centers),
        )


def _calculate_bootstrap_current(*, Z_eff_face, Z_i_face, n_e, n_e_bc, n_i,
                                 n_i_bc, T_e, T_e_bc, T_i, T_i_bc, psi, psi_bc,
                                 q_face):
    f_trap = calculate_f_trap()
    n_e_face = compute_face_value_bc(n_e, jnp.array(g.dx), n_e_bc)
    n_e_face_grad = compute_face_grad_bc(n_e, jnp.array(g.dx), n_e_bc)
    T_e_face = compute_face_value_bc(T_e, jnp.array(g.dx), T_e_bc)
    T_e_face_grad = compute_face_grad_bc(T_e, jnp.array(g.dx), T_e_bc)
    T_i_face = compute_face_value_bc(T_i, jnp.array(g.dx), T_i_bc)
    T_i_face_grad = compute_face_grad_bc(T_i, jnp.array(g.dx), T_i_bc)
    psi_face_grad = compute_face_grad_bc(psi, jnp.array(g.dx), psi_bc)
    n_i_face = compute_face_value_bc(n_i, jnp.array(g.dx), n_i_bc)
    n_i_face_grad = compute_face_grad_bc(n_i, jnp.array(g.dx), n_i_bc)
    f_trap = calculate_f_trap()
    log_lambda_ei = calculate_log_lambda_ei(n_e_face, T_e_face)
    T_i_ev = T_i_face * 1e3
    log_lambda_ii = (30.0 - 0.5 * jnp.log(n_i_face) + 1.5 * jnp.log(T_i_ev) -
                     3.0 * jnp.log(Z_i_face))
    nu_e_star = calculate_nu_e_star(
        q=q_face,
        n_e=n_e_face,
        T_e=T_e_face,
        Z_eff=Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )
    nu_i_star = calculate_nu_i_star(
        q=q_face,
        n_i=n_i_face,
        T_i=T_i_face,
        Z_eff=Z_eff_face,
        log_lambda_ii=log_lambda_ii,
    )
    bootstrap_multiplier = 1.0
    L31 = calculate_L31(f_trap, nu_e_star, Z_eff_face)
    L32 = calculate_L32(f_trap, nu_e_star, Z_eff_face)
    L34 = _calculate_L34(f_trap, nu_e_star, Z_eff_face)
    alpha = _calculate_alpha(f_trap, nu_i_star)
    prefactor = -g.geo_F_face * bootstrap_multiplier * 2 * jnp.pi / g.geo_B_0
    pe = n_e_face * T_e_face * 1e3 * 1.6e-19
    pi = n_i_face * T_i_face * 1e3 * 1.6e-19
    dpsi_drnorm = psi_face_grad
    dlnne_drnorm = n_e_face_grad / n_e_face
    dlnni_drnorm = n_i_face_grad / n_i_face
    dlnte_drnorm = T_e_face_grad / T_e_face
    dlnti_drnorm = T_i_face_grad / T_i_face
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


def gaussian_profile(*, center, width, total):
    r = g.cell_centers
    S = jnp.exp(-((r - center)**2) / (2 * width**2))
    C = total / jnp.sum(S * g.geo_vpr * jnp.array(g.dx))
    return C * S


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QeiInfo:
    implicit_ii: Any
    implicit_ee: Any
    implicit_ie: Any
    implicit_ei: Any

    @classmethod
    def zeros(cls):
        return QeiInfo(
            implicit_ii=jnp.zeros_like(g.geo_rho),
            implicit_ee=jnp.zeros_like(g.geo_rho),
            implicit_ie=jnp.zeros_like(g.geo_rho),
            implicit_ei=jnp.zeros_like(g.geo_rho),
        )


def calculate_total_psi_sources(j_bootstrap, psi_sources_dict):
    total = j_bootstrap
    total += sum(psi_sources_dict.values())
    mu0 = g.mu_0
    prefactor = 8 * g.geo_vpr * jnp.pi**2 * g.geo_B_0 * mu0 * g.geo_Phi_b / g.geo_F**2
    return -total * prefactor


def calculate_total_sources(sources_dict):
    total = sum(sources_dict.values())
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


def calculate_generic_current(unused_state, unused_calculated_source_profiles,
                              unused_conductivity):
    I_generic = g.Ip * g.generic_current_fraction
    generic_current_form = jnp.exp(
        -((g.cell_centers - g.generic_current_location)**2) /
        (2 * g.generic_current_width**2))
    Cext = I_generic / jnp.sum(
        generic_current_form * g.geo_spr * jnp.array(g.dx))
    generic_current_profile = Cext * generic_current_form
    return (generic_current_profile, )


def default_formula(unused_core_profiles, unused_calculated_source_profiles,
                    unused_conductivity):
    absorbed_power = g.generic_heat_P_total * 1.0
    profile = gaussian_profile(center=g.generic_heat_location,
                               width=g.generic_heat_width,
                               total=absorbed_power)
    ion = profile * (1 - g.generic_heat_electron_fraction)
    el = profile * g.generic_heat_electron_fraction
    return (ion, el)


def calc_generic_particle_source(unused_state,
                                 unused_calculated_source_profiles,
                                 unused_conductivity):
    return (gaussian_profile(
        center=g.generic_particle_location,
        width=g.generic_particle_width,
        total=g.generic_particle_S_total,
    ), )


def calc_pellet_source(unused_state, unused_calculated_source_profiles,
                       unused_conductivity):
    return (gaussian_profile(
        center=g.pellet_location,
        width=g.pellet_width,
        total=g.pellet_S_total,
    ), )


def fusion_heat_model_func(core_profiles, unused_calculated_source_profiles,
                           unused_conductivity):
    product = 1.0
    for fraction, symbol in zip(g.main_ion_fractions, g.main_ion_names):
        if symbol == "D" or symbol == "T":
            product *= fraction
            DT_fraction_product = product
            t_face = compute_face_value_bc(core_profiles.T_i, jnp.array(g.dx),
                                           g.T_i_bc)
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
            n_i_face_fusion = compute_face_value_bc(core_profiles.n_i,
                                                    jnp.array(g.dx),
                                                    core_profiles.n_i_bc)
            logPfus = (jnp.log(DT_fraction_product * Efus) +
                       2 * jnp.log(n_i_face_fusion) + logsigmav)
            Pfus_face = jnp.exp(logPfus)
            Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])
            alpha_fraction = 3.5 / 17.6
            birth_energy = 3520
            alpha_mass = 4.002602
            critical_energy = 10 * alpha_mass * core_profiles.T_e
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


def calc_puff_source(unused_state, unused_calculated_source_profiles,
                     unused_conductivity):
    r = g.cell_centers
    S = jnp.exp(-(1.0 - r) / g.gas_puff_decay_length)
    C = g.gas_puff_S_total / jnp.sum(S * g.geo_vpr * jnp.array(g.dx))
    return (C * S, )


def build_source_profiles1(T_i, T_e, n_e, psi, n_i, n_i_bc, n_impurity, Z_i,
                           A_i, Z_impurity, A_impurity, q_face, Z_eff_face,
                           Z_i_face, conductivity):
    log_lambda_ei = calculate_log_lambda_ei(n_e, T_e)
    log_tau_e_Z1 = _calculate_log_tau_e_Z1(
        T_e,
        n_e,
        log_lambda_ei,
    )
    weighted_Z_eff = (n_i * Z_i**2 / A_i +
                      n_impurity * Z_impurity**2 / A_impurity) / n_e
    log_Qei_coef = (jnp.log(g.Qei_multiplier * 1.5 * n_e) +
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
        Z_eff_face=Z_eff_face,
        Z_i_face=Z_i_face,
        n_e=n_e,
        n_e_bc=g.n_e_bc,
        n_i=n_i,
        n_i_bc=n_i_bc,
        T_e=T_e,
        T_e_bc=g.T_e_bc,
        T_i=T_i,
        T_i_bc=g.T_i_bc,
        psi=psi,
        psi_bc=g.psi_bc,
        q_face=q_face,
    )
    bootstrap_current = BootstrapCurrent(
        j_bootstrap=result.j_bootstrap,
        j_bootstrap_face=result.j_bootstrap_face,
    )
    profiles = {
        "bootstrap_current": bootstrap_current,
        "qei": qei,
        "T_e": g.explicit_source_profiles["T_e"],
        "T_i": g.explicit_source_profiles["T_i"],
        "n_e": g.explicit_source_profiles["n_e"],
        "psi": g.explicit_source_profiles["psi"],
    }
    core_profiles_for_sources = CoreProfiles(
        T_i=T_i,
        T_e=T_e,
        n_i=n_i,
        n_i_bc=n_i_bc,
    )
    build_standard_source_profiles(
        calculated_source_profiles=profiles,
        core_profiles=core_profiles_for_sources,
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
        if (not explicit) | calculate_anyway:
            value = handler.eval_fn(
                core_profiles,
                calculated_source_profiles,
                conductivity,
            )
            for profile, affected_core_profile in zip(value,
                                                      handler.affects,
                                                      strict=True):
                match affected_core_profile:
                    case AffectedCoreProfile.PSI:
                        calculated_source_profiles["psi"][
                            source_name] = profile
                    case AffectedCoreProfile.NE:
                        calculated_source_profiles["n_e"][
                            source_name] = profile
                    case AffectedCoreProfile.TEMP_ION:
                        calculated_source_profiles["T_i"][
                            source_name] = profile
                    case AffectedCoreProfile.TEMP_EL:
                        calculated_source_profiles["T_e"][
                            source_name] = profile

    for source_name in g.psi_source_names:
        calculate_source(source_name)
    if psi_only:
        return
    for source_name in g.source_registry.keys():
        if source_name not in g.psi_source_names:
            calculate_source(source_name)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
    chi_face_ion: Any
    chi_face_el: Any
    d_face_el: Any
    v_face_el: Any
    chi_face_el_bohm: Any | None = None
    chi_face_el_gyrobohm: Any | None = None
    chi_face_ion_bohm: Any | None = None
    chi_face_ion_gyrobohm: Any | None = None


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


g.FLUX_NAME_MAP = immutabledict.immutabledict({
    "efiITG": "qi_itg",
    "efeITG": "qe_itg",
    "pfeITG": "pfe_itg",
    "efeTEM": "qe_tem",
    "efiTEM": "qi_tem",
    "pfeTEM": "pfe_tem",
    "efeETG": "qe_etg",
})
g.EPSILON_NN = 1 / 3


def calculate_transport_coeffs(T_i, T_e, n_e, psi, n_i, n_i_bc, n_impurity,
                               n_impurity_bc, q_face, A_i, Z_eff_face):
    T_i_face = compute_face_value_bc(T_i, jnp.array(g.dx), g.T_i_bc)
    T_i_face_grad_rmid = compute_face_grad_bc(T_i,
                                              jnp.array(g.dx),
                                              g.T_i_bc,
                                              x=g.geo_rmid)
    T_e_face = compute_face_value_bc(T_e, jnp.array(g.dx), g.T_e_bc)
    T_e_face_grad_rmid = compute_face_grad_bc(T_e,
                                              jnp.array(g.dx),
                                              g.T_e_bc,
                                              x=g.geo_rmid)
    n_e_face = compute_face_value_bc(n_e, jnp.array(g.dx), g.n_e_bc)
    n_e_face_grad_rmid = compute_face_grad_bc(n_e,
                                              jnp.array(g.dx),
                                              g.n_e_bc,
                                              x=g.geo_rmid)
    n_e_face_grad = compute_face_grad_bc(n_e, jnp.array(g.dx), g.n_e_bc)
    psi_face_grad = compute_face_grad_bc(psi, jnp.array(g.dx), g.psi_bc)
    rmid = g.geo_rmid
    rmid_face = g.geo_rmid_face
    chiGB = ((A_i * g.m_amu)**0.5 / (g.geo_B_0 * g.q_e)**2 *
             (T_i_face * g.keV_to_J)**1.5 / g.geo_a_minor)
    lref_over_lti_result = jnp.where(
        jnp.abs(T_i_face) < g.eps,
        g.eps,
        -g.R_major * T_i_face_grad_rmid / T_i_face,
    )
    lref_over_lti = jnp.where(
        jnp.abs(lref_over_lti_result) < g.eps, g.eps, lref_over_lti_result)
    lref_over_lte_result = jnp.where(
        jnp.abs(T_e_face) < g.eps,
        g.eps,
        -g.R_major * T_e_face_grad_rmid / T_e_face,
    )
    lref_over_lte = jnp.where(
        jnp.abs(lref_over_lte_result) < g.eps, g.eps, lref_over_lte_result)
    lref_over_lne_result = jnp.where(
        jnp.abs(n_e_face) < g.eps,
        g.eps,
        -g.R_major * n_e_face_grad_rmid / n_e_face,
    )
    lref_over_lne = jnp.where(
        jnp.abs(lref_over_lne_result) < g.eps, g.eps, lref_over_lne_result)
    n_i_face = compute_face_value_bc(n_i, jnp.array(g.dx), n_i_bc)
    n_i_face_grad = compute_face_grad_bc(n_i, jnp.array(g.dx), n_i_bc, x=rmid)
    lref_over_lni0_result = jnp.where(
        jnp.abs(n_i_face) < g.eps,
        g.eps,
        -g.R_major * n_i_face_grad / n_i_face,
    )
    lref_over_lni0 = jnp.where(
        jnp.abs(lref_over_lni0_result) < g.eps, g.eps, lref_over_lni0_result)
    n_impurity_face = compute_face_value_bc(n_impurity, jnp.array(g.dx),
                                            n_impurity_bc)
    n_impurity_face_grad = compute_face_grad_bc(n_impurity,
                                                jnp.array(g.dx),
                                                n_impurity_bc,
                                                x=rmid)
    lref_over_lni1_result = jnp.where(
        jnp.abs(n_impurity_face) < g.eps,
        g.eps,
        -g.R_major * n_impurity_face_grad / n_impurity_face,
    )
    lref_over_lni1 = jnp.where(
        jnp.abs(lref_over_lni1_result) < g.eps, g.eps, lref_over_lni1_result)
    q = q_face
    iota_scaled = jnp.abs((psi_face_grad[1:] / g.face_centers[1:]))
    iota_scaled0 = jnp.expand_dims(jnp.abs(psi_face_grad[1] / jnp.array(g.dx)),
                                   axis=0)
    iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
    rmid_face = g.geo_rmid_face
    smag = -rmid_face * jnp.gradient(iota_scaled, rmid_face) / iota_scaled
    epsilon_lcfs = rmid_face[-1] / g.R_major
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < g.eps, g.eps, x)
    Ti_Te = T_i_face / T_e_face
    log_lambda_ei_face = calculate_log_lambda_ei(n_e_face, T_e_face)
    log_tau_e_Z1 = _calculate_log_tau_e_Z1(
        T_e_face,
        n_e_face,
        log_lambda_ei_face,
    )
    nu_e = 1 / jnp.exp(log_tau_e_Z1) * Z_eff_face * g.collisionality_multiplier
    epsilon = g.geo_rho_face / g.R_major
    epsilon = jnp.clip(epsilon, g.eps)
    tau_bounce = (q_face * g.R_major /
                  (epsilon**1.5 * jnp.sqrt(T_e_face * g.keV_to_J / g.m_e)))
    tau_bounce = tau_bounce.at[0].set(tau_bounce[1])
    nu_star = nu_e * tau_bounce
    log_nu_star_face = jnp.log10(nu_star)
    factor_0 = 2 * g.keV_to_J / g.geo_B_0**2 * g.mu_0 * q**2
    alpha = factor_0 * (T_e_face * n_e_face *
                        (lref_over_lte + lref_over_lne) + n_i_face * T_i_face *
                        (lref_over_lti + lref_over_lni0) +
                        n_impurity_face * T_i_face *
                        (lref_over_lti + lref_over_lni1))
    smag = smag - alpha / 2
    smag = jnp.where(q < 1, 0.1, smag)
    q = jnp.where(q < 1, 1, q)
    smag = jnp.where(
        smag - alpha < -0.2,
        alpha - 0.2,
        smag,
    )
    normni = n_i_face / n_e_face
    qualikiz_inputs = QualikizInputs(
        Z_eff_face=Z_eff_face,
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
        x=qualikiz_inputs.x * qualikiz_inputs.epsilon_lcfs / g.EPSILON_NN,
    )
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
        g.FLUX_NAME_MAP.get(flux_name, flux_name): flux_value
        for flux_name, flux_value in model_predictions.items()
    }
    qi_itg_squeezed = model_output["qi_itg"].squeeze()
    qi = qi_itg_squeezed + model_output["qi_tem"].squeeze()
    qe = (model_output["qe_itg"].squeeze() * g.ITG_flux_ratio_correction +
          model_output["qe_tem"].squeeze() +
          model_output["qe_etg"].squeeze() * g.ETG_correction_factor)
    pfe = model_output["pfe_itg"].squeeze() + model_output["pfe_tem"].squeeze()
    gradient_reference_length = g.R_major
    gyrobohm_flux_reference_length = g.geo_a_minor
    pfe_SI = pfe * n_e_face * qualikiz_inputs.chiGB / gyrobohm_flux_reference_length
    chi_face_ion = ((
        (gradient_reference_length / gyrobohm_flux_reference_length) * qi) /
                    qualikiz_inputs.lref_over_lti) * qualikiz_inputs.chiGB
    chi_face_el = ((
        (gradient_reference_length / gyrobohm_flux_reference_length) * qe) /
                   qualikiz_inputs.lref_over_lte) * qualikiz_inputs.chiGB
    Deff = -pfe_SI / (n_e_face_grad * g.geo_g1_over_vpr2_face * g.geo_rho_b +
                      g.eps)
    Veff = pfe_SI / (n_e_face * g.geo_g0_over_vpr_face * g.geo_rho_b)
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
    active_mask = ((g.face_centers > g.transport_rho_min)
                   & (g.face_centers <= g.transport_rho_max)
                   & (g.face_centers <= g.rho_norm_ped_top))
    active_mask = jnp.asarray(active_mask).at[0].set(g.transport_rho_min == 0)
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


g.rho_smoothing_limit = 0.1


def _smooth_savgol(data, idx_limit, polyorder):
    window_length = g.SAVGOL_WINDOW_LENGTH
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
    n_i_bc: dict = dataclasses.field(default_factory=make_bc)
    n_impurity_bc: dict = dataclasses.field(default_factory=make_bc)


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
    return n_e_value


def get_updated_ions(n_e, n_e_bc, T_e, T_e_bc):
    T_e_face = compute_face_value_bc(T_e, jnp.array(g.dx), T_e_bc)
    Z_i_avg, Z_i_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.main_ion_names,
        T_e=T_e,
        fractions=g.main_ion_fractions,
    )
    Z_i = Z_i_Z2_avg / Z_i_avg
    Z_i_face_avg, Z_i_face_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.main_ion_names,
        T_e=T_e_face,
        fractions=g.main_ion_fractions,
    )
    Z_i_face = Z_i_face_Z2_avg / Z_i_face_avg
    Z_impurity_avg, Z_impurity_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.impurity_names,
        T_e=T_e,
        fractions=g.impurity_fractions,
    )
    Z_impurity = Z_impurity_Z2_avg / Z_impurity_avg
    Z_impurity_face_avg, Z_impurity_face_Z2_avg, _ = get_average_charge_state(
        ion_symbols=g.impurity_names,
        T_e=T_e_face,
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
    n_i = n_e * dilution_factor
    n_i_bc = make_bc(
        right_face_grad_constraint=None,
        right_face_constraint=n_e_bc["right_face_constraint"] *
        dilution_factor_edge,
    )
    n_impurity_value = jnp.where(
        dilution_factor == 1.0,
        0.0,
        (n_e - n_i * Z_i) / Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        dilution_factor_edge == 1.0,
        0.0,
        (n_e_bc["right_face_constraint"] -
         n_i_bc["right_face_constraint"] * Z_i_face[-1]) / Z_impurity_face[-1],
    )
    n_impurity = n_impurity_value
    n_impurity_bc = make_bc(
        right_face_grad_constraint=None,
        right_face_constraint=n_impurity_right_face_constraint,
    )
    n_e_face = compute_face_value_bc(n_e, jnp.array(g.dx), n_e_bc)
    n_i_face = compute_face_value_bc(n_i, jnp.array(g.dx), n_i_bc)
    n_impurity_face = compute_face_value_bc(n_impurity, jnp.array(g.dx),
                                            n_impurity_bc)
    Z_eff_face = (Z_i_face**2 * n_i_face +
                  Z_impurity_face**2 * n_impurity_face) / n_e_face
    impurity_fractions_dict = {}
    for i, symbol in enumerate(g.impurity_names):
        fraction = g.impurity_fractions[i]
        impurity_fractions_dict[symbol] = fraction
    return Ions(
        n_i=n_i,
        n_i_bc=n_i_bc,
        n_impurity=n_impurity,
        n_impurity_bc=n_impurity_bc,
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


def evolving_vars_to_solver_x_tuple(T_i, T_e, psi, n_e):
    evolving_dict = {"T_i": T_i, "T_e": T_e, "psi": psi, "n_e": n_e}
    x_tuple_for_solver_list = []
    for name in g.evolving_names:
        original_value = evolving_dict[name]
        original_bc = getattr(g, f"{name}_bc")
        scaling_factor = 1 / SCALING_FACTORS[name]
        operation = lambda x, factor: x * factor if x is not None else None
        scaled_bc = make_bc(
            left_face_constraint=operation(original_bc["left_face_constraint"],
                                           scaling_factor),
            left_face_grad_constraint=operation(
                original_bc["left_face_grad_constraint"], scaling_factor),
            right_face_constraint=operation(
                original_bc["right_face_constraint"], scaling_factor),
            right_face_grad_constraint=operation(
                original_bc["right_face_grad_constraint"], scaling_factor),
        )
        solver_var = (
            operation(original_value, scaling_factor),
            jnp.array(g.dx),
            scaled_bc,
        )
        x_tuple_for_solver_list.append(solver_var)
    return tuple(x_tuple_for_solver_list)


def solver_x_tuple_to_evolving_vars(x_new):
    updated_vars = {}
    for i, var_name in enumerate(g.evolving_names):
        solver_var = x_new[i]
        scaling_factor = SCALING_FACTORS[var_name]
        operation = lambda x, factor: x * factor if x is not None else None
        updated_vars[var_name] = operation(solver_var[0], scaling_factor)
    return updated_vars


OptionalTupleMatrix: TypeAlias = tuple[tuple[Any | None, ...], ...] | None
AuxiliaryOutput: TypeAlias = Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Block1DCoeffs:
    transient_in_cell: tuple[Any, ...]
    transient_out_cell: tuple[Any, ...] | None = None
    d_face: tuple[Any, ...] | None = None
    v_face: tuple[Any, ...] | None = None
    source_mat_cell: OptionalTupleMatrix = None
    source_cell: tuple[Any | None, ...] | None = None
    auxiliary_outputs: AuxiliaryOutput | None = None


g.MIN_DELTA = 1e-7
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
g.impurity_names = ("Ne", )
g.main_ion_names = "D", "T"
g.impurity_fractions = jnp.array([1.0])
g.impurity_fractions_face = jnp.array([1.0])
g.impurity_A_avg = g.A["Ne"]
g.impurity_A_avg_face = g.A["Ne"]
g.main_ion_fractions = jnp.array([0.5, 0.5])
g.main_ion_A_avg = 0.5 * g.A["D"] + 0.5 * g.A["T"]
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
g.T_i_profile_dict = {0.0: 15.0, 1.0: 0.2}
g.T_e_profile_dict = {0.0: 15.0, 1.0: 0.2}
g.n_e_profile_dict = {0.0: 1.5, 1.0: 1.0}
g.T_i_profile_x = np.array(list(g.T_i_profile_dict.keys()))
g.T_i_profile_y = np.array(list(g.T_i_profile_dict.values()))
g.T_e_profile_x = np.array(list(g.T_e_profile_dict.keys()))
g.T_e_profile_y = np.array(list(g.T_e_profile_dict.values()))
g.n_e_profile_x = np.array(list(g.n_e_profile_dict.keys()))
g.n_e_profile_y = np.array(list(g.n_e_profile_dict.values()))
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
int_dl_over_Bp = chease_data[
    "Int(Rdlp/|grad(psi)|)=Int(Jdchi)"] * g.R_major / g.B_0
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
g.geo_rmid = (g.geo_R_out - g.geo_R_in) * 0.5
g.geo_rmid_face = (g.geo_R_out_face - g.geo_R_in_face) * 0.5
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
g.pi_16_squared = 16 * jnp.pi**2
g.pi_16_cubed = 16 * jnp.pi**3
g.toc_temperature_factor = 1.5 * g.geo_vpr**(-2.0 / 3.0) * g.keV_to_J
g.mu0_pi16sq_Phib_sq_over_F_sq = g.mu_0 * g.pi_16_squared * g.geo_Phi_b**2 / g.geo_F**2
g.pi16cubed_mu0_Phib = g.pi_16_cubed * g.mu_0 * g.geo_Phi_b
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
g.transport_rho_min = 0.0
g.transport_rho_max = 1.0
rho_norm_ped_top_idx = jnp.abs(g.cell_centers - g.rho_norm_ped_top).argmin()
g.mask = jnp.zeros_like(g.geo_rho,
                        dtype=bool).at[rho_norm_ped_top_idx].set(True)
g.qei_mode = "ZERO"
g.T_i_bc = make_bc(
    left_face_grad_constraint=jnp.zeros(()),
    right_face_grad_constraint=None,
    right_face_constraint=g.T_i_right_bc,
)
g.T_e_bc = make_bc(
    left_face_grad_constraint=jnp.zeros(()),
    right_face_grad_constraint=None,
    right_face_constraint=g.T_e_right_bc,
)
g.n_e_bc = make_bc(
    right_face_grad_constraint=None,
    right_face_constraint=g.n_e_right_bc,
)
g.dpsi_drhonorm_edge = (g.Ip * 16 * jnp.pi**3 * g.mu_0 * g.geo_Phi_b /
                        (g.geo_g2g3_over_rhon_face[-1] * g.geo_F_face[-1]))
g.psi_bc = make_bc(
    right_face_grad_constraint=g.dpsi_drhonorm_edge,
    right_face_constraint=None,
)
g.explicit_source_profiles = {
    "bootstrap_current": BootstrapCurrent.zeros(),
    "qei": QeiInfo.zeros(),
    "T_e": {},
    "T_i": {},
    "n_e": {},
    "psi": {},
}

Ip_scale_factor = g.Ip / g.geo_Ip_profile_face_base[-1]
T_i = g.T_i
T_e = g.T_e
n_e = get_updated_electron_density()
ions = get_updated_ions(n_e, g.n_e_bc, T_e, g.T_e_bc)
v_loop_lcfs = np.array(0.0, dtype=jnp.float64)
psidot = np.zeros_like(g.geo_rho)
psidot_bc = make_bc()
psi = np.zeros_like(g.geo_rho)
geo_psi_from_Ip_scaled = g.geo_psi_from_Ip_base * Ip_scale_factor
geo_psi_from_Ip_face_scaled = g.geo_psi_from_Ip_face_base * Ip_scale_factor
psi = geo_psi_from_Ip_scaled
psi_face_grad_init = compute_face_grad_bc(psi, jnp.array(g.dx), g.psi_bc)
q_face_init = (jnp.concatenate([
    jnp.expand_dims(
        jnp.abs(
            (2 * g.geo_Phi_b * jnp.array(g.dx)) / psi_face_grad_init[1]), 0),
    jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) / psi_face_grad_init[1:]),
]) * g.geo_q_correction_factor)
source_profiles = {
    "bootstrap_current": BootstrapCurrent.zeros(),
    "qei": QeiInfo.zeros(),
    "T_e": {},
    "T_i": {},
    "n_e": {},
    "psi": {},
}
core_profiles_for_init_sources = CoreProfiles(
    T_i=T_i,
    T_e=T_e,
    n_i=ions.n_i,
    n_i_bc=ions.n_i_bc,
)
build_standard_source_profiles(
    core_profiles=core_profiles_for_init_sources,
    psi_only=True,
    calculate_anyway=True,
    calculated_source_profiles=source_profiles,
)
result = _calculate_bootstrap_current(
    Z_eff_face=ions.Z_eff_face,
    Z_i_face=ions.Z_i_face,
    n_e=n_e,
    n_e_bc=g.n_e_bc,
    n_i=ions.n_i,
    n_i_bc=ions.n_i_bc,
    T_e=T_e,
    T_e_bc=g.T_e_bc,
    T_i=T_i,
    T_i_bc=g.T_i_bc,
    psi=psi,
    psi_bc=g.psi_bc,
    q_face=q_face_init,
)
bootstrap_current = BootstrapCurrent(
    j_bootstrap=result.j_bootstrap,
    j_bootstrap_face=result.j_bootstrap_face,
)
source_profiles["bootstrap_current"] = bootstrap_current
psi_sources = calculate_total_psi_sources(
    source_profiles["bootstrap_current"].j_bootstrap, source_profiles["psi"])
sigma_init, sigma_face_init = calculate_conductivity(n_e, T_e, ions.Z_eff_face,
                                                     q_face_init)
psidot_value = calculate_psidot_from_psi_sources(psi_sources=psi_sources,
                                                 sigma=sigma_init,
                                                 psi=psi,
                                                 psi_bc=g.psi_bc)
v_loop_lcfs = psidot_value[-1]
psidot_bc = make_bc(right_face_constraint=v_loop_lcfs)
psidot = psidot_value
current_T_i = T_i
current_T_e = T_e
current_psi = psi
current_n_e = n_e
current_t = np.array(g.t_initial)
history = [(current_t, current_T_i, current_T_e, current_psi, current_n_e)]
while True:
    # Calculate q_face from current psi
    psi_face_grad = compute_face_grad_bc(current_psi, jnp.array(g.dx),
                                         g.psi_bc)
    current_q_face = (jnp.concatenate([
        jnp.expand_dims(
            jnp.abs(
                (2 * g.geo_Phi_b * jnp.array(g.dx)) / psi_face_grad[1]), 0),
        jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) / psi_face_grad[1:]),
    ]) * g.geo_q_correction_factor)
    # Calculate ions and transport from current state
    ions_for_sources = get_updated_ions(current_n_e, g.n_e_bc, current_T_e,
                                        g.T_e_bc)
    core_transport = calculate_transport_coeffs(
        current_T_i,
        current_T_e,
        current_n_e,
        current_psi,
        ions_for_sources.n_i,
        ions_for_sources.n_i_bc,
        ions_for_sources.n_impurity,
        ions_for_sources.n_impurity_bc,
        current_q_face,
        ions_for_sources.A_i,
        ions_for_sources.Z_eff_face,
    )
    core_profiles_for_sources = CoreProfiles(
        T_i=current_T_i,
        T_e=current_T_e,
        n_i=ions_for_sources.n_i,
        n_i_bc=ions_for_sources.n_i_bc,
    )
    build_standard_source_profiles(
        calculated_source_profiles=g.explicit_source_profiles,
        core_profiles=core_profiles_for_sources,
        explicit=True,
        conductivity=None,
    )
    chi_max = jnp.maximum(
        jnp.max(core_transport.chi_face_ion * g.geo_g1_over_vpr2_face),
        jnp.max(core_transport.chi_face_el * g.geo_g1_over_vpr2_face),
    )
    basic_dt = (3.0 / 4.0) * (jnp.array(g.dx)**2) / chi_max
    initial_dt = jnp.minimum(
        g.chi_timestep_prefactor * basic_dt,
        g.max_dt,
    )
    crosses_t_final = (current_t < g.t_final) * (current_t + initial_dt
                                                 > g.t_final)
    initial_dt = jax.lax.select(
        crosses_t_final,
        g.t_final - current_t,
        initial_dt,
    )
    dt = initial_dt
    while True:
        Z_i_edge = ions_for_sources.Z_i_face[-1]
        Z_impurity_edge = ions_for_sources.Z_impurity_face[-1]
        dilution_factor_edge = (Z_impurity_edge -
                                g.Z_eff) / (Z_i_edge *
                                            (Z_impurity_edge - Z_i_edge))
        n_i_bound_right = g.n_e_right_bc * dilution_factor_edge
        n_impurity_bound_right = (g.n_e_right_bc -
                                  n_i_bound_right * Z_i_edge) / Z_impurity_edge
        x_initial = evolving_vars_to_solver_x_tuple(current_T_i, current_T_e,
                                                    current_psi, current_n_e)
        x_new = x_initial
        tc_in_old = None  # Will be set on first iteration
        for _ in range(0, g.n_corrector_steps + 1):
            x_input = x_new
            # Inlined coeffs_callback body
            evolved = solver_x_tuple_to_evolving_vars(x_input)
            T_i = evolved["T_i"]
            T_e = evolved["T_e"]
            psi = evolved["psi"]
            n_e = evolved["n_e"]
            ions = get_updated_ions(n_e, g.n_e_bc, T_e, g.T_e_bc)
            psi_face_grad = compute_face_grad_bc(psi, jnp.array(g.dx),
                                                 g.psi_bc)
            q_face = (jnp.concatenate([
                jnp.expand_dims(
                    jnp.abs((2 * g.geo_Phi_b * jnp.array(g.dx)) /
                            psi_face_grad[1]), 0),
                jnp.abs((2 * g.geo_Phi_b * g.face_centers[1:]) /
                        psi_face_grad[1:]),
            ]) * g.geo_q_correction_factor)
            T_i_face = compute_face_value_bc(T_i, jnp.array(g.dx), g.T_i_bc)
            T_i_face_grad = compute_face_grad_bc(T_i, jnp.array(g.dx),
                                                 g.T_i_bc)
            T_e_face = compute_face_value_bc(T_e, jnp.array(g.dx), g.T_e_bc)
            T_e_face_grad = compute_face_grad_bc(T_e, jnp.array(g.dx),
                                                 g.T_e_bc)
            n_e_face = compute_face_value_bc(n_e, jnp.array(g.dx), g.n_e_bc)
            n_e_face_grad = compute_face_grad_bc(n_e, jnp.array(g.dx),
                                                 g.n_e_bc)
            sigma, sigma_face = calculate_conductivity(n_e, T_e,
                                                       ions.Z_eff_face, q_face)
            merged_source_profiles = build_source_profiles1(
                T_i,
                T_e,
                n_e,
                psi,
                ions.n_i,
                ions.n_i_bc,
                ions.n_impurity,
                ions.Z_i,
                ions.A_i,
                ions.Z_impurity,
                ions.A_impurity,
                q_face,
                ions.Z_eff_face,
                ions.Z_i_face,
                conductivity=(sigma, sigma_face),
            )
            source_mat_psi = jnp.zeros_like(g.geo_rho)
            source_psi = calculate_total_psi_sources(
                merged_source_profiles["bootstrap_current"].j_bootstrap,
                merged_source_profiles["psi"],
            )
            toc_T_i = g.toc_temperature_factor
            tic_T_i = ions.n_i * g.geo_vpr**(5.0 / 3.0)
            toc_T_e = g.toc_temperature_factor
            tic_T_e = n_e * g.geo_vpr**(5.0 / 3.0)
            toc_psi = (1.0 / g.resistivity_multiplier * g.cell_centers *
                       sigma * g.mu0_pi16sq_Phib_sq_over_F_sq)
            tic_psi = jnp.ones_like(toc_psi)
            toc_dens_el = jnp.ones_like(g.geo_vpr)
            tic_dens_el = g.geo_vpr
            turbulent_transport = calculate_transport_coeffs(
                T_i,
                T_e,
                n_e,
                psi,
                ions.n_i,
                ions.n_i_bc,
                ions.n_impurity,
                ions.n_impurity_bc,
                q_face,
                ions.A_i,
                ions.Z_eff_face,
            )
            chi_face_ion_total = turbulent_transport.chi_face_ion
            chi_face_el_total = turbulent_transport.chi_face_el
            d_face_el_total = turbulent_transport.d_face_el
            v_face_el_total = turbulent_transport.v_face_el
            d_face_psi = g.geo_g2g3_over_rhon_face
            v_face_psi = jnp.zeros_like(d_face_psi)
            n_i_face_chi = compute_face_value_bc(ions.n_i, jnp.array(g.dx),
                                                 ions.n_i_bc)
            full_chi_face_ion = g.geo_g1_over_vpr_face * n_i_face_chi * g.keV_to_J * chi_face_ion_total
            full_chi_face_el = g.geo_g1_over_vpr_face * n_e_face * g.keV_to_J * chi_face_el_total
            full_d_face_el = g.geo_g1_over_vpr_face * d_face_el_total
            full_v_face_el = g.geo_g0_face * v_face_el_total
            source_mat_nn = jnp.zeros_like(g.geo_rho)
            source_n_e = calculate_total_sources(merged_source_profiles["n_e"])
            source_n_e += g.mask * g.adaptive_n_source_prefactor * g.n_e_ped
            source_mat_nn += -(g.mask * g.adaptive_n_source_prefactor)
            geo_factor = jnp.concatenate(
                [jnp.ones(1), g.geo_g1_over_vpr_face[1:] / g.geo_g0_face[1:]])
            chi_face_per_ion = g.geo_g1_over_vpr_face * n_i_face_chi * g.keV_to_J * g.chi_pereverzev
            chi_face_per_el = g.geo_g1_over_vpr_face * n_e_face * g.keV_to_J * g.chi_pereverzev
            d_face_per_el = g.D_pereverzev
            v_face_per_el = n_e_face_grad / n_e_face * d_face_per_el * geo_factor
            chi_face_per_ion = jnp.where(g.face_centers > g.rho_norm_ped_top,
                                         0.0, chi_face_per_ion)
            chi_face_per_el = jnp.where(g.face_centers > g.rho_norm_ped_top,
                                        0.0, chi_face_per_el)
            v_heat_face_ion = T_i_face_grad / T_i_face * chi_face_per_ion
            v_heat_face_el = T_e_face_grad / T_e_face * chi_face_per_el
            d_face_per_el = jnp.where(g.face_centers > g.rho_norm_ped_top, 0.0,
                                      d_face_per_el * g.geo_g1_over_vpr_face)
            v_face_per_el = jnp.where(g.face_centers > g.rho_norm_ped_top, 0.0,
                                      v_face_per_el * g.geo_g0_face)
            chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
            chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])
            full_chi_face_ion += chi_face_per_ion
            full_chi_face_el += chi_face_per_el
            full_d_face_el += d_face_per_el
            full_v_face_el += v_face_per_el
            source_i = calculate_total_sources(merged_source_profiles["T_i"])
            source_e = calculate_total_sources(merged_source_profiles["T_e"])
            qei = merged_source_profiles["qei"]
            source_mat_ii = qei.implicit_ii * g.geo_vpr
            source_mat_ee = qei.implicit_ee * g.geo_vpr
            source_mat_ie = qei.implicit_ie * g.geo_vpr
            source_mat_ei = qei.implicit_ei * g.geo_vpr
            source_i += g.mask * g.adaptive_T_source_prefactor * g.T_i_ped
            source_e += g.mask * g.adaptive_T_source_prefactor * g.T_e_ped
            source_mat_ii -= g.mask * g.adaptive_T_source_prefactor
            source_mat_ee -= g.mask * g.adaptive_T_source_prefactor
            var_to_toc = {
                "T_i": toc_T_i,
                "T_e": toc_T_e,
                "psi": toc_psi,
                "n_e": toc_dens_el
            }
            var_to_tic = {
                "T_i": tic_T_i,
                "T_e": tic_T_e,
                "psi": tic_psi,
                "n_e": tic_dens_el
            }
            transient_out_cell = tuple(var_to_toc[var]
                                       for var in g.evolving_names)
            transient_in_cell = tuple(var_to_tic[var]
                                      for var in g.evolving_names)
            var_to_d_face = {
                "T_i": full_chi_face_ion,
                "T_e": full_chi_face_el,
                "psi": d_face_psi,
                "n_e": full_d_face_el
            }
            d_face = tuple(var_to_d_face[var] for var in g.evolving_names)
            var_to_v_face = {
                "T_i": v_heat_face_ion,
                "T_e": v_heat_face_el,
                "psi": v_face_psi,
                "n_e": full_v_face_el
            }
            v_face = tuple(var_to_v_face.get(var) for var in g.evolving_names)
            d_mat = {
                ("T_i", "T_i"): source_mat_ii,
                ("T_i", "T_e"): source_mat_ie,
                ("T_e", "T_i"): source_mat_ei,
                ("T_e", "T_e"): source_mat_ee,
                ("n_e", "n_e"): source_mat_nn,
                ("psi", "psi"): source_mat_psi,
            }
            source_mat_cell = tuple(
                tuple(
                    d_mat.get((row_block, col_block))
                    for col_block in g.evolving_names)
                for row_block in g.evolving_names)
            var_to_source = {
                "T_i": source_i / SCALING_FACTORS["T_i"],
                "T_e": source_e / SCALING_FACTORS["T_e"],
                "psi": source_psi / SCALING_FACTORS["psi"],
                "n_e": source_n_e / SCALING_FACTORS["n_e"],
            }
            source_cell = tuple(
                var_to_source.get(var) for var in g.evolving_names)
            # End of inlined coeffs_callback
            if tc_in_old is None:
                # First iteration: save transient_in from initial state
                tc_in_old = jnp.concatenate(transient_in_cell)
            x_old_vec = jnp.concatenate([x[0] for x in x_initial])
            x_new_guess_vec = jnp.concatenate([x[0] for x in x_input])
            theta_exp = 1.0 - g.theta_implicit
            tc_out_new = jnp.concatenate(transient_out_cell)
            tc_in_new = jnp.concatenate(transient_in_cell)
            left_transient = jnp.identity(len(x_new_guess_vec))
            right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))
            x = x_input
            num_cells = x[0][0].shape[0]
            num_channels = len(x)
            zero_block = jnp.zeros((num_cells, num_cells))
            zero_row_of_blocks = [zero_block] * num_channels
            zero_vec = jnp.zeros((num_cells))
            zero_block_vec = [zero_vec] * num_channels
            c_mat = [zero_row_of_blocks.copy() for _ in range(num_channels)]
            c = zero_block_vec.copy()
            assert d_face
            for i in range(num_channels):
                (diffusion_mat,
                 diffusion_vec) = make_diffusion_terms(d_face[i], x[i][1],
                                                       x[i][2])
                c_mat[i][i] += diffusion_mat
                c[i] += diffusion_vec
            if v_face is not None:
                for i in range(num_channels):
                    d_face_i = d_face[i] if d_face is not None else None
                    d_face_i = (jnp.zeros_like(v_face[i])
                                if d_face_i is None else d_face_i)
                    (conv_mat,
                     conv_vec) = make_convection_terms(v_face[i], d_face_i,
                                                       x[i][1], x[i][2])
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
            x_new = jnp.split(x_new, len(x_initial))
            out = [(value, var[1], var[2])
                   for var, value in zip(x_input, x_new)]
            x_new = tuple(out)
        solver_numeric_outputs = 0
        loop_output = (
            x_new,
            dt,
            solver_numeric_outputs,
        )
        dt = dt / g.dt_reduction_factor
        solver_outputs = loop_output[2]
        is_nan_next_dt = jnp.isnan(dt)
        solver_did_not_converge = solver_outputs == 1
        at_exact_t_final = jnp.allclose(
            current_t + dt,
            g.t_final,
        )
        next_dt_too_small = dt < g.min_dt
        take_another_step = solver_did_not_converge & (at_exact_t_final
                                                       | ~next_dt_too_small)
        if not (take_another_step & ~is_nan_next_dt):
            break
    result = loop_output
    solved = solver_x_tuple_to_evolving_vars(result[0])
    solved_T_i = solved["T_i"]
    solved_T_e = solved["T_e"]
    solved_psi = solved["psi"]
    solved_n_e = solved["n_e"]
    ions_final = get_updated_ions(solved_n_e, g.n_e_bc, solved_T_e, g.T_e_bc)
    psi_face_new = compute_face_value_bc(solved_psi, jnp.array(g.dx), g.psi_bc)
    psi_face_old = compute_face_value_bc(current_psi, jnp.array(g.dx),
                                         g.psi_bc)
    v_loop_lcfs = (psi_face_new[-1] - psi_face_old[-1]) / result[1]
    psi_face_grad_solved = compute_face_grad_bc(solved_psi, jnp.array(g.dx),
                                                g.psi_bc)
    q_face_solved = (jnp.concatenate([
        jnp.expand_dims(
            jnp.abs(
                (2 * g.geo_Phi_b * jnp.array(g.dx)) / psi_face_grad_solved[1]),
            0,
        ),
        jnp.abs(
            (2 * g.geo_Phi_b * g.face_centers[1:]) / psi_face_grad_solved[1:]),
    ]) * g.geo_q_correction_factor)
    sigma_solved, sigma_face_solved = calculate_conductivity(
        solved_n_e, solved_T_e, ions_final.Z_eff_face, q_face_solved)
    final_source_profiles = build_source_profiles1(
        solved_T_i,
        solved_T_e,
        solved_n_e,
        solved_psi,
        ions_final.n_i,
        ions_final.n_i_bc,
        ions_final.n_impurity,
        ions_final.Z_i,
        ions_final.A_i,
        ions_final.Z_impurity,
        ions_final.A_impurity,
        q_face_solved,
        ions_final.Z_eff_face,
        ions_final.Z_i_face,
        conductivity=(sigma_solved, sigma_face_solved),
    )
    psi_sources = calculate_total_psi_sources(
        final_source_profiles["bootstrap_current"].j_bootstrap,
        final_source_profiles["psi"],
    )
    psidot_value = calculate_psidot_from_psi_sources(
        psi_sources=psi_sources,
        sigma=sigma_solved,
        psi=solved_psi,
        psi_bc=g.psi_bc,
    )
    psidot_bc = make_bc(right_face_constraint=v_loop_lcfs)
    current_t = current_t + result[1]
    current_T_i = solved_T_i
    current_T_e = solved_T_e
    current_psi = solved_psi
    current_n_e = solved_n_e
    history.append((current_t, solved_T_i, solved_T_e, solved_psi, solved_n_e))
    if current_t >= (g.t_final - g.tolerance):
        break
t_history, *var_histories = zip(*history)
t = np.array(t_history)
rho = np.concatenate([[0.0], np.asarray(g.cell_centers), [1.0]])
(nt, ) = np.shape(t)
with open("run.raw", "wb") as f:
    t.tofile(f)
    rho.tofile(f)
    for var_name, var_history in zip(g.evolving_names, var_histories):
        var_bc = getattr(g, f"{var_name}_bc")
        var_data = [
            compute_cell_plus_boundaries_bc(var_value, jnp.array(g.dx), var_bc)
            for var_value in var_history
        ]
        var = np.stack(var_data)
        var.tofile(f)
        lo = np.min(var).item()
        hi = np.max(var).item()
        for j, idx in enumerate([0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]):
            plt.title(f"time: {t[idx]:8.3e}")
            plt.axis([None, None, lo, hi])
            plt.plot(rho, var[idx], "o-")
            plt.savefig(f"{var_name}.{j:04d}.png")
            plt.close()
