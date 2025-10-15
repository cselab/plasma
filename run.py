from fusion_surrogates.qlknn import qlknn_model
from jax import numpy as jnp
from typing import Any
import dataclasses
import jax
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt


class g:
    pass


class s:
    pass

jax.config.update("jax_enable_x64", True)
g.interp_fn = jax.jit(jnp.interp)
g.TOLERANCE = 1e-6
g.keV_to_J = 1e3 * 1.602176634e-19
g.eV_to_J = 1.602176634e-19
g.m_amu = 1.6605390666e-27
g.q_e = 1.602176634e-19
g.m_e = 9.1093837e-31
g.epsilon_0 = 8.85418782e-12
g.mu_0 = 4 * jnp.pi * 1e-7
g.eps = 1e-7
g.EPS_CONVECTION = 1e-20
g.EPS_PECLET = 1e-3
g.SAVGOL_WINDOW_LENGTH = 5
g.T_E_ALLOWED_RANGE = (0.1, 100.0)
g.sym = "D", "T", "Ne"
g.z = dict(zip(g.sym, [1.0, 1.0, 10.0]))
g.A = dict(zip(g.sym, [2.0141, 3.0160, 20.180]))


def compute_face_grad(value,
                      left_face_constraint,
                      right_face_constraint,
                      left_face_grad_constraint,
                      right_face_grad_constraint,
                      x=None):
    if x is None:
        forward_difference = jnp.diff(value) / g.dx_array
    else:
        forward_difference = jnp.diff(value) / jnp.diff(x)
    if left_face_constraint is None:
        left_grad = left_face_grad_constraint
    else:
        dx = g.dx_array if x is None else x[1] - x[0]
        left_grad = (value[0] - left_face_constraint) / (0.5 * dx)
    if right_face_constraint is None:
        right_grad = right_face_grad_constraint
    else:
        dx = g.dx_array if x is None else x[-1] - x[-2]
        right_grad = -(value[-1] - right_face_constraint) / (0.5 * dx)
    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])


def compute_face_value(value, right_face_constraint, right_face_grad_constraint):
    left_face = value[..., 0:1]
    inner = (value[..., :-1] + value[..., 1:]) / 2.0
    if right_face_constraint is not None:
        right_face = jnp.expand_dims(right_face_constraint, axis=-1)
    else:
        right_face = (value[..., -1:] +
                      jnp.expand_dims(right_face_grad_constraint, axis=-1) *
                      jnp.expand_dims(g.dx_array, axis=-1) / 2)
    return jnp.concatenate([left_face, inner, right_face], axis=-1)


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
    vec_value = -v_face[0] * (1.0 - left_alpha[0]) * bc[2]
    mat = mat.at[0, 0].set(mat_value)
    vec = vec.at[0].set(vec_value)
    if bc[1] is not None:
        mat_value = (v_face[-2] * left_alpha[-1] + v_face[-1] *
                     (1.0 - 2.0 * right_alpha[-1])) / dr
        vec_value = (-2.0 * v_face[-1] * (1.0 - right_alpha[-1]) * bc[1]) / dr
    else:
        mat_value = -(v_face[-1] - v_face[-2] * left_alpha[-1]) / dr
        vec_value = (-v_face[-1] * (1.0 - right_alpha[-1]) * bc[3])
    mat = mat.at[-1, -1].set(mat_value)
    vec = vec.at[-1].set(vec_value)
    return mat, vec


def make_diffusion_terms(d_face, dr, bc):
    denom = dr**2
    diag = jnp.asarray(-d_face[1:] - d_face[:-1])
    off = d_face[1:-1]
    vec = jnp.zeros_like(diag)
    diag = diag.at[0].set(-d_face[1])
    vec = vec.at[0].set(-d_face[0] * bc[2] / dr)
    if bc[1] is not None:
        diag = diag.at[-1].set(-2 * d_face[-1] - d_face[-2])
        vec = vec.at[-1].set(2 * d_face[-1] * bc[1] / denom)
    else:
        diag = diag.at[-1].set(-d_face[-2])
        vec = vec.at[-1].set(d_face[-1] * bc[3] / dr)
    mat = (jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)) / denom
    return mat, vec


def calculate_log_lambda_ei(n_e, T_e_keV):
    return 31.3 - 0.5 * jnp.log(n_e) + jnp.log(T_e_keV * 1e3)


def _calculate_log_tau_e_Z1(T_e, n_e, log_lambda_ei):
    return (jnp.log(12 * jnp.pi**1.5 / (n_e * log_lambda_ei)) -
            4 * jnp.log(g.q_e) + 0.5 * jnp.log(g.m_e / 2.0) +
            2 * jnp.log(g.epsilon_0) + 1.5 * jnp.log(T_e * g.keV_to_J))


g.MAVRIN_Z_COEFFS = {
    "Ne":
    np.array([
        [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
        [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
        [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
    ]),
}
g.TEMPERATURE_INTERVALS = {
    "Ne":
    np.array([0.5, 2.0]),
}


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
    C = total / jnp.sum(S * g.geo_vpr * g.dx_array)
    return C * S


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


g.FLUX_NAME_MAP = {
    "efiITG": "qi_itg",
    "efeITG": "qe_itg",
    "pfeITG": "pfe_itg",
    "efeTEM": "qe_tem",
    "efiTEM": "qi_tem",
    "pfeTEM": "pfe_tem",
    "efeETG": "qe_etg",
}
g.EPSILON_NN = 1 / 3


def calculate_transport_coeffs(T_i, T_e, n_e, psi, n_i, n_i_bc, n_impurity,
                               n_impurity_bc, q_face, A_i, Z_eff_face):
    T_i_face = compute_face_value(T_i, g.T_i_bc[1], g.T_i_bc[3])
    T_i_face_grad_rmid = compute_face_grad(T_i, g.T_i_bc[0], g.T_i_bc[1], g.T_i_bc[2], g.T_i_bc[3], x=g.geo_rmid)
    T_e_face = compute_face_value(T_e, g.T_e_bc[1], g.T_e_bc[3])
    T_e_face_grad_rmid = compute_face_grad(T_e, g.T_e_bc[0], g.T_e_bc[1], g.T_e_bc[2], g.T_e_bc[3], x=g.geo_rmid)
    n_e_face = compute_face_value(n_e, g.n_e_bc[1], g.n_e_bc[3])
    n_e_face_grad_rmid = compute_face_grad(n_e, g.n_e_bc[0], g.n_e_bc[1], g.n_e_bc[2], g.n_e_bc[3], x=g.geo_rmid)
    n_e_face_grad = compute_face_grad(n_e, g.n_e_bc[0], g.n_e_bc[1], g.n_e_bc[2], g.n_e_bc[3])
    psi_face_grad = compute_face_grad(psi, g.psi_bc[0], g.psi_bc[1], g.psi_bc[2], g.psi_bc[3])
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
    n_i_face = compute_face_value(n_i, n_i_bc[1], n_i_bc[3])
    n_i_face_grad = compute_face_grad(n_i, n_i_bc[0], n_i_bc[1], n_i_bc[2], n_i_bc[3], x=rmid)
    lref_over_lni0_result = jnp.where(
        jnp.abs(n_i_face) < g.eps,
        g.eps,
        -g.R_major * n_i_face_grad / n_i_face,
    )
    lref_over_lni0 = jnp.where(
        jnp.abs(lref_over_lni0_result) < g.eps, g.eps, lref_over_lni0_result)
    n_impurity_face = compute_face_value(n_impurity, n_impurity_bc[1], n_impurity_bc[3])
    n_impurity_face_grad = compute_face_grad(n_impurity, n_impurity_bc[0], n_impurity_bc[1], n_impurity_bc[2], n_impurity_bc[3], x=rmid)
    lref_over_lni1_result = jnp.where(
        jnp.abs(n_impurity_face) < g.eps,
        g.eps,
        -g.R_major * n_impurity_face_grad / n_impurity_face,
    )
    lref_over_lni1 = jnp.where(
        jnp.abs(lref_over_lni1_result) < g.eps, g.eps, lref_over_lni1_result)
    q = q_face
    iota_scaled = jnp.abs((psi_face_grad[1:] / g.face_centers[1:]))
    iota_scaled0 = jnp.expand_dims(jnp.abs(psi_face_grad[1] / g.dx_array),
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
    active_mask = ((g.face_centers > g.transport_rho_min)
                   & (g.face_centers <= g.transport_rho_max)
                   & (g.face_centers <= g.rho_norm_ped_top))
    active_mask = jnp.asarray(active_mask).at[0].set(g.transport_rho_min == 0)
    chi_face_ion = jnp.where(active_mask, chi_face_ion, 0.0)
    chi_face_el = jnp.where(active_mask, chi_face_el, 0.0)
    d_face_el = jnp.where(active_mask, d_face_el, 0.0)
    v_face_el = jnp.where(active_mask, v_face_el, 0.0)
    chi_face_ion = jnp.clip(chi_face_ion, g.chi_min, g.chi_max)
    chi_face_el = jnp.clip(chi_face_el, g.chi_min, g.chi_max)
    d_face_el = jnp.clip(d_face_el, g.D_e_min, g.D_e_max)
    v_face_el = jnp.clip(v_face_el, g.V_e_min, g.V_e_max)
    chi_face_ion = jnp.where(g.face_centers < g.rho_inner + g.eps, g.chi_i_inner, chi_face_ion)
    chi_face_el = jnp.where(g.face_centers < g.rho_inner + g.eps, g.chi_e_inner, chi_face_el)
    d_face_el = jnp.where(g.face_centers < g.rho_inner + g.eps, g.D_e_inner, d_face_el)
    v_face_el = jnp.where(g.face_centers < g.rho_inner + g.eps, g.V_e_inner, v_face_el)
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

    chi_face_ion = smooth_single_coeff(chi_face_ion)
    chi_face_el = smooth_single_coeff(chi_face_el)
    d_face_el = smooth_single_coeff(d_face_el)
    v_face_el = smooth_single_coeff(v_face_el)
    return (chi_face_ion, chi_face_el, d_face_el, v_face_el)


g.rho_smoothing_limit = 0.1


def _smooth_savgol(data, idx_limit, polyorder):
    window_length = g.SAVGOL_WINDOW_LENGTH
    smoothed_data = scipy.signal.savgol_filter(data,
                                               window_length,
                                               polyorder,
                                               mode="nearest")
    return np.concatenate(
        [np.array([data[0]]), smoothed_data[1:idx_limit], data[idx_limit:]])


g.scaling_T_i = 1.0
g.scaling_T_e = 1.0
g.scaling_n_e = 1e20
g.scaling_psi = 1.0


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
    n_i_bc: Any
    n_impurity_bc: Any


def get_updated_ions(n_e, T_e):
    T_e_face = compute_face_value(T_e, g.T_e_bc[1], g.T_e_bc[3])
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
    n_i_bc = (None, g.n_e_bc[1] * dilution_factor_edge, 0.0, 0.0)
    n_impurity_value = jnp.where(
        dilution_factor == 1.0,
        0.0,
        (n_e - n_i * Z_i) / Z_impurity,
    )
    n_impurity_right_face_constraint = jnp.where(
        dilution_factor_edge == 1.0,
        0.0,
        (g.n_e_bc[1] - n_i_bc[1] * Z_i_face[-1]) / Z_impurity_face[-1],
    )
    n_impurity = n_impurity_value
    n_impurity_bc = (None, n_impurity_right_face_constraint, 0.0, 0.0)
    n_e_face = compute_face_value(n_e, g.n_e_bc[1], g.n_e_bc[3])
    n_i_face = compute_face_value(n_i, n_i_bc[1], n_i_bc[3])
    n_impurity_face = compute_face_value(n_impurity, n_impurity_bc[1], n_impurity_bc[3])
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
g.n_rho = 25
g.dx = 1 / g.n_rho
g.dx_array = jnp.array(g.dx)
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
g.n_e = jnp.interp(g.cell_centers, g.n_e_profile_x, g.n_e_profile_y)
g.chi_pereverzev = 30
g.D_pereverzev = 15
g.theta_implicit = 1.0
g.theta_explicit = 0.0
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
g.fusion_Efus = 17.6 * 1e3 * g.keV_to_J
g.fusion_mrc2 = 1124656
g.fusion_BG = 34.3827
g.fusion_C1 = 1.17302e-9
g.fusion_C2 = 1.51361e-2
g.fusion_C3 = 7.51886e-2
g.fusion_C4 = 4.60643e-3
g.fusion_C5 = 1.35e-2
g.fusion_C6 = -1.0675e-4
g.fusion_C7 = 1.366e-5
g.fusion_alpha_fraction = 3.5 / 17.6
g.fusion_birth_energy = 3520
g.fusion_alpha_mass = 4.002602
g.fusion_critical_energy_coeff = 10 * 4.002602
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
interp = lambda x, y: np.interp(x, rho_norm_intermediate, y)
vpr_face = interp(rho_face_norm, vpr)
vpr = interp(rho_norm, vpr)
spr_face = interp(rho_face_norm, spr)
spr_cell = interp(rho_norm, spr)
spr_hires = interp(rho_hires_norm, spr)
delta_upper_face = interp(rho_face_norm,
                                           chease_data["delta_upper"])
delta_lower_face = interp(rho_face_norm,
                                           chease_data["delta_bottom"])
delta_face = 0.5 * (delta_upper_face + delta_lower_face)
elongation = interp(rho_norm, chease_data["elongation"])
elongation_face = interp(rho_face_norm,
                                          chease_data["elongation"])
Phi_face = interp(rho_face_norm, Phi)
Phi = interp(rho_norm, Phi)
F_face = interp(rho_face_norm, F_chease)
F = interp(rho_norm, F_chease)
F_hires = interp(rho_hires_norm, F_chease)
psi = interp(rho_norm, psi)
psi_from_Ip_face = interp(rho_face_norm, psi_from_Ip)
psi_from_Ip = interp(rho_norm, psi_from_Ip)
j_total_face = interp(rho_face_norm, j_total)
j_total = interp(rho_norm, j_total)
Ip_profile_face = interp(rho_face_norm, Ip_chease)
Rin_face = interp(rho_face_norm, R_in_chease)
Rin = interp(rho_norm, R_in_chease)
Rout_face = interp(rho_face_norm, R_out_chease)
Rout = interp(rho_norm, R_out_chease)
g0_face = interp(rho_face_norm, g0)
g0 = interp(rho_norm, g0)
g1_face = interp(rho_face_norm, g1)
g1 = interp(rho_norm, g1)
g2_face = interp(rho_face_norm, g2)
g2 = interp(rho_norm, g2)
g3_face = interp(rho_face_norm, g3)
g3 = interp(rho_norm, g3)
g2g3_over_rhon_face = interp(rho_face_norm, g2g3_over_rhon)
g2g3_over_rhon_hires = interp(rho_hires_norm, g2g3_over_rhon)
g2g3_over_rhon = interp(rho_norm, g2g3_over_rhon)
gm4 = interp(rho_norm, flux_surf_avg_1_over_B2)
gm4_face = interp(rho_face_norm, flux_surf_avg_1_over_B2)
gm5 = interp(rho_norm, flux_surf_avg_B2)
gm5_face = interp(rho_face_norm, flux_surf_avg_B2)
volume_face = interp(rho_face_norm, volume_intermediate)
volume = interp(rho_norm, volume_intermediate)
area_face = interp(rho_face_norm, area_intermediate)
area = interp(rho_norm, area_intermediate)
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
g.q_factor_axis = 2 * g.geo_Phi_b * g.geo_q_correction_factor
g.q_factor_bulk = g.q_factor_axis
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
g.source_psi_coeff = 8 * g.geo_vpr * jnp.pi**2 * g.geo_B_0 * g.mu_0 * g.geo_Phi_b / g.geo_F**2
g.vpr_5_3 = g.geo_vpr**(5.0 / 3.0)
g.mu0_pi16sq_Phib_sq_over_F_sq = g.mu_0 * g.pi_16_squared * g.geo_Phi_b**2 / g.geo_F**2
g.pi16cubed_mu0_Phib = g.pi_16_cubed * g.mu_0 * g.geo_Phi_b
g.geo_g1_keV = g.geo_g1_over_vpr_face * g.keV_to_J
g.geo_factor_pereverzev = jnp.concatenate(
    [jnp.ones(1), g.geo_g1_over_vpr_face[1:] / g.geo_g0_face[1:]])
epsilon_effective = (
    0.67 * (1.0 - 1.4 * jnp.abs(g.geo_delta_face) * g.geo_delta_face) *
    g.geo_epsilon_face)
aa = (1.0 - g.geo_epsilon_face) / (1.0 + g.geo_epsilon_face)
g.f_trap = 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
    1.0 + 2.0 * jnp.sqrt(epsilon_effective))


g.ETG_correction_factor = 1.0 / 3.0
g.collisionality_multiplier = 1.0
g.smoothing_width = 0.1
g.transport_rho_min = 0.0
g.transport_rho_max = 1.0
rho_norm_ped_top_idx = jnp.abs(g.cell_centers - g.rho_norm_ped_top).argmin()
g.mask = jnp.zeros_like(g.geo_rho,
                        dtype=bool).at[rho_norm_ped_top_idx].set(True)
g.pedestal_mask_face = g.face_centers > g.rho_norm_ped_top
g.mask_adaptive_T = g.mask * g.adaptive_T_source_prefactor
g.mask_adaptive_n = g.mask * g.adaptive_n_source_prefactor
g.T_i_bc = (None, g.T_i_right_bc, 0.0, 0.0)
g.T_e_bc = (None, g.T_e_right_bc, 0.0, 0.0)
g.n_e_bc = (None, g.n_e_right_bc, 0.0, 0.0)
g.dpsi_drhonorm_edge = (g.Ip * g.pi_16_cubed * g.mu_0 * g.geo_Phi_b /
                        (g.geo_g2g3_over_rhon_face[-1] * g.geo_F_face[-1]))
g.psi_bc = (None, None, 0.0, g.dpsi_drhonorm_edge)
g.T_i_bc_scaled = (None, g.T_i_right_bc / g.scaling_T_i, 0.0, 0.0)
g.T_e_bc_scaled = (None, g.T_e_right_bc / g.scaling_T_e, 0.0, 0.0)
g.psi_bc_scaled = (None, None, 0.0, g.dpsi_drhonorm_edge / g.scaling_psi)
g.n_e_bc_scaled = (None, g.n_e_right_bc / g.scaling_n_e, 0.0, 0.0)


g.num_cells = g.n_rho
g.num_channels = 4
g.zero_block = jnp.zeros((g.num_cells, g.num_cells))
g.zero_vec = jnp.zeros(g.num_cells)
g.ones_vec = jnp.ones(g.num_cells)
g.v_face_psi_zero = jnp.zeros_like(g.geo_g2g3_over_rhon_face)
g.ones_like_vpr = jnp.ones_like(g.geo_vpr)

s.T_i = jnp.interp(g.cell_centers, g.T_i_profile_x, g.T_i_profile_y)
s.T_e = jnp.interp(g.cell_centers, g.T_e_profile_x, g.T_e_profile_y)
nGW = g.Ip / 1e6 / (jnp.pi * g.geo_a_minor**2) * 1e20
n_e_value = g.n_e * nGW
n_e_face = jnp.concatenate([
    n_e_value[0:1],
    (n_e_value[:-1] + n_e_value[1:]) / 2.0,
    g.n_e_right_bc[None],
])
a_minor_out = g.geo_R_out_face[-1] - g.geo_R_out_face[0]
nbar_from_n_e_face_inner = (
    jax.scipy.integrate.trapezoid(n_e_face[:-1], g.geo_R_out_face[:-1]) / a_minor_out)
dr_edge = g.geo_R_out_face[-1] - g.geo_R_out_face[-2]
C = (g.nbar * nGW - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
    nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out)
s.n_e = C * n_e_value
s.psi = g.geo_psi_from_Ip_base * (g.Ip / g.geo_Ip_profile_face_base[-1])
s.t = 0.0
history = [(s.t, s.T_i, s.T_e, s.psi, s.n_e)]
while True:
    psi_face_grad = compute_face_grad(s.psi, g.psi_bc[0], g.psi_bc[1], g.psi_bc[2], g.psi_bc[3])
    current_q_face = jnp.concatenate([
        jnp.expand_dims(jnp.abs(g.q_factor_axis * g.dx_array / psi_face_grad[1]), 0),
        jnp.abs(g.q_factor_bulk * g.face_centers[1:] / psi_face_grad[1:]),
    ])
    ions_for_sources = get_updated_ions(s.n_e, s.T_e)
    core_transport = calculate_transport_coeffs(
        s.T_i,
        s.T_e,
        s.n_e,
        s.psi,
        ions_for_sources.n_i,
        ions_for_sources.n_i_bc,
        ions_for_sources.n_impurity,
        ions_for_sources.n_impurity_bc,
        current_q_face,
        ions_for_sources.A_i,
        ions_for_sources.Z_eff_face,
    )
    chi_max = jnp.maximum(
        jnp.max(core_transport[0] * g.geo_g1_over_vpr2_face),
        jnp.max(core_transport[1] * g.geo_g1_over_vpr2_face),
    )
    basic_dt = (3.0 / 4.0) * (g.dx_array**2) / chi_max
    initial_dt = jnp.minimum(
        g.chi_timestep_prefactor * basic_dt,
        g.max_dt,
    )
    crosses_t_final = (s.t < g.t_final) * (s.t + initial_dt > g.t_final)
    initial_dt = jax.lax.select(
        crosses_t_final,
        g.t_final - s.t,
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
        x_T_i = (s.T_i / g.scaling_T_i, g.dx_array, g.T_i_bc_scaled)
        x_T_e = (s.T_e / g.scaling_T_e, g.dx_array, g.T_e_bc_scaled)
        x_psi = (s.psi / g.scaling_psi, g.dx_array, g.psi_bc_scaled)
        x_n_e = (s.n_e / g.scaling_n_e, g.dx_array, g.n_e_bc_scaled)
        x_initial = (x_T_i, x_T_e, x_psi, x_n_e)
        x_new = x_initial
        tc_in_old = None
        for _ in range(g.n_corrector_steps + 1):
            x_input = x_new
            T_i = x_input[0][0] * g.scaling_T_i
            T_e = x_input[1][0] * g.scaling_T_e
            psi = x_input[2][0] * g.scaling_psi
            n_e = x_input[3][0] * g.scaling_n_e
            ions = get_updated_ions(n_e, T_e)
            psi_face_grad = compute_face_grad(psi, g.psi_bc[0], g.psi_bc[1], g.psi_bc[2], g.psi_bc[3])
            q_face = jnp.concatenate([
                jnp.expand_dims(jnp.abs(g.q_factor_axis * g.dx_array / psi_face_grad[1]), 0),
                jnp.abs(g.q_factor_bulk * g.face_centers[1:] / psi_face_grad[1:]),
            ])
            T_i_face = compute_face_value(T_i, g.T_i_bc[1], g.T_i_bc[3])
            T_i_face_grad = compute_face_grad(T_i, g.T_i_bc[0], g.T_i_bc[1], g.T_i_bc[2], g.T_i_bc[3])
            T_e_face = compute_face_value(T_e, g.T_e_bc[1], g.T_e_bc[3])
            T_e_face_grad = compute_face_grad(T_e, g.T_e_bc[0], g.T_e_bc[1], g.T_e_bc[2], g.T_e_bc[3])
            n_e_face = compute_face_value(n_e, g.n_e_bc[1], g.n_e_bc[3])
            n_e_face_grad = compute_face_grad(n_e, g.n_e_bc[0], g.n_e_bc[1], g.n_e_bc[2], g.n_e_bc[3])
            f_trap = g.f_trap
            NZ = 0.58 + 0.74 / (0.76 + ions.Z_eff_face)
            log_lambda_ei = calculate_log_lambda_ei(n_e_face, T_e_face)
            sigsptz = 1.9012e04 * (T_e_face * 1e3)**1.5 / ions.Z_eff_face / NZ / log_lambda_ei
            nu_e_star_face = calculate_nu_e_star(
                q=q_face,
                n_e=n_e_face,
                T_e=T_e_face,
                Z_eff=ions.Z_eff_face,
                log_lambda_ei=log_lambda_ei,
            )
            ft33 = f_trap / (1.0 +
                             (0.55 - 0.1 * f_trap) * jnp.sqrt(nu_e_star_face) + 0.45 *
                             (1.0 - f_trap) * nu_e_star_face / (ions.Z_eff_face**1.5))
            signeo_face = 1.0 - ft33 * (1.0 + 0.36 / ions.Z_eff_face - ft33 *
                                        (0.59 / ions.Z_eff_face - 0.23 / ions.Z_eff_face * ft33))
            sigma_face = sigsptz * signeo_face
            sigma = 0.5 * (sigma_face[:-1] + sigma_face[1:])
            log_lambda_ei = calculate_log_lambda_ei(n_e, T_e)
            log_tau_e_Z1 = _calculate_log_tau_e_Z1(T_e, n_e, log_lambda_ei)
            weighted_Z_eff = (ions.n_i * ions.Z_i**2 / ions.A_i +
                              ions.n_impurity * ions.Z_impurity**2 / ions.A_impurity) / n_e
            log_Qei_coef = (jnp.log(g.Qei_multiplier * 1.5 * n_e) +
                            jnp.log(g.keV_to_J / g.m_amu) + jnp.log(2 * g.m_e) +
                            jnp.log(weighted_Z_eff) - log_tau_e_Z1)
            qei_coef = jnp.exp(log_Qei_coef)
            qei = (-qei_coef, -qei_coef, qei_coef, qei_coef)
            f_trap = g.f_trap
            n_e_face_bootstrap = compute_face_value(n_e, g.n_e_bc[1], g.n_e_bc[3])
            n_e_face_grad_bootstrap = compute_face_grad(n_e, g.n_e_bc[0], g.n_e_bc[1], g.n_e_bc[2], g.n_e_bc[3])
            T_e_face_bootstrap = compute_face_value(T_e, g.T_e_bc[1], g.T_e_bc[3])
            T_e_face_grad_bootstrap = compute_face_grad(T_e, g.T_e_bc[0], g.T_e_bc[1], g.T_e_bc[2], g.T_e_bc[3])
            T_i_face_bootstrap = compute_face_value(T_i, g.T_i_bc[1], g.T_i_bc[3])
            T_i_face_grad_bootstrap = compute_face_grad(T_i, g.T_i_bc[0], g.T_i_bc[1], g.T_i_bc[2], g.T_i_bc[3])
            psi_face_grad_bootstrap = compute_face_grad(psi, g.psi_bc[0], g.psi_bc[1], g.psi_bc[2], g.psi_bc[3])
            n_i_face_bootstrap = compute_face_value(ions.n_i, ions.n_i_bc[1], ions.n_i_bc[3])
            n_i_face_grad_bootstrap = compute_face_grad(ions.n_i, ions.n_i_bc[0], ions.n_i_bc[1], ions.n_i_bc[2], ions.n_i_bc[3])
            log_lambda_ei_bootstrap = calculate_log_lambda_ei(n_e_face_bootstrap, T_e_face_bootstrap)
            T_i_ev = T_i_face_bootstrap * 1e3
            log_lambda_ii = (30.0 - 0.5 * jnp.log(n_i_face_bootstrap) + 1.5 * jnp.log(T_i_ev) -
                             3.0 * jnp.log(ions.Z_i_face))
            nu_e_star = calculate_nu_e_star(
                q=q_face,
                n_e=n_e_face_bootstrap,
                T_e=T_e_face_bootstrap,
                Z_eff=ions.Z_eff_face,
                log_lambda_ei=log_lambda_ei_bootstrap,
            )
            nu_i_star = calculate_nu_i_star(
                q=q_face,
                n_i=n_i_face_bootstrap,
                T_i=T_i_face_bootstrap,
                Z_eff=ions.Z_eff_face,
                log_lambda_ii=log_lambda_ii,
            )
            L31 = calculate_L31(f_trap, nu_e_star, ions.Z_eff_face)
            L32 = calculate_L32(f_trap, nu_e_star, ions.Z_eff_face)
            L34 = _calculate_L34(f_trap, nu_e_star, ions.Z_eff_face)
            alpha = _calculate_alpha(f_trap, nu_i_star)
            prefactor = -g.geo_F_face * 2 * jnp.pi / g.geo_B_0
            pe = n_e_face_bootstrap * T_e_face_bootstrap * 1e3 * 1.6e-19
            pi = n_i_face_bootstrap * T_i_face_bootstrap * 1e3 * 1.6e-19
            dpsi_drnorm = psi_face_grad_bootstrap
            dlnne_drnorm = n_e_face_grad_bootstrap / n_e_face_bootstrap
            dlnni_drnorm = n_i_face_grad_bootstrap / n_i_face_bootstrap
            dlnte_drnorm = T_e_face_grad_bootstrap / T_e_face_bootstrap
            dlnti_drnorm = T_i_face_grad_bootstrap / T_i_face_bootstrap
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
            qei_implicit_ii = qei[0]
            qei_implicit_ee = qei[1]
            qei_implicit_ie = qei[2]
            qei_implicit_ei = qei[3]
            source_T_i = jnp.zeros_like(T_i)
            source_T_e = jnp.zeros_like(T_e)
            source_psi_current = jnp.zeros_like(psi)
            source_n_e = jnp.zeros_like(n_e)
            I_generic = g.Ip * g.generic_current_fraction
            generic_current_form = jnp.exp(
                -((g.cell_centers - g.generic_current_location)**2) /
                (2 * g.generic_current_width**2))
            Cext = I_generic / jnp.sum(generic_current_form * g.geo_spr * g.dx_array)
            source_psi_current += Cext * generic_current_form
            profile = gaussian_profile(center=g.generic_heat_location,
                                       width=g.generic_heat_width,
                                       total=g.generic_heat_P_total)
            source_T_i += profile * (1 - g.generic_heat_electron_fraction)
            source_T_e += profile * g.generic_heat_electron_fraction
            source_n_e += gaussian_profile(
                center=g.generic_particle_location,
                width=g.generic_particle_width,
                total=g.generic_particle_S_total)
            source_n_e += gaussian_profile(
                center=g.pellet_location,
                width=g.pellet_width,
                total=g.pellet_S_total)
            r = g.cell_centers
            S = jnp.exp(-(1.0 - r) / g.gas_puff_decay_length)
            C = g.gas_puff_S_total / jnp.sum(S * g.geo_vpr * g.dx_array)
            source_n_e += C * S
            product = 1.0
            for fraction, symbol in zip(g.main_ion_fractions, g.main_ion_names):
                if symbol == "D" or symbol == "T":
                    product *= fraction
                    DT_fraction_product = product
                    t_face = compute_face_value(T_i, g.T_i_bc[1], g.T_i_bc[3])
                    theta = t_face / (1.0 - (t_face * (g.fusion_C2 + t_face *
                                                       (g.fusion_C4 + t_face * g.fusion_C6))) /
                                      (1.0 + t_face * (g.fusion_C3 + t_face *
                                                       (g.fusion_C5 + t_face * g.fusion_C7))))
                    xi = (g.fusion_BG**2 / (4 * theta))**(1 / 3)
                    logsigmav = (jnp.log(g.fusion_C1 * theta) +
                                 0.5 * jnp.log(xi / (g.fusion_mrc2 * t_face**3)) - 3 * xi -
                                 jnp.log(1e6))
                    n_i_face_fusion = compute_face_value(ions.n_i, ions.n_i_bc[1], ions.n_i_bc[3])
                    logPfus = (jnp.log(DT_fraction_product * g.fusion_Efus) +
                               2 * jnp.log(n_i_face_fusion) + logsigmav)
                    Pfus_face = jnp.exp(logPfus)
                    Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])
                    critical_energy = g.fusion_critical_energy_coeff * T_e
                    x_squared = g.fusion_birth_energy / critical_energy
                    x = jnp.sqrt(x_squared)
                    frac_i = (2 * ((1 / 6) * jnp.log(
                        (1.0 - x + x_squared) /
                        (1.0 + 2.0 * x + x_squared)) + (jnp.arctan(
                            (2.0 * x - 1.0) / jnp.sqrt(3)) + jnp.pi / 6) / jnp.sqrt(3))
                              / x_squared)
                    frac_e = 1.0 - frac_i
                    T_i_fusion = Pfus_cell * frac_i * g.fusion_alpha_fraction
                    T_e_fusion = Pfus_cell * frac_e * g.fusion_alpha_fraction
            source_T_i += T_i_fusion
            source_T_e += T_e_fusion
            source_psi = -(j_bootstrap + source_psi_current) * g.source_psi_coeff
            tic_T_i = ions.n_i * g.vpr_5_3
            tic_T_e = n_e * g.vpr_5_3
            toc_psi = (1.0 / g.resistivity_multiplier * g.cell_centers *
                       sigma * g.mu0_pi16sq_Phib_sq_over_F_sq)
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
            n_i_face_chi = compute_face_value(ions.n_i, ions.n_i_bc[1], ions.n_i_bc[3])
            full_chi_face_ion = g.geo_g1_keV * n_i_face_chi * turbulent_transport[0]
            full_chi_face_el = g.geo_g1_keV * n_e_face * turbulent_transport[1]
            full_d_face_el = g.geo_g1_over_vpr_face * turbulent_transport[2]
            full_v_face_el = g.geo_g0_face * turbulent_transport[3]
            source_n_e = source_n_e * g.geo_vpr + g.mask_adaptive_n * g.n_e_ped
            source_mat_nn = -g.mask_adaptive_n
            chi_face_per_ion = g.geo_g1_keV * n_i_face_chi * g.chi_pereverzev
            chi_face_per_el = g.geo_g1_keV * n_e_face * g.chi_pereverzev
            d_face_per_el = g.D_pereverzev
            v_face_per_el = n_e_face_grad / n_e_face * d_face_per_el * g.geo_factor_pereverzev
            chi_face_per_ion = jnp.where(g.pedestal_mask_face, 0.0, chi_face_per_ion)
            chi_face_per_el = jnp.where(g.pedestal_mask_face, 0.0, chi_face_per_el)
            v_heat_face_ion = T_i_face_grad / T_i_face * chi_face_per_ion
            v_heat_face_el = T_e_face_grad / T_e_face * chi_face_per_el
            d_face_per_el = jnp.where(g.pedestal_mask_face, 0.0,
                                      d_face_per_el * g.geo_g1_over_vpr_face)
            v_face_per_el = jnp.where(g.pedestal_mask_face, 0.0,
                                      v_face_per_el * g.geo_g0_face)
            chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
            chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])
            full_chi_face_ion += chi_face_per_ion
            full_chi_face_el += chi_face_per_el
            full_d_face_el += d_face_per_el
            full_v_face_el += v_face_per_el
            source_i = source_T_i * g.geo_vpr + g.mask_adaptive_T * g.T_i_ped
            source_e = source_T_e * g.geo_vpr + g.mask_adaptive_T * g.T_e_ped
            source_mat_ii = qei_implicit_ii * g.geo_vpr
            source_mat_ee = qei_implicit_ee * g.geo_vpr
            source_mat_ie = qei_implicit_ie * g.geo_vpr
            source_mat_ei = qei_implicit_ei * g.geo_vpr
            source_mat_ii -= g.mask_adaptive_T
            source_mat_ee -= g.mask_adaptive_T
            transient_out_cell = (g.toc_temperature_factor, g.toc_temperature_factor, toc_psi, g.ones_like_vpr)
            transient_in_cell = (tic_T_i, tic_T_e, g.ones_vec, g.geo_vpr)
            d_face = (full_chi_face_ion, full_chi_face_el, g.geo_g2g3_over_rhon_face, full_d_face_el)
            v_face = (v_heat_face_ion, v_heat_face_el, g.v_face_psi_zero, full_v_face_el)
            source_mat_cell = (
                (source_mat_ii, source_mat_ie, None, None),
                (source_mat_ei, source_mat_ee, None, None),
                (None, None, None, None),
                (None, None, None, source_mat_nn),
            )
            source_cell = (
                source_i / g.scaling_T_i,
                source_e / g.scaling_T_e,
                source_psi / g.scaling_psi,
                source_n_e / g.scaling_n_e,
            )
            if tc_in_old is None:
                tc_in_old = jnp.concatenate(transient_in_cell)
            x_old_vec = jnp.concatenate([x[0] for x in x_initial])
            x_new_guess_vec = jnp.concatenate([x[0] for x in x_input])
            tc_out_new = jnp.concatenate(transient_out_cell)
            tc_in_new = jnp.concatenate(transient_in_cell)
            left_transient = jnp.identity(len(x_new_guess_vec))
            right_transient = jnp.diag(jnp.squeeze(tc_in_old / tc_in_new))
            x = x_input
            zero_row_of_blocks = [g.zero_block] * g.num_channels
            zero_block_vec = [g.zero_vec] * g.num_channels
            c_mat = [zero_row_of_blocks.copy() for _ in range(g.num_channels)]
            c = zero_block_vec.copy()
            for i in range(g.num_channels):
                diffusion_mat, diffusion_vec = make_diffusion_terms(d_face[i], x[i][1], x[i][2])
                c_mat[i][i] += diffusion_mat
                c[i] += diffusion_vec
            for i in range(g.num_channels):
                conv_mat, conv_vec = make_convection_terms(v_face[i], d_face[i], x[i][1], x[i][2])
                c_mat[i][i] += conv_mat
                c[i] += conv_vec
            for i in range(g.num_channels):
                for j in range(g.num_channels):
                    source = source_mat_cell[i][j]
                    if source is not None:
                        c_mat[i][j] += jnp.diag(source)
            c = [(c_i + source_i) for c_i, source_i in zip(c, source_cell)]
            c_mat_new = jnp.block(c_mat)
            c_new = jnp.block(c)
            broadcasted = jnp.expand_dims(1 / (tc_out_new * tc_in_new), 1)
            lhs_mat = left_transient - dt * g.theta_implicit * broadcasted * c_mat_new
            lhs_vec = -g.theta_implicit * dt * (1 / (tc_out_new * tc_in_new)) * c_new
            rhs = jnp.dot(right_transient, x_old_vec) - lhs_vec
            x_new_vec = jnp.linalg.solve(lhs_mat, rhs)
            x_new_split = jnp.split(x_new_vec, g.num_channels)
            x_new = tuple((value, x_input[i][1], x_input[i][2])
                          for i, value in enumerate(x_new_split))
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
            s.t + dt,
            g.t_final,
        )
        next_dt_too_small = dt < g.min_dt
        take_another_step = solver_did_not_converge & (at_exact_t_final
                                                       | ~next_dt_too_small)
        if not (take_another_step & ~is_nan_next_dt):
            break
    result = loop_output
    s.t = s.t + result[1]
    s.T_i = result[0][0][0] * g.scaling_T_i
    s.T_e = result[0][1][0] * g.scaling_T_e
    s.psi = result[0][2][0] * g.scaling_psi
    s.n_e = result[0][3][0] * g.scaling_n_e
    history.append((s.t, s.T_i, s.T_e, s.psi, s.n_e))
    if s.t >= (g.t_final - g.tolerance):
        break


t_history, *var_histories = zip(*history)
var_names = ("T_i", "T_e", "psi", "n_e")
var_bcs = (g.T_i_bc, g.T_e_bc, g.psi_bc, g.n_e_bc)
t = np.array(t_history)
rho = np.concatenate([[0.0], np.asarray(g.cell_centers), [1.0]])
(nt, ) = np.shape(t)
with open("run.raw", "wb") as f:
    t.tofile(f)
    rho.tofile(f)
    for var_name, var_bc, var_history in zip(var_names, var_bcs, var_histories):
        var_data = []
        for var_value in var_history:
            left_value = var_value[..., 0:1]
            if var_bc[1] is not None:
                right_value = jnp.expand_dims(var_bc[1], axis=-1)
            else:
                right_value = (var_value[..., -1:] +
                               jnp.expand_dims(var_bc[3], axis=-1) *
                               jnp.expand_dims(g.dx_array, axis=-1) / 2)
            var_data.append(jnp.concatenate([left_value, var_value, right_value], axis=-1))
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
