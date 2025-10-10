
def tridiag(
    diag: jt.Shaped[Array, 'size'],
    above: jt.Shaped[Array, 'size-1'],
    below: jt.Shaped[Array, 'size-1'],
):
    return jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)


@jax.jit
def cell_integration(x, geo):
    return jnp.sum(x * geo.drho_norm)


def area_integration(value, geo):
    return cell_integration(value * geo.spr, geo)


def volume_integration(value, geo):
    return cell_integration(value * geo.vpr, geo)


def line_average(value, geo):
    return cell_integration(value, geo)


def volume_average(value, geo):
    return cell_integration(value * geo.vpr, geo) / geo.volume_face[-1]


@chex.dataclass(frozen=True)
class CellVariable:
    value: jt.Float[chex.Array, 't* cell']
    dr: jt.Float[chex.Array, 't*']
    left_face_constraint: jt.Float[chex.Array, 't*'] | None = None
    right_face_constraint: jt.Float[chex.Array, 't*'] | None = None
    left_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
        dataclasses.field(default_factory=_zero))
    right_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
        dataclasses.field(default_factory=_zero))

    def face_grad(self, x: jt.Float[chex.Array, 'cell'] | None = None):
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
        ):
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

    def face_value(self):
        inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
        return jnp.concatenate(
            [self._left_face_value(), inner,
             self._right_face_value()],
            axis=-1)

    def grad(self):
        face = self.face_value()
        return jnp.diff(face) / jnp.expand_dims(self.dr, axis=-1)


def make_convection_terms(v_face,
                          d_face,
                          var,
                          dirichlet_mode='ghost',
                          neumann_mode='ghost'):
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
    mat = tridiag(diag, above, below)
    vec = jnp.zeros_like(diag)
    mat_value = (v_face[0] - right_alpha[0] * v_face[1]) / var.dr
    vec_value = (-v_face[0] * (1.0 - left_alpha[0]) *
                 var.left_face_grad_constraint)
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


def make_diffusion_terms(d_face: FloatVectorFace, var: CellVariable):
    denom = var.dr**2
    diag = jnp.asarray(-d_face[1:] - d_face[:-1])
    off = d_face[1:-1]
    vec = jnp.zeros_like(diag)
    chex.assert_exactly_one_is_none(var.left_face_grad_constraint,
                                    var.left_face_constraint)
    chex.assert_exactly_one_is_none(var.right_face_grad_constraint,
                                    var.right_face_constraint)
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
    mat = tridiag(diag, off, off) / denom
    return mat, vec


def calculate_plh_scaling_factor(
    geo: Geometry,
    core_profiles: CoreProfiles,
):
    line_avg_n_e = line_average(core_profiles.n_e.value, geo)
    P_LH_hi_dens_D = (2.15 * (line_avg_n_e / 1e20)**0.782 * geo.B_0**0.772 *
                      geo.a_minor**0.975 * geo.R_major**0.999 * 1e6)
    A_deuterium = ION_PROPERTIES_DICT['D'].A
    P_LH_hi_dens = P_LH_hi_dens_D * A_deuterium / core_profiles.A_i
    Ip_total = core_profiles.Ip_profile_face[..., -1]
    n_e_min_P_LH = (0.7 * (Ip_total / 1e6)**0.34 * geo.a_minor**-0.95 *
                    geo.B_0**0.62 * (geo.R_major / geo.a_minor)**0.4 * 1e19)
    P_LH_min_D = (0.36 * (Ip_total / 1e6)**0.27 * geo.B_0**1.25 *
                  geo.R_major**1.23 * (geo.R_major / geo.a_minor)**0.08 * 1e6)
    P_LH_min = P_LH_min_D * A_deuterium / core_profiles.A_i
    P_LH = jnp.maximum(P_LH_min, P_LH_hi_dens)
    return P_LH_hi_dens, P_LH_min, P_LH, n_e_min_P_LH


def calculate_scaling_law_confinement_time(
    geo: Geometry,
    core_profiles: CoreProfiles,
    Ploss: jax.Array,
    scaling_law: str,
):
    params = scaling_params[scaling_law]
    scaled_Ip = core_profiles.Ip_profile_face[-1] / 1e6
    scaled_Ploss = Ploss / 1e6
    B = geo.B_0
    line_avg_n_e = (line_average(core_profiles.n_e.value, geo) / 1e19)
    R = geo.R_major
    inverse_aspect_ratio = geo.a_minor / geo.R_major
    elongation = geo.area_face[-1] / (jnp.pi * geo.a_minor**2)
    effective_mass = core_profiles.A_i
    triangularity = geo.delta_face[-1]
    tau_scaling = (
        params['prefactor'] * scaled_Ip**params['Ip_exponent'] *
        B**params['B_exponent'] *
        line_avg_n_e**params['line_avg_n_e_exponent'] *
        scaled_Ploss**params['Ploss_exponent'] * R**params['R_exponent'] *
        inverse_aspect_ratio**params['inverse_aspect_ratio_exponent'] *
        elongation**params['elongation_exponent'] *
        effective_mass**params['effective_mass_exponent'] *
        (1 + triangularity)**params['triangularity_exponent'])
    return tau_scaling


def calc_q_face(
    geo: Geometry,
    psi: CellVariable,
):
    inv_iota = jnp.abs(
        (2 * geo.Phi_b * geo.rho_face_norm[1:]) / psi.face_grad()[1:])
    inv_iota0 = jnp.expand_dims(
        jnp.abs((2 * geo.Phi_b * geo.drho_norm) / psi.face_grad()[1]), 0)
    q_face = jnp.concatenate([inv_iota0, inv_iota])
    return q_face * geo.q_correction_factor


def calc_j_total(
    geo: Geometry,
    psi: CellVariable,
):
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


def calc_s_face(geo: Geometry, psi: CellVariable):
    iota_scaled = jnp.abs((psi.face_grad()[1:] / geo.rho_face_norm[1:]))
    iota_scaled0 = jnp.expand_dims(jnp.abs(psi.face_grad()[1] / geo.drho_norm),
                                   axis=0)
    iota_scaled = jnp.concatenate([iota_scaled0, iota_scaled])
    s_face = (-geo.rho_face_norm *
              jnp.gradient(iota_scaled, geo.rho_face_norm) / iota_scaled)
    return s_face


def calculate_psidot_from_psi_sources(
    *,
    psi_sources: FloatVector,
    sigma: FloatVector,
    psi: CellVariable,
    geo: Geometry,
) -> jax.Array:
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


def calculate_pressure(
    core_profiles: CoreProfiles, ) -> tuple[CellVariable, ...]:
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


def coll_exchange(
    core_profiles: CoreProfiles,
    Qei_multiplier: float,
) -> jax.Array:
    log_lambda_ei = calculate_log_lambda_ei(core_profiles.T_e.value,
                                            core_profiles.n_e.value)
    log_tau_e_Z1 = _calculate_log_tau_e_Z1(
        core_profiles.T_e.value,
        core_profiles.n_e.value,
        log_lambda_ei,
    )
    weighted_Z_eff = _calculate_weighted_Z_eff(core_profiles)
    log_Qei_coef = (jnp.log(Qei_multiplier * 1.5 * core_profiles.n_e.value) +
                    jnp.log(g.keV_to_J / g.m_amu) + jnp.log(2 * g.m_e) +
                    jnp.log(weighted_Z_eff) - log_tau_e_Z1)
    Qei_coef = jnp.exp(log_Qei_coef)
    return Qei_coef


def exponential_profile(
    geo,
    *,
    decay_start,
    width,
    total,
):
    r = geo.rho_norm
    S = jnp.exp(-(decay_start - r) / width)
    C = total / volume_integration(S, geo)
    return C * S


def gaussian_profile(geo, *, center, width, total):
    r = geo.rho_norm
    S = jnp.exp(-((r - center)**2) / (2 * width**2))
    C = total / volume_integration(S, geo)
    return C * S

@jax.jit
def _calculate_conductivity0(
    *,
    Z_eff_face: FloatVectorFace,
    n_e: CellVariable,
    T_e: CellVariable,
    q_face: FloatVectorFace,
    geo: Geometry,
):
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

@jax.jit
def _calculate_bootstrap_current(*, Z_eff_face, Z_i_face, n_e, n_i, T_e, T_i,
                                 psi, q_face, geo):
    f_trap = calculate_f_trap(geo)
    log_lambda_ei = calculate_log_lambda_ei(T_e.face_value(), n_e.face_value())
    log_lambda_ii = calculate_log_lambda_ii(T_i.face_value(), n_i.face_value(),
                                            Z_i_face)
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
