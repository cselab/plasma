@jax.jit
def corrector_body(carry, _):
    state, pred, tc_old = carry
    i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]
    i_face, i_grad, i_grad_r = g.I_i @ i + g.b_i_face, g.D_i @ i + g.b_i_grad, g.D_i_r @ i + g.b_i_grad_r
    e_face, e_grad, e_grad_r = g.I_e @ e + g.b_e_face, g.D_e @ e + g.b_e_grad, g.D_e_r @ e + g.b_e_grad_r
    n_face, n_grad, n_grad_r = g.I_n @ n + g.b_n_face, g.D_n @ n + g.b_n_grad, g.D_n_r @ n + g.b_n_grad_r
    p_grad = g.D_p @ p + g.b_p_grad
    ions = ions_update(n, e, e_face)
    q_face = jnp.concatenate([jnp.abs(g.q_factor_axis / (p_grad[1] * g.inv_dx))[None],
                              jnp.abs(g.q_factor_bulk * g.face_centers[1:] / p_grad[1:])])
    ni_face, ni_grad, ni_grad_r = (g.I_ni @ ions.n_i + g.b_r * ions.n_i_bc[1],
                                   g.D_ni_rho @ ions.n_i + g.b_r_grad * ions.n_i_bc[1],
                                   g.D_ni_rmid @ ions.n_i + g.b_r_grad_r * ions.n_i_bc[1])
    nz_face, nz_grad_r = (g.I_nimp @ ions.n_impurity + g.b_r * ions.n_impurity_bc[1],
                          g.D_nimp_rmid @ ions.n_impurity + g.b_r_grad_r * ions.n_impurity_bc[1])
    sigma = neoclassical_conductivity(e_face, n_face, q_face, ions.Z_eff_face)
    si_fus, se_fus = fusion_source(e, i_face, ni_face)
    j_bs = bootstrap_current(i_face, e_face, n_face, ni_face, p_grad, q_face, 
                             i_grad, e_grad, n_grad, ni_grad, ions.Z_i_face, ions.Z_eff_face)
    Qei_ii, Qei_ee, Qei_ie, Qei_ei = qei_coupling(e, n, ions.n_i, ions.n_impurity, 
                                                   ions.Z_i, ions.Z_impurity, ions.A_i, ions.A_impurity)
    src_p = -(j_bs + g.source_p_external) * g.source_p_coeff
    src_i = g.source_i_external + si_fus * g.geo_vpr + g.source_i_adaptive
    src_e = g.source_e_external + se_fus * g.geo_vpr + g.source_e_adaptive
    tc_in = jnp.concatenate([ions.n_i * g.vpr_5_3, n * g.vpr_5_3, g.ones_vec, g.geo_vpr])
    tc_out = jnp.concatenate([g.toc_temperature_factor, g.toc_temperature_factor,
                              g.c_p_coeff * sigma, g.ones_vpr])
    chi_i, chi_e, D_n, v_n = turbulent_transport(
        i_face, i_grad_r, e_face, e_grad_r, n_face, n_grad, n_grad_r,
        ni_face, ni_grad_r, nz_face, nz_grad_r, p_grad, q_face, ions)
    v_i, v_e, chi_neo_i, chi_neo_e, D_neo_n, v_neo_n = neoclassical_transport(
        i_face, e_face, n_face, ni_face, i_grad, e_grad, n_grad)
    chi_i += chi_neo_i
    chi_e += chi_neo_e
    D_n += D_neo_n
    v_n += v_neo_n
    A_i, b_i = trans_terms(v_i, chi_i, g.bcs[0])
    A_e, b_e = trans_terms(v_e, chi_e, g.bcs[1])
    A_n, b_n = trans_terms(v_n, D_n, g.bcs[3])
    spatial_mat = jnp.block([
        [A_i + jnp.diag(Qei_ii + g.source_mat_adaptive_T), jnp.diag(Qei_ie), 
         g.zero_block, g.zero_block],
        [jnp.diag(Qei_ei), A_e + jnp.diag(Qei_ee + g.source_mat_adaptive_T), 
         g.zero_block, g.zero_block],
        [g.zero_block, g.zero_block, g.A_p, g.zero_block],
        [g.zero_block, g.zero_block, g.zero_block, 
         A_n + jnp.diag(g.source_mat_adaptive_n)]])
    spatial_vec = jnp.concatenate([b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.source_n_constant])
    tc = 1 / (tc_out * tc_in)
    lhs = g.identity - g.dt * g.theta_imp * jnp.expand_dims(tc, 1) * spatial_mat
    rhs = jnp.dot(jnp.diag(jnp.squeeze(tc_old / tc_in)), state) + g.theta_imp * g.dt * tc * spatial_vec
    new_pred = jnp.linalg.solve(lhs, rhs)
    return (state, new_pred, tc_in), None

@jax.jit
def step(carry):
    state, t = carry
    init_carry = (state, state, jnp.ones(4*g.n))
    (_, pred, _), _ = jax.lax.scan(corrector_body, init_carry, None, length=g.n_corr + 1)
    return (pred, t + g.dt), pred

def save_every_10(carry, _):
    s, t = carry
    (s, t), _ = jax.lax.scan(lambda c, _: (step(c), None), (s, t), None, length=10)
    return (s, t), (t, s)

state = jnp.array(np.concatenate([i_initial, e_initial, p_initial, n_initial]))
n_steps = int(g.t_end / g.dt)
n_saves = n_steps // 10
(_, _), history = jax.lax.scan(save_every_10, (state, 0.0), None, length=n_saves)
t_history, state_history = history

