from jax import numpy as jnp
import jax


def solve_block_tridiag_2x2(lower_aa, diag_aa, upper_aa,
                             lower_bb, diag_bb, upper_bb,
                             coupling_ab, coupling_ba,
                             rhs_a, rhs_b):
    n = len(diag_aa)
    
    def forward_step(carry, inputs):
        U_00_prev, U_01_prev, U_10_prev, U_11_prev, d_a_prev, d_b_prev = carry
        k, lower_a_km1, lower_b_km1, diag_a_k, diag_b_k, upper_a_k, upper_b_k, coup_ab_k, coup_ba_k, rhs_a_k, rhs_b_k = inputs
        
        has_lower = k > 0
        a00 = jnp.where(has_lower, diag_a_k - lower_a_km1 * U_00_prev, diag_a_k)
        a01 = jnp.where(has_lower, coup_ab_k - lower_a_km1 * U_01_prev, coup_ab_k)
        a10 = jnp.where(has_lower, coup_ba_k - lower_b_km1 * U_10_prev, coup_ba_k)
        a11 = jnp.where(has_lower, diag_b_k - lower_b_km1 * U_11_prev, diag_b_k)
        
        r_a = jnp.where(has_lower, rhs_a_k - lower_a_km1 * d_a_prev, rhs_a_k)
        r_b = jnp.where(has_lower, rhs_b_k - lower_b_km1 * d_b_prev, rhs_b_k)
        
        det = a00 * a11 - a01 * a10
        inv_00, inv_01 = a11 / det, -a01 / det
        inv_10, inv_11 = -a10 / det, a00 / det
        
        has_upper = k < n - 1
        U_00_k = jnp.where(has_upper, inv_00 * upper_a_k, 0.0)
        U_01_k = jnp.where(has_upper, inv_01 * upper_b_k, 0.0)
        U_10_k = jnp.where(has_upper, inv_10 * upper_a_k, 0.0)
        U_11_k = jnp.where(has_upper, inv_11 * upper_b_k, 0.0)
        
        d_a_k = inv_00 * r_a + inv_01 * r_b
        d_b_k = inv_10 * r_a + inv_11 * r_b
        
        return (U_00_k, U_01_k, U_10_k, U_11_k, d_a_k, d_b_k), (U_00_k, U_01_k, U_10_k, U_11_k, d_a_k, d_b_k)
    
    indices = jnp.arange(n)
    lower_aa_padded = jnp.r_[0.0, lower_aa]
    lower_bb_padded = jnp.r_[0.0, lower_bb]
    upper_aa_padded = jnp.r_[upper_aa, 0.0]
    upper_bb_padded = jnp.r_[upper_bb, 0.0]
    
    inputs = (indices, lower_aa_padded, lower_bb_padded, 
              diag_aa, diag_bb, upper_aa_padded, upper_bb_padded,
              coupling_ab, coupling_ba, rhs_a, rhs_b)
    
    init_carry = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, (U_00, U_01, U_10, U_11, d_a, d_b) = jax.lax.scan(forward_step, init_carry, inputs)
    
    def backward_step(carry, inputs):
        sol_a_kp1, sol_b_kp1 = carry
        U_00_k, U_01_k, U_10_k, U_11_k, d_a_k, d_b_k = inputs
        
        sol_a_k = d_a_k - U_00_k * sol_a_kp1 - U_01_k * sol_b_kp1
        sol_b_k = d_b_k - U_10_k * sol_a_kp1 - U_11_k * sol_b_kp1
        
        return (sol_a_k, sol_b_k), (sol_a_k, sol_b_k)
    
    backward_inputs = (U_00[::-1][1:], U_01[::-1][1:], U_10[::-1][1:], U_11[::-1][1:], d_a[::-1][1:], d_b[::-1][1:])
    
    init_carry_back = (d_a[-1], d_b[-1])
    _, (sol_a_rev, sol_b_rev) = jax.lax.scan(backward_step, init_carry_back, backward_inputs)
    
    sol_a = jnp.r_[sol_a_rev[::-1], d_a[-1]]
    sol_b = jnp.r_[sol_b_rev[::-1], d_b[-1]]
    
    return sol_a, sol_b

