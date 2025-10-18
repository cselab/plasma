# Plasma Transport Simulation - Matrix Structure and Linear Algebra

## State Vector

The simulation evolves a **state vector** of size `4n` where `n=25`:

```
state = [i, e, p, n]  (100 elements total)
```

Where:
- `i = state[l.i]` - Ion temperature [keV], cells 0:25
- `e = state[l.e]` - Electron temperature [keV], cells 25:50  
- `p = state[l.p]` - Poloidal flux [Wb], cells 50:75
- `n = state[l.n]` - Electron density [m⁻³], cells 75:100

Slices defined as:
```python
l.i = np.s_[:g.n]
l.e = np.s_[g.n:2*g.n]
l.p = np.s_[2*g.n:3*g.n]
l.n = np.s_[3*g.n:4*g.n]
```

## Differential Operators

Three fundamental operators map cell values to face values/gradients.

### Interpolation Operator `I` (Face values)

Matrix `I: R^n → R^(n+1)` computes face values from cell centers:

```
I @ x + b_f = x_f
```

Structure `(n+1) × n`:
```
[1   0   0  ...  0  ]   [x_0]     [x_0        ]
[0.5 0.5 0  ...  0  ]   [x_1]     [(x_0+x_1)/2]
[0  0.5 0.5 ...  0  ] @ [x_2]  +  [(x_1+x_2)/2]
[...            ... ]   [...]  b  [...]
[0   0   0  ... 1  ]   [x_n]     [x_n + b*bc ]
```

Where `b_f` applies right boundary condition (value or gradient-based).

### Gradient Operator `D` (Normalized coordinates)

Matrix `D: R^n → R^(n+1)` computes gradients w.r.t. normalized rho:

```
D @ x + b_g = ∂x/∂ρ_norm
```

Structure `(n+1) × n` with `inv_dx = n`:
```
[0      0     0  ...  0    ]   [x_0]     [bc_left      ]
[-inv_dx inv_dx 0 ...  0   ]   [x_1]     [(x_1-x_0)*n  ]
[0    -inv_dx inv_dx ... 0 ] @ [x_2]  +  [(x_2-x_1)*n  ]
[...                    ... ]   [...]  b  [...]
[special boundary handling  ]   [x_n]     [bc_right     ]
```

Right boundary:
- If gradient BC: `[-2*inv_dx]` in last row, `b[-1] = 2*inv_dx*bc_grad`
- If value BC: different structure

### Gradient Operator `D_r` (Physical coordinates)

Matrix `D_r: R^n → R^(n+1)` computes gradients w.r.t. midplane radius `r_mid`:

```
D_r @ x + b_r = ∂x/∂r_mid
```

Similar to `D` but with non-uniform `inv_dr[i] = 1/Δr_mid[i]`.

## Transport Operators

For each variable, transport has form:

```
∂/∂t(tc_in * x) = (1/tc_out) * [∇·(d∇x - vx) + S_mat*x + S]
```

### Transport Terms (Tridiagonal Vectors)

Function `transport(v, d, bc)` returns `(lower, diag, upper, b)` where:

```
tridiag(lower, diag, upper) @ x + b = ∇·(d∇x - vx)
```

**Returns 4 vectors of length n:**
- `lower[i]` - subdiagonal coefficient for cell i
- `diag[i]` - diagonal coefficient for cell i  
- `upper[i]` - superdiagonal coefficient for cell i
- `b[i]` - boundary/source contribution for cell i

Combines:
- **Diffusion**: `lower_diff, diag_diff, upper_diff, b_diff = diff_terms(d, bc)`
- **Convection**: `lower_conv, diag_conv, upper_conv, b_conv = conv_terms(v, d, bc)`
- **Sum**: Element-wise addition of vectors

### Diffusion Terms

For `∇·(d∇x)` discretization:

```python
diag[i] = -(d_f[i+1] + d_f[i]) * n²
lower[i] = d_f[i+1] * n²
upper[i] = d_f[i+1] * n²
```

Where `n` is grid size and `d_f` are face diffusion coefficients.

This represents a **symmetric negative definite** tridiagonal operator.

### Convection Terms

For `∇·(vx)` with exponential fitting:

```python
diag[i] = (α_L[i] * v[i] - α_R[i] * v[i+1]) * n
upper[i] = -(1 - α_R[i]) * v[i+1] * n
lower[i] = (1 - α_L[i+1]) * v[i+1] * n
```

Where Peclet number `P[i] = v[i]*dx/d[i]` determines upwind factor `α(P)`.

This is a **non-symmetric** tridiagonal operator with upwind bias.

## Block Structure (Conceptual - Not Assembled)

Conceptually, the implicit system has a **4×4 block structure** where each block is `n×n`:

```
[I - θdt*diag(tc)*A] (100×100) = 
┌────────────────────────────────┐
│ A_ii    A_ie      0        0   │  i (25×25 blocks)
│ A_ei    A_ee      0        0   │  e
│ 0       0       A_pp       0   │  p
│ 0       0        0       A_nn  │  n
└────────────────────────────────┘
```

Where:
- `A_ii` - Ion transport (tridiag) + `diag(qei_ii + g.ped_i)` - self-coupling + pedestal feedback
- `A_ee` - Electron transport (tridiag) + `diag(qei_ee + g.ped_e)` - self-coupling + pedestal feedback
- `A_ie = diag(qei_ie)` - Ion ← electron heat exchange (diagonal)
- `A_ei = diag(qei_ei)` - Electron ← ion heat exchange (diagonal)
- `A_pp` - Psi transport (tridiag, precomputed, constant)
- `A_nn` - Density transport (tridiag) + `diag(g.ped_n)` - pedestal feedback

### Solver Strategy (No Assembly)

**This matrix is NEVER assembled.** Instead:

1. **Coupled i-e block** (2×2 at each spatial point):
   - Solved together using `solve_block_tridiag_2x2`
   - Handles tridiagonal + diagonal coupling simultaneously
   - Takes tridiagonal vectors as input (not matrices)

2. **Decoupled p block**:
   - Solved independently using `solve_implicit_tridiag`
   - Uses precomputed constant tridiagonal vectors

3. **Decoupled n block**:
   - Solved independently using `solve_implicit_tridiag`
   - Builds tridiagonal vectors each iteration

**Structure properties:**
- **Tridiagonal** within each variable's transport operator
- **Diagonal coupling** between i and e (handled by specialized solver)
- **Block sparse**: p and n completely decoupled
- **Asymmetric**: Convection terms break symmetry
- **Never materialized**: Only tridiagonal vectors stored

## Transient Coefficients

Each variable has different time derivative scaling:

```python
tc_in = jnp.r_[j * vpr^(5/3),        # i: heat capacity (j = main ion density)
               n * vpr^(5/3),        # e: heat capacity  
               g.ones_vec,           # p: no scaling
               g.geo_vpr]            # n: volume

tc_out = jnp.r_[g.tc_T,              # i: 1.5 * vpr^(-2/3) * keV_to_J
                g.tc_T,              # e: 1.5 * vpr^(-2/3) * keV_to_J
                g.tc_p_base * sigma, # p: conductivity (dynamic)
                g.ones_vpr]          # n: no scaling
```

Combined transient coefficient:
```python
tc = 1 / (tc_out * tc_in)  # element-wise, size 100
```

## Implicit Time Step - Specialized Solvers

The implicit system is **not** assembled as a full 100×100 matrix. Instead, specialized tridiagonal solvers are used for efficiency.

### Right-Hand Side `rhs`

```python
b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]
rhs = ((tc_prev / tc_in) * state + θdt * tc * b)[:, None]
```

Where:
- **First term**: Ratio of old to new transient coefficients (element-wise)
- **Second term**: Implicit source contribution
- **`b`**: Global RHS vector (transport boundaries + sources)
- Shape: `(100, 1)` for compatibility with solvers

### Coupled Ion-Electron Solver

The ion and electron temperatures are solved **simultaneously** as a coupled 2×2 block tridiagonal system:

```python
pred_i, pred_e = solve_implicit_coupled_2x2(
    lower_Ai, diag_Ai + qei_ii + g.ped_i, upper_Ai,
    lower_Ae, diag_Ae + qei_ee + g.ped_e, upper_Ae,
    qei_ie, qei_ei,
    tc[l.i], tc[l.e], θdt, rhs[l.i, 0], rhs[l.e, 0]
)
```

This solves the system:
```
[I - θdt*tc_i*(A_i + diag(qei_ii + ped_i))    -θdt*tc_i*diag(qei_ie)      ] [i_new]   [rhs_i]
[-θdt*tc_e*diag(qei_ei)                I - θdt*tc_e*(A_e + diag(qei_ee + ped_e))] [e_new] = [rhs_e]
```

The function `solve_block_tridiag_2x2` uses a specialized algorithm for block tridiagonal matrices with 2×2 blocks at each spatial point.

Cost: **O(n)** = O(25) operations (linear in grid size)

### Decoupled Tridiagonal Solves

Poloidal flux and density are solved **independently** as standard tridiagonal systems:

```python
sol_p = solve_implicit_tridiag(g.A_p_lower, g.A_p_diag, g.A_p_upper, 
                                tc[l.p], θdt, rhs[l.p])
sol_n = solve_implicit_tridiag(lower_An, diag_An + g.ped_n, upper_An, 
                                tc[l.n], θdt, rhs[l.n])
```

Each solves:
```
[I - θdt*tc*A] @ x_new = rhs
```

Using JAX's built-in `jax.lax.linalg.tridiagonal_solve`:
```python
def solve_implicit_tridiag(lower_A, diag_A, upper_A, tc, θdt, rhs):
    a = jnp.r_[0.0, -θdt * tc[1:] * lower_A]
    b = 1.0 - θdt * tc * diag_A
    c = jnp.r_[-θdt * tc[:-1] * upper_A, 0.0]
    return jax.lax.linalg.tridiagonal_solve(a, b, c, rhs)[:, 0]
```

This uses the **Thomas algorithm** (forward elimination + backward substitution).

Cost: **O(n)** = O(25) operations per solve

### Total Solve Cost

**Per time step**: 3 × O(n) = O(75) operations
- 1× block tridiagonal 2×2 solve (i, e coupled)
- 2× standard tridiagonal solves (p, n decoupled)

This is vastly more efficient than the **O(n³)** = O(1M) cost of dense LU decomposition on the full 100×100 system.

## Operator Precomputation

### Static Operators (Geometry-dependent)

Computed once during initialization:

```python
# Interpolation operators (26×25)
g.I_i, g.b_i_f = face_op(g.bc_i)
g.I_e, g.b_e_f = face_op(g.bc_e)
g.I_n, g.b_n_f = face_op(g.bc_n)
g.I_p, g.b_p_f = face_op(g.bc_p)

# Gradient operators (26×25)  
g.D_i, g.b_i_g = grad_op(g.bc_i)
g.D_e, g.b_e_g = grad_op(g.bc_e)
g.D_n, g.b_n_g = grad_op(g.bc_n)
g.D_p, g.b_p_g = grad_op(g.bc_p)

# Physical coordinate gradients (26×25)
g.D_i_r, g.b_i_r = grad_op_nu(inv_drmid, g.bc_i)
g.D_e_r, g.b_e_r = grad_op_nu(inv_drmid, g.bc_e)
g.D_n_r, g.b_n_r = grad_op_nu(inv_drmid, g.bc_n)

# Ion operators
g.I_j, _ = face_op(1.0, 0.0)
g.D_j, _ = grad_op(dummy_bc)
g.D_j_r, _ = grad_op_nu(inv_drmid, dummy_bc)
g.I_z, _ = face_op(1.0, 0.0)
g.D_z_r, _ = grad_op_nu(inv_drmid, dummy_bc)

# Psi transport (constant diffusion, stored as tridiagonal vectors)
A_p_lower, A_p_diag, A_p_upper, b_p = transport(g.v_p_zero, g.geo_g2g3_over_rhon_face, g.bc_p)
g.A_p_lower = np.asarray(A_p_lower)
g.A_p_diag = np.asarray(A_p_diag)
g.A_p_upper = np.asarray(A_p_upper)
g.b_p = np.asarray(b_p)
```

### Per-Iteration Operators (State-dependent)

Computed each iteration:

```python
# Temperature transport (depends on χ from turbulence model)
lower_Ai, diag_Ai, upper_Ai, b_i = transport(v_i, chi_i, g.bc_i)
lower_Ae, diag_Ae, upper_Ae, b_e = transport(v_e, chi_e, g.bc_e)

# Density transport (depends on chi_n, v_n from turbulence)
lower_An, diag_An, upper_An, b_n = transport(v_n, chi_n, g.bc_n)
```

Each `transport` call builds tridiagonal vectors (3 arrays of length 25) from diffusion and convection terms.

## Face/Gradient Computations

For each state variable, compute three quantities:

```python
# Example for ion temperature
i_f = g.I_i @ i + g.b_i_f        # Size 26 (faces)
i_g = g.D_i @ i + g.b_i_g        # Size 26 (∂i/∂ρ)
i_r = g.D_i_r @ i + g.b_i_r      # Size 26 (∂i/∂r_mid)
```

Pattern repeated for `e`, `n`, `p`, and derived ion quantities (`j`, `z`).

## Boundary Condition Encoding

Boundary conditions stored as 4-tuples:
```python
bc = (left_face, right_face, left_grad, right_grad)
```

Examples:
```python
g.bc_i = (None, 0.2, 0.0, 0.0)      # Right face = 0.2 keV, left grad = 0
g.bc_e = (None, 0.2, 0.0, 0.0)      # Right face = 0.2 keV, left grad = 0
g.bc_n = (None, 0.25e20, 0.0, 0.0)  # Right face = 0.25e20 m⁻³
g.bc_p = (None, None, 0.0, dp_edge) # Right grad from current
```

Operators `I`, `D` use these to set `b_f`, `b_g` vectors.

## Coupling Structure

### Temperature Coupling (Electron-Ion Heat Exchange)

```python
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...)
```

Returns **diagonal coupling coefficients** (size 25 each):
- `qei_ii` - Ion self-coupling (cooling)
- `qei_ee` - Electron self-coupling (cooling)  
- `qei_ie` - Ion←Electron transfer (heating)
- `qei_ei` - Electron←Ion transfer (heating)

These create **off-diagonal 25×25 blocks** in the spatial matrix:
```
[A_i + diag(qei_ii)    diag(qei_ie)    ]
[diag(qei_ei)      A_e + diag(qei_ee)  ]
```

The coupling is **energy-conserving**: `qei_ie = -qei_ei`, `qei_ii = -qei_ee`.

### Decoupled Variables

- **Psi** (poloidal flux): No coupling to other equations
- **Density**: No coupling to other equations (except via physics feedback)

This gives the **block-sparse** structure shown above.

## Source Vector

```python
b = jnp.r_[b_i + src_i,        # i sources (25)
           b_e + src_e,        # e sources (25)
           g.b_p + src_p,      # p sources (25)
           b_n + g.src_n]      # n sources (25)
```

Size 100, combining:
- `b_*` - Boundary condition contributions from operators
- `src_*` - Physical sources (heating, fusion, particles, current)

Sources computed as:
```python
src_i = g.src_i_ext + si_fus * g.geo_vpr + g.src_i_ped
src_e = g.src_e_ext + se_fus * g.geo_vpr + g.src_e_ped
src_p = -(j_bs + g.src_p_ext) * g.source_p_coeff
# g.src_n is precomputed
```

## Solver Assembly Flow

```python
# 1. Extract state components
i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]

# 2. Compute face values and gradients (linear operators)
i_f = g.I_i @ i + g.b_i_f
i_g = g.D_i @ i + g.b_i_g
i_r = g.D_i_r @ i + g.b_i_r
# ... repeat for e, n, p

# 3. Update derived ion quantities
j, z, k, k_f, w, u_f, j_bc, z_bc = ions(n, e, e_f)
j_f = g.I_j @ j + g.b_r * j_bc[1]
j_g = g.D_j @ j + g.b_r_g * j_bc[1]
j_r = g.D_j_r @ j + g.b_r_r * j_bc[1]
z_f = g.I_z @ z + g.b_r * z_bc[1]
z_r = g.D_z_r @ z + g.b_r_r * z_bc[1]

# 4. Compute q (safety factor)
q_f = jnp.abs(g.q_factor_axis * jnp.r_[1.0 / (p_g[1] * g.n), 
                                        g.face_centers[1:] / p_g[1:]])

# 5. Compute transport coefficients (nonlinear)
v_n, chi_i, chi_e, chi_n = turbulent_transport(...)  # QLKNN model
v_i, v_e, v_neo_n, chi_neo_i, chi_neo_e, chi_neo_n = neoclassical_transport(...)
chi_i += chi_neo_i  # Combine turbulent + neoclassical
chi_e += chi_neo_e
chi_n += chi_neo_n
v_n += v_neo_n

# 6. Build transport operators (tridiagonal only - not full matrices)
lower_Ai, diag_Ai, upper_Ai, b_i = transport(v_i, chi_i, g.bc_i)
lower_Ae, diag_Ae, upper_Ae, b_e = transport(v_e, chi_e, g.bc_e)
lower_An, diag_An, upper_An, b_n = transport(v_n, chi_n, g.bc_n)
# g.A_p_lower, g.A_p_diag, g.A_p_upper precomputed

# 7. Compute coupling and feedback terms
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...)
# g.ped_i, g.ped_e, g.ped_n precomputed

# 8. Assemble global RHS (100-vector)
b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]
rhs = ((tc_prev / tc_in) * state + θdt * tc * b)[:, None]

# 9. Solve coupled i-e system (block tridiagonal 2×2)
pred_i, pred_e = solve_implicit_coupled_2x2(
    lower_Ai, diag_Ai + qei_ii + g.ped_i, upper_Ai,
    lower_Ae, diag_Ae + qei_ee + g.ped_e, upper_Ae,
    qei_ie, qei_ei,
    tc[l.i], tc[l.e], θdt, rhs[l.i, 0], rhs[l.e, 0]
)

# 10. Solve decoupled p system (tridiagonal)
sol_p = solve_implicit_tridiag(g.A_p_lower, g.A_p_diag, g.A_p_upper,
                                tc[l.p], θdt, rhs[l.p])

# 11. Solve decoupled n system (tridiagonal)
sol_n = solve_implicit_tridiag(lower_An, diag_An + g.ped_n, upper_An,
                                tc[l.n], θdt, rhs[l.n])

# 12. Combine solutions
pred = jnp.r_[pred_i, pred_e, sol_p, sol_n]
```

## Implicit System

### Predictor-Corrector Loop

```python
pred = state  # Initial guess
tc_in_old = None

for iter in range(g.n_corr + 1):  # 0 (predictor), 1 (corrector)
    # Extract state components
    i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]
    
    # Compute all operators/coefficients from pred
    # ... (face values, gradients, transport coefficients)
    
    # Build transport operators (tridiagonal vectors only)
    lower_Ai, diag_Ai, upper_Ai, b_i = transport(v_i, chi_i, g.bc_i)
    lower_Ae, diag_Ae, upper_Ae, b_e = transport(v_e, chi_e, g.bc_e)
    lower_An, diag_An, upper_An, b_n = transport(v_n, chi_n, g.bc_n)
    
    # Compute coupling and source terms
    qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...)
    src_i = g.src_i_ext + si_fus * g.geo_vpr + g.src_i_ped
    src_e = g.src_e_ext + se_fus * g.geo_vpr + g.src_e_ped
    src_p = -(j_bs + g.src_p_ext) * g.source_p_coeff
    
    # Compute transient coefficients
    tc_in = jnp.r_[j * g.vpr_5_3, n * g.vpr_5_3, g.ones_vec, g.geo_vpr]
    tc_out = jnp.r_[g.tc_T, g.tc_T, g.tc_p_base * sigma, g.ones_vpr]
    tc = 1.0 / (tc_out * tc_in)
    tc_prev = tc_in if tc_in_old is None else tc_in_old
    
    # Build RHS
    θdt = g.theta * dt
    b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]
    rhs = ((tc_prev / tc_in) * state + θdt * tc * b)[:, None]
    
    # Solve coupled i-e system
    pred_i, pred_e = solve_implicit_coupled_2x2(
        lower_Ai, diag_Ai + qei_ii + g.ped_i, upper_Ai,
        lower_Ae, diag_Ae + qei_ee + g.ped_e, upper_Ae,
        qei_ie, qei_ei,
        tc[l.i], tc[l.e], θdt, rhs[l.i, 0], rhs[l.e, 0]
    )
    
    # Solve decoupled p and n systems
    sol_p = solve_implicit_tridiag(g.A_p_lower, g.A_p_diag, g.A_p_upper,
                                    tc[l.p], θdt, rhs[l.p])
    sol_n = solve_implicit_tridiag(lower_An, diag_An + g.ped_n, upper_An,
                                    tc[l.n], θdt, rhs[l.n])
    
    # Combine solutions
    pred = jnp.r_[pred_i, pred_e, sol_p, sol_n]
    tc_in_old = tc_in
```

**Key insight**: After first iteration, `tc_prev ≈ tc_in`, so ratio ≈ 1.

### Solver Properties

**Tridiagonal Operators** (3 vectors of length 25 each):
- **Sparsity**: Only 3 non-zero diagonals per equation
- **Storage**: O(n) per variable instead of O(n²)
- **Coupling**: 2×2 blocks at each spatial point for i-e system
- **Condition number**: ~O(n²) from diffusion, improved by upwinding

**System Structure**:
- Form: `[I - θdt*diag(tc)*A] @ x = rhs` for each subsystem
- **M-matrix** structure for small `dt` (positive diagonal, negative off-diagonals)
- **Thomas algorithm**: Guaranteed stable for M-matrices
- **Invertible** for all `dt > 0` with backward Euler (`θ=1`)

## Numerical Linear Algebra Details

### Operator Costs

Per iteration:
- **Matrix-vector products**: ~15 products (`I@x`, `D@x`, `D_r@x` for 4 fields + ions) → O(n) each
- **Transport operator builds**: 3× `transport()` → O(n) assembly (tridiagonal vectors)
- **Tridiagonal solves**: 1× block 2×2 solve + 2× standard solves → O(n) each
- **Total solve cost**: O(n) instead of O(n³)

Total per iteration: **O(n) operations** = O(25) dominated by transport coefficient computation, not linear algebra.

### Memory Layout

**State vector** (contiguous):
```
state[0:25]   = i  (ion temperature)
state[25:50]  = e  (electron temperature)
state[50:75]  = p  (poloidal flux)
state[75:100] = n  (electron density)
```

**Tridiagonal storage**:
- `A_i, A_e, A_n, A_p`: Stored as 3 vectors of length 25 each (lower, diag, upper)
- No full matrix assembly - only tridiagonal vectors
- Total: ~6KB for all transport operators (much less than 80KB for full matrices)

### JAX Arrays

All computations use JAX:
```python
state = jnp.array(...)                      # JAX device array
lower, diag, upper, b = transport(...)      # JAX operations
pred = jax.lax.linalg.tridiagonal_solve()   # JAX tridiagonal solver
```

**Benefits**:
- Automatic differentiation available
- GPU-ready (not currently used)
- JIT compilation potential
- Immutable arrays (functional style)
- Specialized solvers (tridiagonal) for efficiency

## Exponential Fitting Details

### Peclet Number and Upwind Factor

For convection coefficient at face `i`:

```python
P = v_f[i] * dx / d_f[i]
α(P) = upwind_factor(P)
```

The function `α(P)` smoothly transitions:

```python
def peclet_to_alpha(P):
    if abs(P) < 0.001:
        return 0.5  # Central difference
    elif P > 10:
        return (P - 1) / P  # Full upwind
    elif 0 < P < 10:
        return ((P-1) + (1 - P/10)^5) / P  # Smooth transition
    elif -10 < P < 0:
        return ((1 + P/10)^5 - 1) / P
    else:  # P < -10
        return -1 / P
```

**Key property**: For large |P|, this recovers pure upwinding which is stable.

### Matrix Stencil

The convection matrix stencil for cell `i`:

```
[... , (1-α_L)*v_L/dx , (α_L*v_L - α_R*v_R)/dx , -(1-α_R)*v_R/dx , ...]
       i-1                      i                       i+1
```

Where `v_L = v_f[i]`, `v_R = v_f[i+1]`.

For **pure diffusion** (`v=0`): reduces to standard central difference.
For **pure convection** (`d→0`, `P→∞`): becomes first-order upwind.

## Output Format

### Binary File Structure

File `run.raw` contains:
```
For each time step:
    t (float64)           # 1 value
    state (float64)       # 100 values (4*25)
```

Total size: `nt * (1 + 4*n) * 8` bytes where `nt = t_end/dt ≈ 26`.

### Memory-Mapped Reading

In `plot.py`:
```python
rc = 1 + 4 * g.n  # Record size = 101
sz = np.dtype(np.float64).itemsize  # 8 bytes

# Memory map with custom strides
with open("run.raw", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    nt = len(mm) // (rc * sz)
    
    # Time array with stride = rc*sz (skip state data)
    t_out = np.ndarray(shape=(nt,), dtype=np.float64, buffer=mm,
                       offset=0, strides=(rc*sz,))
    
    # State array with stride = (rc*sz, n*sz, sz)
    states = np.ndarray(shape=(nt, 4, g.n), dtype=np.float64, buffer=mm,
                        offset=sz, strides=(rc*sz, g.n*sz, sz))
```

This enables **zero-copy** access to specific time slices without loading full file.

## Summary

This implementation uses:

1. **Finite volume method** with exponential fitting for stability
2. **Implicit backward Euler** (`θ=1`) for unconditional stability
3. **Specialized tridiagonal solvers** instead of assembling full 100×100 system
4. **Predictor-corrector** for nonlinear transport coefficients
5. **Precomputed operators** (I, D) applied via matrix-vector products
6. **Block tridiagonal 2×2 solver** for coupled ion-electron temperatures
7. **Separate tridiagonal solvers** for decoupled poloidal flux and density
8. **O(n) computational complexity** per iteration instead of O(n³)
9. **JAX arrays** with specialized `jax.lax.linalg.tridiagonal_solve`
10. **Memory-mapped I/O** for efficient post-processing

The implementation exploits **tridiagonal structure** of transport operators and uses **specialized solvers** for maximum efficiency. The i-e temperature coupling is handled via a block tridiagonal solver with 2×2 blocks at each spatial point, while p and n are solved independently. This avoids the need to assemble or invert the full 100×100 system matrix. The code uses single-letter suffixes (`_f`, `_g`, `_r`) and returns tridiagonal vectors (`lower`, `diag`, `upper`) from the `transport()` function.
