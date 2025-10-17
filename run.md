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

### Transport Terms Matrix `A`

Function `transport(v, d, bc)` returns `(A, b)` where:

```
A @ x + b = ∇·(d∇x - vx)
```

Combines:
- **Diffusion**: `A_diff, b_diff = diff_terms(d, bc)`
- **Convection**: `A_conv, b_conv = conv_terms(v, d, bc)`
- **Sum**: `A = A_diff + A_conv`, `b = b_diff + b_conv`

### Diffusion Matrix `A_diff`

For `∇·(d∇x)` discretization (size `n × n`):

```
A_diff = (1/dx²) * tridiag(off, diag, off)
```

Where:
```python
diag[i] = -(d_f[i+1] + d_f[i])
off[i] = d_f[i+1]
```

This is a **symmetric negative definite** tridiagonal matrix.

### Convection Matrix `A_conv`

For `∇·(vx)` with exponential fitting (size `n × n`):

```
A_conv = (1/dx) * tridiag(below, diag, above)
```

Where Peclet number `P[i] = v[i]*dx/d[i]` determines upwind factor `α(P)`:

```python
diag[i] = (α_L[i] * v[i] - α_R[i] * v[i+1]) / dx
above[i] = -(1 - α_R[i]) * v[i+1] / dx
below[i] = (1 - α_L[i+1]) * v[i+1] / dx
```

This is a **non-symmetric** tridiagonal matrix with upwind bias.

## Block Matrix Structure

The full spatial operator is a **4×4 block matrix** where each block is `n×n`:

```
A (100×100) = 
┌────────────────────────────────┐
│ A_ii    A_ie      0        0   │  i (25×25 blocks)
│ A_ei    A_ee      0        0   │  e
│ 0       0       A_pp       0   │  p
│ 0       0        0       A_nn  │  n
└────────────────────────────────┘
```

Where:
- `A_ii = A_i + diag(qei_ii + g.ped_i)` - Ion transport + self-coupling + pedestal feedback
- `A_ee = A_e + diag(qei_ee + g.ped_e)` - Electron transport + self-coupling + pedestal feedback
- `A_ie = diag(qei_ie)` - Ion ← electron heat exchange
- `A_ei = diag(qei_ei)` - Electron ← ion heat exchange
- `A_pp = g.A_p` - Psi transport (precomputed, constant)
- `A_nn = A_n + diag(g.ped_n)` - Density transport + pedestal feedback

### Block Matrix Assembly

```python
# Build block components with coupling and feedback
A_ii = A_i + jnp.diag(qei_ii + g.ped_i)
A_ie = jnp.diag(qei_ie)
A_ei = jnp.diag(qei_ei)
A_ee = A_e + jnp.diag(qei_ee + g.ped_e)
A_pp = g.A_p
A_nn = A_n + jnp.diag(g.ped_n)

# Assemble full spatial operator
A = jnp.block([
    [A_ii,    A_ie,    g.zero, g.zero],
    [A_ei,    A_ee,    g.zero, g.zero],
    [g.zero,  g.zero,  A_pp,   g.zero],
    [g.zero,  g.zero,  g.zero, A_nn  ]
])
```

**Structure properties:**
- **Banded**: Only 3 diagonals per block + coupling terms
- **Block sparse**: Many zero blocks
- **Asymmetric**: Convection terms break symmetry
- **Indefinite**: Mixed signs from different physics

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

## Implicit Time Step - Textbook Form

The implicit system at each time step has the standard form:

```
M @ x_new = rhs
```

### System Matrix `M`

```python
M = g.identity - dt * g.theta * jnp.expand_dims(tc, 1) * A
```

This is a **100×100 matrix**:
- `jnp.expand_dims(tc, 1)` → shape `(100, 1)`
- Broadcasting: `(100, 1) * (100, 100)` → `(100, 100)`
- Each row `i` scaled by `tc[i]`

**Properties:**
- **Non-symmetric** due to convection
- **Diagonally dominant** for small `dt`
- **Well-conditioned** with `θ=1` (backward Euler)

### Right-Hand Side `rhs`

```python
rhs = (tc_prev / tc_in) * state + g.theta * dt * tc * b
```

Where:
- **First term**: Ratio of old to new transient coefficients (element-wise)
- **Second term**: Implicit source contribution
- **`b`**: Global RHS vector (transport boundaries + sources)

### Linear Solve

```python
pred = jnp.linalg.solve(M, rhs)
```

Uses **LU decomposition** (default for `jnp.linalg.solve`):
- Factorization: `P*L*U = M`
- Forward solve: `L*y = P*rhs`
- Backward solve: `U*pred = y`

Cost: **O(n³)** = O(100³) ≈ 1M operations per solve

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

# Psi transport (constant diffusion)
g.A_p, g.b_p = transport(g.v_p_zero, g.geo_g2g3_over_rhon_face, g.bc_p)
```

### Per-Iteration Operators (State-dependent)

Computed each iteration:

```python
# Temperature transport (depends on χ from turbulence model)
A_i, b_i = transport(v_i, chi_i, g.bc_i)
A_e, b_e = transport(v_e, chi_e, g.bc_e)

# Density transport (depends on chi_n, v_n from turbulence)
A_n, b_n = transport(v_n, chi_n, g.bc_n)
```

Each `transport` call builds a tridiagonal matrix from diffusion and convection.

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

## Matrix Assembly Flow

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
q_f = jnp.r_[jnp.abs(g.q_factor_axis / (p_g[1] * g.inv_dx))[None],
             jnp.abs(g.q_factor_bulk * g.face_centers[1:] / p_g[1:])]

# 5. Compute transport coefficients (nonlinear)
chi_i, chi_e, chi_n, v_n = turbulent_transport(...)  # QLKNN model
v_i, v_e, chi_neo_i, chi_neo_e, chi_neo_n, v_neo_n = neoclassical_transport(...)
chi_i += chi_neo_i  # Combine turbulent + neoclassical
chi_e += chi_neo_e
chi_n += chi_neo_n
v_n += v_neo_n

# 6. Build transport matrices (25×25 each)
A_i, b_i = transport(v_i, chi_i, g.bc_i)  # Tridiagonal
A_e, b_e = transport(v_e, chi_e, g.bc_e)
A_n, b_n = transport(v_n, chi_n, g.bc_n)
# g.A_p precomputed (constant diffusion)

# 7. Assemble block components
A_ii = A_i + jnp.diag(qei_ii + g.ped_i)
A_ie = jnp.diag(qei_ie)
A_ei = jnp.diag(qei_ei)
A_ee = A_e + jnp.diag(qei_ee + g.ped_e)
A_pp = g.A_p
A_nn = A_n + jnp.diag(g.ped_n)

# 8. Assemble 100×100 spatial operator
A = jnp.block([
    [A_ii,    A_ie,    g.zero, g.zero],
    [A_ei,    A_ee,    g.zero, g.zero],
    [g.zero,  g.zero,  A_pp,   g.zero],
    [g.zero,  g.zero,  g.zero, A_nn  ]
])

# 9. Assemble 100-vector RHS
b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]
```

## Implicit System

### Predictor-Corrector Loop

```python
pred = state  # Initial guess
tc_prev = None

for iter in range(g.n_corr + 1):  # 0 (predictor), 1 (corrector)
    # Compute all matrices/vectors from pred
    A, b = assemble(pred)
    tc_in, tc_out = compute_transients(pred)
    
    # Build implicit system: M @ x_new = rhs
    tc = 1 / (tc_out * tc_in)
    M = g.identity - dt * g.theta * jnp.expand_dims(tc, 1) * A
    
    # RHS uses ratio of old/new transients
    if tc_prev is None:
        tc_prev = tc_in
    rhs = (tc_prev / tc_in) * state + g.theta * dt * tc * b
    
    # Solve
    pred = jnp.linalg.solve(M, rhs)
    tc_prev = tc_in
```

**Key insight**: After first iteration, `tc_prev ≈ tc_in`, so ratio ≈ 1.

### Matrix Properties

**A** (100×100 spatial operator):
- **Block structure**: 4×4 blocks of 25×25
- **Sparsity**: Tri-diagonal per block + diagonal coupling
- **Bandwidth**: 3 within blocks, full coupling across i-e
- **Condition number**: ~O(n²) from diffusion, improved by upwinding

**M** (100×100 system matrix):
- Form: `I - dt*θ*D*A` where `D=diag(tc)`, `A=spatial operator`
- **M-matrix** structure for small `dt` (positive diagonal, negative off-diagonals)
- **Invertible** for `dt < dt_max` (stability bound)

## Numerical Linear Algebra Details

### Operator Costs

Per iteration:
- **Matrix-vector products**: ~15 products (`I@x`, `D@x`, `D_r@x` for 4 fields + ions) → O(n) each
- **Transport matrix builds**: 3× `transport()` → O(n) assembly
- **Block matrix build**: `jnp.block()` → O(n²) allocation
- **Linear solve**: `jnp.linalg.solve()` → O(n³) = O(1M) operations

Total per iteration: **~1M operations** dominated by dense solve.

### Memory Layout

**State vector** (contiguous):
```
state[0:25]   = i  (ion temperature)
state[25:50]  = e  (electron temperature)
state[50:75]  = p  (poloidal flux)
state[75:100] = n  (electron density)
```

**Matrix storage**:
- `A_i, A_e, A_n, A_p`: Stored as full `(25, 25)` arrays (not sparse)
- `A`: Full `(100, 100)` array
- Total: ~80KB for all matrices

### JAX Arrays

All computations use JAX:
```python
state = jnp.array(...)     # JAX device array
A = jnp.block()            # JAX array operations
pred = jnp.linalg.solve()  # JAX linear solver
```

**Benefits**:
- Automatic differentiation available
- GPU-ready (not currently used)
- JIT compilation potential
- Immutable arrays (functional style)

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
3. **Block-coupled system** (100×100) solving all 4 variables simultaneously
4. **Predictor-corrector** for nonlinear transport coefficients
5. **Precomputed operators** (I, D) applied via matrix-vector products
6. **Tridiagonal blocks** within each variable
7. **Diagonal coupling** between i-e temperatures
8. **Dense linear solver** (LU decomposition) for full system in textbook form `M @ x = rhs`
9. **JAX arrays** for potential GPU acceleration
10. **Memory-mapped I/O** for efficient post-processing

The matrix structure balances **sparsity** (tridiagonal blocks) with **coupling** (off-diagonal blocks), solved via dense methods for simplicity and robustness. The code uses single-letter suffixes (`_f`, `_g`, `_r`) and clear intermediate variables (`A_ii`, `A_ie`, `M`, `rhs`) for readability.
