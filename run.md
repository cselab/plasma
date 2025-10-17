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
nc = 25
l.i = np.s_[:nc]
l.e = np.s_[nc:2*nc]
l.p = np.s_[2*nc:3*nc]
l.n = np.s_[3*nc:4*nc]
```

## Differential Operators

Three fundamental operators map cell values to face values/gradients.

### Interpolation Operator `I` (Face values)

Matrix `I: R^n → R^(n+1)` computes face values from cell centers:

```
I @ x + b_face = x_face
```

Structure `(n+1) × n`:
```
[1   0   0  ...  0  ]   [x_0]     [x_0        ]
[0.5 0.5 0  ...  0  ]   [x_1]     [(x_0+x_1)/2]
[0  0.5 0.5 ...  0  ] @ [x_2]  +  [(x_1+x_2)/2]
[...            ... ]   [...]  b  [...]
[0   0   0  ... 1  ]   [x_n]     [x_n + b*bc ]
```

Where `b_face` applies right boundary condition (value or gradient-based).

### Gradient Operator `D` (Normalized coordinates)

Matrix `D: R^n → R^(n+1)` computes gradients w.r.t. normalized rho:

```
D @ x + b_grad = ∂x/∂ρ_norm
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
D_r @ x + b_grad_r = ∂x/∂r_mid
```

Similar to `D` but with non-uniform `inv_dr[i] = 1/Δr_mid[i]`.

## Transport Operators

For each variable, transport has form:

```
∂/∂t(tic * x) = (1/toc) * [∇·(d∇x - vx) + S_mat*x + S]
```

### Transport Terms Matrix `A`

Function `trans_terms(v, d, bc)` returns `(A, b)` where:

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
diag[i] = -(d_face[i+1] + d_face[i])
off[i] = d_face[i+1]
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
spatial_mat (100×100) = 
┌────────────────────────────────┐
│ A_ii    A_ie      0        0   │  i (25×25 blocks)
│ A_ei    A_ee      0        0   │  e
│ 0       0       A_pp       0   │  p
│ 0       0        0       A_nn  │  n
└────────────────────────────────┘
```

Where:
- `A_ii = A_i + diag(Qei_ii + ped_mat_T)` - Ion transport + self-coupling + pedestal feedback
- `A_ee = A_e + diag(Qei_ee + ped_mat_T)` - Electron transport + self-coupling + pedestal feedback
- `A_ie = diag(Qei_ie)` - Ion ← electron heat exchange
- `A_ei = diag(Qei_ei)` - Electron ← ion heat exchange
- `A_pp = g.A_p` - Psi transport (precomputed, constant)
- `A_nn = A_n + diag(ped_mat_n)` - Density transport + pedestal feedback

### Block Matrix Assembly

```python
A_ii = A_i + jnp.diag(Qei_ii + g.ped_mat_T)
A_ie = jnp.diag(Qei_ie)
A_ei = jnp.diag(Qei_ei)
A_ee = A_e + jnp.diag(Qei_ee + g.ped_mat_T)
A_pp = g.A_p
A_nn = A_n + jnp.diag(g.ped_mat_n)

spatial_mat = jnp.block([
    [A_ii,  A_ie,  g.zero, g.zero],
    [A_ei,  A_ee,  g.zero, g.zero],
    [g.zero, g.zero, A_pp,   g.zero],
    [g.zero, g.zero, g.zero, A_nn  ]
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
tc_in = [ions_n_i * vpr^(5/3),    # i: heat capacity
         n * vpr^(5/3),            # e: heat capacity  
         ones,                     # p: no scaling
         vpr]                      # n: volume

tc_out = [1.5 * vpr^(-2/3) * keV_to_J,  # i: temperature factor
          1.5 * vpr^(-2/3) * keV_to_J,  # e: temperature factor
          c_p_coeff * sigma,             # p: conductivity
          ones]                          # n: no scaling
```

Combined transient coefficient:
```python
tc = 1 / (tc_out * tc_in)  # element-wise, size 100
```

## Implicit Time Step

The implicit system at each time step:

```
[I - dt*θ*diag(tc)*spatial_mat] * pred = (tc_old/tc_in)*state + dt*θ*tc*spatial_vec
```

### Left-Hand Side Matrix

```python
lhs = g.identity - dt * g.theta_imp * jnp.expand_dims(tc, 1) * spatial_mat
```

This is a **100×100 dense matrix** (originally sparse but broadcasting makes dense):
- `jnp.expand_dims(tc, 1)` → shape `(100, 1)`
- Broadcasting: `(100, 1) * (100, 100)` → `(100, 100)`
- Each row `i` scaled by `tc[i]`

**Properties:**
- **Non-symmetric** due to convection
- **Diagonally dominant** for small `dt`
- **Well-conditioned** with `θ=1` (backward Euler)

### Right-Hand Side Vector

```python
rhs = (tc_old / tc_in) * state + g.theta_imp * dt * tc * spatial_vec
```

**First term**: Ratio of old to new transient coefficients (element-wise)
**Second term**: Implicit source contribution

### Linear Solve

```python
pred = jnp.linalg.solve(lhs, rhs)
```

Uses **LU decomposition** (default for `jnp.linalg.solve`):
- Factorization: `P*L*U = lhs`
- Forward solve: `L*y = P*rhs`
- Backward solve: `U*pred = y`

Cost: **O(n³)** = O(100³) ≈ 1M operations per solve

## Operator Precomputation

### Static Operators (Geometry-dependent)

Computed once during initialization:

```python
# Interpolation operators (26×25)
g.I_i, g.b_i_face = face_op(bc_i)
g.I_e, g.b_e_face = face_op(bc_e)
g.I_n, g.b_n_face = face_op(bc_n)
g.I_p, g.b_p_face = face_op(bc_p)

# Gradient operators (26×25)  
g.D_i, g.b_i_grad = grad_op(bc_i)
g.D_e, g.b_e_grad = grad_op(bc_e)
g.D_n, g.b_n_grad = grad_op(bc_n)
g.D_p, g.b_p_grad = grad_op(bc_p)

# Physical coordinate gradients (26×25)
g.D_i_r, g.b_i_grad_r = grad_op_nu(inv_drmid, bc_i)
g.D_e_r, g.b_e_grad_r = grad_op_nu(inv_drmid, bc_e)
g.D_n_r, g.b_n_grad_r = grad_op_nu(inv_drmid, bc_n)

# Psi transport (constant diffusion)
g.A_p, g.b_p = trans_terms(v_zero, g.geo_g2g3_over_rhon_face, bc_p)
```

### Per-Iteration Operators (State-dependent)

Computed each iteration:

```python
# Temperature transport (depends on χ from turbulence model)
A_i, b_i = trans_terms(v_i, chi_i, bc_i)
A_e, b_e = trans_terms(v_e, chi_e, bc_e)

# Density transport (depends on D, V from turbulence)
A_n, b_n = trans_terms(v_n, D_n, bc_n)
```

Each `trans_terms` call builds a tridiagonal matrix from diffusion and convection.

## Face/Gradient Computations

For each state variable, compute three quantities:

```python
# Example for ion temperature
i_face = g.I_i @ i + g.b_i_face          # Size 26 (faces)
i_grad = g.D_i @ i + g.b_i_grad          # Size 26 (∂i/∂ρ)
i_grad_r = g.D_i_r @ i + g.b_i_grad_r    # Size 26 (∂i/∂r_mid)
```

Pattern repeated for `e`, `n`, `p`, and derived quantities (`n_i`, `n_impurity`).

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

Operators `I`, `D` use these to set `b_face`, `b_grad` vectors.

## Coupling Structure

### Temperature Coupling (Electron-Ion Heat Exchange)

```python
Qei_ii, Qei_ee, Qei_ie, Qei_ei = qei_coupling(...)
```

Returns **diagonal coupling coefficients** (size 25 each):
- `Qei_ii` - Ion self-coupling (cooling)
- `Qei_ee` - Electron self-coupling (cooling)  
- `Qei_ie` - Ion←Electron transfer (heating)
- `Qei_ei` - Electron←Ion transfer (heating)

These create **off-diagonal 25×25 blocks** in the spatial matrix:
```
[A_i + diag(Qei_ii)    diag(Qei_ie)    ]
[diag(Qei_ei)      A_e + diag(Qei_ee)  ]
```

The coupling is **energy-conserving**: `Qei_ie = -Qei_ei`, `Qei_ii = -Qei_ee`.

### Decoupled Variables

- **Psi** (poloidal flux): No coupling to other equations
- **Density**: No coupling to other equations (except via physics feedback)

This gives the **block-sparse** structure shown above.

## Source Vector

```python
spatial_vec = jnp.r_[b_i + src_i,           # i sources (25)
                     b_e + src_e,           # e sources (25)
                     g.b_p + src_p,         # p sources (25)
                     b_n + g.source_n_constant]  # n sources (25)
```

Size 100, combining:
- `b_*` - Boundary condition contributions from operators
- `src_*` - Physical sources (heating, fusion, particles, current)

## Matrix Assembly Flow

```python
# 1. Extract state components
i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]

# 2. Compute face values and gradients (linear operators)
i_face = g.I_i @ i + g.b_i_face
i_grad = g.D_i @ i + g.b_i_grad
i_grad_r = g.D_i_r @ i + g.b_i_grad_r
# ... repeat for e, n, p

# 3. Update derived quantities
ions = ions_update(n, e, e_face)  # Returns 8-element tuple
q_face = compute_q(p_grad)

# 4. Compute transport coefficients (nonlinear)
chi_i, chi_e, D_n, v_n = turbulent_transport(...)  # QLKNN model
v_i, v_e, chi_neo_i, chi_neo_e, D_neo_n, v_neo_n = neoclassical_transport(...)
chi_i += chi_neo_i  # Combine turbulent + neoclassical
# ...

# 5. Build transport matrices (25×25 each)
A_i, b_i = trans_terms(v_i, chi_i, g.bcs[0])  # Tridiagonal
A_e, b_e = trans_terms(v_e, chi_e, g.bcs[1])
A_n, b_n = trans_terms(v_n, D_n, g.bcs[3])
# g.A_p precomputed (constant diffusion)

# 6. Assemble 100×100 block matrix
spatial_mat = jnp.block([
    [A_i + diag(...), diag(...), 0, 0],
    [diag(...), A_e + diag(...), 0, 0],
    [0, 0, A_p, 0],
    [0, 0, 0, A_n + diag(...)]
])

# 7. Assemble 100-vector
spatial_vec = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + src_n]
```

## Implicit System

### Predictor-Corrector Loop

```python
pred = state  # Initial guess
tc_in_old = None

for iter in range(g.n_corr + 1):  # 0 (predictor), 1 (corrector)
    # Compute all matrices/vectors from pred
    spatial_mat, spatial_vec = assemble(pred)
    tc_in, tc_out = compute_transients(pred)
    
    # Build implicit system
    tc = 1 / (tc_out * tc_in)
    lhs = I - dt * θ * expand_dims(tc, 1) * spatial_mat
    
    # RHS uses ratio of old/new transients
    tc_old = tc_in if (tc_in_old is None) else tc_in_old
    rhs = (tc_old / tc_in) * state + θ * dt * tc * spatial_vec
    
    # Solve
    pred = solve(lhs, rhs)
    tc_in_old = tc_in
```

**Key insight**: After first iteration, `tc_old ≈ tc_in`, so ratio ≈ 1.

### Matrix Properties

**spatial_mat** (100×100):
- **Block structure**: 4×4 blocks of 25×25
- **Sparsity**: Tri-diagonal per block + diagonal coupling
- **Bandwidth**: 3 within blocks, full coupling across i-e
- **Condition number**: ~O(n²) from diffusion, improved by upwinding

**lhs** (100×100):
- Form: `I - dt*θ*D*A` where `D=diag(tc)`, `A=spatial_mat`
- **M-matrix** structure for small `dt` (positive diagonal, negative off-diagonals)
- **Invertible** for `dt < dt_max` (stability bound)

## Numerical Linear Algebra Details

### Operator Costs

Per iteration:
- **7 matrix-vector products**: `I@x`, `D@x` (×3 coord systems × 4 variables) → O(n) each
- **3 transport matrix builds**: `trans_terms()` → O(n) assembly
- **1 block matrix build**: `jnp.block()` → O(n²) allocation
- **1 linear solve**: `jnp.linalg.solve()` → O(n³) = O(1M) operations

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
- `spatial_mat`: Full `(100, 100)` array
- Total: ~80KB for all matrices

### JAX Arrays

All computations use JAX:
```python
state = jnp.array(...)     # JAX device array
spatial_mat = jnp.block()  # JAX array operations
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
P = v_face[i] * dx / d_face[i]
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

Where `v_L = v_face[i]`, `v_R = v_face[i+1]`.

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
rc = 1 + 4 * n  # Record size = 101
sz = np.float64.itemsize  # 8 bytes

# Memory map with custom strides
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
nt = len(mm) // (rc * sz)

# Time array with stride = rc*sz (skip state data)
t_out = np.ndarray(shape=(nt,), dtype=np.float64, buffer=mm,
                   offset=0, strides=(rc*sz,))

# State array with stride = (rc*sz, n*sz, sz)
states = np.ndarray(shape=(nt, 4, n), dtype=np.float64, buffer=mm,
                    offset=sz, strides=(rc*sz, n*sz, sz))
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
8. **Dense linear solver** (LU decomposition) for full system
9. **JAX arrays** for potential GPU acceleration
10. **Memory-mapped I/O** for efficient post-processing

The matrix structure balances **sparsity** (tridiagonal blocks) with **coupling** (off-diagonal blocks), solved via dense methods for simplicity and robustness.
