# Naming Scheme

## Core State Variables

Single-letter names for evolved quantities (all at cell centers):
- `i` = ion temperature [keV]
- `e` = electron temperature [keV]
- `p` = poloidal flux [Wb]
- `n` = electron density [m⁻³]

Combined in state vector: `state = [i, e, p, n]` (size 100 = 4×25)

## Location/Derivative Suffixes

### Primary Suffixes
- **`_f`** = values at cell faces (n+1 points), e.g., `i_f`, `e_f`, `n_f`
- **`_g`** = gradient w.r.t. normalized ρ (ρ_norm), e.g., `i_g`, `e_g`
- **`_r`** = gradient w.r.t. midplane radius (r_mid), e.g., `i_r`, `e_r`

### Examples
```python
i = pred[l.i]                    # Cell center (25)
i_f = g.I_i @ i + g.b_i_f        # Faces (26)
i_g = g.D_i @ i + g.b_i_g        # Gradient ∂i/∂ρ (26)
i_r = g.D_i_r @ i + g.b_i_r      # Gradient ∂i/∂r (26)
```

Pattern: `{var}_{location}` where location is `f` (face), `g` (gradient ρ), or `r` (gradient r_mid).

## Ion/Impurity Variables

Single ASCII letters for ion-related quantities:
- **`j`** = main ion density [m⁻³] (cell centers)
- **`z`** = impurity density [m⁻³] (cell centers)
- **`k`** = main ion effective charge (cell centers)
- **`k_f`** = main ion effective charge (faces)
- **`w`** = impurity effective charge (cell centers)
- **`u_f`** = Z_eff (effective charge at faces)
- **`j_bc`** = main ion boundary conditions tuple
- **`z_bc`** = impurity boundary conditions tuple

### Ion Function
```python
j, z, k, k_f, w, u_f, j_bc, z_bc = ions(n_e, T_e, T_e_face)
```

Returns:
- `j` = main ion density
- `z` = impurity density
- `k` = main ion charge state Z_i = ⟨Z²⟩/⟨Z⟩
- `k_f` = main ion charge (faces)
- `w` = impurity charge state
- `u_f` = effective charge Z_eff (faces)
- `j_bc` = ion density boundary conditions
- `z_bc` = impurity density boundary conditions

### Derived Ion Quantities
```python
j_f = g.I_j @ j + g.b_r * j_bc[1]
j_g = g.D_j @ j + g.b_r_g * j_bc[1]
j_r = g.D_j_r @ j + g.b_r_r * j_bc[1]
z_f = g.I_z @ z + g.b_r * z_bc[1]
z_r = g.D_z_r @ z + g.b_r_r * z_bc[1]
```

## Matrix/Operator Naming

### Differential Operators (Precomputed)
- **`I_*`** = interpolation operator (cell → face), e.g., `g.I_i`, `g.I_e`, `g.I_n`, `g.I_p`
- **`D_*`** = gradient operator (ρ coords), e.g., `g.D_i`, `g.D_e`, `g.D_n`, `g.D_p`
- **`D_*_r`** = gradient operator (r_mid coords), e.g., `g.D_i_r`, `g.D_e_r`, `g.D_n_r`

Ion operators:
- **`I_j`** = main ion interpolation (was `I_ni`)
- **`D_j`** = main ion gradient ρ (was `D_ni_rho`)
- **`D_j_r`** = main ion gradient r (was `D_ni_rmid`)
- **`I_z`** = impurity interpolation (was `I_nimp`)
- **`D_z_r`** = impurity gradient r (was `D_nimp_rmid`)

All are `(n+1) × n` matrices.

### Boundary Vectors (Precomputed)
- **`b_*_f`** = face boundary contributions, e.g., `g.b_i_f`, `g.b_e_f`
- **`b_*_g`** = gradient boundary contributions, e.g., `g.b_i_g`, `g.b_p_g`
- **`b_*_r`** = r_mid gradient boundaries, e.g., `g.b_i_r`, `g.b_e_r`

Special boundary vectors:
- **`b_r`** = right boundary selector [0,0,...,0,1]
- **`b_r_g`** = right boundary for gradient (with 2*dx factor)
- **`b_r_r`** = right boundary for r_mid gradient

All are size `n+1`.

### Transport Operators (Per-Iteration)

Transport operators stored as **tridiagonal vectors** (not full matrices):
- **`lower_A*`** = subdiagonal coefficients (size `n`), e.g., `lower_Ai`, `lower_Ae`, `lower_An`
- **`diag_A*`** = diagonal coefficients (size `n`), e.g., `diag_Ai`, `diag_Ae`, `diag_An`
- **`upper_A*`** = superdiagonal coefficients (size `n`), e.g., `upper_Ai`, `upper_Ae`, `upper_An`
- **`b_*`** = transport RHS vector (size `n`), e.g., `b_i`, `b_e`, `b_n`

Built from: `lower, diag, upper, b = transport(v_f, d_f, bc)`

For poloidal flux (precomputed constant):
- **`g.A_p_lower`**, **`g.A_p_diag`**, **`g.A_p_upper`** = psi transport tridiagonal vectors

### Implicit Solver Strategy

The system is **never assembled as a full 100×100 matrix**. Instead, specialized tridiagonal solvers are used:

```python
# Build transport operators (tridiagonal vectors only)
lower_Ai, diag_Ai, upper_Ai, b_i = transport(v_i, chi_i, g.bc_i)
lower_Ae, diag_Ae, upper_Ae, b_e = transport(v_e, chi_e, g.bc_e)
lower_An, diag_An, upper_An, b_n = transport(v_n, chi_n, g.bc_n)

# Compute coupling and feedback terms (diagonal vectors)
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...)

# Assemble global RHS
b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]

# Time coefficients
tc = 1 / (tc_out * tc_in)
θdt = g.theta * dt
rhs = ((tc_prev / tc_in) * state + θdt * tc * b)[:, None]

# Solve coupled i-e system (block tridiagonal 2×2)
pred_i, pred_e = solve_implicit_coupled_2x2(
    lower_Ai, diag_Ai + qei_ii + g.ped_i, upper_Ai,
    lower_Ae, diag_Ae + qei_ee + g.ped_e, upper_Ae,
    qei_ie, qei_ei,
    tc[l.i], tc[l.e], θdt, rhs[l.i, 0], rhs[l.e, 0]
)

# Solve decoupled systems (standard tridiagonal)
sol_p = solve_implicit_tridiag(g.A_p_lower, g.A_p_diag, g.A_p_upper,
                                tc[l.p], θdt, rhs[l.p])
sol_n = solve_implicit_tridiag(lower_An, diag_An + g.ped_n, upper_An,
                                tc[l.n], θdt, rhs[l.n])

# Combine solutions
pred = jnp.r_[pred_i, pred_e, sol_p, sol_n]
```

Components:
- **`lower_A*`, `diag_A*`, `upper_A*`** = tridiagonal transport operator vectors
- **`qei_ii`, `qei_ee`, `qei_ie`, `qei_ei`** = diagonal coupling vectors (size 25)
- **`g.ped_i`, `g.ped_e`, `g.ped_n`** = diagonal pedestal feedback vectors (size 25)
- **`tc`** = time coefficients (size 100)
- **`rhs`** = right-hand side vector (size 100)
- **`b`** = spatial RHS before time scaling
- **`θdt`** = `g.theta * dt` (time step with implicit parameter)

## Transport Coefficients

### Diffusivities
- `chi_i`, `chi_e` = thermal diffusivities [m²/s] at faces
- `chi_neo_i`, `chi_neo_e` = neoclassical thermal diffusivities
- `chi_n` = particle diffusivity [m²/s] at faces
- `chi_neo_n` = neoclassical particle diffusivity

### Convection
- `v_i`, `v_e` = heat convection [m²/s] at faces
- `v_n` = particle convection [m/s] at faces  
- `v_neo_n` = neoclassical particle convection

Pattern: Add `_neo` suffix for neoclassical contribution, then sum with turbulent.

## Physics Quantities

### Short Names (Following Physics Convention)
- `q` = safety factor
- `ft` = trapped particle fraction (`f_trap` in global)
- `nue`, `nui` = collisionality parameters (nu_e_star, nu_i_star)
- `pe`, `pi` = electron/ion pressure [Pa]
- `alph` = alpha parameter (avoid `alpha` conflict)
- `eps` = inverse aspect ratio (ε)
- `sigma` = plasma conductivity [S/m]

### Model I/O
- `features` = QLKNN input array (shape: n_faces × 10)
- `out` = QLKNN output dictionary

## Source Terms

Pattern: `src_*` for sources in main loop:
```python
src_i = g.src_i_ext + si_fus * g.geo_vpr + g.src_i_ped
src_e = g.src_e_ext + se_fus * g.geo_vpr + g.src_e_ped  
src_p = -(j_bs + g.src_p_ext) * g.source_p_coeff
# g.src_n is precomputed and constant
```

Global precomputed sources (in `g.*`):
- `g.src_i_ext` = external ion heating
- `g.src_e_ext` = external electron heating
- `g.src_p_ext` = external current drive
- `g.src_n` = constant particle sources (gas puff + pellets + pedestal)
- `g.src_i_ped` = pedestal feedback (ion)
- `g.src_e_ped` = pedestal feedback (electron)

Temporary intermediate sources use descriptive names: `si_fus`, `se_fus`, `j_bs`

## Coupling Terms

Electron-ion heat exchange returns 4 diagonal vectors:
```python
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...)
```

- `qei_ii` = ion self-coupling (implicit cooling)
- `qei_ee` = electron self-coupling (implicit cooling)
- `qei_ie` = ion ← electron heat transfer
- `qei_ei` = electron ← ion heat transfer

Energy conserving: `qei_ie + qei_ei = 0`, `qei_ii + qei_ee = 0`

## Transient Coefficients

Time derivative scaling:
- `tc_in` = coefficient inside time derivative (size 100)
- `tc_out` = coefficient outside time derivative (size 100)
- `tc` = combined `1/(tc_out * tc_in)` (size 100)
- `tc_in_old` = saved `tc_in` from previous iteration for RHS
- `tc_prev` = `tc_in` if first iteration, else `tc_in_old`

## Temporary Variables

### Length Scale Ratios
```python
lti, lte, lne = safe_lref(...)  # Normalized gradient lengths
lni0, lni1 = ...                # Ion/impurity gradient lengths
```

### Angles and Upwind
```python
la, ra = peclet_to_alpha(...)   # Left/right upwind factors
lv, rv = v_f[:-1], v_f[1:]      # Left/right velocities
```

### Tridiagonal Building
```python
lower, diag, upper, vec = ...   # Subdiagonal, diagonal, superdiagonal, RHS vector
```

### Time Integration Helpers
```python
θdt = g.theta * dt              # Implicit parameter × time step
tc_in_old = tc_in               # Saved transient for next iteration
```

## Global Namespace `g.*`

### Physics Constants (Immutable)
```python
g.keV_to_J = 1.602e-16      # Conversion factor
g.keV_m3_to_Pa = 1.6e-16    # n*T → pressure
g.m_amu, g.q_e, g.m_e       # Particle properties
g.epsilon_0, g.mu_0         # EM constants
```

### Geometric Quantities (From Equilibrium)
```python
g.geo_vpr, g.geo_vpr_face           # ∂V/∂ρ (volume derivative)
g.geo_spr, g.geo_spr_face           # ∂A/∂ρ (area derivative)  
g.geo_F, g.geo_F_face               # R*B_φ
g.geo_Phi, g.geo_Phi_face           # Toroidal flux
g.geo_rmid_face                      # Midplane radius
g.geo_g2g3_over_rhon_face           # Metric for psi diffusion
g.geo_g0_over_vpr_face              # Convection factor
g.geo_g1_over_vpr_face              # Diffusion normalization
g.geo_g1_over_vpr2_face             # Particle transport factor
g.geo_g1_keV                         # g1 * keV_to_J
g.geo_epsilon_face                   # Inverse aspect ratio
g.geo_delta_face                     # Triangularity
g.geo_rho_b, g.geo_rho_face         # Minor radius
```

### Precomputed Operators (Sparse Structure)
```python
g.I_i, g.I_e, g.I_n, g.I_p          # Interpolation (26×25)
g.D_i, g.D_e, g.D_n, g.D_p          # Gradient ρ (26×25)
g.D_i_r, g.D_e_r, g.D_n_r           # Gradient r (26×25)
g.I_j, g.D_j, g.D_j_r               # Main ion operators
g.I_z, g.D_z_r                      # Impurity operators
g.b_i_f, g.b_e_f, ...               # Face boundary vectors (26)
g.b_i_g, g.b_e_g, ...               # Gradient boundary vectors (26)
g.b_i_r, g.b_e_r, ...               # r_mid boundary vectors (26)
```

### Configuration
```python
g.n = 25                    # Number of cells
g.dx = 1/25                 # Cell width
g.inv_dx = 25               # 1/dx
g.inv_dx_sq = 625           # 1/dx²
g.face_centers              # 0, 0.04, 0.08, ..., 1.0 (26)
g.cell_centers              # 0.02, 0.06, ..., 0.98 (25)
```

### Precomputed Sources
```python
g.src_i_ext         # External heating (ion)
g.src_e_ext         # External heating (electron)
g.src_p_ext         # External current drive
g.src_n             # Particle sources (gas puff + pellets + pedestal)
g.src_i_ped         # Pedestal control (ion)
g.src_e_ped         # Pedestal control (electron)
```

### Pedestal Control Matrices
```python
g.ped_i             # Temperature implicit control (ion)
g.ped_e             # Temperature implicit control (electron)
g.ped_n             # Density implicit control
```

These are diagonal matrices added to `A_ii`, `A_ee`, `A_nn` respectively.

### Boundary Conditions
```python
g.bc_i = (None, 0.2, 0.0, 0.0)      # (left_face, right_face, left_grad, right_grad)
g.bc_e = (None, 0.2, 0.0, 0.0)
g.bc_p = (None, None, 0.0, dp_edge)
g.bc_n = (None, 0.25e20, 0.0, 0.0)
```

### Time Integration
```python
g.theta = 1.0       # Implicit parameter (1=backward Euler, 0.5=Crank-Nicolson)
g.dt = 0.2          # Time step [s]
g.t_end = 5.0       # End time [s]
g.n_corr = 1        # Number of corrector iterations
```

## Function Return Patterns

### Single Values
```python
chiGB = calculate_bohm_normalization(...)
sigma = neoclassical_conductivity(...)
```

### Tuple Returns (Ordered by Type)
```python
lower, diag, upper, b = transport(v, d, bc)        # Tridiagonal vectors + RHS
lower, diag, upper, b = diff_terms(d, bc)          # Tridiagonal vectors + RHS
lower, diag, upper, b = conv_terms(v, d, bc)       # Tridiagonal vectors + RHS
si_fus, se_fus = fusion_source(e, i_f, j_f)        # Ion, electron
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...) # 4 coupling terms (diagonal vectors)
k1, k2 = z_avg(symbols, T_e, fractions)            # ⟨Z⟩, ⟨Z²⟩
pred_i, pred_e = solve_implicit_coupled_2x2(...)   # Coupled solution
```

### Transport Returns (Velocities, then Diffusivities)
```python
# Turbulent: velocity first (only v_n non-zero for turbulence), then diffusivities
v_n, chi_i, chi_e, chi_n = turbulent_transport(...)

# Neoclassical: velocities first, then diffusivities
v_i, v_e, v_neo_n, chi_neo_i, chi_neo_e, chi_neo_n = neoclassical_transport(...)
```

Order: velocities (i, e, n), then diffusivities (i, e, n)

### Ion Function (8-element tuple)
```python
j, z, k, k_f, w, u_f, j_bc, z_bc = ions(n_e, T_e, T_e_face)
```

## Matrix Variable Patterns

### Operator Matrices (Static)
- Capital letter + variable: `I_i`, `D_i`, `D_i_r`
- Pattern: `{Operator}_{variable}[_{coordinate}]`

### Transport Operators (Dynamic - Tridiagonal Vectors)
- Tridiagonal vectors: `lower_Ai`, `diag_Ai`, `upper_Ai` (size 25)
- Lowercase for RHS: `b_i`, `b_e`, `b_p`, `b_n` (size 25)

Pattern: `{position}_A{variable}` where position is `lower`, `diag`, or `upper`

### Conceptual Block Structure (Not Materialized)

Conceptually represents a 4×4 block system, but **never assembled**:
- Ion-electron coupling solved via `solve_implicit_coupled_2x2` (tridiagonal 2×2 blocks)
- Psi solved via `solve_implicit_tridiag` (standard tridiagonal)
- Density solved via `solve_implicit_tridiag` (standard tridiagonal)

### Utilities
```python
g.zero_vec  # (25,) zeros
g.ones_vec  # (25,) ones  
g.ones_vpr  # (25,) ones
g.v_p_zero  # (26,) zeros for psi velocity (faces)
```

Note: `g.zero` (25×25 zero matrix) and `g.identity` (100×100) were removed since full matrices are no longer assembled.

## Abbreviations

### Physics
- `q` = safety factor (not `q_f` to match physics convention)
- `ft` = trapped fraction (from `g.f_trap`)
- `qei` = electron-ion heat exchange (lowercase)
- `chi` = thermal diffusivity (χ)
- `sigma` = conductivity (σ)
- `eps` = small number OR inverse aspect ratio (context-dependent)
- `alph` = alpha parameter (avoid numpy conflict)

### Geometry
- `vpr` = ∂V/∂ρ (volume prime)
- `spr` = ∂A/∂ρ (surface prime)
- `rmid` = midplane radius
- `Phi` = toroidal flux (Φ)
- `u_f` = Z_eff (effective charge at faces)

### Grids/Indices
- `rc` = record size for binary I/O (101 = 1 + 4×25)
- `sz` = float64 size in bytes (8)
- `nt` = number of time steps

### Model
- `features` = QLKNN input array
- `out` = QLKNN output dictionary
- `lti, lte, lne, lni0, lni1` = normalized gradient scale lengths (L_ref/L_var)

## Array Construction

Use `np.r_[]` or `jnp.r_[]` for row stacking:
```python
x = jnp.r_[a, b, c]           # Stack arrays
x = jnp.r_[0.5, ones, 0.5]    # Scalars automatically broadcast
```

Use `jnp.c_[]` for column stacking:
```python
features = jnp.c_[lti, lte, lne, lni0, q, smag, x, TiTe, nu_star, j_f / n_f]
```

## Variable Scope

### Function-local
Short, descriptive names: `la`, `ra`, `lv`, `rv`, `diag`, `off`, `vec`

### Module-level (via `g.*`)
Fully qualified: `g.src_i_ext`, `g.geo_vpr_face`

### Return values
Match caller expectation: `lower, diag, upper, b` for transport operators, `v_n, chi_i, chi_e, chi_n` for turbulent transport

## Time Loop Variables

```python
state       # Current accepted state (100,)
pred        # Predictor/corrector working state (100,)
tc_in_old   # Saved tc_in from previous iteration (None on first iteration)
dt          # Current timestep [s]
t           # Current time [s]
pred_i      # Ion temperature solution from coupled solver (25,)
pred_e      # Electron temperature solution from coupled solver (25,)
sol_p       # Poloidal flux solution from tridiagonal solver (25,)
sol_n       # Density solution from tridiagonal solver (25,)
```

## Key Design Principles

1. **Single-letter suffixes**: `_f`, `_g`, `_r` instead of `_face`, `_grad`, `_grad_r`
2. **Single-letter ions**: `j`, `z`, `k`, `w`, `u_f` for ion quantities
3. **Lowercase coupling**: `qei_*` not `Qei_*`
4. **Consistent ordering**: i, e, p, n everywhere (functions, returns, blocks)
5. **Tridiagonal vectors**: `lower_A*`, `diag_A*`, `upper_A*` instead of full matrices
6. **Specialized solvers**: Never assemble full 100×100 system, use tridiagonal solvers
7. **No abbreviations in `g.*`**: Full names for globals (`g.src_i_ext` not `g.si`)
8. **Descriptive intermediates**: `si_fus`, `se_fus`, `j_bs` for clarity in physics
9. **O(n) complexity**: Exploit tridiagonal structure for linear scaling with grid size
