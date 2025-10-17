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

### Transport Matrices (Per-Iteration)
- **`A_*`** = transport operator matrix (`n × n`), e.g., `A_i`, `A_e`, `A_n`
- **`b_*`** = transport RHS vector (size `n`), e.g., `b_i`, `b_e`, `b_n`

Built from: `A, b = transport(v_f, d_f, bc)`

### Block Matrix Structure

The implicit solve uses textbook linear algebra form: `M @ x_new = rhs`

```python
# Build block components
A_ii = A_i + jnp.diag(qei_ii + g.ped_i)
A_ie = jnp.diag(qei_ie)
A_ei = jnp.diag(qei_ei)
A_ee = A_e + jnp.diag(qei_ee + g.ped_e)
A_pp = g.A_p
A_nn = A_n + jnp.diag(g.ped_n)

# Assemble spatial operator
A = jnp.block([
    [A_ii,    A_ie,    g.zero, g.zero],
    [A_ei,    A_ee,    g.zero, g.zero],
    [g.zero,  g.zero,  A_pp,   g.zero],
    [g.zero,  g.zero,  g.zero, A_nn  ]
])

# Assemble RHS
b = jnp.r_[b_i + src_i, b_e + src_e, g.b_p + src_p, b_n + g.src_n]

# Time coefficients
tc = 1 / (tc_out * tc_in)

# System matrix and RHS
M = g.identity - dt * g.theta * jnp.expand_dims(tc, 1) * A
rhs = (tc_prev / tc_in) * state + g.theta * dt * tc * b

# Solve
pred = jnp.linalg.solve(M, rhs)
```

Components:
- **`A_ii`, `A_ee`** = diagonal blocks with transport + coupling + pedestal
- **`A_ie`, `A_ei`** = off-diagonal electron-ion coupling
- **`A_pp`, `A_nn`** = decoupled psi and density blocks
- **`A`** = full spatial operator (100×100)
- **`M`** = system matrix (identity - dt*theta*tc*A)
- **`rhs`** = right-hand side vector
- **`b`** = spatial RHS before time scaling

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
- `tc_prev` = saved from previous iteration for RHS

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

### Matrix Building
```python
diag, off, vec = ...  # Diagonal, off-diagonal, RHS vector
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
A, b = transport(v, d, bc)                         # Matrix, vector
diff_mat, diff_vec = diff_terms(d, bc)             # Matrix, vector
conv_mat, conv_vec = conv_terms(v, d, bc)          # Matrix, vector
si_fus, se_fus = fusion_source(e, i_f, j_f)        # Ion, electron
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(...) # 4 coupling terms
k1, k2 = z_avg(symbols, T_e, fractions)            # ⟨Z⟩, ⟨Z²⟩
```

### Transport Returns (Velocities, then Diffusivities)
```python
# Turbulent: diffusivities only (v=0 for turbulence)
chi_i, chi_e, chi_n, v_n = turbulent_transport(...)

# Neoclassical: velocities first, then diffusivities
v_i, v_e, chi_neo_i, chi_neo_e, chi_neo_n, v_neo_n = neoclassical_transport(...)
```

Order: ion, electron, particle

### Ion Function (8-element tuple)
```python
j, z, k, k_f, w, u_f, j_bc, z_bc = ions(n_e, T_e, T_e_face)
```

## Matrix Variable Patterns

### Operator Matrices (Static)
- Capital letter + variable: `I_i`, `D_i`, `D_i_r`
- Pattern: `{Operator}_{variable}[_{coordinate}]`

### Transport Matrices (Dynamic)
- Capital letter: `A_i`, `A_e`, `A_p`, `A_n`
- Lowercase for vectors: `b_i`, `b_e`, `b_p`, `b_n`

### Block Components
- **`A_ii`, `A_ie`, `A_ei`, `A_ee`** = 2×2 ion-electron coupled block
- **`A_pp`** = psi block (decoupled)
- **`A_nn`** = density block (decoupled)
- **`A`** = full spatial operator (100×100)
- **`M`** = system matrix for implicit solve
- **`b`** = global RHS vector

### Block Utilities
```python
g.zero      # (25, 25) zeros (was g.zero_block)
g.zero_vec  # (25,) zeros
g.ones_vec  # (25,) ones  
g.ones_vpr  # (25,) ones
g.identity  # (100, 100) identity
```

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
Match caller expectation: `A, b` for matrix operators, `chi_i, chi_e, chi_n, v_n` for transport

## Time Loop Variables

```python
state      # Current accepted state (100,)
pred       # Predictor/corrector working state (100,)
tc_prev    # Saved transient from previous iteration
dt         # Current timestep [s]
t          # Current time [s]
```

## Key Design Principles

1. **Single-letter suffixes**: `_f`, `_g`, `_r` instead of `_face`, `_grad`, `_grad_r`
2. **Single-letter ions**: `j`, `z`, `k`, `w`, `u_f` for ion quantities
3. **Lowercase coupling**: `qei_*` not `Qei_*`
4. **Consistent ordering**: i, e, p, n everywhere (functions, returns, blocks)
5. **Textbook linear algebra**: `M @ x = rhs` instead of cryptic variable names
6. **No abbreviations in `g.*`**: Full names for globals (`g.src_i_ext` not `g.si`)
7. **Descriptive intermediates**: `si_fus`, `se_fus`, `j_bs` for clarity in physics
