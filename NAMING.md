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
- **`_face`** = values at cell faces (n+1 points), e.g., `i_face`, `e_face`, `n_face`
- **`_grad`** = gradient w.r.t. normalized ρ (ρ_norm), e.g., `i_grad`, `e_grad`
- **`_grad_r`** = gradient w.r.t. midplane radius (r_mid), e.g., `i_grad_r`, `e_grad_r`

### Examples
```python
i = pred[l.i]                        # Cell center (25)
i_face = g.I_i @ i + g.b_i_face      # Faces (26)
i_grad = g.D_i @ i + g.b_i_grad      # Gradient ∂i/∂ρ (26)
i_grad_r = g.D_i_r @ i + g.b_i_grad_r  # Gradient ∂i/∂r (26)
```

Pattern: `{var}_{location}_{coord}` where coordinate is optional (defaults to ρ).

## Ion/Impurity Variables

Prefix convention for derived species:
- **`ions_`** prefix for tuple-unpacked ion quantities
- **`ni`** = main ion density (shorter than `n_i` for symmetry with `n`)
- **`nz`** = impurity density (z for high-Z impurities)

### Ion Tuple Unpacking
```python
(ions_n_i, ions_n_impurity, ions_Z_i, ions_Z_i_face, ions_Z_impurity,
 ions_Z_eff_face, ions_n_i_bc, ions_n_impurity_bc) = ions_update(n, e, e_face)
```

Components:
- `ions_n_i` = main ion density
- `ions_n_impurity` = impurity density
- `ions_Z_i` = main ion charge state
- `ions_Z_i_face` = main ion charge (faces)
- `ions_Z_impurity` = impurity charge state
- `ions_Z_eff_face` = effective charge (faces)
- `ions_n_i_bc` = ion density boundary conditions
- `ions_n_impurity_bc` = impurity density boundary conditions

### Derived Ion Quantities
```python
ni_face = g.I_ni @ ions_n_i + g.b_r * ions_n_i_bc[1]
ni_grad = g.D_ni_rho @ ions_n_i + ...
ni_grad_r = g.D_ni_rmid @ ions_n_i + ...
nz_face = g.I_nimp @ ions_n_impurity + ...
nz_grad_r = g.D_nimp_rmid @ ions_n_impurity + ...
```

## Matrix/Operator Naming

### Differential Operators (Precomputed)
- **`I_*`** = interpolation operator (cell → face), e.g., `g.I_i`, `g.I_e`, `g.I_n`, `g.I_p`
- **`D_*`** = gradient operator (ρ coords), e.g., `g.D_i`, `g.D_e`, `g.D_n`, `g.D_p`
- **`D_*_r`** = gradient operator (r_mid coords), e.g., `g.D_i_r`, `g.D_e_r`, `g.D_n_r`

All are `(n+1) × n` matrices.

### Boundary Vectors (Precomputed)
- **`b_*_face`** = face boundary contributions, e.g., `g.b_i_face`, `g.b_e_face`
- **`b_*_grad`** = gradient boundary contributions, e.g., `g.b_i_grad`, `g.b_p_grad`
- **`b_*_grad_r`** = r_mid gradient boundaries, e.g., `g.b_i_grad_r`, `g.b_e_grad_r`

All are size `n+1`.

### Transport Matrices (Per-Iteration)
- **`A_*`** = transport operator matrix (`n × n`), e.g., `A_i`, `A_e`, `A_n`
- **`b_*`** = transport RHS vector (size `n`), e.g., `b_i`, `b_e`, `b_n`

Built from: `A, b = trans_terms(v_face, d_face, bc)`

### Block Matrix
- **`spatial_mat`** = full 100×100 block-assembled transport operator
- **`spatial_vec`** = full 100-element RHS vector

## Transport Coefficients

### Diffusivities
- `chi_i`, `chi_e` = thermal diffusivities [m²/s] at faces
- `chi_neo_i`, `chi_neo_e` = neoclassical thermal diffusivities
- `D_n` = particle diffusivity [m²/s] at faces
- `D_neo_n` = neoclassical particle diffusivity

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

Pattern: `src_*` for sources after all transformations:
```python
src_i = g.source_i_external + si_fus * g.geo_vpr + g.source_i_adaptive
src_e = g.source_e_external + se_fus * g.geo_vpr + g.source_e_adaptive  
src_p = -(j_bs + g.source_p_external) * g.source_p_coeff
```

Temporary intermediate sources use descriptive names: `si_fus`, `se_fus`, `j_bs`

## Coupling Terms

Electron-ion heat exchange returns 4 diagonal vectors:
```python
Qei_ii, Qei_ee, Qei_ie, Qei_ei = qei_coupling(...)
```

- `Qei_ii` = ion self-coupling (implicit cooling)
- `Qei_ee` = electron self-coupling (implicit cooling)
- `Qei_ie` = ion ← electron heat transfer
- `Qei_ei` = electron ← ion heat transfer

Energy conserving: `Qei_ie + Qei_ei = 0`, `Qei_ii + Qei_ee = 0`

## Transient Coefficients

Time derivative scaling:
- `tc_in` = coefficient inside time derivative (size 100)
- `tc_out` = coefficient outside time derivative (size 100)
- `tc` = combined `1/(tc_out * tc_in)` (size 100)
- `tc_old` = saved from previous iteration for RHS

## Temporary Variables

### Length Scale Ratios
```python
lti, lte, lne = safe_lref(...)  # Normalized gradient lengths
lni0, lni1 = ...                # Ion/impurity gradient lengths
```

### Angles and Upwind
```python
la, ra = peclet_to_alpha(...)   # Left/right upwind factors
lv, rv = v_face[:-1], v_face[1:]  # Left/right velocities
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
g.b_i_face, g.b_e_face, ...         # Boundary vectors (26)
g.b_i_grad, g.b_e_grad, ...         # Boundary vectors (26)
g.b_i_grad_r, g.b_e_grad_r, ...     # Boundary vectors (26)
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
g.source_i_external         # External heating (ion)
g.source_e_external         # External heating (electron)
g.source_p_external         # External current drive
g.source_n_constant         # Particle sources (gas puff + pellets)
g.source_i_adaptive         # Pedestal control (ion)
g.source_e_adaptive         # Pedestal control (electron)
g.source_mat_adaptive_T     # Temperature implicit control
g.source_mat_adaptive_n     # Density implicit control
```

### Boundary Conditions
```python
g.bc_i = (None, 0.2, 0.0, 0.0)      # (left_face, right_face, left_grad, right_grad)
g.bc_e = (None, 0.2, 0.0, 0.0)
g.bc_p = (None, None, 0.0, dp_edge)
g.bc_n = (None, 0.25e20, 0.0, 0.0)
g.bcs = (g.bc_i, g.bc_e, g.bc_p, g.bc_n)
```

## Function Return Patterns

### Single Values
```python
chiGB = calculate_bohm_normalization(...)
sigma = neoclassical_conductivity(...)
```

### Tuple Returns (Ordered by Type)
```python
A, b = trans_terms(v, d, bc)                          # Matrix, vector
diff_mat, diff_vec = diff_terms(d, bc)                # Matrix, vector
conv_mat, conv_vec = conv_terms(v, d, bc)             # Matrix, vector
si_fus, se_fus = fusion_source(e, i_f, ni_f)          # Ion, electron
Qei_ii, Qei_ee, Qei_ie, Qei_ei = qei_coupling(...)    # 4 coupling terms
```

### Transport Returns (4-element)
```python
chi_i, chi_e, D_n, v_n = turbulent_transport(...)
v_i, v_e, chi_neo_i, chi_neo_e, D_neo_n, v_neo_n = neoclassical_transport(...)
```

Order: ion, electron, particle_D, particle_v

### Ion Update (8-element tuple)
```python
(ions_n_i, ions_n_impurity, ions_Z_i, ions_Z_i_face, ions_Z_impurity,
 ions_Z_eff_face, ions_n_i_bc, ions_n_impurity_bc) = ions_update(n, e, e_face)
```

## Matrix Variable Patterns

### Operator Matrices (Static)
- Capital letter + variable: `I_i`, `D_i`, `D_i_r`
- Pattern: `{Operator}_{variable}[_{coordinate}]`

### Transport Matrices (Dynamic)
- Capital letter: `A_i`, `A_e`, `A_p`, `A_n`
- Lowercase for vectors: `b_i`, `b_e`, `b_p`, `b_n`

### Block Utilities
```python
g.zero_block  # (25, 25) zeros
g.zero_vec    # (25,) zeros
g.ones_vec    # (25,) ones  
g.ones_vpr    # (25,) ones
g.identity    # (100, 100) identity
```

## Abbreviations

### Physics
- `q` = safety factor (not `q_face` to match physics convention)
- `ft` = trapped fraction (from `g.f_trap`)
- `Qei` = electron-ion heat exchange
- `chi` = thermal diffusivity (χ)
- `sigma` = conductivity (σ)
- `eps` = small number OR inverse aspect ratio (context-dependent)
- `alph` = alpha parameter (avoid numpy conflict)

### Geometry
- `vpr` = ∂V/∂ρ (volume prime)
- `spr` = ∂A/∂ρ (surface prime)
- `rmid` = midplane radius
- `Phi` = toroidal flux (Φ)
- `Zeff` = effective charge (Z_eff)

### Grids/Indices
- `nc` = num_cells (25)
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
```

Use `jnp.c_[]` for column stacking:
```python
features = jnp.c_[lti, lte, lne, lni0, q, smag, x, TiTe, nu_star, normni]
```

## Variable Scope

### Function-local
Short, descriptive names: `la`, `ra`, `lv`, `rv`, `diag`, `off`, `vec`

### Module-level (via `g.*`)
Fully qualified: `g.source_i_external`, `g.geo_vpr_face`

### Return values
Match caller expectation: `A, b` for matrix operators, `chi_i, chi_e, D_n, v_n` for transport

## Time Loop Variables

```python
state      # Current accepted state (100,)
pred       # Predictor/corrector working state (100,)
tc_in_old  # Saved transient from previous iteration
dt         # Current timestep [s]
t          # Current time [s]
```

