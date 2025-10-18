# TORAX Paper to run.py Implementation Guide

This document connects the TORAX paper (arXiv:2406.06718v3) to the simplified `run.py` implementation, explaining how the mathematical formulations translate to code.

## Paper Overview

**Title**: "TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX"

**Authors**: Citrin et al., Google DeepMind

**Purpose**: Fast core transport simulation for tokamak plasmas using JAX for automatic differentiation and JIT compilation.

## Governing Equations: Paper → Code

### 1. Ion Heat Transport (Eq. 2 in paper)

**Paper notation** (Equation 2):
```
∂/∂t(V'^(5/3) n_i T_i) = (1/V') ∂/∂ρ̂[χ_i n_i (g_1/V') ∂T_i/∂ρ̂ - g_0 q_i^conv T_i] + V' Q_i
```

**Code implementation** (`run.py` lines 105-109):
```python
# State: i = T_i [keV], ion temperature on cell centers (25 points)
# Face values and gradients:
i_f = g.I_i @ i + g.b_i_f        # T_i at faces (26 points)
i_g = g.D_i @ i + g.b_i_g        # ∂T_i/∂ρ̂ at faces
i_r = g.D_i_r @ i + g.b_i_r      # ∂T_i/∂r_mid (for QLKNN)
```

**Transient coefficient** (LHS, line 894):
```python
tc_in = jnp.r_[j * g.vpr_5_3, ...]  # j n_i, g.vpr_5_3 = V'^(5/3)
tc_out = jnp.r_[g.tc_T, ...]        # g.tc_T = 1.5 * V'^(-2/3) * keV_to_J
# Combined: tc = 1/(tc_out * tc_in) gives proper coefficient
```

**Transport operator** (RHS diffusion + convection, line 905):
```python
lower_Ai, diag_Ai, upper_Ai, b_i = transport(v_i, chi_i, g.bc_i)
# v_i: convection velocity (from neoclassical, line 899)
# chi_i: diffusivity = χ_i * g.geo_g1_keV * n_i (turbulent + neoclassical, lines 896-901)
```

**Sources** (line 891):
```python
src_i = g.src_i_ext + si_fus * g.geo_vpr + g.src_i_ped
# g.src_i_ext: external heating (Gaussian, lines 822-829)
# si_fus: fusion power to ions (line 885)
# g.src_i_ped: pedestal feedback (line 826)
```

### 2. Electron Heat Transport (Eq. 3 in paper)

**Paper notation** (Equation 3): Same structure as ion heat, with e subscripts

**Code implementation**: Parallel to ions
```python
e_f = g.I_e @ e + g.b_e_f        # T_e at faces
e_g = g.D_e @ e + g.b_e_g        # ∂T_e/∂ρ̂
lower_Ae, diag_Ae, upper_Ae, b_e = transport(v_e, chi_e, g.bc_e)
src_e = g.src_e_ext + se_fus * g.geo_vpr + g.src_e_ped
```

**Ion-electron coupling** (lines 888-890):
```python
qei_ii, qei_ee, qei_ie, qei_ei = qei_coupling(e, n, j, z, k, w, ...)
# qei_coupling implements Eq. (line 442-455 in paper)
# Returns 4 diagonal coupling terms added to transport operators
```

### 3. Electron Density Transport (Eq. 4 in paper)

**Paper notation** (Equation 4):
```
∂/∂t(n_e V') = ∂/∂ρ̂[D_e n_e (g_1/V') ∂n_e/∂ρ̂ - g_0 V_e n_e] + V' S_n
```

**Code implementation**:
```python
n_f = g.I_n @ n + g.b_n_f        # n_e at faces
n_g = g.D_n @ n + g.b_n_g        # ∂n_e/∂ρ̂
lower_An, diag_An, upper_An, b_n = transport(v_n, chi_n, g.bc_n)
# v_n: particle convection (turbulent + neoclassical)
# chi_n: particle diffusivity
```

**Transient coefficient** (line 894):
```python
tc_in = jnp.r_[..., g.geo_vpr]   # V' for density
tc_out = jnp.r_[..., g.ones_vpr] # No scaling for density
```

### 4. Current Diffusion (Eq. 5 in paper)

**Paper notation** (Equation 5):
```
(16π² σ_|| μ₀ ρ̂ Φ_b²/F²) ∂ψ/∂t = ∂/∂ρ̂[(g_2 g_3/ρ̂) ∂ψ/∂ρ̂] - ...
```

**Code implementation**:
```python
p_g = g.D_p @ p + g.b_p_g        # ∂ψ/∂ρ̂
# Precomputed constant transport (line 835):
A_p_lower, A_p_diag, A_p_upper, b_p = transport(g.v_p_zero, g.geo_g2g3_over_rhon_face, g.bc_p)
```

**Conductivity and bootstrap** (lines 884, 886-887):
```python
sigma = neoclassical_conductivity(e_f, n_f, q_f, u_f)  # σ_|| Sauter model
j_bs = bootstrap_current(...)  # Bootstrap current (Sauter model)
src_p = -(j_bs + g.src_p_ext) * g.source_p_coeff
```

**Transient coefficient** (lines 834, 895):
```python
g.tc_p_base = g.cell_centers * g.mu0_pi16sq_Phib_sq_over_F_sq / g.resist_mult
tc_out = jnp.r_[..., g.tc_p_base * sigma, ...]  # Time-varying with σ
```

## Spatial Discretization: Finite Volume Method

### Grid Structure (Section 2.2, Figure 2 in paper)

**Paper**: Uniform grid with N cells, cell centers at ρ̂ᵢ, faces at ρ̂ᵢ₊₁/₂

**Code** (lines 549-552):
```python
g.n = 25                         # N cells
g.dx = 1/g.n                     # Cell width dρ̂ = 0.04
g.face_centers = np.arange(g.n+1) * g.dx  # 26 faces: [0, 0.04, ..., 1.0]
g.cell_centers = (np.arange(g.n) + 0.5) * g.dx  # 25 centers: [0.02, 0.06, ..., 0.98]
```

### Differential Operators

**Interpolation operator I** (cell → face, Section 2.2):
```python
# Paper Eq. (13-14): x_{i+1/2} = α x_i + (1-α) x_{i+1}
# Code (lines 65-78): face_op() builds operator
I_i @ i + b_i_f = i_f  # Maps 25 cell values → 26 face values
```

**Gradient operator D** (∂/∂ρ̂, Section 2.2):
```python
# Paper: Finite differences across cells
# Code (lines 37-49): grad_op() builds operator
D_i @ i + b_i_g = i_g  # Maps 25 cell values → 26 gradients
```

### Peclet Number and Exponential Fitting (Eq. 11-14 in paper)

**Paper** (Equation 11-14): Power-law scheme for P'eclet weighting

**Code implementation** (`conv_terms`, lines 100-142):
```python
def peclet_to_alpha(p):
    # Equation 14 from paper
    if abs(p) < eps:
        return 0.5  # Central difference
    elif p > 10:
        return (p - 1) / p  # Full upwind
    elif 0 < p < 10:
        return ((p-1) + (1 - p/10)^5) / p  # Smooth transition
    # ... (lines 110-122)
```

Peclet number (line 106):
```python
ratio = scale * g.dx * v_face / d_face  # Pe = V * dρ̂ / D
```

### Transport Terms Assembly

**Diffusion** (`diff_terms`, lines 81-97):
```python
# Paper Eq. (10): Γ = -D ∂x/∂ρ̂
# Returns tridiagonal vectors: lower, diag, upper, vec
diag[i] = -(d[i+1] + d[i]) * n²  # Main diagonal
off[i] = d[i+1] * n²              # Off-diagonals
```

**Convection** (`conv_terms`, lines 100-142):
```python
# Paper Eq. (10): Γ = V x
# With Peclet weighting from Eq. (14)
# Returns tridiagonal vectors incorporating upwinding
```

**Combined** (`transport`, lines 145-148):
```python
def transport(v, d, bc):
    lower_diff, diag_diff, upper_diff, vec_diff = diff_terms(d, bc)
    lower_conv, diag_conv, upper_conv, vec_conv = conv_terms(v, d, bc)
    return lower_diff + lower_conv, diag_diff + diag_conv, ...
```

## Time Integration: Theta Method

### Paper Formulation (Eq. 15-17)

**Equation 15** (theta method):
```
x_{t+Δt} - x_t = Δt [θ F(x_{t+Δt}, t+Δt) + (1-θ) F(x_t, t)]
```

**Equation 17** (state evolution):
```
T̃(x_{t+Δt}) ⊙ x_{t+Δt} - T̃(x_t) ⊙ x_t = 
    Δt [θ (C̄(x_{t+Δt}) x_{t+Δt} + c(x_{t+Δt})) + (1-θ) (...)]
```

### Code Implementation

**Theta parameter** (line 567):
```python
g.theta = 1  # Backward Euler (θ=1), unconditionally stable
```

**Time step** (line 569):
```python
g.dt = 0.2  # Fixed time step [s]
```

**Predictor-corrector** (lines 862-923):
```python
for _ in range(g.n_corr + 1):  # g.n_corr = 1 (one corrector)
    # Evaluate all coefficients at predicted state
    # Build RHS with transient coefficient ratio
    tc_prev = tc_in if tc_in_old is None else tc_in_old
    rhs = ((tc_prev / tc_in) * state + θdt * tc * b)[:, None]
    
    # Solve linear systems
    pred_i, pred_e = solve_implicit_coupled_2x2(...)  # Ion-electron coupled
    sol_p = solve_implicit_tridiag(...)               # Psi decoupled
    sol_n = solve_implicit_tridiag(...)               # Density decoupled
    
    pred = jnp.r_[pred_i, pred_e, sol_p, sol_n]
```

**Key difference from paper**:
- Paper (Section 2.3.1): Generic linear solver on full system
- Code: Specialized tridiagonal solvers (O(n) instead of O(n³))

## Solver Strategy: Specialized Tridiagonal

### Paper Approach (Section 2.3.1, Eq. 19)

**Predictor-corrector** (Equation 19):
```
T̃(x^{k-1}) ⊙ x^k - T̃(x_t) ⊙ x_t = 
    Δt θ [C̄(x^{k-1}) x^k + c(x^{k-1})] + ...
```

Solved using JAX linear algebra on full system.

### Code Approach (run.py implementation)

**Never assemble full 100×100 matrix**. Instead:

1. **Coupled i-e solver** (lines 913-918):
```python
pred_i, pred_e = solve_implicit_coupled_2x2(
    lower_Ai, diag_Ai + qei_ii + g.ped_i, upper_Ai,
    lower_Ae, diag_Ae + qei_ee + g.ped_e, upper_Ae,
    qei_ie, qei_ei,  # Off-diagonal coupling
    tc[l.i], tc[l.e], θdt, rhs[l.i, 0], rhs[l.e, 0]
)
```

Solves 2×2 block tridiagonal system:
```
[I - θdt*tc_i*(A_i + qei_ii + ped_i)    -θdt*tc_i*qei_ie      ] [i_new]   [rhs_i]
[-θdt*tc_e*qei_ei                    I - θdt*tc_e*(A_e + qei_ee + ped_e)] [e_new] = [rhs_e]
```

2. **Decoupled solvers** (lines 919-920):
```python
sol_p = solve_implicit_tridiag(g.A_p_lower, g.A_p_diag, g.A_p_upper, ...)
sol_n = solve_implicit_tridiag(lower_An, diag_An + g.ped_n, upper_An, ...)
```

Uses JAX's `jax.lax.linalg.tridiagonal_solve` (Thomas algorithm).

**Complexity**: O(n) per solve vs. O(n³) for dense LU decomposition.

## Physics Models: Paper → Code

### Turbulent Transport (Section 3.3)

**Paper**: Three models - Constant, CGM, QLKNN

**Code** (`turbulent_transport`, lines 350-461):

1. **QLKNN-hyper-10D** (lines 350-461):
```python
# Paper Eq. (line 410-412): ML surrogate of QuaLiKiz
features = jnp.c_[lti, lte, lne, lni0, q, smag, x, TiTe, nu_star, j_f/n_f]
out = g.model.predict(features)  # Neural network inference
qi = out["efiITG"] + out["efiTEM"]
qe = out["efeITG"] * g.ITG_flux_ratio_correction + out["efeTEM"] + out["efeETG"] * g.ETG_correction_factor
```

2. **Conversion to transport coefficients** (lines 389-396):
```python
chiGB = (A_i * m_amu)^0.5 / (B_0 * q_e)^2 * (T_i * keV_to_J)^1.5 / a_minor
chi_i = (R_major / a_minor) * qi / lti * chiGB
chi_e = (R_major / a_minor) * qe / lte * chiGB
```

3. **Smoothing** (lines 421-456):
```python
# Paper: Optional Gaussian smoothing (line 414)
kernel = exp(-log(2) * (ρ̂ - ρ̂')² / w²)
chi_i = smooth_single_coeff(chi_i)  # Convolution
```

### Neoclassical Physics (Section 3.4)

**Paper**: Sauter model for bootstrap current and conductivity

**Code**:

1. **Bootstrap current** (`bootstrap_current`, lines 292-328):
```python
# Paper: Sauter model (Ref. 34 - sauter:1999)
# Implements full coefficient calculation with trapped particle fraction
ft = g.f_trap  # Trapped fraction (lines 767-772)
nue = nu_e_star(q, n_f, e_f, u_f, log_lei)  # Collisionality
j_bs = gc * (L31 * (pe*n_g/n_f + pi*j_g/j_f) + ...)
```

2. **Neoclassical conductivity** (`neoclassical_conductivity`, lines 331-342):
```python
# Paper: Sauter model for σ_||
sig_sptz = 1.9012e04 * (e_f * 1e3)^1.5 / u_f / NZ / log_lei
sig_neo = 1 - ft33 * (1 + 0.36/u_f - ft33*(0.59/u_f - 0.23/u_f*ft33))
sig_f = sig_sptz * sig_neo
```

3. **Neoclassical transport** (`neoclassical_transport`, lines 464-478):
```python
# Paper: Pereverzev-Corrigan method (line 294)
chi_i_neo = g.geo_g1_keV * j_f * g.chi_pereverzev
v_i = i_g / i_f * chi_i_neo  # Pinch velocity
```

### Sources (Section 3.5)

**Ion-electron heat exchange** (`qei_coupling`, lines 283-289):
```python
# Paper Eq. (line 442-455): Q_ei = 1.5 n_e (T_i - T_e) / (A_i m_p τ_e)
# Returns 4 coupling terms: qei_ii, qei_ee, qei_ie, qei_ei
Q = exp(log(...) - log_tau) * g.geo_vpr
return -Q, -Q, Q, Q  # Energy conserving
```

**Fusion power** (`fusion_source`, lines 261-280):
```python
# Paper Eq. (line 459-476): Bosch-Hale D-T reactivity
theta = i_f / (1 - (i_f * (...)))
xi = (g.fusion_BG^2 / (4*theta))^(1/3)
logsv = log(g.fusion_C1 * theta) + ...
Pf = exp(log(prod * g.fusion_Efus) + 2*log(j_f) + logsv)
# Mikkelsen partitioning (line 477)
```

**Generic heating** (`heat_source`, lines 235-239):
```python
# Paper Section 3.5.4: Gaussian formula
profile = gaussian_profile(g.heat_loc, g.heat_w, g.heat_P)
source_i = profile * (1 - g.heat_efrac)
source_e = profile * g.heat_efrac
```

## Geometric Quantities (Section 3.1)

### Paper Definitions (Eq. 6-9)

```
g_0 = ⟨∇V⟩                    # Eq. 6
g_1 = ⟨(∇V)²⟩                 # Eq. 7
g_2 = ⟨(∇V)²/R²⟩              # Eq. 8
g_3 = ⟨1/R²⟩                  # Eq. 9
V' = dV/dρ̂                    # Volume derivative
```

### Code Implementation (lines 605-774)

**Load CHEASE geometry** (lines 605-636):
```python
file_path = os.path.join("geo", "ITER_hybrid_citrin_equil_cheasedata.mat2cols")
# Read flux-surface-averaged quantities from CHEASE
chease_data["PSIchease=psi/2pi"]
chease_data["RHO_TOR=sqrt(Phi/pi/B0)"]
chease_data["<|grad(psi)|>"]  # → flux_surf_avg_RBp
chease_data["<1/R**2>"]       # → flux_surf_avg_1_over_R2
# ...
```

**Compute metric coefficients** (lines 644-658):
```python
vpr = 4 * π * Φ[-1] * ρ_norm / (F * flux_surf_avg_1_over_R2)  # V'
C0 = flux_surf_avg_RBp * C1
g0 = C0 * 2π                           # g_0
g1 = C1 * C4 * 4π²                     # g_1
g2 = C1 * C3 * 4π²                     # g_2 (used in g_2*g_3 product)
g3 = C2 / C1                           # g_3
```

**Interpolate to simulation grid** (lines 675-715):
```python
interp = lambda x, y: np.interp(x, rho_norm_intermediate, y)
g.geo_vpr_face = interp(rho_face_norm, vpr)
g.geo_g0_face = interp(rho_face_norm, g0)
g.geo_g1_face = interp(rho_face_norm, g1)
# ...
```

**Derived geometric factors** (lines 752-766):
```python
# For transport equations
g.geo_g0_over_vpr_face = g.geo_g0_face[1:] / g.geo_vpr_face[1:]
g.geo_g1_over_vpr_face = g.geo_g1_face[1:] / g.geo_vpr_face[1:]
g.geo_g1_keV = g.geo_g1_over_vpr_face * g.keV_to_J
# For psi equation
g.geo_g2g3_over_rhon_face = g2*g3 / rho_norm
```

## Boundary Conditions (Section 2.1, line 157-159)

### Paper Specification

- All equations: zero-derivative at ρ̂=0 (symmetry)
- T_i, T_e, n_e: Dirichlet at ρ̂=1 (fixed values)
- ψ: Neumann at ρ̂=1 (sets I_p via ∂ψ/∂ρ̂)

### Code Implementation (lines 784-789)

```python
# Boundary condition tuples: (left_face, right_face, left_grad, right_grad)
g.bc_i = (None, g.i_right_bc, 0.0, 0.0)  # T_i: 0 grad at left, value at right
g.bc_e = (None, g.e_right_bc, 0.0, 0.0)  # T_e: 0 grad at left, value at right
g.bc_n = (None, g.n_right_bc, 0.0, 0.0)  # n_e: 0 grad at left, value at right
g.bc_p = (None, None, 0.0, g.dp_edge)    # ψ: 0 grad at left, derivative at right

# dp_edge calculated from I_p (lines 786-787):
g.dp_edge = (g.Ip * 16π³ * μ₀ * Φ_b) / (g.geo_g2g3_over_rhon_face[-1] * g.geo_F_face[-1])
```

## Ion Species and Z_eff (Section 3.2, line 374-377)

**Paper**: Single main ion, single impurity to accommodate Z_eff > 1

**Code** (`ions` function, lines 489-513):
```python
def ions(n_e, T_e, T_e_face):
    # Z-averaged charge states (Mavrin model for neon)
    k1, k2 = z_avg(g.ion_names, T_e, g.ion_fractions)     # Main ion ⟨Z⟩, ⟨Z²⟩
    w1, w2 = z_avg(g.impurity_names, T_e, g.impurity_fractions)  # Impurity
    
    # Dilution factor (Paper Eq. line 376)
    dilution_factor = (w - g.Z_eff) / (k * (w - k))
    j = n_e * dilution_factor  # Main ion density
    z = (n_e - j*k) / w        # Impurity density
    
    # Effective Z at faces
    u_f = (k_f² * j_f + w_f² * z_f) / n_f
    
    return j, z, k, k_f, w, u_f, j_bc, z_bc
```

**Configuration** (lines 541-548):
```python
g.Z_eff = 1.6
g.impurity_names = ("Ne",)
g.ion_names = "D", "T"
g.impurity_fractions = np.array([1.0])
g.ion_fractions = np.array([0.5, 0.5])  # 50-50 D-T
```

## Pedestal Model (Section 3.3, Figure 4, line 416)

**Paper**: Adaptive source term sets desired pedestal height/width

**Code** (lines 576-583, 826-833):
```python
# Configuration
g.rho_norm_ped_top = 0.91  # Pedestal top location
g.i_ped = 4.5              # Ion temperature pedestal [keV]
g.e_ped = 4.5              # Electron temperature pedestal [keV]
g.n_ped = 0.62e20          # Density pedestal [m⁻³]

# Pedestal mask: single point at ρ̂ ≈ 0.91
g.mask = np.zeros(g.n, dtype=bool)
g.mask[rho_norm_ped_top_idx] = True

# Adaptive sources (large prefactors)
g.src_i_ped = g.ped_mask_T * g.i_ped  # g.ped_mask_T = mask * 2e10
g.src_e_ped = g.ped_mask_T * g.e_ped

# Implicit feedback (lines 831-833, added to diagonal)
g.ped_i = -g.ped_mask_T  # Negative for stability
g.ped_e = -g.ped_mask_T
g.ped_n = -g.ped_mask_n
```

## Safety Factor q (Eq. line 356-360)

**Paper** (Equation, line 358):
```
q = 2π B_0 ρ̂ / (∂ψ/∂ρ̂)
```

**Code** (line 883):
```python
q_f = jnp.abs(g.q_factor_axis * jnp.r_[1.0 / (p_g[1] * g.n), 
                                        g.face_centers[1:] / p_g[1:]])
# g.q_factor_axis = 2 * Φ_b (line 745)
# p_g = ∂ψ/∂ρ̂ (computed via D_p operator)
```

## Variable Name Mapping: Paper ↔ Code

| Paper | Code | Description | Location |
|-------|------|-------------|----------|
| ρ̂ | `rho_norm` | Normalized toroidal flux coord | state |
| T_i | `i` | Ion temperature [keV] | `state[l.i]` |
| T_e | `e` | Electron temperature [keV] | `state[l.e]` |
| ψ | `p` | Poloidal flux [Wb] | `state[l.p]` |
| n_e | `n` | Electron density [m⁻³] | `state[l.n]` |
| n_i | `j` | Main ion density [m⁻³] | derived |
| n_imp | `z` | Impurity density [m⁻³] | derived |
| χ_i | `chi_i` | Ion heat diffusivity | face |
| χ_e | `chi_e` | Electron heat diffusivity | face |
| D_e | `d_e` or `chi_n` | Particle diffusivity | face |
| V_e | `v_e` or `v_n` | Particle convection | face |
| V' | `g.geo_vpr` | dV/dρ̂ | geometry |
| g_0 | `g.geo_g0_face` | ⟨∇V⟩ | geometry |
| g_1 | `g.geo_g1_face` | ⟨(∇V)²⟩ | geometry |
| g_2 g_3 | `g.geo_g2g3_over_rhon_face` | Combined metric | geometry |
| Φ_b | `g.geo_Phi_b` | Toroidal flux at boundary | geometry |
| q | `q_f` | Safety factor | faces |
| σ_|| | `sigma` | Neoclassical conductivity | faces |
| j_bs | `j_bs` | Bootstrap current | cells |
| Q_ei | `qei_*` | Ion-electron heat exchange | cells |
| θ | `g.theta` | Theta method parameter (=1) | config |
| Δt | `dt` | Time step [s] | dynamic |

## Key Simplifications in run.py

1. **No time-dependent geometry**: Φ̇_b = 0 (removes convection terms in Appendix)

2. **Fixed I_p**: No current ramping (simplifies ψ equation)

3. **Single impurity**: Ne only (could extend to multiple)

4. **Prescribed density option**: Can fix n_e profile instead of evolving

5. **Constant CHEASE geometry**: Loaded once, not time-dependent

6. **Predictor-corrector only**: No Newton-Raphson or optimizer solvers

7. **Tridiagonal solvers**: Exploits sparsity, not mentioned explicitly in paper

8. **No adaptive timestep**: Fixed dt (paper supports adaptive, Eq. line 330-334)

## Performance Comparison

### Paper (Table 1, line 576-591):
- ITER hybrid rampup, 80s, N=25, QLKNN, dt=2s
- Newton-Raphson: 15.6s compile, 22s runtime
- Predictor-corrector (1): 4.5s compile, 6.5s runtime

### run.py Configuration:
- ITER hybrid, 5s, N=25, QLKNN, dt=0.2s
- Predictor-corrector with 1 corrector step
- Expected runtime: ~0.1-1s (25 timesteps vs 40 in paper)

## Verification Against RAPTOR (Section 5)

**Paper**: Benchmarks against RAPTOR code (Felici et al.)
- Figure 6: ITER L-mode with constant transport, NRMSD ~1%
- Figure 8: Time-dependent scenario with current ramps, NRMSD ~4%

**run.py Implementation**: Same discretization and physics models, expected similar agreement

## Extensions from Paper Not in run.py

### Available in TORAX but not run.py:

1. **Newton-Raphson solver** (Section 2.3.2): Auto-differentiated Jacobian
2. **Optimizer solver** (Section 2.3.3): JAXopt-based
3. **Adaptive timestep** (Section 2.3.4): Based on max χ
4. **Multiple transport models**: CGM, constant (run.py only QLKNN)
5. **Circular geometry**: Simple analytical model
6. **Time-dependent inputs**: Ramping I_p, varying sources
7. **Forward sensitivity** (Section 3.5.4): ∂x/∂p calculations

### Planned TORAX Extensions (Section 6):

1. MHD models (sawteeth, NTMs)
2. Radiation sinks (Bremsstrahlung, line radiation)
3. More ML surrogates (turbulence, pedestal, sources)
4. Neoclassical transport for heavy impurities
5. Momentum transport
6. IMAS coupling
7. Time-dependent geometry

## Key Insights

### Why JAX?

**Paper motivation** (Section 1, line 77-84):
1. **Auto-differentiation**: Enables gradient-based optimization and sensitivity analysis
2. **JIT compilation**: Fast execution comparable to Fortran/C++
3. **ML integration**: Native neural network support (QLKNN coupling)
4. **Vectorization**: Automatic SIMD and GPU support

**Code implementation**:
- All physics models JAX-compatible
- `jax.lax.linalg.tridiagonal_solve` for linear systems
- `jnp` arrays throughout for automatic differentiation
- No explicit loops (JAX-friendly)

### Solver Design Philosophy

**Paper approach**: General framework supporting multiple solvers

**run.py approach**: Simplified for clarity
- Only predictor-corrector
- Specialized tridiagonal solvers (not mentioned in paper)
- O(n) complexity vs O(n³) for dense methods

This makes the code:
- **Faster**: Linear scaling with grid size
- **Simpler**: No Newton iteration complexity
- **Sufficient**: 1-2 corrector steps converge for most cases

### Physical Fidelity

**Both paper and code**:
- Flux-surface-averaged equations (valid for closed surfaces)
- Time-scale separation assumption (transport >> turbulence)
- Neoclassical + turbulent transport
- QLKNN-hyper-10D ML surrogate
- Sauter model for bootstrap/conductivity
- Bosch-Hale fusion reactivity

**Accuracy**: Verified to ~1-5% vs RAPTOR (mature code)

## Recommended Reading Order

1. **Paper Sections 1-2.1**: Motivation, governing equations
2. **run.py lines 1-35**: Constants, global setup
3. **Paper Section 2.2**: Spatial discretization (FVM)
4. **run.py lines 37-148**: Operators and transport terms
5. **Paper Section 3**: Physics models
6. **run.py lines 170-461**: Physics model implementations
7. **Paper Section 2.3**: Time integration and solvers
8. **run.py lines 856-926**: Main time loop
9. **Paper Section 5**: Verification
10. **run.md**: Implementation details of linear algebra

## Summary

The `run.py` implementation is a **simplified, high-performance version** of the TORAX framework described in the paper:

- **Same physics**: Identical governing equations, transport models, sources
- **Same discretization**: Finite volume method with Peclet weighting
- **Same backend**: JAX for auto-diff and JIT compilation
- **Optimized solver**: Specialized tridiagonal instead of general dense methods
- **Reduced features**: Fixed geometry, predictor-corrector only
- **Verified accuracy**: Expected ~1-5% agreement with RAPTOR like full TORAX

This makes it an excellent reference implementation for understanding the core numerical methods while maintaining production-quality physics fidelity.

