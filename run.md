# Tokamak Plasma Transport Simulation - Numerical Implementation

## Overview

This code implements a 1D radial transport simulation for tokamak plasmas, solving coupled partial differential equations for ion temperature (`T_i`), electron temperature (`T_e`), electron density (`n_e`), and poloidal magnetic flux (`psi`). The simulation uses a finite volume method with implicit time stepping.

## Coordinate System and Grid

### Radial Coordinate
- **Primary coordinate**: `rho_norm` - normalized toroidal flux coordinate from 0 (magnetic axis) to 1 (last closed flux surface)
- **Grid resolution**: `g.n_rho = 25` cells
- **Grid spacing**: `g.dx = 1 / g.n_rho = 0.04`

### Grid Structure
```
Cell centers: g.cell_centers = [0.5*dx, 1.5*dx, ..., (n_rho - 0.5)*dx]
Face centers: g.face_centers = [0, dx, 2*dx, ..., n_rho*dx]
```

The grid uses a **staggered mesh**:
- **Cell-centered quantities**: `T_i`, `T_e`, `n_e`, `psi` (unknowns at cell centers)
- **Face-centered quantities**: diffusion coefficients `d_face`, convection velocities `v_face`, gradients

## State Variables

The current simulation state is stored in the global object `s`:
- **`s.T_i`** - Ion temperature [keV]
- **`s.T_e`** - Electron temperature [keV]
- **`s.psi`** - Poloidal magnetic flux [Wb]
- **`s.n_e`** - Electron density [m⁻³]
- **`s.t`** - Current simulation time [s]

These state variables are **read-only during a time step** and only updated after a successful step is completed.

## Evolved Variables

The code evolves **4 coupled variables**: `T_i`, `T_e`, `psi`, `n_e`

### Scaling Factors
Variables are internally scaled for numerical stability:
```python
g.scaling_T_i = 1.0      # keV
g.scaling_T_e = 1.0      # keV
g.scaling_n_e = 1e20     # 10²⁰ m⁻³
g.scaling_psi = 1.0      # Wb
```

## Governing Equations

Each variable satisfies a transport equation of the form:

```
∂/∂t (tic * x) = (1/toc) * [ ∇·(d ∇x - v x) + S_mat * x + S ]
```

Where:
- **`tic`** = transient coefficient in (time derivative)
- **`toc`** = transient coefficient out (normalization)
- **`d`** = diffusion coefficient (face-centered)
- **`v`** = convection velocity (face-centered)
- **`S_mat`** = implicit source matrix coupling
- **`S`** = explicit source term

### Temperature Equations

**Ion temperature** (`T_i`):
```
tic_T_i = n_i * vpr^(5/3)
toc_T_i = 1.5 * vpr^(-2/3) * keV_to_J
d_face = chi_face_ion * (n_i_face * g1_over_vpr_face * keV_to_J)
v_face = v_heat_face_ion (temperature gradient driven)
S = heating sources + fusion power + Q_ei (electron-ion heat exchange)
```

**Electron temperature** (`T_e`):
```
tic_T_e = n_e * vpr^(5/3)
toc_T_e = 1.5 * vpr^(-2/3) * keV_to_J
d_face = chi_face_el * (n_e_face * g1_over_vpr_face * keV_to_J)
v_face = v_heat_face_el
S = heating sources + fusion power - Q_ei
```

**Coupling**: `Q_ei` couples ion and electron temperatures via implicit terms `implicit_ii`, `implicit_ee`, `implicit_ie`, `implicit_ei`.

### Density Equation

**Electron density** (`n_e`):
```
tic_dens_el = vpr
toc_dens_el = 1.0
d_face = full_d_face_el = g1_over_vpr_face * D_el_total
v_face = full_v_face_el = g0_face * V_el_total
S = particle sources (gas puff, pellets, NBI)
```

### Magnetic Flux Equation

**Poloidal flux** (`psi`):
```
tic_psi = 1.0
toc_psi = (1/resistivity_multiplier) * rho_norm * sigma * (mu_0 * 16π² * Phi_b² / F²)
d_face = g2g3_over_rhon_face (geometric factor)
v_face = 0 (no convection)
S = -(8 * vpr * π² * B_0 * mu_0 * Phi_b / F²) * (j_bootstrap + j_external)
```

The safety factor `q` is derived from `psi`:
```
q_face = 2 * Phi_b * rho_face_norm / |∂psi/∂rho_norm|
```

## Boundary Conditions

Boundary conditions are stored as tuples with format:
```python
bc = (left_face, right_face, left_grad, right_grad)
```
Where:
- `left_face`: value at rho=0 (or None for natural BC)
- `right_face`: value at rho=1 (or None for natural BC)
- `left_grad`: gradient at rho=0
- `right_grad`: gradient at rho=1

### Applied Boundary Conditions

**Left boundary (rho=0, magnetic axis)**:
- `T_i_bc`: zero gradient (symmetry)
- `T_e_bc`: zero gradient (symmetry)
- `n_e_bc`: natural (no explicit constraint)
- `psi_bc`: natural

**Right boundary (rho=1, edge)**:
- `T_i_bc`: fixed value = `g.T_i_right_bc = 0.2` keV
- `T_e_bc`: fixed value = `g.T_e_right_bc = 0.2` keV
- `n_e_bc`: fixed value = `g.n_e_right_bc = 0.25e20` m⁻³
- `psi_bc`: fixed gradient from total plasma current `dpsi_drhonorm_edge`

## Numerical Discretization

### Spatial Discretization

The code uses **finite volume method** with exponential fitting for Peclet number control.

#### Face Values from Cell Values
```python
compute_face_value(value, dr, bc):
    left_face = value[0]  # left boundary
    inner = (value[:-1] + value[1:]) / 2.0  # linear interpolation
    right_face = value[-1] + (right_grad * dr / 2)  # from BC
    return [left_face, inner, right_face]
```

#### Face Gradients from Cell Values
```python
compute_face_grad(value, dr, bc):
    forward_diff = diff(value) / dr
    left_grad = constrained_grad(left_bc, value[0])
    right_grad = constrained_grad(right_bc, value[-1])
    return [left_grad, forward_diff, right_grad]
```

### Diffusion Terms

`make_diffusion_terms(d_face, dr, bc)` returns matrices `(mat, vec)` implementing:
```
div(d * grad(x)) ≈ (d[i+1]*(x[i+1]-x[i]) - d[i]*(x[i]-x[i-1])) / dr²
```

Discretization:
```python
diag = -(d_face[1:] + d_face[:-1]) / dr²
off_diag = d_face[1:-1] / dr²
```

**Boundary handling**:
- Left: zero-gradient → `diag[0] = -d_face[1] / dr²`, `vec[0] = -d_face[0] * left_grad / dr`
- Right (fixed value): `diag[-1] = (-2*d_face[-1] - d_face[-2]) / dr²`, `vec[-1] = 2*d_face[-1]*right_value / dr²`
- Right (fixed grad): `diag[-1] = -d_face[-2] / dr²`, `vec[-1] = d_face[-1]*right_grad / dr`

### Convection Terms

`make_convection_terms(v_face, d_face, dr, bc)` uses **exponential fitting scheme** to handle high Peclet numbers.

**Peclet number**:
```python
P[i] = (v_face[i] * dr) / d_face[i]
```

**Upwind factor** `alpha(P)`:
```python
if P > 10:        alpha = (P-1)/P              # full upwind
if 0 < P < 10:    alpha = ((P-1) + (1-P/10)⁵)/P  # smooth transition
if P ≈ 0:         alpha = 0.5                  # central difference
if -10 < P < 0:   alpha = ((1+P/10)⁵ - 1)/P
if P < -10:       alpha = -1/P
```

This prevents oscillations when `|v*dr/d|` is large (convection-dominated flows).

### Temporal Discretization

**Implicit theta method** with `g.theta_implicit = 1.0` (fully implicit/backward Euler):

```
(tic_new / toc_new) * (x_new - x_old) / dt = 
    theta * RHS(x_new) + (1-theta) * RHS(x_old)
```

With `theta=1.0`, this becomes:
```
[I - dt * theta * (1/(toc*tic)) * C_mat] * x_new = x_old + dt * (1/(toc*tic)) * c_vec
```

Where `C_mat` includes diffusion, convection, and source coupling matrices.

**Matrix assembly**:
```python
# Transient scaling
broadcasted = 1 / (tc_out_new * tc_in_new)
lhs_mat = I - dt * theta * broadcasted * c_mat_new
lhs_vec = -theta * dt * broadcasted * c_new

# Right hand side (with old time step)
right_transient = diag(tc_in_old / tc_in_new)
rhs = dot(right_transient, x_old) - lhs_vec

# Solve
x_new = solve(lhs_mat, rhs)
```

## Time Stepping

### Main Time Loop Structure

The code uses **nested adaptive time stepping** with state stored in global `s`:

```python
# Initialize state variables
s.T_i = <initial ion temperature>
s.T_e = <initial electron temperature>
s.n_e = <initial electron density>
s.psi = <initial poloidal flux>
s.t = 0.0

while s.t < t_final:
    # Outer loop: successful time steps
    dt = compute_initial_dt()
    
    while True:
        # Inner loop: retry with smaller dt if needed
        
        for corrector_iter in range(n_corrector_steps + 1):
            # Predictor-corrector iterations
            x_input = x_new
            # Compute coefficients at x_input
            # Build and solve linear system
            x_new = solve_system(x_input)
        
        # Check convergence
        if converged:
            break
        else:
            dt = dt / dt_reduction_factor
    
    # Accept step and update state
    s.t += dt
    s.T_i, s.T_e, s.psi, s.n_e = <unscaled solutions>
    history.append((s.t, s.T_i, s.T_e, s.psi, s.n_e))
```

### Time Step Calculation

**Initial dt estimate**:
```python
chi_max = max(chi_face_ion * g1_over_vpr2_face, 
              chi_face_el * g1_over_vpr2_face)
basic_dt = (3/4) * dx² / chi_max             # CFL-like condition
dt = min(chi_timestep_prefactor * basic_dt, max_dt)

# Ensure we hit t_final exactly
if current_t + dt > t_final:
    dt = t_final - current_t
```

Parameters:
- `g.chi_timestep_prefactor = 50` (safety factor)
- `g.max_dt = 0.5` s
- `g.min_dt = 1e-8` s
- `g.dt_reduction_factor = 3`

### Predictor-Corrector

The code uses **`g.n_corrector_steps = 1`** corrector iteration:

1. **Predictor** (iter 0): Solve with coefficients from `x_old`
2. **Corrector** (iter 1): Re-solve with coefficients from predicted `x_new`

This improves accuracy by making the scheme more consistent with the implicit formulation.

## Transport Coefficients

### Turbulent Transport (QLKNN Model)

The function `calculate_transport_coeffs()` (lines 779-1066) computes turbulent transport via the **QLKNN neural network surrogate** for QuaLiKiz gyrokinetic turbulence:

**Inputs to QLKNN**:
```python
QualikizInputs:
    Ati = lref_over_lti = -R_major * (∇T_i / T_i)     # Normalized T_i gradient
    Ate = lref_over_lte = -R_major * (∇T_e / T_e)     # Normalized T_e gradient
    Ane = lref_over_lne = -R_major * (∇n_e / n_e)     # Normalized n_e gradient
    Ani = lref_over_lni0 = -R_major * (∇n_i / n_i)    # Ion density gradient
    q = safety factor
    smag = magnetic shear = -(r/iota) * (∂iota/∂r)
    x = rho_face_norm * epsilon_lcfs / EPSILON_NN     # Normalized radius
    Ti_Te = T_i / T_e                                  # Temperature ratio
    LogNuStar = log10(nu_star)                        # Collisionality
    Z_eff_face = effective charge
    chiGB = gyro-Bohm normalization
```

**Outputs from QLKNN**:
```python
qi_itg, qi_tem      # Ion heat flux (ITG, TEM modes)
qe_itg, qe_tem, qe_etg  # Electron heat flux (ITG, TEM, ETG modes)
pfe_itg, pfe_tem    # Electron particle flux
```

**Conversion to transport coefficients**:
```python
# Heat diffusivities
chi_face_ion = (qi_itg + qi_tem) * chiGB / lref_over_lti * (R_major / a_minor)
chi_face_el = (qe_itg + qe_tem + qe_etg) * chiGB / lref_over_lte * (R_major / a_minor)

# Particle transport
pfe_SI = pfe * n_e_face * chiGB / a_minor
D_el = -pfe_SI / (∇n_e * geometric_factor)  if consistent sign
V_el = pfe_SI / (n_e * geometric_factor)    otherwise
```

**Masking and bounds** (lines 961-1002):
- Active region: `transport_rho_min < rho < transport_rho_max`
- Clipping: `chi_min = 0.05`, `chi_max = 100`, `D_e_min = 0.05`, `D_e_max = 100`
- Inner core override: `chi_i_inner = 1.0`, `chi_e_inner = 1.0` for `rho < rho_inner`

### Smoothing

Transport coefficients are smoothed via **Gaussian kernel convolution** (lines 1030-1064):
```python
kernel = exp(-log(2) * (rho_i - rho_j)² / smoothing_width²)
chi_smooth = dot(normalized_kernel, chi_raw)
```
with `g.smoothing_width = 0.1`.

### Neoclassical Transport

**Pereverzev model** adds neoclassical transport in pedestal (lines 1877-1896):
```python
if rho > rho_norm_ped_top:
    chi_face_per_ion = g1_over_vpr_face * n_i_face * keV_to_J * chi_pereverzev
    chi_face_per_el = g1_over_vpr_face * n_e_face * keV_to_J * chi_pereverzev
    d_face_per_el = D_pereverzev * g1_over_vpr_face
    v_face_per_el = (∇n_e / n_e) * D_pereverzev * geo_factor
```
with `g.chi_pereverzev = 30`, `g.D_pereverzev = 15`.

## Current and Conductivity

### Bootstrap Current

`_calculate_bootstrap_current()` (lines 392-450) implements the **Sauter-Angioni model**:

```python
# Collisionality parameters
nu_e_star = 6.921e-18 * q * R * n_e * Z_eff * log_lambda / (T_e² * epsilon^1.5)
nu_i_star = 4.9e-18 * q * R * n_i * Z_eff⁴ * log_lambda_ii / (T_i² * epsilon^1.5)

# Trapped particle fraction
f_trap = 1 - sqrt((1-epsilon)/(1+epsilon)) * (1-epsilon_eff) / (1+2*sqrt(epsilon_eff))

# Transport coefficients
L31 = calculate_L31(f_trap, nu_e_star, Z_eff)  # pressure-driven
L32 = calculate_L32(f_trap, nu_e_star, Z_eff)  # temperature-driven electron
L34 = calculate_L34(f_trap, nu_e_star, Z_eff)  # temperature-driven ion
alpha = calculate_alpha(f_trap, nu_i_star)     # ion-electron coupling

# Bootstrap current density
j_bootstrap = (F/(2π * B_0)) * (1/∂psi/∂rho) * [
    L31 * (p_e * ∇ln(n_e) + p_i * ∇ln(n_i)) +
    (L31 + L32) * p_e * ∇ln(T_e) +
    (L31 + alpha*L34) * p_i * ∇ln(T_i)
]
```

### Plasma Conductivity

`calculate_conductivity()` (lines 353-375) uses **Sauter neoclassical resistivity**:

```python
# Spitzer conductivity
sigma_Spitzer = 1.9012e4 * (T_e[keV])^1.5 / (Z_eff * NZ * log_lambda_ei)
where NZ = 0.58 + 0.74/(0.76 + Z_eff)

# Neoclassical correction
ft33 = f_trap / (1 + (0.55 - 0.1*f_trap)*sqrt(nu_e_star) + 
                 0.45*(1-f_trap)*nu_e_star / Z_eff^1.5)
sigma_neo = 1 - ft33 * (1 + 0.36/Z_eff - ft33*(0.59/Z_eff - 0.23*ft33/Z_eff))

sigma = sigma_Spitzer * sigma_neo
```

## Source Terms

Sources are registered in `g.source_registry` (lines 1571-1602) with handlers:

```python
SourceHandler(
    affects: tuple[AffectedCoreProfile],  # Which equations it affects
    eval_fn: callable                     # Function to compute source
)
```

### Generic Heating Source

```python
"generic_heat": (T_i, T_e) sources
    profile = gaussian_profile(center, width, total_power)
    ion_source = profile * (1 - electron_fraction)
    el_source = profile * electron_fraction
```
Parameters: `P_total = 51 MW`, `location = 0.127`, `width = 0.073`, `electron_fraction = 0.68`

### Fusion Source

```python
"fusion": (T_i, T_e) alpha heating
    # D-T fusion reaction rate
    sigma_v = Bosch_Hale_formula(T_i)
    P_fus = DT_fraction² * n_i² * sigma_v * E_fusion
    
    # Alpha slowing down (energy deposition)
    critical_energy = 10 * A_alpha * T_e
    energy_ratio = birth_energy / critical_energy
    frac_i = alpha_slowing_formula(energy_ratio)
    frac_e = 1 - frac_i
    
    P_alpha_i = 0.2 * P_fus  # 3.5/17.6 MeV to alphas, rest to neutrons
    P_alpha_e = 0.2 * P_fus * frac_e
```

### Particle Sources

```python
"generic_particle": Gaussian profile
    S = gaussian_profile(center=0.3, width=0.25, total=2.05e20)

"gas_puff": Edge exponential decay
    S = exp(-(1-rho)/decay_length) normalized to total=6e21

"pellet": Gaussian injection
    S = gaussian_profile(center=0.85, width=0.1, total=0)  # Currently off
```

### Current Source

```python
"generic_current": External current drive (NBI, ECCD, etc.)
    profile = exp(-((rho - location)² / (2*width²)))
    I_external = Ip * generic_current_fraction
    j_external = I_external * profile / integral(profile * spr)
```
Parameters: `fraction = 0.46`, `location = 0.36`, `width = 0.075`

### Adaptive Sources

Pedestal temperature/density control (lines 1873-1907):
```python
source_i += mask * adaptive_prefactor * T_i_pedestal
source_mat_ii -= mask * adaptive_prefactor

# This adds: adaptive_prefactor * (T_i_pedestal - T_i)
# at the pedestal location (rho = rho_norm_ped_top)
```

## Charge State and Impurities

### Ion Composition

Main ions: `g.main_ion_names = ("D", "T")` with equal fractions `[0.5, 0.5]`
Impurity: `g.impurity_names = ("Ne",)` (Neon)
Effective charge: `g.Z_eff = 1.6`

### Density Relationships

Quasi-neutrality: `n_e = Z_i * n_i + Z_impurity * n_impurity`

Given `Z_eff = (Z_i² * n_i + Z_impurity² * n_impurity) / n_e`, solve for dilution:
```python
dilution_factor = (Z_impurity - Z_eff) / (Z_i * (Z_impurity - Z_i))
n_i = n_e * dilution_factor
n_impurity = (n_e - Z_i * n_i) / Z_impurity
```

### Average Charge States

For Neon, use **Mavrin et al. model** (lines 282-293):
```python
if ion in MAVRIN_Z_COEFFS:
    interval_idx = searchsorted(TEMPERATURE_INTERVALS, T_e)
    coeffs = MAVRIN_Z_COEFFS[interval_idx]
    Z_avg = 10^(polyval(coeffs, log10(T_e)))
else:
    Z_avg = Z_nominal
```

For D, T: `Z_avg = 1.0` (fully ionized)

## Geometry

The code uses **CHEASE equilibrium data** from `geo/ITER_hybrid_citrin_equil_cheasedata.mat2cols`.

### Key Geometric Quantities

All quantities interpolated from CHEASE `rho_norm_intermediate` to simulation grid:

**Flux surface volumes**:
- `vpr = dvol/drho_norm` (volume derivative)
- `spr = darea/drho_norm` (surface area derivative)

**Magnetic field geometry**:
- `F = R * B_phi` (poloidal current function)
- `Phi = toroidal flux`
- `g0, g1, g2, g3` = metric coefficients
- `g2g3_over_rhon` = appears in diffusion of psi

**Miller parameters**:
- `elongation` = kappa (vertical elongation)
- `delta_face` = triangularity
- `epsilon_face = (R_out - R_in)/(R_out + R_in)` (inverse aspect ratio)

**Derived quantities**:
- `g0_over_vpr_face` for convection in cylindrical limit
- `g1_over_vpr_face`, `g1_over_vpr2_face` for transport coefficient normalization

### Smoothing (lines 1405-1409)

Some geometric quantities are Savitzky-Golay filtered near the axis:
```python
idx_limit = argmin(|rhon - rho_smoothing_limit|)  # rho < 0.1
smoothed = savgol_filter(data, window_length=5, polyorder=1 or 2)
data[:idx_limit] = smoothed[:idx_limit]
```

This removes noise from numerical differentiation near the magnetic axis.

## Variable Naming Conventions

### Prefixes and Suffixes

- `**_face`: Defined at cell faces (N+1 points)
- `**_cell` or no suffix: Defined at cell centers (N points)
- `**_bc`: Boundary condition dictionary
- `**_hires`: High-resolution grid for interpolation
- `**_profile`: 1D array across radius
- `geo_**: Geometric quantity from equilibrium
- `n_**`: Number/count (e.g., `n_rho`, `n_corrector_steps`)
- `d_face`: Diffusion coefficient
- `v_face`: Convection velocity
- `chi_face`: Heat diffusivity
- `sigma`: Electrical conductivity

### Physics Prefixes

- `j_`: Current density [A/m²]
- `q_` or `Q_`: Heat flux or heating power
- `p_` or `P_`: Power
- `log_`: Natural logarithm (e.g., `log_lambda_ei`)
- `nu_`: Collisionality parameter
- `Z_`: Charge state
- `A_`: Atomic mass
- `f_trap`: Trapped particle fraction

### State Variables

- `current_T_i`, `current_T_e`, `current_psi`, `current_n_e`: Current time step values
- `x_initial`, `x_new`, `x_old`: Solver state tuples (value, dr, bc)
- `tc_in`, `tc_out`: Transient coefficients (in/out of time derivative)
- `tic_**`, `toc_**`: Transient in/out for specific variable

### Matrix/Vector Notation

- `**_mat`: Matrix (2D array or block matrix)
- `**_vec`: Vector (1D array)
- `diffusion_mat`, `diffusion_vec`: Diffusion operator and boundary terms
- `conv_mat`, `conv_vec`: Convection operator and boundary terms
- `c_mat`, `c`: Combined spatial operator matrix and vector
- `source_mat_cell`: Coupling matrix for implicit sources
- `source_cell`: Explicit source vector

## Loop Structure

### Main Time Loop
```python
while True:
    # Compute q_face from current psi
    # Update ions
    # Calculate transport coefficients
    # Build source profiles
    # Compute initial dt
    
    # Inner dt retry loop
    while True:
        # Corrector loop
        for _ in range(n_corrector_steps + 1):
            # Coefficients callback
            # Build and solve linear system
        
        # Check convergence and reduce dt if needed
        if converged: break
    
    # Post-process step
    # Append to history
    # Check if finished
    if current_t >= t_final: break
```

### Coefficients Computation

This large block computes all coefficients needed for the linear system at a given state:

1. **Extract variables** from solver tuple
2. **Update ions** based on quasi-neutrality
3. **Compute q_face** from psi gradients
4. **Compute face values and gradients** for all variables
5. **Calculate conductivity**
6. **Build source profiles** including bootstrap current
7. **Compute transient coefficients**
8. **Calculate transport coefficients** from QLKNN
9. **Build diffusion terms**
10. **Add Pereverzev neoclassical**
11. **Compute sources**
12. **Package as tuples** for solver

### Linear System Assembly

Builds block system for coupled equations:

```python
# Initialize block matrices
c_mat = [[zero_block] * num_channels] * num_channels
c = [zero_vec] * num_channels

# Diffusion blocks (diagonal only)
for i in range(num_channels):
    c_mat[i][i] += diffusion_mat
    c[i] += diffusion_vec

# Convection blocks (diagonal only)
for i in range(num_channels):
    c_mat[i][i] += conv_mat
    c[i] += conv_vec

# Source coupling blocks (all channels)
for i in range(num_channels):
    for j in range(num_channels):
        if source_mat_cell[i][j] is not None:
            c_mat[i][j] += diag(source_mat_cell[i][j])

# Add explicit sources
# Assemble into single matrix
c_mat_new = block(c_mat)
c_new = block(c)

# Build LHS and RHS
lhs_mat = I - dt * theta * (1/(toc*tic)) * c_mat_new
rhs = (tic_old/tic_new) * x_old - dt * theta * (1/(toc*tic)) * c_new

# Solve
x_new = solve(lhs_mat, rhs)
```

### Post-Processing Loop

Write outputs:
```python
for var_name in evolving_names:
    # Compute cell+boundaries for all time steps
    var_data = [compute_cell_plus_boundaries_bc(var[t], dx, bc) 
                for t in range(nt)]
    
    # Write binary data
    var.tofile(f)
    
    # Plot snapshots at 0%, 25%, 50%, 75%, 100% of simulation
    for j, idx in enumerate([0, nt//4, nt//2, 3*nt//4, nt-1]):
        plot(rho, var[idx])
        savefig(f"{var_name}.{j:04d}.png")
```

## Physical Constants

All defined in the global namespace `g` (lines 28-44):
```python
g.keV_to_J = 1.602176634e-16    # keV to Joules
g.eV_to_J = 1.602176634e-19     # eV to Joules
g.m_amu = 1.6605390666e-27      # Atomic mass unit [kg]
g.q_e = 1.602176634e-19         # Elementary charge [C]
g.m_e = 9.1093837e-31           # Electron mass [kg]
g.epsilon_0 = 8.85418782e-12    # Permittivity [F/m]
g.mu_0 = 4π × 10⁻⁷              # Permeability [H/m]
g.k_B = 1.380649e-23            # Boltzmann constant [J/K]
```

## Tolerances and Numerics

```python
g.TOLERANCE = 1e-6              # General tolerance
g.eps = 1e-7                    # Small number to avoid division by zero
g.EPS_CONVECTION = 1e-20        # Minimum diffusivity for convection scheme
g.EPS_PECLET = 1e-3             # Peclet number threshold for alpha calculation
g.tolerance = 1e-7              # Time stepping tolerance
```

## Output Format

Binary file `run.raw` structure:
```
t[nt]           : float64, nt time points
rho[nr]         : float64, nr = n_rho + 2 (with boundaries)
T_i[nt, nr]     : float64
T_e[nt, nr]     : float64  
psi[nt, nr]     : float64
n_e[nt, nr]     : float64
```

PNG plots: `{variable}.{j:04d}.png` where j ∈ {0,1,2,3,4} for time snapshots.

## Key Numerical Features

1. **Implicit method**: Unconditionally stable, allows large time steps
2. **Exponential fitting**: Handles convection-dominated flows without oscillations
3. **Predictor-corrector**: Improves accuracy of nonlinear coupling
4. **Adaptive time stepping**: Automatically reduces dt on convergence failure
5. **Block-coupled system**: All 4 equations solved simultaneously with cross-coupling
6. **Adaptive sources**: Pedestal control via feedback terms
7. **JAX**: Enables future GPU acceleration and automatic differentiation
8. **Staggered grid**: Natural for finite volume, avoids checkerboard instabilities

## Simulation Parameters

**Plasma**:
- Major radius: `R_major = 6.2` m
- Minor radius: `a_minor = 2.0` m  
- Magnetic field: `B_0 = 5.3` T
- Plasma current: `Ip = 10.5` MA
- Effective charge: `Z_eff = 1.6`

**Time**:
- Initial: `t_initial = 0` s
- Final: `t_final = 5` s
- Max timestep: `max_dt = 0.5` s

**Transport**:
- Model: QLKNN-10D neural network
- Smoothing width: `0.1` in rho_norm
- Active region: `0 < rho < 0.91`

**Resistivity**: `resistivity_multiplier = 200` (slows psi evolution for numerical stability)

## Summary

This code implements a sophisticated 1D tokamak transport solver with:
- **4 coupled evolution equations** (T_i, T_e, n_e, psi)
- **Implicit finite volume method** with exponential fitting
- **Adaptive predictor-corrector time stepping**
- **Neural network turbulence model** (QLKNN)
- **Neoclassical effects** (bootstrap current, Sauter conductivity)
- **Realistic sources** (heating, fusion, fueling)
- **ITER-like geometry** from CHEASE equilibrium

The numerical scheme is designed for **robustness** (implicit, adaptive) and **accuracy** (exponential fitting, predictor-corrector) while handling the **stiff multi-physics coupling** inherent to tokamak transport.

