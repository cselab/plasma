# Naming Scheme

## Core Variables
- `i` = ion temperature (T_i)
- `e` = electron temperature (T_e)  
- `p` = poloidal flux (psi)
- `n` = electron density (n_e)

## Suffix Conventions

### Location Suffixes
- **`_f`** = face/interface values (e.g., `i_f`, `e_f`, `n_f`)
- **No suffix** = cell center values (e.g., `i`, `e`, `p`, `n`)

### Derivative/Gradient Suffixes  
- **`_g`** = gradient on rho grid (normalized) 
- **`_r`** = gradient on rmid grid (midplane radius)

### Species/Type Suffixes
- **`ni`** = ion density (n_i) 
- **`nz`** = impurity density (n_impurity)
- **`Zi`** = ion charge
- **`Zz`** = impurity charge  
- **`Zeff`** = effective charge

Note: Ion density uses `ni` (2 chars) to distinguish from electron density `n`

## Function Signatures

### Cell-centered:
```python
qei_coupling(e, n, ni, nz, Zi, Zz, Ai, Az)
```

### Face values:
```python
fusion_source(e, i_f, ni_f)
neoclassical_conductivity(e_f, n_f, q, Zeff_f)
```

### Face and gradients:
```python
bootstrap_current(i_f, e_f, n_f, ni_f, p_g, q, 
                  i_g, e_g, n_g, ni_g, Zi_f, Zeff_f)
neoclassical_transport(i_f, e_f, n_f, ni_f, i_g, e_g, n_g)
```

### Multiple grids:
```python
turbulent_transport(i_f, i_r, e_f, e_r, n_f, n_g, n_r, 
                    ni_f, ni_r, nz_f, nz_r, p_g, q, ions)
```
Where `_r` = rmid gradients, `_g` = rho gradients

## Internal Variable Naming

### Transport Coefficients
- `chi_i`, `chi_e` = thermal diffusivities
- `D_n`, `v_n` = particle diffusivity, convection
- `d_i`, `d_e`, `d_n` = diffusion terms
- `v_i`, `v_e`, `v_n` = convection terms

### Matrices/Operators
- `A_i`, `A_e`, `A_p`, `A_n` = transport matrices
- `b_i`, `b_e`, `b_p`, `b_n` = RHS vectors
- `C_ii`, `C_ie`, `C_ei`, `C_ee` = coupling matrices

### Sources
- `src_i`, `src_e`, `src_p`, `src_n` = source terms
- `si_fus`, `se_fus` = fusion sources (temporary variables)

### Physics Quantities
- `ft` = trapped particle fraction
- `nue`, `nui` = collisionality (nu_e_star, nu_i_star)
- `alph` = alpha (avoiding numpy conflict)
- `eps` = epsilon (aspect ratio)
- `jbs` = bootstrap current
- `gc` = global coefficient
- `pe`, `pi` = electron/ion pressure

## Main Loop Variables

```python
i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]
i_face, i_grad, i_grad_r = ...
e_face, e_grad, e_grad_r = ...
n_face, n_grad, n_grad_r = ...
p_grad = ...
```

Pattern: `{var}_{location}_{grid}` where grid is optional

## Naming Convention Rules

1. **Suffixes are additive from left to right:**
   - Base variable: `i`, `e`, `p`, `n`
   - Add location: `i_f` (face), `e` (cell - no suffix)
   - Add derivative: `i_g` (gradient on rho), `i_r` (gradient on rmid)

2. **Keep it short but clear:**
   - 1 char for state vars: `i`, `e`, `p`, `n`
   - 2 chars only for ion species: `ni`, `nz` (to distinguish from electron density `n`)
   - 1-2 char suffixes: `_f`, `_g`, `_r`

3. **Avoid redundancy:**
   - Don't use `_grad_rmid` when `_r` suffices
   - Don't use `face` when `_f` suffices
