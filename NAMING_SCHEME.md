# Naming Scheme Documentation

## Core Physics Variables (State Vector)
**Ultra-short names for primary quantities:**
- `i` = ion temperature (T_i)
- `e` = electron temperature (T_e)  
- `p` = poloidal flux (psi)
- `n` = electron density (n_e)

## Suffix Conventions

### Location Suffixes
- **`_f`** = face/interface values (e.g., `Ti_f`, `Te_f`, `ne_f`)
- **`_face`** = face values (INCONSISTENT - should be `_f`)
- **Cell center values have no suffix** (e.g., `Te`, `i`, `e`)

### Derivative/Gradient Suffixes  
- **`_grad`** = gradient on rho grid (normalized)
- **`_gr`** = gradient on rmid grid (midplane) - shortened
- **`_r`** = on rmid grid (ultra-short)

### Species/Type Suffixes
- **`ni`** = ion density (n_i)
- **`nz`** = impurity density (n_impurity)
- **`Zi`** = ion charge
- **`Zz`** = impurity charge  
- **`Zeff`** = effective charge

## Function Parameter Patterns

### SHORT (2-3 chars):
```python
qei_coupling(Te, ne, ni, nz, Zi, Zz, Ai, Az)
```
- Bare names for cell-centered quantities
- Capital for Z (charge state convention)

### MEDIUM (with location suffix):
```python
fusion_source(Te, Ti_f, ni_f)
neoclassical_conductivity(Te_f, ne_f, q, Zeff_f)
```
- `_f` suffix for face values
- Bare for cell center

### MIXED (location + derivative):
```python
bootstrap_current(Ti_f, Te_f, ne_f, ni_f, p_grad, q, 
                  Ti_grad, Te_grad, ne_grad, ni_grad, Zi_f, Zeff_f)
```
- `_f` for face values
- `_grad` for gradients
- Bare for q (safety factor)

### INCONSISTENT:
```python
turbulent_transport(Ti_f, Ti_gr, Te_f, Te_gr, ne_f, ne_grad, ne_gr, ...)
                    ^^^^  ^^^^^              ^^^^^^^ ^^^^^^^
                    _f    _gr                _grad   _gr  (MIXED!)
```

## Internal Variable Naming

### Transport Coefficients
- `chi_i`, `chi_e` = thermal diffusivities
- `D_n`, `v_n` = particle diffusivity, convection
- `chi_neo_i`, `chi_neo_e` = neoclassical contributions

### **INCONSISTENT Internal Names:**
- `chi_face_ion`, `chi_face_el` ← uses `_face_` infix (OLD STYLE)
- Should be: `chi_i`, `chi_e` (matches main loop)

### Matrices/Operators
- `A_i`, `A_e`, `A_p`, `A_n` = transport matrices
- `b_i`, `b_e`, `b_p`, `b_n` = RHS vectors
- `C_ii`, `C_ie`, `C_ei`, `C_ee` = coupling matrices

### Sources
- `src_i`, `src_e` = source terms (shortened from `source_i`)
- `si_fus`, `se_fus` = fusion sources (ultra-short)
- `source_p` = psi source (NOT shortened - inconsistent)

### Physics Quantities
- `ft` = trapped particle fraction
- `nue`, `nui` = collisionality (nu_e_star, nu_i_star)
- `alph` = alpha (to avoid conflict with numpy)
- `eps` = epsilon (aspect ratio)
- `jbs` = bootstrap current
- `gc` = global coefficient
- `pe`, `pi` = electron/ion pressure

## Main Loop Variables (Consistent)
```python
i, e, p, n = pred[l.i], pred[l.e], pred[l.p], pred[l.n]
i_face, i_grad, i_grad_r = ...
e_face, e_grad, e_grad_r = ...
n_face, n_grad, n_grad_r = ...
```
**GOOD:** Follows pattern: `{var}_{location}_{grid}`

## IDENTIFIED INCONSISTENCIES

### 1. **Suffix inconsistency in turbulent_transport:**
```python
def turbulent_transport(Ti_f, Ti_gr, Te_f, Te_gr, ne_f, ne_grad, ne_gr, ...)
#                              ^^^^              ^^^^^       ^^^^^^^ ^^^^^
#                              Should all be _gr OR _grad (pick one!)
```

### 2. **Internal variable naming:**
```python
# OLD STYLE (should remove):
chi_face_ion, chi_face_el, d_face_el, v_face_el
# Should be:
chi_i, chi_e, d_e, v_e  (or keep _f suffix)
```

### 3. **Source naming:**
```python
src_i, src_e     # shortened
source_p         # NOT shortened (inconsistent)
si_fus, se_fus   # ultra-shortened
```

### 4. **Main loop vs function signature mismatch:**
```python
# Main loop uses:
i_face, e_face, n_face, ni_face, nz_face

# Functions receive:
Ti_f, Te_f, ne_f, ni_f, nz_f  (different convention!)
```

## RECOMMENDATIONS

### Option 1: Ultra-short everywhere
- Use `_f` for face, `_g` for grad, `_r` for rmid
- Remove all `_face`, `_grad` long forms

### Option 2: Short but readable  
- Keep `_f` for face
- Use `_g` for all gradients (drop `_gr` vs `_grad`)
- Use `_r` suffix for rmid grid quantities

### Option 3: Stay consistent with main loop
- Use full forms in main loop: `i_face`, `i_grad`, `i_grad_r`
- Use short forms in functions: `Ti_f`, `Ti_g`, `Ti_r`
- Makes function signatures compact while main loop is explicit

## Current State Summary

**Strengths:**
- ✅ Core variables (i, e, p, n) are ultra-short and clear
- ✅ Matrix naming (A_i, b_i, C_ii) is consistent
- ✅ Main loop structure is logical

**Weaknesses:**
- ❌ Mixed `_face` vs `_f` suffix usage
- ❌ Mixed `_grad` vs `_gr` vs `_g` for gradients
- ❌ Internal function variables use old `_face_` infix style
- ❌ Source naming inconsistent (src_i vs source_p)
- ❌ Parameter names differ between main loop and function signatures

