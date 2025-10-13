#!/usr/bin/env python3
"""Compare two binary .raw files and report numerical differences."""

import sys
import numpy as np
import os

if len(sys.argv) != 3:
    print("Usage: diff.py <file1.raw> <file2.raw>")
    print("Example: diff.py run.raw ~/run.d2be31f")
    sys.exit(1)

file1 = os.path.expanduser(sys.argv[1])
file2 = os.path.expanduser(sys.argv[2])

try:
    with open(file1, 'rb') as f:
        data1 = np.fromfile(f, dtype=np.float64)
except FileNotFoundError:
    print(f"Error: File '{file1}' not found")
    sys.exit(1)

try:
    with open(file2, 'rb') as f:
        data2 = np.fromfile(f, dtype=np.float64)
except FileNotFoundError:
    print(f"Error: File '{file2}' not found")
    sys.exit(1)

print(f"File 1: {file1}")
print(f"  Size: {len(data1)} float64 values ({len(data1) * 8} bytes)")
print(f"\nFile 2: {file2}")
print(f"  Size: {len(data2)} float64 values ({len(data2) * 8} bytes)")

if len(data1) != len(data2):
    print(f"\nERROR: Size mismatch!")
    print(f"  Difference: {len(data1) - len(data2)} values")
    sys.exit(1)

# Deduce structure: time, rho, then 4 variables (T_i, T_e, psi, n_e)
# File format: nt values (time), nx+2 values (rho), then 4 * nt * (nx+2) values
# Total: nt + (nx+2) + 4*nt*(nx+2)
# Assume nx=25 (fixed)
nx = 25
nrho = nx + 2  # rho has boundaries: [0.0, cell_centers..., 1.0]
n_vars = 4

# Solve for nt: total = nt + nrho + n_vars*nt*nrho
# total = nt(1 + n_vars*nrho) + nrho
# nt = (total - nrho) / (1 + n_vars*nrho)
total_size = len(data1)
nt = (total_size - nrho) // (1 + n_vars * nrho)

print(f"\nDeduced structure:")
print(f"  nx (cells): {nx}")
print(f"  nrho (with boundaries): {nrho}")
print(f"  nt (time steps): {nt}")
print(f"  n_vars (fields): {n_vars}")

# Parse data
offset = 0
time1 = data1[offset:offset+nt]
time2 = data2[offset:offset+nt]
offset += nt

rho1 = data1[offset:offset+nrho]
rho2 = data2[offset:offset+nrho]
offset += nrho

var_names = ['T_i', 'T_e', 'psi', 'n_e']
vars1 = {}
vars2 = {}
for var_name in var_names:
    var_size = nt * nrho
    vars1[var_name] = data1[offset:offset+var_size].reshape(nt, nrho)
    vars2[var_name] = data2[offset:offset+var_size].reshape(nt, nrho)
    offset += var_size

# Check if bitwise identical
if np.array_equal(data1, data2):
    print("\n✓ FILES ARE BITWISE IDENTICAL")
    sys.exit(0)

print(f"\n✗ FILES DIFFER\n")
print("=" * 70)

# Compare time
if np.array_equal(time1, time2):
    print("Time: ✓ IDENTICAL")
else:
    diff_t = np.abs(time2 - time1)
    print(f"Time: ✗ DIFFERS (max abs: {np.max(diff_t):.6e})")

# Compare rho
if np.array_equal(rho1, rho2):
    print("Rho:  ✓ IDENTICAL")
else:
    diff_rho = np.abs(rho2 - rho1)
    print(f"Rho:  ✗ DIFFERS (max abs: {np.max(diff_rho):.6e})")

print()

# Compare each variable
rtol = 1e-12
atol = 1e-12
all_close = True

for var_name in var_names:
    v1 = vars1[var_name]
    v2 = vars2[var_name]
    
    if np.array_equal(v1, v2):
        print(f"{var_name:4s}: ✓ IDENTICAL")
    else:
        diff = v2 - v1
        abs_diff = np.abs(diff)
        rel_diff = np.abs(diff / (np.abs(v1) + 1e-100))
        
        close = np.allclose(v1, v2, rtol=rtol, atol=atol)
        status = "✓ CLOSE" if close else "✗ DIFFER"
        
        print(f"{var_name:4s}: {status}")
        print(f"      Abs: max={np.max(abs_diff):.6e}, mean={np.mean(abs_diff):.6e}")
        print(f"      Rel: max={np.max(rel_diff):.6e}, mean={np.mean(rel_diff):.6e}")
        
        if not close:
            all_close = False

print()
print("=" * 70)

# Overall comparison
diff = data2 - data1
abs_diff = np.abs(diff)
rel_diff = np.abs(diff / (np.abs(data1) + 1e-100))

print(f"\nOverall statistics:")
print(f"  Max abs diff: {np.max(abs_diff):.6e}")
print(f"  Max rel diff: {np.max(rel_diff):.6e}")

if all_close:
    print(f"\n✓ All fields numerically close (rtol={rtol}, atol={atol})")
    sys.exit(0)
else:
    print(f"\n✗ Some fields differ beyond tolerance (rtol={rtol}, atol={atol})")
    sys.exit(1)


