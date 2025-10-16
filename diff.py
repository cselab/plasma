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

nx = 25
n_vars = 4
nt = len(data1) // (n_vars * nx + 1)

print(f"\nDeduced structure:")
print(f"  nx (cells): {nx}")
print(f"  nt (time steps): {nt}")
print(f"  n_vars (fields): {n_vars}")

o = 0
state1 = data1[o:o + nt * n_vars * nx].reshape(nt, n_vars * nx)
state2 = data2[o:o + nt * n_vars * nx].reshape(nt, n_vars * nx)
o += nt * n_vars * nx

time1 = data1[o:o+nt]
time2 = data2[o:o+nt]

rho1 = rho2 = np.linspace(0, 1, nx + 2)[1:-1]

var_names = ['T_i', 'T_e', 'psi', 'n_e']
vars1 = {var_names[i]: state1[:, i*nx:(i+1)*nx] for i in range(n_vars)}
vars2 = {var_names[i]: state2[:, i*nx:(i+1)*nx] for i in range(n_vars)}

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


