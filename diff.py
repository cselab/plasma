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

if np.array_equal(data1, data2):
    print("\n✓ FILES ARE BITWISE IDENTICAL")
    sys.exit(0)

diff = data2 - data1
abs_diff = np.abs(diff)
rel_diff = np.abs(diff / (np.abs(data1) + 1e-100))

print(f"\n✗ FILES DIFFER")
print(f"\nAbsolute differences:")
print(f"  Max:  {np.max(abs_diff):.6e}")
print(f"  Mean: {np.mean(abs_diff):.6e}")
print(f"  RMS:  {np.sqrt(np.mean(abs_diff**2)):.6e}")

print(f"\nRelative differences:")
print(f"  Max:  {np.max(rel_diff):.6e}")
print(f"  Mean: {np.mean(rel_diff):.6e}")
print(f"  RMS:  {np.sqrt(np.mean(rel_diff**2)):.6e}")

max_idx = np.argmax(abs_diff)
print(f"\nLargest difference at index {max_idx}:")
print(f"  File 1: {data1[max_idx]:.15e}")
print(f"  File 2: {data2[max_idx]:.15e}")
print(f"  Diff:   {diff[max_idx]:.15e}")
print(f"  Rel:    {rel_diff[max_idx]:.15e}")

rtol = 1e-12
atol = 1e-12
if np.allclose(data1, data2, rtol=rtol, atol=atol):
    print(f"\n✓ Files are numerically close (rtol={rtol}, atol={atol})")
    sys.exit(0)
else:
    print(f"\n✗ Files differ beyond tolerance (rtol={rtol}, atol={atol})")
    sys.exit(1)


