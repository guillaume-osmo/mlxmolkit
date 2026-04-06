#!/usr/bin/env python3
"""Debug B-integrals comparison."""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

import torch
from seqm.seqm_functions.diat_overlap_PM6_SP import bintgs as pyseqm_bintgs

# Test with beta = -0.1193
beta = -0.1193
x = torch.tensor([beta], dtype=torch.float64)
jcall = torch.tensor([4])
B_pyseqm = pyseqm_bintgs(x, jcall)
print(f"PYSEQM bintgs({beta}):")
print(f"  B = {B_pyseqm[0].numpy()}")

# Also test with beta = 0
B_zero = pyseqm_bintgs(torch.tensor([0.0], dtype=torch.float64), jcall)
print(f"\nPYSEQM bintgs(0.0):")
print(f"  B = {B_zero[0].numpy()}")

# Test with beta = 0.6 (should use exact)
B_large = pyseqm_bintgs(torch.tensor([0.6], dtype=torch.float64), jcall)
print(f"\nPYSEQM bintgs(0.6):")
print(f"  B = {B_large[0].numpy()}")

# Test: what are the actual correct b-values for beta = -0.1193?
# b1 = 2*sinh(x)/x
b = -0.1193
b1_exact = 2 * np.sinh(b) / b if abs(b) > 1e-10 else 2.0
print(f"\nExact b1({b}) = {b1_exact:.10f}")
print(f"Taylor b1 = 2.0")

# Check which bintgs module is actually loaded
import inspect
print(f"\nbintgs source file: {inspect.getfile(pyseqm_bintgs)}")
