#!/usr/bin/env python3
"""Debug S211 for C-C jcall==4 pair."""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

import torch
from mlxmolkit.rm1.params import RM1_PARAMS, ANG_TO_BOHR
from mlxmolkit.rm1.overlap import _aintgs, _bintgs

# C-C at 1.54 A
pA = RM1_PARAMS[6]
pB = RM1_PARAMS[6]
R = 1.54
R_bohr = R * ANG_TO_BOHR
print(f"R_bohr = {R_bohr:.10f}")
print(f"zA_s = {pA.zeta_s:.10f}, zA_p = {pA.zeta_p:.10f}")
print(f"zB_s = {pB.zeta_s:.10f}, zB_p = {pB.zeta_p:.10f}")

# S211: A211, B211 from SET(R, zA_p, zB_s)
alpha_ps = 0.5 * R_bohr * (pA.zeta_p + pB.zeta_s)
beta_ps = 0.5 * R_bohr * (pA.zeta_p - pB.zeta_s)
print(f"\nS211 SET:")
print(f"  alpha = {alpha_ps:.10f}")
print(f"  beta  = {beta_ps:.10f}")

A211 = _aintgs(alpha_ps)
B211 = _bintgs(beta_ps)
print(f"  A = {A211}")
print(f"  B = {B211}")

# S211 formula
coeff = (pB.zeta_s * pA.zeta_p)**2.5 * R_bohr**5
inner = (A211[3]*(B211[0]-B211[2]) - A211[1]*(B211[2]-B211[4])
         + B211[3]*(A211[0]-A211[2]) - B211[1]*(A211[2]-A211[4]))
S211_ours = coeff * inner / (16.0 * np.sqrt(3.0))
print(f"  coeff = {coeff:.10f}")
print(f"  inner = {inner:.10f}")
print(f"  S211 = {S211_ours:.10f}")

# Now PYSEQM's version
from seqm.seqm_functions.diat_overlap_PM6_SP import SET, diatom_overlap_matrix_PM6_SP

# Call SET directly
ni = torch.tensor([6])
nj = torch.tensor([6])
jcall = torch.tensor([4])
rij_t = torch.tensor([R_bohr], dtype=torch.float64)

zA_s_t = torch.tensor([pA.zeta_s], dtype=torch.float64)
zA_p_t = torch.tensor([pA.zeta_p], dtype=torch.float64)
zB_s_t = torch.tensor([pB.zeta_s], dtype=torch.float64)
zB_p_t = torch.tensor([pB.zeta_p], dtype=torch.float64)

A211_pyseqm, B211_pyseqm = SET(rij_t, jcall, zA_p_t, zB_s_t)
print(f"\nPYSEQM SET:")
print(f"  A = {A211_pyseqm[0].numpy()}")
print(f"  B = {B211_pyseqm[0].numpy()}")

# Compare A integrals
print(f"\nA integral comparison (ours vs PYSEQM):")
for i in range(5):
    our_val = A211[i]
    pyseqm_val = A211_pyseqm[0, i].item()
    print(f"  A[{i}] = {our_val:.12f} vs {pyseqm_val:.12f}  diff={abs(our_val-pyseqm_val):.2e}")

print(f"\nB integral comparison (ours vs PYSEQM):")
for i in range(5):
    our_val = B211[i]
    pyseqm_val = B211_pyseqm[0, i].item()
    print(f"  B[{i}] = {our_val:.12f} vs {pyseqm_val:.12f}  diff={abs(our_val-pyseqm_val):.2e}")

# Compute S211 from PYSEQM
A_p = A211_pyseqm[0].numpy()
B_p = B211_pyseqm[0].numpy()
coeff_p = (pB.zeta_s * pA.zeta_p)**2.5 * R_bohr**5
inner_p = (A_p[3]*(B_p[0]-B_p[2]) - A_p[1]*(B_p[2]-B_p[4])
           + B_p[3]*(A_p[0]-A_p[2]) - B_p[1]*(A_p[2]-A_p[4]))
S211_pyseqm_manual = coeff_p * inner_p / (16.0 * np.sqrt(3.0))
print(f"\nS211 from PYSEQM A/B: {S211_pyseqm_manual:.10f}")

# Now get full PYSEQM overlap
xij = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
zeta_a = torch.tensor([[pA.zeta_s, pA.zeta_p]], dtype=torch.float64)
zeta_b = torch.tensor([[pB.zeta_s, pB.zeta_p]], dtype=torch.float64)
qn_int = torch.zeros(10, dtype=torch.long)
qn_int[6] = 2
di = diatom_overlap_matrix_PM6_SP(ni, nj, xij, rij_t, zeta_a, zeta_b, qn_int)
print(f"\nFull PYSEQM di[1,0] (S211) = {di[0,1,0].item():.10f}")
print(f"Full PYSEQM di[0,1] (-S121)= {di[0,0,1].item():.10f}")

# Also check: does PYSEQM use more A/B terms?
print(f"\nPYSEQM A dimensions: {A211_pyseqm.shape}")
