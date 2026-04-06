#!/usr/bin/env python3
"""Compare two-center two-electron integrals: ours vs PYSEQM."""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

from mlxmolkit.rm1.params import RM1_PARAMS, ANG_TO_BOHR
from mlxmolkit.rm1.two_center_integrals import two_center_integrals
from mlxmolkit.rm1.rotation import rotate_integrals_to_molecular_frame
from mlxmolkit.rm1.overlap import overlap_molecular_frame

import torch

# C-H pair at 1.09 A along x-axis
pA = RM1_PARAMS[6]  # C
pB = RM1_PARAMS[1]  # H
R = 1.09
coordA = np.array([0.0, 0.0, 0.0])
coordB = np.array([1.09, 0.0, 0.0])

# Our two-center integrals
ri, core, pair_type = two_center_integrals(pA, pB, R)
print(f"Our local-frame ri ({len(ri)} integrals), pair_type={pair_type}:")
for i in range(len(ri)):
    if abs(ri[i]) > 1e-10:
        print(f"  ri[{i:2d}] = {ri[i]:12.6f}")

# Our rotated integrals
w, e1b, e2a = rotate_integrals_to_molecular_frame(pA, pB, coordA, coordB)
print("\nOur rotated w tensor (non-zero elements):")
for kk in range(4):
    for ll in range(4):
        for mm in range(4):
            for nn in range(4):
                if abs(w[kk,ll,mm,nn]) > 1e-10:
                    print(f"  w[{kk},{ll},{mm},{nn}] = {w[kk,ll,mm,nn]:12.6f}")

print("\ne1b (electron on A, attracted to nucleus B):")
for i in range(4):
    for j in range(4):
        if abs(e1b[i,j]) > 1e-10:
            print(f"  e1b[{i},{j}] = {e1b[i,j]:12.6f}")

print("\ne2a (electron on B, attracted to nucleus A):")
for i in range(4):
    for j in range(4):
        if abs(e2a[i,j]) > 1e-10:
            print(f"  e2a[{i},{j}] = {e2a[i,j]:12.6f}")

# Our overlap
S = overlap_molecular_frame(pA, pB, coordA, coordB)
print("\nOur overlap S:")
for i in range(S.shape[0]):
    row = " ".join(f"{S[i,j]:10.6f}" for j in range(S.shape[1]))
    print(f"  [{row}]")

# Compute H_core elements for this pair
print("\nH_core off-diagonal (0.5*(beta_mu+beta_nu)*S[mu,nu]):")
for mu in range(pA.n_basis):
    beta_mu = pA.beta_s if mu == 0 else pA.beta_p
    for nu in range(pB.n_basis):
        beta_nu = pB.beta_s if nu == 0 else pB.beta_p
        h = 0.5 * (beta_mu + beta_nu) * S[mu, nu]
        if abs(h) > 1e-10:
            print(f"  H[{mu},{nu}] = {h:12.6f} (beta_mu={beta_mu:.3f}, beta_nu={beta_nu:.3f}, S={S[mu,nu]:.6f})")

# Now compare with PYSEQM
print("\n" + "="*60)
print("PYSEQM comparison:")
print("="*60)

# Load PYSEQM two-electron integrals
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int_sp as pyseqm_two_elec
from seqm.seqm_functions.parameters import params as pyseqm_load_params

p = pyseqm_load_params(method='RM1', elements=[0, 1, 6],
                       root_dir='/Users/tgg/Github/pyseqm_ref/seqm/params/')
print(f"PYSEQM RM1 params for C: {p[6].numpy()}")
print(f"PYSEQM RM1 params for H: {p[1].numpy()}")
print(f"Our RM1 params for C: Uss={pA.Uss}, Upp={pA.Upp}, zeta_s={pA.zeta_s}, zeta_p={pA.zeta_p}")
print(f"  beta_s={pA.beta_s}, beta_p={pA.beta_p}")
print(f"  gss={pA.gss}, gsp={pA.gsp}, gpp={pA.gpp}, gp2={pA.gp2}, hsp={pA.hsp}")
print(f"Our RM1 params for H: Uss={pB.Uss}, zeta_s={pB.zeta_s}, beta_s={pB.beta_s}")
print(f"  gss={pB.gss}")
