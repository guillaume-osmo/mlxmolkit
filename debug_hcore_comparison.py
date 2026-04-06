#!/usr/bin/env python3
"""Compare H_core matrix between our code and PYSEQM for H2O."""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

from mlxmolkit.rm1.params import RM1_PARAMS, ANG_TO_BOHR
from mlxmolkit.rm1.scf import _build_basis_info, _build_core_hamiltonian
from mlxmolkit.rm1.rotation import rotate_integrals_to_molecular_frame

# H2O geometry
atoms = [8, 1, 1]
coords = np.array([
    [0.0, 0.0, 0.0],
    [0.9584, 0.0, 0.0],
    [-0.2396, 0.9275, 0.0],
])

info = _build_basis_info(atoms)
H = _build_core_hamiltonian(atoms, coords, info)

print("Our H_core for H2O (6x6):")
labels = ['Os', 'Opx', 'Opy', 'Opz', 'H1s', 'H2s']
for i in range(6):
    row = " ".join(f"{H[i,j]:10.4f}" for j in range(6))
    print(f"  {labels[i]:>4s} [{row}]")

# Get the diagonal (one-center) and off-diagonal (resonance + nucl attract) separately
H_diag = np.zeros_like(H)
for mu in range(6):
    H_diag[mu, mu] = H[mu, mu]

H_offdiag = H - H_diag

print("\nDiagonal elements (Uss/Upp + nuclear attraction):")
for i in range(6):
    print(f"  H[{labels[i]},{labels[i]}] = {H[i,i]:.6f}")

print("\nOff-diagonal elements (resonance integrals):")
for i in range(6):
    for j in range(i+1, 6):
        if abs(H[i,j]) > 1e-10:
            print(f"  H[{labels[i]},{labels[j]}] = {H[i,j]:.6f}")

# Check nuclear attraction separately
print("\n--- Nuclear attraction contributions ---")
params = info['params']
starts = info['atom_basis_start']

for i in range(len(atoms)):
    for j in range(len(atoms)):
        if i == j:
            continue
        _, e1b_ij, _ = rotate_integrals_to_molecular_frame(
            params[i], params[j], coords[i], coords[j],
        )
        nA = params[i].n_basis
        print(f"\ne1b[{i}←{j}] (nucleus {j} attracting electrons on atom {i}):")
        for mu_a in range(nA):
            for nu_a in range(nA):
                if abs(e1b_ij[mu_a, nu_a]) > 1e-10:
                    print(f"  e1b[{mu_a},{nu_a}] = {e1b_ij[mu_a, nu_a]:.6f}")

# Also print the two-electron w tensor for O-H1 pair
print("\n--- Two-electron w tensor for O(0)-H1(1) pair ---")
w, e1b, e2a = rotate_integrals_to_molecular_frame(
    params[0], params[1], coords[0], coords[1],
)
print("Non-zero w elements:")
for kk in range(4):
    for ll in range(4):
        for mm in range(1):  # H has only s
            for nn in range(1):
                if abs(w[kk,ll,mm,nn]) > 1e-10:
                    print(f"  w[{kk},{ll},{mm},{nn}] = {w[kk,ll,mm,nn]:.6f}")
