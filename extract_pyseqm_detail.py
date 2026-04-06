#!/usr/bin/env python3
"""Extract PYSEQM detailed matrices for CH4 RM1."""
import sys
sys.path.insert(0, "/Users/tgg/Github/pyseqm_ref")
sys.path.insert(0, "/Users/tgg/Github/mlxmolkit_phase1")

import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
from seqm.api import Constants, Electronic_Structure, Molecule

torch.set_default_dtype(torch.float64)

# CH4 with SAME coordinates
species = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.int64)
coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.6276, 0.6276, 0.6276],
                              [0.6276, -0.6276, -0.6276],
                              [-0.6276, 0.6276, -0.6276],
                              [-0.6276, -0.6276, 0.6276]]])

const = Constants()
seqm_params = {"method": "RM1", "scf_eps": 1e-8, "scf_converger": [2], "elements": [0, 1, 6]}
mol = Molecule(const, seqm_params, coordinates, species)
esdriver = Electronic_Structure(seqm_params)
esdriver(mol)

# Extract the 8x8 density matrix from the 20x20 padded version
dm_full = mol.dm[0].detach().numpy()  # 20x20
# Basis: C(s,px,py,pz) at indices 0-3, H1(s) at 4, H2(s) at 8, H3(s) at 12, H4(s) at 16
# In PYSEQM: each atom gets 4 slots, so atom i occupies indices [4*i, 4*i+3]
idx = [0, 1, 2, 3, 4, 8, 12, 16]  # active basis indices
dm_8x8 = dm_full[np.ix_(idx, idx)]

print("PYSEQM density matrix P (8x8, CH4 RM1):")
labels = ['Cs', 'Cpx', 'Cpy', 'Cpz', 'H1s', 'H2s', 'H3s', 'H4s']
for i in range(8):
    row = " ".join(f"{dm_8x8[i,j]:8.5f}" for j in range(8))
    print(f"  {labels[i]:>4s} [{row}]")

# Eigenvalues
evals = mol.e_mo[0].detach().numpy()
print(f"\nPYSEQM eigenvalues (all 20, non-zero are first 8):")
active_evals = evals[:8]
print(f"  {active_evals}")

# MO coefficients
mo = mol.molecular_orbitals[0].detach().numpy()
print(f"\nPYSEQM MO coefficients (8x8):")
for i in range(8):
    row = " ".join(f"{mo[i,j]:8.5f}" for j in range(8))
    print(f"  [{row}]")

# Two-electron integrals w
w = mol.w.detach().numpy()  # (10, 10, 10) - 10 pairs, each pair has 10x10 unique integrals
print(f"\nPYSEQM w tensor shape: {w.shape}")
print(f"Pair indices (idxi, idxj):")
for k in range(10):
    i = mol.idxi[k].item()
    j = mol.idxj[k].item()
    print(f"  pair {k}: ({i},{j}) w[0,0]={(w[k,0,0]):.6f}")

# Now run OUR calculation and compare
from mlxmolkit.rm1.scf import rm1_energy, _build_basis_info, _build_core_hamiltonian, _build_fock
from mlxmolkit.rm1.params import RM1_PARAMS

atoms = [6, 1, 1, 1, 1]
coords = np.array([[0.0, 0.0, 0.0],
                    [0.6276, 0.6276, 0.6276],
                    [0.6276, -0.6276, -0.6276],
                    [-0.6276, 0.6276, -0.6276],
                    [-0.6276, -0.6276, 0.6276]])

result = rm1_energy(atoms, coords, max_iter=200, conv_tol=1e-8)
print(f"\n{'='*60}")
print(f"Our CH4 RM1: E_elec={result['electronic_eV']:.8f}, E_nuc={result['nuclear_eV']:.8f}")

# Compare density matrices
info = _build_basis_info(atoms)
our_P = result['density']
print(f"\nOur density matrix P (8x8):")
for i in range(8):
    row = " ".join(f"{our_P[i,j]:8.5f}" for j in range(8))
    print(f"  {labels[i]:>4s} [{row}]")

# Density matrix difference
print(f"\nDensity matrix difference (ours - PYSEQM):")
P_diff = our_P - dm_8x8
max_diff = np.max(np.abs(P_diff))
print(f"  Max |diff| = {max_diff:.6f}")
for i in range(8):
    for j in range(8):
        if abs(P_diff[i,j]) > 0.001:
            print(f"  P[{labels[i]},{labels[j]}]: ours={our_P[i,j]:.5f}  PYSEQM={dm_8x8[i,j]:.5f}  diff={P_diff[i,j]:.5f}")

# Compare eigenvalues
our_evals = result['eigenvalues']
print(f"\nOur eigenvalues: {our_evals}")
print(f"PYSEQM eigenvalues: {active_evals}")
print(f"Eigenvalue diff: {our_evals - active_evals}")
