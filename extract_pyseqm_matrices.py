#!/usr/bin/env python3
"""Extract PYSEQM matrices (Hcore, P, F) for CH4 RM1 comparison."""
import sys
sys.path.insert(0, "/Users/tgg/Github/pyseqm_ref")
sys.path.insert(0, "/Users/tgg/Github/mlxmolkit_phase1")

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from seqm.api import Constants, Electronic_Structure, Molecule

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# CH4 - use SAME coordinates as our code
species = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.int64, device=device)
coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.6276, 0.6276, 0.6276],
                              [0.6276, -0.6276, -0.6276],
                              [-0.6276, 0.6276, -0.6276],
                              [-0.6276, -0.6276, 0.6276]]], device=device)

const = Constants().to(device)
seqm_parameters = {"method": "RM1", "scf_eps": 1.0e-8, "scf_converger": [2]}

mol = Molecule(const, seqm_parameters, coordinates, species).to(device)
esdriver = Electronic_Structure(seqm_parameters).to(device)
esdriver(mol)

print(f"PYSEQM CH4 RM1:")
print(f"  E_elec = {mol.Eelec.item():.8f} eV")
print(f"  E_nuc  = {mol.Enuc.item():.8f} eV")
print(f"  E_tot  = {mol.Etot.item():.8f} eV")

# Extract matrices
print(f"\nAvailable mol attributes:")
for attr in sorted(dir(mol)):
    if not attr.startswith('_'):
        try:
            val = getattr(mol, attr)
            if isinstance(val, torch.Tensor):
                print(f"  {attr}: shape={val.shape}")
        except:
            pass

# Try to get Hcore (M), density (P), Fock (F)
for name in ['M', 'Hcore', 'hcore', 'F', 'Fock', 'P', 'density', 'P0', 'F0']:
    try:
        val = getattr(mol, name)
        if isinstance(val, torch.Tensor):
            print(f"\nmol.{name} shape={val.shape}:")
            v = val.detach().numpy()
            if v.ndim == 2:
                print(v)
            elif v.ndim == 3:
                print(v[0])  # first molecule
    except AttributeError:
        pass

# Try to find the matrices in the electronic structure driver
print(f"\nesdriver attributes:")
for attr in sorted(dir(esdriver)):
    if not attr.startswith('_'):
        try:
            val = getattr(esdriver, attr)
            if isinstance(val, torch.Tensor):
                print(f"  {attr}: shape={val.shape}")
        except:
            pass
