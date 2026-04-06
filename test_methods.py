#!/usr/bin/env python3
"""Test all NDDO methods: RM1, AM1, AM1*, RM1*."""
import sys; sys.path.insert(0, '.')
import numpy as np

from mlxmolkit.rm1.scf import rm1_energy
from mlxmolkit.rm1.pipeline import rm1_from_smiles

smiles_list = ['O', 'C', 'CCO', 'c1ccccc1', 'CC=O', 'CCN']
names = ['water', 'methane', 'ethanol', 'benzene', 'acetaldehyde', 'ethylamine']

methods = ['RM1', 'AM1', 'AM1_STAR', 'RM1_STAR']

print(f"{'Molecule':>15s}", end="")
for m in methods:
    print(f"  {m:>12s}", end="")
print("  (kcal/mol)")
print("-" * 75)

for smi, name in zip(smiles_list, names):
    print(f"{name:>15s}", end="")
    for method in methods:
        try:
            r = rm1_from_smiles(smi, method=method, max_iter=200)
            if r and r['converged']:
                print(f"  {r['heat_of_formation_kcal']:12.1f}", end="")
            elif r:
                print(f"  {'NC':>12s}", end="")
            else:
                print(f"  {'FAIL':>12s}", end="")
        except Exception as e:
            print(f"  {'ERR':>12s}", end="")
    print()
