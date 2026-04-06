#!/usr/bin/env python3
"""Test SMILES → RDKit 3D → RM1 pipeline."""
import sys; sys.path.insert(0, '.')
import time
import numpy as np

from mlxmolkit.rm1.pipeline import rm1_from_smiles, rm1_from_smiles_batch

# Single molecule tests
print("=== Single SMILES → RM1 ===")
test_smiles = {
    'water': 'O',
    'methane': 'C',
    'ethanol': 'CCO',
    'benzene': 'c1ccccc1',
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'caffeine': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
}

for name, smi in test_smiles.items():
    r = rm1_from_smiles(smi)
    if r:
        print(f"  {name:12s} ({smi:30s}): {r['n_atoms']:3d} atoms, "
              f"Hf={r['heat_of_formation_kcal']:8.1f} kcal/mol, "
              f"conv={r['converged']}, {r['n_iter']} iters")
    else:
        print(f"  {name:12s} ({smi:30s}): FAILED")

# Batch test
print("\n=== Batch SMILES → RM1 (CPU) ===")
smiles_batch = ['O', 'C', 'CC', 'CCC', 'CCO', 'c1ccccc1',
                'CC(=O)O', 'CC=O', 'CCCC', 'C(=O)O',
                'c1ccc(O)cc1', 'CCN', 'CC(C)C', 'CCCCC',
                'c1ccc(N)cc1', 'CC(=O)Oc1ccccc1C(=O)O']

t0 = time.time()
results = rm1_from_smiles_batch(smiles_batch, use_metal=False, verbose=True)
t_cpu = time.time() - t0

n_ok = sum(1 for r in results if r is not None)
n_conv = sum(1 for r in results if r and r['converged'])
print(f"\n  {n_ok}/{len(smiles_batch)} computed, {n_conv} converged")
print(f"  Time: {t_cpu:.3f}s ({n_ok/t_cpu:.0f} mol/s)")

for i, (smi, r) in enumerate(zip(smiles_batch, results)):
    if r:
        print(f"    {smi:30s}: {r['n_atoms']:3d} atoms, "
              f"Hf={r['heat_of_formation_kcal']:8.1f} kcal/mol")

# Batch with Metal
print("\n=== Batch SMILES → RM1 (Metal GPU) ===")
t0 = time.time()
results_metal = rm1_from_smiles_batch(smiles_batch, use_metal=True, verbose=True)
t_metal = time.time() - t0

n_ok_m = sum(1 for r in results_metal if r is not None)
n_conv_m = sum(1 for r in results_metal if r and r['converged'])
print(f"\n  {n_ok_m}/{len(smiles_batch)} computed, {n_conv_m} converged")
print(f"  Time: {t_metal:.3f}s ({n_ok_m/t_metal:.0f} mol/s)")

# Compare CPU vs Metal
print("\n=== CPU vs Metal comparison ===")
for smi, rc, rm in zip(smiles_batch, results, results_metal):
    if rc and rm:
        dE = abs(rc['heat_of_formation_kcal'] - rm['heat_of_formation_kcal'])
        print(f"  {smi:30s}: dHf = {dE:.4f} kcal/mol")

# Larger batch benchmark
print("\n=== Large batch: 100 drug-like SMILES (CPU) ===")
drug_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CC=O', 'CCCC', 'c1ccc(O)cc1',
               'CCN', 'CC(C)C', 'CCCCC', 'c1ccc(N)cc1'] * 10

t0 = time.time()
results_100 = rm1_from_smiles_batch(drug_smiles, use_metal=False, verbose=True)
t_100 = time.time() - t0
n_ok_100 = sum(1 for r in results_100 if r is not None and r['converged'])
print(f"  {n_ok_100}/{len(drug_smiles)} converged in {t_100:.2f}s ({n_ok_100/t_100:.0f} mol/s)")
