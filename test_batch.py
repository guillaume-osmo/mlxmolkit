#!/usr/bin/env python3
"""Test batch RM1 against single-molecule results."""
import sys
sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.rm1.scf import rm1_energy, rm1_energy_batch

molecules = {
    'H2': ([1, 1], np.array([[0, 0, 0], [0.74, 0, 0.0]])),
    'H2O': ([8, 1, 1], np.array([[0, 0, 0], [0.9584, 0, 0], [-0.2396, 0.9275, 0.0]])),
    'CH4': ([6, 1, 1, 1, 1], np.array([[0, 0, 0], [0.6276, 0.6276, 0.6276],
             [0.6276, -0.6276, -0.6276], [-0.6276, 0.6276, -0.6276], [-0.6276, -0.6276, 0.6276]])),
    'NH3': ([7, 1, 1, 1], np.array([[0, 0, 0], [0.9377, -0.3816, 0],
             [-0.4689, 0.8119, 0], [-0.4689, -0.4303, 0.8299]])),
}

# Single molecule reference
print("Single molecule reference:")
single_results = {}
for name, (atoms, coords) in molecules.items():
    r = rm1_energy(atoms, coords, max_iter=200, conv_tol=1e-8)
    single_results[name] = r
    print(f"  {name}: E_tot={r['energy_eV']:.8f} eV, Hf={r['heat_of_formation_kcal']:.3f} kcal/mol, conv={r['converged']}")

# Batch (CPU)
print("\nBatch CPU:")
mol_list = [(atoms, coords) for atoms, coords in molecules.values()]
t0 = time.time()
batch_results = rm1_energy_batch(mol_list, max_iter=200, conv_tol=1e-8, use_metal=False, verbose=True)
t_cpu = time.time() - t0
print(f"  Time: {t_cpu:.3f}s")

# Compare
print("\nComparison (single vs batch CPU):")
print(f"  {'Mol':>4s}  {'dE_tot(eV)':>12s}  {'dHf(kcal)':>12s}  {'conv':>6s}")
print("-" * 50)
for i, name in enumerate(molecules):
    sr = single_results[name]
    br = batch_results[i]
    de = br['energy_eV'] - sr['energy_eV']
    dh = br['heat_of_formation_kcal'] - sr['heat_of_formation_kcal']
    print(f"  {name:>4s}  {de:+12.8f}  {dh:+12.6f}  {br['converged']}")

# Test with Metal
print("\nBatch Metal:")
try:
    t0 = time.time()
    batch_metal = rm1_energy_batch(mol_list, max_iter=200, conv_tol=1e-8, use_metal=True, verbose=True)
    t_metal = time.time() - t0
    print(f"  Time: {t_metal:.3f}s")

    print("\nComparison (single vs batch Metal):")
    print(f"  {'Mol':>4s}  {'dE_tot(eV)':>12s}  {'dHf(kcal)':>12s}  {'conv':>6s}")
    print("-" * 50)
    for i, name in enumerate(molecules):
        sr = single_results[name]
        br = batch_metal[i]
        de = br['energy_eV'] - sr['energy_eV']
        dh = br['heat_of_formation_kcal'] - sr['heat_of_formation_kcal']
        print(f"  {name:>4s}  {de:+12.8f}  {dh:+12.6f}  {br['converged']}")
except Exception as e:
    print(f"  Metal failed: {e}")
    import traceback; traceback.print_exc()
