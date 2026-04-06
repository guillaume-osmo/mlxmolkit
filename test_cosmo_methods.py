#!/usr/bin/env python3
"""Test COSMO-RS with all methods including PM3. Compare charge polarization."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.cosmo import smiles_to_cosmo, activity_coefficients_from_smiles

# Step 1: Compare charge polarization across methods
print("=" * 80)
print("  CHARGE POLARIZATION: RM1 vs AM1 vs PM3")
print("=" * 80)

methods = ['RM1', 'AM1', 'PM3']
test_mols = [('O', 'Water'), ('CO', 'Methanol'), ('CCO', 'Ethanol'), ('c1ccccc1', 'Benzene')]

print(f"\n{'Molecule':>12s} {'Method':>8s} {'sigma_min':>10s} {'sigma_max':>10s} {'sigma_rms':>10s} {'area':>8s}")
print("-" * 65)

for smi, name in test_mols:
    for method in methods:
        try:
            r = smiles_to_cosmo(smi, method=method)
            if r:
                sig = r['seg_sigma']
                print(f"{name:>12s} {method:>8s} {sig.min():>10.5f} {sig.max():>10.5f} "
                      f"{np.sqrt(np.mean(sig**2)):>10.5f} {r['cavity_area']:>8.1f}")
        except Exception as e:
            print(f"{name:>12s} {method:>8s} ERROR: {e}")

# Step 2: Activity coefficients comparison
print(f"\n{'='*80}")
print("  ACTIVITY COEFFICIENTS: Water-Ethanol at x_H2O=0.5")
print(f"{'='*80}")

print(f"\n{'Method':>8s} {'gamma_H2O':>10s} {'gamma_EtOH':>10s}")
print("-" * 35)

for method in methods:
    try:
        r = activity_coefficients_from_smiles(["O", "CCO"], x=np.array([0.5, 0.5]),
                                               T=298.15, method=method)
        print(f"{method:>8s} {r['gamma'][0]:>10.4f} {r['gamma'][1]:>10.4f}")
    except Exception as e:
        print(f"{method:>8s} ERROR: {e}")

# Step 3: VLE-like curve (activity coefficients across compositions)
print(f"\n{'='*80}")
print("  Water-Ethanol Activity Coefficients vs Composition (PM3)")
print(f"{'='*80}")

# Experimental reference (approximate from Wilson model):
# At 298K: gamma_H2O(inf) ≈ 2.5, gamma_EtOH(inf) ≈ 5.9
exp_ref = {
    0.0: (2.5, 1.0),   # infinite dilution of water in ethanol
    0.2: (1.8, 1.1),
    0.5: (1.3, 1.5),
    0.8: (1.05, 2.5),
    1.0: (1.0, 5.9),   # infinite dilution of ethanol in water
}

print(f"\n{'x_H2O':>6s} {'gamma_H2O':>10s} {'gamma_EtOH':>10s} {'exp_H2O':>10s} {'exp_EtOH':>10s}")
print("-" * 50)

for x_w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    x = np.array([x_w, 1 - x_w])
    try:
        r = activity_coefficients_from_smiles(["O", "CCO"], x=x, T=298.15, method='PM3')
        print(f"{x_w:>6.1f} {r['gamma'][0]:>10.4f} {r['gamma'][1]:>10.4f}", end="")
        # Approximate experimental interpolation
        if x_w in exp_ref:
            print(f" {exp_ref[x_w][0]:>10.2f} {exp_ref[x_w][1]:>10.2f}")
        else:
            print()
    except Exception as e:
        print(f"{x_w:>6.1f} ERROR: {e}")

# Step 4: Benchmark speed
print(f"\n{'='*80}")
print("  BENCHMARK: COSMO-RS for 20 molecules")
print(f"{'='*80}")

drug_smiles = ['O', 'CO', 'CCO', 'CC=O', 'CC(=O)O', 'c1ccccc1',
               'CC', 'CCC', 'CCCC', 'CCCCC',
               'CN', 'CCN', 'C=O', 'N', 'CC(C)C',
               'c1ccc(O)cc1', 'c1ccc(N)cc1', 'CC(=O)C', 'CCCO', 'OC=O']

for method in methods:
    t0 = time.time()
    results = []
    for smi in drug_smiles:
        try:
            r = smiles_to_cosmo(smi, method=method)
            if r:
                results.append(r)
        except:
            pass
    t = time.time() - t0
    print(f"  {method}: {len(results)}/{len(drug_smiles)} molecules in {t:.2f}s ({len(results)/t:.0f} mol/s)")
