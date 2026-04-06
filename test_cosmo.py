#!/usr/bin/env python3
"""Test COSMO-RS pipeline end-to-end."""
import sys; sys.path.insert(0, '.')
import numpy as np

from mlxmolkit.cosmo import smiles_to_cosmo, activity_coefficients_from_smiles

print("=" * 60)
print("  COSMO-RS Pipeline: SMILES → sigma profile")
print("=" * 60)

# Test 1: Water COSMO surface
print("\n--- Water ---")
result = smiles_to_cosmo("O")
if result:
    print(f"  n_atoms = {result['n_atoms']}")
    print(f"  n_seg = {result['n_seg']}")
    print(f"  cavity_area = {result['cavity_area']:.2f} Å²")
    print(f"  cavity_volume = {result['cavity_volume']:.2f} Å³")
    print(f"  Mulliken charges: {result['mulliken_charges']}")
    print(f"  Total surface charge: {np.sum(result['seg_charge']):.6f} e")
    print(f"  sigma range: [{result['seg_sigma'].min():.4f}, {result['seg_sigma'].max():.4f}] e/Å²")
    print(f"  sigma_moment_0 (area) = {result['sigma_moment_0']:.2f}")
    print(f"  sigma_moment_2 (polarity) = {result['sigma_moment_2']:.6f}")
else:
    print("  FAILED!")

# Test 2: Methanol
print("\n--- Methanol ---")
result = smiles_to_cosmo("CO")
if result:
    print(f"  n_seg = {result['n_seg']}, area = {result['cavity_area']:.2f} Å²")
    print(f"  Mulliken: {result['mulliken_charges']}")
    print(f"  sigma range: [{result['seg_sigma'].min():.4f}, {result['seg_sigma'].max():.4f}]")
else:
    print("  FAILED!")

# Test 3: Ethanol
print("\n--- Ethanol ---")
result = smiles_to_cosmo("CCO")
if result:
    print(f"  n_seg = {result['n_seg']}, area = {result['cavity_area']:.2f} Å²")
    print(f"  sigma range: [{result['seg_sigma'].min():.4f}, {result['seg_sigma'].max():.4f}]")

# Test 4: Benzene
print("\n--- Benzene ---")
result = smiles_to_cosmo("c1ccccc1")
if result:
    print(f"  n_seg = {result['n_seg']}, area = {result['cavity_area']:.2f} Å²")
    print(f"  sigma range: [{result['seg_sigma'].min():.4f}, {result['seg_sigma'].max():.4f}]")

# Test 5: Activity coefficients for water-ethanol
print("\n" + "=" * 60)
print("  Activity Coefficients: Water-Ethanol")
print("=" * 60)

try:
    result = activity_coefficients_from_smiles(
        ["O", "CCO"],
        x=np.array([0.5, 0.5]),
        T=298.15,
    )
    print(f"  x = {result['x']}")
    print(f"  T = {result['T']} K")
    print(f"  ln(γ) = {result['lng']}")
    print(f"  γ = {result['gamma']}")
    print(f"  Water γ = {result['gamma'][0]:.4f}")
    print(f"  Ethanol γ = {result['gamma'][1]:.4f}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback; traceback.print_exc()

# Test 6: Different compositions
print("\n--- Water-Ethanol at different compositions ---")
try:
    for x_water in [0.1, 0.3, 0.5, 0.7, 0.9]:
        x = np.array([x_water, 1 - x_water])
        result = activity_coefficients_from_smiles(["O", "CCO"], x=x, T=298.15)
        print(f"  x_H2O={x_water:.1f}: γ_H2O={result['gamma'][0]:.3f}, γ_EtOH={result['gamma'][1]:.3f}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback; traceback.print_exc()
