#!/usr/bin/env python3
"""Tune COSMO-RS alpha parameter for semi-empirical charges.

The RM1/PM3 Mulliken charges are ~5x less polarized than DFT-CPCM.
We need to scale alpha (misfit energy) to compensate.

Target: Water-Ethanol activity coefficients
  Experimental: gamma_H2O(inf) ≈ 2.5, gamma_EtOH(inf) ≈ 5.9
  At x=0.5: gamma_H2O ≈ 1.3, gamma_EtOH ≈ 1.5
"""
import sys; sys.path.insert(0, '.')
import numpy as np

from mlxmolkit.cosmo.pipeline import smiles_to_cosmo
from mlxmolkit.cosmo.cosmors import activity_coefficients
import mlxmolkit.cosmo.params as cosmo_params

# Pre-compute COSMO surfaces (once)
print("Pre-computing COSMO surfaces...")
h2o = smiles_to_cosmo("O", method='PM3')
etoh = smiles_to_cosmo("CCO", method='PM3')
print(f"  Water: sigma_rms = {np.sqrt(np.mean(h2o['seg_sigma']**2)):.5f}")
print(f"  Ethanol: sigma_rms = {np.sqrt(np.mean(etoh['seg_sigma']**2)):.5f}")

# Scan alpha values
print(f"\n{'alpha':>12s} {'gamma_H2O':>10s} {'gamma_EtOH':>10s}")
print("-" * 35)

# Original alpha: 7.579075e6 → gamma ≈ 0.94 (too close to 1)
# Need ~10-50x larger alpha to get gamma ≈ 1.3-2.5
best_alpha = None
best_err = 1e10
target_g_h2o = 1.3  # at x=0.5
target_g_etoh = 1.5

for log_scale in np.arange(0, 3.0, 0.2):
    scale = 10 ** log_scale
    cosmo_params.MF_ALPHA = 7.579075e6 * scale

    try:
        lng = activity_coefficients([h2o, etoh], np.array([0.5, 0.5]), T=298.15)
        g = np.exp(lng)
        err = abs(g[0] - target_g_h2o) + abs(g[1] - target_g_etoh)

        marker = " <--" if err < best_err else ""
        if err < best_err:
            best_err = err
            best_alpha = cosmo_params.MF_ALPHA

        print(f"{cosmo_params.MF_ALPHA:>12.3e} {g[0]:>10.4f} {g[1]:>10.4f}{marker}")
    except:
        print(f"{cosmo_params.MF_ALPHA:>12.3e} ERROR")

# Also scan HB parameter
print(f"\nBest alpha: {best_alpha:.3e}")
cosmo_params.MF_ALPHA = best_alpha

print(f"\nScanning HB parameter (hb_c)...")
print(f"{'hb_c':>12s} {'gamma_H2O':>10s} {'gamma_EtOH':>10s}")
print("-" * 35)

best_hb = None
best_err = 1e10

for log_scale in np.arange(0, 3.0, 0.2):
    scale = 10 ** log_scale
    cosmo_params.HB_C = 2.7488747e7 * scale

    try:
        lng = activity_coefficients([h2o, etoh], np.array([0.5, 0.5]), T=298.15)
        g = np.exp(lng)
        err = abs(g[0] - target_g_h2o) + abs(g[1] - target_g_etoh)

        marker = " <--" if err < best_err else ""
        if err < best_err:
            best_err = err
            best_hb = cosmo_params.HB_C

        print(f"{cosmo_params.HB_C:>12.3e} {g[0]:>10.4f} {g[1]:>10.4f}{marker}")
    except:
        print(f"{cosmo_params.HB_C:>12.3e} ERROR")

print(f"\nOptimal: alpha={best_alpha:.3e}, hb_c={best_hb:.3e}")

# Final test with optimal parameters
cosmo_params.MF_ALPHA = best_alpha
cosmo_params.HB_C = best_hb

print(f"\n--- Water-Ethanol with tuned params (PM3) ---")
print(f"{'x_H2O':>6s} {'gamma_H2O':>10s} {'gamma_EtOH':>10s}")
for x_w in [0.1, 0.3, 0.5, 0.7, 0.9]:
    x = np.array([x_w, 1 - x_w])
    lng = activity_coefficients([h2o, etoh], x, T=298.15)
    g = np.exp(lng)
    print(f"{x_w:>6.1f} {g[0]:>10.4f} {g[1]:>10.4f}")

# Reset
cosmo_params.MF_ALPHA = 7.579075e6
cosmo_params.HB_C = 2.7488747e7
