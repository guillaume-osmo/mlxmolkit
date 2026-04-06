#!/usr/bin/env python3
"""Benchmark batch RM1: CPU vs Metal GPU, apples-to-apples."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.rm1.scf import rm1_energy, rm1_energy_batch

np.random.seed(42)

# Templates
templates = {
    'CH4': ([6, 1, 1, 1, 1], np.array([[0,0,0],[.6276,.6276,.6276],[.6276,-.6276,-.6276],[-.6276,.6276,-.6276],[-.6276,-.6276,.6276]])),
    'H2O': ([8, 1, 1], np.array([[0,0,0],[.9584,0,0],[-.2396,.9275,0.0]])),
    'NH3': ([7, 1, 1, 1], np.array([[0,0,0],[.9377,-.3816,0],[-.4689,.8119,0],[-.4689,-.4303,.8299]])),
}

def make_batch(N):
    mols = []
    keys = list(templates.keys())
    for i in range(N):
        atoms, coords = templates[keys[i % 3]]
        c = coords + np.random.randn(*coords.shape) * 0.005
        mols.append((atoms, c))
    return mols

print("=" * 70)
print("  RM1 Batch Benchmark: CPU vs Metal GPU (MLX 0.31.1)")
print("=" * 70)

for N in [10, 50, 100, 500, 1000]:
    mols = make_batch(N)

    # Sequential (single-mol loop) — baseline
    if N <= 100:
        t0 = time.time()
        for a, c in mols:
            rm1_energy(a, c, max_iter=100, conv_tol=1e-6)
        t_seq = time.time() - t0
    else:
        t_seq = None

    # Batch CPU
    t0 = time.time()
    r_cpu = rm1_energy_batch(mols, max_iter=100, conv_tol=1e-6, use_metal=False)
    t_cpu = time.time() - t0
    n_conv_cpu = sum(1 for r in r_cpu if r['converged'])

    # Batch Metal GPU
    t0 = time.time()
    r_metal = rm1_energy_batch(mols, max_iter=100, conv_tol=1e-6, use_metal=True)
    t_metal = time.time() - t0
    n_conv_metal = sum(1 for r in r_metal if r['converged'])

    # Compare energies
    max_de = max(abs(r_cpu[i]['energy_eV'] - r_metal[i]['energy_eV'])
                 for i in range(N) if r_cpu[i]['converged'] and r_metal[i]['converged'])

    seq_str = f"{t_seq:.3f}s ({N/t_seq:.0f}/s)" if t_seq else "---"
    print(f"\n  N={N:5d}:")
    print(f"    Sequential : {seq_str}")
    print(f"    Batch CPU  : {t_cpu:.3f}s ({N/t_cpu:.0f} mol/s) [{n_conv_cpu}/{N} conv]")
    print(f"    Batch Metal: {t_metal:.3f}s ({N/t_metal:.0f} mol/s) [{n_conv_metal}/{N} conv]")
    if t_seq:
        print(f"    Speedup: CPU batch {t_seq/t_cpu:.1f}x, Metal {t_seq/t_metal:.1f}x vs sequential")
    else:
        print(f"    Speedup: Metal {t_cpu/t_metal:.1f}x vs CPU batch")
    print(f"    Max |dE| CPU vs Metal: {max_de:.2e} eV")
