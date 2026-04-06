#!/usr/bin/env python3
"""Benchmark batch RM1 SCF."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from mlxmolkit.rm1.scf import rm1_energy, rm1_energy_batch

# CH4 template
ch4_atoms = [6, 1, 1, 1, 1]
ch4_coords = np.array([[0, 0, 0], [0.6276, 0.6276, 0.6276],
    [0.6276, -0.6276, -0.6276], [-0.6276, 0.6276, -0.6276], [-0.6276, -0.6276, 0.6276]])

# H2O template
h2o_atoms = [8, 1, 1]
h2o_coords = np.array([[0, 0, 0], [0.9584, 0, 0], [-0.2396, 0.9275, 0.0]])

# NH3 template
nh3_atoms = [7, 1, 1, 1]
nh3_coords = np.array([[0, 0, 0], [0.9377, -0.3816, 0], [-0.4689, 0.8119, 0], [-0.4689, -0.4303, 0.8299]])

for N in [10, 100, 1000]:
    # Mix of molecules
    mols = []
    for i in range(N):
        choice = i % 3
        if choice == 0:
            # Add random perturbation to coords
            c = ch4_coords + np.random.randn(5, 3) * 0.01
            mols.append((ch4_atoms, c))
        elif choice == 1:
            c = h2o_coords + np.random.randn(3, 3) * 0.01
            mols.append((h2o_atoms, c))
        else:
            c = nh3_coords + np.random.randn(4, 3) * 0.01
            mols.append((nh3_atoms, c))

    # Sequential (single-molecule loop)
    t0 = time.time()
    seq_results = [rm1_energy(a, c, max_iter=100, conv_tol=1e-6) for a, c in mols]
    t_seq = time.time() - t0

    # Batch CPU
    t0 = time.time()
    batch_results = rm1_energy_batch(mols, max_iter=100, conv_tol=1e-6, use_metal=False)
    t_batch = time.time() - t0

    # Verify
    max_de = max(abs(batch_results[i]['energy_eV'] - seq_results[i]['energy_eV']) for i in range(N))
    n_conv_seq = sum(1 for r in seq_results if r['converged'])
    n_conv_batch = sum(1 for r in batch_results if r['converged'])

    print(f"N={N:5d}: seq={t_seq:.3f}s  batch_cpu={t_batch:.3f}s  "
          f"speedup={t_seq/t_batch:.1f}x  max_dE={max_de:.2e}  "
          f"conv_seq={n_conv_seq}/{N}  conv_batch={n_conv_batch}/{N}")
