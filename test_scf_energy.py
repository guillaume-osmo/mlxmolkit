#!/usr/bin/env python3
"""Test SCF energies after overlap fix. Compare RM1 vs PYSEQM."""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
from mlxmolkit.rm1.scf import rm1_energy

molecules = {
    'H2': {
        'atoms': [1, 1],
        'coords': np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
    },
    'H2O': {
        'atoms': [8, 1, 1],
        'coords': np.array([
            [0.0, 0.0, 0.0],
            [0.9584, 0.0, 0.0],
            [-0.2396, 0.9275, 0.0],
        ]),
    },
    'CH4': {
        'atoms': [6, 1, 1, 1, 1],
        'coords': np.array([
            [0.0, 0.0, 0.0],
            [0.6276, 0.6276, 0.6276],
            [0.6276, -0.6276, -0.6276],
            [-0.6276, 0.6276, -0.6276],
            [-0.6276, -0.6276, 0.6276],
        ]),
    },
    'NH3': {
        'atoms': [7, 1, 1, 1],
        'coords': np.array([
            [0.0, 0.0, 0.0],
            [0.9377, -0.3816, 0.0],
            [-0.4689, 0.8119, 0.0],
            [-0.4689, -0.4303, 0.8299],
        ]),
    },
}

print("RM1 SCF Energies (after PYSEQM PM6_SP overlap fix)")
print("="*60)
for name, mol in molecules.items():
    try:
        result = rm1_energy(
            mol['atoms'], mol['coords'],
            max_iter=200, conv_tol=1e-6,
            use_metal=False,
        )
        print(f"\n{name}:")
        print(f"  E_elec    = {result['electronic_eV']:.4f} eV")
        print(f"  E_nuc     = {result['nuclear_eV']:.4f} eV")
        print(f"  E_tot     = {result['energy_eV']:.4f} eV")
        print(f"  ΔHf       = {result['heat_of_formation_kcal']:.2f} kcal/mol")
        print(f"  converged = {result['converged']} in {result['n_iter']} iters")
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")
        import traceback; traceback.print_exc()
