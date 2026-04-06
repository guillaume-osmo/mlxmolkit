#!/usr/bin/env python3
"""Run PYSEQM to get reference energies."""
import sys
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

import torch
import numpy as np

# Use PYSEQM API
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
from seqm.seqm_functions.energy import Energy

molecules = {
    'H2': {
        'species': torch.tensor([[1, 1, 0, 0, 0]]),
        'coords': torch.tensor([[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [0,0,0], [0,0,0], [0,0,0]]], dtype=torch.float64),
    },
    'H2O': {
        'species': torch.tensor([[8, 1, 1, 0, 0]]),
        'coords': torch.tensor([[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.2396, 0.9275, 0.0], [0,0,0], [0,0,0]]], dtype=torch.float64),
    },
    'CH4': {
        'species': torch.tensor([[6, 1, 1, 1, 1]]),
        'coords': torch.tensor([[[0.0, 0.0, 0.0], [0.6276, 0.6276, 0.6276],
                                   [0.6276, -0.6276, -0.6276], [-0.6276, 0.6276, -0.6276],
                                   [-0.6276, -0.6276, 0.6276]]], dtype=torch.float64),
    },
    'NH3': {
        'species': torch.tensor([[7, 1, 1, 1, 0]]),
        'coords': torch.tensor([[[0.0, 0.0, 0.0], [0.9377, -0.3816, 0.0],
                                   [-0.4689, 0.8119, 0.0], [-0.4689, -0.4303, 0.8299],
                                   [0,0,0]]], dtype=torch.float64),
    },
}

for method in ['RM1', 'AM1']:
    print(f"\n{'='*60}")
    print(f"  PYSEQM {method} Reference Energies")
    print(f"{'='*60}")
    for name, mol in molecules.items():
        try:
            species = mol['species']
            coords = mol['coords']
            elements = sorted(set(species[0].tolist()) - {0})

            const = Constants()
            p = params(method=method, elements=[0]+elements,
                      root_dir='/Users/tgg/Github/pyseqm_ref/seqm/params/')

            eng = Energy(const, method, species, p)
            result = eng(coords)

            e_tot = result[0].item()  # total energy in eV
            # Try to get components
            print(f"\n{name} ({method}):")
            print(f"  E_total = {e_tot:.6f} eV = {e_tot*23.061:.2f} kcal/mol")
        except Exception as e:
            print(f"\n{name} ({method}): ERROR - {e}")
            import traceback; traceback.print_exc()
