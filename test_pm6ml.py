#!/usr/bin/env python3
"""Test PM6-ML model loading and correction."""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")

# Step 1: Compute PM6 energies (no torch import yet)
import numpy as np
from mlxmolkit.rm1.scf import rm1_energy
from mlxmolkit.rm1.pipeline import _smiles_to_3d

pm6_data = {}
for smi, name in [('O','Water'), ('CCO','Ethanol'), ('c1ccccc1','Benzene'), ('N','NH3')]:
    atoms, coords = _smiles_to_3d(smi)
    r = rm1_energy(list(atoms), coords, method='PM6')
    pm6_data[name] = {'atoms': list(atoms), 'coords': coords, 'energy': r['energy_eV'], 'hf': r['heat_of_formation_kcal']}

# Step 2: Load torch model and compute corrections
import torch
from torchmdnet.models.model import load_model

model = load_model("/Users/tgg/Github/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt", derivative=False)
model.eval()

Z_TO_ATYPE = {35:1, 6:3, 20:5, 17:7, 9:9, 1:10, 53:12, 19:13, 3:14, 12:15, 7:17, 11:19, 8:21, 15:23, 16:26}

print("PM6-ML Energy Corrections:")
print(f"{'Mol':>8s} {'PM6(eV)':>10s} {'ML_corr(eV)':>12s} {'PM6-ML(eV)':>12s} {'Hf_PM6':>8s}")
print("-" * 55)

for name, data in pm6_data.items():
    types = torch.tensor([Z_TO_ATYPE[z] for z in data['atoms']], dtype=torch.long)
    pos = torch.tensor(data['coords'].astype(np.float32))

    with torch.no_grad():
        out = model(types, pos)
    energy = out[0] if isinstance(out, tuple) else out
    e_corr_ev = energy.item() / 96.485
    total = data['energy'] + e_corr_ev

    print(f"{name:>8s} {data['energy']:>10.4f} {e_corr_ev:>+12.4f} {total:>12.4f} {data['hf']:>8.1f}")
