"""
Run PYSEQM single-point energy calculations for H2, H2O, CH4, NH3
using both AM1 and RM1 methods.

Reports: E_elec (eV), E_nuc (eV), E_total (eV), Heat of formation (kcal/mol)
"""
import sys
sys.path.insert(0, "/Users/tgg/Github/pyseqm_ref")

import warnings
warnings.filterwarnings("ignore")

import torch
from seqm.api import Constants, Electronic_Structure, Molecule

torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# Conversion factor: 1 eV = 23.061 kcal/mol (MOPAC convention)
EV_TO_KCAL = 23.061

# Define molecules
molecules_data = {
    "H2": {
        "species": [1, 1],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.74, 0.0, 0.0],
        ],
    },
    "H2O": {
        "species": [8, 1, 1],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.9584, 0.0, 0.0],
            [-0.2396, 0.9275, 0.0],
        ],
    },
    "CH4": {
        "species": [6, 1, 1, 1, 1],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.6276, 0.6276, 0.6276],
            [-0.6276, -0.6276, 0.6276],
            [-0.6276, 0.6276, -0.6276],
            [0.6276, -0.6276, -0.6276],
        ],
    },
    "NH3": {
        "species": [7, 1, 1, 1],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.9377, -0.3816, 0.0],
            [-0.4689, 0.8119, 0.0],
            [-0.4689, -0.4303, 0.8299],
        ],
    },
}

methods = ["AM1", "RM1"]

print("=" * 80)
print("PYSEQM Reference Energies")
print("=" * 80)

results = {}

for method in methods:
    print(f"\n{'='*80}")
    print(f"Method: {method}")
    print(f"{'='*80}")
    results[method] = {}

    for mol_name, mol_data in molecules_data.items():
        species = torch.as_tensor(
            [mol_data["species"]],
            dtype=torch.int64,
            device=device,
        )
        coordinates = torch.tensor(
            [mol_data["coordinates"]],
            device=device,
        )

        const = Constants().to(device)

        seqm_parameters = {
            "method": method,
            "scf_eps": 1.0e-8,
            "scf_converger": [2],
        }

        mol = Molecule(const, seqm_parameters, coordinates, species).to(device)
        esdriver = Electronic_Structure(seqm_parameters).to(device)
        esdriver(mol)

        E_elec = mol.Eelec.item()
        E_nuc = mol.Enuc.item()
        E_total = mol.Etot.item()
        Hf_eV = mol.Hf.item()
        Hf_kcal = Hf_eV * EV_TO_KCAL

        results[method][mol_name] = {
            "E_elec_eV": E_elec,
            "E_nuc_eV": E_nuc,
            "E_total_eV": E_total,
            "Hf_eV": Hf_eV,
            "Hf_kcal": Hf_kcal,
        }

        print(f"\n  {mol_name}:")
        print(f"    E_elec  = {E_elec:16.8f} eV")
        print(f"    E_nuc   = {E_nuc:16.8f} eV")
        print(f"    E_total = {E_total:16.8f} eV")
        print(f"    Hf      = {Hf_eV:16.8f} eV  =  {Hf_kcal:12.6f} kcal/mol")

# Summary table
print(f"\n\n{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}")

header = f"{'Molecule':<8} {'Method':<6} {'E_elec (eV)':>16} {'E_nuc (eV)':>16} {'E_total (eV)':>16} {'Hf (kcal/mol)':>16}"
print(header)
print("-" * len(header))

for method in methods:
    for mol_name in molecules_data:
        r = results[method][mol_name]
        print(
            f"{mol_name:<8} {method:<6} {r['E_elec_eV']:>16.8f} {r['E_nuc_eV']:>16.8f} "
            f"{r['E_total_eV']:>16.8f} {r['Hf_kcal']:>16.6f}"
        )
    print()
