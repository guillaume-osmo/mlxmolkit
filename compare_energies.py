#!/usr/bin/env python3
"""Compare our RM1 energies vs PYSEQM RM1 reference."""
import sys; sys.path.insert(0, '.')
import numpy as np
from mlxmolkit.rm1.scf import rm1_energy

pyseqm_rm1 = {
    'H2':  {'E_elec': -42.27915, 'E_nuc': 13.78091, 'E_tot': -28.49824, 'Hf': -1.344},
    'H2O': {'E_elec': -488.95138, 'E_nuc': 143.38085, 'E_tot': -345.57053, 'Hf': -57.826},
    'CH4': {'E_elec': -391.80408, 'E_nuc': 209.04487, 'E_tot': -182.75921, 'Hf': -13.873},
    'NH3': {'E_elec': -439.11996, 'E_nuc': 186.81211, 'E_tot': -252.30785, 'Hf': 7.836},
}

mols = {
    'H2':  ([1,1], [[0,0,0],[0.74,0,0]]),
    'H2O': ([8,1,1], [[0,0,0],[0.9584,0,0],[-0.2396,0.9275,0]]),
    'CH4': ([6,1,1,1,1], [[0,0,0],[0.6276,0.6276,0.6276],[0.6276,-0.6276,-0.6276],[-0.6276,0.6276,-0.6276],[-0.6276,-0.6276,0.6276]]),
    'NH3': ([7,1,1,1], [[0,0,0],[0.9377,-0.3816,0],[-0.4689,0.8119,0],[-0.4689,-0.4303,0.8299]]),
}

hdr = "  Mol     dE_elec(eV)    dE_nuc(eV)    dE_tot(eV)    dHf(kcal)"
print(hdr)
print("-" * len(hdr))

for name in ['H2', 'H2O', 'CH4', 'NH3']:
    atoms, coords = mols[name]
    r = rm1_energy(atoms, np.array(coords), max_iter=200, conv_tol=1e-8)
    ref = pyseqm_rm1[name]
    de = r['electronic_eV'] - ref['E_elec']
    dn = r['nuclear_eV'] - ref['E_nuc']
    dt = r['energy_eV'] - ref['E_tot']
    dh = r['heat_of_formation_kcal'] - ref['Hf']
    print(f"  {name:3s}    {de:+10.4f}      {dn:+10.4f}      {dt:+10.4f}      {dh:+10.2f}")
    print(f"        ours={r['electronic_eV']:.4f}  ref={ref['E_elec']:.4f}  |  ours={r['nuclear_eV']:.4f}  ref={ref['E_nuc']:.4f}")
