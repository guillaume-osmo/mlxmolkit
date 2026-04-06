#!/usr/bin/env python3
"""Extract PYSEQM's Hcore matrix for H2O."""
import sys
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')
sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')

import torch
import numpy as np

# Use PYSEQM's full calculation pipeline
from seqm.basics import Pack_Parameters, Parser
from seqm.seqm_functions.parameters import params
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.hcore import hcore

# Load RM1 parameters
elements = [0, 1, 8]
p = params(method='RM1', elements=elements,
           root_dir='/Users/tgg/Github/pyseqm_ref/seqm/params/')

# H2O
species = torch.tensor([[8, 1, 1]], dtype=torch.long)
coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.9584, 0.0, 0.0],
                              [-0.2396, 0.9275, 0.0]]], dtype=torch.float64)

const = Constants()

# Extract parameters for H2O
# The packed parameters are indexed by atomic number
print("PYSEQM RM1 parameters:")
print(f"  H: {p[1].numpy()}")
print(f"  O: {p[8].numpy()}")

# Now let's use hcore function
# Need to understand the calling convention
import inspect
sig = inspect.signature(hcore)
print(f"\nhcore signature: {sig}")

# Try the PYSEQM approach from basics.py
# Let me look for how Pack_Parameters works
print("\nPYSEQM hcore function requires specific inputs.")
print("Let me trace through basics.py to understand the API...")

# Alternative: use the full seqm module
try:
    from seqm import SEQM_singlepoint
    sp = SEQM_singlepoint(species, coordinates, method='RM1')
    print(f"\nPYSEQM single-point energy: {sp}")
except Exception as e:
    print(f"SEQM_singlepoint failed: {e}")

# Try manual approach
try:
    from seqm.basics import Parser, Pack_Parameters

    # Create parser with proper settings
    settings = {
        'method': 'RM1',
        'scf_eps': 1.0e-8,
        'scf_converger': [2, 0.0],
        'sp2': [False, 1.0e-5],
        'elements': [0, 1, 8],
        'learned': [],
        'pair_outer_cutoff': 1.0e10,
    }

    parser = Parser(settings)
    pp = Pack_Parameters(parser, p)

    # Get molecule info
    nmol = 1
    molsize = 3

    # Pack the coordinates
    # In PYSEQM, coordinates are in Angstrom

    # Get the hcore
    # This is complex... let me try the high-level API instead
    from seqm import SEQM_singlepoint
except Exception as e:
    print(f"Manual approach failed: {e}")
    import traceback; traceback.print_exc()

# Simplest approach: just call the relevant functions directly
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from seqm.seqm_functions.two_elec_two_center_int import two_elec_two_center_int as pyseqm_two_elec
from seqm.seqm_functions.two_elec_two_center_int import rotate_with_quaternion
from mlxmolkit.rm1.params import RM1_PARAMS, ANG_TO_BOHR

# O-H1 pair
Z_O, Z_H = 8, 1
pO = RM1_PARAMS[Z_O]
pH = RM1_PARAMS[Z_H]

coordO = np.array([0.0, 0.0, 0.0])
coordH = np.array([0.9584, 0.0, 0.0])
R_vec = coordH - coordO
R = np.linalg.norm(R_vec)
R_bohr = R * ANG_TO_BOHR

# Get PYSEQM's two-electron integrals for O-H1
ni = torch.tensor([Z_O])
nj = torch.tensor([Z_H])
xij = torch.tensor(R_vec / R, dtype=torch.float64).unsqueeze(0)
rij = torch.tensor([R_bohr], dtype=torch.float64)

# Zeta from OUR RM1 params (same values)
zeta_a = torch.tensor([[pO.zeta_s, pO.zeta_p]], dtype=torch.float64)
zeta_b = torch.tensor([[pH.zeta_s, pH.zeta_p]], dtype=torch.float64)

qn_int = torch.zeros(100, dtype=torch.long)
qn_int[1] = 1; qn_int[6] = 2; qn_int[7] = 2; qn_int[8] = 2

# Overlap
di = diatom_overlap_matrix_PM6_SP(ni, nj, xij, rij, zeta_a, zeta_b, qn_int)
print(f"\nPYSEQM O-H overlap (4x4):")
print(di[0].numpy())

# Two-electron integrals
# Need: gss, gsp, gpp, gp2, hsp, alpha for both atoms
gss = torch.tensor([pO.gss, pH.gss], dtype=torch.float64)
gsp = torch.tensor([pO.gsp, 0.0], dtype=torch.float64)
gpp = torch.tensor([pO.gpp, 0.0], dtype=torch.float64)
gp2 = torch.tensor([pO.gp2, 0.0], dtype=torch.float64)
hsp = torch.tensor([pO.hsp, 0.0], dtype=torch.float64)

# Try calling two_elec function
print(f"\nPYSEQM two_elec_two_center_int signature:")
print(inspect.signature(pyseqm_two_elec))
