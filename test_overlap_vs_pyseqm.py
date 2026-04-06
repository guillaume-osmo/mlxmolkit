#!/usr/bin/env python3
"""
Test our overlap implementation against PYSEQM PM6_SP.
Uses IDENTICAL zeta parameters (our RM1 zetas fed directly to PYSEQM).
"""
import sys
import numpy as np

sys.path.insert(0, '/Users/tgg/Github/mlxmolkit_phase1')
sys.path.insert(0, '/Users/tgg/Github/pyseqm_ref')

from mlxmolkit.rm1.params import RM1_PARAMS, ANG_TO_BOHR
from mlxmolkit.rm1.overlap import overlap_molecular_frame

import torch
from seqm.seqm_functions.diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP


def get_pyseqm_overlap_with_zetas(Z_A, Z_B, coordA, coordB, zs_A, zp_A, zs_B, zp_B):
    """Call PYSEQM with explicit zeta values."""
    ni = torch.tensor([Z_A])
    nj = torch.tensor([Z_B])

    cA = np.array(coordA)
    cB = np.array(coordB)
    R_vec = cB - cA
    R = np.linalg.norm(R_vec)
    R_bohr = R * ANG_TO_BOHR

    xij = torch.tensor(R_vec / R, dtype=torch.float64).unsqueeze(0)
    rij = torch.tensor([R_bohr], dtype=torch.float64)

    zeta_a = torch.tensor([[zs_A, zp_A]], dtype=torch.float64)
    zeta_b = torch.tensor([[zs_B, zp_B]], dtype=torch.float64)

    qn_int_dict = {1: 1, 2: 1, 6: 2, 7: 2, 8: 2, 9: 2, 15: 2, 16: 2, 17: 2, 35: 2, 53: 2}
    max_Z = max(Z_A, Z_B, 53)
    qn_int = torch.zeros(max_Z + 1, dtype=torch.long)
    for z, qn in qn_int_dict.items():
        if z <= max_Z:
            qn_int[z] = qn

    di = diatom_overlap_matrix_PM6_SP(ni, nj, xij, rij, zeta_a, zeta_b, qn_int)
    return di[0].numpy()


def test_pair(Z_A, Z_B, coordA, coordB, label):
    """Compare with identical zeta parameters."""
    pA = RM1_PARAMS[Z_A]
    pB = RM1_PARAMS[Z_B]

    # Our implementation
    our_di = overlap_molecular_frame(pA, pB, np.array(coordA), np.array(coordB))

    # PYSEQM with our EXACT zeta values
    qnA = 1 if Z_A <= 2 else 2
    qnB = 1 if Z_B <= 2 else 2

    # PYSEQM requires heavier atom first
    if qnA < qnB:
        # Swap for PYSEQM call
        pyseqm_di = get_pyseqm_overlap_with_zetas(
            Z_B, Z_A, coordB, coordA,
            pB.zeta_s, pB.zeta_p, pA.zeta_s, pA.zeta_p
        )
        pyseqm_di = pyseqm_di.T  # transpose back
    else:
        pyseqm_di = get_pyseqm_overlap_with_zetas(
            Z_A, Z_B, coordA, coordB,
            pA.zeta_s, pA.zeta_p, pB.zeta_s, pB.zeta_p
        )

    # Pad our result to 4x4
    our_4x4 = np.zeros((4, 4))
    nA, nB = our_di.shape
    our_4x4[:nA, :nB] = our_di

    diff = np.abs(our_4x4 - pyseqm_di)
    max_diff = np.max(diff)

    if max_diff < 1e-6:
        print(f"  ✅ {label}: MATCH (max diff = {max_diff:.2e})")
        return True
    else:
        print(f"  ❌ {label}: MISMATCH (max diff = {max_diff:.2e})")
        labels = ['s', 'px', 'py', 'pz']
        n = max(nA, nB)
        for i in range(n):
            for j in range(n):
                if diff[i,j] > 1e-8:
                    print(f"      [{labels[i]},{labels[j]}] PYSEQM={pyseqm_di[i,j]:12.8f}  Ours={our_4x4[i,j]:12.8f}  diff={diff[i,j]:.2e}")
        return False


def main():
    print("Comparing overlap: Ours vs PYSEQM (SAME zeta values)")
    print("="*60)

    results = []

    # H-H along x
    results.append(test_pair(1, 1, [0,0,0], [0.74,0,0], "H-H along x"))
    # C-H along x (jcall=3)
    results.append(test_pair(6, 1, [0,0,0], [1.09,0,0], "C-H along x"))
    # H-C along x (swap)
    results.append(test_pair(1, 6, [0,0,0], [1.09,0,0], "H-C along x (swap)"))
    # C-C along x (jcall=4)
    results.append(test_pair(6, 6, [0,0,0], [1.54,0,0], "C-C along x"))
    # C-N along x (heteroatomic jcall=4)
    results.append(test_pair(6, 7, [0,0,0], [1.47,0,0], "C-N along x"))
    # C-H along z (rotation)
    results.append(test_pair(6, 1, [0,0,0], [0,0,1.09], "C-H along z"))
    # C-H along diagonal
    d = 1.09 / np.sqrt(3)
    results.append(test_pair(6, 1, [0,0,0], [d,d,d], "C-H diagonal"))
    # C-O along y
    results.append(test_pair(6, 8, [0,0,0], [0,1.43,0], "C-O along y"))
    # N-H along z
    results.append(test_pair(7, 1, [0,0,0], [0,0,1.01], "N-H along z"))
    # O-H arbitrary direction
    results.append(test_pair(8, 1, [0,0,0], [0.5,0.7,0.3], "O-H arbitrary"))
    # C-O along x (key heteroatomic test)
    results.append(test_pair(6, 8, [0,0,0], [1.43,0,0], "C-O along x"))
    # N-O along x
    results.append(test_pair(7, 8, [0,0,0], [1.40,0,0], "N-O along x"))

    print(f"\n{'='*60}")
    n_pass = sum(results)
    print(f"  Results: {n_pass}/{len(results)} tests passed")
    if n_pass == len(results):
        print("  ✅ ALL TESTS PASS!")
    else:
        print(f"  ❌ {len(results) - n_pass} tests FAILED")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
