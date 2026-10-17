"""xTB: a molecule's result must not depend on what it is batched with.

``gfn0_energy_batch`` pads every molecule's (H, S) to the largest basis in the
batch (H = 0, S = I on the padding) for one batched generalized eigensolve.
Each padded row contributes an eigenpair with eigenvalue exactly 0, which
sorts in among the real orbital energies; slicing the first ``n_basis`` then
returned spurious zeros in ``eigenvalues`` and dropped the top virtuals for
every molecule smaller than the largest (water batched with ibuprofen: 2 zeros,
eigenvalues off by 0.22 Eh). ``eigenvalues[n_occ]`` read as the LUMO was 0.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.xtb.eeq import eeq_charges_and_energy
from mlxmolkit.xtb.energy import gfn0_energy, gfn0_energy_batch
from tests._mlx_cache import poison_buffer_cache

pytest.importorskip("mlx_addons.linalg", reason="gfn0_energy_batch needs mlx_addons")

# Different basis sizes and net charges; ibuprofen is the largest.
SMILES = ["O", "CC(=O)[O-]", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C[NH3+]"]


def _molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return atoms, mol.GetConformer().GetPositions(), Chem.GetFormalCharge(mol)


@pytest.fixture(scope="module")
def molecules():
    return [_molecule(s) for s in SMILES]


def test_gfn0_batch_matches_single(molecules):
    singles = [gfn0_energy(a, x, charge=q) for a, x, q in molecules]

    poison_buffer_cache()
    batch = gfn0_energy_batch([(a, x) for a, x, _ in molecules], charges=[q for *_, q in molecules])

    for smi, s, b in zip(SMILES, singles, batch):
        assert b["n_basis"] == s["n_basis"], smi
        # The batch solves in float32 (mlx_addons gen_eigh), the single path
        # in float64 (scipy): measured gaps are <= 4e-5 eV, 2e-6 in P.
        np.testing.assert_allclose(b["energy_eV"], s["energy_eV"], atol=1e-3, err_msg=smi)
        np.testing.assert_allclose(b["eigenvalues"], s["eigenvalues"], atol=1e-4, err_msg=smi)
        np.testing.assert_allclose(b["density"], s["density"], atol=1e-4, err_msg=smi)
        np.testing.assert_allclose(b["charges"], s["charges"], atol=1e-5, err_msg=smi)


def test_eeq_mixed_batch_matches_single(molecules):
    """Padded EEQ: charges and the EEQ energy, slot by slot."""
    neutral = [(a, x) for a, x, q in molecules if q == 0]
    n_max = max(len(a) for a, _ in neutral)

    singles = []
    for atoms, coords in neutral:
        q, e = eeq_charges_and_energy(
            mx.array(coords.astype(np.float32)), mx.array(np.array(atoms, np.int32))
        )
        mx.eval(q, e)
        singles.append((np.array(q), float(e)))

    Z = np.zeros((len(neutral), n_max), np.int32)
    X = np.zeros((len(neutral), n_max, 3), np.float32)
    for i, (atoms, coords) in enumerate(neutral):
        Z[i, : len(atoms)] = atoms
        X[i, : len(atoms)] = coords
    n_atoms = mx.array([len(a) for a, _ in neutral], dtype=mx.int32)

    poison_buffer_cache()
    q_b, e_b = eeq_charges_and_energy(mx.array(X), mx.array(Z), n_atoms_arr=n_atoms)
    mx.eval(q_b, e_b)
    q_b, e_b = np.array(q_b), np.array(e_b)

    for i, ((atoms, _), (q_s, e_s)) in enumerate(zip(neutral, singles)):
        n = len(atoms)
        np.testing.assert_allclose(q_b[i, :n], q_s, atol=1e-5)
        np.testing.assert_array_equal(q_b[i, n:], 0.0)
        np.testing.assert_allclose(e_b[i], e_s, rtol=1e-5, atol=1e-6)
