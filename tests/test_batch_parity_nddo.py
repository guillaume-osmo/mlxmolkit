"""NDDO: a molecule's result must not depend on what it is batched with.

``nddo_energy_batch(use_metal=True)`` buckets molecules by basis size, pads
each bucket to a common basis and runs the SCF in float32 on the GPU: basis
<= 32 in "eigh" mode, larger buckets in "sign" mode, where molecules sharing a
bucket also share DIIS resets and the sign-settled flag. Existing tests compare
the CPU batch, and small eigh-mode hydrides, with the scalar path; nothing
compared sign mode, multi-bucket batches, charged molecules in a padded
bucket, the batched gradient, or batched AM1-BCC.

Tolerances are measured, not guessed. The reference is the float64 CPU SCF;
the Metal batch is float32 (conv_tol raised to >= 1e-5), and its energy error
is float32 accumulation, so it scales with the energy: up to 7e-7 |E|
(ibuprofen, 1.75e-3 eV of 2554 eV) while the CPU batch matches to 1e-12. The
energy bound is therefore relative, 2e-6 |E|. Charges agree to 7e-5 e and
gradients to 9e-5 eV/A. Padding leaking into a molecule, or a result landing
in the wrong slot, is orders of magnitude larger.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.am1bcc import (
    BCCParameter,
    am1_bcc_charges_from_rdkit_mol,
    am1_bcc_charges_from_rdkit_mols,
)
from mlxmolkit.nddo import nddo_energy, nddo_energy_batch, nddo_gradient
from mlxmolkit.nddo.gradient import nddo_gradient_batch
from mlxmolkit.nddo.pipeline import rm1_from_smiles, rm1_from_smiles_batch
from tests._mlx_cache import poison_buffer_cache

# Basis sizes (sp methods): 6 and 19 share the eigh bucket; 34, 36, 41, 42 share
# the 48 sign bucket (41 is an anion); 60 and 78 sit alone in theirs.
SMILES = [
    "O",
    "CC(=O)[O-]",
    "Oc1ccccc1",
    "Cc1ccccc1",
    "[O-]C(=O)c1ccccc1",
    "OC(=O)c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
]
CHARGE_E, GRAD_EV_A = 2e-4, 5e-4
KCAL_PER_EV = 23.060548


def _energy_tol_ev(energy_ev):
    """float32 energy accumulation bound for the Metal batch (see docstring)."""
    return max(5e-4, 2e-6 * abs(energy_ev))


def _rdkit_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture(scope="module")
def rdkit_mols():
    return [_rdkit_mol(s) for s in SMILES]


@pytest.fixture(scope="module")
def molecules(rdkit_mols):
    atoms = [[a.GetAtomicNum() for a in m.GetAtoms()] for m in rdkit_mols]
    coords = [m.GetConformer().GetPositions() for m in rdkit_mols]
    charges = [Chem.GetFormalCharge(m) for m in rdkit_mols]
    return list(zip(atoms, coords)), charges


@pytest.mark.parametrize("method", ["PM6", "AM1"])
def test_metal_energy_batch_matches_single(molecules, method):
    mols, charges = molecules
    singles = [
        nddo_energy(a, x, method=method, molecular_charge=q)
        for (a, x), q in zip(mols, charges)
    ]

    poison_buffer_cache()
    batch = nddo_energy_batch(mols, method=method, molecular_charges=charges, use_metal=True)

    solvers = {b.get("density_solver") for b in batch}
    assert {"eigh", "sign"} <= solvers, solvers  # both code paths exercised
    for smi, s, b in zip(SMILES, singles, batch):
        assert b["converged"] and s["converged"], smi
        assert b["n_basis"] == s["n_basis"], smi
        tol = _energy_tol_ev(s["energy_eV"])
        np.testing.assert_allclose(b["energy_eV"], s["energy_eV"], atol=tol, err_msg=smi)
        np.testing.assert_allclose(
            b["heat_of_formation_kcal"], s["heat_of_formation_kcal"],
            atol=tol * KCAL_PER_EV, err_msg=smi,
        )
        np.testing.assert_allclose(b["charges"], s["charges"], atol=CHARGE_E, err_msg=smi)


def test_molecule_alone_in_its_bucket_is_bitwise_batch_invariant(molecules):
    """Aspirin and ibuprofen each own a bucket, so their neighbours elsewhere
    in the batch cannot touch them at all."""
    mols, charges = molecules
    batch = nddo_energy_batch(mols, method="PM6", molecular_charges=charges, use_metal=True)
    for i in (6, 7):
        alone = nddo_energy_batch([mols[i]], method="PM6", molecular_charges=[charges[i]], use_metal=True)[0]
        assert batch[i]["energy_eV"] == alone["energy_eV"], SMILES[i]
        np.testing.assert_array_equal(batch[i]["charges"], alone["charges"])


def test_gradient_batch_matches_single(molecules):
    mols, charges = molecules
    singles = [
        nddo_gradient(a, x, method="AM1", molecular_charge=q)
        for (a, x), q in zip(mols, charges)
    ]

    poison_buffer_cache()
    energies, gradients = nddo_gradient_batch(mols, method="AM1", molecular_charges=charges)

    for smi, (e_s, g_s), e_b, g_b in zip(SMILES, singles, energies, gradients):
        assert np.shape(g_b) == np.shape(g_s), smi
        np.testing.assert_allclose(e_b, e_s, atol=_energy_tol_ev(e_s), err_msg=smi)
        np.testing.assert_allclose(g_b, g_s, atol=GRAD_EV_A, err_msg=smi)


def test_am1_bcc_batch_matches_single(rdkit_mols):
    """Plumbing parity, independent of the (untracked) published BCC table:
    any parameter set must give the same per-molecule answer both ways."""
    params = [
        BCCParameter("[#6X4:1]-[#1:2]", 0.02),
        BCCParameter("[#6X3:1]-[#1:2]", 0.03),
        BCCParameter("[#6X3:1]=[#8X1:2]", -0.05),
        BCCParameter("[#6:1]-[#8X2H1:2]", 0.04),
        BCCParameter("[#8:1]-[#1:2]", -0.07),
    ]
    kw = dict(parameters=params, validate_bcc_coverage=False)
    singles = [am1_bcc_charges_from_rdkit_mol(m, **kw) for m in rdkit_mols]

    poison_buffer_cache()
    batch = am1_bcc_charges_from_rdkit_mols(rdkit_mols, **kw)

    for smi, s, b in zip(SMILES, singles, batch):
        np.testing.assert_array_equal(b.bcc_corrections, s.bcc_corrections, err_msg=smi)
        np.testing.assert_allclose(b.am1_charges, s.am1_charges, atol=CHARGE_E, err_msg=smi)
        np.testing.assert_allclose(b.charges, s.charges, atol=CHARGE_E, err_msg=smi)
        assert b.total_charge == s.total_charge
        np.testing.assert_allclose(b.charges.sum(), s.total_charge, atol=1e-6, err_msg=smi)


def test_smiles_batch_keeps_results_in_their_slots():
    """Failed entries are dropped before the SCF and results mapped back by
    index; every survivor must land where its SMILES was. Entry i is embedded
    with seed 42 + i, so that is the single-molecule seed to compare with."""
    smiles = [
        "CCO",
        "not_a_smiles",
        "Oc1ccccc1",
        "[U]",  # no AM1 parameters
        "CC(=O)[O-]",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    ]
    poison_buffer_cache()
    batch = rm1_from_smiles_batch(smiles, method="AM1")

    assert len(batch) == len(smiles)
    assert batch[1] is None and batch[3] is None
    for i in (0, 2, 4, 5):
        single = rm1_from_smiles(smiles[i], seed=42 + i, method="AM1")
        assert batch[i]["smiles"] == smiles[i]
        assert batch[i]["converged"] and single["converged"], smiles[i]
        np.testing.assert_allclose(
            batch[i]["energy_eV"], single["energy_eV"],
            atol=_energy_tol_ev(single["energy_eV"]), err_msg=smiles[i],
        )
