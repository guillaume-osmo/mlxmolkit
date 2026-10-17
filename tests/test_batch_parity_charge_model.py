"""Charge model: a molecule's charges must not depend on what it is batched with.

``predict_partial_charges_batch_mlx`` pads every molecule to the widest one
(or to ``pad_to``) and runs one forward pass; ``predict_partial_charges_mlx``
runs one molecule alone. Padded atoms are masked out of the attention keys,
the residual stream and the total-charge projection, so each batch slot must
reproduce the molecule's own charges, padded atoms must come back exactly 0,
and each molecule must keep its own total charge. Both readouts are covered:
``direct`` (shift to the target total) and ``qeq`` (charge equilibration).

Not bitwise across widths: padding changes the length of the attention and
charge sums and the GEMM shapes, and four transformer layers carry the
rounding through, so charges move by up to 4.8e-7 (|q| <= 0.4, three seeds).
That figure is flat from 34 to 512 padded atoms, where a leak would grow with
the padding. Reordering a batch keeps every shape, and is bitwise; so is
filling the padded slots with junk. A masking or offset bug moves charges by
O(0.1) or yields NaN.

The model is randomly initialised from a fixed seed: parity does not need
trained weights, and random ones exercise every path just as well.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.charge_model import (
    ChargeModelConfig,
    GeometricChargePredictor,
    bond_matrix_from_rdkit_mol,
    charge_model_batch,
    predict_partial_charges_batch_mlx,
    predict_partial_charges_from_rdkit_mols_mlx,
    predict_partial_charges_from_smiles_mlx,
    predict_partial_charges_mlx,
)
from tests._mlx_cache import poison_buffer_cache

# 1 to 33 atoms and total charges -1, 0, +1, so slot widths, padded tails
# and charge targets all differ.
SMILES = [
    "[Na+]",
    "CCO",
    "C[NH3+]",
    "CC(=O)[O-]",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
]
PAD_TO = 96
U = 2.0**-24  # float32 unit roundoff
# ~8 float32 ulps at unit charge, 2x the worst width-induced drift measured.
ATOL = 1e-6


def _embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    return mol


@pytest.fixture(scope="module")
def molecules():
    mols = [_embed(smi, 7 + i) for i, smi in enumerate(SMILES)]
    return dict(
        mols=mols,
        atoms=[[a.GetAtomicNum() for a in m.GetAtoms()] for m in mols],
        xyz=[m.GetConformer().GetPositions().astype(np.float32) for m in mols],
        bonds=[bond_matrix_from_rdkit_mol(m) for m in mols],
        charge=[float(Chem.GetFormalCharge(m)) for m in mols],
    )


@pytest.fixture(scope="module", params=["direct", "qeq"])
def model(request):
    mx.random.seed(0)
    model = GeometricChargePredictor(ChargeModelConfig(readout=request.param))
    mx.eval(model.parameters())
    return model


def _single(model, m):
    return [
        predict_partial_charges_mlx(model, a, x, b, total_charge=q)
        for a, x, b, q in zip(m["atoms"], m["xyz"], m["bonds"], m["charge"])
    ]


def _batch(m, order=None, pad_to=None):
    order = range(len(m["atoms"])) if order is None else order
    atoms, xyz, bonds, charge = (
        [m[key][i] for i in order] for key in ("atoms", "xyz", "bonds", "charge")
    )
    return charge_model_batch(atoms, xyz, bonds, total_charges=charge, pad_to=pad_to)


def _raw(model, batch):
    out = model(
        batch.atomic_numbers, batch.coords, batch.bond_matrix, batch.mask, batch.total_charge
    )
    mx.eval(out)
    return np.array(out)


@pytest.mark.parametrize("pad_to", [None, PAD_TO], ids=["tight", "pad96"])
def test_batch_slot_matches_single_molecule(model, molecules, pad_to):
    single = _single(model, molecules)
    poison_buffer_cache()
    batched = predict_partial_charges_batch_mlx(model, _batch(molecules, pad_to=pad_to))

    assert len(batched) == len(single)
    for smi, b, s in zip(SMILES, batched, single):
        assert b.shape == s.shape, smi
        np.testing.assert_allclose(b, s, rtol=0, atol=ATOL, err_msg=smi)


def test_reordering_the_batch_permutes_the_result_exactly(model, molecules):
    order = [4, 0, 5, 2, 1, 3]
    ref = predict_partial_charges_batch_mlx(model, _batch(molecules, pad_to=PAD_TO))
    poison_buffer_cache()
    perm = predict_partial_charges_batch_mlx(model, _batch(molecules, order, pad_to=PAD_TO))
    for slot, i in enumerate(order):
        np.testing.assert_array_equal(perm[slot], ref[i], err_msg=SMILES[i])


def test_padded_atoms_are_zero_and_total_charge_is_kept(model, molecules):
    poison_buffer_cache()
    q = _raw(model, _batch(molecules, pad_to=PAD_TO)).astype(np.float64)
    assert np.isfinite(q).all()

    for i, (smi, atoms, target) in enumerate(zip(SMILES, molecules["atoms"], molecules["charge"])):
        n = len(atoms)
        np.testing.assert_array_equal(q[i, n:], 0.0, err_msg=smi)
        # Float32 rounding of an n-term sum and the per-atom shift: ~n ulps
        # of the summed magnitudes.
        tol = 2 * n * U * (np.abs(q[i, :n]).sum() + abs(target))
        assert abs(q[i, :n].sum() - target) <= tol, (smi, q[i, :n].sum(), target, tol)


def test_padded_slots_are_ignored_whatever_they_hold(model, molecules):
    """Masked entries must not leak in, even if they hold real-looking atoms."""
    clean = _batch(molecules, pad_to=PAD_TO)
    real = np.array(clean.mask) > 0
    pair = real[:, :, None] & real[:, None, :]
    rng = np.random.default_rng(0)

    z = np.array(clean.atomic_numbers)
    xyz = np.array(clean.coords)
    bonds = np.array(clean.bond_matrix)
    z[~real] = rng.integers(1, 10, (~real).sum())
    xyz[~real] = rng.normal(0.0, 1.5, ((~real).sum(), 3))
    bonds[~pair] = rng.integers(1, 5, (~pair).sum())
    junk = dataclasses.replace(
        clean, atomic_numbers=mx.array(z), coords=mx.array(xyz), bond_matrix=mx.array(bonds)
    )

    np.testing.assert_array_equal(_raw(model, junk), _raw(model, clean))


def test_rdkit_and_smiles_entry_points_match_single_molecule(model, molecules):
    single = _single(model, molecules)
    poison_buffer_cache()
    from_mols = predict_partial_charges_from_rdkit_mols_mlx(
        model, molecules["mols"], pad_to=PAD_TO
    )
    for smi, b, s in zip(SMILES, from_mols, single):
        np.testing.assert_allclose(b, s, rtol=0, atol=ATOL, err_msg=smi)

    # The SMILES path seeds molecule i's conformer with random_seed + i, so a
    # one-molecule call with that seed must see the same geometry.
    seed = 100
    poison_buffer_cache()
    batched = predict_partial_charges_from_smiles_mlx(model, SMILES, random_seed=seed)
    for i, smi in enumerate(SMILES):
        alone = predict_partial_charges_from_smiles_mlx(model, [smi], random_seed=seed + i)[0]
        np.testing.assert_allclose(batched[i], alone, rtol=0, atol=ATOL, err_msg=smi)
