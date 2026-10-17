"""CHEESE: a molecule's scores and embedding must not depend on its batch.

``cheese_batch`` pads every molecule to the widest one (or ``pad_to``) with
Z=0 atoms at the origin, and every overlap sum multiplies by the atom mask.
Each parity test scores molecules inside a padded multi-molecule batch, with
MLX's buffer cache full of NaN, and compares every entry with the same
molecule(s) scored in one-molecule, unpadded batches.

Default padding cannot reveal a missing mask: Z=0 maps to radius 0 (clamped
to a 1e-4 A needle Gaussian) and padded charges are 0, so with the mask set
to all ones the scores come out bitwise identical. The masks are pinned
separately by filling the padded slots with real-looking atoms (Z 1-9,
random coordinates, charges and bonds): the output must not change by a bit.

Tolerances, all from float32 rounding:

* Gaussian shape overlaps sum positive terms, so the order of summation
  moves them by a few ulps (measured <= 2.1e-7 relative): ``rtol=1e-6``.
* ESP-Sim overlaps sum signed ``q_a q_b K_ab`` terms that cancel: here
  ``sum |terms|`` is 200-3000 for overlaps as small as 0.003, and changing
  the padded width from 33 to 300 atoms moves the Carbo index by up to 2e-5.
  That is pure summation order (the single-molecule values carry errors of
  the same size against a float64 reference), so the bound is the
  first-order rounding error of the score given those ``sum |terms|``:
  ``2u * kappa`` (measured <= 0.32u * kappa). A leaked padded atom or stale
  buffer would move scores by O(0.1) or give NaN.

Widths up to 64 atoms happen to be bitwise; 300 is not, and is tested.

Also pinned: ``CheeseGraphTransformer.encode_batch`` on ``[A]`` vs inside a
padded ``[A, B, C]`` batch (random weights, fixed seed), where padding moves
the unit-norm embeddings by a few ulps (measured <= 8.2e-8): ``atol=1e-6``.
"""
from __future__ import annotations

import dataclasses
import functools

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mlxmolkit.charge_model import bond_matrix_from_rdkit_mol
from mlxmolkit.cheese import (
    cheese_batch,
    cheese_similarity_pairs_mlx,
    electrostatic_overlap_matrix_mlx,
    electrostatic_similarity_matrix_mlx,
    shape_carbo_matrix_mlx,
    shape_tanimoto_matrix_mlx,
)
from mlxmolkit.cheese_embedding import (
    CheeseEmbeddingConfig,
    CheeseGraphTransformer,
    cheese_embedding_batch,
)
from tests._mlx_cache import poison_buffer_cache

# 9, 24 and 33 atoms.
SMILES = ["CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
WIDE = 300
U = 2.0**-24  # float32 unit roundoff
SHAPE_RTOL = 1e-6
EMBED_ATOL = 1e-6

SHAPE_SCORES = {"tanimoto": shape_tanimoto_matrix_mlx, "carbo": shape_carbo_matrix_mlx}


def _embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    AllChem.ComputeGasteigerCharges(mol)
    return mol


@pytest.fixture(scope="module")
def molecules():
    mols = [_embed(smi, 7 + i) for i, smi in enumerate(SMILES)]
    return dict(
        z=[[a.GetAtomicNum() for a in m.GetAtoms()] for m in mols],
        xyz=[m.GetConformer().GetPositions().astype(np.float32) for m in mols],
        q=[
            np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()], np.float32)
            for m in mols
        ],
        bonds=[bond_matrix_from_rdkit_mol(m) for m in mols],
    )


def _batch(m, order, pad_to=None):
    z, xyz, q = ([m[key][i] for i in order] for key in ("z", "xyz", "q"))
    return cheese_batch(z, xyz, q, pad_to=pad_to)


def _pairwise_alone(m, score):
    """``score`` of every (i, j), each from two one-molecule unpadded batches."""
    alone = [_batch(m, [i]) for i in range(len(m["z"]))]
    return np.array([[np.array(score(a, b))[0, 0] for b in alone] for a in alone])


def _esp_rounding_scale(m, metric):
    """First-order float32 error of each ESP score, in units of ``u``.

    The kernel ``K_ab`` is positive, so the overlap of ``|q|`` with ``|q|``
    is ``sum |q_a q_b K_ab|``, the magnitude each summed overlap is rounded
    against. Propagate those through the score.
    """
    cross = _pairwise_alone(m, electrostatic_overlap_matrix_mlx).astype(np.float64)
    mag = _pairwise_alone(
        m,
        lambda a, b: electrostatic_overlap_matrix_mlx(
            dataclasses.replace(a, charges=mx.abs(a.charges)),
            dataclasses.replace(b, charges=mx.abs(b.charges)),
        ),
    ).astype(np.float64)
    s, s_mag = np.diag(cross), np.diag(mag)
    a, b = s[:, None], s[None, :]
    a_mag, b_mag = s_mag[:, None], s_mag[None, :]
    if metric == "carbo":
        norm = np.sqrt(a * b)
        return mag / norm + np.abs(cross) / norm * 0.5 * (a_mag / a + b_mag / b)
    denom = a + b - cross
    return ((a + b) * mag + np.abs(cross) * (a_mag + b_mag)) / denom**2


@pytest.mark.parametrize("pad_to", [None, WIDE], ids=["tight", "wide"])
@pytest.mark.parametrize("metric", list(SHAPE_SCORES))
def test_shape_matrix_matches_one_molecule_batches(molecules, metric, pad_to):
    score = SHAPE_SCORES[metric]
    alone = _pairwise_alone(molecules, score)
    batch = _batch(molecules, [0, 1, 2], pad_to=pad_to)
    poison_buffer_cache()
    batched = np.array(score(batch, batch))
    np.testing.assert_allclose(batched, alone, rtol=SHAPE_RTOL, atol=0)


@pytest.mark.parametrize("pad_to", [None, WIDE], ids=["tight", "wide"])
@pytest.mark.parametrize("metric", ["carbo", "tanimoto"])
def test_electrostatic_matrix_matches_one_molecule_batches(molecules, metric, pad_to):
    score = functools.partial(electrostatic_similarity_matrix_mlx, metric=metric)
    alone = _pairwise_alone(molecules, score)
    tol = 2 * U * _esp_rounding_scale(molecules, metric)
    batch = _batch(molecules, [0, 1, 2], pad_to=pad_to)
    poison_buffer_cache()
    batched = np.array(score(batch, batch))
    assert np.isfinite(batched).all()
    err = np.abs(batched - alone)
    assert (err <= tol).all(), f"max err/tol {np.max(err / tol):.2f}\n{err}\n{tol}"


def test_paired_scores_match_one_pair_at_a_time(molecules):
    """Probe and reference padded to different widths, in different orders."""
    probe_order, ref_order = [0, 1, 2], [2, 0, 1]
    probe = _batch(molecules, probe_order)
    reference = _batch(molecules, ref_order, pad_to=WIDE)
    esp_tol = 2 * U * _esp_rounding_scale(molecules, "carbo")
    poison_buffer_cache()
    batched = cheese_similarity_pairs_mlx(probe, reference)

    for k, (i, j) in enumerate(zip(probe_order, ref_order)):
        alone = cheese_similarity_pairs_mlx(_batch(molecules, [i]), _batch(molecules, [j]))
        shape_b, shape_a = np.array(batched.shape)[k], np.array(alone.shape)[0]
        esp_b, esp_a = np.array(batched.electrostatic)[k], np.array(alone.electrostatic)[0]
        np.testing.assert_allclose(shape_b, shape_a, rtol=SHAPE_RTOL, atol=0)
        assert abs(esp_b - esp_a) <= esp_tol[i, j], (i, j, esp_b, esp_a)
        # combined = (shape + (esp + 1) / 2) / 2
        np.testing.assert_allclose(
            np.array(batched.combined)[k],
            np.array(alone.combined)[0],
            rtol=0,
            atol=SHAPE_RTOL * shape_a / 2 + esp_tol[i, j] / 4,
        )


@pytest.fixture(scope="module")
def embedder():
    mx.random.seed(0)
    model = CheeseGraphTransformer(CheeseEmbeddingConfig())
    mx.eval(model.parameters())
    return model


def _embedding_batch(m, order, pad_to=None):
    pick = {key: [m[key][i] for i in order] for key in ("z", "xyz", "bonds", "q")}
    return cheese_embedding_batch(pick["z"], pick["xyz"], pick["bonds"], pick["q"], pad_to=pad_to)


@pytest.mark.parametrize("pad_to", [None, WIDE], ids=["tight", "wide"])
def test_encode_batch_matches_one_molecule(molecules, embedder, pad_to):
    alone = [np.array(embedder.encode_batch(_embedding_batch(molecules, [i])))[0] for i in range(3)]
    poison_buffer_cache()
    pair = np.array(embedder.encode_batch(_embedding_batch(molecules, [0, 1], pad_to=pad_to)))
    poison_buffer_cache()
    triple = np.array(embedder.encode_batch(_embedding_batch(molecules, [2, 0, 1], pad_to=pad_to)))

    for i in range(2):
        np.testing.assert_allclose(pair[i], alone[i], rtol=0, atol=EMBED_ATOL)
    for slot, i in enumerate([2, 0, 1]):
        np.testing.assert_allclose(triple[slot], alone[i], rtol=0, atol=EMBED_ATOL)


def _junk_in_padding(batch, fields, seed=0):
    """Same batch with real-looking atoms written into every masked slot."""
    rng = np.random.default_rng(seed)
    real = np.array(batch.mask) > 0
    pad, pad_pairs = ~real, ~(real[:, :, None] & real[:, None, :])
    junk = {}
    for name in fields:
        a = np.array(getattr(batch, name))
        if name == "coords":
            a[pad] = rng.normal(0.0, 1.5, (pad.sum(), 3))
        elif name == "bond_matrix":
            a[pad_pairs] = rng.integers(1, 5, pad_pairs.sum())
        elif name == "atomic_numbers":
            a[pad] = rng.integers(1, 10, pad.sum())
        else:
            a[pad] = rng.normal(0.0, 0.5, pad.sum())
        junk[name] = mx.array(a)
    return dataclasses.replace(batch, **junk)


@pytest.mark.parametrize(
    "score",
    [
        shape_tanimoto_matrix_mlx,
        shape_carbo_matrix_mlx,
        functools.partial(electrostatic_similarity_matrix_mlx, metric="carbo"),
        functools.partial(electrostatic_similarity_matrix_mlx, metric="tanimoto"),
    ],
    ids=["shape_tanimoto", "shape_carbo", "esp_carbo", "esp_tanimoto"],
)
def test_similarity_ignores_whatever_padded_slots_hold(molecules, score):
    clean = _batch(molecules, [0, 1, 2], pad_to=48)
    junk = _junk_in_padding(clean, ["atomic_numbers", "coords", "charges"])
    np.testing.assert_array_equal(np.array(score(junk, junk)), np.array(score(clean, clean)))


def test_encode_batch_ignores_whatever_padded_slots_hold(molecules, embedder):
    clean = _embedding_batch(molecules, [0, 1, 2], pad_to=48)
    junk = _junk_in_padding(
        clean, ["atomic_numbers", "coords", "bond_matrix", "charges", "chiral_features"]
    )
    np.testing.assert_array_equal(
        np.array(embedder.encode_batch(junk)), np.array(embedder.encode_batch(clean))
    )
