"""tools.conformer_source must hand its attempt budget to RDKit, under its real name."""

from rdkit import Chem

from tools import conformer_source as cs


def test_explicit_attempt_budget_reaches_rdkit():
    params = cs.make_etkdg_params(seed=1, max_attempts=37)
    assert params.maxIterations == 37


def test_default_is_rdkit_default():
    # Historical behaviour: the old default of 1000 never reached RDKit, so every
    # existing cache was built with RDKit's own default (0 = 10 x atoms).
    params = cs.make_etkdg_params(seed=1)
    assert cs.DEFAULT_MAX_EMBED_ATTEMPTS is None
    assert params.maxIterations == 0


def test_small_budget_gives_up_on_an_impossible_isomer_without_a_conformer():
    # 7-azanorbornane with both bridgeheads specified in a configuration the cage
    # cannot adopt (an enumerated isomer of CHEMBL104700). RDKit must refuse it,
    # and a small budget must make it refuse quickly instead of after 10 x atoms
    # attempts (and then a random-coordinates fallback).
    mol = Chem.AddHs(Chem.MolFromSmiles("c1cncc(CN2[C@H]3CC[C@H]2CC3)c1"))
    ids = cs.embed_conformers(mol, n_conformers=1, min_conformers=1, seed=1, max_attempts=5,
                              random_coords_fallback=False)
    assert list(ids) == []
