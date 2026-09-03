"""Lock in the g-xTB atomic-charge accuracy against the real binary.

The reference charges in ``tests/data/gxtb_oracle_charges.json`` come from
``xtb --gxtb --acc 0.2`` (xtb 6.7.1, commit 26dd68d — the g-xTB author's own
build).
Geometries are the deterministic RDKit ones the oracle saw, carried in the same
file, so this runs with no binary and no network.

Measured over the 3 molecules / 50 atoms in the fixture, mean |dq| against
the binary, each term switched off from the default configuration:

    default                       5.6e-06 e
    without the first order       1.1e-01 e
    with the carbon-p patch       7.8e-02 e   <- now a LOSS, see its test
    without the ACP Hamiltonian   6.0e-02 e
    without Mulliken Fock exch.   4.8e-02 e
    without AES                   1.8e-02 e
    without anisotropic H0        2.0e-03 e
    without two-body third order  1.6e-03 e

Every term is load-bearing and every one improves the answer, which was not
true of the previous solver: there AES had to be switched on to help, and the
ACP Hamiltonian and the anisotropic H0 were measured losses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mlxmolkit.xtb.scf_gxtb import gxtb_energy

FIXTURE = Path(__file__).parent / "data" / "gxtb_oracle_charges.json"


def _cases():
    return json.loads(FIXTURE.read_text())["molecules"]


def _mae(**kwargs) -> float:
    diffs = []
    for case in _cases():
        res = gxtb_energy(
            case["atoms"], np.asarray(case["coords_ang"]), use_d4srev=False, **kwargs
        )
        assert res["converged"], f"{case['name']} did not converge for {kwargs}"
        q = np.asarray(res["atom_charges"], dtype=float)
        ref = np.asarray(case["oracle_charges"], dtype=float)
        assert q.shape == ref.shape
        diffs.append(q - ref)
    return float(np.abs(np.concatenate(diffs)).mean())


def test_fixture_is_self_consistent():
    for case in _cases():
        assert len(case["atoms"]) == len(case["coords_ang"]) == len(case["oracle_charges"])
        # A neutral molecule's Mulliken charges must sum to zero in the oracle too.
        assert abs(sum(case["oracle_charges"])) < 1e-5


def test_default_charge_mae_against_the_binary():
    """The default configuration reproduces the binary's charges.

    This is the headline number: the solver is not approximating the binary,
    it is reproducing it, and the residual is convergence noise rather than a
    model difference.
    """
    assert _mae() < 1.0e-04


def test_mulliken_fock_exchange_is_load_bearing():
    """Turning MFX off must visibly regress — it is a large charge term."""
    without = _mae(use_mfx_exchange=False)
    assert without > _mae()
    assert without > 0.010


def test_acp_hamiltonian_is_load_bearing():
    """⚠️ This assertion is INVERTED from what it used to be.

    Against the previous solver the ACP Hamiltonian was a measured net loss,
    and this file guarded against enabling it. With the terms it interacts
    with in place it is worth ~6e-02 e, the second largest of them.
    """
    assert _mae(use_acp_hamiltonian=False) > _mae() + 0.010


def test_aes_is_load_bearing():
    """AES must stay enabled-able and must stay a win.

    It could not run at all until the one-centre exchange table shipped
    upstream (#80): mlxmolkit.xtb.gxtb_aes used to load that untracked file at
    module scope, which killed use_aes, use_aniso_h0 and use_twobody_third_order
    outright. This pins the loss from removing it, so a regression in the table
    or its loader surfaces as a number rather than a silent skip.
    """
    assert _mae(use_aes=False) > _mae() + 0.005


def test_anisotropic_h0_is_load_bearing():
    """⚠️ Also inverted: this used to assert the term was not worth enabling.

    The gain is small — ~2e-03 e — but it is a gain, and it is three orders of
    magnitude above the default residual, so it is not noise.
    """
    assert _mae(use_aniso_h0=False) > _mae() + 5.0e-04


def test_carbon_plevel_patch_is_now_harmful():
    """⚠️ Inverted, and the most informative of the four.

    The carbon-p shift was fitted against the oracle to repair charges the
    model was getting wrong for a different reason. It was load-bearing then —
    removing it cost ~2.6x. Now that the terms it was standing in for are
    present, switching it back ON is the single largest regression available,
    which is what an empirical patch should look like once the physics it
    substituted for arrives.
    """
    assert _mae(use_carbon_plevel_shift=True) > _mae() + 0.020


def test_two_body_third_order_is_load_bearing():
    """The smallest of the terms, and the one whose flag was unusable.

    Passing ``use_twobody_third_order=False`` used to raise: the disabled
    branch returned a 3-tuple where the routine returns 2, so the term could
    not be switched off at all and no test could measure it.
    """
    assert _mae(use_twobody_third_order=False) > _mae() + 5.0e-04


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_charges_are_neutral_and_finite(case):
    res = gxtb_energy(case["atoms"], np.asarray(case["coords_ang"]), use_d4srev=False)
    q = np.asarray(res["atom_charges"], dtype=float)
    assert np.isfinite(q).all()
    assert abs(q.sum()) < 1e-6
