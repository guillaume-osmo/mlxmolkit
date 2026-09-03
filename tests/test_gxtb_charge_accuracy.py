"""Lock in the g-xTB atomic-charge accuracy against the real binary.

The reference charges in ``tests/data/gxtb_oracle_charges.json`` come from
``xtb --gxtb --acc 0.2`` (xtb 6.7.1, commit 26dd68d — the g-xTB author's own
build), run on ``union`` because no g-xTB binary exists for macOS-arm64 here.
Geometries are the deterministic RDKit ones the oracle saw, carried in the same
file, so this runs with no binary and no network.

Full-set numbers behind the thresholds (20 molecules / 516 atoms):

    MFX off                   0.04617 e   0.37 s/mol
    MFX on (today's default)  0.02494 e   0.37 s/mol  <- wins on 20/20 molecules
    MFX + AES                 0.02062 e   1.61 s/mol  <- best MAE, 4.3x the cost
    MFX + AES + ACP           0.02175 e   1.59 s/mol  <- best RMSE and worst-case
    MFX + two-body 3rd order  0.02283 e   0.37 s/mol
    MFX + anisotropic H0      0.02972 e   1.09 s/mol  <- worse than plain MFX
    MFX, carbon patch off     0.06408 e
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
    """Defaults must stay at least as good as the measured MFX-on result."""
    assert _mae() < 0.032


def test_mulliken_fock_exchange_is_what_earns_that():
    """Turning MFX off must visibly regress — it is the largest charge term."""
    with_mfx = _mae()
    without = _mae(use_mfx_exchange=False)
    assert without > with_mfx
    assert without - with_mfx > 0.010


def test_acp_hamiltonian_does_not_help_on_top_of_mfx_alone():
    """Guards a tempting-but-wrong change.

    On water the ACP Hamiltonian repairs the oxygen s-shell population almost
    exactly (1.6916 -> 1.8202 against the binary's 1.8050), which makes it look
    like the missing piece. Without AES it is a net loss across real molecules:
    it halves the oxygen residual but degrades H and C by more.

    With AES on it becomes a mean-vs-tail trade instead, which is why this test
    is scoped to "on top of MFX alone" — see
    test_aes_is_the_best_accuracy_lever_and_costs_time.
    """
    assert _mae(use_acp_hamiltonian=True) > _mae()


def test_aes_is_the_best_accuracy_lever_on_top_of_mfx():
    """AES must stay enabled-able and must stay a win.

    It could not run at all until the one-centre exchange table shipped
    upstream (#80): mlxmolkit.xtb.gxtb_aes used to load that untracked file at
    module scope, which killed use_aes, use_aniso_h0 and use_twobody_third_order
    outright. This pins the gain so a regression in the table or its loader
    surfaces as a number, not a silent skip.
    """
    assert _mae(use_aes=True) < _mae() - 0.002


def test_anisotropic_h0_is_not_worth_enabling():
    """Measured negative: worse than plain MFX, and 3x the cost."""
    assert _mae(use_aniso_h0=True) > _mae()


def test_carbon_plevel_patch_is_not_redundant_with_mfx():
    """The oracle-fitted carbon-p patch was suspected of standing in for MFX.

    It is not: removing it with MFX on costs ~2.6x, so both are load-bearing.
    """
    assert _mae(use_carbon_plevel_shift=False) > _mae() + 0.02


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_charges_are_neutral_and_finite(case):
    res = gxtb_energy(case["atoms"], np.asarray(case["coords_ang"]), use_d4srev=False)
    q = np.asarray(res["atom_charges"], dtype=float)
    assert np.isfinite(q).all()
    assert abs(q.sum()) < 1e-6
