"""Regression guards for g-xTB terms that were silently unrunnable.

Three g-xTB contributions — ``use_aes``, ``use_aniso_h0`` and
``use_twobody_third_order`` — could not execute at all, on any clean checkout.
Not because of their own physics: :mod:`mlxmolkit.xtb.gxtb_aes` loaded
``data/gxtb_onecxints_extracted.npz`` at *module scope*, that file is an
untracked extraction artifact with no regeneration script, and so importing the
module raised ``FileNotFoundError`` before any of those terms ran. None of them
actually reads the table. The suite stayed green because the one test that
touches it skips on the same missing file.

The load is lazy now, so these tests pin that the terms execute.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.xtb import gxtb_aes
from mlxmolkit.xtb.scf_gxtb import gxtb_energy

# Water at the geometry used by the oracle harness.
ATOMS = [8, 1, 1]
COORDS = np.array(
    [
        [0.0, 0.0, 0.117790],
        [0.0, 0.755453, -0.471160],
        [0.0, -0.755453, -0.471160],
    ]
)


def _q(**kwargs) -> np.ndarray:
    res = gxtb_energy(ATOMS, COORDS, use_d4srev=False, **kwargs)
    assert res["converged"], f"SCF did not converge for {kwargs}"
    return np.asarray(res["atom_charges"], dtype=float)


def test_gxtb_aes_imports_without_the_untracked_table():
    """The module must import whether or not the extraction artifact is present."""
    assert gxtb_aes is not None
    assert hasattr(gxtb_aes, "_onecx_tables")


@pytest.mark.skipif(
    gxtb_aes._os.path.exists(gxtb_aes._ONECX_PATH),
    reason="the onecxints table is present here, so the missing-file path cannot be exercised",
)
def test_missing_onecx_table_explains_itself():
    with pytest.raises(FileNotFoundError) as excinfo:
        gxtb_aes._onecx_tables()
    msg = str(excinfo.value)
    assert "gxtb_onecxints_extracted.npz" in msg
    # It must say what breaks and how to get the table back, not just print a path.
    assert "use_aes" in msg
    assert "libxtb" in msg


@pytest.mark.parametrize(
    "flag",
    ["use_aes", "use_aniso_h0", "use_twobody_third_order"],
)
def test_previously_dead_term_runs(flag):
    """Each term must reach a converged SCF rather than dying on the import."""
    q = _q(**{flag: True})
    assert q.shape == (3,)
    assert np.isfinite(q).all()
    assert abs(q.sum()) < 1e-6, "a neutral molecule's charges must still sum to zero"


@pytest.mark.parametrize("flag", ["use_aes", "use_twobody_third_order"])
def test_previously_dead_term_actually_changes_the_density(flag):
    """A term that runs but moves nothing would be dead in a subtler way.

    ⚠️ Switched around: these terms are ON in the default configuration now,
    so enabling them is a no-op and the question is whether DISABLING them
    moves anything. ``use_aniso_h0`` stays excluded: on water it shifts q(O)
    by only ~0.002 e, which is real but too small to assert as a guard.
    """
    base = _q()
    without_term = _q(**{flag: False})
    assert np.abs(without_term - base).max() > 1e-3
