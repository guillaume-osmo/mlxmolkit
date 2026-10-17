"""QMSession: the facade must be the low-level functions, not a fork of them.

docs/API_LOGIC.md advertised ``from mlxmolkit.qm import QMSession`` before the
module existed. These tests pin the facade to ``nddo_energy`` and
``pm6_esp_resp_charge_labels`` bit for bit, so it cannot drift from them.
"""
import numpy as np
import pytest

from mlxmolkit.esp_resp import pm6_esp_resp_charge_labels
from mlxmolkit.nddo import nddo_energy, pm6_d3h4_correction
from mlxmolkit.qm import QMSession

# Water dimer: the H4 hydrogen-bond and D3 terms are both non-zero.
DIMER_ATOMS = [8, 1, 1, 8, 1, 1]
DIMER_COORDS = np.array([
    [-1.551007, -0.114520, 0.000000],
    [-1.934259, 0.762503, 0.000000],
    [-0.599677, 0.040712, 0.000000],
    [1.350625, 0.111469, 0.000000],
    [1.680398, -0.373741, -0.758561],
    [1.680398, -0.373741, 0.758561],
])
# Acetate, so the charge models have a non-zero total to conserve.
ACETATE_ATOMS = [6, 6, 8, 8, 1, 1, 1]
ACETATE_COORDS = np.array([
    [0.000, 0.000, 0.000],
    [1.520, 0.000, 0.000],
    [2.130, 1.090, 0.000],
    [2.130, -1.090, 0.000],
    [-0.370, 1.030, 0.000],
    [-0.370, -0.510, 0.890],
    [-0.370, -0.510, -0.890],
])


def test_documented_example_runs():
    qm = QMSession(method="PM6_D3H4", backend="mlx", charge_model="resp")
    result = qm.energy(DIMER_ATOMS, DIMER_COORDS, molecular_charge=0)
    charges = qm.charges(DIMER_ATOMS, DIMER_COORDS, molecular_charge=0)
    assert result["converged"]
    assert charges.shape == (len(DIMER_ATOMS),)
    assert abs(charges.sum()) < 1e-6


def test_energy_is_nddo_energy():
    got = QMSession("PM6_D3H4").energy(DIMER_ATOMS, DIMER_COORDS)
    want = nddo_energy(DIMER_ATOMS, DIMER_COORDS, method="PM6_D3H4")
    assert got.keys() == want.keys()
    for key in ("energy_eV", "heat_of_formation_kcal", "n_iter", "converged"):
        assert got[key] == want[key], key
    np.testing.assert_array_equal(got["charges"], want["charges"])
    np.testing.assert_array_equal(got["density"], want["density"])


def test_d3h4_lands_in_the_heat_of_formation_only():
    """The documented convention: post-SCF terms are in HOF, not energy_eV."""
    pm6 = QMSession("PM6").energy(DIMER_ATOMS, DIMER_COORDS)
    d3h4 = QMSession("PM6_D3H4").energy(DIMER_ATOMS, DIMER_COORDS)
    correction = pm6_d3h4_correction(DIMER_ATOMS, DIMER_COORDS)
    assert correction["e_hb"] < 0.0 and correction["e_disp"] < 0.0
    assert d3h4["energy_eV"] == pm6["energy_eV"]
    np.testing.assert_allclose(
        d3h4["heat_of_formation_kcal"] - pm6["heat_of_formation_kcal"],
        correction["e_total"],
        atol=1e-9,
    )


@pytest.mark.parametrize("charge_model", ["mulliken", "esp", "resp"])
def test_charges_match_the_low_level_functions(charge_model):
    qm = QMSession("PM6", charge_model=charge_model)
    got = qm.charges(ACETATE_ATOMS, ACETATE_COORDS, molecular_charge=-1)
    if charge_model == "mulliken":
        want = nddo_energy(
            ACETATE_ATOMS, ACETATE_COORDS, method="PM6", molecular_charge=-1
        )["charges"]
    else:
        esp, resp = pm6_esp_resp_charge_labels(
            ACETATE_ATOMS, ACETATE_COORDS, total_charge=-1, method="PM6"
        )
        want = (esp if charge_model == "esp" else resp).charges
    np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(got.sum(), -1.0, atol=1e-6)


def test_method_spelling_is_normalised():
    assert QMSession("pm6-d3h4").method == "PM6_D3H4"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(method="NOT_A_METHOD"), "Unknown method"),
        (dict(backend="xtb"), "backend"),
        (dict(charge_model="am1bcc"), "charge_model"),
    ],
)
def test_bad_configuration_fails_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        QMSession(**kwargs)
