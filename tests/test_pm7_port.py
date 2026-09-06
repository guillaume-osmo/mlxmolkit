"""Native PM7 gates against current MOPAC, including complete forces and batches."""
import numpy as np
import pytest

from mlxmolkit.nddo.scf import nddo_energy, nddo_energy_batch
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.constants import EV_TO_KCAL
from mlxmolkit.nddo.pm7_corrections import corrections
from mlxmolkit.nddo.anal_grad import analytical_gradient
from tests.test_mndo_port import CASES, hydride
from tests.test_mopac_pm6_parity import geometry
from tools.pm7_mopac_probe import MOPAC, reference

ORGANICS = ['CO', 'CCO', 'CC(=O)N', 'CC(=O)O', 'C#C', 'c1ccccc1',
            'CCN', 'CS', 'CP', 'CCl', 'CBr', 'CI', 'C[Si](C)(C)O']


@pytest.mark.skipif(MOPAC is None, reason='MOPAC executable not installed')
@pytest.mark.parametrize('case', CASES)
def test_pm7_original_hydrides_against_mopac(case, tmp_path):
    a, x = hydride(*case)
    oracle = reference(a, x, tmp_path)
    result = nddo_energy(a, x, method='PM7', max_iter=300, conv_tol=1e-9)
    assert result['converged']
    assert result['heat_of_formation_kcal'] == pytest.approx(oracle['heat'], abs=.002)
    np.testing.assert_allclose(result['charges'], oracle['charges'], atol=3e-4, rtol=0)


@pytest.mark.skipif(MOPAC is None, reason='MOPAC executable not installed')
@pytest.mark.parametrize('smiles', ORGANICS)
def test_pm7_molecular_energy_and_charges(smiles, tmp_path):
    a, x = geometry(smiles)
    oracle = reference(a, x, tmp_path)
    result = nddo_energy(a, x, method='PM7', max_iter=300, conv_tol=1e-9)
    assert result['converged']
    assert result['heat_of_formation_kcal'] == pytest.approx(oracle['heat'], abs=.002)
    np.testing.assert_allclose(result['charges'], oracle['charges'], atol=3e-4, rtol=0)


@pytest.mark.skipif(MOPAC is None, reason='MOPAC executable not installed')
@pytest.mark.parametrize('smiles', ['O', 'N', 'C(=O)O', 'NC=O'])
@pytest.mark.parametrize('separation', [2.6, 2.8, 3.2, 4.0])
def test_pm7_correction_terms_against_mopac(smiles, separation, tmp_path):
    a, x = geometry(smiles)
    a, x = a+a, np.vstack([x, x+[separation, 0, 0]])
    oracle = reference(a, x, tmp_path)
    actual = corrections(a, x)
    for key in ['dispersion', 'hydrogen_bond']:
        assert key in oracle['terms']
        assert actual[key] == pytest.approx(oracle['terms'][key], abs=1e-5)


@pytest.mark.parametrize('metal', [False, True])
def test_pm7_mixed_batch(metal):
    mols = [hydride(*c) for c in CASES] + [geometry(s) for s in ['CP', 'CC(=O)N']]
    results = nddo_energy_batch(mols, method='PM7', use_metal=metal,
                                max_iter=300, conv_tol=1e-7)
    for (a, x), r in zip(mols, results):
        expected = nddo_energy(a, x, method='PM7', max_iter=300, conv_tol=1e-9)
        assert r['converged'] and expected['converged']
        assert r['heat_of_formation_kcal'] == pytest.approx(
            expected['heat_of_formation_kcal'], abs=.01 if metal else 1e-5)
        np.testing.assert_allclose(r['charges'], expected['charges'], atol=3e-4, rtol=0)
        assert r['energy_eV'] == pytest.approx(
            r['scf_energy_eV']+r['geometry_correction_eV'], abs=1e-10)


@pytest.mark.parametrize('smiles', ['CP', 'CC(=O)N', 'C#C', 'C[Si](C)(C)O', 'water_dimer'])
def test_pm7_complete_gradient(smiles):
    if smiles == 'water_dimer':
        a, x = geometry('O')
        a, x = a+a, np.vstack([x, x+[2.8, 0, 0]])
    else:
        a, x = geometry(smiles)
    result, grad = analytical_gradient(a, x, method='PM7')
    step = 1e-4
    numeric = np.zeros_like(x)
    for i in range(len(a)):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += step
            xm[i, k] -= step
            rp = nddo_energy(a, xp, method='PM7', max_iter=300, conv_tol=1e-9)
            rm = nddo_energy(a, xm, method='PM7', max_iter=300, conv_tol=1e-9)
            assert rp['converged'] and rm['converged']
            numeric[i, k] = (rp['heat_of_formation_kcal']-rm['heat_of_formation_kcal'])/(2*step*EV_TO_KCAL)
    assert np.sqrt(np.mean((grad-numeric)**2)) < 1e-5
    assert result['energy_eV'] == pytest.approx(
        result['electronic_eV']+result['nuclear_eV']+sum(corrections(a, x).values())/EV_TO_KCAL)


def test_pm7_point_charge_limit_and_method_isolation():
    from mlxmolkit.nddo.rotation import rotate_integrals_to_molecular_frame
    from mlxmolkit.nddo.point_charge import COULOMB_EV_ANG
    a, x = geometry('CP')
    before = nddo_energy(a, x, method='PM6')
    nddo_energy(a, x, method='PM7')
    after = nddo_energy(a, x, method='PM6')
    np.testing.assert_allclose(before['density'], after['density'], atol=1e-12)
    p = get_params('PM7')
    w, e1, e2 = rotate_integrals_to_molecular_frame(p[6], p[8], np.zeros(3), np.array([8., 0, 0]))
    point = COULOMB_EV_ANG/8
    np.testing.assert_allclose(w, point*np.einsum('ij,kl->ijkl', np.eye(4), np.eye(4)), atol=1e-12)
    np.testing.assert_allclose(e1, -6*point*np.eye(4), atol=1e-12)
    np.testing.assert_allclose(e2, -4*point*np.eye(4), atol=1e-12)
    # The rotation helpers retain their zero-distance sentinel contract.
    from mlxmolkit.nddo.rotation_batch import rotate_pairs
    zero = rotate_pairs([(p[6], p[8])], [(np.zeros(3), np.zeros(3))])
    np.testing.assert_array_equal(zero, 0.)


def test_pm7_rotation_and_permutation():
    a, x = geometry('CP')
    base = nddo_energy(a, x, method='PM7', conv_tol=1e-9)
    q, _ = np.linalg.qr(np.random.default_rng(13).normal(size=(3, 3)))
    order = np.random.default_rng(14).permutation(len(a))
    changed = nddo_energy([a[i] for i in order], (x@q+[2., -3., 1.])[order],
                          method='PM7', conv_tol=1e-9)
    assert changed['energy_eV'] == pytest.approx(base['energy_eV'], abs=1e-6)
    np.testing.assert_allclose(changed['charges'], base['charges'][order], atol=1e-5)


@pytest.mark.parametrize('entry', ['scalar', 'batch'])
def test_pm7_unsupported_elements_fail_explicitly(entry):
    with pytest.raises(ValueError, match='PM7 element Z=79 is not ported'):
        if entry == 'scalar':
            nddo_energy([79], np.zeros((1, 3)), method='PM7')
        else:
            nddo_energy_batch([([79], np.zeros((1, 3)))], method='PM7')


def test_pm7_case_normalization_and_fused_metal(monkeypatch):
    mols = [geometry(s) for s in ['CCO', 'CP']]
    monkeypatch.setenv('MLXMOLKIT_BATCH_ROTATION_METAL', '1')
    monkeypatch.setenv('MLXMOLKIT_SCF_FUSED_UPDATE', '1')
    batch = nddo_energy_batch(mols, method='pm7', use_metal=True, max_iter=300)
    for (a, x), r in zip(mols, batch):
        expected = nddo_energy(a, x, method='PM7', conv_tol=1e-9)
        lower = nddo_energy(a, x, method='pm7', conv_tol=1e-9)
        assert lower['energy_eV'] == pytest.approx(expected['energy_eV'], abs=1e-10)
        assert r['converged']
        assert r['heat_of_formation_kcal'] == pytest.approx(expected['heat_of_formation_kcal'], abs=.01)
