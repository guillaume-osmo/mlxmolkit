"""All 40 PM7 elements against independently captured OpenMOPAC integrals."""
from pathlib import Path
import json
import os
import numpy as np
import pytest
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.scf import _pair_core_attraction
from mlxmolkit.nddo.batch import _two_centre_packed
from mlxmolkit.nddo.d_two_center import pair_cache
from mlxmolkit.nddo.overlap_d import overlap_d_molecular_frame
from mlxmolkit.nddo.overlap import overlap_molecular_frame
from mlxmolkit.nddo.pm7 import pair_repulsion

DATA = Path(os.environ.get('MLXMOLKIT_PM7_REFERENCE_DIR', Path(__file__).parent/'data'))
REFERENCE = np.load(DATA/'pm7_element_integrals.npz')
RECORDS = json.loads((DATA/'pm7_element_integrals.json').read_text())['records']


def test_pm7_coverage_and_atomic_reference():
    pm7, pm6 = get_params('PM7'), get_params('PM6')
    assert len(pm7) == 40 and set(pm7) == set(pm6)
    assert set(pm7) == set(REFERENCE['elements'])
    for i, z in enumerate(REFERENCE['elements']):
        p = pm7[int(z)]
        assert p.n_basis == REFERENCE['n_basis'][i]
        assert p.n_valence == REFERENCE['tore'][i]
        for name in ['gss','gsp','gpp','gp2','hsp','eisol']:
            assert getattr(p,name) == pytest.approx(REFERENCE[name][i], abs=1e-8)
    assert pm7[30].n_valence == pm7[48].n_valence == 2


@pytest.mark.parametrize('case', RECORDS, ids=lambda c:f"{c['a']}-{c['b']}")
def test_pm7_all_elements_mopac_integrals(case):
    params = get_params('PM7');a,b = params[case['a']],params[case['b']]
    ra,rb = np.zeros(3),np.array(case['displacement'])
    key = f'{a.Z}_{b.Z}'
    for cached in [False, True]:
        from contextlib import nullcontext
        with pair_cache([(a,b,ra,rb)]) if cached else nullcontext():
            w = _two_centre_packed(a,b,ra,rb)
            np.testing.assert_allclose(w, REFERENCE[key+'_w'], atol=2e-6, rtol=0)
            np.testing.assert_allclose(_two_centre_packed(b,a,rb,ra), w.T, atol=2e-6, rtol=0)
            for x,y,rx,ry,suffix in [(a,b,ra,rb,'a'),(b,a,rb,ra,'b')]:
                np.testing.assert_allclose(_pair_core_attraction(x,y,rx,ry),
                                           REFERENCE[key+'_core_'+suffix], atol=1e-5, rtol=0)
    overlap = (overlap_d_molecular_frame if a.has_d or b.has_d else overlap_molecular_frame)(a,b,ra,rb)
    np.testing.assert_allclose(overlap[:a.n_basis,:b.n_basis], REFERENCE[key+'_overlap'], atol=2e-6, rtol=0)
    assert pair_repulsion(a,b,ra,rb) == pytest.approx(float(REFERENCE[key+'_nuclear']), abs=1e-4)

from tests.test_mndo_port import hydride
from mlxmolkit.nddo.pm7_corrections import _COV_RADII
from mlxmolkit.nddo.scf import nddo_energy, nddo_energy_batch
from tools.pm7_mopac_probe import MOPAC, reference

def molecule(z):
 p=get_params('PM7')[z]
 if z==24:
  directions=np.vstack([np.eye(3),-np.eye(3)])
  return [24]+[9]*6,np.vstack([np.zeros(3),directions*1.65])
 if z==28:
  _,x=hydride(28,4,1.8)
  return [28]+[6]*4+[8]*4,np.vstack([x,x[1:]*(2.95/1.8)])
 if z==38:
  a,x=hydride(38,2,2.3)
  x[1:]+=np.random.default_rng(73).normal(scale=.07,size=x[1:].shape)
  return [38,9,9],x
 n={21:3,22:4,23:3,25:3,26:4,27:3,29:1}.get(z,min(p.n_valence,8-p.n_valence))
 a,x=hydride(z,n,_COV_RADII[z]+.31)
 if p.d_electrons:x[1:]+=np.random.default_rng(73).normal(scale=.07,size=x[1:].shape)
 return a,x


@pytest.mark.skipif(MOPAC is None, reason='MOPAC executable not installed')
@pytest.mark.parametrize('z', REFERENCE['elements'].tolist())
def test_pm7_expanded_molecular_oracle(z, tmp_path):
    atoms, coords = molecule(z)
    oracle = reference(atoms, coords, tmp_path)
    result = nddo_energy(atoms, coords, method='PM7', max_iter=500, conv_tol=1e-9)
    assert result['converged']
    assert result['heat_of_formation_kcal'] == pytest.approx(oracle['heat'], abs=.002)
    np.testing.assert_allclose(result['charges'], oracle['charges'], atol=3e-4, rtol=0)


@pytest.mark.parametrize('metal', [False, True])
def test_pm7_expanded_batch(metal, monkeypatch):
    if metal:
        monkeypatch.setenv('MLXMOLKIT_BATCH_ROTATION_METAL', '1')
        monkeypatch.setenv('MLXMOLKIT_SCF_FUSED_UPDATE', '1')
    molecules = [molecule(z) for z in REFERENCE['elements']]
    batch = nddo_energy_batch(molecules, method='PM7', use_metal=metal,
                             max_iter=500, conv_tol=1e-7)
    for (atoms, coords), result in zip(molecules, batch):
        scalar = nddo_energy(atoms, coords, method='PM7', max_iter=500, conv_tol=1e-9)
        assert result['converged'] and scalar['converged'], atoms
        assert result['heat_of_formation_kcal'] == pytest.approx(
            scalar['heat_of_formation_kcal'], abs=.02 if metal else 1e-5), atoms
        np.testing.assert_allclose(result['charges'], scalar['charges'], atol=3e-4, rtol=0)


@pytest.mark.parametrize('z', [13, 21, 26, 28, 33, 51])
def test_pm7_expanded_gradient(z):
    from mlxmolkit.nddo.anal_grad import analytical_gradient
    from mlxmolkit.nddo.constants import EV_TO_KCAL
    atoms, coords = molecule(z)
    result = nddo_energy(atoms, coords, method='PM7', max_iter=500, conv_tol=1e-9)
    _, actual = analytical_gradient(atoms, coords, method='PM7', scf_result=result)
    numeric = np.zeros_like(coords)
    for i in range(len(atoms)):
        for axis in range(3):
            xp, xm = coords.copy(), coords.copy()
            xp[i,axis] += 1e-4; xm[i,axis] -= 1e-4
            results = [nddo_energy(atoms, x, method='PM7', max_iter=500,
                                   conv_tol=1e-9, P_init=result['density']) for x in [xp,xm]]
            assert all(r['converged'] for r in results)
            numeric[i,axis] = (results[0]['heat_of_formation_kcal']-
                               results[1]['heat_of_formation_kcal'])/(2e-4*EV_TO_KCAL)
    np.testing.assert_allclose(actual, numeric, atol=5e-5, rtol=0)


@pytest.mark.parametrize('batch', [False, True])
def test_pm7_open_shell_remains_explicit(batch):
    with pytest.raises(ValueError, match='open.shell|odd|even|closed.shell'):
        if batch:
            nddo_energy_batch([([27], np.zeros((1,3)))], method='PM7')
        else:
            nddo_energy([27], np.zeros((1,3)), method='PM7')
