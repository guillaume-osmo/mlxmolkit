"""Independent executable gates for the MNDO port; PM5 must fail explicitly."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.scf import nddo_energy, nddo_energy_batch

MOPAC = os.environ.get('MOPAC_BIN') or shutil.which('mopac')
if not MOPAC:
    candidate = Path.home() / 'miniconda3/bin/mopac'
    MOPAC = str(candidate) if candidate.exists() else None

# Fixed Cartesian geometries, no embedding or force-field dependency.
def hydride(z, count, length):
    directions = np.array([[1., 1., 1.], [1., -1., -1.],
                           [-1., 1., -1.], [-1., -1., 1.]]) / np.sqrt(3)
    return [z] + [1] * count, np.vstack([np.zeros(3), directions[:count] * length])


CASES = [(1, 1, .74), (6, 4, 1.09), (7, 3, 1.01), (8, 2, .96),
         (9, 1, .92), (14, 4, 1.48), (15, 3, 1.42), (16, 2, 1.34),
         (17, 1, 1.27), (35, 1, 1.41), (53, 1, 1.61)]
SYMBOLS = {z: p.symbol for z, p in get_params('MNDO').items()}


def mopac_reference(atoms, coords, directory):
    job = directory / 'reference.mop'
    job.write_text('MNDO 1SCF SCFCRT=1.D-9\nMNDO port gate\n\n' + ''.join(
        f'{SYMBOLS[z]} {x:.12f} 0 {y:.12f} 0 {w:.12f} 0\n'
        for z, (x, y, w) in zip(atoms, coords)))
    process = subprocess.run([MOPAC, str(job)], cwd=directory,
                             capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr
    text = job.with_suffix('.out').read_text()
    # An executable exists but failed, or silently substituted a method:
    # that is a failed oracle, never a skipped test.
    assert 'METHOD NOT SUPPORTED' not in text
    assert re.search(r'MNDO\s+-.*Hamiltonian', text, re.I), text
    heat = re.search(r'FINAL HEAT OF FORMATION =\s*([-+\d.]+)\s*KCAL', text)
    assert heat, text
    charges = re.search(r'NET ATOMIC CHARGES.*?\n(.*?)\n\s*DIPOLE', text, re.S)
    assert charges, text
    q = [float(row.split()[2]) for row in charges[1].splitlines()
         if len(row.split()) >= 4 and row.split()[0].isdigit()]
    assert len(q) == len(atoms)
    return float(heat[1]), q


@pytest.mark.skipif(MOPAC is None, reason='MOPAC executable not installed')
@pytest.mark.parametrize('z,count,length', CASES)
def test_mndo_energy_and_charges_against_mopac(z, count, length, tmp_path):
    atoms, coords = hydride(z, count, length)
    reference_heat, reference_q = mopac_reference(atoms, coords, tmp_path)
    result = nddo_energy(atoms, coords, method='MNDO', max_iter=300, conv_tol=1e-9)
    assert result['converged']
    assert result['heat_of_formation_kcal'] == pytest.approx(reference_heat, abs=.05)
    np.testing.assert_allclose(result['charges'], reference_q, atol=5e-4, rtol=0)


@pytest.mark.parametrize('use_metal', [False, True])
def test_mndo_mixed_element_batch_matches_scalar(use_metal):
    molecules = [hydride(*case) for case in CASES]
    batch = nddo_energy_batch(molecules, method='MNDO', max_iter=300,
                              conv_tol=1e-7, use_metal=use_metal)
    for (atoms, coords), actual in zip(molecules, batch):
        expected = nddo_energy(atoms, coords, method='MNDO', max_iter=300, conv_tol=1e-8)
        assert actual['converged'] and expected['converged']
        np.testing.assert_allclose(actual['charges'], expected['charges'], atol=5e-4, rtol=0)
        assert actual['heat_of_formation_kcal'] == pytest.approx(
            expected['heat_of_formation_kcal'], abs=.01)


@pytest.mark.parametrize('method', ['MNDO', 'RM1', 'AM1', 'PM3', 'PM6'])
def test_symmetric_hydrogen_does_not_false_converge_under_singular_diis(method):
    atoms, coords = hydride(1, 1, .74)
    actual = nddo_energy_batch([(atoms, coords)], method=method,
                               use_metal=True, density_solver='eigh')[0]
    reference = nddo_energy(atoms, coords, method=method)
    assert actual['converged'] and reference['converged']
    np.testing.assert_allclose(actual['charges'], [0, 0], atol=1e-5)
    np.testing.assert_allclose(actual['density'], reference['density'], atol=1e-5)
    assert actual['heat_of_formation_kcal'] == pytest.approx(
        reference['heat_of_formation_kcal'], abs=.001)


@pytest.mark.parametrize('method', ['PM5', 'pm5'])
@pytest.mark.parametrize('entry', ['parameters', 'scalar', 'batch'])
def test_pm5_is_never_silently_substituted(method, entry):
    atoms, coords = hydride(8, 2, .96)
    with pytest.raises(ValueError, match='PM5.*not available'):
        if entry == 'parameters':
            get_params(method)
        elif entry == 'scalar':
            nddo_energy(atoms, coords, method=method)
        else:
            nddo_energy_batch([(atoms, coords)], method=method)
