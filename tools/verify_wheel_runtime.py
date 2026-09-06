"""Check an unpacked wheel in an isolated interpreter, outside the checkout.

Usage: python tools/verify_wheel_runtime.py /path/to/mlxmolkit.whl
Uses installed third-party dependencies but imports mlxmolkit from the wheel.
The frozen binary-charge fixture is input data, not an import from the repo.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


CHECK = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
import mlxmolkit
assert Path(mlxmolkit.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
from mlxmolkit.xtb.gxtb_scf import _install_onecxints
from mlxmolkit.xtb.gxtb_aes import _onecx_tables
from mlxmolkit.xtb.scf_gxtb import gxtb_energy
assert _install_onecxints()
assert _onecx_tables()[0].shape == (103, 10)
cases = json.loads(Path(sys.argv[2]).read_text())['molecules']
errors = []
for case in cases:
    result = gxtb_energy(case['atoms'], np.array(case['coords_ang']), use_d4srev=False)
    assert result['converged'], case['name']
    q = np.asarray(result['atom_charges'])
    errors.extend(np.abs(q - case['oracle_charges']).tolist())
mae = float(np.mean(errors))
assert mae < 1e-4, mae
# Exercise the D4Srev package data as well, independently of the charge gate.
case = cases[0]
assert gxtb_energy(case['atoms'], np.array(case['coords_ang']), use_d4srev=True)['converged']
from mlxmolkit.nddo import nddo_energy
result = nddo_energy([1, 1], np.array([[0., 0., 0.], [0., 0., .74]]), method='MNDO')
assert result['converged']
np.testing.assert_allclose(result['charges'], [0, 0], atol=1e-6)
print(json.dumps(dict(gxtb_molecules=len(cases), gxtb_atoms=len(errors),
                     gxtb_charge_mae_e=mae, gxtb_charge_max_error_e=max(errors),
                     onecentre_table=True, d4srev=True, mndo=True), indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / 'tests/data/gxtb_oracle_charges.json'
    with tempfile.TemporaryDirectory(prefix='mlxmolkit-wheel-') as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(directory)
        subprocess.run([sys.executable, '-I', '-c', CHECK, directory, str(fixture)],
                       cwd=directory, check=True)


if __name__ == '__main__':
    main()
