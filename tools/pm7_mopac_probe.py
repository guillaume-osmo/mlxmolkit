"""Development oracle: current MOPAC executable, frozen Cartesian geometry."""
from pathlib import Path
import os
import re
import shutil
import subprocess

MOPAC = os.environ.get('MOPAC_BIN') or shutil.which('mopac')
if not MOPAC:
    path = Path.home() / 'miniconda3/bin/mopac'
    MOPAC = str(path) if path.exists() else None


def reference(atoms, coords, directory, method='PM7'):
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    p = Path(directory) / 'reference.mop'
    p.write_text(f'{method} 1SCF SCFCRT=1.D-9 DISP PRT DEBUG\nport gate\n\n' + ''.join(
        f'{table.GetElementSymbol(int(z))} {x:.12f} 0 {y:.12f} 0 {w:.12f} 0\n'
        for z, (x, y, w) in zip(atoms, coords)))
    run = subprocess.run([MOPAC, str(p)], cwd=directory, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    out = p.with_suffix('.out').read_text()
    assert 'UNRECOGNIZED KEY' not in out and 'METHOD NOT SUPPORTED' not in out, out
    assert re.search(rf'{method}\s+-.*Hamiltonian', out, re.I), out
    def value(pattern):
        match = re.search(pattern + r'\s*([-+\d.]+)', out)
        assert match, out
        return float(match[1])
    qblock = re.search(r'NET ATOMIC CHARGES.*?\n(.*?)\n\s*DIPOLE', out, re.S)
    assert qblock, out
    q = [float(row.split()[2]) for row in qblock[1].splitlines()
         if len(row.split()) >= 4 and row.split()[0].isdigit()]
    assert len(q) == len(atoms)
    terms = {}
    for label, pattern in [('dispersion', r'DISPERSION ENERGY'),
                           ('hydrogen_bond', r'H-BOND ENERGY')]:
        match = re.search(pattern + r'\s*=?\s*([-+\d.]+)', out)
        if match:
            terms[label] = float(match[1])
    return dict(heat=value(r'FINAL HEAT OF FORMATION ='), charges=q,
                terms=terms, output=out)
