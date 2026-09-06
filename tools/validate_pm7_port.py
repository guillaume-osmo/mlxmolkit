"""Capture reproducible PM7 MOPAC parity evidence. Run with PYTHONPATH=."""
import json
import tempfile
from pathlib import Path
import numpy as np

from tools.pm7_mopac_probe import reference, MOPAC
from tests.test_mndo_port import CASES, hydride
from tests.test_mopac_pm6_parity import geometry
from tests.test_pm7_port import ORGANICS
from mlxmolkit.nddo.scf import nddo_energy


def validate():
    cases = [(f'hydride_Z{c[0]}', hydride(*c)) for c in CASES]
    cases += [(s, geometry(s)) for s in ORGANICS]
    records = []
    for name, (atoms, coords) in cases:
        with tempfile.TemporaryDirectory() as directory:
            oracle = reference(atoms, coords, directory)
        result = nddo_energy(atoms, coords, method='PM7', max_iter=300, conv_tol=1e-9)
        delta = result['heat_of_formation_kcal']-oracle['heat']
        dq = float(np.max(np.abs(result['charges']-oracle['charges'])))
        assert result['converged'] and abs(delta) < .002 and dq < 3e-4, name
        records.append(dict(name=name, atoms=atoms, coords_ang=coords.tolist(),
                            native_heat_kcal=result['heat_of_formation_kcal'],
                            oracle_heat_kcal=oracle['heat'], heat_error_kcal=delta,
                            native_charges=result['charges'].tolist(),
                            oracle_charges=oracle['charges'], max_charge_error_e=dq))
    return dict(oracle=f'{MOPAC}: PM7 1SCF SCFCRT=1.D-9 DISP PRT DEBUG',
                physical_constants='current OpenMOPAC CODATA, no OLDFPC',
                model_source='openmopac/mopac@1d9d92b0283f197616f1e9e76d1ee09e2bc21e72',
                count=len(records),
                max_heat_error_kcal=max(abs(r['heat_error_kcal']) for r in records),
                max_charge_error_e=max(r['max_charge_error_e'] for r in records),
                molecules=records)


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
