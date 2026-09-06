"""Reproduce expanded PM7 molecular and CPU/Metal batch parity evidence."""
import json
import tempfile
import numpy as np
from tests.test_pm7_elements import molecule
from tools.pm7_mopac_probe import reference, MOPAC
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.scf import nddo_energy, nddo_energy_batch


def validate():
    zs = list(get_params('PM7'))
    molecules = [molecule(z) for z in zs]
    rows = []
    for z, (atoms, coords) in zip(zs, molecules):
        with tempfile.TemporaryDirectory() as directory:
            oracle = reference(atoms, coords, directory)
        native = nddo_energy(atoms, coords, method='PM7', max_iter=500, conv_tol=1e-9)
        dh = float(native['heat_of_formation_kcal']-oracle['heat'])
        dq = float(np.max(np.abs(native['charges']-oracle['charges'])))
        assert native['converged'] and abs(dh) < .002 and dq < 3e-4, z
        rows.append(dict(element=z, atoms=[int(a) for a in atoms], coords_ang=coords.tolist(),
                         oracle_heat_kcal=oracle['heat'], oracle_charges=oracle['charges'],
                         native_heat_kcal=float(native['heat_of_formation_kcal']),
                         native_charges=native['charges'].tolist(),
                         heat_error_kcal=dh, max_charge_error_e=dq))
    batch_max = {}
    for metal in [False, True]:
        results = nddo_energy_batch(molecules, method='PM7', use_metal=metal,
                                    max_iter=500, conv_tol=1e-7)
        key = 'metal_batch' if metal else 'cpu_batch'
        for row, result in zip(rows, results):
            dh = float(result['heat_of_formation_kcal']-row['oracle_heat_kcal'])
            dq = float(np.max(np.abs(result['charges']-row['oracle_charges'])))
            assert result['converged'] and abs(dh) < (.02 if metal else .002) and dq < 3e-4
            row[key] = dict(heat_error_kcal=dh, max_charge_error_e=dq)
        batch_max[key] = dict(max_heat_error_kcal=max(abs(r[key]['heat_error_kcal']) for r in rows),
                             max_charge_error_e=max(r[key]['max_charge_error_e'] for r in rows))
    return dict(oracle=f'{MOPAC}: PM7 CHARGE=0 1SCF SCFCRT=1.D-9 DISP PRT DEBUG',
                constants='current CODATA, no OLDFPC', count=len(rows),
                max_heat_error_kcal=max(abs(r['heat_error_kcal']) for r in rows),
                max_charge_error_e=max(r['max_charge_error_e'] for r in rows),
                batch_max=batch_max, molecules=rows)


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
