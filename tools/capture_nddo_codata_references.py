"""Read reference energies/overlaps from a gfortran-built libmopac.

Developer-only diagnostic: private Fortran symbols are never used by the
runtime or tests. Run with PYTHONPATH=. and MOPAC_LIB as needed. Output JSON
is reproducible evidence for the CODATA migration, not a native self-oracle.
"""
import ctypes as C
import json
import numpy as np
from tools.mopac_api import _lib, scf, _find_libmopac
from tests.test_nddo_scf import MOLS


def capture():
    lib = _lib()
    results = {}
    for method in ['RM1', 'AM1', 'PM6']:
        results[method] = {}
        for name, (atoms, coords) in MOLS.items():
            scf(atoms, coords, model=method)
            electronic = C.c_double.in_dll(lib, '__molkst_c_MOD_elect').value
            nuclear = C.c_double.in_dll(lib, '__molkst_c_MOD_enuclr').value
            results[method][name] = dict(E_elec=electronic, E_nuc=nuclear,
                                         E_tot=electronic+nuclear)
    overlaps = []
    # All relevant PM6 elements are initialized by MOPAC even on water.
    for a,b,r in [(1,1,.74),(6,1,1.09),(6,6,1.54),(16,1,1.336),(16,6,1.81),(16,16,2.05)]:
        z1,z2=C.c_int(a),C.c_int(b)
        vector=np.array([r,0.,0.]);out=np.zeros((9,9),order='F')
        lib.diat_(C.byref(z1),C.byref(z2),vector.ctypes.data_as(C.c_void_p),
                  out.ctypes.data_as(C.c_void_p))
        overlaps.append(dict(a=a,b=b,r=r,S00=float(out[0,0])))
    return dict(library=_find_libmopac(), convention='default CODATA',
                results=results, overlaps=overlaps)


if __name__ == '__main__':
    print(json.dumps(capture(), indent=2))
