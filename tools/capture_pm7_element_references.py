"""Capture independent PM7 element/integral gates from gfortran libmopac.

Developer-only private symbols. The installed runtime never loads MOPAC.
Source: OpenMOPAC 1d9d92b0283f197616f1e9e76d1ee09e2bc21e72 (Apache-2.0).
Run from the repository: OMP_NUM_THREADS=1 python -m tools.capture_pm7_element_references
"""
import ctypes as C
import json
from pathlib import Path
import numpy as np
from tools.import_pm7_params import ELEMENTS
from tools.mopac_api import _lib, scf, _find_libmopac


def capture(destination, all_pairs=False):
    # Initializes all PM7 element tables with the binary's default CODATA constants.
    scf([8, 1, 1], [[0,0,0], [.96,0,0], [-.24,.93,0]], model='PM7')
    lib = _lib()
    ptr = lambda a: a.ctypes.data_as(C.c_void_p)
    values, records = {}, []
    elements = list(ELEMENTS)
    for name in ['gss', 'gsp', 'gpp', 'gp2', 'hsp', 'eisol', 'tore']:
        table = np.ctypeslib.as_array((C.c_double*107).in_dll(lib, '__parameters_c_MOD_'+name))
        values[name] = table[np.array(elements)-1].copy()
    nb = (C.c_int*107).in_dll(lib, '__parameters_c_MOD_natorb')
    values['elements'] = np.array(elements)
    values['n_basis'] = np.array([nb[z-1] for z in elements])
    for a in elements:
        for b in (elements if all_pairs else [1, 6, 16, 26, 53]):
            if all_pairs and b > a:
                continue
            na, nbasis = nb[a-1], nb[b-1]
            # Generic orientation probes every real p/d harmonic.
            ra, rb = np.zeros(3), np.array([1.73, .41, -.69])
            za, zb = C.c_int(a), C.c_int(b)
            count, en = C.c_int(), C.c_double()
            w = np.zeros(2025)
            lib.rotatd_(C.byref(za), C.byref(zb), ptr(ra), ptr(rb), ptr(w), C.byref(count), C.byref(en))
            size = na*(na+1)//2 * (nbasis*(nbasis+1)//2)
            assert count.value == size
            key = f'{a}_{b}'
            values[key+'_w'] = w[:size].reshape(na*(na+1)//2, nbasis*(nbasis+1)//2).copy()
            h = np.zeros(171)
            indices = [C.c_int(i) for i in (1, na, na+1, na+nbasis)]
            lib.elenuc_(*(C.byref(i) for i in indices), ptr(h))
            n = na+nbasis
            ii,jj = np.tril_indices(n)
            full = np.zeros((n,n)); full[ii,jj] = h[:len(ii)]; full[jj,ii] = h[:len(ii)]
            values[key+'_core_a'], values[key+'_core_b'] = full[:na,:na], full[na:,na:]
            overlap = np.zeros((9,9), order='F')
            lib.diat_(C.byref(za), C.byref(zb), ptr(rb), ptr(overlap))
            values[key+'_overlap'] = overlap[:na,:nbasis].copy()
            values[key+'_nuclear'] = en.value
            records.append(dict(a=a, b=b, displacement=rb.tolist()))
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination/'pm7_element_integrals.npz', **values)
    (destination/'pm7_element_integrals.json').write_text(json.dumps(dict(
        library=_find_libmopac(), source_revision='1d9d92b0283f197616f1e9e76d1ee09e2bc21e72',
        constants='default CODATA', records=records), indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-pairs', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'tests/data')
    args = parser.parse_args()
    capture(args.output, all_pairs=args.all_pairs)
