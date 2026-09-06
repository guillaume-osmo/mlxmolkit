"""Extract PM7 parameters from the pinned OpenMOPAC source.

Apache-2.0 source, copyright 2021 Virginia Polytechnic Institute and State
University. Source revision: 1d9d92b0283f197616f1e9e76d1ee09e2bc21e72.
Usage: python tools/import_pm7_params.py /path/to/openmopac
"""
import csv
from pathlib import Path
import re
import sys

# Valence populations from parameters_C.F90 (ios + iop + iod).
ELEMENTS = dict(zip(
    [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,37,38,48,49,50,51,52,53],
    zip('H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Rb Sr Cd In Sn Sb Te I'.split(),
        [1,1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7,8,9,10,11,2,3,4,5,6,7,1,2,2,3,4,5,6,7])))
FIELDS = dict(Uss='uss7', Upp='upp7', Udd='udd7', zeta_s='zs7',
              zeta_p='zp7', zeta_d='zd7', beta_s='betas7', beta_p='betap7',
              beta_d='betad7', gss='gss7', gsp='gsp7', gpp='gpp7',
              gp2='gp27', hsp='hsp7', alpha='alp7', F0SD='f0sd7',
              G2SD='g2sd7', rho_core='poc_7', tail_s='zsn7',
              tail_p='zpn7', tail_d='zdn7')


def extract(source):
    scalars = {(name.lower(), int(z)): float(value.lower().replace('d', 'e'))
               for name, z, value in re.findall(
                   r'data\s+(\w+)\(\s*(\d+)\s*\)\s*/\s*([-+\d.dDeE]+)\s*/', source, re.I)}
    gaussians = {(name.lower(), int(z), int(k)): float(v.lower().replace('d', 'e'))
                 for name, z, k, v in re.findall(
                     r'data\s+(gues7[123])\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*([-+\d.dDeE]+)\s*/', source, re.I)}
    rows = []
    for z, (symbol, valence) in ELEMENTS.items():
        assert ('uss7', z) in scalars
        row = dict(Z=z, symbol=symbol, n_valence=valence)
        row.update({field: scalars.get((name, z), 0.0) for field, name in FIELDS.items()})
        for k in range(1, 5):
            for field, name in [('K', 'gues71'), ('L', 'gues72'), ('M', 'gues73')]:
                row[f'{field}{k}'] = gaussians.get((name, z, k), 0.0)
        rows.append(row)
    pairs = {}
    for name, za, zb, value in re.findall(
            r'(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-+\d.dDeE]+)', source, re.I):
        a, b = sorted((int(za), int(zb)))
        if a in ELEMENTS and b in ELEMENTS:
            pairs.setdefault((a, b), {})[name.lower()] = float(value.lower().replace('d', 'e'))
    return rows, [dict(ZA=a, ZB=b, alpha=p['alpb'], chi=p['xfac'])
                  for (a, b), p in sorted(pairs.items())]


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'mlxmolkit/nddo/data'
    source = Path(sys.argv[1]) / 'src/models/parameters_for_PM7_C.F90'
    rows, pairs = extract(source.read_text())
    for name, values in [('parameters_PM7_MOPAC.csv', rows), ('PWCCT_PM7_MOPAC.csv', pairs)]:
        with (root / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
