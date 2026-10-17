"""Extract the supported sp MNDO parameters from an OpenMOPAC checkout.

Usage: python tools/import_mndo_params.py /path/to/mopac
The source is Apache-2.0, copyright 2021 Virginia Polytechnic Institute
and State University. Only explicitly supported elements are exported;
transition-metal MNDO extensions need separate integral validation.
"""
import argparse
import csv
from pathlib import Path
import re

ELEMENTS = {1: ('H', 1), 6: ('C', 4), 7: ('N', 5), 8: ('O', 6),
            9: ('F', 7), 14: ('Si', 4), 15: ('P', 5), 16: ('S', 6),
            17: ('Cl', 7), 35: ('Br', 7), 53: ('I', 7)}
FIELDS = dict(Uss='ussm', Upp='uppm', zeta_s='zsm', zeta_p='zpm',
              beta_s='betasm', beta_p='betapm', gss='gssm', gsp='gspm',
              gpp='gppm', gp2='gp2m', hsp='hspm', alpha='alpm')


def extract(source):
    values = {(name.lower(), int(z)): float(value.lower().replace('d', 'e'))
              for name, z, value in re.findall(
                  r'data\s+(\w+)\(\s*(\d+)\s*\)\s*/\s*([-+\d.dDeE]+)\s*/',
                  source, re.I)}
    rows = []
    for z, (symbol, valence) in ELEMENTS.items():
        if values.get(('zdm', z), 0) != 0:
            raise ValueError(f'{symbol} requires a d basis')
        row = dict(Z=z, symbol=symbol, n_valence=valence)
        for field, name in FIELDS.items():
            # Hydrogen has no p shell; every heavy-atom field is required.
            row[field] = (values.get((name, z), 0) if z == 1
                          else values[name, z])
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    source = args.checkout / 'src/models/parameters_for_mndo_C.F90'
    rows = extract(source.read_text())
    target = Path(__file__).resolve().parents[1] / 'mlxmolkit/nddo/data/parameters_MNDO_MOPAC.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
