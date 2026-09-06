"""PM7 main-group model, ported from OpenMOPAC (Apache-2.0).

Source: 1d9d92b0283f197616f1e9e76d1ee09e2bc21e72, models/parameters_for_PM7_C.F90
and integrals/ccrep.F90. Copyright 2021 Virginia Polytechnic Institute and
State University. See data/LICENSE_OpenMOPAC.txt.
"""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .params import ElementParams, _compute_eisol, ANG_TO_BOHR
from .atomic_heats import atomic_heat

DATA = Path(__file__).resolve().parent / 'data'


class PM7Parameters(dict):
    def __missing__(self, z):
        raise ValueError(f'PM7 element Z={z} is not ported; supported elements are '
                         'H, C, N, O, F, Si, P, S, Cl, Br, I')


def load_params():
    params = PM7Parameters()
    with (DATA / 'parameters_PM7_MOPAC.csv').open() as handle:
        for row in csv.DictReader(handle):
            z = int(row.pop('Z'))
            symbol = row.pop('symbol')
            valence = int(row.pop('n_valence'))
            values = {k: float(v) for k, v in row.items()}
            gaussians = {f'gauss_{k}': [values.pop(f'{k}{i}') for i in range(1, 5)]
                         for k in ('K', 'L', 'M')}
            tails = tuple(values.pop('tail_' + k) for k in ('s', 'p', 'd'))
            has_d = values['zeta_d'] != 0
            p = ElementParams(Z=z, symbol=symbol, n_valence=valence,
                              n_basis=1 if z == 1 else 9 if has_d else 4,
                              eheat=atomic_heat(z), has_d=has_d, feather=True,
                              tail_exponents=tails, **gaussians, **values)
            p.eisol = _compute_eisol(p)
            params[z] = p
    return params


@lru_cache(maxsize=1)
def pair_parameters():
    table = {}
    with (DATA / 'PWCCT_PM7_MOPAC.csv').open() as handle:
        for row in csv.DictReader(handle):
            table[int(row['ZA']), int(row['ZB'])] = float(row['chi']), float(row['alpha'])
    return table


def pair_repulsion(p_a, p_b, coord_a, coord_b):
    """PM7 core repulsion, eV, including O-H, C-C and Si-O special terms."""
    r = float(np.linalg.norm(np.asarray(coord_a) - np.asarray(coord_b)))
    a, b = sorted((p_a.Z, p_b.Z))
    table = pair_parameters()
    chi, alpha = table.get((a, b), (0., 0.))
    if abs(chi) < 1e-5:
        # MOPAC fills missing heteronuclear pairs from homonuclear entries.
        chi = .5 * (table.get((a, a), (0., 0.))[0] + table.get((b, b), (0., 0.))[0])
        alpha = .5 * (table.get((a, a), (0., 0.))[1] + table.get((b, b), (0., 0.))[1])
    from .constants import HARTREE_TO_EV as ev
    rho = .5 * ev / p_a.gss + .5 * ev / p_b.gss
    enuc = p_a.n_valence * p_b.n_valence * ev / np.sqrt((r * ANG_TO_BOHR)**2 + rho**2)
    from .point_charge import factors
    keep, point = factors(r)
    enuc = enuc*keep + p_a.n_valence*p_b.n_valence*point
    if abs(chi) > 1e-5:
        if alpha < 1e-6:
            alpha = 1.2
        power = r*r if a == 1 and b in (6, 7, 8) else r + .0003*r**6
        scale = 1 + 2*chi*np.exp(-alpha*power)
        if (a, b) == (1, 8):
            scale -= -0.012037*np.exp(-0.701333*r*2)
        if (a, b) == (6, 6):
            scale += 8.947612*np.exp(-6.024265*r)
        if (a, b) == (8, 14):
            scale -= .0007*np.exp(-(r-2.9)**2)
    else:
        scale = 1 + 10*np.exp(-2.18*r)
    correction = 0.
    for p in (p_a, p_b):
        exponent = p.gauss_L[0]*(r-p.gauss_M[0])**2
        if exponent < 25:
            correction += p.gauss_K[0]*np.exp(-exponent)
    correction *= p_a.n_valence * p_b.n_valence / r
    unpolarizable = 1e-8*((a**(1/3)+b**(1/3))/r)**12
    return float(enuc*scale + correction + unpolarizable)
