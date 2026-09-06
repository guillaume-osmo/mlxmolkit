"""PM7 model, ported from OpenMOPAC (Apache-2.0).

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
        supported = ', '.join(p.symbol for p in self.values())
        raise ValueError(f'PM7 element Z={z} is not ported; supported elements are {supported}')


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
            from .params import _eisol_coefficients
            p.d_quantum_number = (3 if z <= 29 else 4 if z <= 47 else 5) if has_d else None
            p.d_electrons = _eisol_coefficients(z)[2]
            # calpar clamps diffuse p orbitals before any integral evaluation.
            if z != 1:
                p.zeta_p = max(.3, p.zeta_p)
            if p.d_electrons:
                _transition_parameters(p)
            if z != 1:
                p.hsp = max(1e-7, p.hsp)
            p.eisol = _compute_eisol(p) + _d_isolated_correction(p)
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
    rho = core_radius(p_a) + core_radius(p_b)
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
    # 0.3333 is the fitted model exponent, not a physical conversion constant.
    ax = r/(a**.3333+b**.3333)
    unpolarizable = min(1e-8/ax**12, 1e5) if ax < 3 else 0.
    return float(enuc*scale + correction + unpolarizable)


def core_radius(p):
    from .constants import HARTREE_TO_EV
    return p.rho_core if p.rho_core and p.rho_core > 1e-5 else .5*HARTREE_TO_EV/p.gss


def core_partner(p):
    """A nucleus uses po(9), while its electron monopole uses po(1)."""
    if not p.feather or not p.rho_core or p.rho_core <= 1e-5:
        return p
    from dataclasses import replace
    from .constants import HARTREE_TO_EV
    return replace(p, gss=.5*HARTREE_TO_EV/p.rho_core)


def _transition_parameters(p):
    """OpenMOPAC sp_two_electron: derive transition-metal sp integrals."""
    from .params import principal_qn
    from .w_integrals import slater_condon_parameter as sc
    n = principal_qn(p.Z)
    s, q, _ = p.tail_exponents
    def r(k,a,b,c,d):
        return sc(k,n,a,n,b,n,c,n,d)
    p.gss = r(0,s,s,s,s)
    p.gsp = r(0,s,s,q,q)
    p.hsp = r(1,s,q,s,q)/3
    f0, f2 = r(0,q,q,q,q), r(2,q,q,q,q)
    p.gpp, p.gp2 = f0+.16*f2, f0-.08*f2


def _d_isolated_correction(p):
    """OpenMOPAC inighd/eiscor, occupied d shells Sc through Cu."""
    if not p.d_electrons:
        return 0.
    from .params import principal_qn
    from .w_integrals import slater_condon_parameter as sc
    ns, nd = principal_qn(p.Z), p.d_quantum_number
    s, _, d = p.tail_exponents
    r016 = p.F0SD if p.F0SD > .001 else sc(0,ns,s,ns,s,nd,d,nd,d)
    r244 = p.G2SD if p.G2SD > .001 else sc(2,ns,s,nd,d,ns,s,nd,d)
    r066, r266, r466 = [sc(k,nd,d,nd,d,nd,d,nd,d) for k in (0,2,4)]
    i = p.Z-21
    return ([2,4,6,5,10,12,14,16,10][i]*r016
            + [0,1,3,10,10,15,21,28,45][i]*r066
            - [1,2,3,5,5,6,7,8,5][i]*r244/5
            - [0,8,15,35,35,35,43,50,70][i]*r266/49
            - [0,1,8,35,35,35,36,43,70][i]*r466/49)


def transition_core(block, p, partner):
    """PM7 rotatd shell-average Coulomb correction on a transition atom."""
    if p.feather and p.d_electrons:
        for shell in ([range(1,4), range(4,9)] if partner.n_basis > 1 else [range(4,9)]):
            ii = np.array(list(shell))
            block[ii,ii] += block[0,0]-np.mean(block[ii,ii])
    return block


def transition_packed(w, a, b):
    """PM7 rotatd corrections; packed orbital-pair storage in A,B order."""
    if not (a.feather and b.feather and (a.d_electrons or b.d_electrons)):
        return w
    da = np.arange(a.n_basis)*(np.arange(a.n_basis)+3)//2
    db = np.arange(b.n_basis)*(np.arange(b.n_basis)+3)//2
    def shift(ia,ib):
        ix = np.ix_(da[ia],db[ib])
        w[ix] += w[0,0]-w[ix].mean()
    if a.d_electrons:
        if b.n_basis > 1:
            shift(slice(4,9),slice(1,4))
        shift(slice(4,9),slice(0,1))
    if b.d_electrons:
        if a.d_electrons:
            shift(slice(4,9),slice(4,9))
        if a.n_basis > 1:
            shift(slice(1,4),slice(4,9))
        shift(slice(0,1),slice(4,9))
    return w
