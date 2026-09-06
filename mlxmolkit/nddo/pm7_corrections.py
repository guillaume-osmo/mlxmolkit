"""Molecular PM7 corrections from OpenMOPAC, Apache-2.0.

Sources: corrections/Hydrogen_bond_corrections.F90 (setup_DH_Plus),
H_bond_correction_EH_plus.F90, H_bond_correction_bits.F90,
set_up_dentate.F90, geometry/dihed.F90. Copyright 2021 Virginia
Polytechnic Institute and State University. Nonperiodic scope.
"""
import math
import numpy as np
from scipy.special import expit
from .constants import BOHR_TO_ANG, HARTREE_TO_EV, EV_TO_KCAL

# OpenMOPAC covrad and atom_radius_covalent, Z=1..54.
_HB_RADII = dict(enumerate([0.32, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31], 1))
_COV_RADII = dict(enumerate([0.37, 0.32, 1.34, 0.9, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69, 1.54, 1.3, 1.18, 1.11, 1.06, 1.02, 0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31, 1.26, 1.22, 1.19, 1.16, 1.14, 1.1, 2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48, 1.44, 1.41, 1.38, 1.35, 1.33, 1.3], 1))

def bonded(atoms, coords):
    """Ordinary MOPAC bond graph (distinct from the EH+ neighbor graph)."""
    radii = _COV_RADII
    result = [[] for _ in atoms]
    for i, a in enumerate(atoms):
        for j in range(i):
            b = atoms[j]
            pair = tuple(sorted((a, b)))
            safety = 1.0 if pair == (5, 7) else 1.25 if pair == (1, 6) else 1.2 if pair in (
                (6, 6), (6, 7), (16, 16)) else 1.1
            if np.linalg.norm(coords[i]-coords[j]) < safety*(radii[a]+radii[b]):
                result[i].append(j)
                result[j].append(i)
    for i, z in enumerate(atoms):
        if len(result[i]) > 15:
            raise ValueError('PM7 supports at most 15 bonded neighbors per atom')
        if z == 1 and len(result[i]) >= 2:
            result[i] = [j for j in result[i] if atoms[j] != 1]
    return result


def _angle(x, i, j, k):
    a, b = x[i]-x[j], x[k]-x[j]
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return math.acos(float(np.clip(a@b/norm, -1, 1))) if norm > 1e-20 else 0.


def _torsion(x, i, j, k, l):
    """MOPAC's dihed/dang convention, including degenerate projections."""
    u, v, w = x[i]-x[k], x[j]-x[k], x[l]-x[k]
    d = np.linalg.norm(v)
    if d < 1e-20:
        return 0.
    costh = float(np.clip(v[2]/d, -1, 1))
    xy = d*math.sqrt(max(0., 1-costh*costh))
    if xy > 1e-6:
        cp, sp = v[1]/xy, v[0]/xy
        ux, uy = u[0]*cp-u[1]*sp, u[0]*sp+u[1]*cp
        wx, wy = w[0]*cp-w[1]*sp, w[0]*sp+w[1]*cp
        sinth = (v[0]*sp+v[1]*cp)/d
    else:
        ux, uy, wx, wy, sinth = u[0], u[1], w[0], w[1], 0.
    uy, wy = uy*costh-u[2]*sinth, wy*costh-w[2]*sinth
    if max(abs(ux), abs(uy)) < 1e-6 or max(abs(wx), abs(wy)) < 1e-6:
        return 0.
    theta = math.atan2(wx*uy-wy*ux, wx*ux+wy*uy)
    return 0. if abs(theta) < 4e-5 else theta


def _fold(angle):
    return -math.pi-angle if angle < 0 else math.pi-angle


def _neighbors(z, distance, radii, scale):
    r = np.array([radii[int(a)] for a in z])
    adj = distance < scale*(r[:, None]+r[None, :])
    np.fill_diagonal(adj, False)
    return [list(np.flatnonzero(row)) for row in adj]


def _hb_frame(center, h, neighbors, distance, bonded):
    # setup_DH_Plus retains the four nearest bonds, keeping atom-index order.
    near = list(neighbors[center])
    while len(near) > 4:
        near.remove(max(near, key=lambda j: distance[center, j]))
    ordered = sorted(near, key=lambda j: -distance[h, j])
    count = len(near)
    if count >= 3:
        frame = ordered[:3]
    elif count == 2:
        frame = ordered + [h if bonded else center]
    elif count == 1:
        first = near[0]
        other = max(neighbors[first], key=lambda j: distance[h, j])
        frame = [first, other, h if bonded else center]
    else:
        frame = [h if bonded else center]*3
    return near, count, frame


def _orientation(z, x, dist, center, h, count, frame, *, second):
    a, b, c = frame
    carbonyl = z[center] == 8 and count == 1
    nr3 = z[center] == 7 and count != 2
    if carbonyl:
        shift, shift2, torsion_shift = math.pi, math.radians(120), 0.
    elif z[center] == 7 and count == 2:
        shift = shift2 = math.radians(120)
        torsion_shift = 0.
    else:
        shift = shift2 = math.radians(109.48)
        torsion_shift = math.radians(54.74)
    check = 0.
    if nr3:
        check = _fold(_torsion(x, b, a, center, c))
        fraction = (54.74-abs(math.degrees(check)))/54.74
        torsion_shift += math.radians(fraction*35.26)
        shift -= math.radians(fraction*19.48)
        shift2 = shift
    theta = _angle(x, a, center, h)
    angle_cos = max(math.cos(shift-theta), math.cos(shift2-theta))
    if angle_cos <= 0:
        return None
    torsion = _torsion(x, b, a, center, h)
    if not carbonyl or abs(torsion) > math.pi/2:
        torsion = _fold(torsion)
    if check < 0:
        tc = math.cos(torsion_shift-torsion)
    elif check > 0:
        tc = math.cos(-torsion_shift-torsion)
    else:
        tc = max(math.cos(torsion_shift-torsion), math.cos(-torsion_shift-torsion))
    if carbonyl and dist[h, center] > dist[h, a]:
        tc = 0.
    if b == c or h == c:
        tc = 1.
    # EH_plus deliberately uses abs on the second centre only.
    if not second and tc < 0:
        return None
    tc = abs(tc) if second else tc
    return angle_cos*tc


def hydrogen_bonds(atoms, coords):
    """PM7 EH+ correction in kcal/mol, including the short O-H-O term."""
    z, x = np.asarray(atoms), np.asarray(coords, dtype=float)
    dist = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
    neighbors = _neighbors(z, dist, _HB_RADII, 4/3)
    centers, hs = np.flatnonzero(np.isin(z, [7, 8])), np.flatnonzero(z == 1)
    seen = set()
    energy = 0.
    for a in centers:
        for h in hs:
            if dist[a, h] >= 1.4:
                continue
            for b in centers:
                key = (min(a, b), h, max(a, b))
                if a == b or key in seen or dist[a, b] >= 7:
                    continue
                angle_cos = -math.cos(_angle(x, a, h, b))
                if angle_cos <= 0:
                    continue
                seen.add(key)
                near_a, count_a, frame_a = _hb_frame(a, h, neighbors, dist, h in neighbors[a])
                near_b, count_b, frame_b = _hb_frame(b, h, neighbors, dist, h in neighbors[b])
                if b in near_a or (set(near_a) & set(near_b)) - {h}:
                    continue  # 1-3 and 1-4 exclusions
                fa = _orientation(z, x, dist, a, h, count_a, frame_a, second=False)
                fb = _orientation(z, x, dist, b, h, count_b, frame_b, second=True)
                if fa is None or fb is None:
                    continue
                short, long = sorted((dist[a, h], dist[b, h]))
                damping = expit(-60*(short/1.2-1)) if long-short > .5 else 1.
                r = dist[a, b]
                damping *= expit(100*(r/2.4-1))*expit(-10*(r/7-1))
                scale = .5*sum(-.171271 if z[j] == 7 else -.098822 for j in (a, b))
                scale *= BOHR_TO_ANG**2 * HARTREE_TO_EV * EV_TO_KCAL
                energy += scale/r**2*angle_cos**2*(1-(1-fa*fb)**2)*damping
                if z[a] == z[b] == 8:
                    energy -= 2.5*math.exp(-80*max(r-2.67, 0)**2)*angle_cos**4
    return float(energy)


def corrections(atoms, coords):
    from .pm6_dh import pm6_dh_dispersion
    from .pwcct import c_triple_bond_correction, nhco_dihedral_correction
    z, x = np.asarray(atoms), np.asarray(coords, dtype=float)
    bonds = bonded(atoms, x)
    si_oh = 0.
    for o in np.flatnonzero(z == 8):
        sis = [j for j in bonds[o] if z[j] == 14]
        hs = [j for j in bonds[o] if z[j] == 1]
        if sis and hs:
            si, h = sis[-1], hs[-1]
            so, oh = np.sum((x[si]-x[o])**2)-1.7**2, np.sum((x[h]-x[o])**2)-1.
            si_oh += 15*(_angle(x, si, o, h)-math.radians(125))**2 * math.exp(-33*max(0, so)-68*max(0, oh))
    return dict(dispersion=pm6_dh_dispersion(atoms, x, pm7=True,
                                          neighbour_counts=list(map(len, bonds))),
                hydrogen_bond=hydrogen_bonds(atoms, x),
                triple_bond=c_triple_bond_correction(atoms, x),
                amide=nhco_dihedral_correction(atoms, x, htype=3.1595),
                silicon_hydroxyl=float(si_oh))


def correction_gradient(atoms, coords, step=1e-5):
    """Central differences of geometry-only corrections, in eV/Angstrom.

    No displaced SCFs are required. Together with the frozen-density SCF
    derivative this differentiates the complete PM7 optimization energy.
    """
    from .params import EV_TO_KCAL
    x = np.asarray(coords, dtype=float).copy()
    grad = np.zeros_like(x)
    for i in range(len(atoms)):
        for axis in range(3):
            value = x[i, axis]
            x[i, axis] = value + step
            plus = sum(corrections(atoms, x).values())
            x[i, axis] = value - step
            minus = sum(corrections(atoms, x).values())
            x[i, axis] = value
            grad[i, axis] = (plus-minus)/(2*step*EV_TO_KCAL)
    return grad
