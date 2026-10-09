"""RDKit's ETKDG 3D force field, term by term, as a reference for the ETK kernel.

RDKit builds this field in ``DistGeom::construct3DForceField`` (useBasicKnowledge)
or ``constructPlain3DForceField`` (``Code/DistGeom/DistGeomUtils.cpp``) from the
``CrystalFFDetails`` that ``getExperimentalTorsions`` fills
(``Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.cpp``), and does
not expose it to Python. This module rebuilds it from the source of the
2025.09 release series (tag ``Release_2025_09_1b1``; the installed RDKit is
2025.09.4), as follows:

* distance and angle constraints are RDKit's own contribs, evaluated by an
  ``rdForceField.ForceField`` (``AddDistanceConstraint`` is
  ``ForceFields::DistanceConstraintContrib``: 0.5 k (d - bound)^2;
  ``UFFAddAngleConstraint`` is ``ForceFields::AngleConstraintContrib``:
  k (theta_deg - bound)^2) on an otherwise empty force field;
* the CSD / flat-ring torsions (``CrystalFF::TorsionAngleContribs``) and the
  UFF inversion (``UFF::InversionContribs``) are not reachable from Python and
  are evaluated here in float64 with their C++ formulas; the gradients of
  those two are the exact derivatives of their energies. (RDKit's own torsion
  gradient uses V[4]/sign[4] in place of V[5]/sign[5] for the cos(6 phi) term,
  a bug in the C++; it matters only for torsions with V[5] != 0.)

The term selection follows the C++: every bond is a 1-2 term at its reference
length +/- 0.01 A (k 100); every bond angle a 1-3 term, which is a 179-180 deg
angle constraint (k 1) at a triple bond or between two double bonds at a
degree-2 atom, the bounds-matrix window (k 100) when its centre carries an
improper, else the reference distance +/- 0.01 A (k 100); every other pair,
except the end atoms of each torsion term, is held to its bounds-matrix window
with k = 10 * boundsMatForceScaling. Impropers sit on C/N/O atoms that are SP2
with exactly three neighbours, as three inversion terms of force constant
10 * (50 if C bound to an SP2 O else 6) / 3. Flat-ring torsions (V2 = 100) are
added for 4- to 6-membered rings whose four consecutive atoms are SP2, on
bonds no CSD torsion uses. One approximation: RDKit also skips flat-ring
torsions on a bond of a bridged ring system (or in more than three rings)
when a CSD pattern matched it, although it then adds no CSD torsion there.
Which bonds a pattern matched is not visible from Python, and the ETKDG
patterns are for acyclic bonds apart from the macrocycle ones, so such bonds
are taken as unmatched: matching RDKit's own pattern files
(torsionPreferences_v2.in + _macrocycles.in) selects the same flat-ring
torsions on all 677 probe isomers, while taking them as matched would
differ on 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from mlxmolkit.etk_extract import ETKDG_VARIANTS

SP2 = Chem.HybridizationType.SP2
KNOWN_DIST_TOL = 0.01
KNOWN_DIST_FORCE_CONSTANT = 100.0


@dataclass
class RDKitETKTerms:
    torsions: list   # (i, j, k, l, V[6], signs[6])
    inversions: list  # (I, J(centre), K, L, force constant)
    bonds: list      # (i, j)
    angles: list     # (i, j(centre), k, is_linear)
    improper_centres: set


def rdkit_etk_terms(mol: Chem.Mol, variant: str = "ETKDGv3") -> RDKitETKTerms:
    use_exp, use_basic = ETKDG_VARIANTS[variant][0], ETKDG_VARIANTS[variant][1]
    torsions = []
    done = set()
    if use_exp or use_basic:
        factory = getattr(rdDistGeom, variant, None)
        params = factory() if factory is not None else None
        if params is None:  # a union variant without an RDKit factory
            _, _, sr, mc, _, ver = ETKDG_VARIANTS[variant]
            exp = rdDistGeom.GetExperimentalTorsions(
                mol, useExpTorsionAnglePrefs=use_exp, useSmallRingTorsions=sr,
                useMacrocycleTorsions=mc, useBasicKnowledge=use_basic, ETversion=ver)
        else:
            exp = rdDistGeom.GetExperimentalTorsions(mol, params)
        for t in exp:
            q = list(t["atomIndices"])
            torsions.append((*q, np.array(t["V"], float), np.array(t["signs"], float)))
            done.add(mol.GetBondBetweenAtoms(q[1], q[2]).GetIdx())
    inversions, centres = [], set()
    if use_basic:
        for a in mol.GetAtoms():
            if a.GetAtomicNum() in (6, 7, 8) and a.GetHybridization() == SP2 and a.GetDegree() == 3:
                nb = [x.GetIdx() for x in a.GetNeighbors()]
                bound_o = a.GetAtomicNum() == 6 and any(
                    x.GetAtomicNum() == 8 and x.GetHybridization() == SP2 for x in a.GetNeighbors())
                k = 10.0 * (50.0 if bound_o else 6.0) / 3.0
                c = a.GetIdx()
                for (i, kk, l) in ((nb[0], nb[1], nb[2]), (nb[0], nb[2], nb[1]), (nb[1], nb[2], nb[0])):
                    inversions.append((i, c, kk, l, k))
                centres.add(c)
        for ring in mol.GetRingInfo().AtomRings():
            n = len(ring)
            if n < 4 or n > 6:
                continue
            for i in range(n):
                q = [ring[i], ring[(i + 1) % n], ring[(i + 2) % n], ring[(i + 3) % n]]
                bid = mol.GetBondBetweenAtoms(q[1], q[2]).GetIdx()
                if bid in done or any(mol.GetAtomWithIdx(x).GetHybridization() != SP2 for x in q):
                    continue
                done.add(bid)
                v = np.zeros(6); v[1] = 100.0
                s = np.ones(6); s[1] = -1.0
                torsions.append((*q, v, s))
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    angles = []
    for bi in mol.GetBonds():
        for bj in mol.GetBonds():
            if bj.GetIdx() <= bi.GetIdx():
                continue
            common = {bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()} & {bj.GetBeginAtomIdx(), bj.GetEndAtomIdx()}
            if len(common) != 1:
                continue
            c = common.pop()
            lin = (bi.GetBondType() == Chem.BondType.TRIPLE or bj.GetBondType() == Chem.BondType.TRIPLE
                   or (bi.GetBondType() == Chem.BondType.DOUBLE and bj.GetBondType() == Chem.BondType.DOUBLE
                       and mol.GetAtomWithIdx(c).GetDegree() == 2))
            angles.append((bi.GetOtherAtomIdx(c), c, bj.GetOtherAtomIdx(c), lin and use_basic))
    return RDKitETKTerms(torsions, inversions, bonds, angles, centres)


def _empty_field(n):
    """An rdForceField.ForceField over n points with no terms of its own."""
    m = Chem.MolFromSmiles(".".join(["[He]"] * n))
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, (1.0e4 * i, 0.0, 0.0))
    m.AddConformer(conf)
    block = rdBase.BlockLogs()  # helium has no UFF type: say nothing about it
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, vdwThresh=1.0e-6)
    del block
    ff.Initialize()
    assert ff.CalcEnergy() == 0.0
    return ff


def _eval_ff(ff, x):
    pos = [float(v) for v in np.asarray(x, np.float64).ravel()]
    return float(ff.CalcEnergy(pos)), np.array(ff.CalcGrad(pos), np.float64)


def _torsion(x, q, V, s):
    """CrystalFF::TorsionAngleContribs energy and its exact gradient."""
    i, j, k, l = q
    r1 = x[i] - x[j]; r2 = x[k] - x[j]; r3 = -r2; r4 = x[l] - x[k]
    t1 = np.cross(r1, r2); t2 = np.cross(r3, r4)
    d1 = np.linalg.norm(t1); d2 = np.linalg.norm(t2)
    g = np.zeros_like(x)
    if d1 < 1e-16 or d2 < 1e-16:
        return float(_tors_e(0.0, V, s)), g
    t1n = t1 / d1; t2n = t2 / d2
    c = float(np.clip(t1n @ t2n, -1.0, 1.0))
    e = _tors_e(c, V, s)
    dEdc = _tors_dedc(c, V, s)
    dc_dt1 = (t2n - c * t1n) / d1
    dc_dt2 = (t1n - c * t2n) / d2
    # t1 = r1 x r2, t2 = r3 x r4 = (x_j - x_k) x (x_l - x_k)
    g[i] += dEdc * np.cross(r2, dc_dt1)
    g[k] += dEdc * np.cross(dc_dt1, r1)
    g[j] -= dEdc * (np.cross(r2, dc_dt1) + np.cross(dc_dt1, r1))
    g[j] += dEdc * np.cross(r4, dc_dt2)
    g[l] += dEdc * np.cross(dc_dt2, r3)
    g[k] -= dEdc * (np.cross(r4, dc_dt2) + np.cross(dc_dt2, r3))
    return float(e), g


def _cheb(c):
    return [c, 2 * c**2 - 1, 4 * c**3 - 3 * c, 8 * c**4 - 8 * c**2 + 1,
            16 * c**5 - 20 * c**3 + 5 * c, 32 * c**6 - 48 * c**4 + 18 * c**2 - 1]


def _cheb_d(c):
    return [1.0, 4 * c, 12 * c**2 - 3, 32 * c**3 - 16 * c, 80 * c**4 - 60 * c**2 + 5,
            192 * c**5 - 192 * c**3 + 36 * c]


def _tors_e(c, V, s):
    return float(sum(V[m] * (1 + s[m] * t) for m, t in enumerate(_cheb(c))))


def _tors_dedc(c, V, s):
    return float(sum(V[m] * s[m] * t for m, t in enumerate(_cheb_d(c))))


def _inversion(x, q, K):
    """UFF::InversionContrib (C0 = 1, C1 = -1, C2 = 0): K (1 - sin Y), exact gradient."""
    I, J, Kk, L = q
    rJI = x[I] - x[J]; rJK = x[Kk] - x[J]; rJL = x[L] - x[J]
    a, b, d = np.linalg.norm(rJI), np.linalg.norm(rJK), np.linalg.norm(rJL)
    n = np.cross(rJI, rJK) / (a * b)
    ln = np.linalg.norm(n)
    g = np.zeros_like(x)
    if ln < 1e-8:
        return float(K * (1.0 - 1.0)), g  # cosY = 0 -> sinY = 1
    cy = float(n @ rJL / (d * ln))
    sy = np.sqrt(max(1.0 - cy * cy, 0.0))
    e = K * (1.0 - sy)
    if sy < 1e-8:
        return float(e), g
    dEdcy = K * cy / sy
    # cy = (u x v) . w / (|u x v| |w|) with u = rJI, v = rJK, w = rJL
    cr = np.cross(rJI, rJK); lc = np.linalg.norm(cr)
    nh = cr / lc; wh = rJL / d
    dcy_dw = (nh - cy * wh) / d
    dcy_dcr = (wh - cy * nh) / lc
    dcy_du = np.cross(rJK, dcy_dcr)
    dcy_dv = np.cross(dcy_dcr, rJI)
    g[I] += dEdcy * dcy_du
    g[Kk] += dEdcy * dcy_dv
    g[L] += dEdcy * dcy_dw
    g[J] -= dEdcy * (dcy_du + dcy_dv + dcy_dw)
    return float(e), g


def rdkit_etk_energy_terms(mol, bmat, x_ref, x, variant="ETKDGv3", bounds_force_scaling=1.0):
    """{group: (energy, gradient (n, 3))} of RDKit's ETK field at x.

    1-2 and 1-3 reference lengths are measured on x_ref (RDKit measures them on
    the conformer that enters the ETK stage). Groups: torsion, improper, 1-2,
    1-3 (distance and linear-angle terms over bond angles), long_range, total.
    """
    T = rdkit_etk_terms(mol, variant)
    n = mol.GetNumAtoms()
    x = np.asarray(x, np.float64).reshape(n, 3)
    x_ref = np.asarray(x_ref, np.float64).reshape(n, 3)
    out = {}
    e = 0.0; g = np.zeros((n, 3))
    for (i, j, k, l, V, s) in T.torsions:
        ee, gg = _torsion(x, (i, j, k, l), V, s); e += ee; g += gg
    out["torsion"] = (e, g)
    e = 0.0; g = np.zeros((n, 3))
    for (I, J, K, L, kc) in T.inversions:
        ee, gg = _inversion(x, (I, J, K, L), kc); e += ee; g += gg
    out["improper"] = (e, g)
    pairs = set()
    for (i, j, k, l, _, _) in T.torsions:
        pairs.add((min(i, l), max(i, l)))
    ff = _empty_field(n)
    for i, j in T.bonds:
        d = float(np.linalg.norm(x_ref[i] - x_ref[j]))
        ff.AddDistanceConstraint(i, j, d - KNOWN_DIST_TOL, d + KNOWN_DIST_TOL, KNOWN_DIST_FORCE_CONSTANT)
        pairs.add((min(i, j), max(i, j)))
    out["1-2"] = _eval_ff(ff, x)
    ff = _empty_field(n)
    for i, j, k, lin in T.angles:
        pairs.add((min(i, k), max(i, k)))
        if lin:
            ff.UFFAddAngleConstraint(i, j, k, False, 179.0, 180.0, 1.0)
        elif j in T.improper_centres:
            lo, hi = min(i, k), max(i, k)
            ff.AddDistanceConstraint(i, k, float(bmat[hi, lo]), float(bmat[lo, hi]),
                                     KNOWN_DIST_FORCE_CONSTANT)
        else:
            d = float(np.linalg.norm(x_ref[i] - x_ref[k]))
            ff.AddDistanceConstraint(i, k, d - KNOWN_DIST_TOL, d + KNOWN_DIST_TOL, KNOWN_DIST_FORCE_CONSTANT)
    out["1-3"] = _eval_ff(ff, x)
    ff = _empty_field(n)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in pairs:
                ff.AddDistanceConstraint(i, j, float(bmat[j, i]), float(bmat[i, j]),
                                         10.0 * bounds_force_scaling)
    out["long_range"] = _eval_ff(ff, x)
    out = {k: (v[0], np.asarray(v[1]).reshape(n, 3)) for k, v in out.items()}
    out["total"] = (sum(v[0] for v in out.values()), sum(v[1] for v in out.values()))
    return out
