"""OpenMOPAC PM7 integral feathering (mndod.F90:to_point).

The default transition is at 7 Angstrom with exponent 0.22 Angstrom^-2.
It applies to Coulomb integrals and core attraction, never overlap.
"""
import numpy as np
from .constants import COULOMB_EV_ANG


def factors(distance):
    r = np.asarray(distance, dtype=float)
    weight = np.exp(-.22*np.maximum(7.-r, 0.)**2)
    return 1.-weight, weight*COULOMB_EV_ANG/r


def tensor(w, distance, n_a, n_b):
    """Blend (...,a,a,b,b) tensors, preserving zero-padded orbitals."""
    keep, point = factors(distance)
    result = w * keep[..., None, None, None, None]
    a = np.arange(w.shape[-4]) < np.asarray(n_a)[..., None]
    b = np.arange(w.shape[-2]) < np.asarray(n_b)[..., None]
    eye_a = np.eye(w.shape[-4])*a[..., :, None]
    eye_b = np.eye(w.shape[-2])*b[..., :, None]
    return result + point[..., None, None, None, None]*np.einsum(
        '...ij,...kl->...ijkl', eye_a, eye_b)


def matrix(w, distance, charge=1.):
    """Blend electron/core or YH matrices; charge includes the energy sign."""
    keep, point = factors(distance)
    return w*keep[..., None, None] + (point*charge)[..., None, None]*np.eye(w.shape[-1])
