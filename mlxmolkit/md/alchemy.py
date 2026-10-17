"""Alchemical decoupling of one solute per box, lambda state per box (for HREX / MBAR).

Same thermodynamic path as openmmtools' AbsoluteAlchemicalFactory with
annihilate_electrostatics=True, annihilate_sterics=False (energies match an OpenMM
implementation of this path to 1e-4 kJ/mol):
  * solute charges scaled by lambda_elec everywhere (real space, PME, exceptions, Ewald self
    term), so intramolecular electrostatics are annihilated too;
  * solute-solvent LJ: soft-core U = lam 4 eps x (x - 1), x = 1/(alpha (1 - lam) + (r/sigma)^6);
  * intramolecular LJ untouched; dispersion tail = solvent-solvent + lam_sterics * solute-solvent.
Every box b carries its own (lam_s[b], lam_e[b]); evaluating energies of one configuration at
all K states is the same call on K copies with K lambda pairs.
"""

import mlx.core as mx
import numpy as np

from mlxmolkit.md import forcefield as ff
from mlxmolkit.md import kernels
from mlxmolkit.md.integrator import Dynamics


class Alchemy:
    def __init__(self, t: ff.Topology, n_solute: int, e_disp_solvent: float, e_tail_solute: float, box_len: np.ndarray, softcore_alpha: float = 0.5):
        self.t = t
        self.n_solute = n_solute
        al = np.zeros(t.n, np.int32)
        al[:n_solute] = 1
        self.alch = mx.array(al)
        self.solute_mask = mx.array(al.astype(bool))
        exc = np.asarray(t.exc_idx)
        self.exc_solute = mx.array(((exc[:, 0] < n_solute) & (exc[:, 1] < n_solute)).astype(np.int32))
        self.e_disp_solvent, self.e_tail_solute = e_disp_solvent, e_tail_solute
        self.alpha_sc = softcore_alpha
        self.pme = ff.PME(t, box_len)
        self.bonded = kernels.BondedTerms(t)

    def energy_forces(self, x, box, nbr, nnb, maxnb, lam_s, lam_e):
        t = self.t
        qb = t.q[None, :] * mx.where(self.solute_mask[None, :], lam_e[:, None], 1.0)
        ed, fd = kernels.direct_fused(x, box, nbr, nnb, maxnb, t, qb=qb, alch=self.alch, lam_s=lam_s, softcore_alpha=self.alpha_sc)
        er, fr = ff.pme_energy_forces_fused(self.pme, x, box, qb=qb)
        eb, fb = self.bonded.energy_forces(x, qb=qb, lam_e=lam_e, exc_solute=self.exc_solute)
        v = box[:, 0] * box[:, 1] * box[:, 2]
        self_e = -ff.ONE_4PI_EPS0 * t.alpha / np.sqrt(np.pi) * (qb * qb).sum(-1)
        disp = (self.e_disp_solvent + lam_s * self.e_tail_solute) * t.v0 / v
        return ed + er + eb + self_e + disp, fd + fr + fb


class AlchemicalDynamics(Dynamics):
    """Dynamics where box b samples lambda state state_of[b] (lam_s, lam_e ladders)."""

    def __init__(self, t, d, alch: Alchemy, lam_s: np.ndarray, lam_e: np.ndarray, state_of: np.ndarray, **kw):
        super().__init__(t, d, B=len(state_of), **kw)
        self.al = alch
        self.lam_s_ladder = mx.array(np.asarray(lam_s, np.float32))
        self.lam_e_ladder = mx.array(np.asarray(lam_e, np.float32))
        self.set_states(state_of)

    def set_states(self, state_of):
        self.state_of = np.asarray(state_of)
        idx = mx.array(self.state_of.astype(np.int32))
        self.lam_s, self.lam_e = self.lam_s_ladder[idx], self.lam_e_ladder[idx]

    def forces(self, x, box, nbr, nnb):
        return self.al.energy_forces(x, box, nbr, nnb, self.maxnb, self.lam_s, self.lam_e)

    def reduced_energies_all_states(self, x, box, nbr, nnb, kT):
        """u[b, k] = E(x_b at state k) / kT for every box b and state k (one batched call)."""
        B, N = x.shape[0], x.shape[1]
        K = self.lam_s_ladder.shape[0]
        xs = mx.repeat(x, K, axis=0)  # [B*K, N, 3], box-major
        bs = mx.repeat(box, K, axis=0)
        nb = mx.repeat(nbr.reshape(B, -1), K, axis=0).reshape(-1)
        nn = mx.repeat(nnb.reshape(B, -1), K, axis=0).reshape(-1)
        ls = mx.tile(self.lam_s_ladder, (B,))
        le = mx.tile(self.lam_e_ladder, (B,))
        e, _ = self.al.energy_forces(xs, bs, nb, nn, self.maxnb, ls, le)
        return (e / kT).reshape(B, K)
