"""Monte Carlo barostat (OpenMM's MonteCarloBarostat), batched: every box makes its own move.

A move (OpenMM's reference implementation, isotropic, molecular scaling):
  dV = scale * 2 (u - 0.5);  V' = V + dV;  s = (V'/V)^(1/3)
  every molecule is translated rigidly so that its geometric centre c goes to s c (constraints
  and internal geometry untouched), the box edges are multiplied by s;
  w = U(x') - U(x) + P dV - N_mol kT ln(V'/V);  accept if w <= 0 or rand < exp(-w/kT).
Step size: 1 % of the volume to start; after >= 10 attempts the scale shrinks by 1.1 below 25 %
acceptance and grows by 1.1 (capped at 0.3 V) above 75 %, the counters resetting only then.
Velocities are not scaled (OpenMM does not either).
The energy U is whatever `dyn.forces` returns for the box (for AlchemicalDynamics, the box's own
lambda state). In an HREX/MBAR run the P V term is the same for a sample at every state, so it
cancels from swap probabilities and from MBAR; boxes keep their volumes when states swap.
"""

import mlx.core as mx
import numpy as np

KB = 0.0083144626  # kJ/mol/K
BAR_TO_KJ_MOL_NM3 = 0.0602214076  # 1 bar = 1e5 J/m^3 = 1e-22 J/nm^3, times N_A


def molecule_index(n: int, pairs: np.ndarray) -> np.ndarray:
    """Molecule id of every atom: connected components of the given atom pairs (bonds, constraints,
    exceptions), numbered in order of each molecule's first atom."""
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, mol = np.unique(roots, return_inverse=True)
    return mol


class MonteCarloBarostat:
    def __init__(self, dyn, d, pressure_bar: float = 1.0, frequency: int = 25, seed: int = 0):
        self.dyn = dyn
        n = dyn.t.n
        pairs = [np.asarray(d["bond_idx"]).reshape(-1, 2), np.asarray(d["exc_idx"]).reshape(-1, 2)]
        if len(d["constraints"]):
            pairs.append(np.asarray(d["constraints"])[:, :2])
        mol = molecule_index(n, np.concatenate(pairs).astype(np.int64))
        self.n_mol = int(mol.max()) + 1
        counts = np.bincount(mol)
        first = np.full(self.n_mol, n, np.int64)
        np.minimum.at(first, mol, np.arange(n))  # lowest atom index of each molecule
        w = np.zeros((self.n_mol, n), np.float32)
        w[mol, np.arange(n)] = 1.0 / counts[mol]
        self.w = mx.array(w)  # [M, N]: geometric-centre weights
        self.mol = mx.array(mol.astype(np.int32))
        self.first = mx.array(first.astype(np.int32))
        self.ref = mx.array(first[mol].astype(np.int32))  # each atom's molecule reference atom
        self.p = pressure_bar * BAR_TO_KJ_MOL_NM3
        self.kt = KB * dyn.T
        self.frequency = frequency
        self.rng = np.random.default_rng(seed)
        self.scale = None
        self.n_att = np.zeros(dyn.B, np.int64)
        self.n_acc = np.zeros(dyn.B, np.int64)
        self.total_att = 0
        self.total_acc = np.zeros(dyn.B, np.int64)

    def centres(self, x: mx.array, box: mx.array) -> mx.array:
        """Geometric centre of every molecule, [B, M, 3]; min-imaged to the molecule's first atom so a
        molecule split across the boundary still has one centre."""
        L = box[:, None, :]
        dx = x - x[:, self.ref, :]
        dx = dx - L * mx.round(dx / L)
        return x[:, self.first, :] + mx.matmul(self.w, dx)

    def step(self, x, box, e, f, nbr, nnb):
        """One volume move in every box. Returns (x, box, e, f, nbr, nnb, accepted[B])."""
        B, N = x.shape[0], x.shape[1]
        mx.eval(box, e)
        bl = np.asarray(box, np.float64)
        vol = bl.prod(axis=1)
        if self.scale is None:
            self.scale = 0.01 * vol
        dv = self.scale * 2.0 * (self.rng.random(B) - 0.5)
        v_new = vol + dv
        s = (v_new / vol) ** (1.0 / 3.0)
        s_mx = mx.array(s.astype(np.float32))
        c = self.centres(x, box)[:, self.mol, :]  # [B, N, 3], centre of each atom's molecule
        x_new = x + (s_mx - 1.0)[:, None, None] * c
        box_new = mx.array((bl * s[:, None]).astype(np.float32))
        nbr_n, nnb_n, _ = self.dyn.nlist(x_new, box_new)
        e_n, f_n = self.dyn.forces(x_new, box_new, nbr_n, nnb_n)
        mx.eval(e_n)
        w = (np.asarray(e_n, np.float64) - np.asarray(e, np.float64)) + self.p * dv - self.n_mol * self.kt * np.log(v_new / vol)
        acc = (w <= 0.0) | (self.rng.random(B) < np.exp(-np.maximum(w, 0.0) / self.kt))
        # OpenMM's step-size adaptation
        self.n_att += 1
        self.n_acc += acc
        self.total_att += 1
        self.total_acc += acc
        ready = self.n_att >= 10
        low = ready & (self.n_acc < 0.25 * self.n_att)
        high = ready & ~low & (self.n_acc > 0.75 * self.n_att)
        self.scale = np.where(low, self.scale / 1.1, self.scale)
        self.scale = np.where(high, np.minimum(self.scale * 1.1, 0.3 * vol), self.scale)
        self.n_att[low | high] = 0
        self.n_acc[low | high] = 0
        a = mx.array(acc)
        x = mx.where(a[:, None, None], x_new, x)
        box = mx.where(a[:, None], box_new, box)
        e = mx.where(a, e_n, e)
        f = mx.where(a[:, None, None], f_n, f)
        nbr = mx.where(a[:, None], nbr_n.reshape(B, -1), nbr.reshape(B, -1)).reshape(-1)
        nnb = mx.where(a[:, None], nnb_n.reshape(B, N), nnb.reshape(B, N)).reshape(-1)
        return x, box, e, f, nbr, nnb, acc

    def acceptance(self) -> np.ndarray:
        return self.total_acc / max(self.total_att, 1)
