"""
Parallel Preconditioned L-BFGS (PP-LBFGS) optimizer.

Port of Cuby4 optimizer_PLBFGS.rb (Klemsa & Řezáč, Chem. Phys. Lett. 2013).

Key idea: In each optimization step, compute 4 gradients in parallel:
  1. Gradient at current geometry (standard)
  2-4. Gradients at displaced geometries (one color × 3 XYZ directions)

The displaced gradients update a sparse Hessian preconditioning matrix
built from the molecular connectivity graph (covalent bonds only).
This gives 2-4x faster convergence than plain L-BFGS.

Graph coloring with distance-2 ensures atoms of the same color don't
share neighbors, so displacing all atoms of one color simultaneously
gives correct Hessian elements via finite differences.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def graph_color_d2(connectivity: np.ndarray) -> np.ndarray:
    """Greedy graph coloring with distance 2.

    Two atoms get the same color only if they don't share any neighbor.
    This means: displacing all atoms of one color simultaneously gives
    valid finite-difference Hessian elements for bonded pairs.

    Args:
        connectivity: (n_atoms, n_atoms) boolean adjacency matrix

    Returns:
        colors: (n_atoms,) integer color assignment
    """
    n = len(connectivity)
    colors = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        # Forbidden colors: colors of neighbors AND second-order neighbors
        forbidden = set()
        neighbors_i = np.where(connectivity[i])[0]
        for j in neighbors_i:
            if colors[j] >= 0:
                forbidden.add(colors[j])
            # Second-order neighbors
            neighbors_j = np.where(connectivity[j])[0]
            for k in neighbors_j:
                if k != i and colors[k] >= 0:
                    forbidden.add(colors[k])

        # Find minimum available color
        c = 0
        while c in forbidden:
            c += 1
        colors[i] = c

    return colors


def build_connectivity(atoms: list[int], coords: np.ndarray, scale: float = 1.3) -> np.ndarray:
    """Build covalent bond connectivity matrix from geometry.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        scale: scale factor for covalent radii sum

    Returns:
        connectivity: (n_atoms, n_atoms) boolean
    """
    # Covalent radii (Angstrom)
    COVALENT_RADII = {
        1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
    }

    n = len(atoms)
    conn = np.zeros((n, n), dtype=bool)

    for i in range(n):
        ri = COVALENT_RADII.get(atoms[i], 1.5)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(atoms[j], 1.5)
            d = np.linalg.norm(coords[i] - coords[j])
            if d < (ri + rj) * scale:
                conn[i, j] = True
                conn[j, i] = True

    return conn


class PPLBFGSOptimizer:
    """Parallel Preconditioned L-BFGS geometry optimizer.

    Usage:
        opt = PPLBFGSOptimizer(atoms, coords)
        for step in range(max_steps):
            # Compute gradient at current + displaced geometries (parallel!)
            grad, displaced_grads = opt.get_gradient_requests()
            # ... compute all gradients in batch ...
            opt.feed_gradients(grad, displaced_grads)
            converged = opt.step()
            if converged: break
    """

    def __init__(
        self,
        atoms: list[int],
        coords: np.ndarray,
        lbfgs_n: int = 20,
        trust_radius: float = 0.3,
        fd_step: float = 0.001,
        n_colors_per_step: int = 1,
    ):
        self.atoms = list(atoms)
        self.n_atoms = len(atoms)
        self.n_vars = self.n_atoms * 3
        self.coords = np.asarray(coords, dtype=np.float64).copy()
        self.vector = self.coords.flatten()

        # L-BFGS parameters
        self.lbfgs_n = lbfgs_n
        self.trust_radius = trust_radius
        self.fd_step = fd_step
        self.n_colors_per_step = n_colors_per_step

        # Build connectivity and graph coloring
        self.connectivity = build_connectivity(atoms, coords)
        self.colors = graph_color_d2(self.connectivity)
        self.n_colors = int(self.colors.max()) + 1

        # Sparse Hessian preconditioning matrix
        self.H0 = lil_matrix((self.n_vars, self.n_vars))

        # L-BFGS history
        self.old_grads = []
        self.old_vecs = []

        # State
        self.cycle = 0
        self.color_idx = 0
        self.energy = None
        self.gradient = None

    def get_displaced_geometries(self) -> list[np.ndarray]:
        """Get list of displaced geometries for parallel gradient computation.

        Returns list of (n_atoms, 3) coordinate arrays.
        First element is the current geometry.
        Remaining elements are finite-difference displacements for graph-colored atoms.
        """
        geometries = [self.coords.copy()]  # Current geometry

        if self.cycle == 0:
            # First cycle: all colors
            colors_to_compute = range(self.n_colors)
        else:
            # Subsequent: only n_colors_per_step colors (rotating)
            colors_to_compute = [
                (self.color_idx + i) % self.n_colors
                for i in range(self.n_colors_per_step)
            ]

        for color in colors_to_compute:
            for xyz in range(3):
                disp = self.coords.copy()
                for j in range(self.n_atoms):
                    if self.colors[j] == color:
                        disp[j, xyz] += self.fd_step
                geometries.append(disp)

        self._current_colors = list(colors_to_compute)
        return geometries

    def feed_results(
        self,
        energies: list[float],
        gradients: list[np.ndarray],
    ) -> bool:
        """Feed computed energies and gradients, perform optimization step.

        Args:
            energies: list of energies (first = current, rest = displaced)
            gradients: list of (n_atoms, 3) gradient arrays

        Returns:
            converged: True if gradient RMS < tolerance
        """
        self.energy = energies[0]
        self.gradient = gradients[0].flatten()

        # Update sparse Hessian from displaced gradients
        idx = 1  # skip current geometry
        for color in self._current_colors:
            dif_grads = []
            for xyz in range(3):
                dg = gradients[idx].flatten() - self.gradient
                dif_grads.append(dg)
                idx += 1

            # Update Hessian blocks
            for j in range(self.n_atoms):
                if self.colors[j] == color:
                    # Diagonal 3×3 block
                    for k in range(3):
                        for l in range(3):
                            val = dif_grads[l][3*j+k] / self.fd_step
                            self.H0[3*j+k, 3*j+l] = val
                            self.H0[3*j+l, 3*j+k] = val
                else:
                    # Off-diagonal: bonded neighbor with this color
                    neighbors = np.where(self.connectivity[j])[0]
                    for i in neighbors:
                        if self.colors[i] == color:
                            for k in range(3):
                                for l in range(3):
                                    val = dif_grads[k][3*j+l] / self.fd_step
                                    self.H0[3*i+k, 3*j+l] = val
                                    self.H0[3*j+l, 3*i+k] = val
                            break

        # Rotate color index
        if self.cycle > 0:
            self.color_idx = (self.color_idx + self.n_colors_per_step) % self.n_colors

        # Compute L-BFGS direction with sparse preconditioning
        direction = self._compute_direction()

        # Trust radius scaling
        step_norm = np.linalg.norm(direction)
        if step_norm > self.trust_radius:
            direction *= self.trust_radius / step_norm

        # Update coordinates
        self.vector += direction
        self.coords = self.vector.reshape(self.n_atoms, 3)

        # Save history
        self.old_grads.insert(0, self.gradient.copy())
        self.old_vecs.insert(0, self.vector.copy())
        if len(self.old_grads) > self.lbfgs_n + 1:
            self.old_grads.pop()
            self.old_vecs.pop()

        self.cycle += 1

        # Check convergence
        grad_rms = np.sqrt(np.mean(self.gradient ** 2))
        return grad_rms

    def _compute_direction(self) -> np.ndarray:
        """L-BFGS direction with sparse Hessian preconditioning."""
        g = self.gradient
        m = min(len(self.old_grads) - 1, self.lbfgs_n) if self.old_grads else 0

        if m == 0:
            # First step: use preconditioned steepest descent
            H0_csr = csr_matrix(self.H0)
            # Add diagonal regularization (beta * I)
            beta = 1.0 / 5000.0  # initial Hessian diagonal estimate
            H0_reg = H0_csr + beta * csr_matrix(np.eye(self.n_vars))
            try:
                z = spsolve(H0_reg, g)
            except Exception:
                z = g / 5000.0
            return -z

        # L-BFGS two-loop recursion with sparse preconditioning
        q = g.copy()
        s_list, y_list, rho_list, alpha_list = [], [], [], []

        for i in range(1, m + 1):
            if i >= len(self.old_vecs):
                break
            s = self.old_vecs[i-1] - self.old_vecs[i]
            y = self.old_grads[i-1] - self.old_grads[i]
            sy = y.dot(s)
            if abs(sy) < 1e-20:
                continue
            rho = 1.0 / sy
            alpha = rho * s.dot(q)
            q -= alpha * y
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)
            alpha_list.append(alpha)

        # Preconditioning: solve H0 * z = q
        if self.cycle > 0:
            s_last = self.old_vecs[0] - self.old_vecs[1] if len(self.old_vecs) > 1 else np.ones(self.n_vars)
            y_last = self.old_grads[0] - self.old_grads[1] if len(self.old_grads) > 1 else np.ones(self.n_vars)
            h0_s = self.H0 @ s_last
            beta_I = np.linalg.norm(y_last - h0_s) / (np.linalg.norm(s_last) + 1e-30)
        else:
            beta_I = 1.0 / 5000.0

        H0_csr = csr_matrix(self.H0)
        H0_reg = H0_csr.copy()
        # Add beta*I to diagonal
        for i in range(self.n_vars):
            H0_reg[i, i] += beta_I

        try:
            z = spsolve(csr_matrix(H0_reg), q)
        except Exception:
            z = q * beta_I

        # Check direction is descent
        if z.dot(g) < 0:
            z = q * beta_I  # Fallback to scaled gradient

        # Second loop
        for i in range(len(s_list) - 1, -1, -1):
            beta = rho_list[i] * y_list[i].dot(z)
            z += s_list[i] * (alpha_list[i] - beta)

        return -z
