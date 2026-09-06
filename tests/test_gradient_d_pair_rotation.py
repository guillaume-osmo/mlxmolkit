"""Batched sp corners of d gradient pairs must match the scalar pair builder."""
import numpy as np
import pytest

from mlxmolkit.nddo.anal_grad import _pair_energy_many, _pair_terms
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.scf import _build_basis_info


@pytest.mark.parametrize('method', ['PM6', 'PM6_ORG'])
@pytest.mark.parametrize('pair', [(16, 16), (16, 6), (6, 16), (16, 1), (1, 16)])
def test_displaced_d_pair_energy_matches_scalar(method, pair):
    # Spectator atoms keep the pair's basis blocks away from zero and adjacent
    # offsets. Random symmetric densities exercise every Coulomb/exchange term.
    atoms = [1, pair[0], 8, pair[1]]
    table = get_params(method)
    charge = sum(table[z].n_valence for z in atoms) % 2
    info = _build_basis_info(atoms, table, molecular_charge=charge)
    params, starts, n = info['params'], info['atom_basis_start'], info['n_basis']
    rng = np.random.default_rng(732)
    coords = rng.normal(size=(4, 3))
    matrix = rng.normal(size=(n, n))
    density = matrix + matrix.T
    for delta in (None, np.array([1e-5, 0., 0.]), np.array([0., -1e-5, 0.])):
        displaced = coords.copy()
        if delta is not None:
            displaced[3] += delta
        h, t = _pair_terms(params, displaced, 1, 3, starts, density, n)
        reference = np.sum(density * (2 * h + t))
        actual = _pair_energy_many(params, coords, [(1, 3)], starts, density, n,
                                   shift=delta)[(1, 3)]
        assert actual == pytest.approx(reference, rel=1e-12, abs=1e-10)
