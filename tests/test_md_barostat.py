"""Unit tests for the mlxmolkit.md modules that need no exported force-field system."""

import importlib

import numpy as np
import pytest

from mlxmolkit.md.barostat import BAR_TO_KJ_MOL_NM3, molecule_index


def test_molecule_index_connected_components():
    # atoms 0-1-2 bonded, 3 alone, 4-5 bonded; ids follow each molecule's first atom
    mol = molecule_index(6, np.array([[0, 1], [2, 1], [4, 5]]))
    assert mol.tolist() == [0, 0, 0, 1, 2, 2]


def test_molecule_index_without_pairs():
    assert molecule_index(3, np.zeros((0, 2), np.int64)).tolist() == [0, 1, 2]


def test_bar_in_kj_per_mol_per_nm3():
    # 1 bar = 1e5 Pa = 1e5 J/m^3 = 1e-22 J/nm^3, per mole
    assert BAR_TO_KJ_MOL_NM3 == pytest.approx(1e5 * 1e-27 * 6.02214076e23 / 1e3, rel=1e-12)


@pytest.mark.parametrize("name", ["forcefield", "kernels", "integrator", "alchemy", "barostat"])
def test_md_modules_import(name):
    importlib.import_module(f"mlxmolkit.md.{name}")
