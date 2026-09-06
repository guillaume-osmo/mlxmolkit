"""The one-centre exchange table must ship, and must load lazily.

Two regressions are covered:

  * the table was matched by the blanket `*.npz` in .gitignore and never
    committed, so a clean checkout did not have it at all;
  * the load sat at module scope, so importing gxtb_aes raised FileNotFoundError
    outright instead of failing only in the terms that need the table.
"""
import importlib

import numpy as np
import pytest


def test_module_imports_without_touching_the_table():
    m = importlib.import_module("mlxmolkit.xtb.gxtb_aes")
    # importing must not have forced the load
    assert hasattr(m, "_onecx_tables")


def test_table_ships_with_the_repo():
    from mlxmolkit.xtb.gxtb_aes import _ONECX_PATH
    import os
    assert os.path.exists(_ONECX_PATH), (
        "gxtb_onecxints_extracted.npz is missing; it has no regeneration script, "
        "so it must be committed"
    )
    from importlib.resources import files
    assert files('mlxmolkit.xtb').joinpath('data/gxtb_onecxints_extracted.npz').is_file()


def test_solver_cannot_silently_drop_missing_onsite_exchange(monkeypatch, tmp_path):
    from mlxmolkit.xtb import gxtb_aes, gxtb_scf
    monkeypatch.setattr(gxtb_aes, '_ONEC', None)
    monkeypatch.setattr(gxtb_aes, '_ONECX_PATH', str(tmp_path / 'missing.npz'))
    with pytest.raises(FileNotFoundError, match='one-centre exchange table'):
        gxtb_scf._install_onecxints()


def test_table_shapes_and_content():
    from mlxmolkit.xtb.gxtb_aes import _onecx_tables
    tbl, lidx = _onecx_tables()
    assert tbl.shape == (103, 10)
    assert lidx.shape == (4, 4)
    assert np.isfinite(tbl).all()
    # s-s one-centre exchange is absorbed into the second-order Hubbard term,
    # so the s-s column is identically zero for every element.
    assert not np.any(tbl[:, int(lidx[0, 0]) - 1])
    # carbon carries non-zero sp/pp entries
    assert tbl[5, int(lidx[0, 1]) - 1] != 0.0


def test_legacy_attribute_names_still_resolve():
    from mlxmolkit.xtb import gxtb_aes
    assert gxtb_aes.ONECX_TBL.shape == (103, 10)
    assert gxtb_aes.ONECX_LIDX.shape == (4, 4)
