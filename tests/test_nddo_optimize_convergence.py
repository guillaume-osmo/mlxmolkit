"""nddo_optimize must report its *own* convergence, not the SCF's.

The return dict was built as

    {..., 'opt_converged': False, 'converged': True, ...,
     **{k: v for k, v in result.items() if k != 'coords'}}

with the SCF `result` splatted **last**, so its `converged`, `n_iter` and
`method` overwrote the optimizer's. Two consequences:

1. `converged` was the SCF's flag for the final single-point — essentially
   always True — so a geometry optimization that exhausted `max_iter`
   reported success. The non-converged branch also hardcoded `'converged':
   True` outright.
2. `mlxmolkit.nddo.pipeline` read `opt_result['converged']` and
   `opt_result['n_iter']` into its own `opt_converged`/`opt_n_iter`, so the
   public pipeline surfaced the SCF's convergence and iteration count as the
   geometry optimization's.

Measured on menthol at the old default of max_iter=50: the optimizer stopped
at grad_rms=0.014 (~3x grad_tol) while the energy was still falling, and
reported converged=True. See #28.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from mlxmolkit.nddo.gradient import nddo_optimize

RDLogger.DisableLog("rdApp.*")

# Flexible enough that it cannot converge in a couple of iterations, small
# enough to stay quick.
FLEXIBLE = "CCCCO"


def geometry(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        pytest.skip(f"embedding failed for {smiles}")
    return ([a.GetAtomicNum() for a in mol.GetAtoms()],
            mol.GetConformer().GetPositions())


def test_exhausting_max_iter_reports_not_converged():
    """The bug in one assertion: starved budget must not report success."""
    atoms, coords = geometry(FLEXIBLE)
    res = nddo_optimize(atoms, coords, max_iter=2, grad_tol=1e-8)

    assert res["converged"] is False
    assert res["opt_converged"] is False
    assert res["n_iter"] == 2
    assert res["opt_n_iter"] == 2
    assert res["grad_rms"] > 1e-8


def test_scf_convergence_is_preserved_under_its_own_key():
    """Fixing `converged` must not throw the SCF's flag away."""
    atoms, coords = geometry(FLEXIBLE)
    res = nddo_optimize(atoms, coords, max_iter=2, grad_tol=1e-8)

    # The SCF converges at every geometry even when the optimizer does not.
    assert res["scf_converged"] is True
    assert res["converged"] is not res["scf_converged"]
    assert isinstance(res["scf_n_iter"], (int, np.integer))


def test_converged_run_agrees_across_both_key_spellings():
    atoms, coords = geometry("CCO")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True
    assert res["converged"] == res["opt_converged"]
    assert res["n_iter"] == res["opt_n_iter"]
    assert res["grad_rms"] < 0.005
    # n_iter is the optimizer's, and the two counters are independent: the
    # SCF's is per single-point and much smaller than the optimizer's total.
    assert res["n_iter"] != res["scf_n_iter"]


def test_pipeline_reports_the_optimizers_convergence_not_the_scfs():
    """The pipeline read the SCF keys, so opt_converged was always True."""
    from mlxmolkit.nddo.pipeline import rm1_from_smiles

    res = rm1_from_smiles(FLEXIBLE, optimize=True, opt_max_iter=2,
                          opt_grad_tol=1e-8)
    if res is None:
        pytest.skip("pipeline declined this molecule")

    assert res["opt_converged"] is False
    assert res["opt_n_iter"] == 2
    # The SCF converges at every geometry, so a pipeline that echoed the SCF
    # flag here would report True no matter how starved the optimizer was.
    assert res["converged"] is True


@pytest.mark.slow
def test_menthol_converges_within_the_default_budget():
    """The molecule that exposed the too-small default cap.

    It needs 94 iterations under RM1, so the old default of 50 stopped it
    short at grad_rms ~0.014. Against MOPAC's own PM6 minimum the agreement
    improves from 0.4646 to 0.2009 kcal/mol once it is allowed to finish.
    """
    atoms, coords = geometry("CC(C)C1CCC(C)CC1O")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True, (
        f"stopped at {res['n_iter']} iterations, grad_rms={res['grad_rms']:.5f}"
    )
    assert res["grad_rms"] < 0.005
    assert res["n_iter"] > 50, (
        "menthol converging in under 50 would mean this regression test no "
        "longer covers the case that motivated the larger default"
    )


def test_easy_molecule_still_exits_early():
    """The bigger cap is a bound, not a cost — the loop must still return
    as soon as grad_tol is met."""
    atoms, coords = geometry("Clc1ccccc1")
    res = nddo_optimize(atoms, coords)

    assert res["converged"] is True
    assert res["n_iter"] < 50, f"took {res['n_iter']} of a 200 cap"
