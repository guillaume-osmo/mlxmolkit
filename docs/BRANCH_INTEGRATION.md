# Integration of the g-xTB and NDDO branches

Remote comparison on 2026-09-06:

| Ref | Commit | Integration status |
|---|---|---|
| Previous main | b777b439ac48ea151fc03cd72e68081d2bcff02f | Separate clean snapshot history |
| publish-clean | 3e1a710 | Same tree as a63bea3; its separate merge commit is also retained |
| perf/nddo-fock-plan | d1b5784 | Already merged into fix/nddo-mopac-gates |
| perf/nddo-round2 | 85b2eab | Already merged into fix/nddo-mopac-gates |
| fix/nddo-mopac-gates | 0a508e98f958fcffea1483179e8d69cca8a2c751 | Contains the complete published branch stack |

The integration preserves both clean histories. The branch stack's tree
supersedes the older main snapshot: its 49-file difference includes the
g-xTB v2.0.1 solver, C++ overlap kernels, method-specific PM6-ORG integrals,
dipole correction, and batched Fock/rotation optimizations. No historical
archive branches or private recovery notes are imported.

## Parameter archive reconciliation

An array-by-array comparison against previous main found the following
runtime tables unchanged in the consolidated gxtb_v2.npz archive:

- g-xTB: 44 arrays.
- q-vSZP: 15 arrays.
- EEQBC: 10 arrays.
- MCTC: 2 arrays.

Only extraction metadata was omitted from those four old archives.
The D4Srev archive retains 17 identical runtime arrays and removes 17
raw-object/cache dump entries; no retained array changes value. The newer
D4 implementation separately adds the three-body scale and cutoff behavior.
The one-centre exchange table is preserved byte-for-byte and moved into
`mlxmolkit/xtb/data` so it is included in wheels.

## Additional local work included

- MNDO sp port and independent MOPAC energy/charge gates.
- Float64 batched sp rotations within d-bearing gradient pair evaluation.
- Optional fused Metal sp rotation and density updates.
- GPU convergence-density freezing and singular-DIIS fallback.
- Experimental g-xTB batch coordination with checked eigensolver refinement.
- Stronger dipole, MOPAC failure, and g-xTB missing-table tests.
- Portable performance-harness geometry path and current energy fields.
- Packaged one-centre exchange and D4Srev tanh tables. An isolated wheel
  previously failed to load these despite the source suite passing. The
  onsite loader now raises when its required data is missing instead of
  silently dropping that term from the model.

## Final validation

The reconciled tree was built in a fresh worktree, including both C++
extensions. Full suite including slow tests: **1,095 passed, 24 skipped,
4 expected failures**, with three existing COSMO warnings. See
`validation/main_integration_tests.txt` for all exclusions and warnings.

An isolated interpreter imported only the built wheel's mlxmolkit package
from a temporary directory. On the frozen binary fixture it reproduced
three g-xTB molecules / 50 atomic charges with **5.615e-6 e MAE** and
**2.639e-5 e maximum error** (rounded upward). The check also exercises
one-centre table loading, a D4Srev-enabled calculation and MNDO. Run:

```sh
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python tools/verify_wheel_runtime.py dist/<wheel-name>.whl
```

Results are stored in `validation/wheel_runtime.json`. This small frozen
fixture does not establish accuracy across the full periodic table or
replace the absent external-reference tests.

The new Metal experiments stay opt-in:
`MLXMOLKIT_BATCH_ROTATION_METAL=1` and `MLXMOLKIT_SCF_FUSED_UPDATE=1`.
Experimental g-xTB coordination is available from
`mlxmolkit.xtb.gxtb_batch.gxtb_energy_batch`, with `backend='cpu'` or `'mlx'`.
It is not made the default: previous measurements on this M4 Pro found it
slower than scalar calls for the tested small-molecule workloads. No claim
that GPU batching beats a warm parallel MOPAC library is made by this merge.

## Scientific limits retained explicitly

PM6-family optimization currently differentiates `energy_eV`, while some
geometry-dependent corrections are added only to the heat-of-formation
fields. The corresponding correction gradients remain missing. The prior
acetylene reproduction at a 1.25 Angstrom C-C distance gave 4.594623 eV/A
for the current optimizer's component, versus 0.073900 for the derivative
of corrected heat and 0.072645 from MOPAC. Passing a finite difference test
of the uncorrected energy does not close that gap.

The [MOPAC discussion of individual methods](https://openmopac.net/Discussions/Individual%20Methods%20in%20MOPAC.html)
explains why corrections intended for intermolecular binding cannot simply
be treated as interchangeable heat-of-formation models. PM7 modifies the
model itself; it is not a PM6 parameter alias. Validation should separately
measure single-molecule heats, geometries/forces and intermolecular binding,
including large complexes. The page's PM8 passage is prospective historical
discussion: the inspected current OpenMOPAC source already contains a PM8
parameter module. Its presence alone is not a validated mlxmolkit port.

PM5, MNDO/d, PM7, PM7-TS, PM8 and INDO are not introduced by this merge.
See MOPAC_PORT_STATUS.md for supported coverage and unresolved test exclusions.
