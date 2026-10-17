# MOPAC port and test review — 2026-09-06

Worktree: `codex/nddo-review-speed`, based on `fix/nddo-mopac-gates`
at `0a508e98f958fcffea1483179e8d69cca8a2c751`. Integrated with the published optimization stack; see BRANCH_INTEGRATION.md.

The subsequent [PM7 port and CODATA migration](PM7_PORT.md) adds native PM7.
The MNDO measurements below are historical, before that constants migration.

## Implemented in this update

**MNDO**, for H, C, N, O, F, Si, P, S, Cl, Br and I. The port uses its
own parameters and the shared native NDDO integral/SCF machinery. Scalar,
CPU batch, MLX/Metal batch, analytical gradient and geometry optimization
entry points accept `method="MNDO"`. This is a closed-shell sp subset,
not every element supported by modern MOPAC's MNDO extensions.

```python
from mlxmolkit.nddo import nddo_energy_batch

results = nddo_energy_batch(
    molecules,  # [(atomic_numbers, coordinates_in_angstrom), ...]
    method="MNDO", use_metal=True,
)
```

The parameter source is [OpenMOPAC's MNDO table at a pinned revision](https://github.com/openmopac/mopac/blob/1d9d92b0283f197616f1e9e76d1ee09e2bc21e72/src/models/parameters_for_mndo_C.F90).
`tools/import_mndo_params.py` reproduces the bundled CSV from that checkout.
The dataset includes its Apache-2.0 license and provenance in `nddo/data`.

Independent MOPAC executable checks on eleven fixed hydride geometries:
maximum heat-of-formation difference **0.020304 kcal/mol**, maximum atomic
charge difference **0.000194 e** (rounded upward). These are small-system
validation results, not accuracy guarantees for arbitrary chemistry.
Raw geometries and measurements: `docs/validation/mndo_oracle_results.json`.
Tests additionally cover CPU/Metal mixed-element batches, ethanol gradients
against central differences, and ethanol optimization convergence.

## PM5 and remaining model ports

| Model | Status |
|---|---|
| PM5 | Blocked: genuine model/parameters unavailable in OpenMOPAC. Explicit error in parameter, scalar and batch APIs, tested for both name cases. |
| MNDO | New eleven-element sp port; wider MNDO element coverage remains unported. |
| AM1, RM1, PM3 | Existing native ports, included in the full regression run. |
| PM6, PM6-ORG, PM6-D3/H4/X variants | Existing native ports; PM6_D is an alias, not another Hamiltonian. |
| MNDO/d | Not ported: requires method-specific d/core parameters and diatomic repulsion, with independent energy/gradient gates. |
| PM7 | Native 40-element closed-shell port, including Sc–Cu, complete corrections/gradient and CPU/Metal batch; see [validation](PM7_PORT.md). |
| PM7-TS | Not ported; requires its own model parameters and validation. |
| PM8 | Present in the inspected current OpenMOPAC source, not ported here. Needs source/version-specific validation. |
| INDO and Sparkles | Not ported; these need their own model/basis semantics and validation. |

The [official historical archive](https://github.com/openmopac/mopac-archive)
identifies PM5 as the proprietary SCIGRESS/MO-G model. OpenMOPAC's
[input reporting code](https://github.com/openmopac/mopac/blob/1d9d92b0283f197616f1e9e76d1ee09e2bc21e72/src/input/wrtkey.F90#L870)
states that PM5 is unsupported and uses the default method instead. Thus
running `mopac` with a PM5 keyword cannot serve as a PM5 oracle. A real PM5
port needs the actual implementation or complete equations and parameters.
This update does **not** claim to implement PM5 or all missing MOPAC models.

## Test review findings and fixes

1. **False GPU SCF convergence from singular DIIS.** H2 produced charges
   (-1,+1) and a converged flag: a singular Pulay solve returned NaNs,
   and the GPU eigensolver produced identity vectors from that input.
   The extrapolation now falls back per molecule to the current Fock matrix
   when the extrapolated matrix is non-finite, with no host synchronization.
   Regression tests cover H2 under MNDO, RM1, AM1, PM3 and PM6.
2. **MOPAC oracle failures hidden as skips.** The existing PM6 heat test
   skipped when an installed executable produced no result. It now fails;
   only absence of an optional executable should skip its oracle tests.
3. **g-xTB missing-file test never ran with the table installed.** It now
   isolates a nonexistent path and cache with pytest monkeypatch, exercises
   the actual loader and checks the restoration guidance. The recovered
   table remains untouched.
4. **Absent-shell numerical integration.** The iodine overlap fallback
   evaluated zero-exponent d shells for sp MNDO, producing warnings/NaNs in
   unused entries. Absent shells are now explicitly zero. The MNDO and
   g-xTB loader tests pass with warnings treated as errors.
5. **Stale public method documentation.** The README advertised PM6_SP and
   described PM6 as sp-only. It now reflects the registry and identifies
   the MNDO subset and unported PM5 explicitly.

The full pre-port suite: **1,063 passed, 25 skipped, 4 expected failures**.
The final full suite: **1,094 passed, 24 skipped, 4 expected failures** in
79.02 seconds, including slow tests. The missing-file test change turns
one previous skip into a passing test. The new MNDO and g-xTB loader tests
also pass with warnings treated as errors (31 tests). The distribution
wheel builds and includes the MNDO table, provenance and license.
Final full-run output and all exclusion reasons are recorded in
`docs/validation/full_suite_final_ports.txt`.

Outstanding exclusions include missing PYSEQM, tblite, dftd4, Python xtb,
AM1-BCC data, and an xtb executable accepting `--gxtb`. Existing expected
failures concern the reconstructed g-xTB heteronuclear radius combination
and three GFN0-xTB contract/parameter expectations. They are not evidence
of validation and were not removed to obtain a green suite.

Existing COSMO overflow warnings and a documented direct-solver fallback
also remain. Existing PM6-family heat corrections still lack their matching
optimization gradients: a passing finite difference test of `energy_eV`
does not validate the corrected heat objective. See the prior
`docs/BRANCH_INTEGRATION.md` for the acetylene/MOPAC reproduction.

No speed improvement over MOPAC is claimed from this port. The existing
g-xTB recovery and optional GPU batching prototype are preserved.
