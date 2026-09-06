# PM7 native port and CODATA migration — 2026-09-06

PM7 is available through the native scalar SCF, CPU batch, MLX/Metal batch,
gradient and geometry optimization APIs. Coverage is closed-shell,
nonperiodic H, C, N, O, F, Si, P, S, Cl, Br and I. The latter six elements
retain PM7's nine-orbital bases. Unsupported elements fail explicitly.
Transition metals, PM7-TS, PM7-HH and periodic PM7 are outside this port.

```python
from mlxmolkit.nddo import nddo_energy_batch, nddo_optimize_batch

energies = nddo_energy_batch(molecules, method="PM7", use_metal=True)
optimized = nddo_optimize_batch(molecules, method="PM7")
```

## Model implementation

The port uses its own one-center and diatomic parameters, extracted from
[OpenMOPAC at a pinned revision](https://github.com/openmopac/mopac/blob/1d9d92b0283f197616f1e9e76d1ee09e2bc21e72/src/models/parameters_for_PM7_C.F90).
Run `python tools/import_pm7_params.py /path/to/openmopac` to reproduce the
CSV files. The package includes provenance and the Apache-2.0 license.
Explicit zero PM7 parameters do not fall back to PM6 values.

The calculation includes:

- PM7 pair core repulsion, including C–C, O–H, Si–O and Gaussian terms.
- The smooth transition to point-charge electron–electron, electron–nuclear
  and core interactions, with MOPAC's default 7 Å and 0.22 Å⁻² parameters.
- PM7 Slater–Kirkwood dispersion, EH+ hydrogen bonding, the short O–H–O
  contribution, C≡C, amide torsion and Si–O–H corrections.
- Separate MOPAC bond definitions for dispersion/Si–O–H and EH+ geometry.

For PM7, `energy_eV` includes the geometry corrections. `scf_energy_eV`
reports `electronic_eV + nuclear_eV`, and `geometry_correction_eV` reports
the difference. Heat of formation includes the same correction exactly once.
The optimization gradient differentiates this complete energy: frozen-density
SCF derivatives plus central differences of the geometry-only corrections,
without displaced SCF solves for those corrections. These correction
calculations currently run on the CPU.

Other methods retain their existing energy-field conventions. In particular,
the pre-existing PM6-family omission of heat-only corrections from its
optimization gradient remains a separate issue.

## Current physical constants

Every native NDDO integral path now uses the default current OpenMOPAC
[CODATA constants](https://github.com/openmopac/mopac/blob/1d9d92b0283f197616f1e9e76d1ee09e2bc21e72/src/conref_C.F90):
27.211386245988 eV/Hartree, 0.529177210903 Å/Bohr and
23.060547830619029 kcal/mol/eV. `nddo/constants.py` supplies the scalar,
vectorized and vendored integral routines, energy conversions and NEB.
GPU paths consume integrals calculated with the same constants. NDDO's
D3 conversion constants were unified as well. g-xTB is unchanged.

This intentionally changes numerical results previously computed with
MOPAC7-era constants. The old frozen PYSEQM SCF references were replaced
with independent default-MOPAC electronic and nuclear energy references,
retaining the existing 0.001 eV bounds. Algorithm-only integral regression
snapshots were refreshed and labeled as snapshots rather than independent
accuracy evidence. New constants tests also pin MOPAC overlap values across
six shell combinations and reject reintroduced legacy literals.

Reproduce the independent constants evidence with:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. python tools/capture_nddo_codata_references.py
```

That developer diagnostic uses private gfortran symbols in libmopac;
these symbols are never required by the package runtime or unit tests.

## Validation

On 24 fixed geometries (11 hydrides and 13 organic molecules), maximum
absolute heat-of-formation disagreement with default MOPAC is
**0.000935 kcal/mol**, and maximum atomic-charge disagreement is
**0.0000697 e**. These are measured results for this set, not universal
accuracy guarantees. The oracle uses `PM7 1SCF SCFCRT=1.D-9`, never `OLDFPC`.
Raw coordinates and both sets of results are in
[pm7_oracle_results.json](validation/pm7_oracle_results.json).

```sh
OMP_NUM_THREADS=1 PYTHONPATH=. python tools/validate_pm7_port.py
OMP_NUM_THREADS=1 python -m pytest tests/test_pm7_port.py tests/test_nddo_constants.py -q
```

Tests separately compare printed MOPAC dispersion and hydrogen-bond terms
on water, ammonia, formic-acid and formamide dimers at four separations.
Other gates cover all-element CPU/Metal batches, corrected gradients on
five systems, method isolation, rotation/permutation, the long-range
point-charge limit, case normalization, unsupported elements, and the
optional fused Metal rotation/SCF update. The shared optimizer test includes
PM7 ethanol convergence. Installed-wheel checks exercise PM7 parameter loading
alongside the preserved g-xTB and MNDO checks.

Full suite: **1,160 passed, 24 skipped, 4 expected failures**, with three
existing COSMO warnings, in 72.41 seconds. Full output and exclusion reasons:
[pm7_full_suite.txt](validation/pm7_full_suite.txt). The isolated installed-wheel
check passed for PM7, MNDO and g-xTB, with unchanged g-xTB frozen charge errors:
[pm7_wheel_runtime.json](validation/pm7_wheel_runtime.json).

The port establishes numerical correctness and batch compatibility. It does
not establish a speed advantage over MOPAC; especially the CPU correction
gradients remain a throughput optimization target.
