#!/usr/bin/env python3
"""Build a clean-room pseudo-C++ map from the g-xTB release binary.

This script does not decompile source. It uses public symbol names plus
targeted ARM64 disassembly from the release binary to produce a compact
reverse-engineering report:

* g-xTB/tblite symbol map
* private parameter-array sizes inferred from adjacent const symbols
* small call neighborhoods for selected functions
* hand-written pseudo-C++ skeletons anchored to those symbols

Run from the repo root, for example:

    /Users/guillaume-osmo/miniconda3/envs/osmo/bin/python3 \
        tools/gxtb_disasm_pseudocpp.py \
        --lib /tmp/gxtb-v2-macos/lib/libxtb.dylib \
        --out gxtb_retro_pseudocpp.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYMBOL_PATTERNS = (
    "tblite_xtb_gxtb",
    "tblite_repulsion_gxtb",
    "tblite_coulomb_multipole_gxtb",
    "multicharge_model_eeqbc",
    "dftd4_model_d4srev",
)

SELECTED_FUNCTIONS = [
    "___tblite_xtb_gxtb_MOD_new_gxtb_calculator",
    "___tblite_xtb_gxtb_MOD_new_gxtb_h0spec",
    "___tblite_xtb_gxtb_MOD_add_repulsion",
    "___tblite_xtb_gxtb_MOD_add_exchange",
    "___tblite_xtb_gxtb_MOD_add_coulomb",
    "___tblite_xtb_gxtb_MOD_get_hscale",
    "___tblite_xtb_gxtb_MOD_get_anisotropy",
    "___tblite_repulsion_gxtb_MOD_get_repulsion_derivs._omp_fn.0",
    "___tblite_repulsion_gxtb_MOD_get_repulsion_matrix._omp_fn.0",
    "___tblite_repulsion_gxtb_MOD_get_energy",
    "___tblite_repulsion_gxtb_MOD_get_gradient",
    "___tblite_repulsion_gxtb_MOD_get_potential",
    "___tblite_repulsion_gxtb_MOD_get_scaled_zeff.constprop.0.isra.0",
    "___tblite_repulsion_gxtb_MOD_update",
    "___tblite_repulsion_gxtb_MOD_new_repulsion_gxtb",
    "___tblite_coulomb_multipole_gxtb_MOD_get_mrad_pair",
    "___tblite_coulomb_multipole_gxtb_MOD_get_damping_pair",
    "___tblite_coulomb_multipole_gxtb_MOD_get_damping_derivs",
    "___tblite_coulomb_multipole_gxtb_MOD_update",
    "___tblite_coulomb_multipole_gxtb_MOD_new_gxtb_multipole",
    "___multicharge_model_eeqbc_MOD_new_eeqbc_model",
    "___multicharge_model_eeqbc_MOD_get_xvec",
    "___multicharge_model_eeqbc_MOD_get_xvec_derivs",
    "___multicharge_model_eeqbc_MOD_get_coulomb_matrix",
    "___multicharge_model_eeqbc_MOD_get_coulomb_derivs",
    "___dftd4_model_d4srev_MOD_new_d4srev_model",
]

TOOL_CANDIDATES = (
    "nm",
    "otool",
    "objdump",
    "dwarfdump",
    "ghidra",
    "r2",
    "rizin",
    "jtool2",
    "binaryninja",
    "hopper",
)


@dataclass(frozen=True)
class Symbol:
    addr: int
    section: str
    linkage: str
    name: str


@dataclass(frozen=True)
class Section:
    section: str
    segment: str
    addr: int
    size: int
    offset: int


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def which(tool: str) -> str | None:
    from shutil import which as _which

    return _which(tool)


def detect_tools() -> dict[str, str | None]:
    return {tool: which(tool) for tool in TOOL_CANDIDATES}


def load_symbols(lib: Path) -> list[Symbol]:
    out = run(["nm", "-nm", str(lib)])
    symbols: list[Symbol] = []
    pat = re.compile(
        r"^([0-9a-fA-F]+)\s+\(([^)]+)\)\s+((?:non-)?external(?: \(was a private external\))?)\s+(\S+)$"
    )
    for line in out.splitlines():
        m = pat.match(line)
        if not m:
            continue
        symbols.append(
            Symbol(
                addr=int(m.group(1), 16),
                section=m.group(2),
                linkage=m.group(3),
                name=m.group(4),
            )
        )
    return symbols


def load_sections(lib: Path) -> list[Section]:
    out = run(["otool", "-l", str(lib)])
    sections: list[Section] = []
    current: dict[str, Any] | None = None
    for raw in out.splitlines():
        line = raw.strip()
        if line.startswith("sectname "):
            if current and {"section", "segment", "addr", "size", "offset"} <= current.keys():
                sections.append(Section(**current))
            current = {"section": line.split(None, 1)[1]}
        elif current is not None and line.startswith("segname "):
            current["segment"] = line.split(None, 1)[1]
        elif current is not None and line.startswith("addr "):
            current["addr"] = int(line.split()[1], 16)
        elif current is not None and line.startswith("size "):
            current["size"] = int(line.split()[1], 16)
        elif current is not None and line.startswith("offset "):
            current["offset"] = int(line.split()[1], 10)
        elif current is not None and line.startswith("flags "):
            if {"section", "segment", "addr", "size", "offset"} <= current.keys():
                sections.append(Section(**current))
            current = None
    return sections


def addr_to_file_offset(sections: list[Section], addr: int, nbytes: int) -> int:
    for sec in sections:
        if sec.addr <= addr and addr + nbytes <= sec.addr + sec.size:
            return sec.offset + (addr - sec.addr)
    raise ValueError(f"address 0x{addr:x}+{nbytes} is not inside a known section")


def relevant_symbols(symbols: list[Symbol]) -> list[Symbol]:
    return [
        sym
        for sym in symbols
        if any(pattern in sym.name for pattern in DEFAULT_SYMBOL_PATTERNS)
    ]


def gxtb_const_arrays(symbols: list[Symbol]) -> list[tuple[Symbol, int | None]]:
    consts = [
        sym
        for sym in symbols
        if sym.section == "__TEXT,__const"
        and sym.name.startswith("___tblite_xtb_gxtb_MOD_")
        and (
            "_MOD_pa_" in sym.name
            or "_MOD_ps_" in sym.name
            or "_MOD_pg_" in sym.name
        )
    ]
    out: list[tuple[Symbol, int | None]] = []
    for i, sym in enumerate(consts):
        next_addr = consts[i + 1].addr if i + 1 < len(consts) else None
        out.append((sym, None if next_addr is None else next_addr - sym.addr))
    return out


def infer_gxtb_param_spec(short: str, inferred_size: int | None) -> tuple[str, tuple[int, ...], int]:
    """Infer dtype/shape/nbytes for g-xTB parameter arrays.

    The symbol table exposes enough repeated dimensions to make this reliable for
    the named arrays below:
    * ``ps_*`` shell tables are mostly 103 elements x 4 shells, float64.
    * ``pa_*`` atom tables are 103 elements, float64, except integer count/index
      tables.
    * ``pg_*`` tables are tiny float64 lookup vectors.
    """

    if short.startswith("ps_h0_diat_scale"):
        return "float64", (310,), 310 * 8
    if short.startswith("ps_"):
        return "float64", (103, 4), 412 * 8
    if short.startswith("pg_"):
        return "float64", (4,), 4 * 8
    if short in {"pa_nshell", "pa_nacp"}:
        return "int32", (103,), 103 * 4
    if short == "pa_l_acp":
        return "int32", (103, 4), 412 * 4
    if short.startswith("pa_"):
        return "float64", (103,), 103 * 8
    if inferred_size is None:
        raise ValueError(f"cannot infer parameter array size for {short}")
    return "raw", (inferred_size,), inferred_size


def dump_gxtb_params(
    lib: Path,
    symbols: list[Symbol],
    arrays: list[tuple[Symbol, int | None]],
    npz_path: Path,
    csv_path: Path | None,
) -> dict[str, Any]:
    import numpy as np

    sections = load_sections(lib)
    blob = lib.read_bytes()
    payload: dict[str, Any] = {}
    meta: list[dict[str, Any]] = []

    for sym, inferred_size in arrays:
        short = short_name(sym.name)
        dtype_name, shape, nbytes = infer_gxtb_param_spec(short, inferred_size)
        file_offset = addr_to_file_offset(sections, sym.addr, nbytes)
        raw = blob[file_offset : file_offset + nbytes]
        if dtype_name == "float64":
            array = np.frombuffer(raw, dtype="<f8").copy().reshape(shape)
        elif dtype_name == "int32":
            array = np.frombuffer(raw, dtype="<i4").copy().reshape(shape)
        else:
            array = np.frombuffer(raw, dtype="u1").copy()
        payload[short] = array
        meta.append(
            {
                "name": short,
                "symbol": sym.name,
                "address": f"0x{sym.addr:08x}",
                "file_offset": file_offset,
                "dtype": dtype_name,
                "shape": shape,
                "bytes_dumped": nbytes,
                "bytes_to_next_gxtb_symbol": inferred_size,
            }
        )

    payload["__meta_json__"] = np.array(json.dumps(meta, indent=2))
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "address",
                    "file_offset",
                    "dtype",
                    "shape",
                    "bytes_dumped",
                    "bytes_to_next_gxtb_symbol",
                    "preview",
                ],
            )
            writer.writeheader()
            for item in meta:
                arr = payload[item["name"]].reshape(-1)
                preview_vals = arr[: min(12, arr.size)].tolist()
                writer.writerow(
                    {
                        **{k: item[k] for k in writer.fieldnames if k in item},
                        "shape": "x".join(map(str, item["shape"])),
                        "preview": json.dumps(preview_vals),
                    }
                )

    return {"npz": str(npz_path), "csv": str(csv_path) if csv_path else None, "meta": meta}


def disassemble(lib: Path) -> dict[str, list[str]]:
    text = run(["otool", "-tvV", str(lib)])
    funcs: dict[str, list[str]] = {}
    current: str | None = None
    label = re.compile(r"^(_{2,}\S+):$")
    for line in text.splitlines():
        m = label.match(line.strip())
        if m:
            current = m.group(1)
            funcs[current] = [line]
            continue
        if current is not None:
            funcs[current].append(line)
    return funcs


def objdump_disassemble_symbol(lib: Path, symbol: str) -> list[str]:
    """Return objdump's single-symbol Mach-O disassembly when available."""

    if which("objdump") is None:
        return []
    try:
        out = run(
            [
                "objdump",
                "--macho",
                "--disassemble",
                "--dis-symname",
                symbol,
                str(lib),
            ]
        )
    except subprocess.CalledProcessError:
        return []
    return out.splitlines()


def dump_selected_asm(lib: Path, funcs: dict[str, list[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = ["backend\tfile\tsymbol\tinstructions"]
    for name in SELECTED_FUNCTIONS:
        filename = short_name(name).replace("::", "_").replace("/", "_")
        lines = funcs.get(name)
        if lines:
            path = out_dir / f"{filename}.otool.s"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest.append(f"otool\t{path.name}\t{name}\t{max(0, len(lines) - 1)}")

        objdump_lines = objdump_disassemble_symbol(lib, name)
        if objdump_lines:
            path = out_dir / f"{filename}.objdump.s"
            path.write_text("\n".join(objdump_lines) + "\n", encoding="utf-8")
            instruction_count = sum(1 for line in objdump_lines if re.match(r"^\s*[0-9a-fA-F]+:", line))
            manifest.append(f"objdump\t{path.name}\t{name}\t{instruction_count}")
    (out_dir / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def call_targets(lines: list[str]) -> list[str]:
    targets: list[str] = []
    for line in lines:
        if "\tbl\t" not in line and "\tblr\t" not in line:
            continue
        if "; symbol stub for:" in line:
            target = line.split("; symbol stub for:", 1)[1].strip()
        else:
            target = line.rsplit("\t", 1)[-1].strip()
        targets.append(target)
    return targets


def short_name(name: str) -> str:
    return name.replace("___tblite_xtb_gxtb_MOD_", "").replace(
        "___tblite_repulsion_gxtb_MOD_", "repulsion::"
    ).replace("___tblite_coulomb_multipole_gxtb_MOD_", "multipole::").replace(
        "___multicharge_model_eeqbc_MOD_", "eeqbc::"
    ).replace(
        "___dftd4_model_d4srev_MOD_", "d4srev::"
    )


PSEUDOCODE = r"""
## Pseudo-C++ Skeletons

These are not recovered Fortran. They are clean-room skeletons anchored to
symbol names, call neighborhoods, output behavior, and parameter-array layout.
Names with `?` are inferred fields, not confirmed source identifiers.

### Assembly-Derived Microkernels

```cpp
// Anchor: tblite_repulsion_gxtb::get_scaled_zeff.constprop.0.isra.0
// Observed loop shape:
//   zero output vector
//   for each atom i:
//       Zi = atomic_number[i] - 1
//       out[i] = zeff[Zi] * (1.0 - scale[Zi] * descriptor[i])
// The call sites decide whether `descriptor` is shell charge, CN-like data,
// or a precombined repulsion descriptor.
void get_scaled_zeff(
    span<double> out,
    span<const int> atomic_number,
    span<const double> zeff_by_Z,
    span<const double> scale_by_Z,
    span<const double> descriptor
) {
    fill(out.begin(), out.end(), 0.0);
    for (int i = 0; i != atomic_number.size(); ++i) {
        const int z = atomic_number[i] - 1;
        out[i] = zeff_by_Z[z] * (1.0 - scale_by_Z[z] * descriptor[i]);
    }
}
```

```cpp
// Anchor: tblite_repulsion_gxtb::update
// Observed in two repeated loops with literals 1e-12 and 1e-6:
//   value = base[Z] * (1 + slope[Z] * (sqrt(cn + 1e-12) - 1e-6))
//   deriv = base[Z] * slope[Z] / (2 * sqrt(cn + 1e-12))
void cn_scaled_parameter(span<double> value, span<double> deriv,
                         span<const int> Z, span<const double> base,
                         span<const double> slope, span<const double> cn) {
    for (int i = 0; i != Z.size(); ++i) {
        double root = sqrt(cn[i] + 1.0e-12);
        int z = Z[i] - 1;
        value[i] = base[z] * (1.0 + slope[z] * (root - 1.0e-6));
        deriv[i] = base[z] * slope[z] / (2.0 * root);
    }
}
```

```cpp
// Anchors: tblite_repulsion_gxtb::{get_repulsion_matrix._omp_fn.0,
//          get_repulsion_derivs._omp_fn.0}
// Scalar literals passed by add_repulsion into new_repulsion_gxtb, decoded from
// __TEXT,__const:
//   0x73b268 1.5
//   0x73b270 2.068
//   0x73b278 2.0
//   0x73b280 0.73
//   0x73b288 0.0046511298
//   0x73b290 0.011607795128002491
//   0x73b298 0.011095539524126988
//   0x73b2a0 0.012098131381864387
//   0x73b2a8 0.008544252691968662
// The precise parameter-to-field map is still being wired, but the inner
// algebraic shape is visible from get_repulsion_matrix._omp_fn.0: alpha mixing,
// one linear 1/R term, higher polynomial terms in (rvdw_pair/R), and two
// exponentials in (R + offset).
double repulsion_pair_value(double R, double alphaA, double alphaB,
                            double rvdw_pair, double roffset,
                            double c1_pair, double c2_pair,
                            double c3_global, double c4_global,
                            double p1, double p2,
                            double exp2scale, double exp2weight) {
    double invR = 1.0 / R;
    double x = rvdw_pair * invR;
    double poly = 1.0 + c1_pair*invR
                      + c2_pair*x*x
                      + c3_global*x*x*x
                      + c4_global*x*x*x*x;
    double alpha = alphaA * alphaB / (alphaA + alphaB);
    double rho = R + roffset;
    return poly * (exp(-alpha * pow(rho, p1))
                 + exp2weight * exp(-alpha * exp2scale * pow(rho, p2)));
}

void repulsion_matrix_energy_gradient(...) {
    for (int A = 0; A != nat; ++A) {
        for (int B = A + 1; B != nat; ++B) {
            double R = norm(xyz[A] - xyz[B]);
            double pair = repulsion_pair_value(R, alpha[A], alpha[B],
                                               rvdw_pair[A][B], offset[A][B], ...);
            matrix[A][B] = matrix[B][A] = pair;
            matvec[A] += pair * scaled_zeff[B];
            matvec[B] += pair * scaled_zeff[A];
            energy += 2.0 * scaled_zeff[A] * scaled_zeff[B] * pair;
            // gradient uses the analytic d(pair)/dR from the same scalar kernel.
        }
    }
}
```

```cpp
// Anchor: tblite_coulomb_multipole_gxtb::get_damping_pair
// The function loads two scalar positions/radii, takes delta = a - b, and
// writes four damped coefficients using paired amplitudes and erf slopes.
void get_damping_pair(const GxtbMultipole& mp, double a, double b,
                      double& d00, double& d01, double& d10, double& d11) {
    const double delta = a - b;
    d00 = 0.5 * mp.amp00 * (1.0 + erf(delta * mp.beta00));
    d01 = 0.5 * mp.amp01 * (1.0 + erf(delta * mp.beta01));
    d10 = 0.5 * mp.amp10 * (1.0 + erf(delta * mp.beta10));
    d11 = 0.5 * mp.amp11 * (1.0 + erf(delta * mp.beta11));
}

double get_mrad_pair(const Matrix& table, int i, int j) {
    return table(i, j);
}
```

```cpp
// g-xTB top-level calculator constructor.
// Anchor: tblite_xtb_gxtb::new_gxtb_calculator
void new_gxtb_calculator(XtbCalculator& calc, const Structure& mol, Error& err) {
    calc.release_owned_components();
    calc.method = "gxtb";

    // Basis setup uses EEQ_BC-like charges for AO expansion/contraction.
    auto eeqbc_basis_charge = eeqbc::get_charges_for_basis(mol);
    calc.basis = make_xtb_basis(mol, eeqbc_basis_charge, gxtb_param);

    calc.h0 = new_gxtb_h0spec(mol, calc.basis, gxtb_param);

    add_repulsion(calc, mol, calc.basis, gxtb_param);
    add_exchange(calc, mol, calc.basis, gxtb_param);   // INDO-like one-center exchange
    add_coulomb(calc, mol, calc.basis, gxtb_param);    // shell-charge SCC terms

    calc.multipole = multipole::new_gxtb_multipole(mol, calc.basis, gxtb_param);
    calc.dispersion = d4srev::new_d4srev_model(mol, /* reference */ "gxtb");
    calc.acp = make_p_acp_terms_for_H_to_F(mol, calc.basis, gxtb_param);

    // CP2K PR confirms this exists in save_tblite: a native DIIS/potential mixer.
    calc.iterator = make_native_potential_scf_iterator(calc.variable_info());
}
```

```cpp
// H0 specification. The binary exposes many getters, each pulling from
// parameter arrays shaped mostly as [Z=1..103][shell=0..3].
// Anchors: new_gxtb_h0spec, get_selfenergy, get_cnshift, get_hscale,
//          get_anisotropy, get_reference_occ, get_diat_scale.
struct GxtbH0Spec {
    double reference_occ[103][4];
    double selfenergy[103][4];
    double selfenergy_cn[103][4];
    double qvszp_exp_scale[103][4];
    double diat_scale[103][/* sparse shell pairs */];
    double h0_dip_scale[103];
    double h0_qvszp_k0_scale[103];
    double h0_qvszp_k2_scale[103];
    double h0_qvszp_k3_scale[103];

    void get_hscale(const Basis& bas, Tensor3& out) const {
        out.fill(0.0);
        for (int A = 0; A != bas.natoms; ++A) {
            for (int B = 0; B != bas.natoms; ++B) {
                for (Shell sA : bas.shells_on_atom(A)) {
                    for (Shell sB : bas.shells_on_atom(B)) {
                        // Disassembly shows nested atom/shell loops and stores
                        // values from a const parameter table into a strided tensor.
                        out(sA.index, sB.index, /*spin/channel?*/0) =
                            hscale_lookup(sA.angular, sB.angular, bas.Z[A], bas.Z[B]);
                    }
                }
            }
        }
    }

    void add_anisotropic_H0_terms(/* H0, shell charges, multipoles */) const {
        // Binary exposes additional anisotropy getters beyond GFN2.
        // Treat this as shell-charge dependent dipole/quadrupole H0 correction.
    }
};
```

```cpp
// g-xTB repulsion component.
// Anchors: tblite_xtb_gxtb::add_repulsion,
//          tblite_repulsion_gxtb::{new_repulsion_gxtb,get_energy,get_gradient}.
struct GxtbRepulsion {
    double cutoff_bohr = 25.0;           // literal fmov #25.0 observed in add_repulsion
    double zeff[103], alpha[103], k1[103], q[103], cn[103], roffset[103];
    double rvdw_scale[103], cn_average[103], cn_rcov[103];
    Matrix<double> pair_rvdw;            // add_repulsion calls mctc VdW pair radii

    double scaled_zeff(int Z, double q_shell_or_atom, double cn) const;

    double energy(const Structure& mol, const ShellCharges& q) const {
        double E = 0.0;
        for (Pair AB : neighbor_pairs(mol, cutoff_bohr)) {
            double R = norm(mol.xyz[A] - mol.xyz[B]);
            auto pA = atom_params(A, q, mol.cn[A]);
            auto pB = atom_params(B, q, mol.cn[B]);
            // Exact algebra is not source-recovered; symbols show a flexible
            // charge/CN-dependent pair repulsion rather than GFN2's fixed form.
            E += repulsion_pair(R, pA, pB, pair_rvdw(A, B));
        }
        return E;
    }

    void gradient(const Structure& mol, const ShellCharges& q, Vec3* grad) const {
        for (Pair AB : neighbor_pairs(mol, cutoff_bohr)) {
            Vec3 dE_dR = repulsion_pair_derivative(AB, q);
            grad[A] += dE_dR;
            grad[B] -= dE_dR;
        }
    }
};
```

```cpp
// EEQ_BC model used for basis setup and D4Srev references.
// Anchors: multicharge_model_eeqbc::{new_eeqbc_model,get_xvec,get_coulomb_matrix,...}
struct EEQBCModel {
    // Parameter symbols exist for chi, eta, cov_radii, rad, kqchi, kcnchi,
    // cap, average CN, and rvdw_scale.
    Vector xvec(const Structure& mol) const;          // electronegativity RHS
    Matrix coulomb_matrix(const Structure& mol) const;
    Matrix constraint_matrix(const Structure& mol) const;

    Vector solve_charges(const Structure& mol, int total_charge) const {
        // [A C; C^T 0] [q; lambda] = [-x; Q]
        return constrained_charge_solve(coulomb_matrix(mol), xvec(mol), total_charge);
    }
};
```

```cpp
// g-xTB multipole Coulomb damping.
// Anchors: tblite_coulomb_multipole_gxtb::{new_gxtb_multipole,
//          get_mrad_pair,get_damping_pair,get_damping_derivs}.
struct GxtbMultipole {
    double mrad_pair(int ZA, int ZB, int shellA, int shellB) const;
    double damping_pair(double R, int ZA, int ZB, const ShellPair& sh) const;
    Deriv damping_derivs(double R, int ZA, int ZB, const ShellPair& sh) const;
};
```
"""


def build_report(
    lib: Path,
    dump_info: dict[str, Any] | None = None,
    asm_dir: Path | None = None,
) -> str:
    symbols = load_symbols(lib)
    rel = relevant_symbols(symbols)
    funcs = disassemble(lib)

    lines: list[str] = []
    lines.append("# g-xTB Binary Retro Map\n")
    lines.append(f"Binary: `{lib}`\n")
    lines.append("This report is generated from symbol tables and targeted ARM64 disassembly.\n")

    lines.append("## Local Tooling\n")
    for tool, path in detect_tools().items():
        lines.append(f"- `{tool}`: `{path}`" if path else f"- `{tool}`: not found")
    if asm_dir is not None:
        lines.append(f"- selected assembly slices: `{asm_dir}`")
    lines.append("")

    lines.append("## Module Symbols\n")
    for pattern in DEFAULT_SYMBOL_PATTERNS:
        chunk = [sym for sym in rel if pattern in sym.name and sym.section == "__TEXT,__text"]
        lines.append(f"### `{pattern}` ({len(chunk)} text symbols)\n")
        for sym in chunk:
            lines.append(f"- `0x{sym.addr:08x}` `{short_name(sym.name)}`")
        lines.append("")

    lines.append("## g-xTB Parameter Arrays\n")
    lines.append("| address | bytes | as f64 | as i32 | symbol |")
    lines.append("|---:|---:|---:|---:|---|")
    for sym, size in gxtb_const_arrays(symbols):
        name = short_name(sym.name)
        if size is None:
            lines.append(f"| `0x{sym.addr:08x}` | ? | ? | ? | `{name}` |")
        else:
            lines.append(
                f"| `0x{sym.addr:08x}` | {size} | {size // 8} | {size // 4} | `{name}` |"
            )
    lines.append("")

    if dump_info is not None:
        lines.append("## Extracted Parameter Bundle\n")
        lines.append(f"- NPZ: `{dump_info['npz']}`")
        if dump_info.get("csv"):
            lines.append(f"- CSV summary: `{dump_info['csv']}`")
        lines.append(
            "- Interpretation is name/size based: shell tables are `(103, 4)`, "
            "atom tables are `(103,)`, `pg_*` tables are `(4,)`, and count/index "
            "tables are `int32`."
        )
        lines.append("- First extracted arrays:")
        for item in dump_info["meta"][:12]:
            lines.append(
                f"  - `{item['name']}` `{item['dtype']}` shape `{tuple(item['shape'])}` "
                f"at `{item['address']}`"
            )
        if len(dump_info["meta"]) > 12:
            lines.append(f"  - ... {len(dump_info['meta']) - 12} more")
        lines.append("")

    lines.append("## Selected Call Neighborhoods\n")
    for name in SELECTED_FUNCTIONS:
        f_lines = funcs.get(name)
        if not f_lines:
            lines.append(f"### `{short_name(name)}`\n\n_not found_\n")
            continue
        calls = call_targets(f_lines)
        lines.append(f"### `{short_name(name)}`")
        lines.append(f"- instructions captured: {max(0, len(f_lines) - 1)}")
        if calls:
            lines.append("- calls:")
            for target in calls[:40]:
                lines.append(f"  - `{short_name(target)}`")
            if len(calls) > 40:
                lines.append(f"  - ... {len(calls) - 40} more")
        else:
            lines.append("- calls: none visible in this function")
        lines.append("- first instructions:")
        lines.append("```asm")
        lines.extend(f_lines[:28])
        if len(f_lines) > 28:
            lines.append("    ...")
        lines.append("```\n")

    lines.append(PSEUDOCODE.strip())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=Path("/tmp/gxtb-v2-macos/lib/libxtb.dylib"))
    parser.add_argument("--out", type=Path, default=Path("gxtb_retro_pseudocpp.md"))
    parser.add_argument(
        "--dump-npz",
        type=Path,
        default=None,
        help="Optionally dump interpreted g-xTB const arrays to a compressed NumPy bundle.",
    )
    parser.add_argument(
        "--dump-csv",
        type=Path,
        default=None,
        help="Optional CSV summary for --dump-npz.",
    )
    parser.add_argument(
        "--dump-asm-dir",
        type=Path,
        default=None,
        help="Optional directory for raw selected-function assembly slices.",
    )
    args = parser.parse_args()

    dump_info = None
    symbols = load_symbols(args.lib)
    arrays = gxtb_const_arrays(symbols)
    if args.dump_npz is not None:
        dump_info = dump_gxtb_params(args.lib, symbols, arrays, args.dump_npz, args.dump_csv)

    if args.dump_asm_dir is not None:
        dump_selected_asm(args.lib, disassemble(args.lib), args.dump_asm_dir)

    report = build_report(args.lib, dump_info=dump_info, asm_dir=args.dump_asm_dir)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report.splitlines())} lines)")
    if dump_info is not None:
        print(f"wrote {dump_info['npz']}")
        if dump_info.get("csv"):
            print(f"wrote {dump_info['csv']}")


if __name__ == "__main__":
    main()
