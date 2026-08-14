# g-xTB Binary Retro Map

Binary: `/tmp/gxtb-v2-macos/lib/libxtb.dylib`

This report is generated from symbol tables and targeted ARM64 disassembly.

## Local Tooling

- `nm`: `/usr/bin/nm`
- `otool`: `/usr/bin/otool`
- `objdump`: `/usr/bin/objdump`
- `dwarfdump`: `/usr/bin/dwarfdump`
- `ghidra`: not found
- `r2`: not found
- `rizin`: not found
- `jtool2`: not found
- `binaryninja`: not found
- `hopper`: not found
- selected assembly slices: `data/gxtb_asm_slices`

## Module Symbols

### `tblite_xtb_gxtb` (17 text symbols)

- `0x00416640` `__copy_tblite_xtb_gxtb_Gxtb_h0spec`
- `0x00416664` `get_diat_scale`
- `0x004169a0` `get_reference_occ`
- `0x00416acc` `get_shpoly4`
- `0x00416c40` `get_shpoly2`
- `0x00416da0` `get_cnshift`
- `0x00416ecc` `get_selfenergy`
- `0x00417000` `get_hscale`
- `0x00417308` `get_rad`
- `0x004174c0` `get_anisotropy`
- `0x00417684` `get_rvdw`
- `0x00417840` `add_repulsion`
- `0x004182c0` `add_exchange`
- `0x00419360` `add_coulomb`
- `0x0041a820` `export_gxtb_param`
- `0x0041cfe0` `new_gxtb_h0spec`
- `0x0041d080` `new_gxtb_calculator`

### `tblite_repulsion_gxtb` (10 text symbols)

- `0x003c7640` `repulsion::__copy_tblite_repulsion_gxtb_Gxtb_repulsion`
- `0x003c790c` `repulsion::__final_tblite_repulsion_gxtb_Gxtb_repulsion`
- `0x003c7b80` `repulsion::get_repulsion_derivs._omp_fn.0`
- `0x003c8a60` `repulsion::get_repulsion_matrix._omp_fn.0`
- `0x003c8f84` `repulsion::get_scaled_zeff.constprop.0.isra.0`
- `0x003c9040` `repulsion::get_gradient`
- `0x003c96b0` `repulsion::get_potential`
- `0x003c97c8` `repulsion::get_energy`
- `0x003c98e0` `repulsion::update`
- `0x003ca28c` `repulsion::new_repulsion_gxtb`

### `tblite_coulomb_multipole_gxtb` (8 text symbols)

- `0x00324aa0` `multipole::get_mrad_derivs`
- `0x00324ae0` `multipole::get_mrad_pair`
- `0x00324b08` `multipole::__copy_tblite_coulomb_multipole_gxtb_Gxtb_multipole`
- `0x00324c90` `multipole::__final_tblite_coulomb_multipole_gxtb_Gxtb_multipole`
- `0x00324e80` `multipole::get_damping_pair`
- `0x00324f60` `multipole::get_damping_derivs`
- `0x00325120` `multipole::update`
- `0x00325748` `multipole::new_gxtb_multipole`

### `multicharge_model_eeqbc` (33 text symbols)

- `0x0046de00` `eeqbc::__copy_multicharge_model_eeqbc_Eeqbc_model`
- `0x0046e100` `eeqbc::__copy_multicharge_model_eeqbc_Eeqbc_cache`
- `0x0046e4f0` `eeqbc::__final_multicharge_model_eeqbc_Eeqbc_model`
- `0x0046e7a8` `eeqbc::__final_multicharge_model_eeqbc_Eeqbc_cache`
- `0x0046ea20` `eeqbc::get_dcmat_3d`
- `0x0046edc0` `eeqbc::get_dcmat_0d`
- `0x0046f0e0` `eeqbc::get_cmat_3d`
- `0x0046f270` `eeqbc::get_cmat_0d`
- `0x0046f3c0` `eeqbc::get_damat_0d._omp_fn.0`
- `0x00471950` `eeqbc::get_amat_0d._omp_fn.0`
- `0x004720e0` `eeqbc::get_xvec_derivs._omp_fn.0`
- `0x00472c08` `eeqbc::get_xvec_derivs._omp_fn.2`
- `0x00473aa0` `eeqbc::get_xvec_derivs`
- `0x00474360` `eeqbc::get_xvec._omp_fn.0`
- `0x00474440` `eeqbc::get_xvec`
- `0x00474640` `eeqbc::update`
- `0x00475a84` `eeqbc::get_coulomb_derivs`
- `0x00476ec0` `eeqbc::get_coulomb_matrix`
- `0x004773c0` `eeqbc::get_damat_dir.isra.0`
- `0x00477788` `eeqbc::get_dcpair.isra.0`
- `0x004779ac` `eeqbc::get_damat_dc_dir.isra.0`
- `0x00477bc0` `eeqbc::get_dcmat_0d._omp_fn.0`
- `0x00478750` `eeqbc::get_dcpair_dir`
- `0x00478944` `eeqbc::get_dcmat_3d._omp_fn.0`
- `0x0047976c` `eeqbc::get_damat_3d._omp_fn.0`
- `0x0047bca4` `eeqbc::get_cmat_0d._omp_fn.0`
- `0x0047c320` `eeqbc::get_cpair_dir`
- `0x0047c4c8` `eeqbc::get_cmat_3d._omp_fn.0`
- `0x0047cda8` `eeqbc::get_xvec_derivs._omp_fn.1`
- `0x0047ddc0` `eeqbc::get_xvec._omp_fn.1`
- `0x0047e1c0` `eeqbc::get_amat_dir_3d.isra.0`
- `0x0047e388` `eeqbc::get_amat_3d._omp_fn.0`
- `0x0047ee00` `eeqbc::new_eeqbc_model`

### `dftd4_model_d4srev` (20 text symbols)

- `0x0045a5e0` `d4srev::get_2b_derivs`
- `0x0045a6c8` `d4srev::get_2b_coeffs`
- `0x0045a720` `d4srev::get_polarizabilities`
- `0x0045aa60` `d4srev::__copy_dftd4_model_d4srev_D4srev_model`
- `0x0045ae88` `d4srev::__final_dftd4_model_d4srev_D4srev_model`
- `0x0045b188` `d4srev::get_3b_rdamp`
- `0x0045b1c0` `d4srev::get_3b_derivs`
- `0x0045b304` `d4srev::get_3b_coeffs`
- `0x0045b360` `d4srev::weight_references._omp_fn.0`
- `0x0045bb80` `d4srev::weight_references._omp_fn.1`
- `0x0045c130` `d4srev::get_atomic_pol._omp_fn.0`
- `0x0045c664` `d4srev::get_atomic_pol._omp_fn.1`
- `0x0045c86c` `d4srev::get_atomic_c6._omp_fn.0`
- `0x0045ce20` `d4srev::get_atomic_c6._omp_fn.1`
- `0x0045d0e0` `d4srev::get_atomic_pol.constprop.0.isra.0`
- `0x0045d9c0` `d4srev::get_atomic_c6.constprop.0.isra.0`
- `0x0045dec4` `d4srev::weight_references.constprop.0.isra.0`
- `0x0045e520` `d4srev::update`
- `0x0045f02c` `d4srev::get_2b_rdamp`
- `0x0045f060` `d4srev::new_d4srev_model`

## g-xTB Parameter Arrays

| address | bytes | as f64 | as i32 | symbol |
|---:|---:|---:|---:|---|
| `0x0073b600` | 3296 | 412 | 824 | `ps_tb2_shell_hubbard` |
| `0x0073c2e0` | 3296 | 412 | 824 | `ps_tb1_zeffsh` |
| `0x0073cfc0` | 3296 | 412 | 824 | `ps_tb1_ipea` |
| `0x0073dca0` | 3296 | 412 | 824 | `ps_reference_occ` |
| `0x0073e980` | 3296 | 412 | 824 | `ps_h0_selfenergy_cn` |
| `0x0073f660` | 3296 | 412 | 824 | `ps_h0_selfenergy` |
| `0x00740340` | 3296 | 412 | 824 | `ps_h0_qvszp_exp_scal` |
| `0x00741020` | 2480 | 310 | 620 | `ps_h0_diat_scale` |
| `0x007419d0` | 3296 | 412 | 824 | `ps_fock_shell_hubbard` |
| `0x007426b0` | 3296 | 412 | 824 | `ps_fock_avg_exp` |
| `0x00743390` | 3296 | 412 | 824 | `ps_acp_level` |
| `0x00744070` | 3296 | 412 | 824 | `ps_acp_exp` |
| `0x00744d50` | 32 | 4 | 8 | `pg_tb4_kshell` |
| `0x00744d70` | 32 | 4 | 8 | `pg_tb3_kshell` |
| `0x00744d90` | 32 | 4 | 8 | `pg_h0_shpoly2` |
| `0x00744db0` | 32 | 4 | 8 | `pg_h0_kshell` |
| `0x00744dd0` | 32 | 4 | 8 | `pg_fock_offdiag_l` |
| `0x00744df0` | 32 | 4 | 8 | `pg_fock_kq` |
| `0x00744e10` | 824 | 103 | 206 | `pa_wll_scale` |
| `0x00745148` | 824 | 103 | 206 | `pa_tb3_hubbard_derivs` |
| `0x00745480` | 824 | 103 | 206 | `pa_tb2_hubbard_cn` |
| `0x007457b8` | 824 | 103 | 206 | `pa_tb1_ipea_cn` |
| `0x00745af0` | 824 | 103 | 206 | `pa_rvdw_scale` |
| `0x00745e28` | 824 | 103 | 206 | `pa_rep_zeff` |
| `0x00746160` | 824 | 103 | 206 | `pa_rep_roffset` |
| `0x00746498` | 824 | 103 | 206 | `pa_rep_q` |
| `0x007467d0` | 824 | 103 | 206 | `pa_rep_k1` |
| `0x00746b08` | 824 | 103 | 206 | `pa_rep_cn` |
| `0x00746e40` | 824 | 103 | 206 | `pa_rep_alpha` |
| `0x00747178` | 412 | 51 | 103 | `pa_nshell` |
| `0x00747314` | 412 | 51 | 103 | `pa_nacp` |
| `0x007474b0` | 1648 | 206 | 412 | `pa_l_acp` |
| `0x00747b20` | 824 | 103 | 206 | `pa_increment` |
| `0x00747e58` | 824 | 103 | 206 | `pa_hubbard_parameter` |
| `0x00748190` | 824 | 103 | 206 | `pa_h0_shpoly2` |
| `0x007484c8` | 824 | 103 | 206 | `pa_h0_qvszp_k3_scal` |
| `0x00748800` | 824 | 103 | 206 | `pa_h0_qvszp_k2_scal` |
| `0x00748b38` | 824 | 103 | 206 | `pa_h0_qvszp_k0_scal` |
| `0x00748e70` | 824 | 103 | 206 | `pa_h0_dip_scale` |
| `0x007491a8` | 824 | 103 | 206 | `pa_fock_cscale` |
| `0x007494e0` | 824 | 103 | 206 | `pa_fock_crad` |
| `0x00749818` | 824 | 103 | 206 | `pa_cn_rcov` |
| `0x00749b50` | 824 | 103 | 206 | `pa_cn_average` |
| `0x00749e88` | ? | ? | ? | `pa_aes_dip_scale` |

## Extracted Parameter Bundle

- NPZ: `data/gxtb_binary_params.npz`
- CSV summary: `data/gxtb_binary_params_summary.csv`
- Interpretation is name/size based: shell tables are `(103, 4)`, atom tables are `(103,)`, `pg_*` tables are `(4,)`, and count/index tables are `int32`.
- First extracted arrays:
  - `ps_tb2_shell_hubbard` `float64` shape `(103, 4)` at `0x0073b600`
  - `ps_tb1_zeffsh` `float64` shape `(103, 4)` at `0x0073c2e0`
  - `ps_tb1_ipea` `float64` shape `(103, 4)` at `0x0073cfc0`
  - `ps_reference_occ` `float64` shape `(103, 4)` at `0x0073dca0`
  - `ps_h0_selfenergy_cn` `float64` shape `(103, 4)` at `0x0073e980`
  - `ps_h0_selfenergy` `float64` shape `(103, 4)` at `0x0073f660`
  - `ps_h0_qvszp_exp_scal` `float64` shape `(103, 4)` at `0x00740340`
  - `ps_h0_diat_scale` `float64` shape `(310,)` at `0x00741020`
  - `ps_fock_shell_hubbard` `float64` shape `(103, 4)` at `0x007419d0`
  - `ps_fock_avg_exp` `float64` shape `(103, 4)` at `0x007426b0`
  - `ps_acp_level` `float64` shape `(103, 4)` at `0x00743390`
  - `ps_acp_exp` `float64` shape `(103, 4)` at `0x00744070`
  - ... 32 more

## Selected Call Neighborhoods

### `new_gxtb_calculator`
- instructions captured: 3848
- calls:
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `x3`
  - `_free`
  - `x3`
  - `_free`
  - ... 272 more
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_new_gxtb_calculator:
000000000041d080	sub	sp, sp, #0x850
000000000041d084	mov	x9, x0
000000000041d088	stp	x29, x30, [sp, #0x30]
000000000041d08c	add	x29, sp, #0x30
000000000041d090	stp	x19, x20, [sp, #0x40]
000000000041d094	mov	x19, x2
000000000041d098	stp	x21, x22, [sp, #0x50]
000000000041d09c	stp	x23, x24, [sp, #0x60]
000000000041d0a0	stp	x25, x26, [sp, #0x70]
000000000041d0a4	stp	x27, x28, [sp, #0x80]
000000000041d0a8	str	x3, [x29, #0xd0]
000000000041d0ac	str	x1, [x29, #0x148]
000000000041d0b0	ldr	x1, [x0]
000000000041d0b4	cbz	x1, 0x41d3c4
000000000041d0b8	ldr	x0, [x0, #0x8]
000000000041d0bc	ldr	x3, [x0, #0x28]
000000000041d0c0	cbz	x3, 0x41d0f8
000000000041d0c4	mov	x2, #0x50000000000
000000000041d0c8	str	x1, [x29, #0x190]
000000000041d0cc	mov	x1, #0x358
000000000041d0d0	str	x9, [x29, #0x168]
000000000041d0d4	stp	x1, x2, [x29, #0x1a0]
000000000041d0d8	mov	w2, #0x1
000000000041d0dc	str	x1, [x29, #0x1b0]
000000000041d0e0	ldr	x1, [x0, #0x8]
000000000041d0e4	add	x0, x29, #0x190
000000000041d0e8	blr	x3
    ...
```

### `new_gxtb_h0spec`
- instructions captured: 40
- calls: none visible in this function
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_new_gxtb_h0spec:
000000000041cfe0	adrp	x3, 814 ; 0x74a000
000000000041cfe4	fmov.2d	v30, #0.50000000
000000000041cfe8	sub	sp, sp, #0x80
000000000041cfec	adrp	x2, 808 ; 0x744000
000000000041cff0	mov	x1, sp
000000000041cff4	ldr	q31, [x3, #0x240]
000000000041cff8	adrp	x3, 814 ; 0x74a000
000000000041cffc	mov	x0, #0x0
000000000041d000	add	x2, x2, #0xdb0
000000000041d004	ldr	q29, [x3, #0x250]
000000000041d008	ldr	d28, [x2, x0, lsl #3]
000000000041d00c	add	x0, x0, #0x1
000000000041d010	dup.2d	v28, v28[0]
000000000041d014	fadd.2d	v27, v28, v31
000000000041d018	fadd.2d	v28, v28, v29
000000000041d01c	fmul.2d	v27, v27, v30
000000000041d020	fmul.2d	v28, v28, v30
000000000041d024	stp	q27, q28, [x1], #0x20
000000000041d028	cmp	x0, #0x4
000000000041d02c	b.ne	0x41d008
000000000041d030	ldr	d26, [sp, #0x58]
000000000041d034	fmov	d25, #1.50000000
000000000041d038	ldr	d0, [sp, #0x70]
000000000041d03c	fmul	d26, d26, d25
000000000041d040	fmul	d0, d0, d25
000000000041d044	ldp	q25, q24, [sp]
000000000041d048	str	d26, [sp, #0x58]
    ...
```

### `add_repulsion`
- instructions captured: 672
- calls:
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_free`
  - `___tblite_utils_average_MOD_new_average`
  - `_malloc`
  - `___mctc_data_vdwrad_MOD_get_vdw_rad_pair_num`
  - `x5`
  - `_malloc`
  - `_free`
  - `_free`
  - `repulsion::new_repulsion_gxtb`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_malloc`
  - `_memcpy`
  - `_malloc`
  - `_free`
  - ... 30 more
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_add_repulsion:
0000000000417840	sub	sp, sp, #0x3b0
0000000000417844	stp	x29, x30, [sp, #0x70]
0000000000417848	add	x29, sp, #0x70
000000000041784c	stp	x27, x28, [sp, #0xc0]
0000000000417850	mov	x27, #0x1
0000000000417854	stp	x19, x20, [sp, #0x80]
0000000000417858	str	x0, [x29, #0x88]
000000000041785c	mov	x0, x1
0000000000417860	stp	x21, x22, [sp, #0x90]
0000000000417864	stp	x23, x24, [sp, #0xa0]
0000000000417868	stp	x25, x26, [sp, #0xb0]
000000000041786c	stp	x1, x2, [x29, #0xb8]
0000000000417870	str	x27, [x29, #0x150]
0000000000417874	ldp	x1, x2, [x1, #0x50]
0000000000417878	stp	x2, x1, [x29, #0xc8]
000000000041787c	ldp	x24, x3, [x0, #0x80]
0000000000417880	sub	x22, x3, x24
0000000000417884	add	x23, x22, x27
0000000000417888	lsl	x20, x23, #3
000000000041788c	cmp	x20, #0x0
0000000000417890	csel	x20, x20, x27, ne
0000000000417894	tbz	x22, #0x3f, 0x417dbc
0000000000417898	mov	x22, #-0x1
000000000041789c	mov	x0, x20
00000000004178a0	str	x23, [x29, #0x158]
00000000004178a4	bl	0x4e2bb8 ; symbol stub for: _malloc
00000000004178a8	str	x0, [x29, #0xa8]
    ...
```

### `add_exchange`
- instructions captured: 1064
- calls:
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `___mctc_data_vdwrad_MOD_get_vdw_rad_pair_num`
  - `x5`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `___tblite_utils_average_MOD_new_average`
  - `___tblite_utils_average_MOD_new_average`
  - `___tblite_utils_average_MOD_new_average`
  - `_malloc`
  - `_memset`
  - `_memcpy`
  - `_malloc`
  - `_memset`
  - `___tblite_data_onecxints_MOD_get_onecxints_number`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_free`
  - `___tblite_exchange_fock_MOD_new_exchange_fock`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - ... 22 more
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_add_exchange:
00000000004182c0	sub	sp, sp, #0x4b0
00000000004182c4	stp	x29, x30, [sp, #0x80]
00000000004182c8	add	x29, sp, #0x80
00000000004182cc	stp	x1, x0, [x29, #0xe0]
00000000004182d0	mov	x0, #0x448
00000000004182d4	stp	xzr, xzr, [x29, #0x170]
00000000004182d8	stp	xzr, xzr, [x29, #0x1b0]
00000000004182dc	bl	0x4e2bb8 ; symbol stub for: _malloc
00000000004182e0	stp	x19, x20, [x29, #0x10]
00000000004182e4	cbz	x0, 0x419240
00000000004182e8	mov	x19, x0
00000000004182ec	stp	x27, x28, [x29, #0x50]
00000000004182f0	stp	xzr, xzr, [x0]
00000000004182f4	mov	x0, #0x8
00000000004182f8	str	x0, [x29, #0x370]
00000000004182fc	mov	x0, #0x30200000000
0000000000418300	str	xzr, [x19, #0x38]
0000000000418304	str	x0, [x29, #0x378]
0000000000418308	ldr	x0, [x29, #0xe0]
000000000041830c	str	xzr, [x19, #0x78]
0000000000418310	str	xzr, [x19, #0xb8]
0000000000418314	str	xzr, [x19, #0xf8]
0000000000418318	str	xzr, [x19, #0x138]
000000000041831c	ldr	w3, [x0, #0x4]
0000000000418320	str	xzr, [x19, #0x1c0]
0000000000418324	str	xzr, [x19, #0x238]
0000000000418328	str	xzr, [x19, #0x2d8]
    ...
```

### `add_coulomb`
- instructions captured: 1328
- calls:
  - `_malloc`
  - `_malloc`
  - `_memcpy`
  - `_malloc`
  - `_free`
  - `_free`
  - `_free`
  - `___tblite_xtb_coulomb_MOD_new_coulomb`
  - `_free`
  - `_free`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_memset`
  - `_memcpy`
  - `_malloc`
  - `_memcpy`
  - `_malloc`
  - `_free`
  - `_malloc`
  - `_memset`
  - `___tblite_coulomb_firstorder_MOD_new_onsite_firstorder`
  - `_malloc`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_memcpy`
  - `_malloc`
  - `_free`
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `___tblite_coulomb_charge_effective_MOD_new_effective_coulomb`
  - `x3`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `_memset`
  - `___tblite_coulomb_thirdorder_twobody_MOD_new_twobody_thirdorder`
  - ... 48 more
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_add_coulomb:
0000000000419360	sub	sp, sp, #0x490
0000000000419364	mov	x3, x0
0000000000419368	stp	x29, x30, [sp, #0x20]
000000000041936c	add	x29, sp, #0x20
0000000000419370	stp	x21, x22, [sp, #0x40]
0000000000419374	mov	x22, #0x30100000000
0000000000419378	stp	x25, x26, [sp, #0x60]
000000000041937c	mov	x25, #0x8
0000000000419380	str	x0, [x29, #0x108]
0000000000419384	mov	x0, #0x30200000000
0000000000419388	str	x1, [x29, #0xf0]
000000000041938c	stp	x25, x22, [x29, #0x160]
0000000000419390	stp	xzr, xzr, [x29, #0x1a0]
0000000000419394	stp	xzr, xzr, [x29, #0x1e0]
0000000000419398	str	x25, [x29, #0x2a0]
000000000041939c	str	x0, [x29, #0x2a8]
00000000004193a0	str	x25, [x29, #0x300]
00000000004193a4	str	x0, [x29, #0x308]
00000000004193a8	str	x25, [x29, #0x360]
00000000004193ac	str	x0, [x29, #0x368]
00000000004193b0	str	x25, [x29, #0x420]
00000000004193b4	str	x0, [x29, #0x428]
00000000004193b8	ldr	x0, [x3, #0x5f8]
00000000004193bc	stp	x19, x20, [x29, #0x10]
00000000004193c0	stp	x23, x24, [x29, #0x30]
00000000004193c4	cbnz	x0, 0x41a6d4
00000000004193c8	mov	x0, #0xa0
    ...
```

### `get_hscale`
- instructions captured: 194
- calls: none visible in this function
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_get_hscale:
0000000000417000	stp	x29, x30, [sp, #-0x50]!
0000000000417004	mov	x29, sp
0000000000417008	stp	x19, x20, [sp, #0x10]
000000000041700c	stp	x21, x22, [sp, #0x20]
0000000000417010	stp	x23, x24, [sp, #0x30]
0000000000417014	ldr	x11, [x3, #0x28]
0000000000417018	str	x25, [sp, #0x40]
000000000041701c	neg	x4, x11
0000000000417020	cbnz	x11, 0x41702c
0000000000417024	mov	x4, #-0x1
0000000000417028	mov	x11, #0x1
000000000041702c	ldr	x8, [x3, #0x40]
0000000000417030	ldp	x21, x5, [x3, #0x70]
0000000000417034	ldr	x23, [x3, #0x80]
0000000000417038	sub	x19, x4, x8
000000000041703c	ldr	x15, [x3, #0x58]
0000000000417040	ldr	x24, [x3]
0000000000417044	subs	x23, x23, x5
0000000000417048	sub	x19, x19, x15
000000000041704c	b.mi	0x417158
0000000000417050	ldp	x4, x16, [x3, #0x30]
0000000000417054	add	x12, x24, x11, lsl #3
0000000000417058	mov	x25, x19
000000000041705c	lsl	x6, x11, #3
0000000000417060	mov	x20, #0x0
0000000000417064	add	x30, x15, x8
0000000000417068	sub	x16, x16, x4
    ...
```

### `get_anisotropy`
- instructions captured: 113
- calls:
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `x4`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___tblite_xtb_gxtb_MOD_get_anisotropy:
00000000004174c0	sub	sp, sp, #0x80
00000000004174c4	stp	x29, x30, [sp, #0x10]
00000000004174c8	add	x29, sp, #0x10
00000000004174cc	stp	x19, x20, [sp, #0x20]
00000000004174d0	mov	x20, x1
00000000004174d4	stp	x21, x22, [sp, #0x30]
00000000004174d8	stp	x23, x24, [sp, #0x40]
00000000004174dc	stp	x25, x26, [sp, #0x50]
00000000004174e0	stp	x27, x28, [sp, #0x60]
00000000004174e4	ldr	x21, [x3, #0x28]
00000000004174e8	neg	x26, x21
00000000004174ec	cbnz	x21, 0x4174f8
00000000004174f0	mov	x26, #-0x1
00000000004174f4	mov	x21, #0x1
00000000004174f8	ldr	x7, [x3, #0x50]
00000000004174fc	ldp	x1, x0, [x3, #0x40]
0000000000417500	ldr	x19, [x3]
0000000000417504	subs	x7, x7, x0
0000000000417508	str	x1, [x29, #0x68]
000000000041750c	b.mi	0x417588
0000000000417510	ldp	x1, x0, [x3, #0x30]
0000000000417514	subs	x0, x0, x1
0000000000417518	b.mi	0x41756c
000000000041751c	add	x8, x19, x21, lsl #3
0000000000417520	mov	x5, x26
0000000000417524	lsl	x3, x21, #3
0000000000417528	mov	x6, #0x0
    ...
```

### `repulsion::get_repulsion_derivs._omp_fn.0`
- instructions captured: 952
- calls:
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_memset`
  - `_malloc`
  - `_memset`
  - `_GOMP_loop_maybe_nonmonotonic_runtime_start`
  - `_pow`
  - `_pow`
  - `_exp`
  - `_exp`
  - `__gfortran_spread`
  - `__gfortran_spread`
  - `_GOMP_loop_maybe_nonmonotonic_runtime_next`
  - `_GOMP_loop_end`
  - `_GOMP_critical_name_start`
  - `_GOMP_critical_name_end`
  - `_free`
  - `_free`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error_at`
  - `__gfortran_runtime_error`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error_at`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_repulsion_derivs._omp_fn.0:
00000000003c7b80	sub	sp, sp, #0x4d0
00000000003c7b84	stp	x29, x30, [sp, #0x10]
00000000003c7b88	add	x29, sp, #0x10
00000000003c7b8c	stp	x19, x20, [sp, #0x20]
00000000003c7b90	stp	x25, x26, [sp, #0x50]
00000000003c7b94	mov	x25, x0
00000000003c7b98	ldr	x0, [x0, #0x220]
00000000003c7b9c	ldr	x20, [x0]
00000000003c7ba0	cbz	x20, 0x3c7bcc
00000000003c7ba4	ldr	x1, [x0, #0x50]
00000000003c7ba8	ldp	x0, x2, [x0, #0x40]
00000000003c7bac	sub	x1, x1, x2
00000000003c7bb0	madd	x0, x1, x0, x0
00000000003c7bb4	lsl	x19, x0, #3
00000000003c7bb8	cmp	x19, #0x0
00000000003c7bbc	csinc	x0, x19, xzr, ne
00000000003c7bc0	bl	0x4e2bb8 ; symbol stub for: _malloc
00000000003c7bc4	mov	x20, x0
00000000003c7bc8	cbz	x0, 0x3c8a2c
00000000003c7bcc	stp	x27, x28, [x29, #0x50]
00000000003c7bd0	ldr	x0, [x25, #0x218]
00000000003c7bd4	ldr	x27, [x0]
00000000003c7bd8	cbz	x27, 0x3c7c04
00000000003c7bdc	ldr	x1, [x0, #0x50]
00000000003c7be0	ldp	x0, x2, [x0, #0x40]
00000000003c7be4	sub	x1, x1, x2
00000000003c7be8	madd	x0, x1, x0, x0
    ...
```

### `repulsion::get_repulsion_matrix._omp_fn.0`
- instructions captured: 329
- calls:
  - `_GOMP_loop_maybe_nonmonotonic_runtime_start`
  - `_pow`
  - `_exp`
  - `_pow`
  - `_exp`
  - `_GOMP_loop_maybe_nonmonotonic_runtime_next`
  - `_GOMP_loop_end_nowait`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_repulsion_matrix._omp_fn.0:
00000000003c8a60	stp	x29, x30, [sp, #-0x1d0]!
00000000003c8a64	mov	x29, sp
00000000003c8a68	add	x4, x29, #0x1c8
00000000003c8a6c	mov	x2, #0x1
00000000003c8a70	stp	x19, x20, [sp, #0x10]
00000000003c8a74	add	x20, x29, #0x1c0
00000000003c8a78	mov	x3, x20
00000000003c8a7c	stp	x21, x22, [sp, #0x20]
00000000003c8a80	stp	x23, x24, [sp, #0x30]
00000000003c8a84	stp	x25, x26, [sp, #0x40]
00000000003c8a88	mov	x26, x0
00000000003c8a8c	stp	x27, x28, [sp, #0x50]
00000000003c8a90	stp	d12, d13, [sp, #0x80]
00000000003c8a94	ldr	x23, [x0, #0x80]
00000000003c8a98	ldp	x24, x19, [x0, #0xa0]
00000000003c8a9c	ldp	x21, x22, [x0, #0x90]
00000000003c8aa0	str	x23, [x29, #0xb8]
00000000003c8aa4	ldr	d13, [x0, #0x140]
00000000003c8aa8	ldp	x0, x15, [x0, #0x70]
00000000003c8aac	str	x0, [x29, #0x108]
00000000003c8ab0	str	x15, [x29, #0x1b8]
00000000003c8ab4	ldp	x0, x1, [x26, #0x60]
00000000003c8ab8	str	x1, [x29, #0x100]
00000000003c8abc	str	x0, [x29, #0x148]
00000000003c8ac0	ldp	x0, x1, [x26, #0x50]
00000000003c8ac4	stp	x0, x1, [x29, #0xf0]
00000000003c8ac8	ldp	x25, x0, [x26, #0x40]
    ...
```

### `repulsion::get_energy`
- instructions captured: 70
- calls:
  - `repulsion::get_scaled_zeff.constprop.0.isra.0`
  - `___tblite_blas_level2_MOD_wrap_dsymv`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_energy:
00000000003c97c8	stp	x29, x30, [sp, #-0x40]!
00000000003c97cc	mov	x29, sp
00000000003c97d0	stp	x19, x20, [sp, #0x10]
00000000003c97d4	stp	x21, x22, [sp, #0x20]
00000000003c97d8	mov	x22, #0x0
00000000003c97dc	ldr	x21, [x4, #0x28]
00000000003c97e0	ldr	x20, [x4]
00000000003c97e4	ldr	x4, [x2, #0x10]
00000000003c97e8	str	x23, [sp, #0x30]
00000000003c97ec	mov	x23, x1
00000000003c97f0	cmp	x21, #0x0
00000000003c97f4	adrp	x1, 2744 ; 0xe81000
00000000003c97f8	add	x1, x1, #0x4f0
00000000003c97fc	csinc	x21, x21, xzr, ne
00000000003c9800	cmp	x4, x1
00000000003c9804	b.ne	0x3c980c
00000000003c9808	ldr	x22, [x2, #0x8]
00000000003c980c	mov	x19, #0x1
00000000003c9810	add	x6, x22, #0x58
00000000003c9814	ldr	x1, [x0]
00000000003c9818	ldr	x0, [x3, #0x2b8]
00000000003c981c	ldr	x2, [x3, #0x2b0]
00000000003c9820	ldr	x5, [x3, #0x270]
00000000003c9824	sub	x0, x19, x0
00000000003c9828	ldr	x4, [x1, #0x100]
00000000003c982c	mul	x0, x0, x2
00000000003c9830	ldr	x2, [x1, #0x80]
    ...
```

### `repulsion::get_gradient`
- instructions captured: 412
- calls:
  - `___tblite_cutoff_MOD_get_lattice_points_cutoff`
  - `repulsion::get_scaled_zeff.constprop.0.isra.0`
  - `_malloc`
  - `_memset`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
  - `_free`
  - `x6`
  - `_free`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_gradient:
00000000003c9040	sub	sp, sp, #0x4e0
00000000003c9044	stp	x29, x30, [sp, #0x10]
00000000003c9048	add	x29, sp, #0x10
00000000003c904c	stp	x19, x20, [sp, #0x20]
00000000003c9050	stp	x21, x22, [sp, #0x30]
00000000003c9054	mov	x21, #0x0
00000000003c9058	stp	x23, x24, [sp, #0x40]
00000000003c905c	stp	x25, x26, [sp, #0x50]
00000000003c9060	mov	x26, x1
00000000003c9064	stp	x27, x28, [sp, #0x60]
00000000003c9068	mov	x28, x3
00000000003c906c	ldr	x1, [x4]
00000000003c9070	ldr	x20, [x4, #0x50]
00000000003c9074	ldr	x19, [x5, #0x50]
00000000003c9078	str	x1, [x29, #0xf8]
00000000003c907c	ldp	x1, x23, [x4, #0x30]
00000000003c9080	str	x1, [x29, #0xd8]
00000000003c9084	ldp	x1, x3, [x4, #0x40]
00000000003c9088	str	x1, [x29, #0x108]
00000000003c908c	ldr	x1, [x5]
00000000003c9090	str	x3, [x29, #0xd0]
00000000003c9094	ldr	x3, [x2, #0x10]
00000000003c9098	str	x1, [x29, #0xf0]
00000000003c909c	ldp	x1, x22, [x5, #0x30]
00000000003c90a0	str	x1, [x29, #0xc8]
00000000003c90a4	mov	x1, #0x8
00000000003c90a8	ldp	x25, x24, [x5, #0x40]
    ...
```

### `repulsion::get_potential`
- instructions captured: 70
- calls:
  - `repulsion::get_scaled_zeff.constprop.0.isra.0`
  - `___tblite_blas_level2_MOD_wrap_dsymv`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_potential:
00000000003c96b0	stp	x29, x30, [sp, #-0x40]!
00000000003c96b4	mov	x29, sp
00000000003c96b8	stp	x19, x20, [sp, #0x10]
00000000003c96bc	stp	x21, x22, [sp, #0x20]
00000000003c96c0	mov	x22, x1
00000000003c96c4	mov	x21, #0x0
00000000003c96c8	adrp	x1, 2744 ; 0xe81000
00000000003c96cc	add	x1, x1, #0x4f0
00000000003c96d0	str	x23, [sp, #0x30]
00000000003c96d4	mov	x23, x4
00000000003c96d8	ldr	x4, [x2, #0x10]
00000000003c96dc	cmp	x4, x1
00000000003c96e0	b.ne	0x3c96e8
00000000003c96e4	ldr	x21, [x2, #0x8]
00000000003c96e8	mov	x19, #0x1
00000000003c96ec	add	x6, x21, #0x58
00000000003c96f0	ldr	x20, [x0]
00000000003c96f4	ldr	x0, [x3, #0x2b8]
00000000003c96f8	ldr	x1, [x3, #0x2b0]
00000000003c96fc	ldr	x5, [x3, #0x270]
00000000003c9700	sub	x0, x19, x0
00000000003c9704	ldr	x2, [x20, #0x80]
00000000003c9708	mul	x0, x0, x1
00000000003c970c	ldr	x1, [x20, #0x58]
00000000003c9710	ldr	x3, [x20, #0xd8]
00000000003c9714	add	x5, x5, x0, lsl #3
00000000003c9718	mov	x0, x22
    ...
```

### `repulsion::get_scaled_zeff.constprop.0.isra.0`
- instructions captured: 47
- calls: none visible in this function
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_get_scaled_zeff.constprop.0.isra.0:
00000000003c8f84	ldp	x9, x8, [x6, #0x28]
00000000003c8f88	ldr	x10, [x6, #0x38]
00000000003c8f8c	cmp	x9, #0x0
00000000003c8f90	csinc	x9, x9, xzr, ne
00000000003c8f94	cmp	x4, #0x0
00000000003c8f98	csinc	x4, x4, xzr, ne
00000000003c8f9c	ldr	x7, [x6]
00000000003c8fa0	cmp	x2, #0x0
00000000003c8fa4	csinc	x2, x2, xzr, ne
00000000003c8fa8	subs	x10, x10, x8
00000000003c8fac	b.mi	0x3c8fd4
00000000003c8fb0	lsl	x11, x9, #3
00000000003c8fb4	mov	x8, x7
00000000003c8fb8	mov	x6, #0x0
00000000003c8fbc	add	x10, x10, #0x1
00000000003c8fc0	add	x6, x6, #0x1
00000000003c8fc4	str	xzr, [x8]
00000000003c8fc8	add	x8, x8, x11
00000000003c8fcc	cmp	x10, x6
00000000003c8fd0	b.ne	0x3c8fc0
00000000003c8fd4	ldr	w11, [x0]
00000000003c8fd8	cmp	w11, #0x0
00000000003c8fdc	b.le	0x3c9038
00000000003c8fe0	ldp	x0, x10, [x0, #0x10]
00000000003c8fe4	fmov	d31, #1.00000000
00000000003c8fe8	lsl	x9, x9, #3
00000000003c8fec	ubfx	x11, x11, #0, #32
    ...
```

### `repulsion::update`
- instructions captured: 619
- calls:
  - `x3`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `___tblite_cutoff_MOD_get_lattice_points_cutoff`
  - `x7`
  - `_memset`
  - `_GOMP_parallel`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_update:
00000000003c98e0	sub	sp, sp, #0x2b0
00000000003c98e4	stp	x29, x30, [sp, #0x10]
00000000003c98e8	add	x29, sp, #0x10
00000000003c98ec	stp	x19, x20, [sp, #0x20]
00000000003c98f0	mov	x20, x1
00000000003c98f4	mov	x1, #0x30200000000
00000000003c98f8	mov	x19, x2
00000000003c98fc	stp	x21, x22, [sp, #0x30]
00000000003c9900	mov	x21, x0
00000000003c9904	mov	x0, #0x8
00000000003c9908	stp	x23, x24, [sp, #0x40]
00000000003c990c	stp	x25, x26, [sp, #0x50]
00000000003c9910	stp	x27, x28, [sp, #0x60]
00000000003c9914	ldr	x26, [x2, #0x8]
00000000003c9918	stp	x0, x1, [x29, #0xe8]
00000000003c991c	cbz	x26, 0x3c9974
00000000003c9920	ldr	x1, [x19, #0x10]
00000000003c9924	adrp	x2, 2744 ; 0xe81000
00000000003c9928	add	x2, x2, #0x4f0
00000000003c992c	cmp	x1, x2
00000000003c9930	b.eq	0x3c9f08
00000000003c9934	cbz	x1, 0x3c9964
00000000003c9938	ldr	x3, [x1, #0x28]
00000000003c993c	cbz	x3, 0x3c9964
00000000003c9940	mov	x2, #0xa0000000000
00000000003c9944	ldr	x1, [x1, #0x8]
00000000003c9948	str	x26, [x29, #0x130]
    ...
```

### `repulsion::new_repulsion_gxtb`
- instructions captured: 733
- calls:
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x4`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `_malloc`
  - `x3`
  - `_malloc`
  - `_malloc`
  - `___tblite_utils_average_MOD_new_average`
  - `x7`
  - `x7`
  - `_realloc`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `___mctc_ncoord_MOD_new_ncoord`
  - `_malloc`
  - `__gfortran_runtime_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error_at`
  - `__gfortran_runtime_error`
- first instructions:
```asm
___tblite_repulsion_gxtb_MOD_new_repulsion_gxtb:
00000000003ca28c	sub	sp, sp, #0x230
00000000003ca290	stp	x29, x30, [sp, #0x10]
00000000003ca294	add	x29, sp, #0x10
00000000003ca298	stp	x27, x28, [sp, #0x60]
00000000003ca29c	mov	x28, x0
00000000003ca2a0	ldr	x0, [x29, #0x228]
00000000003ca2a4	str	x7, [x29, #0xc0]
00000000003ca2a8	ldr	x7, [x29, #0x270]
00000000003ca2ac	stp	x19, x20, [sp, #0x20]
00000000003ca2b0	str	x0, [x29, #0xa0]
00000000003ca2b4	ldr	x0, [x29, #0x230]
00000000003ca2b8	str	x7, [x29, #0xd0]
00000000003ca2bc	ldr	x7, [x29, #0x278]
00000000003ca2c0	stp	x21, x22, [sp, #0x30]
00000000003ca2c4	str	x0, [x29, #0x98]
00000000003ca2c8	ldr	x0, [x29, #0x238]
00000000003ca2cc	str	x7, [x29, #0x150]
00000000003ca2d0	ldr	x7, [x29, #0x280]
00000000003ca2d4	stp	x23, x24, [sp, #0x40]
00000000003ca2d8	str	x0, [x29, #0xf8]
00000000003ca2dc	ldr	x0, [x29, #0x240]
00000000003ca2e0	stp	x25, x26, [sp, #0x50]
00000000003ca2e4	str	x1, [x29, #0x158]
00000000003ca2e8	str	x0, [x29, #0xf0]
00000000003ca2ec	ldr	x0, [x29, #0x248]
00000000003ca2f0	ldr	x9, [x29, #0x220]
00000000003ca2f4	ldr	x1, [x29, #0x268]
    ...
```

### `multipole::get_mrad_pair`
- instructions captured: 10
- calls: none visible in this function
- first instructions:
```asm
___tblite_coulomb_multipole_gxtb_MOD_get_mrad_pair:
0000000000324ae0	ldr	x1, [x0]
0000000000324ae4	ldrsw	x2, [x4]
0000000000324ae8	ldrsw	x0, [x5]
0000000000324aec	ldr	x3, [x1, #0x150]
0000000000324af0	madd	x0, x0, x3, x2
0000000000324af4	ldp	x1, x2, [x1, #0x110]
0000000000324af8	add	x0, x0, x2
0000000000324afc	ldr	d31, [x1, x0, lsl #3]
0000000000324b00	str	d31, [x6]
0000000000324b04	ret
```

### `multipole::get_damping_pair`
- instructions captured: 56
- calls:
  - `_erf`
  - `_erf`
  - `_erf`
  - `_erf`
- first instructions:
```asm
___tblite_coulomb_multipole_gxtb_MOD_get_damping_pair:
0000000000324e80	stp	x29, x30, [sp, #-0x60]!
0000000000324e84	mov	x29, sp
0000000000324e88	ldr	d31, [x1]
0000000000324e8c	stp	x19, x20, [sp, #0x10]
0000000000324e90	mov	x20, x7
0000000000324e94	ldr	d30, [x3]
0000000000324e98	stp	x21, x22, [sp, #0x20]
0000000000324e9c	mov	x22, x5
0000000000324ea0	mov	x21, x6
0000000000324ea4	ldr	x19, [x0]
0000000000324ea8	str	x23, [sp, #0x30]
0000000000324eac	mov	x23, x4
0000000000324eb0	str	d15, [sp, #0x38]
0000000000324eb4	stp	d11, d12, [sp, #0x40]
0000000000324eb8	fsub	d12, d31, d30
0000000000324ebc	stp	d13, d14, [sp, #0x50]
0000000000324ec0	ldp	d0, d11, [x19, #0x30]
0000000000324ec4	fmul	d0, d12, d0
0000000000324ec8	bl	0x4e2af8 ; symbol stub for: _erf
0000000000324ecc	ldp	d29, d15, [x19, #0x10]
0000000000324ed0	fmov	d13, #0.50000000
0000000000324ed4	fmov	d14, #1.00000000
0000000000324ed8	fadd	d0, d0, d14
0000000000324edc	fmul	d29, d29, d13
0000000000324ee0	fmul	d29, d29, d0
0000000000324ee4	fmul	d0, d12, d11
0000000000324ee8	str	d29, [x23]
    ...
```

### `multipole::get_damping_derivs`
- instructions captured: 112
- calls:
  - `_erf`
  - `_erf`
  - `_erf`
  - `_erf`
  - `_exp`
  - `_exp`
  - `_exp`
  - `_exp`
- first instructions:
```asm
___tblite_coulomb_multipole_gxtb_MOD_get_damping_derivs:
0000000000324f60	stp	x29, x30, [sp, #-0xa0]!
0000000000324f64	mov	x29, sp
0000000000324f68	ldr	d31, [x1]
0000000000324f6c	stp	x19, x20, [sp, #0x10]
0000000000324f70	mov	x20, x7
0000000000324f74	ldr	d30, [x3]
0000000000324f78	stp	x21, x22, [sp, #0x20]
0000000000324f7c	mov	x22, x5
0000000000324f80	mov	x21, x6
0000000000324f84	stp	d8, d9, [sp, #0x40]
0000000000324f88	stp	d10, d11, [sp, #0x50]
0000000000324f8c	stp	d12, d13, [sp, #0x60]
0000000000324f90	fsub	d9, d31, d30
0000000000324f94	stp	d14, d15, [sp, #0x70]
0000000000324f98	ldr	x19, [x0]
0000000000324f9c	str	x23, [sp, #0x30]
0000000000324fa0	mov	x23, x4
0000000000324fa4	fmul	d12, d9, d9
0000000000324fa8	ldp	d11, d13, [x19, #0x30]
0000000000324fac	fmul	d0, d11, d9
0000000000324fb0	bl	0x4e2af8 ; symbol stub for: _erf
0000000000324fb4	ldp	d30, d14, [x19, #0x10]
0000000000324fb8	fmov	d15, #1.00000000
0000000000324fbc	fmov	d10, #0.50000000
0000000000324fc0	fadd	d0, d0, d15
0000000000324fc4	fmul	d31, d30, d10
0000000000324fc8	str	d30, [x29, #0x80]
    ...
```

### `multipole::update`
- instructions captured: 394
- calls:
  - `x3`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `x8`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `__gfortran_runtime_error`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___tblite_coulomb_multipole_gxtb_MOD_update:
0000000000325120	sub	sp, sp, #0xa0
0000000000325124	stp	x29, x30, [sp, #0x10]
0000000000325128	add	x29, sp, #0x10
000000000032512c	stp	x19, x20, [sp, #0x20]
0000000000325130	mov	x20, x2
0000000000325134	stp	x21, x22, [sp, #0x30]
0000000000325138	mov	x22, x0
000000000032513c	mov	x21, x1
0000000000325140	ldr	x19, [x2, #0x8]
0000000000325144	str	x23, [sp, #0x40]
0000000000325148	cbz	x19, 0x3251a4
000000000032514c	ldr	x0, [x2, #0x10]
0000000000325150	adrp	x1, 2906 ; 0xe7f000
0000000000325154	add	x1, x1, #0x920
0000000000325158	cmp	x0, x1
000000000032515c	b.eq	0x325318
0000000000325160	cbz	x0, 0x325194
0000000000325164	ldr	x3, [x0, #0x28]
0000000000325168	cbz	x3, 0x325194
000000000032516c	mov	x1, #0x8
0000000000325170	mov	x2, #0xa0000000000
0000000000325174	str	x19, [x29, #0x68]
0000000000325178	stp	x1, x2, [x29, #0x78]
000000000032517c	mov	w2, #0x0
0000000000325180	str	x1, [x29, #0x88]
0000000000325184	ldr	x1, [x0, #0x8]
0000000000325188	add	x0, x29, #0x68
    ...
```

### `multipole::new_gxtb_multipole`
- instructions captured: 190
- calls:
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `_malloc`
- first instructions:
```asm
___tblite_coulomb_multipole_gxtb_MOD_new_gxtb_multipole:
0000000000325748	stp	x29, x30, [sp, #-0x90]!
000000000032574c	mov	x29, sp
0000000000325750	stp	x19, x20, [sp, #0x10]
0000000000325754	stp	x21, x22, [sp, #0x20]
0000000000325758	stp	x23, x24, [sp, #0x30]
000000000032575c	mov	x24, x5
0000000000325760	stp	x25, x26, [sp, #0x40]
0000000000325764	mov	x25, x4
0000000000325768	stp	x27, x28, [sp, #0x50]
000000000032576c	mov	x28, x0
0000000000325770	ldr	x20, [x2, #0x28]
0000000000325774	stp	x7, x6, [x29, #0x80]
0000000000325778	cbz	x20, 0x325a20
000000000032577c	neg	x0, x20
0000000000325780	str	x0, [x29, #0x68]
0000000000325784	ldr	x0, [x2]
0000000000325788	ldr	x7, [x2, #0x50]
000000000032578c	ldr	x1, [x3]
0000000000325790	str	x0, [x29, #0x70]
0000000000325794	ldp	x0, x27, [x2, #0x30]
0000000000325798	ldr	x6, [x3, #0x38]
000000000032579c	sub	x27, x27, x0
00000000003257a0	ldp	x23, x0, [x2, #0x40]
00000000003257a4	add	x27, x27, #0x1
00000000003257a8	sub	x22, x7, x0
00000000003257ac	ldp	x19, x0, [x3, #0x28]
00000000003257b0	add	x21, x22, #0x1
    ...
```

### `eeqbc::new_eeqbc_model`
- instructions captured: 616
- calls:
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x5`
  - `_free`
  - `x5`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_malloc`
  - `_free`
  - `_free`
  - `___mctc_ncoord_MOD_new_ncoord`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `___mctc_ncoord_MOD_new_ncoord`
- first instructions:
```asm
___multicharge_model_eeqbc_MOD_new_eeqbc_model:
000000000047ee00	sub	sp, sp, #0x260
000000000047ee04	stp	x29, x30, [sp, #0x10]
000000000047ee08	add	x29, sp, #0x10
000000000047ee0c	stp	x19, x20, [sp, #0x20]
000000000047ee10	stp	x21, x22, [sp, #0x30]
000000000047ee14	stp	x23, x24, [sp, #0x40]
000000000047ee18	stp	x25, x26, [sp, #0x50]
000000000047ee1c	mov	x26, x0
000000000047ee20	stp	x27, x28, [sp, #0x60]
000000000047ee24	str	x1, [x29, #0x120]
000000000047ee28	str	x2, [x29, #0x140]
000000000047ee2c	ldp	x8, x11, [x3, #0x28]
000000000047ee30	ldr	x24, [x4, #0x38]
000000000047ee34	cmp	x8, #0x0
000000000047ee38	csinc	x8, x8, xzr, ne
000000000047ee3c	str	x8, [x29, #0xf0]
000000000047ee40	ldr	x8, [x3]
000000000047ee44	ldr	x3, [x3, #0x38]
000000000047ee48	ldr	x2, [x29, #0x258]
000000000047ee4c	str	x8, [x29, #0xe8]
000000000047ee50	ldr	x1, [x29, #0x260]
000000000047ee54	sub	x3, x3, x11
000000000047ee58	add	x27, x3, #0x1
000000000047ee5c	ldp	x3, x11, [x4, #0x28]
000000000047ee60	ldr	x0, [x29, #0x268]
000000000047ee64	cmp	x3, #0x0
000000000047ee68	csinc	x3, x3, xzr, ne
    ...
```

### `eeqbc::get_xvec`
- instructions captured: 128
- calls:
  - `_GOMP_parallel`
  - `___multicharge_blas_MOD_mchrg_dgemv`
  - `_free`
  - `___multicharge_model_type_MOD_get_dir_trans`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
- first instructions:
```asm
___multicharge_model_eeqbc_MOD_get_xvec:
0000000000474440	stp	x29, x30, [sp, #-0x170]!
0000000000474444	mov	x29, sp
0000000000474448	stp	x19, x20, [sp, #0x10]
000000000047444c	mov	x20, x0
0000000000474450	stp	x21, x22, [sp, #0x20]
0000000000474454	stp	x23, x24, [sp, #0x30]
0000000000474458	stp	x25, x26, [sp, #0x40]
000000000047445c	mov	x25, x1
0000000000474460	stp	x27, x28, [sp, #0x50]
0000000000474464	ldr	x27, [x3, #0x28]
0000000000474468	neg	x22, x27
000000000047446c	cbnz	x27, 0x474478
0000000000474470	mov	x22, #-0x1
0000000000474474	mov	x27, #0x1
0000000000474478	ldp	x0, x19, [x3, #0x30]
000000000047447c	mov	x1, #0x30200000000
0000000000474480	str	xzr, [x29, #0x118]
0000000000474484	ldr	x21, [x3]
0000000000474488	sub	x19, x19, x0
000000000047448c	mov	x0, #0x8
0000000000474490	stp	xzr, xzr, [x29, #0x78]
0000000000474494	add	x19, x19, #0x1
0000000000474498	stp	x0, x1, [x29, #0x128]
000000000047449c	ldr	x1, [x2, #0x8]
00000000004744a0	str	x0, [x29, #0x90]
00000000004744a4	mov	x0, #0x30100000000
00000000004744a8	str	x0, [x29, #0x98]
    ...
```

### `eeqbc::get_xvec_derivs`
- instructions captured: 560
- calls:
  - `_malloc`
  - `_malloc`
  - `_memset`
  - `_memset`
  - `_memset`
  - `_memset`
  - `_GOMP_parallel`
  - `___multicharge_blas_MOD_mchrg_dgemm323`
  - `___multicharge_blas_MOD_mchrg_dgemm323`
  - `_free`
  - `___multicharge_model_type_MOD_get_dir_trans`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_GOMP_parallel`
  - `__gfortran_os_error_at`
  - `__gfortran_runtime_error`
- first instructions:
```asm
___multicharge_model_eeqbc_MOD_get_xvec_derivs:
0000000000473aa0	sub	sp, sp, #0x4e0
0000000000473aa4	stp	x29, x30, [sp, #0x10]
0000000000473aa8	add	x29, sp, #0x10
0000000000473aac	add	x11, x29, #0x260
0000000000473ab0	add	x12, x29, #0x340
0000000000473ab4	stp	x19, x20, [sp, #0x20]
0000000000473ab8	stp	x21, x22, [sp, #0x30]
0000000000473abc	stp	x23, x24, [sp, #0x40]
0000000000473ac0	stp	x25, x26, [sp, #0x50]
0000000000473ac4	stp	x27, x28, [sp, #0x60]
0000000000473ac8	str	x0, [x29, #0x90]
0000000000473acc	ldr	x0, [x3]
0000000000473ad0	str	x1, [x29, #0xf8]
0000000000473ad4	str	x0, [x29, #0xe8]
0000000000473ad8	ldp	x27, x0, [x3, #0x50]
0000000000473adc	ldp	x28, x24, [x3, #0x30]
0000000000473ae0	ldp	x19, x22, [x3, #0x40]
0000000000473ae4	str	x0, [x29, #0x108]
0000000000473ae8	ldr	x0, [x4]
0000000000473aec	ldp	x23, x26, [x3, #0x60]
0000000000473af0	mov	x3, #0x30200000000
0000000000473af4	str	x0, [x29, #0xe0]
0000000000473af8	ldp	x20, x0, [x4, #0x40]
0000000000473afc	ldp	x25, x21, [x4, #0x30]
0000000000473b00	str	x0, [x29, #0xc0]
0000000000473b04	ldp	x0, x1, [x4, #0x50]
0000000000473b08	str	x0, [x29, #0xf0]
    ...
```

### `eeqbc::get_coulomb_matrix`
- instructions captured: 320
- calls:
  - `___multicharge_model_type_MOD_get_dir_trans`
  - `_GOMP_parallel`
  - `_free`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
- first instructions:
```asm
___multicharge_model_eeqbc_MOD_get_coulomb_matrix:
0000000000476ec0	sub	sp, sp, #0x230
0000000000476ec4	stp	x29, x30, [sp]
0000000000476ec8	mov	x29, sp
0000000000476ecc	stp	x19, x20, [sp, #0x10]
0000000000476ed0	stp	x21, x22, [sp, #0x20]
0000000000476ed4	stp	x23, x24, [sp, #0x30]
0000000000476ed8	mov	x24, x0
0000000000476edc	mov	x23, x1
0000000000476ee0	ldr	x19, [x3, #0x28]
0000000000476ee4	neg	x7, x19
0000000000476ee8	cbnz	x19, 0x476ef4
0000000000476eec	mov	x7, #-0x1
0000000000476ef0	mov	x19, #0x1
0000000000476ef4	ldp	x0, x4, [x3, #0x30]
0000000000476ef8	mov	x1, #0x0
0000000000476efc	ldr	x8, [x3, #0x50]
0000000000476f00	sub	x4, x4, x0
0000000000476f04	ldp	x20, x0, [x3, #0x40]
0000000000476f08	add	x4, x4, #0x1
0000000000476f0c	ldr	x10, [x3]
0000000000476f10	sub	x8, x8, x0
0000000000476f14	adrp	x0, 2734 ; 0xf24000
0000000000476f18	ldr	x3, [x2, #0x8]
0000000000476f1c	add	x0, x0, #0x4d0
0000000000476f20	add	x9, x8, #0x1
0000000000476f24	cmp	x3, x0
0000000000476f28	b.ne	0x476f30
    ...
```

### `eeqbc::get_coulomb_derivs`
- instructions captured: 1295
- calls:
  - `___multicharge_model_type_MOD_get_dir_trans`
  - `_malloc`
  - `_malloc`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_GOMP_parallel`
  - `_free`
  - `_free`
  - `_free`
  - `__gfortran_runtime_error_at`
  - `__gfortran_os_error_at`
  - `__gfortran_os_error_at`
- first instructions:
```asm
___multicharge_model_eeqbc_MOD_get_coulomb_derivs:
0000000000475a84	sub	sp, sp, #0x750
0000000000475a88	mov	x8, x1
0000000000475a8c	stp	x29, x30, [sp, #0x10]
0000000000475a90	add	x29, sp, #0x10
0000000000475a94	stp	x19, x20, [sp, #0x20]
0000000000475a98	stp	x21, x22, [sp, #0x30]
0000000000475a9c	stp	x23, x24, [sp, #0x40]
0000000000475aa0	stp	x25, x26, [sp, #0x50]
0000000000475aa4	stp	x27, x28, [sp, #0x60]
0000000000475aa8	ldr	x12, [x3, #0x28]
0000000000475aac	str	x0, [x29, #0x298]
0000000000475ab0	mov	x0, x2
0000000000475ab4	cbz	x12, 0x476544
0000000000475ab8	neg	x1, x12
0000000000475abc	str	x1, [x29, #0x290]
0000000000475ac0	ldr	x1, [x3]
0000000000475ac4	ldr	x11, [x4, #0x28]
0000000000475ac8	str	x1, [x29, #0x2a8]
0000000000475acc	ldp	x2, x1, [x3, #0x30]
0000000000475ad0	sub	x1, x1, x2
0000000000475ad4	neg	x2, x11
0000000000475ad8	add	x1, x1, #0x1
0000000000475adc	str	x1, [x29, #0x2a0]
0000000000475ae0	cbnz	x11, 0x475aec
0000000000475ae4	mov	x2, #-0x1
0000000000475ae8	mov	x11, #0x1
0000000000475aec	ldr	x1, [x4]
    ...
```

### `d4srev::new_d4srev_model`
- instructions captured: 1800
- calls:
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `x3`
  - `_free`
  - `_free`
  - `_free`
  - `_free`
  - `__gfortran_string_trim`
  - `_malloc`
  - `__gfortran_concat_string`
  - `_free`
  - `_malloc`
  - `__gfortran_concat_string`
  - `_free`
  - `_free`
  - `_free`
  - `___mctc_env_error_MOD_fatal_error`
  - `_malloc`
  - `_malloc`
  - ... 74 more
- first instructions:
```asm
___dftd4_model_d4srev_MOD_new_d4srev_model:
000000000045f060	sub	sp, sp, #0x2a0
000000000045f064	stp	x29, x30, [sp, #0x10]
000000000045f068	add	x29, sp, #0x10
000000000045f06c	stp	x19, x20, [sp, #0x20]
000000000045f070	stp	x21, x22, [sp, #0x30]
000000000045f074	mov	x21, x2
000000000045f078	stp	x25, x26, [sp, #0x50]
000000000045f07c	mov	x25, x0
000000000045f080	stp	x27, x28, [sp, #0x60]
000000000045f084	mov	x27, x1
000000000045f088	ldr	x0, [x1, #0x18]
000000000045f08c	str	x3, [x29, #0x80]
000000000045f090	cbz	x0, 0x45f09c
000000000045f094	bl	0x4e2b58 ; symbol stub for: _free
000000000045f098	str	xzr, [x27, #0x18]
000000000045f09c	ldr	x0, [x27, #0x58]
000000000045f0a0	cbz	x0, 0x45f0ac
000000000045f0a4	bl	0x4e2b58 ; symbol stub for: _free
000000000045f0a8	str	xzr, [x27, #0x58]
000000000045f0ac	ldr	x0, [x27, #0x98]
000000000045f0b0	cbz	x0, 0x45f0bc
000000000045f0b4	bl	0x4e2b58 ; symbol stub for: _free
000000000045f0b8	str	xzr, [x27, #0x98]
000000000045f0bc	ldr	x0, [x27, #0xd8]
000000000045f0c0	cbz	x0, 0x45f0cc
000000000045f0c4	bl	0x4e2b58 ; symbol stub for: _free
000000000045f0c8	str	xzr, [x27, #0xd8]
    ...
```

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
// The pair builder is now mapped as:
//   pair_rvdw = mctc_data_vdwrad_pair(ZA,ZB) * geometric(pa_rvdw_scale)
//   c1_pair   = arithmetic(pa_rep_k1)
//   c2_pair   = arithmetic(Z < 3 ? 0.012098131381864387
//                                : 0.008544252691968662)
// g-xTB main construction also builds mctc_ncoord type 3, i.e. ERF CN:
//   cn_ij = 0.5 * (1 + erf(-2.068 * (R - (rcov_i+rcov_j))/(rcov_i+rcov_j)))
// using pa_cn_rcov as custom radii.
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
    Matrix<double> pair_rvdw;            // exact packed MCTC VdW pair radii, 103*104/2

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
    // Recovered 103-element EEQ_BC 2025 tables:
    //   rvdw_scale, rad, kqchi, kcnchi, eta, cov_radii, chi, cap, avg_cn.
    // Constructor creates two mctc_ncoord objects:
    //   ID 3 = erf_ncoord and ID 4 = erf_en_ncoord.
    // The xvec/coulomb kernels are still being reduced from the OpenMP bodies.
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
