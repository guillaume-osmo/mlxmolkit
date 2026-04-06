#!/usr/bin/env python3
"""Extract PM3 params from PYSEQM CSV."""
import csv

VALENCE = {1:1, 6:4, 7:5, 8:6, 9:7, 15:5, 16:6, 17:7, 35:7, 53:7}
EHEAT = {1:52.102, 6:170.89, 7:113.00, 8:59.559, 9:18.86, 15:75.42, 16:66.40, 17:28.99, 35:26.74, 53:25.517}

fn = '/Users/tgg/Github/pyseqm_ref/seqm/params/parameters_PM3_MOPAC.csv'
with open(fn) as f:
    reader = csv.reader(f)
    header = [h.strip() for h in next(reader)]
    for row in reader:
        Z = int(row[0].strip())
        if Z not in VALENCE:
            continue
        v = {header[i]: row[i].strip() for i in range(len(header))}
        nb = 1 if Z <= 2 else 4
        nv = VALENCE[Z]
        eh = EHEAT[Z]
        print(f"    {Z}: ElementParams(")
        print(f"        Z={Z}, symbol=\"{v['sym']}\", n_basis={nb}, n_valence={nv}, eheat={eh},")
        print(f"        Uss={v['U_ss']}, Upp={v['U_pp']},")
        print(f"        zeta_s={v['zeta_s']}, zeta_p={v['zeta_p']},")
        print(f"        beta_s={v['beta_s']}, beta_p={v['beta_p']},")
        print(f"        gss={v['g_ss']}, gsp={v['g_sp']}, gpp={v['g_pp']}, gp2={v['g_p2']}, hsp={v['h_sp']},")
        print(f"        alpha={v['alpha']},")
        print(f"        gauss_K=[{v['Gaussian1_K']}, {v['Gaussian2_K']}, 0.0, 0.0],")
        print(f"        gauss_L=[{v['Gaussian1_L']}, {v['Gaussian2_L']}, 0.0, 0.0],")
        print(f"        gauss_M=[{v['Gaussian1_M']}, {v['Gaussian2_M']}, 0.0, 0.0],")
        print(f"    ),")
