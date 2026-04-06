#!/usr/bin/env python3
import csv

VALENCE = {1:1, 6:4, 7:5, 8:6, 9:7, 15:5, 16:6, 17:7, 35:7, 53:7}
EHEAT = {1:52.102, 6:170.89, 7:113.00, 8:59.559, 9:18.86, 15:75.42, 16:66.40, 17:28.99, 35:26.74, 53:25.517}

fn = '/Users/tgg/Github/pyseqm_ref/seqm/params/parameters_PM6_MOPAC.csv'
with open(fn) as f:
    reader = csv.reader(f)
    header = [h.strip() for h in next(reader)]
    for row in reader:
        Z = int(row[0].strip())
        if Z not in VALENCE:
            continue
        v = {}
        for i in range(len(header)):
            v[header[i]] = row[i].strip()
        nb = 1 if Z <= 2 else 4
        nv = VALENCE[Z]
        eh = EHEAT[Z]
        gk = [v.get('Gaussian'+str(i)+'_K', '0') for i in range(1,5)]
        gl = [v.get('Gaussian'+str(i)+'_L', '0') for i in range(1,5)]
        gm = [v.get('Gaussian'+str(i)+'_M', '0') for i in range(1,5)]
        print(f"    {Z}: ElementParams(")
        print(f"        Z={Z}, symbol=\"{v['sym']}\", n_basis={nb}, n_valence={nv}, eheat={eh},")
        print(f"        Uss={v['U_ss']}, Upp={v['U_pp']},")
        print(f"        zeta_s={v['zeta_s']}, zeta_p={v['zeta_p']},")
        print(f"        beta_s={v['beta_s']}, beta_p={v['beta_p']},")
        print(f"        gss={v['g_ss']}, gsp={v['g_sp']}, gpp={v['g_pp']}, gp2={v['g_p2']}, hsp={v['h_sp']},")
        print(f"        alpha={v['alpha']},")
        print(f"        gauss_K=[{', '.join(gk)}],")
        print(f"        gauss_L=[{', '.join(gl)}],")
        print(f"        gauss_M=[{', '.join(gm)}],")
        print(f"    ),")
