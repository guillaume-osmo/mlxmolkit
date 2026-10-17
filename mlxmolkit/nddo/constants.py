"""Physical constants used by current OpenMOPAC (2018 CODATA / 2019 SI).

Source: OpenMOPAC src/conref_C.F90, fpcref(1,:), revision
1d9d92b0283f197616f1e9e76d1ee09e2bc21e72. These are the default constants,
not the OLDFPC values. All native NDDO integral and energy paths share them.
"""

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
EV_TO_KCAL = 23.060547830619029
COULOMB_EV_ANG = 14.399645478456
ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 2.99792458e8
