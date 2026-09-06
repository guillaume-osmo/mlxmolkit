"""Default NDDO constants must agree with current MOPAC, in every integral path."""
import importlib
from pathlib import Path
import re

import numpy as np
import pytest
from mlxmolkit.nddo.constants import HARTREE_TO_EV, BOHR_TO_ANG, EV_TO_KCAL


def test_codata_values_and_all_integral_consumers():
    # Independent literal values from OpenMOPAC conref_C fpcref(1,:).
    assert HARTREE_TO_EV == 27.211386245988
    assert BOHR_TO_ANG == .529177210903
    assert EV_TO_KCAL == 23.060547830619029
    for name in ['integrals', 'two_center_integrals', 'pwcct', 'w_integrals',
                 'yy_integrals', 'tetci_yh', 'tetci_multipole_pyseqm']:
        assert importlib.import_module('mlxmolkit.nddo.'+name).EV == HARTREE_TO_EV
    from mlxmolkit.nddo._pyseqm_port import constants_np as c
    assert c.ev == HARTREE_TO_EV and c.a0 == BOHR_TO_ANG
    assert c.ev_kcalpmol == EV_TO_KCAL
    # Prevent reintroduction of a legacy literal into a lazy/imported path.
    root = Path(__file__).parents[1]/'mlxmolkit/nddo'
    for path in root.rglob('*.py'):
        assert not re.search(r'\b(?:27\.21|23\.061|0\.529167|14\.399)\b', path.read_text()), path


@pytest.mark.parametrize('a,b,r,expected', [
    # MOPAC diat_ overlap[0,0], PM6, default CODATA. Reproduce via
    # tools/capture_nddo_codata_references.py; independent of native kernels.
    (1, 1, 0.74, 0.6485699211231968),
    (6, 1, 1.09, 0.4243574513583935),
    (6, 6, 1.54, 0.19183346680166632),
    (16, 1, 1.336, 0.37099677485006843),
    (16, 6, 1.81, 0.15902666860906234),
    (16, 16, 2.05, 0.14294593695156926),
])
def test_mopac_overlap_codata(a,b,r,expected):
    from mlxmolkit.nddo.methods import get_params
    from mlxmolkit.nddo.overlap_d import overlap_d_molecular_frame
    p = get_params('PM6')
    overlap = overlap_d_molecular_frame(p[a],p[b],np.zeros(3),np.array([r,0,0]))
    assert overlap[0,0] == pytest.approx(expected, abs=1e-6)
