import numpy as np

from mlxmolkit.xtb.params_gxtb import (
    GXTB_PARAMS,
    GXTB_REPULSION_LITERALS,
    GXTB_REPULSION_LITERAL_SEQUENCE,
    SHELL_LABELS,
)


def test_gxtb_parameter_shapes_and_global_tables():
    p = GXTB_PARAMS
    assert p["ps_reference_occ"].shape == (103, 4)
    assert p["pa_rep_zeff"].shape == (103,)
    assert p["pa_nshell"].shape == (103,)
    assert p["pg_tb4_kshell"].shape == (4,)
    np.testing.assert_allclose(p["pg_tb4_kshell"], [1.0, 1.15, 1.3, 1.45])
    np.testing.assert_allclose(p["pg_h0_shpoly2"], [1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(p["pg_fock_kq"], [1.1, 0.55, 0.275, 0.1375])


def test_gxtb_repulsion_scalar_literals_from_add_repulsion():
    """The constants the recovered repulsion routine uses, by name.

    The two exponential powers and their scale and weight are what
    ``damp = exp(-(rr**p1 * zz)) + exp(-(rr**p2 * zz * s)) * w`` needs; the
    cubic and quartic coefficients multiply the polynomial in ``rad/r``.
    ``erf_cn_steepness`` shares the block but belongs to the coordination
    number, which is why it is not one of the repulsion powers.
    """

    lit = GXTB_REPULSION_LITERALS

    assert lit["exp_power_1"] == 1.5
    assert lit["exp_power_2"] == 2.0
    assert lit["exp2_scale"] == 0.73
    assert lit["exp2_weight"] == 0.0046511298
    assert lit["cubic_coeff"] == 0.011095539524126988
    assert lit["quartic_coeff"] == 0.011607795128002491
    assert lit["erf_cn_steepness"] == 2.068

    # The sequence is the block's own order, which the constants builder and
    # the C++ extension both index positionally.
    assert GXTB_REPULSION_LITERAL_SEQUENCE == tuple(lit.values())
    assert len(GXTB_REPULSION_LITERAL_SEQUENCE) == 9


def test_gxtb_h_c_o_s_element_views():
    h = GXTB_PARAMS.element(1)
    c = GXTB_PARAMS.element(6)
    o = GXTB_PARAMS.element(8)
    s = GXTB_PARAMS.element(16)

    assert h.n_shell == 1
    assert c.n_shell == 2
    assert o.n_shell == 2
    assert s.n_shell == 3
    assert tuple(shell.label for shell in s.shells) == SHELL_LABELS[:3]

    np.testing.assert_allclose(h.reference_occ, [1.0])
    np.testing.assert_allclose(c.reference_occ, [1.03539398945965, 2.96460601049032])
    np.testing.assert_allclose(o.reference_occ, [1.67303036949422, 4.32696963045594])
    np.testing.assert_allclose(
        s.reference_occ,
        [1.75759024471622, 4.08981250079158, 0.15259725444982],
    )

    assert np.isclose(o.rep_zeff, 2.7937777512)
    assert np.isclose(s.aes_dip_scale, 0.1724210314)
