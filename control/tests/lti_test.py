"""lti_test.py"""

import numpy as np
import pytest

from control import c2d, tf, tf2ss, NonlinearIOSystem
from control.lti import (LTI, damp, dcgain, isctime, isdtime,
                         issiso, pole, timebaseEqual, zero)
from control.tests.conftest import slycotonly


class TestLTI:

    def test_pole(self):
        sys = tf(126, [-1, 42])
        np.testing.assert_equal(sys.pole(), 42)
        np.testing.assert_equal(pole(sys), 42)

    def test_zero(self):
        sys = tf([-1, 42], [1, 10])
        np.testing.assert_equal(sys.zero(), 42)
        np.testing.assert_equal(zero(sys), 42)

    def test_issiso(self):
        assert issiso(1)
        with pytest.raises(ValueError):
            issiso(1, strict=True)

        # SISO transfer function
        sys = tf([-1, 42], [1, 10])
        assert issiso(sys)
        assert issiso(sys, strict=True)

        # SISO state space system
        sys = tf2ss(sys)
        assert issiso(sys)
        assert issiso(sys, strict=True)

    @slycotonly
    def test_issiso_mimo(self):
        # MIMO transfer function
        sys = tf([[[-1, 41], [1]], [[1, 2], [3, 4]]],
                 [[[1, 10], [1, 20]], [[1, 30], [1, 40]]]);
        assert not issiso(sys)
        assert not issiso(sys, strict=True)

        # MIMO state space system
        sys = tf2ss(sys)
        assert not issiso(sys)
        assert not issiso(sys, strict=True)

    def test_damp(self):
        # Test the continuous time case.
        zeta = 0.1
        wn = 42
        p = -wn * zeta + 1j * wn * np.sqrt(1 - zeta**2)
        sys = tf(1, [1, 2 * zeta * wn, wn**2])
        expected = ([wn, wn], [zeta, zeta], [p, p.conjugate()])
        np.testing.assert_equal(sys.damp(), expected)
        np.testing.assert_equal(damp(sys), expected)

        # Also test the discrete time case.
        dt = 0.001
        sys_dt = c2d(sys, dt, method='matched')
        p_zplane = np.exp(p*dt)
        expected_dt = ([wn, wn], [zeta, zeta],
                       [p_zplane, p_zplane.conjugate()])
        np.testing.assert_almost_equal(sys_dt.damp(), expected_dt)
        np.testing.assert_almost_equal(damp(sys_dt), expected_dt)

    def test_dcgain(self):
        sys = tf(84, [1, 2])
        np.testing.assert_equal(sys.dcgain(), 42)
        np.testing.assert_equal(dcgain(sys), 42)
