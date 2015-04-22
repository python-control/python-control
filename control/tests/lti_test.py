#!/usr/bin/env python

import unittest
import numpy as np
from control.lti import *
from control.xferfcn import tf
import numpy as np

class TestUtils(unittest.TestCase):
    def test_pole(self):
        sys = tf(126, [-1, 42])
        np.testing.assert_equal(sys.pole(), 42)
        np.testing.assert_equal(pole(sys), 42)

    def test_zero(self):
        sys = tf([-1, 42], [1, 10])
        np.testing.assert_equal(sys.zero(), 42)
        np.testing.assert_equal(zero(sys), 42)

    def test_damp(self):
        zeta = 0.1
        wn = 42
        p = -wn * zeta + 1j * wn * np.sqrt(1 - zeta**2)
        sys = tf(1, [1, 2 * zeta * wn, wn**2])
        expected = ([wn, wn], [zeta, zeta], [p, p.conjugate()])
        np.testing.assert_equal(sys.damp(), expected)
        np.testing.assert_equal(damp(sys), expected)

    def test_dcgain(self):
        sys = tf(84, [1, 2])
        np.testing.assert_equal(sys.dcgain(), 42)
        np.testing.assert_equal(dcgain(sys), 42)
