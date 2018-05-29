# input_element_int_test.py
#
# Author: Kangwon Lee (kangwonlee)
# Date: 22 Oct 2017
#
# Unit tests contributed as part of PR #158, "SISO tf() may not work
# with numpy arrays with numpy.int elements"
#
# Modified:
# * 29 Dec 2017, RMM - updated file name and added header

import unittest
import numpy as np
import control as ctl

class TestTfInputIntElement(unittest.TestCase):
    # currently these do not pass
    def test_tf_den_with_numpy_int_element(self):
        num = 1
        den = np.convolve([1, 2, 1], [1, 1, 1])

        sys = ctl.tf(num, den)

        self.assertAlmostEqual(1.0, ctl.dcgain(sys))

    def test_tf_num_with_numpy_int_element(self):
        num = np.convolve([1], [1, 1])
        den = np.convolve([1, 2, 1], [1, 1, 1])

        sys = ctl.tf(num, den)

        self.assertAlmostEqual(1.0, ctl.dcgain(sys))

    # currently these pass
    def test_tf_input_with_int_element_works(self):
        num = 1
        den = np.convolve([1.0, 2, 1], [1, 1, 1])

        sys = ctl.tf(num, den)

        self.assertAlmostEqual(1.0, ctl.dcgain(sys))

    def test_ss_input_with_int_element(self):
        ident = np.matrix(np.identity(2), dtype=int)
        a = np.matrix([[0, 1],
                       [-1, -2]], dtype=int) * ident
        b = np.matrix([[0],
                       [1]], dtype=int)
        c = np.matrix([[0, 1]], dtype=int)
        d = 0

        sys = ctl.ss(a, b, c, d)
        sys2 = ctl.ss2tf(sys)
        self.assertAlmostEqual(ctl.dcgain(sys), ctl.dcgain(sys2))
