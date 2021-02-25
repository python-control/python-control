"""input_element_int_test.py

Author: Kangwon Lee (kangwonlee)
Date: 22 Oct 2017

Modified:
* 29 Dec 2017, RMM - updated file name and added header
"""

import numpy as np
from control import dcgain, ss, tf

class TestTfInputIntElement:
    """input_element_int_test

    Unit tests contributed as part of PR gh-158, "SISO tf() may not work
    with numpy arrays with numpy.int elements
    """

    def test_tf_den_with_numpy_int_element(self):
        num = 1
        den = np.convolve([1, 2, 1], [1, 1, 1])

        sys = tf(num, den)

        np.testing.assert_almost_equal(1., dcgain(sys))

    def test_tf_num_with_numpy_int_element(self):
        num = np.convolve([1], [1, 1])
        den = np.convolve([1, 2, 1], [1, 1, 1])

        sys = tf(num, den)

        np.testing.assert_almost_equal(1., dcgain(sys))

    # currently these pass
    def test_tf_input_with_int_element(self):
        num = 1
        den = np.convolve([1.0, 2, 1], [1, 1, 1])

        sys = tf(num, den)

        np.testing.assert_almost_equal(1., dcgain(sys))

    def test_ss_input_with_int_element(self):
        a = np.array([[0, 1],
                      [-1, -2]], dtype=int)
        b = np.array([[0],
                       [1]], dtype=int)
        c = np.array([[0, 1]], dtype=int)
        d = np.array([[1]], dtype=int)

        sys = ss(a, b, c, d)
        sys2 = tf(sys)
        np.testing.assert_almost_equal(dcgain(sys), dcgain(sys2))

    def test_ss_input_with_0int_dcgain(self):
        a = np.array([[0, 1],
                      [-1, -2]], dtype=int)
        b = np.array([[0],
                       [1]], dtype=int)
        c = np.array([[0, 1]], dtype=int)
        d = 0
        sys = ss(a, b, c, d)
        np.testing.assert_allclose(dcgain(sys), 0,
                                   atol=np.finfo(float).epsneg)
