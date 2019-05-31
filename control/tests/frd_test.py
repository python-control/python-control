#!/usr/bin/env python
#
# frd_test.py - test FRD class
# RvP, 4 Oct 2012


import unittest
import sys as pysys
import numpy as np
import control as ct
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.frdata import FRD, _convertToFRD
from control import bdalg
from control import freqplot
from control.exception import slycot_check
import matplotlib.pyplot as plt


class TestFRD(unittest.TestCase):
    """These are tests for functionality and correct reporting of the
    frequency response data class."""

    def testBadInputType(self):
        """Give the constructor invalid input types."""
        self.assertRaises(ValueError, FRD)
        self.assertRaises(TypeError, FRD, [1])

    def testInconsistentDimension(self):
        self.assertRaises(TypeError, FRD, [1, 1], [1, 2, 3])

    def testSISOtf(self):
        # get a SISO transfer function
        h = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        frd = FRD(h, omega)
        assert isinstance(frd, FRD)

        np.testing.assert_array_almost_equal(
            frd.freqresp([1.0]), h.freqresp([1.0]))

    def testOperators(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)

        np.testing.assert_array_almost_equal(
            (f1 + f2).freqresp([0.1, 1.0, 10])[0],
            (h1 + h2).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 + f2).freqresp([0.1, 1.0, 10])[1],
            (h1 + h2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 - f2).freqresp([0.1, 1.0, 10])[0],
            (h1 - h2).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 - f2).freqresp([0.1, 1.0, 10])[1],
            (h1 - h2).freqresp([0.1, 1.0, 10])[1])

        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * f2).freqresp([0.1, 1.0, 10])[1],
            (h1 * h2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / f2).freqresp([0.1, 1.0, 10])[1],
            (h1 / h2).freqresp([0.1, 1.0, 10])[1])

        # with default conversion from scalar
        np.testing.assert_array_almost_equal(
            (f1 * 1.5).freqresp([0.1, 1.0, 10])[1],
            (h1 * 1.5).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / 1.7).freqresp([0.1, 1.0, 10])[1],
            (h1 / 1.7).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (2.2 * f2).freqresp([0.1, 1.0, 10])[1],
            (2.2 * h2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (1.3 / f2).freqresp([0.1, 1.0, 10])[1],
            (1.3 / h2).freqresp([0.1, 1.0, 10])[1])

    def testOperatorsTf(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)
        f2  # reference to avoid pyflakes error

        np.testing.assert_array_almost_equal(
            (f1 + h2).freqresp([0.1, 1.0, 10])[0],
            (h1 + h2).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 + h2).freqresp([0.1, 1.0, 10])[1],
            (h1 + h2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 - h2).freqresp([0.1, 1.0, 10])[0],
            (h1 - h2).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 - h2).freqresp([0.1, 1.0, 10])[1],
            (h1 - h2).freqresp([0.1, 1.0, 10])[1])
        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * h2).freqresp([0.1, 1.0, 10])[1],
            (h1 * h2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / h2).freqresp([0.1, 1.0, 10])[1],
            (h1 / h2).freqresp([0.1, 1.0, 10])[1])
        # the reverse does not work

    def testbdalg(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)

        np.testing.assert_array_almost_equal(
            (bdalg.series(f1, f2)).freqresp([0.1, 1.0, 10])[0],
            (bdalg.series(h1, h2)).freqresp([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.parallel(f1, f2)).freqresp([0.1, 1.0, 10])[0],
            (bdalg.parallel(h1, h2)).freqresp([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.feedback(f1, f2)).freqresp([0.1, 1.0, 10])[0],
            (bdalg.feedback(h1, h2)).freqresp([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.negate(f1)).freqresp([0.1, 1.0, 10])[0],
            (bdalg.negate(h1)).freqresp([0.1, 1.0, 10])[0])

#       append() and connect() not implemented for FRD objects
#        np.testing.assert_array_almost_equal(
#            (bdalg.append(f1, f2)).freqresp([0.1, 1.0, 10])[0],
#            (bdalg.append(h1, h2)).freqresp([0.1, 1.0, 10])[0])
#
#        f3 = bdalg.append(f1, f2, f2)
#        h3 = bdalg.append(h1, h2, h2)
#        Q = np.mat([ [1, 2], [2, -1] ])
#        np.testing.assert_array_almost_equal(
#           (bdalg.connect(f3, Q, [2], [1])).freqresp([0.1, 1.0, 10])[0],
#            (bdalg.connect(h3, Q, [2], [1])).freqresp([0.1, 1.0, 10])[0])

    def testFeedback(self):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        np.testing.assert_array_almost_equal(
            f1.feedback(1).freqresp([0.1, 1.0, 10])[0],
            h1.feedback(1).freqresp([0.1, 1.0, 10])[0])

        # Make sure default argument also works
        np.testing.assert_array_almost_equal(
            f1.feedback().freqresp([0.1, 1.0, 10])[0],
            h1.feedback().freqresp([0.1, 1.0, 10])[0])

    def testFeedback2(self):
        h2 = StateSpace([[-1.0, 0], [0, -2.0]], [[0.4], [0.1]],
                        [[1.0, 0], [0, 1]], [[0.0], [0.0]])
        # h2.feedback([[0.3, 0.2], [0.1, 0.1]])

    def testAuto(self):
        omega = np.logspace(-1, 2, 10)
        f1 = _convertToFRD(1, omega)
        f2 = _convertToFRD(np.matrix([[1, 0], [0.1, -1]]), omega)
        f2 = _convertToFRD([[1, 0], [0.1, -1]], omega)
        f1, f2  # reference to avoid pyflakes error

    def testNyquist(self):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 40)
        f1 = FRD(h1, omega, smooth=True)
        freqplot.nyquist(f1, np.logspace(-1, 2, 100))
        # plt.savefig('/dev/null', format='svg')
        plt.figure(2)
        freqplot.nyquist(f1, f1.omega)
        # plt.savefig('/dev/null', format='svg')

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMO(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            sys.freqresp([0.1, 1.0, 10])[0],
            f1.freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            sys.freqresp([0.1, 1.0, 10])[1],
            f1.freqresp([0.1, 1.0, 10])[1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMOfb(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega).feedback([[0.1, 0.3], [0.0, 1.0]])
        f2 = FRD(sys.feedback([[0.1, 0.3], [0.0, 1.0]]), omega)
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[0],
            f2.freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[1],
            f2.freqresp([0.1, 1.0, 10])[1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMOfb2(self):
        sys = StateSpace(np.matrix('-2.0 0 0; 0 -1 1; 0 0 -3'),
                         np.matrix('1.0 0; 0 0; 0 1'),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        K = np.matrix('1 0.3 0; 0.1 0 0')
        f1 = FRD(sys, omega).feedback(K)
        f2 = FRD(sys.feedback(K), omega)
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[0],
            f2.freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[1],
            f2.freqresp([0.1, 1.0, 10])[1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMOMult(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        f2 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1*f2).freqresp([0.1, 1.0, 10])[0],
            (sys*sys).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).freqresp([0.1, 1.0, 10])[1],
            (sys*sys).freqresp([0.1, 1.0, 10])[1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMIMOSmooth(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        sys2 = np.matrix([[1, 0, 0], [0, 1, 0]]) * sys
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega, smooth=True)
        f2 = FRD(sys2, omega, smooth=True)
        np.testing.assert_array_almost_equal(
            (f1*f2).freqresp([0.1, 1.0, 10])[0],
            (sys*sys2).freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).freqresp([0.1, 1.0, 10])[1],
            (sys*sys2).freqresp([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1*f2).freqresp([0.1, 1.0, 10])[2],
            (sys*sys2).freqresp([0.1, 1.0, 10])[2])

    def testAgainstOctave(self):
        # with data from octave:
        # sys = ss([-2 0 0; 0 -1 1; 0 0 -3],
        #  [1 0; 0 0; 0 1], eye(3), zeros(3,2))
        # bfr = frd(bsys, [1])
        sys = StateSpace(np.matrix('-2.0 0 0; 0 -1 1; 0 0 -3'),
                         np.matrix('1.0 0; 0 0; 0 1'),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1.freqresp([1.0])[0] *
             np.exp(1j*f1.freqresp([1.0])[1])).reshape(3, 2),
            np.matrix('0.4-0.2j 0; 0 0.1-0.2j; 0 0.3-0.1j'))

    def test_string_representation(self):
        sys = FRD([1, 2, 3], [4, 5, 6])
        print(sys)              # Just print without checking

    def test_frequency_mismatch(self):
        # Overlapping but non-equal frequency ranges
        sys1 = FRD([1, 2, 3], [4, 5, 6])
        sys2 = FRD([2, 3, 4], [5, 6, 7])
        self.assertRaises(NotImplementedError, FRD.__add__, sys1, sys2)

        # One frequency range is a subset of another
        sys1 = FRD([1, 2, 3], [4, 5, 6])
        sys2 = FRD([2, 3], [4, 5])
        self.assertRaises(NotImplementedError, FRD.__add__, sys1, sys2)

    def test_size_mismatch(self):
        sys1 = FRD(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))

        # Different number of inputs
        sys2 = FRD(ct.rss(3, 1, 2), np.logspace(-1, 1, 10))
        self.assertRaises(ValueError, FRD.__add__, sys1, sys2)

        # Different number of outputs
        sys2 = FRD(ct.rss(3, 2, 1), np.logspace(-1, 1, 10))
        self.assertRaises(ValueError, FRD.__add__, sys1, sys2)

        # Inputs and outputs don't match
        self.assertRaises(ValueError, FRD.__mul__, sys2, sys1)

        # Feedback mismatch
        self.assertRaises(ValueError, FRD.feedback, sys2, sys1)

    def test_operator_conversion(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = FRD(sys_tf, np.logspace(-1, 1, 10))
        frd_2 = FRD(2 * np.ones(10), np.logspace(-1, 1, 10))

        # Make sure that we can add, multiply, and feedback constants
        sys_add = frd_tf + 2
        chk_add = frd_tf + frd_2
        np.testing.assert_array_almost_equal(sys_add.omega, chk_add.omega)
        np.testing.assert_array_almost_equal(sys_add.fresp, chk_add.fresp)

        sys_radd = 2 + frd_tf
        chk_radd = frd_2 + frd_tf
        np.testing.assert_array_almost_equal(sys_radd.omega, chk_radd.omega)
        np.testing.assert_array_almost_equal(sys_radd.fresp, chk_radd.fresp)

        sys_sub = frd_tf - 2
        chk_sub = frd_tf - frd_2
        np.testing.assert_array_almost_equal(sys_sub.omega, chk_sub.omega)
        np.testing.assert_array_almost_equal(sys_sub.fresp, chk_sub.fresp)

        sys_rsub = 2 - frd_tf
        chk_rsub = frd_2 - frd_tf
        np.testing.assert_array_almost_equal(sys_rsub.omega, chk_rsub.omega)
        np.testing.assert_array_almost_equal(sys_rsub.fresp, chk_rsub.fresp)

        sys_mul = frd_tf * 2
        chk_mul = frd_tf * frd_2
        np.testing.assert_array_almost_equal(sys_mul.omega, chk_mul.omega)
        np.testing.assert_array_almost_equal(sys_mul.fresp, chk_mul.fresp)

        sys_rmul = 2 * frd_tf
        chk_rmul = frd_2 * frd_tf
        np.testing.assert_array_almost_equal(sys_rmul.omega, chk_rmul.omega)
        np.testing.assert_array_almost_equal(sys_rmul.fresp, chk_rmul.fresp)

        sys_rdiv = 2 / frd_tf
        chk_rdiv = frd_2 / frd_tf
        np.testing.assert_array_almost_equal(sys_rdiv.omega, chk_rdiv.omega)
        np.testing.assert_array_almost_equal(sys_rdiv.fresp, chk_rdiv.fresp)

        sys_pow = frd_tf**2
        chk_pow = FRD(sys_tf**2, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(sys_pow.omega, chk_pow.omega)
        np.testing.assert_array_almost_equal(sys_pow.fresp, chk_pow.fresp)

        sys_pow = frd_tf**-2
        chk_pow = FRD(sys_tf**-2, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(sys_pow.omega, chk_pow.omega)
        np.testing.assert_array_almost_equal(sys_pow.fresp, chk_pow.fresp)

        # Assertion error if we try to raise to a non-integer power
        self.assertRaises(ValueError, FRD.__pow__, frd_tf, 0.5)

        # Selected testing on transfer function conversion
        sys_add = frd_2 + sys_tf
        chk_add = frd_2 + frd_tf
        np.testing.assert_array_almost_equal(sys_add.omega, chk_add.omega)
        np.testing.assert_array_almost_equal(sys_add.fresp, chk_add.fresp)

        # Input/output mismatch size mismatch in  rmul
        sys1 = FRD(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))
        self.assertRaises(ValueError, FRD.__rmul__, frd_2, sys1)

        # Make sure conversion of something random generates exception
        self.assertRaises(TypeError,  FRD.__add__, frd_tf, 'string')

    def test_eval(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = FRD(sys_tf, np.logspace(-1, 1, 3))
        np.testing.assert_almost_equal(sys_tf.evalfr(1), frd_tf.eval(1))

        # Should get an error if we evaluate at an unknown frequency
        self.assertRaises(ValueError, frd_tf.eval, 2)

    # This test only works in Python 3 due to a conflict with the same
    # warning type in other test modules (frd_test.py).  See
    # https://bugs.python.org/issue4180 for more details
    @unittest.skipIf(pysys.version_info < (3, 0), "test requires Python 3+")
    def test_evalfr_deprecated(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = FRD(sys_tf, np.logspace(-1, 1, 3))

        # Deprecated version of the call (should generate warning)
        import warnings
        with warnings.catch_warnings():
            # Make warnings generate an exception
            warnings.simplefilter('error')

            # Make sure that we get a pending deprecation warning
            self.assertRaises(PendingDeprecationWarning, frd_tf.evalfr, 1.)

        # FRD.evalfr() is being deprecated
        import warnings
        with warnings.catch_warnings():
            # Make warnings generate an exception
            warnings.simplefilter('error')

            # Make sure that we get a pending deprecation warning
            self.assertRaises(PendingDeprecationWarning, frd_tf.evalfr, 1.)

            
def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestFRD)

if __name__ == "__main__":
    unittest.main()
