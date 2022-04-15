"""frd_test.py - test FRD class

RvP, 4 Oct 2012
"""

import sys as pysys

import numpy as np
import matplotlib.pyplot as plt
import pytest

import control as ct
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.frdata import FRD, _convert_to_FRD, FrequencyResponseData
from control import bdalg, evalfr, freqplot
from control.tests.conftest import slycotonly
from control.exception import pandas_check


class TestFRD:
    """These are tests for functionality and correct reporting of the
    frequency response data class."""

    def testBadInputType(self):
        """Give the constructor invalid input types."""
        with pytest.raises(ValueError):
            FRD()
        with pytest.raises(TypeError):
            FRD([1])

    def testInconsistentDimension(self):
        with pytest.raises(TypeError):
            FRD([1, 1], [1, 2, 3])

    def testSISOtf(self):
        # get a SISO transfer function
        h = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        frd = FRD(h, omega)
        assert isinstance(frd, FRD)

        mag1, phase1, omega1 = frd.frequency_response([1.0])
        mag2, phase2, omega2 = h.frequency_response([1.0])
        np.testing.assert_array_almost_equal(mag1, mag2)
        np.testing.assert_array_almost_equal(phase1, phase2)
        np.testing.assert_array_almost_equal(omega1, omega2)

    def testOperators(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)

        np.testing.assert_array_almost_equal(
            (f1 + f2).frequency_response([0.1, 1.0, 10])[0],
            (h1 + h2).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 + f2).frequency_response([0.1, 1.0, 10])[1],
            (h1 + h2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 - f2).frequency_response([0.1, 1.0, 10])[0],
            (h1 - h2).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 - f2).frequency_response([0.1, 1.0, 10])[1],
            (h1 - h2).frequency_response([0.1, 1.0, 10])[1])

        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * f2).frequency_response([0.1, 1.0, 10])[1],
            (h1 * h2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / f2).frequency_response([0.1, 1.0, 10])[1],
            (h1 / h2).frequency_response([0.1, 1.0, 10])[1])

        # with default conversion from scalar
        np.testing.assert_array_almost_equal(
            (f1 * 1.5).frequency_response([0.1, 1.0, 10])[1],
            (h1 * 1.5).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / 1.7).frequency_response([0.1, 1.0, 10])[1],
            (h1 / 1.7).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (2.2 * f2).frequency_response([0.1, 1.0, 10])[1],
            (2.2 * h2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (1.3 / f2).frequency_response([0.1, 1.0, 10])[1],
            (1.3 / h2).frequency_response([0.1, 1.0, 10])[1])

    def testOperatorsTf(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)
        f2  # reference to avoid pyflakes error

        np.testing.assert_array_almost_equal(
            (f1 + h2).frequency_response([0.1, 1.0, 10])[0],
            (h1 + h2).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 + h2).frequency_response([0.1, 1.0, 10])[1],
            (h1 + h2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 - h2).frequency_response([0.1, 1.0, 10])[0],
            (h1 - h2).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1 - h2).frequency_response([0.1, 1.0, 10])[1],
            (h1 - h2).frequency_response([0.1, 1.0, 10])[1])
        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * h2).frequency_response([0.1, 1.0, 10])[1],
            (h1 * h2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1 / h2).frequency_response([0.1, 1.0, 10])[1],
            (h1 / h2).frequency_response([0.1, 1.0, 10])[1])
        # the reverse does not work

    def testbdalg(self):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        f2 = FRD(h2, omega)

        np.testing.assert_array_almost_equal(
            (bdalg.series(f1, f2)).frequency_response([0.1, 1.0, 10])[0],
            (bdalg.series(h1, h2)).frequency_response([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.parallel(f1, f2)).frequency_response([0.1, 1.0, 10])[0],
            (bdalg.parallel(h1, h2)).frequency_response([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.feedback(f1, f2)).frequency_response([0.1, 1.0, 10])[0],
            (bdalg.feedback(h1, h2)).frequency_response([0.1, 1.0, 10])[0])

        np.testing.assert_array_almost_equal(
            (bdalg.negate(f1)).frequency_response([0.1, 1.0, 10])[0],
            (bdalg.negate(h1)).frequency_response([0.1, 1.0, 10])[0])

#       append() and connect() not implemented for FRD objects
#        np.testing.assert_array_almost_equal(
#            (bdalg.append(f1, f2)).frequency_response([0.1, 1.0, 10])[0],
#            (bdalg.append(h1, h2)).frequency_response([0.1, 1.0, 10])[0])
#
#        f3 = bdalg.append(f1, f2, f2)
#        h3 = bdalg.append(h1, h2, h2)
#        Q = np.mat([ [1, 2], [2, -1] ])
#        np.testing.assert_array_almost_equal(
#           (bdalg.connect(f3, Q, [2], [1])).frequency_response([0.1, 1.0, 10])[0],
#            (bdalg.connect(h3, Q, [2], [1])).frequency_response([0.1, 1.0, 10])[0])

    def testFeedback(self):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(h1, omega)
        np.testing.assert_array_almost_equal(
            f1.feedback(1).frequency_response([0.1, 1.0, 10])[0],
            h1.feedback(1).frequency_response([0.1, 1.0, 10])[0])

        # Make sure default argument also works
        np.testing.assert_array_almost_equal(
            f1.feedback().frequency_response([0.1, 1.0, 10])[0],
            h1.feedback().frequency_response([0.1, 1.0, 10])[0])

    def testFeedback2(self):
        h2 = StateSpace([[-1.0, 0], [0, -2.0]], [[0.4], [0.1]],
                        [[1.0, 0], [0, 1]], [[0.0], [0.0]])
        # h2.feedback([[0.3, 0.2], [0.1, 0.1]])

    def testAuto(self):
        omega = np.logspace(-1, 2, 10)
        f1 = _convert_to_FRD(1, omega)
        f2 = _convert_to_FRD(np.array([[1, 0], [0.1, -1]]), omega)
        f2 = _convert_to_FRD([[1, 0], [0.1, -1]], omega)
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

    @slycotonly
    def testMIMO(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            sys.frequency_response([0.1, 1.0, 10])[0],
            f1.frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            sys.frequency_response([0.1, 1.0, 10])[1],
            f1.frequency_response([0.1, 1.0, 10])[1])

    @slycotonly
    def testMIMOfb(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega).feedback([[0.1, 0.3], [0.0, 1.0]])
        f2 = FRD(sys.feedback([[0.1, 0.3], [0.0, 1.0]]), omega)
        np.testing.assert_array_almost_equal(
            f1.frequency_response([0.1, 1.0, 10])[0],
            f2.frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            f1.frequency_response([0.1, 1.0, 10])[1],
            f2.frequency_response([0.1, 1.0, 10])[1])

    @slycotonly
    def testMIMOfb2(self):
        sys = StateSpace(np.array([[-2.0, 0, 0],
                                   [0, -1, 1],
                                   [0, 0, -3]]),
                         np.array([[1.0, 0], [0, 0], [0, 1]]),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        K = np.array([[1, 0.3, 0], [0.1, 0, 0]])
        f1 = FRD(sys, omega).feedback(K)
        f2 = FRD(sys.feedback(K), omega)
        np.testing.assert_array_almost_equal(
            f1.frequency_response([0.1, 1.0, 10])[0],
            f2.frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            f1.frequency_response([0.1, 1.0, 10])[1],
            f2.frequency_response([0.1, 1.0, 10])[1])

    @slycotonly
    def testMIMOMult(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        f2 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response([0.1, 1.0, 10])[0],
            (sys*sys).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response([0.1, 1.0, 10])[1],
            (sys*sys).frequency_response([0.1, 1.0, 10])[1])

    @slycotonly
    def testMIMOSmooth(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        sys2 = np.array([[1, 0, 0], [0, 1, 0]]) * sys
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega, smooth=True)
        f2 = FRD(sys2, omega, smooth=True)
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response([0.1, 1.0, 10])[0],
            (sys*sys2).frequency_response([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response([0.1, 1.0, 10])[1],
            (sys*sys2).frequency_response([0.1, 1.0, 10])[1])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response([0.1, 1.0, 10])[2],
            (sys*sys2).frequency_response([0.1, 1.0, 10])[2])

    def testAgainstOctave(self):
        # with data from octave:
        # sys = ss([-2 0 0; 0 -1 1; 0 0 -3],
        #  [1 0; 0 0; 0 1], eye(3), zeros(3,2))
        # bfr = frd(bsys, [1])
        sys = StateSpace(np.array([[-2.0, 0, 0], [0, -1, 1], [0, 0, -3]]),
                         np.array([[1.0, 0], [0, 0], [0, 1]]),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1.frequency_response([1.0])[0] *
             np.exp(1j * f1.frequency_response([1.0])[1])).reshape(3, 2),
            np.array([[0.4 - 0.2j, 0], [0, 0.1 - 0.2j], [0, 0.3 - 0.1j]]))

    def test_string_representation(self, capsys):
        sys = FRD([1, 2, 3], [4, 5, 6])
        print(sys)              # Just print without checking

    def test_frequency_mismatch(self, recwarn):
        # recwarn: there may be a warning before the error!
        # Overlapping but non-equal frequency ranges
        sys1 = FRD([1, 2, 3], [4, 5, 6])
        sys2 = FRD([2, 3, 4], [5, 6, 7])
        with pytest.raises(NotImplementedError):
            FRD.__add__(sys1, sys2)

        # One frequency range is a subset of another
        sys1 = FRD([1, 2, 3], [4, 5, 6])
        sys2 = FRD([2, 3], [4, 5])
        with pytest.raises(NotImplementedError):
            FRD.__add__(sys1, sys2)

    def test_size_mismatch(self):
        sys1 = FRD(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))

        # Different number of inputs
        sys2 = FRD(ct.rss(3, 1, 2), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            FRD.__add__(sys1, sys2)

        # Different number of outputs
        sys2 = FRD(ct.rss(3, 2, 1), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            FRD.__add__(sys1, sys2)

        # Inputs and outputs don't match
        with pytest.raises(ValueError):
            FRD.__mul__(sys2, sys1)

        # Feedback mismatch
        with pytest.raises(ValueError):
            FRD.feedback(sys2, sys1)

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
        with pytest.raises(ValueError):
            FRD.__pow__(frd_tf, 0.5)

        # Selected testing on transfer function conversion
        sys_add = frd_2 + sys_tf
        chk_add = frd_2 + frd_tf
        np.testing.assert_array_almost_equal(sys_add.omega, chk_add.omega)
        np.testing.assert_array_almost_equal(sys_add.fresp, chk_add.fresp)

        # Input/output mismatch size mismatch in  rmul
        sys1 = FRD(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            FRD.__rmul__(frd_2, sys1)

        # Make sure conversion of something random generates exception
        with pytest.raises(TypeError):
            FRD.__add__(frd_tf, 'string')

    def test_eval(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = FRD(sys_tf, np.logspace(-1, 1, 3))
        np.testing.assert_almost_equal(sys_tf(1j), frd_tf.eval(1))
        np.testing.assert_almost_equal(sys_tf(1j), frd_tf(1j))

        # Should get an error if we evaluate at an unknown frequency
        with pytest.raises(ValueError, match="not .* in frequency list"):
            frd_tf.eval(2)

        # Should get an error if we evaluate at an complex number
        with pytest.raises(ValueError, match="can only accept real-valued"):
            frd_tf.eval(2 + 1j)

        # Should get an error if we use __call__ at real-valued frequency
        with pytest.raises(ValueError, match="only accept purely imaginary"):
            frd_tf(2)

    def test_freqresp_deprecated(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = FRD(sys_tf, np.logspace(-1, 1, 3))
        with pytest.warns(DeprecationWarning):
            frd_tf.freqresp(1.)

    def test_repr_str(self):
        # repr printing
        array = np.array
        sys0 = FrequencyResponseData([1.0, 0.9+0.1j, 0.1+2j, 0.05+3j],
                                     [0.1, 1.0, 10.0, 100.0])
        sys1 = FrequencyResponseData(sys0.fresp, sys0.omega, smooth=True)
        ref0 = "FrequencyResponseData(" \
            "array([[[1.  +0.j , 0.9 +0.1j, 0.1 +2.j , 0.05+3.j ]]])," \
            " array([  0.1,   1. ,  10. , 100. ]))"
        ref1 = ref0[:-1] + ", smooth=True)"
        sysm = FrequencyResponseData(
            np.matmul(array([[1],[2]]), sys0.fresp), sys0.omega)

        assert repr(sys0) == ref0
        assert repr(sys1) == ref1
        sys0r = eval(repr(sys0))
        np.testing.assert_array_almost_equal(sys0r.fresp, sys0.fresp)
        np.testing.assert_array_almost_equal(sys0r.omega, sys0.omega)
        sys1r = eval(repr(sys1))
        np.testing.assert_array_almost_equal(sys1r.fresp, sys1.fresp)
        np.testing.assert_array_almost_equal(sys1r.omega, sys1.omega)
        assert(sys1.ifunc is not None)

        refs = """Frequency response data
Freq [rad/s]  Response
------------  ---------------------
       0.100           1        +0j
       1.000         0.9      +0.1j
      10.000         0.1        +2j
     100.000        0.05        +3j"""
        assert str(sys0) == refs
        assert str(sys1) == refs

        # print multi-input system
        refm = """Frequency response data
Input 1 to output 1:
Freq [rad/s]  Response
------------  ---------------------
       0.100           1        +0j
       1.000         0.9      +0.1j
      10.000         0.1        +2j
     100.000        0.05        +3j
Input 2 to output 1:
Freq [rad/s]  Response
------------  ---------------------
       0.100           2        +0j
       1.000         1.8      +0.2j
      10.000         0.2        +4j
     100.000         0.1        +6j"""
        assert str(sysm) == refm

    def test_unrecognized_keyword(self):
        h = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        with pytest.raises(TypeError, match="unrecognized keyword"):
            frd = FRD(h, omega, unknown=None)


def test_named_signals():
    ct.namedio.NamedIOSystem._idCounter = 0
    h1 = TransferFunction([1], [1, 2, 2])
    h2 = TransferFunction([1], [0.1, 1])
    omega = np.logspace(-1, 2, 10)
    f1 = FRD(h1, omega)
    f2 = FRD(h2, omega)

    # Make sure that systems were properly named
    assert f1.name == 'sys[2]'
    assert f2.name == 'sys[3]'
    assert f1.ninputs == 1
    assert f1.input_labels == ['u[0]']
    assert f1.noutputs == 1
    assert f1.output_labels == ['y[0]']

    # Change names
    f1 = FRD(h1, omega, name='mysys', inputs='u0', outputs='y0')
    assert f1.name == 'mysys'
    assert f1.ninputs == 1
    assert f1.input_labels == ['u0']
    assert f1.noutputs == 1
    assert f1.output_labels == ['y0']


@pytest.mark.skipif(not pandas_check(), reason="pandas not installed")
def test_to_pandas():
    # Create a SISO frequency response
    h1 = TransferFunction([1], [1, 2, 2])
    omega = np.logspace(-1, 2, 10)
    resp = FRD(h1, omega)

    # Convert to pandas
    df = resp.to_pandas()

    # Check to make sure the data make senses
    np.testing.assert_equal(df['omega'], resp.omega)
    np.testing.assert_equal(df['H_{y[0], u[0]}'], resp.fresp[0, 0])


def test_frequency_response():
    # Create an SISO frequence response
    sys = ct.rss(2, 2, 2)
    omega = np.logspace(-2, 2, 20)
    resp = ct.frequency_response(sys, omega)
    eval = sys(omega*1j)

    # Make sure we get the right answers in various ways
    np.testing.assert_equal(resp.magnitude, np.abs(eval))
    np.testing.assert_equal(resp.phase, np.angle(eval))
    np.testing.assert_equal(resp.omega, omega)

    # Make sure that we can change the properties of the response
    sys = ct.rss(2, 1, 1)
    resp_default = ct.frequency_response(sys, omega)
    mag_default, phase_default, omega_default = resp_default
    assert mag_default.ndim == 1
    assert phase_default.ndim == 1
    assert omega_default.ndim == 1
    assert mag_default.shape[0] == omega_default.shape[0]
    assert phase_default.shape[0] == omega_default.shape[0]

    resp_nosqueeze = ct.frequency_response(sys, omega, squeeze=False)
    mag_nosqueeze, phase_nosqueeze, omega_nosqueeze = resp_nosqueeze
    assert mag_nosqueeze.ndim == 3
    assert phase_nosqueeze.ndim == 3
    assert omega_nosqueeze.ndim == 1
    assert mag_nosqueeze.shape[2] == omega_nosqueeze.shape[0]
    assert phase_nosqueeze.shape[2] == omega_nosqueeze.shape[0]

    # Try changing the response
    resp_def_nosq = resp_default(squeeze=False)
    mag_def_nosq, phase_def_nosq, omega_def_nosq = resp_def_nosq
    assert mag_def_nosq.shape == mag_nosqueeze.shape
    assert phase_def_nosq.shape == phase_nosqueeze.shape
    assert omega_def_nosq.shape == omega_nosqueeze.shape

    resp_nosq_sq = resp_nosqueeze(squeeze=True)
    mag_nosq_sq, phase_nosq_sq, omega_nosq_sq = resp_nosq_sq
    assert mag_nosq_sq.shape == mag_default.shape
    assert phase_nosq_sq.shape == phase_default.shape
    assert omega_nosq_sq.shape == omega_default.shape
