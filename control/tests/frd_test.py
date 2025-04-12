"""frd_test.py - test FRD class

RvP, 4 Oct 2012
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

import control as ct
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.frdata import frd, _convert_to_frd, FrequencyResponseData
from control import bdalg, freqplot
from control.tests.conftest import slycotonly
from control.exception import pandas_check


class TestFRD:
    """These are tests for functionality and correct reporting of the
    frequency response data class."""

    def testBadInputType(self):
        """Give the constructor invalid input types."""
        with pytest.raises(ValueError):
            frd()
        with pytest.raises(TypeError):
            frd([1])

    def testInconsistentDimension(self):
        with pytest.raises(TypeError):
            frd([1, 1], [1, 2, 3])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testSISOtf(self, frd_fcn):
        # get a SISO transfer function
        h = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        sys = frd_fcn(h, omega)
        assert isinstance(sys, FrequencyResponseData)

        mag1, phase1, omega1 = sys.frequency_response([1.0])
        mag2, phase2, omega2 = h.frequency_response([1.0])
        np.testing.assert_array_almost_equal(mag1, mag2)
        np.testing.assert_array_almost_equal(phase1, phase2)
        np.testing.assert_array_almost_equal(omega1, omega2)

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testOperators(self, frd_fcn):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(h1, omega)
        f2 = frd_fcn(h2, omega)

        np.testing.assert_array_almost_equal(
            (f1 + f2).frequency_response(chkpts)[0],
            (h1 + h2).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1 + f2).frequency_response(chkpts)[1],
            (h1 + h2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1 - f2).frequency_response(chkpts)[0],
            (h1 - h2).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1 - f2).frequency_response(chkpts)[1],
            (h1 - h2).frequency_response(chkpts)[1])

        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * f2).frequency_response(chkpts)[1],
            (h1 * h2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1 / f2).frequency_response(chkpts)[1],
            (h1 / h2).frequency_response(chkpts)[1])

        # with default conversion from scalar
        np.testing.assert_array_almost_equal(
            (f1 * 1.5).frequency_response(chkpts)[1],
            (h1 * 1.5).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1 / 1.7).frequency_response(chkpts)[1],
            (h1 / 1.7).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (2.2 * f2).frequency_response(chkpts)[1],
            (2.2 * h2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (1.3 / f2).frequency_response(chkpts)[1],
            (1.3 / h2).frequency_response(chkpts)[1])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testOperatorsTf(self, frd_fcn):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(h1, omega)
        f2 = frd_fcn(h2, omega)
        f2  # reference to avoid pyflakes error

        np.testing.assert_array_almost_equal(
            (f1 + h2).frequency_response(chkpts)[0],
            (h1 + h2).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1 + h2).frequency_response(chkpts)[1],
            (h1 + h2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1 - h2).frequency_response(chkpts)[0],
            (h1 - h2).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1 - h2).frequency_response(chkpts)[1],
            (h1 - h2).frequency_response(chkpts)[1])
        # multiplication and division
        np.testing.assert_array_almost_equal(
            (f1 * h2).frequency_response(chkpts)[1],
            (h1 * h2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1 / h2).frequency_response(chkpts)[1],
            (h1 / h2).frequency_response(chkpts)[1])
        # the reverse does not work

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testbdalg(self, frd_fcn):
        # get two SISO transfer functions
        h1 = TransferFunction([1], [1, 2, 2])
        h2 = TransferFunction([1], [0.1, 1])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(h1, omega)
        f2 = frd_fcn(h2, omega)

        np.testing.assert_array_almost_equal(
            (bdalg.series(f1, f2)).frequency_response(chkpts)[0],
            (bdalg.series(h1, h2)).frequency_response(chkpts)[0])

        np.testing.assert_array_almost_equal(
            (bdalg.parallel(f1, f2)).frequency_response(chkpts)[0],
            (bdalg.parallel(h1, h2)).frequency_response(chkpts)[0])

        np.testing.assert_array_almost_equal(
            (bdalg.feedback(f1, f2)).frequency_response(chkpts)[0],
            (bdalg.feedback(h1, h2)).frequency_response(chkpts)[0])

        np.testing.assert_array_almost_equal(
            (bdalg.negate(f1)).frequency_response(chkpts)[0],
            (bdalg.negate(h1)).frequency_response(chkpts)[0])

#       append() and connect() not implemented for FRD objects
#        np.testing.assert_array_almost_equal(
#            (bdalg.append(f1, f2)).frequency_response(chkpts)[0],
#            (bdalg.append(h1, h2)).frequency_response(chkpts)[0])
#
#        f3 = bdalg.append(f1, f2, f2)
#        h3 = bdalg.append(h1, h2, h2)
#        Q = np.mat([ [1, 2], [2, -1] ])
#        np.testing.assert_array_almost_equal(
#           (bdalg.connect(f3, Q, [2], [1])).frequency_response(chkpts)[0],
#            (bdalg.connect(h3, Q, [2], [1])).frequency_response(chkpts)[0])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testFeedback(self, frd_fcn):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(h1, omega)
        np.testing.assert_array_almost_equal(
            f1.feedback(1).frequency_response(chkpts)[0],
            h1.feedback(1).frequency_response(chkpts)[0])

        # Make sure default argument also works
        np.testing.assert_array_almost_equal(
            f1.feedback().frequency_response(chkpts)[0],
            h1.feedback().frequency_response(chkpts)[0])

    def testAppendSiso(self):
        # Create frequency responses
        d1 = np.array([1 + 2j, 1 - 2j, 1 + 4j, 1 - 4j, 1 + 6j, 1 - 6j])
        d2 = d1 + 2
        d3 = d1 - 1j
        w = np.arange(d1.shape[-1])
        frd1 = FrequencyResponseData(d1, w)
        frd2 = FrequencyResponseData(d2, w)
        frd3 = FrequencyResponseData(d3, w)
        # Create appended frequency responses
        d_app_1 = np.zeros((2, 2, d1.shape[-1]), dtype=complex)
        d_app_1[0, 0, :] = d1
        d_app_1[1, 1, :] = d2
        d_app_2 = np.zeros((3, 3, d1.shape[-1]), dtype=complex)
        d_app_2[0, 0, :] = d1
        d_app_2[1, 1, :] = d2
        d_app_2[2, 2, :] = d3
        # Test appending two FRDs
        frd_app_1 = frd1.append(frd2)
        np.testing.assert_allclose(d_app_1, frd_app_1.frdata)
        # Test appending three FRDs
        frd_app_2 = frd1.append(frd2).append(frd3)
        np.testing.assert_allclose(d_app_2, frd_app_2.frdata)

    def testAppendMimo(self):
        # Create frequency responses
        rng = np.random.default_rng(1234)
        n = 100
        w = np.arange(n)
        d1 = rng.uniform(size=(2, 2, n)) + 1j * rng.uniform(size=(2, 2, n))
        d2 = rng.uniform(size=(3, 1, n)) + 1j * rng.uniform(size=(3, 1, n))
        d3 = rng.uniform(size=(1, 2, n)) + 1j * rng.uniform(size=(1, 2, n))
        frd1 = FrequencyResponseData(d1, w)
        frd2 = FrequencyResponseData(d2, w)
        frd3 = FrequencyResponseData(d3, w)
        # Create appended frequency responses
        d_app_1 = np.zeros((5, 3, d1.shape[-1]), dtype=complex)
        d_app_1[:2, :2, :] = d1
        d_app_1[2:, 2:, :] = d2
        d_app_2 = np.zeros((6, 5, d1.shape[-1]), dtype=complex)
        d_app_2[:2, :2, :] = d1
        d_app_2[2:5, 2:3, :] = d2
        d_app_2[5:, 3:, :] = d3
        # Test appending two FRDs
        frd_app_1 = frd1.append(frd2)
        np.testing.assert_allclose(d_app_1, frd_app_1.frdata)
        # Test appending three FRDs
        frd_app_2 = frd1.append(frd2).append(frd3)
        np.testing.assert_allclose(d_app_2, frd_app_2.frdata)

    def testAuto(self):
        omega = np.logspace(-1, 2, 10)
        f1 = _convert_to_frd(1, omega)
        f2 = _convert_to_frd(np.array([[1, 0], [0.1, -1]]), omega)
        f2 = _convert_to_frd([[1, 0], [0.1, -1]], omega)
        f1, f2  # reference to avoid pyflakes error

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testNyquist(self, frd_fcn):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 40)
        f1 = frd_fcn(h1, omega, smooth=True)
        freqplot.nyquist(f1, np.logspace(-1, 2, 100))
        # plt.savefig('/dev/null', format='svg')
        plt.figure(2)
        freqplot.nyquist(f1, f1.omega)
        plt.figure(3)
        freqplot.nyquist(f1)
        # plt.savefig('/dev/null', format='svg')

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testMIMO(self, frd_fcn):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(sys, omega)
        np.testing.assert_array_almost_equal(
            sys.frequency_response(chkpts)[0],
            f1.frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            sys.frequency_response(chkpts)[1],
            f1.frequency_response(chkpts)[1])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testMIMOfb(self, frd_fcn):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(sys, omega).feedback([[0.1, 0.3], [0.0, 1.0]])
        f2 = frd_fcn(sys.feedback([[0.1, 0.3], [0.0, 1.0]]), omega)
        np.testing.assert_array_almost_equal(
            f1.frequency_response(chkpts)[0],
            f2.frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            f1.frequency_response(chkpts)[1],
            f2.frequency_response(chkpts)[1])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testMIMOfb2(self, frd_fcn):
        sys = StateSpace(np.array([[-2.0, 0, 0],
                                   [0, -1, 1],
                                   [0, 0, -3]]),
                         np.array([[1.0, 0], [0, 0], [0, 1]]),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        K = np.array([[1, 0.3, 0], [0.1, 0, 0]])
        f1 = frd_fcn(sys, omega).feedback(K)
        f2 = frd_fcn(sys.feedback(K), omega)
        np.testing.assert_array_almost_equal(
            f1.frequency_response(chkpts)[0],
            f2.frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            f1.frequency_response(chkpts)[1],
            f2.frequency_response(chkpts)[1])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testMIMOMult(self, frd_fcn):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(sys, omega)
        f2 = frd_fcn(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response(chkpts)[0],
            (sys*sys).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response(chkpts)[1],
            (sys*sys).frequency_response(chkpts)[1])

    @pytest.mark.parametrize(
        "frd_fcn", [ct.frd, ct.FRD, ct.FrequencyResponseData])
    def testMIMOSmooth(self, frd_fcn):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]],
                         [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                         [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        sys2 = np.array([[1, 0, 0], [0, 1, 0]]) * sys
        omega = np.logspace(-1, 2, 10)
        chkpts = omega[::3]
        f1 = frd_fcn(sys, omega, smooth=True)
        f2 = frd_fcn(sys2, omega, smooth=True)
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response(chkpts)[0],
            (sys*sys2).frequency_response(chkpts)[0])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response(chkpts)[1],
            (sys*sys2).frequency_response(chkpts)[1])
        np.testing.assert_array_almost_equal(
            (f1*f2).frequency_response(chkpts)[2],
            (sys*sys2).frequency_response(chkpts)[2])

    def testAgainstOctave(self):
        # with data from octave:
        # sys = ss([-2 0 0; 0 -1 1; 0 0 -3],
        #  [1 0; 0 0; 0 1], eye(3), zeros(3,2))
        # bfr = frd(bsys, [1])
        sys = StateSpace(np.array([[-2.0, 0, 0], [0, -1, 1], [0, 0, -3]]),
                         np.array([[1.0, 0], [0, 0], [0, 1]]),
                         np.eye(3), np.zeros((3, 2)))
        omega = np.logspace(-1, 2, 10)
        f1 = frd(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1.frequency_response([1.0])[0] *
             np.exp(1j * f1.frequency_response([1.0])[1])).reshape(3, 2),
            np.array([[0.4 - 0.2j, 0], [0, 0.1 - 0.2j], [0, 0.3 - 0.1j]]))

    def test_string_representation(self, capsys):
        sys = frd([1, 2, 3], [4, 5, 6])
        print(sys)              # Just print without checking

    def test_frequency_mismatch(self, recwarn):
        # recwarn: there may be a warning before the error!
        # Overlapping but non-equal frequency ranges
        sys1 = frd([1, 2, 3], [4, 5, 6])
        sys2 = frd([2, 3, 4], [5, 6, 7])
        with pytest.raises(NotImplementedError):
            sys1 + sys2

        # One frequency range is a subset of another
        sys1 = frd([1, 2, 3], [4, 5, 6])
        sys2 = frd([2, 3], [4, 5])
        with pytest.raises(NotImplementedError):
            sys1 + sys2

    def test_size_mismatch(self):
        sys1 = frd(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))

        # Different number of inputs
        sys2 = frd(ct.rss(3, 1, 2), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            sys1 + sys2

        # Different number of outputs
        sys2 = frd(ct.rss(3, 2, 1), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            sys1 + sys2

        # Inputs and outputs don't match
        with pytest.raises(ValueError):
            sys2 * sys1

        # Feedback mismatch
        with pytest.raises(ValueError):
            ct.feedback(sys2, sys1)

    def test_operator_conversion(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = frd(sys_tf, np.logspace(-1, 1, 10))
        frd_2 = frd(2 * np.ones(10), np.logspace(-1, 1, 10))

        # Make sure that we can add, multiply, and feedback constants
        sys_add = frd_tf + 2
        chk_add = frd_tf + frd_2
        np.testing.assert_array_almost_equal(sys_add.omega, chk_add.omega)
        np.testing.assert_array_almost_equal(sys_add.frdata, chk_add.frdata)

        sys_radd = 2 + frd_tf
        chk_radd = frd_2 + frd_tf
        np.testing.assert_array_almost_equal(sys_radd.omega, chk_radd.omega)
        np.testing.assert_array_almost_equal(sys_radd.frdata, chk_radd.frdata)

        sys_sub = frd_tf - 2
        chk_sub = frd_tf - frd_2
        np.testing.assert_array_almost_equal(sys_sub.omega, chk_sub.omega)
        np.testing.assert_array_almost_equal(sys_sub.frdata, chk_sub.frdata)

        sys_rsub = 2 - frd_tf
        chk_rsub = frd_2 - frd_tf
        np.testing.assert_array_almost_equal(sys_rsub.omega, chk_rsub.omega)
        np.testing.assert_array_almost_equal(sys_rsub.frdata, chk_rsub.frdata)

        sys_mul = frd_tf * 2
        chk_mul = frd_tf * frd_2
        np.testing.assert_array_almost_equal(sys_mul.omega, chk_mul.omega)
        np.testing.assert_array_almost_equal(sys_mul.frdata, chk_mul.frdata)

        sys_rmul = 2 * frd_tf
        chk_rmul = frd_2 * frd_tf
        np.testing.assert_array_almost_equal(sys_rmul.omega, chk_rmul.omega)
        np.testing.assert_array_almost_equal(sys_rmul.frdata, chk_rmul.frdata)

        sys_rdiv = 2 / frd_tf
        chk_rdiv = frd_2 / frd_tf
        np.testing.assert_array_almost_equal(sys_rdiv.omega, chk_rdiv.omega)
        np.testing.assert_array_almost_equal(sys_rdiv.frdata, chk_rdiv.frdata)

        sys_pow = frd_tf**2
        chk_pow = frd(sys_tf**2, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(sys_pow.omega, chk_pow.omega)
        np.testing.assert_array_almost_equal(sys_pow.frdata, chk_pow.frdata)

        sys_pow = frd_tf**-2
        chk_pow = frd(sys_tf**-2, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(sys_pow.omega, chk_pow.omega)
        np.testing.assert_array_almost_equal(sys_pow.frdata, chk_pow.frdata)

        # Assertion error if we try to raise to a non-integer power
        with pytest.raises(ValueError):
            frd_tf**0.5

        # Selected testing on transfer function conversion
        sys_add = frd_2 + sys_tf
        chk_add = frd_2 + frd_tf
        np.testing.assert_array_almost_equal(sys_add.omega, chk_add.omega)
        np.testing.assert_array_almost_equal(sys_add.frdata, chk_add.frdata)

        # Test broadcasting with SISO system
        sys_tf_mimo = TransferFunction([1], [1, 0]) * np.eye(2)
        frd_tf_mimo = frd(sys_tf_mimo, np.logspace(-1, 1, 10))
        result = FrequencyResponseData.__rmul__(frd_tf, frd_tf_mimo)
        expected = frd(sys_tf_mimo * sys_tf, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

        # Input/output mismatch size mismatch in rmul
        sys1 = frd(ct.rss(2, 2, 2), np.logspace(-1, 1, 10))
        sys2 = frd(ct.rss(3, 3, 3), np.logspace(-1, 1, 10))
        with pytest.raises(ValueError):
            FrequencyResponseData.__rmul__(sys2, sys1)

        # Make sure conversion of something random generates exception
        with pytest.raises(TypeError):
            FrequencyResponseData.__add__(frd_tf, 'string')

    def test_add_sub_mimo_siso(self):
        omega = np.logspace(-1, 1, 10)
        sys_mimo = frd(ct.rss(2, 2, 2), omega)
        sys_siso = frd(ct.rss(2, 1, 1), omega)

        for op, expected_fresp in [
            (FrequencyResponseData.__add__, sys_mimo.frdata + sys_siso.frdata),
            (FrequencyResponseData.__radd__, sys_mimo.frdata + sys_siso.frdata),
            (FrequencyResponseData.__sub__, sys_mimo.frdata - sys_siso.frdata),
            (FrequencyResponseData.__rsub__, -sys_mimo.frdata + sys_siso.frdata),
        ]:
            result = op(sys_mimo, sys_siso)
            np.testing.assert_array_almost_equal(omega, result.omega)
            np.testing.assert_array_almost_equal(expected_fresp, result.frdata)

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (
                TransferFunction([2], [1, 0]),
                TransferFunction(
                    [
                        [[2], [1]],
                        [[-1], [4]],
                    ],
                    [
                        [[10, 1], [20, 1]],
                        [[20, 1], [30, 1]],
                    ],
                ),
                TransferFunction(
                    [
                        [[4], [2]],
                        [[-2], [8]],
                    ],
                    [
                        [[10, 1, 0], [20, 1, 0]],
                        [[20, 1, 0], [30, 1, 0]],
                    ],
                ),
            ),
            (
                TransferFunction(
                    [
                        [[2], [1]],
                        [[-1], [4]],
                    ],
                    [
                        [[10, 1], [20, 1]],
                        [[20, 1], [30, 1]],
                    ],
                ),
                TransferFunction([2], [1, 0]),
                TransferFunction(
                    [
                        [[4], [2]],
                        [[-2], [8]],
                    ],
                    [
                        [[10, 1, 0], [20, 1, 0]],
                        [[20, 1, 0], [30, 1, 0]],
                    ],
                ),
            ),
            (
                TransferFunction([2], [1, 0]),
                np.eye(3),
                TransferFunction(
                    [
                        [[2], [0], [0]],
                        [[0], [2], [0]],
                        [[0], [0], [2]],
                    ],
                    [
                        [[1, 0], [1], [1]],
                        [[1], [1, 0], [1]],
                        [[1], [1], [1, 0]],
                    ],
                ),
            ),
        ]
    )
    def test_mul_mimo_siso(self, left, right, expected):
        result = frd(left, np.logspace(-1, 1, 10)).__mul__(right)
        expected_frd = frd(expected, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(expected_frd.omega, result.omega)
        np.testing.assert_array_almost_equal(expected_frd.frdata, result.frdata)

    @slycotonly
    def test_truediv_mimo_siso(self):
        omega = np.logspace(-1, 1, 10)
        tf_mimo = TransferFunction([1], [1, 0]) * np.eye(2)
        frd_mimo = frd(tf_mimo, omega)
        tf_siso = TransferFunction([1], [1, 1])
        frd_siso = frd(tf_siso, omega)
        expected = frd(tf_mimo.__truediv__(tf_siso), omega)
        ss_siso = ct.tf2ss(tf_siso)

        # Test division of MIMO FRD by SISO FRD
        result = frd_mimo.__truediv__(frd_siso)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

        # Test division of MIMO FRD by SISO TF
        result = frd_mimo.__truediv__(tf_siso)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

        # Test division of MIMO FRD by SISO TF
        result = frd_mimo.__truediv__(ss_siso)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

    @slycotonly
    def test_rtruediv_mimo_siso(self):
        omega = np.logspace(-1, 1, 10)
        tf_mimo = TransferFunction([1], [1, 0]) * np.eye(2)
        frd_mimo = frd(tf_mimo, omega)
        ss_mimo = ct.tf2ss(tf_mimo)
        tf_siso = TransferFunction([1], [1, 1])
        frd_siso = frd(tf_siso, omega)
        expected = frd(tf_siso.__rtruediv__(tf_mimo), omega)

        # Test division of MIMO FRD by SISO FRD
        result = frd_siso.__rtruediv__(frd_mimo)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

        # Test division of MIMO TF by SISO FRD
        result = frd_siso.__rtruediv__(tf_mimo)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)

        # Test division of MIMO SS by SISO FRD
        result = frd_siso.__rtruediv__(ss_mimo)
        np.testing.assert_array_almost_equal(expected.omega, result.omega)
        np.testing.assert_array_almost_equal(expected.frdata, result.frdata)


    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (
                TransferFunction([2], [1, 0]),
                TransferFunction(
                    [
                        [[2], [1]],
                        [[-1], [4]],
                    ],
                    [
                        [[10, 1], [20, 1]],
                        [[20, 1], [30, 1]],
                    ],
                ),
                TransferFunction(
                    [
                        [[4], [2]],
                        [[-2], [8]],
                    ],
                    [
                        [[10, 1, 0], [20, 1, 0]],
                        [[20, 1, 0], [30, 1, 0]],
                    ],
                ),
            ),
            (
                TransferFunction(
                    [
                        [[2], [1]],
                        [[-1], [4]],
                    ],
                    [
                        [[10, 1], [20, 1]],
                        [[20, 1], [30, 1]],
                    ],
                ),
                TransferFunction([2], [1, 0]),
                TransferFunction(
                    [
                        [[4], [2]],
                        [[-2], [8]],
                    ],
                    [
                        [[10, 1, 0], [20, 1, 0]],
                        [[20, 1, 0], [30, 1, 0]],
                    ],
                ),
            ),
            (
                np.eye(3),
                TransferFunction([2], [1, 0]),
                TransferFunction(
                    [
                        [[2], [0], [0]],
                        [[0], [2], [0]],
                        [[0], [0], [2]],
                    ],
                    [
                        [[1, 0], [1], [1]],
                        [[1], [1, 0], [1]],
                        [[1], [1], [1, 0]],
                    ],
                ),
            ),
        ]
    )
    def test_rmul_mimo_siso(self, left, right, expected):
        result = frd(right, np.logspace(-1, 1, 10)).__rmul__(left)
        expected_frd = frd(expected, np.logspace(-1, 1, 10))
        np.testing.assert_array_almost_equal(expected_frd.omega, result.omega)
        np.testing.assert_array_almost_equal(expected_frd.frdata, result.frdata)

    def test_eval(self):
        sys_tf = ct.tf([1], [1, 2, 1])
        frd_tf = frd(sys_tf, np.logspace(-1, 1, 3))
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
        frd_tf = frd(sys_tf, np.logspace(-1, 1, 3))
        with pytest.warns(FutureWarning):
            frd_tf.freqresp(1.)

        with pytest.warns(FutureWarning, match="use complex"):
            np.testing.assert_equal(frd_tf.response, frd_tf.complex)

        with pytest.warns(FutureWarning, match="use frdata"):
            np.testing.assert_equal(frd_tf.fresp, frd_tf.frdata)

    def test_repr_str(self):
        # repr printing
        array = np.array
        sys0 = ct.frd(
            [1.0, 0.9+0.1j, 0.1+2j, 0.05+3j],
            [0.1, 1.0, 10.0, 100.0], name='sys0')
        sys1 = ct.frd(
            sys0.frdata, sys0.omega, smooth=True, name='sys1')
        ref_common = "FrequencyResponseData(\n" \
            "array([[[1.  +0.j , 0.9 +0.1j, 0.1 +2.j , 0.05+3.j ]]]),\n" \
            "array([  0.1,   1. ,  10. , 100. ]),"
        ref0 = ref_common + "\nname='sys0', outputs=1, inputs=1)"
        ref1 = ref_common + " smooth=True," + \
            "\nname='sys1', outputs=1, inputs=1)"
        sysm = ct.frd(
            np.matmul(array([[1], [2]]), sys0.frdata), sys0.omega, name='sysm')

        assert ct.iosys_repr(sys0, format='eval') == ref0
        assert ct.iosys_repr(sys1, format='eval') == ref1

        sys0r = eval(ct.iosys_repr(sys0, format='eval'))
        np.testing.assert_array_almost_equal(sys0r.frdata, sys0.frdata)
        np.testing.assert_array_almost_equal(sys0r.omega, sys0.omega)

        sys1r = eval(ct.iosys_repr(sys1, format='eval'))
        np.testing.assert_array_almost_equal(sys1r.frdata, sys1.frdata)
        np.testing.assert_array_almost_equal(sys1r.omega, sys1.omega)
        assert(sys1._ifunc is not None)

        refs = """<FrequencyResponseData>: {sysname}
Inputs (1): ['u[0]']
Outputs (1): ['y[0]']

Freq [rad/s]  Response
------------  ---------------------
       0.100           1        +0j
       1.000         0.9      +0.1j
      10.000         0.1        +2j
     100.000        0.05        +3j"""
        assert str(sys0) == refs.format(sysname='sys0')
        assert str(sys1) == refs.format(sysname='sys1')

        # print multi-input system
        refm = """<FrequencyResponseData>: sysm
Inputs (2): ['u[0]', 'u[1]']
Outputs (1): ['y[0]']

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
            FrequencyResponseData(h, omega, unknown=None)
        with pytest.raises(TypeError, match="unrecognized keyword"):
            ct.frd(h, omega, unknown=None)


def test_named_signals():
    ct.iosys.InputOutputSystem._idCounter = 0
    h1 = TransferFunction([1], [1, 2, 2])
    h2 = TransferFunction([1], [0.1, 1])
    omega = np.logspace(-1, 2, 10)
    f1 = frd(h1, omega)
    f2 = frd(h2, omega)

    # Make sure that systems were properly named
    assert f1.name == 'sys[2]'
    assert f2.name == 'sys[3]'
    assert f1.ninputs == 1
    assert f1.input_labels == ['u[0]']
    assert f1.noutputs == 1
    assert f1.output_labels == ['y[0]']

    # Change names
    f1 = frd(h1, omega, name='mysys', inputs='u0', outputs='y0')
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
    resp = frd(h1, omega)

    # Convert to pandas
    df = resp.to_pandas()

    # Check to make sure the data make senses
    np.testing.assert_equal(df['omega'], resp.omega)
    np.testing.assert_equal(df['H_{y[0], u[0]}'], resp.frdata[0, 0])


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


def test_signal_labels():
    # Create a system response for a SISO system
    sys = ct.rss(4, 1, 1)
    fresp = ct.frequency_response(sys)

    # Make sure access via strings works
    np.testing.assert_equal(
        fresp.magnitude['y[0]'], fresp.magnitude)
    np.testing.assert_equal(
        fresp.phase['y[0]'], fresp.phase)

    # Make sure errors are generated if key is unknown
    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        fresp.magnitude['bad']

    # Create a system response for a MIMO system
    sys = ct.rss(4, 2, 2)
    fresp = ct.frequency_response(sys)

    # Make sure access via strings works
    np.testing.assert_equal(
        fresp.magnitude['y[0]', 'u[1]'],
        fresp.magnitude[0, 1])
    np.testing.assert_equal(
        fresp.phase['y[0]', 'u[1]'],
        fresp.phase[0, 1])
    np.testing.assert_equal(
        fresp.complex['y[0]', 'u[1]'],
        fresp.complex[0, 1])

    # Make sure access via lists of strings works
    np.testing.assert_equal(
        fresp.complex[['y[1]', 'y[0]'], 'u[0]'],
        fresp.complex[[1, 0], 0])

    # Make sure errors are generated if key is unknown
    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        fresp.magnitude['bad']

    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        fresp.complex[['y[1]', 'bad']]

    with pytest.raises(ValueError, match=r"unknown signal name 'y\[0\]'"):
        fresp.complex['y[1]', 'y[0]']         # second index = input name
