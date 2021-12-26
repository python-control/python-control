#!/usr/bin/env python

"""convert_test.py

Test state space and transfer function conversion.

Currently, this unit test script is not complete.  It converts several random
state spaces back and forth between state space and transfer function
representations.  Ideally, it should be able to assert that the conversion
outputs are correct.  This is not yet implemented.

Also, the conversion seems to enter an infinite loop once in a while.  The cause
of this is unknown.

"""


import numpy as np
import pytest

from control import rss, ss, ss2tf, tf, tf2ss
from control.statesp import _mimo2siso
from control.statefbk import ctrb, obsv
from control.freqplot import bode
from control.exception import slycot_check
from control.tests.conftest import slycotonly


# Set to True to print systems to the output.
verbose = False
# Maximum number of states to test + 1
maxStates = 4
# Maximum number of inputs and outputs to test + 1
# If slycot is not installed, just check SISO
maxIO = 5 if slycot_check() else 2


@pytest.fixture
def fixedseed(scope='module'):
    """Get consistent results"""
    np.random.seed(7)


class TestConvert:
    """Test state space and transfer function conversions."""

    def printSys(self, sys, ind):
        """Print system to the standard output."""
        print("sys%i:\n" % ind)
        print(sys)

    @pytest.mark.parametrize("states", range(1, maxStates))
    @pytest.mark.parametrize("inputs", range(1, maxIO))
    @pytest.mark.parametrize("outputs", range(1, maxIO))
    def testConvert(self, fixedseed, states, inputs, outputs):
        """Test state space to transfer function conversion.

        start with a random SS system and transform to TF then
        back to SS, check that the matrices are the same.
        """
        ssOriginal = rss(states, outputs, inputs)
        if verbose:
            self.printSys(ssOriginal, 1)

        # Make sure the system is not degenerate
        Cmat = ctrb(ssOriginal.A, ssOriginal.B)
        if (np.linalg.matrix_rank(Cmat) != states):
            pytest.skip("not reachable")
        Omat = obsv(ssOriginal.A, ssOriginal.C)
        if (np.linalg.matrix_rank(Omat) != states):
            pytest.skip("not observable")

        tfOriginal = tf(ssOriginal)
        if (verbose):
            self.printSys(tfOriginal, 2)

        ssTransformed = ss(tfOriginal)
        if (verbose):
            self.printSys(ssTransformed, 3)

        tfTransformed = tf(ssTransformed)
        if (verbose):
            self.printSys(tfTransformed, 4)

        # Check to see if the state space systems have same dim
        if (ssOriginal.nstates != ssTransformed.nstates) and verbose:
            print("WARNING: state space dimension mismatch: %d versus %d" %
                  (ssOriginal.nstates, ssTransformed.nstates))

        # Now make sure the frequency responses match
        # Since bode() only handles SISO, go through each I/O pair
        # For phase, take sine and cosine to avoid +/- 360 offset
        for inputNum in range(inputs):
            for outputNum in range(outputs):
                if (verbose):
                    print("Checking input %d, output %d"
                          % (inputNum, outputNum))
                ssorig_mag, ssorig_phase, ssorig_omega = \
                    bode(_mimo2siso(ssOriginal, inputNum, outputNum),
                         deg=False, plot=False)
                ssorig_real = ssorig_mag * np.cos(ssorig_phase)
                ssorig_imag = ssorig_mag * np.sin(ssorig_phase)

                #
                # Make sure TF has same frequency response
                #
                num = tfOriginal.num[outputNum][inputNum]
                den = tfOriginal.den[outputNum][inputNum]
                tforig = tf(num, den)

                tforig_mag, tforig_phase, tforig_omega = \
                    bode(tforig, ssorig_omega,
                         deg=False, plot=False)

                tforig_real = tforig_mag * np.cos(tforig_phase)
                tforig_imag = tforig_mag * np.sin(tforig_phase)
                np.testing.assert_array_almost_equal(
                    ssorig_real, tforig_real)
                np.testing.assert_array_almost_equal(
                    ssorig_imag, tforig_imag)

                #
                # Make sure xform'd SS has same frequency response
                #
                ssxfrm_mag, ssxfrm_phase, ssxfrm_omega = \
                    bode(_mimo2siso(ssTransformed,
                                    inputNum, outputNum),
                         ssorig_omega,
                         deg=False, plot=False)
                ssxfrm_real = ssxfrm_mag * np.cos(ssxfrm_phase)
                ssxfrm_imag = ssxfrm_mag * np.sin(ssxfrm_phase)
                np.testing.assert_array_almost_equal(
                    ssorig_real, ssxfrm_real, decimal=5)
                np.testing.assert_array_almost_equal(
                    ssorig_imag, ssxfrm_imag, decimal=5)

                # Make sure xform'd TF has same frequency response
                #
                num = tfTransformed.num[outputNum][inputNum]
                den = tfTransformed.den[outputNum][inputNum]
                tfxfrm = tf(num, den)
                tfxfrm_mag, tfxfrm_phase, tfxfrm_omega = \
                    bode(tfxfrm, ssorig_omega,
                         deg=False, plot=False)

                tfxfrm_real = tfxfrm_mag * np.cos(tfxfrm_phase)
                tfxfrm_imag = tfxfrm_mag * np.sin(tfxfrm_phase)
                np.testing.assert_array_almost_equal(
                    ssorig_real, tfxfrm_real, decimal=5)
                np.testing.assert_array_almost_equal(
                    ssorig_imag, tfxfrm_imag, decimal=5)

    def testConvertMIMO(self):
        """Test state space to transfer function conversion.

        Do a MIMO conversion and make sure that it is processed
        correctly both with and without slycot

        Example from issue gh-120, jgoppert
        """

        # Set up a 1x3 transfer function (should always work)
        tsys = tf([[[-235, 1.146e4],
                    [-235, 1.146E4],
                    [-235, 1.146E4, 0]]],
                  [[[1, 48.78, 0],
                    [1, 48.78, 0, 0],
                    [0.008, 1.39, 48.78]]])

        # Convert to state space and look for an error
        if (not slycot_check()):
            with pytest.raises(TypeError):
                tf2ss(tsys)
        else:
            ssys = tf2ss(tsys)
            assert ssys.B.shape[1] == 3
            assert ssys.C.shape[0] == 1

    def testTf2ssStaticSiso(self):
        """Regression: tf2ss for SISO static gain"""
        gsiso = tf2ss(tf(23, 46))
        assert 0 == gsiso.nstates
        assert 1 == gsiso.ninputs
        assert 1 == gsiso.noutputs
        np.testing.assert_allclose([[0.5]], gsiso.D)

    def testTf2ssStaticMimo(self):
        """Regression: tf2ss for MIMO static gain"""
        # 2x3 TFM
        gmimo = tf2ss(tf(
                [[ [23],   [3],  [5] ], [ [-1],  [0.125],  [101.3] ]],
                [[ [46], [0.1], [80] ], [  [2],   [-0.1],      [1] ]]))
        assert 0 == gmimo.nstates
        assert 3 == gmimo.ninputs
        assert 2 == gmimo.noutputs
        d = np.array([[0.5, 30, 0.0625], [-0.5, -1.25, 101.3]])
        np.testing.assert_allclose(d, gmimo.D)

    def testSs2tfStaticSiso(self):
        """Regression: ss2tf for SISO static gain"""
        gsiso = ss2tf(ss([], [], [], 0.5))
        np.testing.assert_allclose([[[0.5]]], gsiso.num)
        np.testing.assert_allclose([[[1.]]], gsiso.den)

    def testSs2tfStaticMimo(self):
        """Regression: ss2tf for MIMO static gain"""
        # 2x3 TFM
        a = []
        b = []
        c = []
        d = np.array([[0.5, 30, 0.0625], [-0.5, -1.25, 101.3]])
        gtf = ss2tf(ss(a, b, c, d))

        # we need a 3x2x1 array to compare with gtf.num
        numref = d[..., np.newaxis]
        np.testing.assert_allclose(numref,
                                   np.array(gtf.num) / np.array(gtf.den))

    @slycotonly
    def testTf2SsDuplicatePoles(self):
        """Tests for 'too few poles for MIMO tf gh-111'"""
        num = [[[1], [0]],
               [[0], [1]]]
        den = [[[1, 0], [1]],
               [[1], [1, 0]]]
        g = tf(num, den)
        s = ss(g)
        np.testing.assert_allclose(g.pole(), s.pole())

    @slycotonly
    def test_tf2ss_robustness(self):
        """Unit test to make sure that tf2ss is working correctly. gh-240"""
        num =  [ [[0], [1]],           [[1],   [0]] ]
        den1 = [ [[1], [1,1]],         [[1,4], [1]] ]
        sys1tf = tf(num, den1)
        sys1ss = tf2ss(sys1tf)

        # slight perturbation
        den2 = [ [[1], [1e-10, 1, 1]], [[1,4], [1]] ]
        sys2tf = tf(num, den2)
        sys2ss = tf2ss(sys2tf)

        # Make sure that the poles match for StateSpace and TransferFunction
        np.testing.assert_array_almost_equal(np.sort(sys1tf.pole()),
                                             np.sort(sys1ss.pole()))
        np.testing.assert_array_almost_equal(np.sort(sys2tf.pole()),
                                             np.sort(sys2ss.pole()))

    def test_tf2ss_nonproper(self):
        """Unit tests for non-proper transfer functions"""
        # Easy case: input 2 to output 1 is 's'
        num =  [ [[0], [1, 0]],  [[1],   [0]] ]
        den1 = [ [[1], [1]],     [[1,4], [1]] ]
        with pytest.raises(ValueError):
            tf2ss(tf(num, den1))

        # Trickier case (make sure that leading zeros in den are handled)
        num =  [ [[0], [1, 0]],  [[1],   [0]] ]
        den1 = [ [[1], [0, 1]],  [[1,4], [1]] ]
        with pytest.raises(ValueError):
            tf2ss(tf(num, den1))
