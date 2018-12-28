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

from __future__ import print_function
import unittest
import numpy as np
from control import matlab
from control.statesp import _mimo2siso
from control.statefbk import ctrb, obsv
from control.freqplot import bode
from control.matlab import tf
from control.exception import slycot_check

class TestConvert(unittest.TestCase):
    """Test state space and transfer function conversions."""

    def setUp(self):
        """Set up testing parameters."""

        # Number of times to run each of the randomized tests.
        self.numTests = 1  # almost guarantees failure
        # Maximum number of states to test + 1
        self.maxStates = 4
        # Maximum number of inputs and outputs to test + 1
        # If slycot is not installed, just check SISO
        self.maxIO = 5 if slycot_check() else 2
        # Set to True to print systems to the output.
        self.debug = False
        # get consistent results
        np.random.seed(7)

    def printSys(self, sys, ind):
        """Print system to the standard output."""

        if self.debug:
            print("sys%i:\n" % ind)
            print(sys)

    def testConvert(self):
        """Test state space to transfer function conversion."""
        verbose = self.debug

        # print __doc__

        # Machine precision for floats.
        # eps = np.finfo(float).eps

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    # start with a random SS system and transform to TF then
                    # back to SS, check that the matrices are the same.
                    ssOriginal = matlab.rss(states, outputs, inputs)
                    if (verbose):
                        self.printSys(ssOriginal, 1)

                    # Make sure the system is not degenerate
                    Cmat = ctrb(ssOriginal.A, ssOriginal.B)
                    if (np.linalg.matrix_rank(Cmat) != states):
                        if (verbose):
                            print("  skipping (not reachable)")
                        continue
                    Omat = obsv(ssOriginal.A, ssOriginal.C)
                    if (np.linalg.matrix_rank(Omat) != states):
                        if (verbose):
                            print("  skipping (not observable)")
                        continue

                    tfOriginal = matlab.tf(ssOriginal)
                    if (verbose):
                        self.printSys(tfOriginal, 2)

                    ssTransformed = matlab.ss(tfOriginal)
                    if (verbose):
                        self.printSys(ssTransformed, 3)

                    tfTransformed = matlab.tf(ssTransformed)
                    if (verbose):
                        self.printSys(tfTransformed, 4)

                    # Check to see if the state space systems have same dim
                    if (ssOriginal.states != ssTransformed.states):
                        print("WARNING: state space dimension mismatch: " + \
                            "%d versus %d" % \
                            (ssOriginal.states, ssTransformed.states))

                    # Now make sure the frequency responses match
                    # Since bode() only handles SISO, go through each I/O pair
                    # For phase, take sine and cosine to avoid +/- 360 offset
                    for inputNum in range(inputs):
                        for outputNum in range(outputs):
                            if (verbose):
                                print("Checking input %d, output %d" \
                                    % (inputNum, outputNum))
                            ssorig_mag, ssorig_phase, ssorig_omega = \
                                bode(_mimo2siso(ssOriginal, \
                                                        inputNum, outputNum), \
                                                 deg=False, Plot=False)
                            ssorig_real = ssorig_mag * np.cos(ssorig_phase)
                            ssorig_imag = ssorig_mag * np.sin(ssorig_phase)

                            #
                            # Make sure TF has same frequency response
                            #
                            num = tfOriginal.num[outputNum][inputNum]
                            den = tfOriginal.den[outputNum][inputNum]
                            tforig = tf(num, den)

                            tforig_mag, tforig_phase, tforig_omega = \
                                bode(tforig, ssorig_omega, \
                                                 deg=False, Plot=False)

                            tforig_real = tforig_mag * np.cos(tforig_phase)
                            tforig_imag = tforig_mag * np.sin(tforig_phase)
                            np.testing.assert_array_almost_equal( \
                                ssorig_real, tforig_real)
                            np.testing.assert_array_almost_equal( \
                                ssorig_imag, tforig_imag)

                            #
                            # Make sure xform'd SS has same frequency response
                            #
                            ssxfrm_mag, ssxfrm_phase, ssxfrm_omega = \
                                bode(_mimo2siso(ssTransformed, \
                                                        inputNum, outputNum), \
                                                 ssorig_omega, \
                                                 deg=False, Plot=False)
                            ssxfrm_real = ssxfrm_mag * np.cos(ssxfrm_phase)
                            ssxfrm_imag = ssxfrm_mag * np.sin(ssxfrm_phase)
                            np.testing.assert_array_almost_equal( \
                            ssorig_real, ssxfrm_real)
                            np.testing.assert_array_almost_equal( \
                            ssorig_imag, ssxfrm_imag)
                            #
                            # Make sure xform'd TF has same frequency response
                            #
                            num = tfTransformed.num[outputNum][inputNum]
                            den = tfTransformed.den[outputNum][inputNum]
                            tfxfrm = tf(num, den)
                            tfxfrm_mag, tfxfrm_phase, tfxfrm_omega = \
                                bode(tfxfrm, ssorig_omega, \
                                                 deg=False, Plot=False)

                            tfxfrm_real = tfxfrm_mag * np.cos(tfxfrm_phase)
                            tfxfrm_imag = tfxfrm_mag * np.sin(tfxfrm_phase)
                            np.testing.assert_array_almost_equal( \
                                ssorig_real, tfxfrm_real)
                            np.testing.assert_array_almost_equal( \
                                ssorig_imag, tfxfrm_imag)

    def testConvertMIMO(self):
        """Test state space to transfer function conversion."""
        verbose = self.debug

        # Do a MIMO conversation and make sure that it is processed
        # correctly both with and without slycot
        #
        # Example from issue #120, jgoppert
        import control

        # Set up a transfer function (should always work)
        tfcn = control.tf([[[-235, 1.146e4],
                            [-235, 1.146E4],
                            [-235, 1.146E4, 0]]],
                          [[[1, 48.78, 0],
                            [1, 48.78, 0, 0],
                            [0.008, 1.39, 48.78]]])

        # Convert to state space and look for an error
        if (not slycot_check()):
            self.assertRaises(TypeError, control.tf2ss, tfcn)

    def testTf2ssStaticSiso(self):
        """Regression: tf2ss for SISO static gain"""
        import control
        gsiso = control.tf2ss(control.tf(23, 46))
        self.assertEqual(0, gsiso.states)
        self.assertEqual(1, gsiso.inputs)
        self.assertEqual(1, gsiso.outputs)
        # in all cases ratios are exactly representable, so assert_array_equal is fine
        np.testing.assert_array_equal([[0.5]], gsiso.D)

    def testTf2ssStaticMimo(self):
        """Regression: tf2ss for MIMO static gain"""
        import control
        # 2x3 TFM
        gmimo = control.tf2ss(control.tf(
                [[ [23],   [3],  [5] ], [ [-1],  [0.125],  [101.3] ]],
                [[ [46], [0.1], [80] ], [  [2],   [-0.1],      [1] ]]))
        self.assertEqual(0, gmimo.states)
        self.assertEqual(3, gmimo.inputs)
        self.assertEqual(2, gmimo.outputs)
        d = np.matrix([[0.5, 30, 0.0625], [-0.5, -1.25, 101.3]])
        np.testing.assert_array_equal(d, gmimo.D)

    def testSs2tfStaticSiso(self):
        """Regression: ss2tf for SISO static gain"""
        import control
        gsiso = control.ss2tf(control.ss([], [], [], 0.5))
        np.testing.assert_array_equal([[[0.5]]], gsiso.num)
        np.testing.assert_array_equal([[[1.]]], gsiso.den)

    def testSs2tfStaticMimo(self):
        """Regression: ss2tf for MIMO static gain"""
        import control
        # 2x3 TFM
        a = []
        b = []
        c = []
        d = np.matrix([[0.5, 30, 0.0625], [-0.5, -1.25, 101.3]])
        gtf = control.ss2tf(control.ss(a,b,c,d))

        # we need a 3x2x1 array to compare with gtf.num
        # np.testing.assert_array_equal doesn't seem to like a matrices
        # with an extra dimension, so convert to ndarray
        numref = np.asarray(d)[...,np.newaxis]
        np.testing.assert_array_equal(numref, np.array(gtf.num) / np.array(gtf.den))

    def testTf2SsDuplicatePoles(self):
        """Tests for "too few poles for MIMO tf #111" """
        import control
        try:
            import slycot
            num = [ [ [1], [0] ],
                   [ [0], [1] ] ]

            den = [ [ [1,0], [1] ],
                [ [1],   [1,0] ] ]
            g = control.tf(num, den)
            s = control.ss(g)
            np.testing.assert_array_equal(g.pole(), s.pole())
        except ImportError:
            print("Slycot not present, skipping")

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_tf2ss_robustness(self):
        """Unit test to make sure that tf2ss is working correctly.
         Source: https://github.com/python-control/python-control/issues/240
        """
        import control
        
        num =  [ [[0], [1]],           [[1],   [0]] ]
        den1 = [ [[1], [1,1]],         [[1,4], [1]] ]
        sys1tf = control.tf(num, den1)
        sys1ss = control.tf2ss(sys1tf)

        # slight perturbation
        den2 = [ [[1], [1e-10, 1, 1]], [[1,4], [1]] ]
        sys2tf = control.tf(num, den2)
        sys2ss = control.tf2ss(sys2tf)

        # Make sure that the poles match for StateSpace and TransferFunction
        np.testing.assert_array_almost_equal(np.sort(sys1tf.pole()),
                                             np.sort(sys1ss.pole()))
        np.testing.assert_array_almost_equal(np.sort(sys2tf.pole()),
                                             np.sort(sys2ss.pole()))

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestConvert)

if __name__ == "__main__":
    unittest.main()
