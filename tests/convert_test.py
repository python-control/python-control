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
import control
import control.matlab as matlab

class TestConvert(unittest.TestCase):
    """Test state space and transfer function conversions."""

    def setUp(self):
        """Set up testing parameters."""

        # Number of times to run each of the randomized tests.
        self.numTests = 1 #almost guarantees failure
        # Maximum number of states to test + 1
        self.maxStates = 4
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5
        # Set to True to print systems to the output.
        self.debug = False
        # get consistent results
        np.random.seed(9)

    def printSys(self, sys, ind):
        """Print system to the standard output."""

        if self.debug:
            print("sys%i:\n" % ind)
            print(sys)

    def testConvert(self):
        """Test state space to transfer function conversion."""
        verbose = self.debug
        from control.statesp import _mimo2siso
        
        #print __doc__

        # Machine precision for floats.
        eps = np.finfo(float).eps

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    # start with a random SS system and transform to TF then
                    # back to SS, check that the matrices are the same.
                    ssOriginal = matlab.rss(states, outputs, inputs)
                    if (verbose):
                        self.printSys(ssOriginal, 1)

                    # Make sure the system is not degenerate
                    Cmat = control.ctrb(ssOriginal.A, ssOriginal.B)
                    if (np.linalg.matrix_rank(Cmat) != states):
                        if (verbose):
                            print("  skipping (not reachable)")
                        continue
                    Omat = control.obsv(ssOriginal.A, ssOriginal.C)
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
                                control.bode(_mimo2siso(ssOriginal, \
                                                        inputNum, outputNum), \
                                                 deg=False, Plot=False)
                            ssorig_real = ssorig_mag * np.cos(ssorig_phase)
                            ssorig_imag = ssorig_mag * np.sin(ssorig_phase)

                            #
                            # Make sure TF has same frequency response
                            #
                            num = tfOriginal.num[outputNum][inputNum]
                            den = tfOriginal.den[outputNum][inputNum]
                            tforig = control.tf(num, den)
                                                
                            tforig_mag, tforig_phase, tforig_omega = \
                                control.bode(tforig, ssorig_omega, \
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
                                control.bode(_mimo2siso(ssTransformed, \
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
                            tfxfrm = control.tf(num, den)
                            tfxfrm_mag, tfxfrm_phase, tfxfrm_omega = \
                                control.bode(tfxfrm, ssorig_omega, \
                                                 deg=False, Plot=False)
                            
                            tfxfrm_real = tfxfrm_mag * np.cos(tfxfrm_phase)
                            tfxfrm_imag = tfxfrm_mag * np.sin(tfxfrm_phase)
                            np.testing.assert_array_almost_equal( \
                                ssorig_real, tfxfrm_real)
                            np.testing.assert_array_almost_equal( \
                                ssorig_imag, tfxfrm_imag)

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestConvert)

if __name__ == "__main__":
    unittest.main()
