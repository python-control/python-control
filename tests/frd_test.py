#!/usr/bin/env python
#
# frd_test.py - test FRD class
# RvP, 4 Oct 2012


import unittest
import numpy as np
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.frdata import FRD, _convertToFRD
from control.matlab import bode
import control.bdalg as bdalg
import control.freqplot
import matplotlib.pyplot as plt

class TestFRD(unittest.TestCase):
    """These are tests for functionality and correct reporting of the
    frequency response data class."""

    def testBadInputType(self):
        """Give the constructor invalid input types."""
        self.assertRaises(ValueError, FRD)
        
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
        #h2.feedback([[0.3, 0.2],[0.1, 0.1]])
    
    def testAuto(self):
        omega = np.logspace(-1, 2, 10)
        f1 = _convertToFRD(1, omega)
        f2 = _convertToFRD(np.matrix([[1, 0], [0.1, -1]]), omega)
        f2 = _convertToFRD([[1, 0], [0.1, -1]], omega)

    def testNyquist(self):
        h1 = TransferFunction([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 40)
        f1 = FRD(h1, omega, smooth=True)
        control.freqplot.nyquist(f1, np.logspace(-1, 2, 100))
        plt.savefig('/dev/null', format='svg')
        plt.figure(2)
        control.freqplot.nyquist(f1, f1.omega)
        plt.savefig('/dev/null', format='svg')

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

    def testMIMOfb(self):
        sys = StateSpace([[-0.5, 0.0], [0.0, -1.0]], 
                         [[1.0, 0.0], [0.0, 1.0]], 
                         [[1.0, 0.0], [0.0, 1.0]], 
                         [[0.0, 0.0], [0.0, 0.0]])
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega).feedback([[0.1, 0.3],[0.0, 1.0]])
        f2 = FRD(sys.feedback([[0.1, 0.3],[0.0, 1.0]]), omega)
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[0],
            f2.freqresp([0.1, 1.0, 10])[0])
        np.testing.assert_array_almost_equal(
            f1.freqresp([0.1, 1.0, 10])[1],
            f2.freqresp([0.1, 1.0, 10])[1])

    def testMIMOfb2(self):
        sys = StateSpace(np.matrix('-2.0 0 0; 0 -1 1; 0 0 -3'), 
                         np.matrix('1.0 0; 0 0; 0 1'), 
                         np.eye(3), np.zeros((3,2)))
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
        #sys = ss([-2 0 0; 0 -1 1; 0 0 -3], [1 0; 0 0; 0 1], eye(3), zeros(3,2))
        #bfr = frd(bsys, [1])
        sys = StateSpace(np.matrix('-2.0 0 0; 0 -1 1; 0 0 -3'), 
                         np.matrix('1.0 0; 0 0; 0 1'), 
                         np.eye(3), np.zeros((3,2)))
        omega = np.logspace(-1, 2, 10)
        f1 = FRD(sys, omega)
        np.testing.assert_array_almost_equal(
            (f1.freqresp([1.0])[0] * 
             np.exp(1j*f1.freqresp([1.0])[1])).reshape(3,2),
            np.matrix('0.4-0.2j 0; 0 0.1-0.2j; 0 0.3-0.1j'))

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestFRD)

if __name__ == "__main__":
    unittest.main()

