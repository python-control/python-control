#!/usr/bin/env python
#
# frd_test.py - test FRD class
# RvP, 4 Oct 2012


import unittest
import numpy as np
from control.statesp import StateSpace
from control.xferfcn import TransferFunction, _convertToTransferFunction
from control.frdata import FRD

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
        


if __name__ == "__main__":
    unittest.main()
