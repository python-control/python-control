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

if __name__ == "__main__":
    unittest.main()
