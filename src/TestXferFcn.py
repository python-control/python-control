#!/usr/bin/env python

import numpy as np
from xferfcn import xTransferFunction
import unittest

class TestXferFcn(unittest.TestCase):
    """These are tests for functionality and correct reporting of the transfer
    function class."""
    
    def testTruncateCoeff(self):
        """Remove extraneous zeros in polynomial representations."""
        
        sys1 = xTransferFunction([0., 0., 1., 2.], [[[0., 0., 0., 3., 2., 1.]]])
        
        np.testing.assert_array_equal(sys1.num, [[[1., 2.]]])
        np.testing.assert_array_equal(sys1.den, [[[3., 2., 1.]]])
    
    def testAddSISO1(self):
        """Add two direct feedthrough systems."""
        
        # Try different input formats, too.
        sys1 = xTransferFunction(1., [[[1.]]])
        sys2 = xTransferFunction([2.], [1.])
        sys3 = sys1 + sys2
        
        np.testing.assert_array_equal(sys3.num, 3.)
        np.testing.assert_array_equal(sys3.den, 1.)

    def testAddSISO2(self):
        """Add two SISO systems."""
        
        # Try different input formats, too.
        sys1 = xTransferFunction([1., 3., 5], [1., 6., 2., -1])
        sys2 = xTransferFunction([[[-1., 3.]]], [[[1., 0., -1.]]])
        sys3 = sys1 + sys2
        
        # If sys3.num is [[[0., 20., 4., -8.]]], then this is wrong!
        np.testing.assert_array_equal(sys3.num, [[[20., 4., -8]]])
        np.testing.assert_array_equal(sys3.den, [[[1., 6., 1., -7., -2., 1.]]])
        
    def testAddMIMO(self):
        """Add two MIMO systems."""
        
        num1 = [[[1., 2.], [0., 3.], [2., -1.]],
                [[1.], [4., 0.], [1., -4., 3.]]]
        den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
                [[3., 0., .0], [2., -1., -1.], [1.]]]
        num2 = [[[0., 0., -1], [2.], [-1., -1.]],
                [[1., 2.], [-1., -2.], [4.]]]
        den2 = [[[-1.], [1., 2., 3.], [-1., -1.]],
                [[-4., -3., 2.], [0., 1.], [1., 0.]]]
        num3 = [[[3., -3., -6], [5., 6., 9.], [-4., -2., 2]],
                [[3., 2., -3., 2], [-2., -3., 7., 2.], [1., -4., 3., 4]]]
        den3 = [[[3., -2., -4.], [1., 2., 3., 0., 0.], [-2., -1., 1.]],
                [[-12., -9., 6., 0., 0.], [2., -1., -1.], [1., 0.]]]
                
        sys1 = xTransferFunction(num1, den1)
        sys2 = xTransferFunction(num2, den2)
        
        sys3 = sys1 + sys2

        for i in range(sys3.outputs):
            for j in range(sys3.inputs):
                np.testing.assert_array_equal(sys3.num[i][j], num3[i][j])
                np.testing.assert_array_equal(sys3.den[i][j], den3[i][j])

if __name__ == "__main__":
    unittest.main()