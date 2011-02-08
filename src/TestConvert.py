#!/usr/bin/env python

"""TestConvert.py

Test state space and transfer function conversion.

Currently, this unit test script is not complete.  It converts several random
state spaces back and forth between state space and transfer function
representations.  Ideally, it should be able to assert that the conversion
outputs are correct.  This is not yet implemented.

Also, the conversion seems to enter an infinite loop once in a while.  The cause
of this is unknown.

"""

import numpy as np
import matlab
import unittest

class TestConvert(unittest.TestCase):
    """Test state space and transfer function conversions."""

    def setUp(self):
        """Set up testing parameters."""

        # Number of times to run each of the randomized tests.
        self.numTests = 10 #almost guarantees failure
        # Maximum number of states to test + 1
        self.maxStates = 20
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 20
        # Set to True to print systems to the output.
        self.debug = False

    def printSys(self, sys, ind):
        """Print system to the standard output."""

        if self.debug:
            print "sys%i:\n" % ind
            print sys

    def testConvert(self):
        """Test state space to transfer function conversion."""
        #Currently it only tests that a TF->SS->TF generates an unchanged TF
        
        #print __doc__

        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    #start with a random SS system and transform to TF
                    #then back to SS, check that the matrices are the same.
                    ssOriginal = matlab.rss(states, inputs, outputs)
                    self.printSys(ssOriginal, 1)

                    tfOriginal = matlab.tf(ssOriginal)
                    self.printSys(tfOriginal, 2)
                    
                    ssTransformed = matlab.ss(tfOriginal)
                    self.printSys(ssTransformed, 3)

                    tfTransformed = matlab.tf(ssTransformed)
                    self.printSys(tfTransformed, 4)
                    
                    for inputNum in range(inputs):
                        for outputNum in range(outputs):
                            np.testing.assert_array_almost_equal(\
                              tfOriginal.num[outputNum][inputNum],\
                            tfTransformed.num[outputNum][inputNum])
                            
                            np.testing.assert_array_almost_equal(\
                              tfOriginal.den[outputNum][inputNum],\
                                tfTransformed.den[outputNum][inputNum])
                    
                    #To test the ss systems is harder because they aren't the same
                    #realization. This could be done with checking that they have the 
                    #same freq response with bode, but apparently it doesn't work
                    #the way it should right now:
                    ## Bode should work like this:
                    #[mag,phase,freq]=bode(sys)
                    #it doesn't seem to......
                    #This should be added.
                    
                    

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestConvert)

if __name__ == "__main__":
    unittest.main()
