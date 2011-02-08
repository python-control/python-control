#!/usr/bin/env python

import numpy as np
import matlab
import unittest

class TestRss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.rss."""
    
    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maxmimum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5
        
    def testShape(self):
        """Test that rss outputs have the right state, input, and output
        size."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, inputs, outputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)
    
    def testPole(self):
        """Test that the poles of rss outputs have a negative real part."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.rss(states, inputs, outputs)
                    p = sys.poles()
                    for z in p:
                        self.assertTrue(z.real < 0)

class TestDrss(unittest.TestCase):
    """These are tests for the proper functionality of statesp.drss."""
    
    def setUp(self):
        # Number of times to run each of the randomized tests.
        self.numTests = 100
        # Maxmimum number of states to test + 1
        self.maxStates = 10
        # Maximum number of inputs and outputs to test + 1
        self.maxIO = 5
        
    def testShape(self):
        """Test that drss outputs have the right state, input, and output
        size."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, inputs, outputs)
                    self.assertEqual(sys.states, states)
                    self.assertEqual(sys.inputs, inputs)
                    self.assertEqual(sys.outputs, outputs)
    
    def testPole(self):
        """Test that the poles of drss outputs have less than unit magnitude."""
        
        for states in range(1, self.maxStates):
            for inputs in range(1, self.maxIO):
                for outputs in range(1, self.maxIO):
                    sys = matlab.drss(states, inputs, outputs)
                    p = sys.poles()
                    for z in p:
                        self.assertTrue(abs(z) < 1)
                        
if __name__ == "__main__":
    unittest.main()