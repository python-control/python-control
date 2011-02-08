#!/usr/bin/env python

import numpy as np
import statesp as ss
import unittest

class TestRss(unittest.TestCase):
	"""These are tests for the proper functionality of statesp.rss."""
	
	def setUp(self):
		# Number of times to run each of the randomized tests.
		self.numTests = 100
		
	def testShape(self):
		"""Test that rss outputs have the right state, input, and output
		size."""
		
		for states in range(1, 10):
			for inputs in range(1, 5):
				for outputs in range(1, 5):
					sys = ss.rss(states, inputs, outputs)
					self.assertEqual(sys.states, states)
					self.assertEqual(sys.inputs, inputs)
					self.assertEqual(sys.outputs, outputs)
	
	def testPole(self):
		"""Test that the poles of rss outputs have a negative real part."""
		
		for states in range(1, 10):
			for inputs in range(1, 5):
				for outputs in range(1, 5):
					sys = ss.rss(states, inputs, outputs)
					p = sys.poles()
					for z in p:
						self.assertTrue(z.real < 0)
						
if __name__ == "__main__":
	unittest.main()