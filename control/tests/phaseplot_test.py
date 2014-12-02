#!/usr/bin/env python
#
# phaseplot_test.py - test phase plot functions
# RMM, 17 24 2011 (based on TestMatlab from v0.4c)
#
# This test suite calls various phaseplot functions.  Since the plots
# themselves can't be verified, this is mainly here to make sure all
# of the function arguments are handled correctly.  If you run an
# individual test by itself and then type show(), it should pop open
# the figures so that you can check them visually.

import unittest
import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl
from control.phaseplot import *
from numpy import pi

class TestPhasePlot(unittest.TestCase):
    def setUp(self):
        pass;

    def testInvPendNoSims(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10));

    def testInvPendSims(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10), 
                  X0 = ([1,1], [-1,1]));

    def testInvPendTimePoints(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10), 
                  X0 = ([1,1], [-1,1]), T=np.linspace(0,5,100));

    def testInvPendLogtime(self):
        phase_plot(self.invpend_ode, X0 = 
                  [ [-2*pi, 1.6], [-2*pi, 0.5], [-1.8, 2.1],
                    [-1, 2.1], [4.2, 2.1], [5, 2.1],
                    [2*pi, -1.6], [2*pi, -0.5], [1.8, -2.1],
                    [1, -2.1], [-4.2, -2.1], [-5, -2.1] ], 
                  T = np.linspace(0, 40, 200),
                  logtime=(3, 0.7),
                  verbose=False)

    def testInvPendAuto(self):
        phase_plot(self.invpend_ode, lingrid = 0, X0=
                  [[-2.3056, 2.1], [2.3056, -2.1]], T=6, verbose=False)

    def testOscillatorParams(self):
        m = 1; b = 1; k = 1;			# default values
        phase_plot(self.oscillator_ode, timepts = [0.3, 1, 2, 3], X0 = 
                  [[-1,1], [-0.3,1], [0,1], [0.25,1], [0.5,1], [0.7,1],
                   [1,1], [1.3,1], [1,-1], [0.3,-1], [0,-1], [-0.25,-1],
                   [-0.5,-1], [-0.7,-1], [-1,-1], [-1.3,-1]],
                  T = np.linspace(0, 10, 100), parms = (m, b, k));

    # Sample dynamical systems - inverted pendulum
    def invpend_ode(self, x, t, m=1., l=1., b=0, g=9.8):
        import numpy as np
        return (x[1], -b/m*x[1] + (g*l/m) * np.sin(x[0]))

    # Sample dynamical systems - oscillator
    def oscillator_ode(self, x, t, m=1., b=1, k=1, extra=None):
        return (x[1], -k/m*x[0] - b/m*x[1])

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestPhasePlot)

if __name__ == '__main__':
    unittest.main()
