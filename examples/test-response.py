# test-response.py - Unit tests for system response functions
# RMM, 11 Sep 2010

from matplotlib.pyplot import * # Grab MATLAB plotting functions
from control.matlab import *    # Load the controls systems library
from scipy import arange        # function to create range of numbers

# Create several systems for testing
sys1 = tf([1], [1, 2, 1])
sys2 = tf([1, 1], [1, 1, 0])

# Generate step responses
(y1a, T1a) = step(sys1)
(y1b, T1b) = step(sys1, T = arange(0, 10, 0.1))
(y1c, T1c) = step(sys1, X0 = [1, 0])
(y2a, T2a) = step(sys2, T = arange(0, 10, 0.1))

plot(T1a, y1a, T1b, y1b, T1c, y1c, T2a, y2a)
