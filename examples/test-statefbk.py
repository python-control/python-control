# test-statefbk.py - Unit tests for state feedback code
# RMM, 6 Sep 2010

import numpy as np              # Numerical library
from scipy import *             # Load the scipy functions
from control.matlab import *    # Load the controls systems library

# Parameters defining the system
m = 250.0			# system mass
k = 40.0			# spring constant
b = 60.0			# damping constant

# System matrices
A = matrix([[1, -1, 1.], [1, -k/m, -b/m], [1, 1, 1]])
B = matrix([[0], [1/m], [1]])
C = matrix([[1., 0, 1.]])
sys = ss(A, B, C, 0);

# Controllability
Wc = ctrb(A, B)
print "Wc = ", Wc

# Observability
Wo = obsv(A, C)
print "Wo = ", Wo


