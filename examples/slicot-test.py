# Simple test function to test out SLICOT interface
# RMM, 28 May 09

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

# Eigenvalue placement
#from slycot import sb01bd
K = place(A, B, [-3, -2, -1])
print "Pole place: K = ", K
print "Pole place: eigs = ", np.linalg.eig(A - B * K)[0]

# from slycot import ab01md
# Ac, Bc, ncont, info = ab01md('I', A, B)
