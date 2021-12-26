""" check-controllability-and-observability.py

Example to check the controllability and the observability of a state space system.
RMM, 6 Sep 2010
"""

import numpy as np  # Load the scipy functions
from control.matlab import *  # Load the controls systems library

# Parameters defining the system

m = 250.0  # system mass
k = 40.0   # spring constant
b = 60.0   # damping constant

# System matrices
A = np.array([[1, -1, 1.],
             [1, -k/m, -b/m],
             [1, 1, 1]])

B = np.array([[0],
             [1/m],
             [1]])

C = np.array([[1., 0, 1.]])

sys = ss(A, B, C, 0)

# Check controllability
Wc = ctrb(A, B)
print("Wc = ", Wc)

# Check Observability
Wo = obsv(A, C)
print("Wo = ", Wo)
