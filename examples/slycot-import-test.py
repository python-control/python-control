""" slycot-import-test.py

Simple example script to test Slycot import
RMM, 28 May 09
"""

import numpy as np
from control.matlab import *
from control.exception import slycot_check

# Parameters defining the system
m = 250.0  # system mass
k = 40.0   # spring constant
b = 60.0   # damping constant

# System matrices
A = np.array([[1, -1, 1.], [1, -k/m, -b/m], [1, 1, 1]])
B = np.array([[0], [1/m], [1]])
C = np.array([[1., 0, 1.]])
sys = ss(A, B, C, 0)

# Python control may be used without slycot, for example for a pole placement.
# Eigenvalue placement
w = [-3, -2, -1]
K = place(A, B, w)
print("[python-control (from scipy)] K = ", K)
print("[python-control (from scipy)] eigs = ", np.linalg.eig(A - B*K)[0])

# Before using one of its routine, check that slycot is installed.
w = np.array([-3, -2, -1])
if slycot_check():
    # Import routine sb01bd used for pole placement.
    from slycot import sb01bd

    n = 3        # Number of states
    m = 1        # Number of inputs
    npp = 3      # Number of placed eigen values
    alpha = 1    # Maximum threshold for eigen values
    dico = 'D'   # Discrete system
    _, _, _, _, _, K, _ = sb01bd(n, m, npp, alpha, A, B, w, dico, tol=0.0, ldwork=None)
    print("[slycot] K = ", K)
    print("[slycot] eigs = ", np.linalg.eig(A + B @ K)[0])
else:
    print("Slycot is not installed.")
