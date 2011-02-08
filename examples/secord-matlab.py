# secord.py - demonstrate some standard MATLAB commands 
# RMM, 25 May 09

from matplotlib.pyplot import * # Grab MATLAB plotting functions
from control.matlab import *    # MATLAB-like functions

# Parameters defining the system
m = 250.0			# system mass
k = 40.0			# spring constant
b = 60.0			# damping constant

# System matrices
A = [[0, 1.], [-k/m, -b/m]]
B = [[0], [1/m]]
C = [[1., 0]]
sys = ss(A, B, C, 0);

# Step response for the system
figure(1)
T, yout = step(sys)
plot(T, yout)

# Bode plot for the system
figure(2)
bode(sys, logspace(-2, 2))

# Nyquist plot for the system
figure(3)
nyquist(sys, logspace(-2, 2))
