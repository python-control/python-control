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
yout, T = step(sys)
plot(T.T, yout.T)

# Bode plot for the system
figure(2)
mag,phase,om = bode(sys, logspace(-2, 2),Plot=True)

# Nyquist plot for the system
figure(3)
nyquist(sys, logspace(-2, 2))

# Root lcous plut for the system
figure(4)
rlocus(sys)
