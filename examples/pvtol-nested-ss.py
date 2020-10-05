# pvtol-nested.py - inner/outer design for vectored thrust aircraft
# RMM, 5 Sep 09
#
# This file works through a fairly complicated control design and
# analysis, corresponding to the planar vertical takeoff and landing
# (PVTOL) aircraft in Astrom and Mruray, Chapter 11.  It is intended
# to demonstrate the basic functionality of the python-control
# package.
#

import os
import matplotlib.pyplot as plt  # MATLAB plotting functions
from control.matlab import *    # MATLAB-like functions
import numpy as np

# System parameters
m = 4           # mass of aircraft
J = 0.0475      # inertia around pitch axis
r = 0.25        # distance to center of force
g = 9.8         # gravitational constant
c = 0.05        # damping factor (estimated)

# Transfer functions for dynamics
Pi = tf([r], [J, 0, 0])  # inner loop (roll)
Po = tf([1], [m, c, 0])  # outer loop (position)

# Use state space versions
Pi = tf2ss(Pi)
Po = tf2ss(Po)

#
# Inner loop control design
#
# This is the controller for the pitch dynamics.  Goal is to have
# fast response for the pitch dynamics so that we can use this as a 
# control for the lateral dynamics
#

# Design a simple lead controller for the system
k, a, b = 200, 2, 50
Ci = k*tf([1, a], [1, b])  # lead compensator

# Convert to statespace
Ci = tf2ss(Ci)

# Compute the loop transfer function for the inner loop
Li = Pi*Ci


# Bode plot for the open loop process
plt.figure(1)
bode(Pi)

# Bode plot for the loop transfer function, with margins
plt.figure(2)
bode(Li)

# Compute out the gain and phase margins
#! Not implemented
# (gm, pm, wcg, wcp) = margin(Li);

# Compute the sensitivity and complementary sensitivity functions
Si = feedback(1, Li)
Ti = Li*Si

# Check to make sure that the specification is met
plt.figure(3)
gangof4(Pi, Ci)

# Compute out the actual transfer function from u1 to v1 (see L8.2 notes)
# Hi = Ci*(1-m*g*Pi)/(1+Ci*Pi);
Hi = parallel(feedback(Ci, Pi), -m*g*feedback(Ci*Pi, 1))

plt.figure(4)
plt.clf()
plt.subplot(221)
bode(Hi)

# Now design the lateral control system
a, b, K = 0.02, 5, 2
Co = -K*tf([1, 0.3], [1, 10])  # another lead compensator

# Convert to statespace
Co = tf2ss(Co)

# Compute the loop transfer function for the outer loop
Lo = -m*g*Po*Co

plt.figure(5)
bode(Lo)  # margin(Lo)

# Finally compute the real outer-loop loop gain + responses
L = Co*Hi*Po
S = feedback(1, L)
T = feedback(L, 1)

# Compute stability margins
#! Not yet implemented
# (gm, pm, wgc, wpc) = margin(L); 

plt.figure(6)
plt.clf()
bode(L, logspace(-4, 3))

# Add crossover line to magnitude plot
for ax in plt.gcf().axes:
    if ax.get_label() == 'control-bode-magnitude':
        break
ax.semilogx([1e-4, 1e3], 20*np.log10([1, 1]), 'k-')

# Re-plot phase starting at -90 degrees
mag, phase, w = freqresp(L, logspace(-4, 3))
phase = phase - 360

for ax in plt.gcf().axes:
    if ax.get_label() == 'control-bode-phase':
        break
ax.semilogx([1e-4, 1e3], [-180, -180], 'k-')
ax.semilogx(w, np.squeeze(phase), 'b-')
ax.axis([1e-4, 1e3, -360, 0])
plt.xlabel('Frequency [deg]')
plt.ylabel('Phase [deg]')
# plt.set(gca, 'YTick', [-360, -270, -180, -90, 0])
# plt.set(gca, 'XTick', [10^-4, 10^-2, 1, 100])

#
# Nyquist plot for complete design
#
plt.figure(7)
plt.clf()
plt.axis([-700, 5300, -3000, 3000])
nyquist(L, (0.0001, 1000))
plt.axis([-700, 5300, -3000, 3000])

# Add a box in the region we are going to expand
plt.plot([-400, -400, 200, 200, -400], [-100, 100, 100, -100, -100], 'r-')

# Expanded region  
plt.figure(8)
plt.clf()
plt.subplot(231)
plt.axis([-10, 5, -20, 20])
nyquist(L)
plt.axis([-10, 5, -20, 20])

# set up the color
color = 'b'

# Add arrows to the plot
# H1 = L.evalfr(0.4); H2 = L.evalfr(0.41);
# arrow([real(H1), imag(H1)], [real(H2), imag(H2)], AM_normal_arrowsize, \
#  'EdgeColor', color, 'FaceColor', color);

# H1 = freqresp(L, 0.35); H2 = freqresp(L, 0.36);
# arrow([real(H2), -imag(H2)], [real(H1), -imag(H1)], AM_normal_arrowsize, \
#  'EdgeColor', color, 'FaceColor', color);

plt.figure(9)
Yvec, Tvec = step(T, linspace(1, 20))
plt.plot(Tvec.T, Yvec.T)

Yvec, Tvec = step(Co*S, linspace(1, 20))
plt.plot(Tvec.T, Yvec.T)

#TODO: PZmap for statespace systems has not yet been implemented.
plt.figure(10)
plt.clf()
# P, Z = pzmap(T, Plot=True)
# print("Closed loop poles and zeros: ", P, Z)

# Gang of Four
plt.figure(11)
plt.clf()
gangof4(Hi*Po, Co, linspace(-2, 3))

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
