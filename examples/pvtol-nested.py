# pvtol-nested.py - inner/outer design for vectored thrust aircraft
# RMM, 5 Sep 09
#
# This file works through a fairly complicated control design and
# analysis, corresponding to the planar vertical takeoff and landing
# (PVTOL) aircraft in Astrom and Murray, Chapter 11.  It is intended
# to demonstrate the basic functionality of the python-control
# package.
#

import os
import matplotlib.pyplot as plt  # MATLAB-like plotting functions
import control as ct
import numpy as np

# System parameters
m = 4               # mass of aircraft
J = 0.0475          # inertia around pitch axis
r = 0.25            # distance to center of force
g = 9.8             # gravitational constant
c = 0.05            # damping factor (estimated)

# Transfer functions for dynamics
Pi = ct.tf([r], [J, 0, 0])  # inner loop (roll)
Po = ct.tf([1], [m, c, 0])  # outer loop (position)

#
# Inner loop control design
#
# This is the controller for the pitch dynamics.  Goal is to have
# fast response for the pitch dynamics so that we can use this as a
# control for the lateral dynamics
#

# Design a simple lead controller for the system
k, a, b = 200, 2, 50
Ci = k * ct.tf([1, a], [1, b])  # lead compensator
Li = Pi * Ci

# Bode plot for the open loop process
plt.figure(1)
ct.bode_plot(Pi)

# Bode plot for the loop transfer function, with margins
plt.figure(2)
ct.bode_plot(Li)

# Compute out the gain and phase margins
gm, pm, wcg, wcp = ct.margin(Li)

# Compute the sensitivity and complementary sensitivity functions
Si = ct.feedback(1, Li)
Ti = Li * Si

# Check to make sure that the specification is met
plt.figure(3)
ct.gangof4(Pi, Ci)

# Compute out the actual transfer function from u1 to v1 (see L8.2 notes)
# Hi = Ci*(1-m*g*Pi)/(1+Ci*Pi)
Hi = ct.parallel(ct.feedback(Ci, Pi), -m * g *ct.feedback(Ci * Pi, 1))

plt.figure(4)
plt.clf()
plt.subplot(221)
ct.bode_plot(Hi)

# Now design the lateral control system
a, b, K = 0.02, 5, 2
Co = -K * ct.tf([1, 0.3], [1, 10])  # another lead compensator
Lo = -m*g*Po*Co

plt.figure(5)
ct.bode_plot(Lo)  # margin(Lo)

# Finally compute the real outer-loop loop gain + responses
L = Co * Hi * Po
S = ct.feedback(1, L)
T = ct.feedback(L, 1)

# Compute stability margins
gm, pm, wgc, wpc = ct.margin(L)
print("Gain margin: %g at %g" % (gm, wgc))
print("Phase margin: %g at %g" % (pm, wpc))

plt.figure(6)
plt.clf()
ct.bode_plot(L, np.logspace(-4, 3))

# Add crossover line to the magnitude plot
#
# Note: in matplotlib before v2.1, the following code worked:
#
#   plt.subplot(211); hold(True);
#   loglog([1e-4, 1e3], [1, 1], 'k-')
#
# In later versions of matplotlib the call to plt.subplot will clear the
# axes and so we have to extract the axes that we want to use by hand.
# In addition, hold() is deprecated so we no longer require it.
#
for ax in plt.gcf().axes:
    if ax.get_label() == 'control-bode-magnitude':
        break
ax.semilogx([1e-4, 1e3], 20*np.log10([1, 1]), 'k-')

#
# Replot phase starting at -90 degrees
#
# Get the phase plot axes
for ax in plt.gcf().axes:
    if ax.get_label() == 'control-bode-phase':
        break

# Recreate the frequency response and shift the phase
mag, phase, w = ct.freqresp(L, np.logspace(-4, 3))
phase = phase - 360

# Replot the phase by hand
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
ct.nyquist_plot(L, (0.0001, 1000))

# Add a box in the region we are going to expand
plt.plot([-2, -2, 1, 1, -2], [-4, 4, 4, -4, -4], 'r-')

# Expanded region
plt.figure(8)
plt.clf()
ct.nyquist_plot(L)
plt.axis([-2, 1, -4, 4])

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
Tvec, Yvec = ct.step_response(T, np.linspace(0, 20))
plt.plot(Tvec.T, Yvec.T)

Tvec, Yvec = ct.step_response(Co*S, np.linspace(0, 20))
plt.plot(Tvec.T, Yvec.T)

plt.figure(10)
plt.clf()
P, Z = ct.pzmap(T, plot=True, grid=True)
print("Closed loop poles and zeros: ", P, Z)

# Gang of Four
plt.figure(11)
plt.clf()
ct.gangof4_plot(Hi * Po, Co)

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
