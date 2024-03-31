# phase_plane_plots.py - phase portrait examples
# RMM, 25 Mar 2024
#
# This file contains a number of examples of phase plane plots generated
# using the phaseplot module.  Most of these figures line up with examples
# in FBS2e, with different display options shown as different subplots.

import time
import warnings
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np

import control as ct
import control.phaseplot as pp

#
# Example 1: Dampled oscillator systems
#

# Oscillator parameters
damposc_params = {'m': 1, 'b': 1, 'k': 1}

# System model (as ODE)
def damposc_update(t, x, u, params):
    m, b, k = params['m'], params['b'], params['k']
    return np.array([x[1], -k/m * x[0] - b/m * x[1]])
damposc = ct.nlsys(damposc_update, states=2, inputs=0, params=damposc_params)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.3: damped oscillator")

ct.phase_plane_plot(damposc, [-1, 1, -1, 1], 8, ax=ax1)
ax1.set_title("boxgrid [-1, 1, -1, 1], 8")

ct.phase_plane_plot(damposc, [-1, 1, -1, 1], ax=ax2, gridtype='meshgrid')
ax2.set_title("meshgrid [-1, 1, -1, 1]")

ct.phase_plane_plot(
    damposc, [-1, 1, -1, 1], 4, ax=ax3, gridtype='circlegrid', dir='both')
ax3.set_title("circlegrid [0, 0, 1], 4, both")

ct.phase_plane_plot(
    damposc, [-1, 1, -1, 1], ax=ax4, gridtype='circlegrid',
    dir='reverse', gridspec=[0.1, 12], timedata=5)
ax4.set_title("circlegrid [0, 0, 0.1], reverse")

#
# Example 2: Inverted pendulum
#

def invpend_update(t, x, u, params):
    m, l, b, g = params['m'], params['l'], params['b'], params['g']
    return [x[1], -b/m * x[1] + (g * l / m) * np.sin(x[0])]
invpend = ct.nlsys(
    invpend_update, states=2, inputs=0,
    params={'m': 1, 'l': 1, 'b': 0.2, 'g': 1})

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.4: inverted pendulum")

ct.phase_plane_plot(
    invpend, [-2*pi, 2*pi, -2, 2], 5, ax=ax1)
ax1.set_title("default, 5")

ct.phase_plane_plot(
    invpend, [-2*pi, 2*pi, -2, 2], gridtype='meshgrid', ax=ax2)
ax2.set_title("meshgrid")

ct.phase_plane_plot(
    invpend, [-2*pi, 2*pi, -2, 2], 1, gridtype='meshgrid',
    gridspec=[12, 9], ax=ax3, arrows=1)
ax3.set_title("denser grid")

ct.phase_plane_plot(
    invpend, [-2*pi, 2*pi, -2, 2], 4, gridspec=[6, 6],
    plot_separatrices={'timedata': 20, 'arrows': 4}, ax=ax4)
ax4.set_title("custom")

#
# Example 3: Limit cycle (nonlinear oscillator)
#

def oscillator_update(t, x, u, params):
    return [
        x[1] + x[0] * (1 - x[0]**2 - x[1]**2),
        -x[0] + x[1] * (1 - x[0]**2 - x[1]**2)
    ]
oscillator = ct.nlsys(oscillator_update, states=2, inputs=0)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.5: Nonlinear oscillator")

ct.phase_plane_plot(oscillator, [-1.5, 1.5, -1.5, 1.5], 3, ax=ax1)
ax1.set_title("default, 3")
ax1.set_aspect('equal')

try:
    ct.phase_plane_plot(
        oscillator, [-1.5, 1.5, -1.5, 1.5], 1, gridtype='meshgrid',
        dir='forward', ax=ax2)
except RuntimeError as inst:
    axs[0,1].text(0, 0, "Runtime Error")
    warnings.warn(inst.__str__())
ax2.set_title("meshgrid, forward, 0.5")
ax2.set_aspect('equal')

ct.phase_plane_plot(oscillator, [-1.5, 1.5, -1.5, 1.5], ax=ax3)
pp.streamlines(
    oscillator, [-0.5, 0.5, -0.5, 0.5], dir='both', ax=ax3)
ax3.set_title("outer + inner")
ax3.set_aspect('equal')

ct.phase_plane_plot(
    oscillator, [-1.5, 1.5, -1.5, 1.5], 0.9, ax=ax4)
pp.streamlines(
    oscillator, np.array([[0, 0]]), 1.5,
    gridtype='circlegrid', gridspec=[0.5, 6], dir='both', ax=ax4)
pp.streamlines(
    oscillator, np.array([[1, 0]]), 2*pi, arrows=6, ax=ax4, color='b')
ax4.set_title("custom")
ax4.set_aspect('equal')

#
# Example 4: Simple saddle
#

def saddle_update(t, x, u, params):
    return [x[0] - 3*x[1], -3*x[0] + x[1]]
saddle = ct.nlsys(saddle_update, states=2, inputs=0)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.9: Saddle")

ct.phase_plane_plot(saddle, [-1, 1, -1, 1], ax=ax1)
ax1.set_title("default")

ct.phase_plane_plot(
    saddle, [-1, 1, -1, 1], 0.5, gridtype='meshgrid', ax=ax2)
ax2.set_title("meshgrid")

ct.phase_plane_plot(
    saddle, [-1, 1, -1, 1], gridspec=[16, 12], ax=ax3, 
    plot_vectorfield=True, plot_streamlines=False, plot_separatrices=False)
ax3.set_title("vectorfield")

ct.phase_plane_plot(
    saddle, [-1, 1, -1, 1], 0.3,
    gridtype='meshgrid', gridspec=[5, 7], ax=ax4)
ax3.set_title("custom")

#
# Example 5: Internet congestion control
#

def _congctrl_update(t, x, u, params):
    # Number of sources per state of the simulation
    M = x.size - 1                      # general case
    assert M == 1                       # make sure nothing funny happens here

    # Remaining parameters
    N = params.get('N', M)              # number of sources
    rho = params.get('rho', 2e-4)       # RED parameter = pbar / (bupper-blower)
    c = params.get('c', 10)             # link capacity (Mp/ms)

    # Compute the derivative (last state = bdot)
    return np.append(
        c / x[M] - (rho * c) * (1 + (x[:-1]**2) / 2),
        N/M * np.sum(x[:-1]) * c / x[M] - c)
congctrl = ct.nlsys(
    _congctrl_update, states=2, inputs=0,
    params={'N': 60, 'rho': 2e-4, 'c': 10})

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_tight_layout(True)
plt.suptitle("FBS Figure 5.10: Congestion control")

try:
    ct.phase_plane_plot(
        congctrl, [0, 10, 100, 500], 120, ax=ax1)
except RuntimeError as inst:
    ax1.text(5, 250, "Runtime Error")
    warnings.warn(inst.__str__())
ax1.set_title("default, T=120")

try:
    ct.phase_plane_plot(
        congctrl, [0, 10, 100, 500], 120,
        params={'rho': 4e-4, 'c': 20}, ax=ax2)
except RuntimeError as inst:
    ax2.text(5, 250, "Runtime Error")
    warnings.warn(inst.__str__())
ax2.set_title("updated param")

ct.phase_plane_plot(
    congctrl, [0, 10, 100, 500], ax=ax3,
    plot_vectorfield=True, plot_streamlines=False)
ax3.set_title("vector field")

ct.phase_plane_plot(
    congctrl, [2, 6, 200, 300], 100,
    params={'rho': 4e-4, 'c': 20},
    ax=ax4, plot_vectorfield={'gridspec': [12, 9]})
ax4.set_title("vector field + streamlines")

#
# End of examples
#

plt.show(block=False)
