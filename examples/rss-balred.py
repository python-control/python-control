#!/usr/bin/env python

import os

import numpy as np
import control.modelsimp as msimp
import control.matlab as mt
from control.statesp import StateSpace
import matplotlib.pyplot as plt

plt.close('all')

# controllable canonical realization computed in MATLAB for the
# transfer function: num = [1 11 45 32], den = [1 15 60 200 60]
A = np.array([
    [-15., -7.5, -6.25, -1.875],
    [8., 0., 0., 0.],
    [0., 4., 0., 0.],
    [0., 0., 1., 0.]
])
B = np.array([
    [2.],
    [0.],
    [0.],
    [0.]
])
C = np.array([[0.5, 0.6875, 0.7031, 0.5]])
D = np.array([[0.]])

# The full system
fsys = StateSpace(A, B, C, D)

# The reduced system, truncating the order by 1
n = 3
rsys = msimp.balred(fsys, n, method='truncate')

# Comparison of the step responses of the full and reduced systems
plt.figure(1)
y, t = mt.step(fsys)
yr, tr = mt.step(rsys)
plt.plot(t.T, y.T)
plt.plot(tr.T, yr.T)

# Repeat balanced reduction, now with 100-dimensional random state space
sysrand = mt.rss(100, 1, 1)
rsysrand = msimp.balred(sysrand, 10, method='truncate')

# Comparison of the impulse responses of the full and reduced random systems
plt.figure(2)
yrand, trand = mt.impulse(sysrand)
yrandr, trandr = mt.impulse(rsysrand)
plt.plot(trand.T, yrand.T, trandr.T, yrandr.T)

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
