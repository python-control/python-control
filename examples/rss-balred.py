#!/usr/bin/env python

import numpy as np
import control.modelsimp as msimp
import control.matlab as mt
from control.statesp import StateSpace
import matplotlib.pyplot as plt

plt.close('all')

#controlable canonical realization computed in matlab for the transfer function:
# num = [1 11 45 32], den = [1 15 60 200 60]
A = np.matrix('-15., -7.5, -6.25, -1.875; \
8., 0., 0., 0.; \
0., 4., 0., 0.; \
0., 0., 1., 0.')
B = np.matrix('2.; 0.; 0.; 0.')
C = np.matrix('0.5, 0.6875, 0.7031, 0.5')
D = np.matrix('0.')

# The full system
fsys = StateSpace(A,B,C,D)
# The reduced system, truncating the order by 1
ord = 3
rsys = msimp.balred(fsys,ord, method = 'truncate')

# Comparison of the step responses of the full and reduced systems
plt.figure(1)
y, t = mt.step(fsys)
yr, tr = mt.step(rsys)
plt.plot(t.T, y.T)
plt.hold(True)
plt.plot(tr.T, yr.T)

# Repeat balanced reduction, now with 100-dimensional random state space
sysrand = mt.rss(100, 1, 1)
rsysrand = msimp.balred(sysrand,10,method ='truncate')

# Comparison of the impulse responses of the full and reduced random systems
plt.figure(2)
yrand, trand = mt.impulse(sysrand)
yrandr, trandr = mt.impulse(rsysrand)
plt.plot(trand.T, yrand.T, trandr.T, yrandr.T) 

