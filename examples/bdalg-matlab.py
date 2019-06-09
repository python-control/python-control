# bdalg-matlab.py - demonstrate some MATLAB commands for block diagram altebra
# RMM, 29 May 09

from control.matlab import *    # MATLAB-like functions

# System matrices
A1 = [[0, 1.], [-4, -1]]
B1 = [[0], [1.]]
C1 = [[1., 0]]
sys1ss = ss(A1, B1, C1, 0)
sys1tf = ss2tf(sys1ss)

sys2tf = tf([1, 0.5], [1, 5])
sys2ss = tf2ss(sys2tf)

# Series composition
series1 = sys1ss + sys2ss
