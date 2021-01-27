# test-response.py - Unit tests for system response functions
# RMM, 11 Sep 2010

import os
import matplotlib.pyplot as plt  # MATLAB plotting functions
from control.matlab import *    # Load the controls systems library
from numpy import arange        # function to create range of numbers

from control import reachable_form

# Create several systems for testing
sys1 = tf([1], [1, 2, 1])
sys2 = tf([1, 1], [1, 1, 0])

# Generate step responses
(y1a, T1a) = step(sys1)
(y1b, T1b) = step(sys1, T=arange(0, 10, 0.1))
# convert to reachable canonical SS to specify initial state
sys1_ss = reachable_form(ss(sys1))[0]
(y1c, T1c) = step(sys1_ss, X0=[1, 0])
(y2a, T2a) = step(sys2, T=arange(0, 10, 0.1))

plt.plot(T1a, y1a, label='$g_1$ (default)', linewidth=5)
plt.plot(T1b, y1b, label='$g_1$ (w/ spec. times)', linestyle='--')
plt.plot(T1c, y1c, label='$g_1$ (w/ init cond.)')
plt.plot(T2a, y2a, label='$g_2$ (w/ spec. times)')
plt.xlabel('time')
plt.ylabel('output')
plt.legend()

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
