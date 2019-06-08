# type2_type3.py - demonstration for type2 versus type3 control comparing
#   tracking and disturbance rejection for two proposed controllers
# Gunnar Ristroph, 15 January 2010

import os
import matplotlib.pyplot as plt  # Grab MATLAB plotting functions
from control.matlab import *     # MATLAB-like functions
from scipy import pi
integrator = tf([0, 1], [1, 0])  # 1/s

# Parameters defining the system
J = 1.0
b = 10.0
Kp = 110.
Ki = Kp/2.
Kii = Ki

# Plant transfer function from torque to rate
inertia = integrator*1/J
friction = b  # transfer function from rate to torque
P = inertia  # friction is modelled as a separate block

# Gyro transfer function from rate to rate
gyro = 1.  # for now, our gyro is perfect

# Controller transfer function from rate error to torque
C_type2 = (1. + Ki*integrator)*Kp*1.5
C_type3 = (1. + Ki*integrator)*(1. + Kii*integrator)*Kp

# System Transfer Functions
# tricky because the disturbance (base motion) is coupled in by friction
closed_loop_type2 = feedback(C_type2*feedback(P, friction), gyro)
disturbance_rejection_type2 = P*friction/(1. + P*friction+P*C_type2)
closed_loop_type3 = feedback(C_type3*feedback(P, friction), gyro)
disturbance_rejection_type3 = P*friction/(1. + P*friction + P*C_type3)

# Bode plot for the system
plt.figure(1)
bode(closed_loop_type2, logspace(0, 2)*2*pi, dB=True, Hz=True)  # blue
bode(closed_loop_type3, logspace(0, 2)*2*pi, dB=True, Hz=True)  # green
plt.show(block=False)

plt.figure(2)
bode(disturbance_rejection_type2, logspace(0, 2)*2*pi, Hz=True)  # blue
bode(disturbance_rejection_type3, logspace(0, 2)*2*pi, Hz=True)  # green

if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
