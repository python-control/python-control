# steering-gainsched.py - gain scheduled control for vehicle steering
# RMM, 8 May 2019
#
# This file works through Example 1.1 in the "Optimization-Based Control"
# course notes by Richard Murray (avaliable at http://fbsbook.org, in the
# optimization-based control supplement).  It is intended to demonstrate the
# functionality for nonlinear input/output systems in the python-control
# package.

import numpy as np
import control as ct
from cmath import sqrt
import matplotlib.pyplot as plt

#
# Vehicle steering dynamics
#
# The vehicle dynamics are given by a simple bicycle model.  We take the state
# of the system as (x, y, theta) where (x, y) is the position of the vehicle
# in the plane and theta is the angle of the vehicle with respect to
# horizontal.  The vehicle input is given by (v, phi) where v is the forward
# velocity of the vehicle and phi is the angle of the steering wheel.  The
# model includes saturation of the vehicle steering angle.
#
# System state: x, y, theta
# System input: v, phi
# System output: x, y
# System parameters: wheelbase, maxsteer
#
def vehicle_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input
    phi = np.clip(u[1], -phimax, phimax)

    # Return the derivative of the state
    return np.array([
        np.cos(x[2]) * u[0],            # xdot = cos(theta) v
        np.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * np.tan(phi)        # thdot = v/l tan(phi)
    ])

def vehicle_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Define the vehicle steering dynamics as an input/output system
vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'phi'),
    outputs=('x', 'y', 'theta'))

#
# Gain scheduled controller
#
# For this system we use a simple schedule on the forward vehicle velocity and
# place the poles of the system at fixed values.  The controller takes the
# current vehicle position and orientation plus the velocity velocity as
# inputs, and returns the velocity and steering commands.
#
# System state: none
# System input: ex, ey, etheta, vd, phid
# System output: v, phi
# System parameters: longpole, latpole1, latpole2
#
def control_output(t, x, u, params):
    # Get the controller parameters
    longpole = params.get('longpole', -2.)
    latpole1 = params.get('latpole1', -1/2 + sqrt(-7)/2)
    latpole2 = params.get('latpole2', -1/2 - sqrt(-7)/2)
    l = params.get('wheelbase', 3)
    
    # Extract the system inputs
    ex, ey, etheta, vd, phid = u

    # Determine the controller gains
    alpha1 = -np.real(latpole1 + latpole2)
    alpha2 = np.real(latpole1 * latpole2)

    # Compute and return the control law
    v = -longpole * ex          # Note: no feedfwd (to make plot interesting)
    if vd != 0:
        phi = phid + (alpha1 * l) / vd * ey + (alpha2 * l) / vd * etheta
    else:
        # We aren't moving, so don't turn the steering wheel
        phi = phid
    
    return  np.array([v, phi])

# Define the controller as an input/output system
controller = ct.NonlinearIOSystem(
    None, control_output, name='controller',        # static system
    inputs=('ex', 'ey', 'etheta', 'vd', 'phid'),    # system inputs
    outputs=('v', 'phi')                            # system outputs
)

#
# Reference trajectory subsystem
#
# The reference trajectory block generates a simple trajectory for the system
# given the desired speed (vref) and lateral position (yref).  The trajectory
# consists of a straight line of the form (vref * t, yref, 0) with nominal
# input (vref, 0).
#
# System state: none
# System input: vref, yref
# System output: xd, yd, thetad, vd, phid
# System parameters: none
#
def trajgen_output(t, x, u, params):
    vref, yref = u
    return np.array([vref * t, yref, 0, vref, 0])

# Define the trajectory generator as an input/output system
trajgen = ct.NonlinearIOSystem(
    None, trajgen_output, name='trajgen',
    inputs=('vref', 'yref'),
    outputs=('xd', 'yd', 'thetad', 'vd', 'phid'))

#
# System construction
#
# The input to the full closed loop system is the desired lateral position and
# the desired forward velocity.  The output for the system is taken as the
# full vehicle state plus the velocity of the vehicle.  The following diagram
# summarizes the interconnections:
#
#                        +---------+       +---------------> v
#                        |         |       |
# [ yref ]               |         v       |
# [      ] ---> trajgen -+-+-> controller -+-> vehicle -+-> [x, y, theta]
# [ vref ]                 ^                            |
#                          |                            |
#                          +----------- [-1] -----------+
#
# We construct the system using the InterconnectedSystem constructor and using
# signal labels to keep track of everything.  

steering = ct.interconnect(
    # List of subsystems
    (trajgen, controller, vehicle), name='steering',

    # Interconnections between  subsystems
    connections=(
        ['controller.ex', 'trajgen.xd', '-vehicle.x'],
        ['controller.ey', 'trajgen.yd', '-vehicle.y'],
        ['controller.etheta', 'trajgen.thetad', '-vehicle.theta'],
        ['controller.vd', 'trajgen.vd'],
        ['controller.phid', 'trajgen.phid'],
        ['vehicle.v', 'controller.v'],
        ['vehicle.phi', 'controller.phi']
    ),

    # System inputs
    inplist=['trajgen.vref', 'trajgen.yref'],
    inputs=['yref', 'vref'],

    #  System outputs
    outlist=['vehicle.x', 'vehicle.y', 'vehicle.theta', 'controller.v',
             'controller.phi'],
    outputs=['x', 'y', 'theta', 'v', 'phi']
)

# Set up the simulation conditions
yref = 1
T = np.linspace(0, 5, 100)

# Set up a figure for plotting the results
plt.figure();

# Plot the reference trajectory for the y position
plt.plot([0, 5], [yref, yref], 'k--')

# Find the signals we want to plot
y_index = steering.find_output('y')
v_index = steering.find_output('v')

# Do an iteration through different speeds
for vref in [8, 10, 12]:
    # Simulate the closed loop controller response
    tout, yout = ct.input_output_response(
        steering, T, [vref * np.ones(len(T)), yref * np.ones(len(T))])

    # Plot the reference speed
    plt.plot([0, 5], [vref, vref], 'k--')

    # Plot the system output
    y_line, = plt.plot(tout, yout[y_index, :], 'r')  # lateral position
    v_line, = plt.plot(tout, yout[v_index, :], 'b')  # vehicle velocity

# Add axis labels
plt.xlabel('Time (s)')
plt.ylabel('x vel (m/s), y pos (m)')
plt.legend((v_line, y_line), ('v', 'y'), loc='center right', frameon=False)
