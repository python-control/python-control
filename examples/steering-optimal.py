# steering-optimal.py - optimal control for vehicle steering
# RMM, 18 Feb 2021
#
# This file works through an optimal control example for the vehicle
# steering system.  It is intended to demonstrate the functionality
# for optimization-based control (obc) module in the python-control
# package.

import numpy as np
import control as ct
import control.obc as obc
import matplotlib.pyplot as plt
import logging

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
# Utility function to plot the results
#
def plot_results(t, y, u, figure=None, yf=None):
    plt.figure(figure)

    # Plot the xy trajectory
    plt.subplot(3, 1, 1)
    plt.plot(y[0], y[1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if yf:
        plt.plot(yf[0], yf[1], 'ro')

    # Plot the inputs as a function of time
    plt.subplot(3, 1, 2)
    plt.plot(t, u[0])
    plt.xlabel("t [sec]")
    plt.ylabel("velocity [m/s]")

    plt.subplot(3, 1, 3)
    plt.plot(t, u[1])
    plt.xlabel("t [sec]")
    plt.ylabel("steering [rad/s]")

    plt.suptitle("Lane change manuever")
    plt.tight_layout()
    plt.show(block=False)

#
# Optimal control problem
#
# Perform a "lane change" manuever over the course of 10 seconds.
#

# Initial and final conditions
x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [100., 2., 0.]; uf = [10., 0.]
Tf = 10

# Set up the cost functions
Q = np.diag([0.1, 1, 0.1])      # keep lateral error low
R = np.eye(2)                   # minimize applied inputs
cost = obc.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)

#
# Set up different types of constraints to demonstrate
#

# Input constraints
constraints = [ obc.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]

# Terminal constraints (optional)
terminal = [ obc.state_range_constraint(vehicle, xf, xf) ]

# Time horizon and possible initial guessses
horizon = np.linspace(0, Tf, 10, endpoint=True)
straight = [10, 0]              # straight trajectory
bend_left = [10, 0.01]          # slight left veer

#
# Solve the optimal control problem in dififerent ways
#

# Basic setup: quadratic cost, no terminal constraint, straight initial path
logging.basicConfig(
    level=logging.DEBUG, filename="steering-straight.log",
    filemode='w', force=True)
result = obc.compute_optimal_input(
    vehicle, horizon, x0, cost, initial_guess=straight,
    log=True, options={'eps': 0.01})
t1, u1 = result.time, result.inputs
t1, y1 = ct.input_output_response(vehicle, horizon, u1, x0)
plot_results(t1, y1, u1, figure=1, yf=xf[0:2])

# Add constraint on the input to avoid high steering angles
logging.basicConfig(
    level=logging.INFO, filename="./steering-bendleft.log",
    filemode='w', force=True)
result = obc.compute_optimal_input(
    vehicle, horizon, x0, cost, constraints, initial_guess=bend_left,
    log=True, options={'eps': 0.01})
t2, u2 = result.time, result.inputs
t2, y2 = ct.input_output_response(vehicle, horizon, u2, x0)
plot_results(t2, y2, u2, figure=2, yf=xf[0:2])

# Resolve with a terminal constraint (starting with previous result)
logging.basicConfig(
    level=logging.WARN, filename="./steering-terminal.log",
    filemode='w', force=True)
result = obc.compute_optimal_input(
    vehicle, horizon, x0, cost, constraints,
    terminal_constraints=terminal, initial_guess=u2,
    log=True, options={'eps': 0.01})
t3, u3 = result.time, result.inputs
t3, y3 = ct.input_output_response(vehicle, horizon, u3, x0)
plot_results(t3, y3, u3, figure=3, yf=xf[0:2])
