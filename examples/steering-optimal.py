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

#
# Approach 1: standard quadratic cost
#
# We can set up the optimal control problem as trying to minimize the
# distance form the desired final point while at the same time as not
# exerting too much control effort to achieve our goal.
#
# Note: depending on what version of SciPy you are using, you might get a
# warning message about precision loss, but the solution is pretty good.
#

# Set up the cost functions
Q = np.diag([1, 10, 1])     # keep lateral error low
R = np.diag([1, 1])         # minimize applied inputs
cost1 = obc.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)

# Define the time horizon (and spacing) for the optimization
horizon = np.linspace(0, Tf, 10, endpoint=True)

# Provide an intial guess (will be extended to entire horizon)
bend_left = [10, 0.01]          # slight left veer

# Turn on debug level logging so that we can see what the optimizer is doing
logging.basicConfig(
    level=logging.DEBUG, filename="steering-integral_cost.log",
    filemode='w', force=True)

# Compute the optimal control, setting step size for gradient calculation (eps)
result1 = obc.compute_optimal_input(
    vehicle, horizon, x0, cost1, initial_guess=bend_left, log=True,
    options={'eps': 0.01})

# Extract and plot the results (+ state trajectory)
t1, u1 = result1.time, result1.inputs
t1, y1 = ct.input_output_response(vehicle, horizon, u1, x0)
plot_results(t1, y1, u1, figure=1, yf=xf[0:2])

#
# Approach 2: input cost, input constraints, terminal cost
#
# The previous solution integrates the position error for the entire
# horizon, and so the car changes lanes very quickly (at the cost of larger
# inputs).  Instead, we can penalize the final state and impose a higher
# cost on the inputs, resuling in a more graduate lane change.
#
# We also set the solver explicitly (its actually the default one, but shows
# how to do this).
#

# Add input constraint, input cost, terminal cost
constraints = [ obc.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
traj_cost = obc.quadratic_cost(vehicle, None, np.diag([0.1, 1]), u0=uf)
term_cost = obc.quadratic_cost(vehicle, np.diag([1, 10, 10]), None, x0=xf)

# Change logging to keep less information
logging.basicConfig(
    level=logging.INFO, filename="./steering-terminal_cost.log",
    filemode='w', force=True)

# Compute the optimal control
result2 = obc.compute_optimal_input(
    vehicle, horizon, x0, traj_cost, constraints, terminal_cost=term_cost,
    initial_guess=bend_left, log=True,
    method='SLSQP', options={'eps': 0.01})

# Extract and plot the results (+ state trajectory)
t2, u2 = result2.time, result2.inputs
t2, y2 = ct.input_output_response(vehicle, horizon, u2, x0)
plot_results(t2, y2, u2, figure=2, yf=xf[0:2])

#
# Approach 3: terminal constraints and new solver
#
# As a final example, we can remove the cost function on the state and
# replace it with a terminal *constraint* on the state.  If a solution is
# found, it guarantees we get to exactly the final state.
#
# To speeds things up a bit, we initalize the problem using the previous
# optimal controller (which didn't quite hit the final value).
#

# Input cost and terminal constraints
cost3 = obc.quadratic_cost(vehicle, np.zeros((3,3)), R, u0=uf)
terminal = [ obc.state_range_constraint(vehicle, xf, xf) ]

# Reset logging to its default values
logging.basicConfig(level=logging.WARN, force=True)

# Compute the optimal control
result3 = obc.compute_optimal_input(
    vehicle, horizon, x0, cost3, constraints,
    terminal_constraints=terminal, initial_guess=u2, log=True,
    options={'eps': 0.01})

# Extract and plot the results (+ state trajectory)
t3, u3 = result3.time, result3.inputs
t3, y3 = ct.input_output_response(vehicle, horizon, u3, x0)
plot_results(t3, y3, u3, figure=3, yf=xf[0:2])
