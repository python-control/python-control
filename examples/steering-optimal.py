# steering-optimal.py - optimal control for vehicle steering
# RMM, 18 Feb 2021
#
# This file works through an optimal control example for the vehicle
# steering system.  It is intended to demonstrate the functionality for
# optimal control module (control.optimal) in the python-control package.

import numpy as np
import math
import control as ct
import control.optimal as opt
import matplotlib.pyplot as plt
import logging
import time
import os

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

    # Saturate the steering input (use min/max instead of clip for speed)
    phi = max(-phimax, min(u[1], phimax))

    # Return the derivative of the state
    return np.array([
        math.cos(x[2]) * u[0],            # xdot = cos(theta) v
        math.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * math.tan(phi)        # thdot = v/l tan(phi)
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
def plot_lanechange(t, y, u, yf=None, figure=None):
    plt.figure(figure)

    # Plot the xy trajectory
    plt.subplot(3, 1, 1)
    plt.plot(y[0], y[1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if yf is not None:
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
x0 = np.array([0., -2., 0.]); u0 = np.array([10., 0.])
xf = np.array([100., 2., 0.]); uf = np.array([10., 0.])
Tf = 10

#
# Approach 1: standard quadratic cost
#
# We can set up the optimal control problem as trying to minimize the
# distance form the desired final point while at the same time as not
# exerting too much control effort to achieve our goal.
#
print("Approach 1: standard quadratic cost")

# Set up the cost functions
Q = np.diag([.1, 10, .1])       # keep lateral error low
R = np.diag([.1, 1])            # minimize applied inputs
quad_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)

# Define the time horizon (and spacing) for the optimization
timepts = np.linspace(0, Tf, 20, endpoint=True)

# Provide an initial guess
straight_line = (
    np.array([x0 + (xf - x0) * time/Tf for time in timepts]).transpose(),
    np.outer(u0, np.ones_like(timepts))
)

# Turn on debug level logging so that we can see what the optimizer is doing
logging.basicConfig(
    level=logging.DEBUG, filename="steering-integral_cost.log",
    filemode='w', force=True)

# Compute the optimal control, setting step size for gradient calculation (eps)
start_time = time.process_time()
result1 = opt.solve_ocp(
    vehicle, timepts, x0, quad_cost, initial_guess=straight_line, log=True,
    # minimize_method='trust-constr',
    # minimize_options={'finite_diff_rel_step': 0.01},
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result1.success

# Plot the results from the optimization
plot_lanechange(timepts, result1.states, result1.inputs, xf, figure=1)
print("Final computed state: ", result1.states[:,-1])

# Simulate the system and see what happens
t1, u1 = result1.time, result1.inputs
t1, y1 = ct.input_output_response(vehicle, timepts, u1, x0)
plot_lanechange(t1, y1, u1, yf=xf[0:2], figure=1)
print("Final simulated state:", y1[:,-1])

#
# Approach 2: input cost, input constraints, terminal cost
#
# The previous solution integrates the position error for the entire
# horizon, and so the car changes lanes very quickly (at the cost of larger
# inputs).  Instead, we can penalize the final state and impose a higher
# cost on the inputs, resuling in a more graduate lane change.
#
# We also set the solver explicitly.
#
print("\nApproach 2: input cost and constraints plus terminal cost")

# Add input constraint, input cost, terminal cost
constraints = [ opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
traj_cost = opt.quadratic_cost(vehicle, None, np.diag([0.1, 1]), u0=uf)
term_cost = opt.quadratic_cost(vehicle, np.diag([1, 10, 10]), None, x0=xf)

# Change logging to keep less information
logging.basicConfig(
    level=logging.INFO, filename="./steering-terminal_cost.log",
    filemode='w', force=True)

# Use a straight line between initial and final position as initial guesss
input_guess = np.outer(u0, np.ones((1, timepts.size)))
state_guess = np.array([
    x0 + (xf - x0) * time/Tf for time in timepts]).transpose()
straight_line = (state_guess, input_guess)

# Compute the optimal control
start_time = time.process_time()
result2 = opt.solve_ocp(
    vehicle, timepts, x0, traj_cost, constraints, terminal_cost=term_cost,
    initial_guess=straight_line, log=True,
    # minimize_method='SLSQP', minimize_options={'eps': 0.01}
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result2.success

# Plot the results from the optimization
plot_lanechange(timepts, result2.states, result2.inputs, xf, figure=2)
print("Final computed state: ", result2.states[:,-1])

# Simulate the system and see what happens
t2, u2 = result2.time, result2.inputs
t2, y2 = ct.input_output_response(vehicle, timepts, u2, x0)
plot_lanechange(t2, y2, u2, yf=xf[0:2], figure=2)
print("Final simulated state:", y2[:,-1])

#
# Approach 3: terminal constraints
#
# We can also remove the cost function on the state and replace it
# with a terminal *constraint* on the state.  If a solution is found,
# it guarantees we get to exactly the final state.
#
print("\nApproach 3: terminal constraints")

# Input cost and terminal constraints
R = np.diag([1, 1])                 # minimize applied inputs
cost3 = opt.quadratic_cost(vehicle, np.zeros((3,3)), R, u0=uf)
constraints = [
    opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
terminal = [ opt.state_range_constraint(vehicle, xf, xf) ]

# Reset logging to its default values
logging.basicConfig(
    level=logging.DEBUG, filename="./steering-terminal_constraint.log",
    filemode='w', force=True)

# Compute the optimal control
start_time = time.process_time()
result3 = opt.solve_ocp(
    vehicle, timepts, x0, cost3, constraints,
    terminal_constraints=terminal, initial_guess=straight_line, log=False,
    # solve_ivp_kwargs={'atol': 1e-3, 'rtol': 1e-2},
    # minimize_method='trust-constr',
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result3.success

# Plot the results from the optimization
plot_lanechange(timepts, result3.states, result3.inputs, xf, figure=3)
print("Final computed state: ", result3.states[:,-1])

# Simulate the system and see what happens
t3, u3 = result3.time, result3.inputs
t3, y3 = ct.input_output_response(vehicle, timepts, u3, x0)
plot_lanechange(t3, y3, u3, yf=xf[0:2], figure=3)
print("Final simulated state:", y3[:,-1])

#
# Approach 4: terminal constraints w/ basis functions
#
# As a final example, we can use a basis function to reduce the size
# of the problem and get faster answers with more temporal resolution.
# Here we parameterize the input by a set of 4 Bezier curves but solve
# for a much more time resolved set of inputs.

print("\nApproach 4: Bezier basis")
import control.flatsys as flat

# Compute the optimal control
start_time = time.process_time()
result4 = opt.solve_ocp(
    vehicle, timepts, x0, quad_cost,
    constraints,
    terminal_constraints=terminal,
    initial_guess=straight_line,
    basis=flat.BezierFamily(6, T=Tf),
    # solve_ivp_kwargs={'method': 'RK45', 'atol': 1e-2, 'rtol': 1e-2},
    # solve_ivp_kwargs={'atol': 1e-3, 'rtol': 1e-2},
    # minimize_method='trust-constr', minimize_options={'disp': True},
    log=False
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result4.success

# Plot the results from the optimization
plot_lanechange(timepts, result4.states, result4.inputs, xf, figure=4)
print("Final computed state: ", result3.states[:,-1])

# Simulate the system and see what happens
t4, u4 = result4.time, result4.inputs
t4, y4 = ct.input_output_response(vehicle, timepts, u4, x0)
plot_lanechange(t4, y4, u4, yf=xf[0:2], figure=4)
print("Final simulated state: ", y4[:,-1])

# If we are not running CI tests, display the results
if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
