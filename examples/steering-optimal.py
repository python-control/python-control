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
print("Approach 1: standard quadratic cost")

# Set up the cost functions
Q = np.diag([.1, 10, .1])       # keep lateral error low
R = np.diag([.1, 1])            # minimize applied inputs
quad_cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)

# Define the time horizon (and spacing) for the optimization
horizon = np.linspace(0, Tf, 10, endpoint=True)

# Provide an intial guess (will be extended to entire horizon)
bend_left = [10, 0.01]          # slight left veer

# Turn on debug level logging so that we can see what the optimizer is doing
logging.basicConfig(
    level=logging.DEBUG, filename="steering-integral_cost.log",
    filemode='w', force=True)

# Compute the optimal control, setting step size for gradient calculation (eps)
start_time = time.process_time()
result1 = opt.solve_ocp(
    vehicle, horizon, x0, quad_cost, initial_guess=bend_left, log=True,
    minimize_method='trust-constr',
    minimize_options={'finite_diff_rel_step': 0.01},
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result1.success

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
# We also set the solver explicitly.
#
print("Approach 2: input cost and constraints plus terminal cost")

# Add input constraint, input cost, terminal cost
constraints = [ opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
traj_cost = opt.quadratic_cost(vehicle, None, np.diag([0.1, 1]), u0=uf)
term_cost = opt.quadratic_cost(vehicle, np.diag([1, 10, 10]), None, x0=xf)

# Change logging to keep less information
logging.basicConfig(
    level=logging.INFO, filename="./steering-terminal_cost.log",
    filemode='w', force=True)

# Compute the optimal control
start_time = time.process_time()
result2 = opt.solve_ocp(
    vehicle, horizon, x0, traj_cost, constraints, terminal_cost=term_cost,
    initial_guess=bend_left, log=True,
    minimize_method='SLSQP', minimize_options={'eps': 0.01})
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result2.success

# Extract and plot the results (+ state trajectory)
t2, u2 = result2.time, result2.inputs
t2, y2 = ct.input_output_response(vehicle, horizon, u2, x0)
plot_results(t2, y2, u2, figure=2, yf=xf[0:2])

#
# Approach 3: terminal constraints
#
# We can also remove the cost function on the state and replace it
# with a terminal *constraint* on the state.  If a solution is found,
# it guarantees we get to exactly the final state.
#
print("Approach 3: terminal constraints")

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
    vehicle, horizon, x0, cost3, constraints,
    terminal_constraints=terminal, initial_guess=bend_left, log=False,
    solve_ivp_kwargs={'atol': 1e-3, 'rtol': 1e-2},
    minimize_method='trust-constr',
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result3.success

# Extract and plot the results (+ state trajectory)
t3, u3 = result3.time, result3.inputs
t3, y3 = ct.input_output_response(vehicle, horizon, u3, x0)
plot_results(t3, y3, u3, figure=3, yf=xf[0:2])

#
# Approach 4: terminal constraints w/ basis functions
#
# As a final example, we can use a basis function to reduce the size
# of the problem and get faster answers with more temporal resolution.
# Here we parameterize the input by a set of 4 Bezier curves but solve
# for a much more time resolved set of inputs.

print("Approach 4: Bezier basis")
import control.flatsys as flat

# Compute the optimal control
start_time = time.process_time()
result4 = opt.solve_ocp(
    vehicle, horizon, x0, quad_cost,
    constraints,
    terminal_constraints=terminal,
    initial_guess=bend_left,
    basis=flat.BezierFamily(4, T=Tf),
    # solve_ivp_kwargs={'method': 'RK45', 'atol': 1e-2, 'rtol': 1e-2},
    solve_ivp_kwargs={'atol': 1e-3, 'rtol': 1e-2},
    minimize_method='trust-constr', minimize_options={'disp': True},
    log=False
)
print("* Total time = %5g seconds\n" % (time.process_time() - start_time))

# If we are running CI tests, make sure we succeeded
if 'PYCONTROL_TEST_EXAMPLES' in os.environ:
    assert result4.success

# Extract and plot the results (+ state trajectory)
t4, u4 = result4.time, result4.inputs
t4, y4 = ct.input_output_response(vehicle, horizon, u4, x0)
plot_results(t4, y4, u4, figure=4, yf=xf[0:2])

# If we are not running CI tests, display the results
if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
