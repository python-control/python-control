# kincar-flatsys.py - differentially flat systems example
# RMM, 3 Jul 2019
#
# This example demonstrates the use of the `flatsys` module for generating
# trajectories for differnetially flat systems by computing a trajectory for a
# kinematic (bicycle) model of a car changing lanes.

import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
import control.optimal as opt

#
# System model and utility functions
#

# Function to take states, inputs and return the flat flag
def vehicle_flat_forward(x, u, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)

    # Create a list of arrays to store the flat output and its derivatives
    zflag = [np.zeros(3), np.zeros(3)]

    # Flat output is the x, y position of the rear wheels
    zflag[0][0] = x[0]
    zflag[1][0] = x[1]

    # First derivatives of the flat output
    zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt
    zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

    # First derivative of the angle
    thdot = (u[0]/b) * np.tan(u[1])

    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])

    return zflag


# Function to take the flat flag and return states, inputs
def vehicle_flat_reverse(zflag, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)

    # Create a vector to store the state and inputs
    x = np.zeros(3)
    u = np.zeros(2)

    # Given the flat variables, solve for the state
    x[0] = zflag[0][0]  # x position
    x[1] = zflag[1][0]  # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot

    # And next solve for the inputs
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)

    return x, u

# Function to compute the RHS of the system dynamics
def vehicle_update(t, x, u, params):
    b = params.get('wheelbase', 3.)             # get parameter values
    dx = np.array([
        np.cos(x[2]) * u[0],
        np.sin(x[2]) * u[0],
        (u[0]/b) * np.tan(u[1])
    ])
    return dx

# Plot the trajectory in xy coordinates
def plot_results(t, x, ud, rescale=True):
    plt.subplot(4, 1, 2)
    plt.plot(x[0], x[1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    if rescale:
        plt.axis([x0[0], xf[0], x0[1]-1, xf[1]+1])

    # Time traces of the state and input
    plt.subplot(2, 4, 5)
    plt.plot(t, x[1])
    plt.ylabel('y [m]')

    plt.subplot(2, 4, 6)
    plt.plot(t, x[2])
    plt.ylabel('theta [rad]')

    plt.subplot(2, 4, 7)
    plt.plot(t, ud[0])
    plt.xlabel('Time t [sec]')
    plt.ylabel('v [m/s]')
    if rescale:
        plt.axis([0, Tf, u0[0] - 1, uf[0] + 1])

    plt.subplot(2, 4, 8)
    plt.plot(t, ud[1])
    plt.xlabel('Ttime t [sec]')
    plt.ylabel('$\delta$ [rad]')
    plt.tight_layout()

#
# Approach 1: point to point solution, no cost or constraints
#

# Create differentially flat input/output system
vehicle_flat = fs.FlatSystem(
    vehicle_flat_forward, vehicle_flat_reverse, vehicle_update,
    inputs=('v', 'delta'), outputs=('x', 'y'),
    states=('x', 'y', 'theta'))

# Define the endpoints of the trajectory
x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [40., 2., 0.]; uf = [10., 0.]
Tf = 4

# Define a set of basis functions to use for the trajectories
poly = fs.PolyFamily(6)

# Find a trajectory between the initial condition and the final condition
traj1 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)

# Create the desired trajectory between the initial and final condition
T = np.linspace(0, Tf, 500)
xd, ud = traj1.eval(T)

# Simulation the open system dynamics with the full input
t, y, x = ct.input_output_response(
    vehicle_flat, T, ud, x0, return_x=True)

# Plot the open loop system dynamics
plt.figure(1)
plt.suptitle("Open loop trajectory for kinematic car lane change")
plot_results(t, x, ud)

#
# Approach #2: add cost function to make lane change quicker
#

# Define timepoints for evaluation plus basis function to use
timepts = np.linspace(0, Tf, 10)
basis = fs.PolyFamily(8)

# Define the cost function (penalize lateral error and steering)
traj_cost = opt.quadratic_cost(
    vehicle_flat, np.diag([0, 0.1, 0]), np.diag([0.1, 1]), x0=xf, u0=uf)

# Solve for an optimal solution
traj2 = fs.point_to_point(
    vehicle_flat, timepts, x0, u0, xf, uf, cost=traj_cost, basis=basis,
)
xd, ud = traj2.eval(T)

plt.figure(2)
plt.suptitle("Lane change with lateral error + steering penalties")
plot_results(T, xd, ud)

#
# Approach #3: optimal cost with trajectory constraints
#
# Resolve the problem with constraints on the inputs
#

# Constraint the input values
constraints = [
    opt.input_range_constraint(vehicle_flat, [8, -0.1], [12, 0.1]) ]

# TEST: Change the basis to use B-splines
basis = fs.BSplineFamily([0, Tf/2, Tf], 6)

# Solve for an optimal solution
traj3 = fs.point_to_point(
    vehicle_flat, timepts, x0, u0, xf, uf, cost=traj_cost,
    constraints=constraints, basis=basis,
)
xd, ud = traj3.eval(T)

plt.figure(3)
plt.suptitle("Lane change with penalty + steering constraints")
plot_results(T, xd, ud)

# Show the results unless we are running in batch mode
if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()


#
# Approach #4: optimal trajectory, final cost with trajectory constraints
#
# Resolve the problem with constraints on the inputs and also replacing the
# point to point problem with one using a terminal cost to set the final
# state.
#

# Define the cost function (mainly penalize steering angle)
traj_cost = opt.quadratic_cost(
    vehicle_flat, None, np.diag([0.1, 10]), x0=xf, u0=uf)

# Set terminal cost to bring us close to xf
terminal_cost = opt.quadratic_cost(vehicle_flat, 1e3 * np.eye(3), None, x0=xf)

# Change the basis to use B-splines
basis = fs.BSplineFamily([0, Tf/2, Tf], [4, 6], vars=2)

# Use a straight line as an initial guess for the trajectory
initial_guess = np.array(
    [x0[i] + (xf[i] - x0[i]) * timepts/Tf for i in (0, 1)])

# Solve for an optimal solution
traj4 = fs.solve_flat_ocp(
    vehicle_flat, timepts, x0, u0, cost=traj_cost, constraints=constraints,
    terminal_cost=terminal_cost, basis=basis, initial_guess=initial_guess,
    # minimize_kwargs={'method': 'trust-constr'},
)
xd, ud = traj4.eval(T)

plt.figure(4)
plt.suptitle("Lane change with terminal cost + steering constraints")
plot_results(T, xd, ud, rescale=False)  # TODO: remove rescale

# Show the results unless we are running in batch mode
if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
    plt.show()
