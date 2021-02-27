# optimal_bench.py - benchmarks for optimal control package
# RMM, 27 Feb 2020
#
# This benchmark tests the timing for the optimal control module
# (control.optimal) and is intended to be used for helping tune the
# performance of the functions used for optimization-base control.

import numpy as np
import math
import control as ct
import control.flatsys as flat
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

vehicle = ct.NonlinearIOSystem(
    vehicle_update, vehicle_output, states=3, name='vehicle',
    inputs=('v', 'phi'), outputs=('x', 'y', 'theta'))

x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [100., 2., 0.]; uf = [10., 0.]
Tf = 10

# Define the time horizon (and spacing) for the optimization
horizon = np.linspace(0, Tf, 10, endpoint=True)

# Provide an intial guess (will be extended to entire horizon)
bend_left = [10, 0.01]          # slight left veer

def time_integrated_cost():
    # Set up the cost functions
    Q = np.diag([.1, 10, .1])     # keep lateral error low
    R = np.diag([.1, 1])          # minimize applied inputs
    quad_cost = opt.quadratic_cost(
        vehicle, Q, R, x0=xf, u0=uf)

    res = opt.solve_ocp(
        vehicle, horizon, x0, quad_cost,
        initial_guess=bend_left, print_summary=False,
        # solve_ivp_kwargs={'atol': 1e-2, 'rtol': 1e-2},
        minimize_method='trust-constr',
        minimize_options={'finite_diff_rel_step': 0.01},
    )

    # Only count this as a benchmark if we converged
    assert res.success

def time_terminal_cost():
    # Define cost and constraints
    traj_cost = opt.quadratic_cost(
        vehicle, None, np.diag([0.1, 1]), u0=uf)
    term_cost = opt.quadratic_cost(
        vehicle, np.diag([1, 10, 10]), None, x0=xf)
    constraints = [
        opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]

    res = opt.solve_ocp(
        vehicle, horizon, x0, traj_cost, constraints,
        terminal_cost=term_cost, initial_guess=bend_left, print_summary=False,
        # solve_ivp_kwargs={'atol': 1e-4, 'rtol': 1e-2},
        minimize_method='SLSQP', minimize_options={'eps': 0.01}
    )

    # Only count this as a benchmark if we converged
    assert res.success

def time_terminal_constraint():
    # Input cost and terminal constraints
    R = np.diag([1, 1])                 # minimize applied inputs
    cost = opt.quadratic_cost(vehicle, np.zeros((3,3)), R, u0=uf)
    constraints = [
        opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
    terminal = [ opt.state_range_constraint(vehicle, xf, xf) ]

    res = opt.solve_ocp(
        vehicle, horizon, x0, cost, constraints,
        terminal_constraints=terminal, initial_guess=bend_left, log=False,
        solve_ivp_kwargs={'atol': 1e-3, 'rtol': 1e-2},
        # solve_ivp_kwargs={'atol': 1e-4, 'rtol': 1e-2},
        minimize_method='trust-constr',
        # minimize_method='SLSQP', minimize_options={'eps': 0.01}
    )

    # Only count this as a benchmark if we converged
    assert res.success

# Reset the timeout value to allow for longer runs
time_terminal_constraint.timeout = 120

def time_optimal_basis_vehicle():
    # Set up costs and constriants
    Q = np.diag([.1, 10, .1])           # keep lateral error low
    R = np.diag([1, 1])                 # minimize applied inputs
    cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    constraints = [ opt.input_range_constraint(vehicle, [0, -0.1], [20, 0.1]) ]
    terminal = [ opt.state_range_constraint(vehicle, xf, xf) ]
    bend_left = [10, 0.05]              # slight left veer
    near_optimal = [
        [ 1.15073736e+01,  1.16838616e+01,  1.15413395e+01,
          1.11585544e+01,  1.06142537e+01,  9.98718468e+00,
          9.35609454e+00,  8.79973057e+00,  8.39684004e+00,
          8.22617023e+00],
        [ -9.99830506e-02,  8.98139594e-03,  5.26385615e-02,
          4.96635954e-02,  1.87316470e-02, -2.14821345e-02,
          -5.23025996e-02, -5.50545990e-02, -1.10629834e-02,
          9.83473965e-02] ]

    # Set up horizon
    horizon = np.linspace(0, Tf, 10, endpoint=True)

    # Set up the optimal control problem
    res = opt.solve_ocp(
        vehicle, horizon, x0, cost,
        constraints,
        terminal_constraints=terminal,
        initial_guess=near_optimal,
        basis=flat.BezierFamily(4, T=Tf),
        minimize_method='trust-constr',
        # minimize_method='SLSQP', minimize_options={'eps': 0.01},
        solve_ivp_kwargs={'atol': 1e-4, 'rtol': 1e-2},
        return_states=True, print_summary=False
    )
    t, u, x = res.time, res.inputs, res.states

    # Make sure we found a valid solution
    assert res.success
    np.testing.assert_almost_equal(x[:, -1], xf, decimal=4)

def time_mpc_iosystem():
    # model of an aircraft discretized with 0.2s sampling time
    # Source: https://www.mpt3.org/UI/RegulationProblem
    A = [[0.99, 0.01, 0.18, -0.09,   0],
         [   0, 0.94,    0,  0.29,   0],
         [   0, 0.14, 0.81,  -0.9,   0],
         [   0, -0.2,    0,  0.95,   0],
         [   0, 0.09,    0,     0, 0.9]]
    B = [[ 0.01, -0.02],
         [-0.14,     0],
         [ 0.05,  -0.2],
         [ 0.02,     0],
         [-0.01, 0]]
    C = [[0, 1, 0, 0, -1],
         [0, 0, 1, 0,  0],
         [0, 0, 0, 1,  0],
         [1, 0, 0, 0,  0]]
    model = ct.ss2io(ct.ss(A, B, C, 0, 0.2))

    # For the simulation we need the full state output
    sys = ct.ss2io(ct.ss(A, B, np.eye(5), 0, 0.2))

    # compute the steady state values for a particular value of the input
    ud = np.array([0.8, -0.3])
    xd = np.linalg.inv(np.eye(5) - A) @ B @ ud
    yd = C @ xd

    # provide constraints on the system signals
    constraints = [opt.input_range_constraint(sys, [-5, -6], [5, 6])]

    # provide penalties on the system signals
    Q = model.C.transpose() @ np.diag([10, 10, 10, 10]) @ model.C
    R = np.diag([3, 2])
    cost = opt.quadratic_cost(model, Q, R, x0=xd, u0=ud)

    # online MPC controller object is constructed with a horizon 6
    ctrl = opt.create_mpc_iosystem(
        model, np.arange(0, 6) * 0.2, cost, constraints)

    # Define an I/O system implementing model predictive control
    loop = ct.feedback(sys, ctrl, 1)

    # Choose a nearby initial condition to speed up computation
    X0 = np.hstack([xd, np.kron(ud, np.ones(6))]) * 0.99

    Nsim = 12
    tout, xout = ct.input_output_response(
        loop, np.arange(0, Nsim) * 0.2, 0, X0)

    # Make sure the system converged to the desired state
    np.testing.assert_allclose(
        xout[0:sys.nstates, -1], xd, atol=0.1, rtol=0.01)
