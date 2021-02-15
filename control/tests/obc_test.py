"""obc_test.py - tests for optimization based control

RMM, 17 Apr 2019 check the functionality for optimization based control.
RMM, 30 Dec 2020 convert to pytest
"""

import pytest
import warnings
import numpy as np
import scipy as sp
import control as ct
import control.obc as obc
from control.tests.conftest import slycotonly

def test_finite_horizon_mpc_simple():
    # Define a linear system with constraints
    # Source: https://www.mpt3.org/UI/RegulationProblem

    # LTI prediction model
    sys = ct.ss2io(ct.ss([[1, 1], [0, 1]], [[1], [0.5]], np.eye(2), 0, 1))

    # State and input constraints
    constraints = [
        (sp.optimize.LinearConstraint, np.eye(3), [-5, -5, -1], [5, 5, 1]),
    ]

    # Quadratic state and input penalty
    Q = [[1, 0], [0, 1]]
    R = [[1]]
    cost = obc.quadratic_cost(sys, Q, R)

    # Create a model predictive controller system
    time = np.arange(0, 5, 1)
    optctrl = obc.OptimalControlProblem(sys, time, cost, constraints)
    mpc = optctrl.mpc

    # Optimal control input for a given value of the initial state
    x0 = [4, 0]
    u = mpc(x0)
    np.testing.assert_almost_equal(u, -1)

    # Retrieve the full open-loop predictions
    t, u_openloop = optctrl.compute_trajectory(x0, squeeze=True)
    np.testing.assert_almost_equal(
        u_openloop, [-1, -1, 0.1393, 0.3361, -5.204e-16], decimal=4)

    # Convert controller to an explicit form (not implemented yet)
    # mpc_explicit = obc.explicit_mpc();

    # Test explicit controller
    # u_explicit = mpc_explicit(x0)
    # np.testing.assert_array_almost_equal(u_openloop, u_explicit)


@slycotonly
def test_finite_horizon_mpc_oscillator():
    # oscillator model defined in 2D
    # Source: https://www.mpt3.org/UI/RegulationProblem
    A = [[0.5403, -0.8415], [0.8415, 0.5403]]
    B = [[-0.4597], [0.8415]]
    C = [[1, 0]]
    D = [[0]]

    # Linear discrete-time model with sample time 1
    sys = ct.ss2io(ct.ss(A, B, C, D, 1))

    # state and input constraints
    trajectory_constraints = [
        (sp.optimize.LinearConstraint, np.eye(3), [-10, -10, -1], [10, 10, 1]),
    ]

    # Include weights on states/inputs
    Q = np.eye(2)
    R = 1
    K, S, E = ct.lqr(A, B, Q, R)

    # Compute the integral and terminal cost
    integral_cost = obc.quadratic_cost(sys, Q, R)
    terminal_cost = obc.quadratic_cost(sys, S, 0)

    # Formulate finite horizon MPC problem
    time = np.arange(0, 5, 1)
    optctrl = obc.OptimalControlProblem(
        sys, time, integral_cost, trajectory_constraints, terminal_cost)

    # Add tests to make sure everything works
    t, u_openloop = optctrl.compute_trajectory([1, 1])


def test_mpc_iosystem():
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
    constraints = [obc.input_range_constraint(sys, [-5, -6], [5, 6])]

    # provide penalties on the system signals
    Q = model.C.transpose() @ np.diag([10, 10, 10, 10]) @ model.C
    R = np.diag([3, 2])
    cost = obc.quadratic_cost(model, Q, R, x0=xd, u0=ud)

    # online MPC controller object is constructed with a horizon 6
    optctrl = obc.OptimalControlProblem(
        model, np.arange(0, 6) * 0.2, cost, constraints)

    # Define an I/O system implementing model predictive control
    ctrl = optctrl.create_mpc_iosystem()
    loop = ct.feedback(sys, ctrl, 1)

    # Choose a nearby initial condition to speed up computation
    X0 = np.hstack([xd, np.kron(ud, np.ones(6))]) * 0.99

    Nsim = 10
    tout, xout = ct.input_output_response(
        loop, np.arange(0, Nsim) * 0.2, 0, X0)

    # Make sure the system converged to the desired state
    np.testing.assert_almost_equal(xout[0:sys.nstates, -1], xd, decimal=1)
