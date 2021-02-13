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
import polytope as pc

def test_finite_horizon_mpc_simple():
    # Define a linear system with constraints
    # Source: https://www.mpt3.org/UI/RegulationProblem

    # LTI prediction model
    sys = ct.ss2io(ct.ss([[1, 1], [0, 1]], [[1], [0.5]], np.eye(2), 0, 1))

    # State and input constraints
    constraints = [
        obc.state_poly_constraint(sys, pc.box2poly([[-5, 5], [-5, 5]])),
        obc.input_poly_constraint(sys, pc.box2poly([[-1, 1]])),
    ]

    # Quadratic state and input penalty
    Q = [[1, 0], [0, 1]]
    R = [[1]]
    cost = obc.quadratic_cost(sys, Q, R)

    # Create a model predictive controller system
    time = np.arange(0, 5, 1)
    mpc = obc.ModelPredictiveController(sys, time, cost, constraints)

    # Optimal control input for a given value of the initial state
    x0 = [4, 0]
    u = mpc(x0)
    np.testing.assert_almost_equal(u, -1)

    # Retrieve the full open-loop predictions
    t, u_openloop = mpc.compute_trajectory(x0, squeeze=True)
    np.testing.assert_almost_equal(
        u_openloop, [-1, -1, 0.1393, 0.3361, -5.204e-16], decimal=4)

    # Convert controller to an explicit form (not implemented yet)
    # mpc_explicit = mpc.explicit();

    # Test explicit controller 
    # u_explicit = mpc_explicit(x0)
    # np.testing.assert_array_almost_equal(u_openloop, u_explicit)

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
        obc.state_poly_constraint(sys, pc.box2poly([[-10, 10]])),
        obc.input_poly_constraint(sys, pc.box2poly([[-1, 1]]))
    ]

    # Include weights on states/inputs
    Q = np.eye(2)
    R = 1
    K, S, E = ct.lqr(A, B, Q, R)

    # Compute the integral and terminal cost
    integral_cost = obc.quadratic_cost(sys, Q, R)
    terminal_cost = obc.quadratic_cost(sys, S, 0)

    # Formulate finite horizon MPC problem
    time = np.arange(0, 5, 5)
    mpc = obc.ModelPredictiveController(
        sys, time, integral_cost, trajectory_constraints, terminal_cost)

    # Add tests to make sure everything works
