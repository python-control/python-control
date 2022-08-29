# optimal_bench.py - benchmarks for optimal control package
# RMM, 27 Feb 2021
#
# This benchmark tests the timing for the optimal control module
# (control.optimal) and is intended to be used for helping tune the
# performance of the functions used for optimization-base control.

import numpy as np
import math
import control as ct
import control.flatsys as fs
import control.optimal as opt

#
# Benchmark test parameters
#

# Define integrator and minimizer methods and options/keywords
integrator_table = {
    'default': (None, {}),
    'RK23': ('RK23', {}),
    'RK23_sloppy': ('RK23', {'atol': 1e-4, 'rtol': 1e-2}),
    'RK45': ('RK45', {}),
    'RK45': ('RK45', {}),
    'RK45_sloppy': ('RK45', {'atol': 1e-4, 'rtol': 1e-2}),
    'LSODA': ('LSODA', {}),
}

minimizer_table = {
    'default': (None, {}),
    'trust': ('trust-constr', {}),
    'trust_bigstep': ('trust-constr', {'finite_diff_rel_step': 0.01}),
    'SLSQP': ('SLSQP', {}),
    'SLSQP_bigstep': ('SLSQP', {'eps': 0.01}),
    'COBYLA': ('COBYLA', {}),
}

# Utility function to create a basis of a given size
def get_basis(name, size, Tf):
    if name == 'poly':
        basis = fs.PolyFamily(size, T=Tf)
    elif name == 'bezier':
        basis = fs.BezierFamily(size, T=Tf)
    elif name == 'bspline':
        basis = fs.BSplineFamily([0, Tf/2, Tf], size)
    return basis


#
# Optimal trajectory generation with linear quadratic cost
#

def time_optimal_lq_basis(basis_name, basis_size, npoints):
    # Create a sufficiently controllable random system to control
    ntrys = 20
    while ntrys > 0:
        # Create a random system
        sys = ct.rss(2, 2, 2)

        # Compute the controllability Gramian
        Wc = ct.gram(sys, 'c')

        # Make sure the condition number is reasonable
        if np.linalg.cond(Wc) < 1e6:
            break

        ntrys -= 1
    assert ntrys > 0            # Something wrong if we needed > 20 tries

    # Define cost functions
    Q = np.eye(sys.nstates)
    R = np.eye(sys.ninputs) * 10

    # Figure out the LQR solution (for terminal cost)
    K, S, E = ct.lqr(sys, Q, R)

    # Define the cost functions
    traj_cost = opt.quadratic_cost(sys, Q, R)
    term_cost = opt.quadratic_cost(sys, S, None)
    constraints = opt.input_range_constraint(
        sys, -np.ones(sys.ninputs), np.ones(sys.ninputs))

    # Define the initial condition, time horizon, and time points
    x0 = np.ones(sys.nstates)
    Tf = 10
    timepts = np.linspace(0, Tf, npoints)

    # Create the basis function to use
    basis = get_basis(basis_name, basis_size, Tf)

    res = opt.solve_ocp(
        sys, timepts, x0, traj_cost, constraints, terminal_cost=term_cost,
        basis=basis,
    )
    # Only count this as a benchmark if we converged
    assert res.success

# Parameterize the test against different choices of integrator and minimizer
time_optimal_lq_basis.param_names = ['basis', 'size', 'npoints']
time_optimal_lq_basis.params = (
    ['poly', 'bezier', 'bspline'], [8, 10, 12], [5, 10, 20])


def time_optimal_lq_methods(integrator_name, minimizer_name):
    # Get the integrator and minimizer parameters to use
    integrator = integrator_table[integrator_name]
    minimizer = minimizer_table[minimizer_name]

    # Create a random system to control
    sys = ct.rss(2, 1, 1)

    # Define cost functions
    Q = np.eye(sys.nstates)
    R = np.eye(sys.ninputs) * 10

    # Figure out the LQR solution (for terminal cost)
    K, S, E = ct.lqr(sys, Q, R)

    # Define the cost functions
    traj_cost = opt.quadratic_cost(sys, Q, R)
    term_cost = opt.quadratic_cost(sys, S, None)
    constraints = opt.input_range_constraint(
        sys, -np.ones(sys.ninputs), np.ones(sys.ninputs))

    # Define the initial condition, time horizon, and time points
    x0 = np.ones(sys.nstates)
    Tf = 10
    timepts = np.linspace(0, Tf, 20)

    # Create the basis function to use
    basis = get_basis('poly', 12, Tf)

    res = opt.solve_ocp(
        sys, timepts, x0, traj_cost, constraints, terminal_cost=term_cost,
        solve_ivp_method=integrator[0], solve_ivp_kwargs=integrator[1],
        minimize_method=minimizer[0], minimize_options=minimizer[1],
    )
    # Only count this as a benchmark if we converged
    assert res.success

# Parameterize the test against different choices of integrator and minimizer
time_optimal_lq_methods.param_names = ['integrator', 'minimizer']
time_optimal_lq_methods.params = (
    ['RK23', 'RK45', 'LSODA'], ['trust', 'SLSQP', 'COBYLA'])


def time_optimal_lq_size(nstates, ninputs, npoints):
    # Create a sufficiently controllable random system to control
    ntrys = 20
    while ntrys > 0:
        # Create a random system
        sys = ct.rss(nstates, ninputs, ninputs)

        # Compute the controllability Gramian
        Wc = ct.gram(sys, 'c')

        # Make sure the condition number is reasonable
        if np.linalg.cond(Wc) < 1e6:
            break

        ntrys -= 1
    assert ntrys > 0            # Something wrong if we needed > 20 tries

    # Define cost functions
    Q = np.eye(sys.nstates)
    R = np.eye(sys.ninputs) * 10

    # Figure out the LQR solution (for terminal cost)
    K, S, E = ct.lqr(sys, Q, R)

    # Define the cost functions
    traj_cost = opt.quadratic_cost(sys, Q, R)
    term_cost = opt.quadratic_cost(sys, S, None)
    constraints = opt.input_range_constraint(
        sys, -np.ones(sys.ninputs), np.ones(sys.ninputs))

    # Define the initial condition, time horizon, and time points
    x0 = np.ones(sys.nstates)
    Tf = 10
    timepts = np.linspace(0, Tf, npoints)

    res = opt.solve_ocp(
        sys, timepts, x0, traj_cost, constraints, terminal_cost=term_cost,
    )
    # Only count this as a benchmark if we converged
    assert res.success

# Parameterize the test against different choices of integrator and minimizer
time_optimal_lq_size.param_names = ['nstates', 'ninputs', 'npoints']
time_optimal_lq_size.params = ([1, 2, 4], [1, 2, 4], [5, 10, 20])


#
# Aircraft MPC example (from multi-parametric toolbox)
#

def time_discrete_aircraft_mpc(minimizer_name):
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

    # Set the time horizon and time points
    Tf = 3
    timepts = np.arange(0, 6) * 0.2

    # Get the minimizer parameters to use
    minimizer = minimizer_table[minimizer_name]

    # online MPC controller object is constructed with a horizon 6
    ctrl = opt.create_mpc_iosystem(
        model, timepts, cost, constraints,
        minimize_method=minimizer[0], minimize_options=minimizer[1],
    )

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

# Parameterize the test against different choices of minimizer and basis
time_discrete_aircraft_mpc.param_names = ['minimizer']
time_discrete_aircraft_mpc.params = (
    ['trust', 'trust_bigstep', 'SLSQP', 'SLSQP_bigstep', 'COBYLA'])
