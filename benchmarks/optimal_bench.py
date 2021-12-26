# optimal_bench.py - benchmarks for optimal control package
# RMM, 27 Feb 2021
#
# This benchmark tests the timing for the optimal control module
# (control.optimal) and is intended to be used for helping tune the
# performance of the functions used for optimization-base control.

import numpy as np
import math
import control as ct
import control.flatsys as flat
import control.optimal as opt

# Vehicle steering dynamics
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

# Initial and final conditions
x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [100., 2., 0.]; uf = [10., 0.]
Tf = 10

# Define the time horizon (and spacing) for the optimization
horizon = np.linspace(0, Tf, 10, endpoint=True)

# Provide an intial guess (will be extended to entire horizon)
bend_left = [10, 0.01]          # slight left veer

def time_steering_integrated_cost():
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

def time_steering_terminal_cost():
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
        solve_ivp_kwargs={'atol': 1e-4, 'rtol': 1e-2},
        # minimize_method='SLSQP', minimize_options={'eps': 0.01}
        minimize_method='trust-constr',
        minimize_options={'finite_diff_rel_step': 0.01},
    )
    # Only count this as a benchmark if we converged
    assert res.success

# Define integrator and minimizer methods and options/keywords
integrator_table = {
    'RK23_default': ('RK23', {'atol': 1e-4, 'rtol': 1e-2}),
    'RK23_sloppy': ('RK23', {}),
    'RK45_default': ('RK45', {}),
    'RK45_sloppy': ('RK45', {'atol': 1e-4, 'rtol': 1e-2}),
}

minimizer_table = {
    'trust_default': ('trust-constr', {}),
    'trust_bigstep': ('trust-constr', {'finite_diff_rel_step': 0.01}),
    'SLSQP_default': ('SLSQP', {}),
    'SLSQP_bigstep': ('SLSQP', {'eps': 0.01}),
}


def time_steering_terminal_constraint(integrator_name, minimizer_name):
    # Get the integrator and minimizer parameters to use
    integrator = integrator_table[integrator_name]
    minimizer = minimizer_table[minimizer_name]

    # Input cost and terminal constraints
    R = np.diag([1, 1])                 # minimize applied inputs
    cost = opt.quadratic_cost(vehicle, np.zeros((3,3)), R, u0=uf)
    constraints = [
        opt.input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
    terminal = [ opt.state_range_constraint(vehicle, xf, xf) ]

    res = opt.solve_ocp(
        vehicle, horizon, x0, cost, constraints,
        terminal_constraints=terminal, initial_guess=bend_left, log=False,
        solve_ivp_method=integrator[0], solve_ivp_kwargs=integrator[1],
        minimize_method=minimizer[0], minimize_options=minimizer[1],
    )
    # Only count this as a benchmark if we converged
    assert res.success

# Reset the timeout value to allow for longer runs
time_steering_terminal_constraint.timeout = 120

# Parameterize the test against different choices of integrator and minimizer
time_steering_terminal_constraint.param_names = ['integrator', 'minimizer']
time_steering_terminal_constraint.params = (
    ['RK23_default', 'RK23_sloppy', 'RK45_default', 'RK45_sloppy'],
    ['trust_default', 'trust_bigstep', 'SLSQP_default', 'SLSQP_bigstep']
)

def time_steering_bezier_basis(nbasis, ntimes):
    # Set up costs and constriants
    Q = np.diag([.1, 10, .1])           # keep lateral error low
    R = np.diag([1, 1])                 # minimize applied inputs
    cost = opt.quadratic_cost(vehicle, Q, R, x0=xf, u0=uf)
    constraints = [ opt.input_range_constraint(vehicle, [0, -0.1], [20, 0.1]) ]
    terminal = [ opt.state_range_constraint(vehicle, xf, xf) ]

    # Set up horizon
    horizon = np.linspace(0, Tf, ntimes, endpoint=True)

    # Set up the optimal control problem
    res = opt.solve_ocp(
        vehicle, horizon, x0, cost,
        constraints,
        terminal_constraints=terminal,
        initial_guess=bend_left,
        basis=flat.BezierFamily(nbasis, T=Tf),
        # solve_ivp_kwargs={'atol': 1e-4, 'rtol': 1e-2},
        minimize_method='trust-constr',
        minimize_options={'finite_diff_rel_step': 0.01},
        # minimize_method='SLSQP', minimize_options={'eps': 0.01},
        return_states=True, print_summary=False
    )
    t, u, x = res.time, res.inputs, res.states

    # Make sure we found a valid solution
    assert res.success

# Reset the timeout value to allow for longer runs
time_steering_bezier_basis.timeout = 120

# Set the parameter values for the number of times and basis vectors
time_steering_bezier_basis.param_names = ['nbasis', 'ntimes']
time_steering_bezier_basis.params = ([2, 4, 6], [5, 10, 20])

def time_aircraft_mpc():
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
