# optestim_bench.py - benchmarks for optimal/moving horizon estimation
# RMM, 14 Mar 2023
#
# This benchmark tests the timing for the optimal estimation routines and
# is intended to be used for helping tune the performance of the functions
# used for optimization-based estimation.

import numpy as np
import control as ct
import control.optimal as opt

minimizer_table = {
    'default': (None, {}),
    'trust': ('trust-constr', {}),
    'trust_bigstep': ('trust-constr', {'finite_diff_rel_step': 0.01}),
    'SLSQP': ('SLSQP', {}),
    'SLSQP_bigstep': ('SLSQP', {'eps': 0.01}),
    'COBYLA': ('COBYLA', {}),
}

# Table to turn on and off process disturbances and measurement noise
noise_table = {
    'noisy': (1e-1, 1e-3),
    'nodist': (0, 1e-3),
    'nomeas': (1e-1, 0),
    'clean': (0, 0)
}


# Assess performance as a function of optimization and integration methods
def time_oep_minimizer_methods(minimizer_name, noise_name, initial_guess):
    # Use fixed system to avoid randome errors (was csys = ct.rss(4, 2, 5))
    csys = ct.ss(
        [[-0.5, 1, 0, 0], [0, -1, 1, 0], [0, 0, -2, 1], [0, 0, 0, -3]], # A
        [[0, 0.1], [0, 0.1], [0, 0.1], [1, 0.1]],                       # B
        [[1, 0, 0, 0], [0, 0, 1, 0]],                                   # C
        0, dt=0)
    # dsys = ct.c2d(csys, dt)
    # sys = csys if dt == 0 else dsys
    sys = csys

    # Decide on process disturbances and measurement noise
    dist_mag, meas_mag = noise_table[noise_name]

    # Create disturbances and noise (fixed, to avoid random errors)
    Rv = 0.1 * np.eye(1)                # scalar disturbance
    Rw = 0.01 * np.eye(sys.noutputs)
    timepts = np.arange(0, 10.1, 1)
    V = np.array(
        [0 if t % 2 == 1 else 1 if t % 4 == 0 else -1 for t in timepts]
    ).reshape(1, -1) * dist_mag
    W = np.vstack([np.sin(2*timepts), np.cos(3*timepts)]) * meas_mag

    # Generate system data
    U = np.sin(timepts).reshape(1, -1)
    res = ct.input_output_response(sys, timepts, [U, V])
    Y = res.outputs + W

    # Decide on the initial guess to use
    if initial_guess == 'xhat':
        initial_guess = (res.states, V*0)
    elif initial_guess == 'both':
        initial_guess = (res.states, V)
    else:
        initial_guess = None

    # Set up optimal estimation function using Gaussian likelihoods for cost
    traj_cost = opt.gaussian_likelihood_cost(sys, Rv, Rw)
    init_cost = lambda xhat, x: (xhat - x) @ (xhat - x)
    oep = opt.OptimalEstimationProblem(
        sys, timepts, traj_cost, terminal_cost=init_cost)

    # Noise and disturbances (the standard case)
    est = oep.compute_estimate(Y, U, initial_guess=initial_guess)
    assert est.success
    np.testing.assert_allclose(
        est.states[:, -1], res.states[:, -1], atol=1e-1, rtol=1e-2)


# Parameterize the test against different choices of integrator and minimizer
time_oep_minimizer_methods.param_names = ['minimizer', 'noise', 'initial']
time_oep_minimizer_methods.params = (
    ['default', 'trust', 'SLSQP', 'COBYLA'],
    ['noisy', 'nodist', 'nomeas', 'clean'],
    ['none', 'xhat', 'both'])
