# optimal.py - optimization based control module
#
# RMM, 11 Feb 2021
#

"""Optimization-based control module.

This module provides support for optimization-based controllers for
nonlinear systems with state and input constraints.  An optimal
control problem can be solved using the `solve_optimal_trajectory`
function or set up using the `OptimalControlProblem` class and then
solved using the `~OptimalControlProblem.compute_trajectory` method.
Utility functions are available to define common cost functions and
input/state constraints.  Optimal estimation problems can be solved
using the `solve_optimal_estimate` function or by using the
`OptimalEstimationProblem` class and the
`~OptimalEstimationProblem.compute_estimate` method..

The docstring examples assume the following import commands::

  >>> import numpy as np
  >>> import control as ct
  >>> import control.optimal as opt

"""

import logging
import time
import warnings

import numpy as np
import scipy as sp
import scipy.optimize as opt

import control as ct

from . import config
from .config import _process_param, _process_kwargs
from .iosys import _process_control_disturbance_indices, _process_labels
from .timeresp import _timeresp_aliases

# Define module default parameter values
_optimal_trajectory_methods = {'shooting', 'collocation'}
_optimal_defaults = {
    'optimal.minimize_method': None,
    'optimal.minimize_options': {},
    'optimal.minimize_kwargs': {},
    'optimal.solve_ivp_method': None,
    'optimal.solve_ivp_options': {},
}

# Parameter and keyword aliases
_optimal_aliases = {
    # param:                  ([alias, ...],                [legacy, ...])
    'integral_cost':          (['trajectory_cost', 'cost'], []),
    'initial_state':          (['x0', 'X0'],                []),
    'initial_input':          (['u0', 'U0'],                []),
    'final_state':            (['xf'],                      []),
    'final_input':            (['uf'],                      []),
    'initial_time':           (['T0'],                      []),
    'trajectory_constraints': (['constraints'],             []),
    'return_states':          (['return_x'],                []),
}


class OptimalControlProblem():
    """Description of a finite horizon, optimal control problem.

    The `OptimalControlProblem` class holds all of the information required
    to specify an optimal control problem: the system dynamics, cost
    function, and constraints.  As much as possible, the information used
    to specify an optimal control problem matches the notation and
    terminology of `scipy.optimize` module, with the hope that
    this makes it easier to remember how to describe a problem.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the optimal input will be computed.
    timepts : 1D array_like
        List of times at which the optimal input should be computed.
    integral_cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as integral_cost(x, u).
    trajectory_constraints : list of constraints, optional
       List of constraints that should hold at each point in the time
       vector.  Each element of the list should be an object of type
       `scipy.optimize.LinearConstraint` with arguments ``(A, lb,
       ub)`` or `scipy.optimize.NonlinearConstraint` with arguments
       ``(fun, lb, ub)``.  The constraints will be applied at each time point
       along the trajectory.
    terminal_cost : callable, optional
        Function that returns the terminal cost given the final state
        and input.  Called as terminal_cost(x, u).
    trajectory_method : string, optional
        Method to use for carrying out the optimization. Currently supported
        methods are 'shooting' and 'collocation' (continuous time only). The
        default value is 'shooting' for discrete-time systems and
        'collocation' for continuous-time systems.
    initial_guess : (tuple of) 1D or 2D array_like
        Initial states and/or inputs to use as a guess for the optimal
        trajectory.  For shooting methods, an array of inputs for each time
        point should be specified.  For collocation methods, the initial
        guess is either the input vector or a tuple consisting guesses for
        the state and the input.  Guess should either be a 2D vector of
        shape (ninputs, ntimepts) or a 1D input of shape (ninputs,) that
        will be broadcast by extension of the time axis.
    log : bool, optional
        If True, turn on logging messages (using Python logging module).
        Use `logging.basicConfig` to enable logging output
        (e.g., to a file).

    Attributes
    ----------
    constraint: list of SciPy constraint objects
        List of `scipy.optimize.LinearConstraint` or
        `scipy.optimize.NonlinearConstraint` objects.
    constraint_lb, constrain_ub, eqconst_value : list of float
        List of constraint bounds.

    Other Parameters
    ----------------
    basis : `BasisFamily`, optional
        Use the given set of basis functions for the inputs instead of
        setting the value of the input at each point in the timepts vector.
    terminal_constraints : list of constraints, optional
        List of constraints that should hold at the terminal point in time,
        in the same form as `trajectory_constraints`.
    solve_ivp_method : str, optional
        Set the method used by `scipy.integrate.solve_ivp`.
    solve_ivp_kwargs : str, optional
        Pass additional keywords to `scipy.integrate.solve_ivp`.
    minimize_method : str, optional
        Set the method used by `scipy.optimize.minimize`.
    minimize_options : str, optional
        Set the options keyword used by `scipy.optimize.minimize`.
    minimize_kwargs : str, optional
        Pass additional keywords to `scipy.optimize.minimize`.

    Notes
    -----
    To describe an optimal control problem we need an input/output system,
    a set of time points over a a fixed horizon, a cost function, and
    (optionally) a set of constraints on the state and/or input, either
    along the trajectory and at the terminal time.  This class sets up an
    optimization over the inputs at each point in time, using the integral
    and terminal costs as well as the trajectory and terminal constraints.
    The `compute_trajectory` method sets up an optimization problem that
    can be solved using `scipy.optimize.minimize`.

    The `_cost_function` method takes the information computes the cost of
    the trajectory generated by the proposed input.  It does this by calling
    a user-defined function for the integral_cost given the current states
    and inputs at each point along the trajectory and then adding the value
    of a user-defined terminal cost at the final point in the trajectory.

    The `_constraint_function` method evaluates the constraint functions
    along the trajectory generated by the proposed input.  As in the case
    of the cost function, the constraints are evaluated at the state and
    input along each time point on the trajectory.  This information is
    compared against the constraint upper and lower bounds.  The constraint
    function is processed in the class initializer, so that it only needs
    to be computed once.

    If `basis` is specified, then the optimization is done over coefficients
    of the basis elements.  Otherwise, the optimization is performed over the
    values of the input at the specified times (using linear interpolation
    for continuous systems).

    The default values for `minimize_method`, `minimize_options`,
    `minimize_kwargs`, `solve_ivp_method`, and `solve_ivp_options` can
    be set using `config.defaults['optimal.<keyword>']`.

    """
    def __init__(
            self, sys, timepts, integral_cost, trajectory_constraints=None,
            terminal_cost=None, terminal_constraints=None, initial_guess=None,
            trajectory_method=None, basis=None, log=False, _kwargs_check=True,
            **kwargs):
        """Set up an optimal control problem."""
        # Save the basic information for use later
        self.system = sys
        self.timepts = timepts
        self.integral_cost = integral_cost
        self.terminal_cost = terminal_cost
        self.terminal_constraints = terminal_constraints
        self.basis = basis

        # Keep track of what type of trajectory method we are using
        if trajectory_method is None:
            trajectory_method = 'collocation' if sys.isctime() else 'shooting'
        elif trajectory_method not in _optimal_trajectory_methods:
            raise NotImplementedError(f"Unknown method {trajectory_method}")

        self.shooting = trajectory_method in {'shooting'}
        self.collocation = trajectory_method in {'collocation'}

        # Process keyword arguments
        self.solve_ivp_kwargs = {}
        self.solve_ivp_kwargs['method'] = kwargs.pop(
            'solve_ivp_method', config.defaults['optimal.solve_ivp_method'])
        self.solve_ivp_kwargs.update(kwargs.pop(
            'solve_ivp_kwargs', config.defaults['optimal.solve_ivp_options']))

        self.minimize_kwargs = {}
        self.minimize_kwargs['method'] = kwargs.pop(
            'minimize_method', config.defaults['optimal.minimize_method'])
        self.minimize_kwargs['options'] = kwargs.pop(
            'minimize_options', config.defaults['optimal.minimize_options'])
        self.minimize_kwargs.update(kwargs.pop(
            'minimize_kwargs', config.defaults['optimal.minimize_kwargs']))

        # Check to make sure arguments for discrete-time systems are OK
        if sys.isdtime(strict=True):
            if self.solve_ivp_kwargs['method'] is not None or \
                 len(self.solve_ivp_kwargs) > 1:
                raise TypeError(
                    "solve_ivp method, kwargs not allowed for"
                    " discrete-time systems")

        # Make sure there were no extraneous keywords
        if _kwargs_check and kwargs:
            raise TypeError("unrecognized keyword(s): ", str(kwargs))

        self.trajectory_constraints = _process_constraints(
            trajectory_constraints, "trajectory")
        self.terminal_constraints = _process_constraints(
            terminal_constraints, "terminal")

        #
        # Compute and store constraints
        #
        # While the constraints are evaluated during the execution of the
        # SciPy optimization method itself, we go ahead and pre-compute the
        # `scipy.optimize.NonlinearConstraint` function that will be passed to
        # the optimizer on initialization, since it doesn't change.  This is
        # mainly a matter of computing the lower and upper bound vectors,
        # which we need to "stack" to account for the evaluation at each
        # trajectory time point plus any terminal constraints (in a way that
        # is consistent with the `_constraint_function` that is used at
        # evaluation time.
        #
        # TODO: when using the collocation method, linear constraints on the
        # states and inputs can potentially maintain their linear structure
        # rather than being converted to nonlinear constraints.
        #
        constraint_lb, constraint_ub, eqconst_value = [], [], []

        # Go through each time point and stack the bounds
        for t in self.timepts:
            for type, fun, lb, ub in self.trajectory_constraints:
                if np.all(lb == ub):
                    # Equality constraint
                    eqconst_value.append(lb)
                else:
                    # Inequality constraint
                    constraint_lb.append(lb)
                    constraint_ub.append(ub)

        # Add on the terminal constraints
        for type, fun, lb, ub in self.terminal_constraints:
            if np.all(lb == ub):
                # Equality constraint
                eqconst_value.append(lb)
            else:
                # Inequality constraint
                constraint_lb.append(lb)
                constraint_ub.append(ub)

        # Turn constraint vectors into 1D arrays
        self.constraint_lb = np.hstack(constraint_lb) if constraint_lb else []
        self.constraint_ub = np.hstack(constraint_ub) if constraint_ub else []
        self.eqconst_value = np.hstack(eqconst_value) if eqconst_value else []

        # Create the constraints (inequality and equality)
        # TODO: for collocation method, keep track of linear vs nonlinear
        self.constraints = []

        if len(self.constraint_lb) != 0:
            self.constraints.append(sp.optimize.NonlinearConstraint(
                self._constraint_function, self.constraint_lb,
                self.constraint_ub))

        if len(self.eqconst_value) != 0:
            self.constraints.append(sp.optimize.NonlinearConstraint(
                self._eqconst_function, self.eqconst_value,
                self.eqconst_value))

        if self.collocation:
            # Add the collocation constraints
            colloc_zeros = np.zeros(sys.nstates * self.timepts.size)
            self.colloc_vals = np.zeros((sys.nstates, self.timepts.size))
            self.constraints.append(sp.optimize.NonlinearConstraint(
                self._collocation_constraint, colloc_zeros, colloc_zeros))

        # Initialize run-time statistics
        self._reset_statistics(log)

        # Process the initial guess
        self.initial_guess = self._process_initial_guess(initial_guess)

        # Store states, input (used later to minimize re-computation)
        self.last_x = np.full(self.system.nstates, np.nan)
        self.last_coeffs = np.full(self.initial_guess.shape, np.nan)

        # Log information
        if log:
            logging.info("New optimal control problem initialized")

    #
    # Cost function
    #
    # For collocation methods we are given the states and inputs at each
    # time point and we use a trapezoidal approximation to compute the
    # integral cost, then add on the terminal cost.
    #
    # For shooting methods, given the input U = [u[t_0], ... u[t_N]] we need
    # to compute the cost of the trajectory generated by that input.  This
    # means we have to simulate the system to get the state trajectory X =
    # [x[t_0], ..., x[t_N]] and then compute the cost at each point:
    #
    #   cost = sum_k integral_cost(x[t_k], u[t_k])
    #          + terminal_cost(x[t_N], u[t_N])
    #
    # The actual calculation is a bit more complex: for continuous-time
    # systems, we use a trapezoidal approximation for the integral cost.
    #
    # The initial state used for generating the simulation is stored in the
    # class parameter `x` prior to calling the optimization algorithm.
    #
    def _cost_function(self, coeffs):
        if self.log:
            start_time = time.process_time()
            logging.info("_cost_function called at: %g", start_time)

        # Compute the states and inputs
        states, inputs = self._compute_states_inputs(coeffs)

        # Trajectory cost
        if ct.isctime(self.system):
            # Evaluate the costs
            costs = [self.integral_cost(states[:, i], inputs[:, i]) for
                     i in range(self.timepts.size)]

            # Compute the time intervals
            dt = np.diff(self.timepts)

            # Integrate the cost
            costs = np.array(costs)

            # Approximate the integral using trapezoidal rule
            cost = np.sum(0.5 * (costs[:-1] + costs[1:]) * dt)

        else:
            # Sum the integral cost over the time (second) indices
            # cost += self.integral_cost(states[:,i], inputs[:,i])
            cost = sum(map(
                self.integral_cost, states[:, :-1].transpose(),
                inputs[:, :-1].transpose()))

        # Terminal cost
        if self.terminal_cost is not None:
            cost += self.terminal_cost(states[:, -1], inputs[:, -1])

        # Update statistics
        self.cost_evaluations += 1
        if self.log:
            stop_time = time.process_time()
            self.cost_process_time += stop_time - start_time
            logging.info(
                "_cost_function returning %g; elapsed time: %g",
                cost, stop_time - start_time)

        # Return the total cost for this input sequence
        return cost

    #
    # Constraints
    #
    # We are given the constraints along the trajectory and the terminal
    # constraints, which each take inputs [x, u] and evaluate the
    # constraint.  How we handle these depends on the type of constraint:
    #
    # * For linear constraints (LinearConstraint), a combined (hstack'd)
    #   vector of the state and input is multiplied by the polytope A matrix
    #   for comparison against the upper and lower bounds.
    #
    # * For nonlinear constraints (NonlinearConstraint), a user-specific
    #   constraint function having the form
    #
    #      constraint_fun(x, u)
    #
    #   is called at each point along the trajectory and compared against the
    #   upper and lower bounds.
    #
    # * If the upper and lower bound for the constraint are identical, then
    #   we separate out the evaluation into two different constraints, which
    #   allows the SciPy optimizers to be more efficient (and stops them
    #   from generating a warning about mixed constraints).  This is handled
    #   through the use of the `_eqconst_function` and `eqconst_value`
    #   members.
    #
    # In both cases, the constraint is specified at a single point, but we
    # extend this to apply to each point in the trajectory.  This means
    # that for N time points with m trajectory constraints and p terminal
    # constraints we need to compute N*m + p constraints, each of which
    # holds at a specific point in time, and implements the original
    # constraint.
    #
    # For collocation methods, we can directly evaluate the constraints at
    # the collocation points.
    #
    # For shooting methods, we do this by creating a function that simulates
    # the system dynamics and returns a vector of values corresponding to
    # the value of the function at each time.  The class initialization
    # methods takes care of replicating the upper and lower bounds for each
    # point in time so that the SciPy optimization algorithm can do the
    # proper evaluation.
    #
    def _constraint_function(self, coeffs):
        if self.log:
            start_time = time.process_time()
            logging.info("_constraint_function called at: %g", start_time)

        # Compute the states and inputs
        states, inputs = self._compute_states_inputs(coeffs)

        #
        # Evaluate the constraint function along the trajectory
        #
        value = []
        for i, t in enumerate(self.timepts):
            for ctype, fun, lb, ub in self.trajectory_constraints:
                if np.all(lb == ub):
                    # Skip equality constraints
                    continue
                elif ctype == opt.LinearConstraint:
                    # `fun` is the A matrix associated with the polytope...
                    value.append(fun @ np.hstack([states[:, i], inputs[:, i]]))
                elif ctype == opt.NonlinearConstraint:
                    value.append(fun(states[:, i], inputs[:, i]))
                else:      # pragma: no cover
                    # Checked above => we should never get here
                    raise TypeError(f"unknown constraint type {ctype}")

        # Evaluate the terminal constraint functions
        for ctype, fun, lb, ub in self.terminal_constraints:
            if np.all(lb == ub):
                # Skip equality constraints
                continue
            elif ctype == opt.LinearConstraint:
                value.append(fun @ np.hstack([states[:, -1], inputs[:, -1]]))
            elif ctype == opt.NonlinearConstraint:
                value.append(fun(states[:, -1], inputs[:, -1]))
            else:      # pragma: no cover
                # Checked above => we should never get here
                raise TypeError(f"unknown constraint type {ctype}")

        # Update statistics
        self.constraint_evaluations += 1
        if self.log:
            stop_time = time.process_time()
            self.constraint_process_time += stop_time - start_time
            logging.info(
                "_constraint_function elapsed time: %g",
                stop_time - start_time)

            # Debugging information
            logging.debug(
                "constraint values\n" + str(value) + "\n" +
                "lb, ub =\n" + str(self.constraint_lb) + "\n" +
                str(self.constraint_ub))

        # Return the value of the constraint function
        return np.hstack(value)

    def _eqconst_function(self, coeffs):
        if self.log:
            start_time = time.process_time()
            logging.info("_eqconst_function called at: %g", start_time)

        # Compute the states and inputs
        states, inputs = self._compute_states_inputs(coeffs)

        # Evaluate the constraint function along the trajectory
        value = []
        for i, t in enumerate(self.timepts):
            for ctype, fun, lb, ub in self.trajectory_constraints:
                if np.any(lb != ub):
                    # Skip inequality constraints
                    continue
                elif ctype == opt.LinearConstraint:
                    # `fun` is the A matrix associated with the polytope...
                    value.append(fun @ np.hstack([states[:, i], inputs[:, i]]))
                elif ctype == opt.NonlinearConstraint:
                    value.append(fun(states[:, i], inputs[:, i]))
                else:      # pragma: no cover
                    # Checked above => we should never get here
                    raise TypeError(f"unknown constraint type {ctype}")

        # Evaluate the terminal constraint functions
        for ctype, fun, lb, ub in self.terminal_constraints:
            if np.any(lb != ub):
                # Skip inequality constraints
                continue
            elif ctype == opt.LinearConstraint:
                value.append(fun @ np.hstack([states[:, -1], inputs[:, -1]]))
            elif ctype == opt.NonlinearConstraint:
                value.append(fun(states[:, -1], inputs[:, -1]))
            else:      # pragma: no cover
                # Checked above => we should never get here
                raise TypeError("unknown constraint type {ctype}")

        # Update statistics
        self.eqconst_evaluations += 1
        if self.log:
            stop_time = time.process_time()
            self.eqconst_process_time += stop_time - start_time
            logging.info(
                "_eqconst_function elapsed time: %g", stop_time - start_time)

            # Debugging information
            logging.debug(
                "eqconst values\n" + str(value) + "\n" +
                "desired =\n" + str(self.eqconst_value))

        # Return the value of the constraint function
        return np.hstack(value)

    def _collocation_constraint(self, coeffs):
        # Compute the states and inputs
        states, inputs = self._compute_states_inputs(coeffs)

        if self.system.isctime():
            # Compute the collocation constraints
            for i, t in enumerate(self.timepts):
                if i == 0:
                    # Initial condition constraint (self.x = initial point)
                    self.colloc_vals[:, 0] = states[:, 0] - self.x
                    fk = self.system._rhs(
                        self.timepts[0], states[:, 0], inputs[:, 0])
                    continue

                # From M. Kelly, SIAM Review (2017), equation (3.2), i = k+1
                # x[k+1] - x[k] = 0.5 hk (f(x[k+1], u[k+1] + f(x[k], u[k]))
                fkp1 = self.system._rhs(t, states[:, i], inputs[:, i])
                self.colloc_vals[:, i] = states[:, i] - states[:, i-1] - \
                    0.5 * (self.timepts[i] - self.timepts[i-1]) * (fkp1 + fk)
                fk = fkp1
        else:
            raise NotImplementedError(
                "collocation not yet implemented for discrete-time systems")

        # Return the value of the constraint function
        return self.colloc_vals.reshape(-1)

    #
    # Initial guess processing
    #
    # We store an initial guess in case it is not specified later.  Note
    # that create_mpc_iosystem() will reset the initial guess based on
    # the current state of the MPC controller.
    #
    # The functions below are used to process the initial guess, which can
    # either consist of an input only (for shooting methods) or an input
    # and/or state trajectory (for collocation methods).
    #
    # Note: The initial input guess is passed as the inputs at the given time
    # vector.  If a basis is specified, this is converted to coefficient
    # values (which are generally of smaller dimension).
    #
    def _process_initial_guess(self, initial_guess):
        # Sort out the input guess and the state guess
        if self.collocation and initial_guess is not None and \
           isinstance(initial_guess, tuple):
            state_guess, input_guess = initial_guess
        else:
            state_guess, input_guess = None, initial_guess

        # Process the input guess
        if input_guess is not None:
            input_guess = self._broadcast_initial_guess(
                input_guess, (self.system.ninputs, self.timepts.size))

            # If we were given a basis, project onto the basis elements
            if self.basis is not None:
                input_guess = self._inputs_to_coeffs(input_guess)
        else:
            input_guess = np.zeros(
                self.system.ninputs *
                (self.timepts.size if self.basis is None else self.basis.N))

        # Process the state guess
        if self.collocation:
            if state_guess is None:
                # Run a simulation to get the initial guess
                if self.basis:
                    inputs = self._coeffs_to_inputs(input_guess)
                else:
                    inputs = input_guess.reshape(self.system.ninputs, -1)
                state_guess = self._simulate_states(
                    np.zeros(self.system.nstates), inputs)
            else:
                state_guess = self._broadcast_initial_guess(
                    state_guess, (self.system.nstates, self.timepts.size))

            # Reshape for use by scipy.optimize.minimize()
            return np.hstack([
                input_guess.reshape(-1), state_guess.reshape(-1)])
        else:
            # Reshape for use by scipy.optimize.minimize()
            return input_guess.reshape(-1)

    # Utility function to broadcast an initial guess to the right shape
    def _broadcast_initial_guess(self, initial_guess, shape):
        # Convert to a 1D array (or higher)
        initial_guess = np.atleast_1d(initial_guess)

        # See whether we got entire guess or just first time point
        if initial_guess.ndim == 1:
            # Broadcast inputs to entire time vector
            try:
                initial_guess = np.broadcast_to(
                    initial_guess.reshape(-1, 1), shape)
            except ValueError:
                raise ValueError("initial guess is the wrong shape")

        elif initial_guess.shape != shape:
            raise ValueError("initial guess is the wrong shape")

        return initial_guess

    #
    # Utility function to convert input vector to coefficient vector
    #
    # Initial guesses from the user are passed as input vectors as a
    # function of time, but internally we store the guess in terms of the
    # basis coefficients.  We do this by solving a least squares problem to
    # find coefficients that match the input functions at the time points
    # (as much as possible, if the problem is under-determined).
    #
    def _inputs_to_coeffs(self, inputs):
        # If there is no basis function, just return inputs as coeffs
        if self.basis is None:
            return inputs

        # Solve least squares problems (M x = b) for coeffs on each input
        coeffs = []
        for i in range(self.system.ninputs):
            # Set up the matrices to get inputs
            M = np.zeros((self.timepts.size, self.basis.var_ncoefs(i)))
            b = np.zeros(self.timepts.size)

            # Evaluate at each time point and for each basis function
            # TODO: vectorize
            for j, t in enumerate(self.timepts):
                for k in range(self.basis.var_ncoefs(i)):
                    M[j, k] = self.basis(k, t)
                b[j] = inputs[i, j]

            # Solve a least squares problem for the coefficients
            alpha, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
            coeffs.append(alpha)

        return np.hstack(coeffs)

    # Utility function to convert coefficient vector to input vector
    def _coeffs_to_inputs(self, coeffs):
        # TODO: vectorize
        # TODO: use BasisFamily eval() method (if more efficient)?
        inputs = np.zeros((self.system.ninputs, self.timepts.size))
        offset = 0
        for i in range(self.system.ninputs):
            length = self.basis.var_ncoefs(i)
            for j, t in enumerate(self.timepts):
                for k in range(length):
                    inputs[i, j] += coeffs[offset + k] * self.basis(k, t)
            offset += length
        return inputs

    #
    # Log and statistics
    #
    # To allow some insight into where time is being spent, we keep track of
    # the number of times that various functions are called and (optionally)
    # how long we spent inside each function.
    #
    def _reset_statistics(self, log=False):
        """Reset counters for keeping track of statistics"""
        self.log = log
        self.cost_evaluations, self.cost_process_time = 0, 0
        self.constraint_evaluations, self.constraint_process_time = 0, 0
        self.eqconst_evaluations, self.eqconst_process_time = 0, 0
        self.system_simulations = 0

    def _print_statistics(self, reset=True):
        """Print out summary statistics from last run"""
        print("Summary statistics:")
        print("* Cost function calls:", self.cost_evaluations)
        if self.log:
            print("* Cost function process time:", self.cost_process_time)
        if self.constraint_evaluations:
            print("* Constraint calls:", self.constraint_evaluations)
            if self.log:
                print(
                    "* Constraint process time:", self.constraint_process_time)
        if self.eqconst_evaluations:
            print("* Eqconst calls:", self.eqconst_evaluations)
            if self.log:
                print(
                    "* Eqconst process time:", self.eqconst_process_time)
        print("* System simulations:", self.system_simulations)
        if reset:
            self._reset_statistics(self.log)

    #
    # Compute the states and inputs from the coefficient vector
    #
    # These internal functions return the states and inputs at the
    # collocation points given the coefficient (optimizer state) vector.
    # They keep track of whether a shooting method is being used or not and
    # simulate the dynamics if needed.
    #

    # Compute the states and inputs from the coefficients
    def _compute_states_inputs(self, coeffs):
        #
        # Compute out the states and inputs
        #
        if self.collocation:
            # States are appended to end of (input) coefficients
            states = coeffs[-self.system.nstates * self.timepts.size:].reshape(
                self.system.nstates, -1)
            coeffs = coeffs[:-self.system.nstates * self.timepts.size]

        # Compute input at time points
        if self.basis:
            inputs = self._coeffs_to_inputs(coeffs)
        else:
            inputs = coeffs.reshape((self.system.ninputs, -1))

        if self.shooting:
            # See if we already have a simulation for this condition
            if np.array_equal(coeffs, self.last_coeffs) \
               and np.array_equal(self.x, self.last_x):
                states = self.last_states
            else:
                states = self._simulate_states(self.x, inputs)
                self.last_x = self.x.copy()             # save initial state
                self.last_states = states               # always a new object
                self.last_coeffs = coeffs.copy()        # save coefficients

        return states, inputs

    # Simulate the system dynamics to retrieve the state
    def _simulate_states(self, x0, inputs):
        if self.log:
            logging.debug(
                "calling input_output_response from state\n" + str(x0))
            logging.debug("input =\n" + str(inputs))

        # Simulate the system to get the state
        # TODO: update to use response object; remove return_x
        _, _, states = ct.input_output_response(
            self.system, self.timepts, inputs, x0, return_x=True,
            solve_ivp_kwargs=self.solve_ivp_kwargs, t_eval=self.timepts)
        self.system_simulations += 1

        if self.log:
            logging.debug(
                "input_output_response returned states\n" + str(states))

        return states

    #
    # Optimal control computations
    #

    # Compute the optimal trajectory from the current state
    def compute_trajectory(
            self, x, squeeze=None, transpose=None, return_states=True,
            initial_guess=None, print_summary=True, **kwargs):
        """Compute the optimal trajectory starting at state x.

        Parameters
        ----------
        x : array_like or number, optional
            Initial state for the system.
        initial_guess : (tuple of) 1D or 2D array_like
            Initial states and/or inputs to use as a guess for the optimal
            trajectory.  For shooting methods, an array of inputs for each
            time point should be specified.  For collocation methods, the
            initial guess is either the input vector or a tuple consisting
            guesses for the state and the input.  Guess should either be a
            2D vector of shape (ninputs, ntimepts) or a 1D input of shape
            (ninputs,) that will be broadcast by extension of the time axis.
        return_states : bool, optional
            If True (default), return the values of the state at each time.
        squeeze : bool, optional
            If True and if the system has a single output, return
            the system output as a 1D array rather than a 2D array.  If
            False, return the system output as a 2D array even if
            the system is SISO.  Default value set by
            `config.defaults['control.squeeze_time_response']`.
        transpose : bool, optional
            If True, assume that 2D input arrays are transposed from the
            standard format.  Used to convert MATLAB-style inputs to our
            format.
        print_summary : bool, optional
            If True (default), print a short summary of the computation.

        Returns
        -------
        res : `OptimalControlResult`
            Bundle object with the results of the optimal control problem.
        res.success : bool
            Boolean flag indicating whether the optimization was successful.
        res.time : array
            Time values of the input.
        res.inputs : array
            Optimal inputs for the system.  If the system is SISO and
            squeeze is not True, the array is 1D (indexed by time).
            If the system is not SISO or squeeze is False, the array
            is 2D (indexed by the output number and time).
        res.states : array
            Time evolution of the state vector.

        """
        # Allow 'return_x` as a synonym for 'return_states'
        return_states = ct.config._get_param(
            'optimal', 'return_x', kwargs, return_states, pop=True, last=True)

        # Store the initial state (for use in _constraint_function)
        self.x = x

        # Allow the initial guess to be overridden
        if initial_guess is None:
            initial_guess = self.initial_guess
        else:
            initial_guess = self._process_initial_guess(initial_guess)

        # Call SciPy optimizer
        res = sp.optimize.minimize(
            self._cost_function, initial_guess,
            constraints=self.constraints, **self.minimize_kwargs)

        # Process and return the results
        return OptimalControlResult(
            self, res, transpose=transpose, return_states=return_states,
            squeeze=squeeze, print_summary=print_summary)

    # Compute the current input to apply from the current state (MPC style)
    def compute_mpc(self, x, squeeze=None):
        """Compute the optimal input at state x.

        This function calls the :meth:`compute_trajectory` method and returns
        the input at the first time point.

        Parameters
        ----------
        x : array_like or number, optional
            Initial state for the system.
        squeeze : bool, optional
            If True and if the system has a single output, return
            the system output as a 1D array rather than a 2D array.  If
            False, return the system output as a 2D array even if
            the system is SISO.  Default value set by
            `config.defaults['control.squeeze_time_response']`.

        Returns
        -------
        input : array
            Optimal input for the system at the current time, as a 1D array
            (even in the SISO case).  Set to None if the optimization failed.

        """
        res = self.compute_trajectory(x, squeeze=squeeze)
        return res.inputs[:, 0]

    # Create an input/output system implementing an MPC controller
    def create_mpc_iosystem(self, **kwargs):
        """Create an I/O system implementing an MPC controller.

        For a discrete-time system, creates an input/output system taking
        the current state x and returning the control u to apply at the
        current time step.

        Parameters
        ----------
        name : str, optional
            Name for the system controller.  Defaults to a generic system
            name of the form 'sys[i]'.
        inputs : list of str, optional
            Labels for the controller inputs.  Defaults to the system state
            labels.
        outputs : list of str, optional
            Labels for the controller outputs.  Defaults to the system input
            labels.
        states : list of str, optional
            Labels for the internal controller states, which consist either
            of the input values over the horizon of the controller or the
            coefficients of the basis functions.  Defaults to strings of
            the form 'x[i]'.

        Returns
        -------
        `NonlinearIOSystem`

        Notes
        -----
        Only works for discrete-time systems.

        """
        # Check to make sure we are in discrete time
        if self.system.dt == 0:
            raise ct.ControlNotImplemented(
                "MPC for continuous-time systems not implemented")

        def _update(t, x, u, params={}):
            coeffs = x.reshape((self.system.ninputs, -1))
            if self.basis:
                # Keep the coefficients unchanged
                # TODO: could compute input vector, shift, and re-project (?)
                self.initial_guess = coeffs
            else:
                # Shift the basis elements by one time step
                self.initial_guess = np.hstack(
                    [coeffs[:, 1:], coeffs[:, -1:]]).reshape(-1)
            res = self.compute_trajectory(u, print_summary=False)

            # New state is the new input vector
            return res.inputs.reshape(-1)

        def _output(t, x, u, params={}):
            # Start with initial guess and recompute based on input state (u)
            self.initial_guess = x
            res = self.compute_trajectory(u, print_summary=False)
            return res.inputs[:, 0]

        # Define signal names, if they are not already given
        if kwargs.get('inputs') is None:
            kwargs['inputs'] = self.system.state_labels
        if kwargs.get('outputs') is None:
            kwargs['outputs'] = self.system.input_labels
        if kwargs.get('states') is None:
            kwargs['states'] = self.system.ninputs * \
                (self.timepts.size if self.basis is None else self.basis.N)

        return ct.NonlinearIOSystem(
            _update, _output, dt=self.system.dt, **kwargs)


# Optimal control result
class OptimalControlResult(sp.optimize.OptimizeResult):
    """Result from solving an optimal control problem.

    This class is a subclass of `scipy.optimize.OptimizeResult` with
    additional attributes associated with solving optimal control problems.
    It is used as the return type for optimal control problems.

    Parameters
    ----------
    ocp : OptimalControlProblem
        Optimal control problem that generated this solution.
    res : scipy.minimize.OptimizeResult
        Result of optimization.
    print_summary : bool, optional
        If True (default), print a short summary of the computation.
    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default
        value set by `config.defaults['control.squeeze_time_response']`.

    Attributes
    ----------
    inputs : ndarray
        The optimal inputs associated with the optimal control problem.
    states : ndarray
        If `return_states` was set to true, stores the state trajectory
        associated with the optimal input.
    success : bool
        Whether or not the optimizer exited successful.
    cost : float
        Final cost of the return solution.
    system_simulations, cost_evaluations, constraint_evaluations, \
    eqconst_evaluations : int
        Number of system simulations and evaluations of the cost function,
        (inequality) constraint function, and equality constraint function
        performed during the optimization.
    cost_process_time, constraint_process_time, eqconst_process_time : float
        If logging was enabled, the amount of time spent evaluating the cost
        and constraint functions.

    """
    def __init__(
            self, ocp, res, return_states=True, print_summary=False,
            transpose=None, squeeze=None):
        """Create a OptimalControlResult object"""

        # Copy all of the fields we were sent by sp.optimize.minimize()
        for key, val in res.items():
            setattr(self, key, val)

        # Remember the optimal control problem that we solved
        self.problem = ocp

        # Parse the optimization variables into states and inputs
        states, inputs = ocp._compute_states_inputs(res.x)

        # See if we got an answer
        if not res.success:
            warnings.warn(
                "unable to solve optimal control problem\n"
                "scipy.optimize.minimize returned " + res.message, UserWarning)

        # Save the final cost
        self.cost = res.fun

        # Optionally print summary information
        if print_summary:
            ocp._print_statistics()
            print("* Final cost:", self.cost)

        # Process data as a time response (with "outputs" = inputs)
        response = ct.TimeResponseData(
            ocp.timepts, inputs, states, issiso=ocp.system.issiso(),
            transpose=transpose, return_x=return_states, squeeze=squeeze)

        self.time = response.time
        self.inputs = response.outputs
        self.states = response.states


# Compute the input for a nonlinear, (constrained) optimal control problem
def solve_optimal_trajectory(
        sys, timepts, initial_state=None, integral_cost=None,
        trajectory_constraints=None, terminal_cost=None,
        terminal_constraints=None, initial_guess=None,
        basis=None, squeeze=None, transpose=None, return_states=True,
        print_summary=True, log=False, **kwargs):

    r"""Compute the solution to an optimal control problem.

    The optimal trajectory (states and inputs) is computed so as to
    approximately minimize a cost function of the following form (for
    continuous-time systems):

      J(x(.), u(.)) = \int_0^T L(x(t), u(t)) dt + V(x(T)),

    where T is the time horizon.

    Discrete time systems use a similar formulation, with the integral
    replaced by a sum:

      J(x[.], u[.]) = \sum_0^{N-1} L(x_k, u_k) + V(x_N),

    where N is the time horizon (corresponding to timepts[-1]).

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the optimal input will be computed.
    timepts : 1D array_like
        List of times at which the optimal input should be computed.
    initial_state (or X0) : array_like or number, optional
        Initial condition (default = 0).
    integral_cost (or cost) : callable
        Function that returns the integral cost (L) given the current state
        and input.  Called as ``integral_cost(x, u)``.
    trajectory_constraints (or constraints) : list of tuples, optional
        List of constraints that should hold at each point in the time
        vector.  Each element of the list should consist of a tuple with
        first element given by `scipy.optimize.LinearConstraint` or
        `scipy.optimize.NonlinearConstraint` and the remaining elements of
        the tuple are the arguments that would be passed to those functions.
        The following tuples are supported:

        * (LinearConstraint, A, lb, ub): The matrix A is multiplied by
          stacked vector of the state and input at each point on the
          trajectory for comparison against the upper and lower bounds.

        * (NonlinearConstraint, fun, lb, ub): a user-specific constraint
          function ``fun(x, u)`` is called at each point along the trajectory
          and compared against the upper and lower bounds.

        The constraints are applied at each time point along the trajectory.
    terminal_cost : callable, optional
        Function that returns the terminal cost (V) given the final state
        and input.  Called as terminal_cost(x, u).  (For compatibility with
        the form of the cost function, u is passed even though it is often
        not part of the terminal cost.)
    terminal_constraints : list of tuples, optional
        List of constraints that should hold at the end of the trajectory.
        Same format as `constraints`.
    initial_guess : 1D or 2D array_like
        Initial inputs to use as a guess for the optimal input.  The inputs
        should either be a 2D vector of shape (ninputs, len(timepts)) or a
        1D input of shape (ninputs,) that will be broadcast by extension of
        the time axis.
    basis : `BasisFamily`, optional
        Use the given set of basis functions for the inputs instead of
        setting the value of the input at each point in the timepts vector.

    Returns
    -------
    res : `OptimalControlResult`
        Bundle object with the results of the optimal control problem.
    res.success : bool
        Boolean flag indicating whether the optimization was successful.
    res.time : array
        Time values of the input.
    res.inputs : array
        Optimal inputs for the system.  If the system is SISO and squeeze is
        not True, the array is 1D (indexed by time).  If the system is not
        SISO or squeeze is False, the array is 2D (indexed by the output
        number and time).
    res.states : array
        Time evolution of the state vector.

    Other Parameters
    ----------------
    log : bool, optional
        If True, turn on logging messages (using Python logging module).
    minimize_method : str, optional
        Set the method used by `scipy.optimize.minimize`.
    print_summary : bool, optional
        If True (default), print a short summary of the computation.
    return_states : bool, optional
        If True (default), return the values of the state at each time.
    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default
        value set by `config.defaults['control.squeeze_time_response']`.
    trajectory_method : string, optional
        Method to use for carrying out the optimization. Currently supported
        methods are 'shooting' and 'collocation' (continuous time only). The
        default value is 'shooting' for discrete-time systems and
        'collocation' for continuous-time systems.
    transpose : bool, optional
        If True, assume that 2D input arrays are transposed from the standard
        format.  Used to convert MATLAB-style inputs to our format.

    Notes
    -----
    For discrete-time systems, the final value of the timepts vector
    specifies the final time t_N, and the trajectory cost is computed from
    time t_0 to t_{N-1}.  Note that the input u_N does not affect the state
    x_N and so it should always be returned as 0.  Further, if neither a
    terminal cost nor a terminal constraint is given, then the input at
    time point t_{N-1} does not affect the cost function and hence u_{N-1}
    will also be returned as zero.  If you want the trajectory cost to
    include state costs at time t_{N}, then you can set `terminal_cost` to
    be the same function as `cost`.

    Additional keyword parameters can be used to fine-tune the behavior of
    the underlying optimization and integration functions.  See
    `OptimalControlProblem` for more information.

    """
    # Process parameter and keyword arguments
    _process_kwargs(kwargs, _optimal_aliases)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, _optimal_aliases, sigval=None)
    cost = _process_param(
        'integral_cost', integral_cost, kwargs, _optimal_aliases)
    trajectory_constraints = _process_param(
        'trajectory_constraints', trajectory_constraints, kwargs,
        _optimal_aliases)
    return_states = _process_param(
        'return_states', return_states, kwargs, _optimal_aliases, sigval=True)

    # Process (legacy) method keyword (could be minimize or trajectory)
    if kwargs.get('method'):
        method = kwargs.pop('method')
        if method not in _optimal_trajectory_methods:
            if kwargs.get('minimize_method'):
                raise ValueError("'minimize_method' specified more than once")
            warnings.warn(
                "'method' parameter is deprecated; assuming minimize_method",
                FutureWarning)
            kwargs['minimize_method'] = method
        else:
            if kwargs.get('trajectory_method'):
                raise ValueError(
                    "'trajectory_method' specified more than once")
            warnings.warn(
                "'method' parameter is deprecated; assuming trajectory_method",
                FutureWarning)
            kwargs['trajectory_method'] = method

    # Set up the optimal control problem
    ocp = OptimalControlProblem(
        sys, timepts, cost, trajectory_constraints=trajectory_constraints,
        terminal_cost=terminal_cost, terminal_constraints=terminal_constraints,
        initial_guess=initial_guess, basis=basis, log=log, **kwargs)

    # Solve for the optimal input from the current state
    return ocp.compute_trajectory(
        X0, squeeze=squeeze, transpose=transpose, print_summary=print_summary,
        return_states=return_states)


# Create a model predictive controller for an optimal control problem
def create_mpc_iosystem(
        sys, timepts, integral_cost=None, trajectory_constraints=None,
        terminal_cost=None, terminal_constraints=None, log=False, **kwargs):
    """Create a model predictive I/O control system.

    This function creates an input/output system that implements a model
    predictive control for a system given the time points, cost function and
    constraints that define the finite-horizon optimization that should be
    carried out at each state.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the optimal input will be computed.
    timepts : 1D array_like
        List of times at which the optimal input should be computed.
    integral_cost (or cost) : callable
        Function that returns the integral cost given the current state
        and input.  Called as ``integral_cost(x, u)``.
    trajectory_constraints (or constraints) : list of tuples, optional
        List of constraints that should hold at each point in the time
        vector.  See `solve_optimal_trajectory` for more details.
    terminal_cost : callable, optional
        Function that returns the terminal cost given the final state
        and input.  Called as terminal_cost(x, u).
    terminal_constraints : list of tuples, optional
        List of constraints that should hold at the end of the trajectory.
        Same format as `constraints`.
    **kwargs
        Additional parameters, passed to `scipy.optimize.minimize` and
        `~control.NonlinearIOSystem`.

    Returns
    -------
    ctrl : `InputOutputSystem`
        An I/O system taking the current state of the model system and
        returning the current input to be applied that minimizes the cost
        function while satisfying the constraints.

    Other Parameters
    ----------------
    inputs, outputs, states : int or list of str, optional
        Set the names of the inputs, outputs, and states, as described in
        `InputOutputSystem`.
    log : bool, optional
        If True, turn on logging messages (using Python logging module).
        Use `logging.basicConfig` to enable logging output
        (e.g., to a file).
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Notes
    -----
    Additional keyword parameters can be used to fine-tune the behavior of
    the underlying optimization and integration functions.  See
    `OptimalControlProblem` for more information.

    """
    from .iosys import InputOutputSystem

    # Process parameter and keyword arguments
    _process_kwargs(kwargs, _optimal_aliases)
    cost = _process_param(
        'integral_cost', integral_cost, kwargs, _optimal_aliases)
    constraints = _process_param(
        'trajectory_constraints', trajectory_constraints, kwargs,
        _optimal_aliases)

    # Grab the keyword arguments known by this function
    iosys_kwargs = {}
    for kw in InputOutputSystem._kwargs_list:
        if kw in kwargs:
            iosys_kwargs[kw] = kwargs.pop(kw)

    # Set up the optimal control problem
    ocp = OptimalControlProblem(
        sys, timepts, cost, trajectory_constraints=constraints,
        terminal_cost=terminal_cost, terminal_constraints=terminal_constraints,
        log=log, **kwargs)

    # Return an I/O system implementing the model predictive controller
    return ocp.create_mpc_iosystem(**iosys_kwargs)


#
# Optimal (moving horizon) estimation problem
#

class OptimalEstimationProblem():
    """Description of a finite horizon, optimal estimation problem.

    The `OptimalEstimationProblem` class holds all of the information
    required to specify an optimal estimation problem: the system dynamics,
    cost function (negative of the log likelihood), and constraints.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the optimal input will be computed.
    timepts : 1D array
            Set up time points at which the inputs and outputs are given.
    integral_cost : callable
        Function that returns the integral cost given the estimated state,
        system inputs, and output error.  Called as integral_cost(xhat, u,
        v, w) where xhat is the estimated state, u is the applied input to
        the system, v is the estimated disturbance input, and w is the
        difference between the measured and the estimated output.
    trajectory_constraints : list of constraints, optional
       List of constraints that should hold at each point in the time
       vector.  Each element of the list should be an object of type
       `scipy.optimize.LinearConstraint` with arguments ``(A, lb,
       ub)`` or `scipy.optimize.NonlinearConstraint` with arguments
       ``(fun, lb, ub)``.  The constraints will be applied at each time point
       along the trajectory.
    terminal_cost : callable, optional
        Function that returns the terminal cost given the initial estimated
        state and expected value.  Called as terminal_cost(xhat, x0).
    control_indices : int, slice, or list of int or string, optional
        Specify the indices in the system input vector that correspond to
        the control inputs.  These inputs will be used as known control
        inputs for the estimator.  If value is an integer `m`, the first
        `m` system inputs are used.  Otherwise, the value should be a slice
        or a list of indices.  The list of indices can be specified as
        either integer offsets or as system input signal names.  If not
        specified, defaults to the complement of the disturbance indices
        (see also notes below).
    disturbance_indices : int, list of int, or slice, optional
        Specify the indices in the system input vector that correspond to
        the process disturbances.  If value is an integer `m`, the last `m`
        system inputs are used.  Otherwise, the value should be a slice or
        a list of indices, as described for `control_indices`.  If not
        specified, defaults to the complement of the control indices (see
        also notes below).

    Attributes
    ----------
    constraint: list of SciPy constraint objects
        List of `scipy.optimize.LinearConstraint` or
        `scipy.optimize.NonlinearConstraint` objects.
    constraint_lb, constrain_ub, eqconst_value : list of float
        List of constraint bounds.

    Other Parameters
    ----------------
    terminal_constraints : list of constraints, optional
        List of constraints that should hold at the terminal point in time,
        in the same form as `trajectory_constraints`.
    solve_ivp_method : str, optional
        Set the method used by `scipy.integrate.solve_ivp`.
    solve_ivp_kwargs : str, optional
        Pass additional keywords to `scipy.integrate.solve_ivp`.
    minimize_method : str, optional
        Set the method used by `scipy.optimize.minimize`.
    minimize_options : str, optional
        Set the options keyword used by `scipy.optimize.minimize`.
    minimize_kwargs : str, optional
        Pass additional keywords to `scipy.optimize.minimize`.

    Notes
    -----
    To describe an optimal estimation problem we need an input/output
    system, a set of time points, applied inputs and measured outputs, a
    cost function, and (optionally) a set of constraints on the state
    and/or inputs along the trajectory (and at the terminal time).  This
    class sets up an optimization over the state and disturbances at
    each point in time, using the integral and terminal costs as well as
    the trajectory constraints.  The `compute_estimate` method solves
    the underling optimization problem using `scipy.optimize.minimize`.

    The control input and disturbance indices can be specified using the
    `control_indices` and `disturbance_indices` keywords.  If only one is
    given, the other is assumed to be the remaining indices in the system
    input.  If neither is given, the disturbance inputs are assumed to be
    the same as the control inputs.

    The "cost" (e.g. negative of the log likelihood) of the estimated
    trajectory is computed using the estimated state, the disturbances and
    noise, and the measured output.  This is done by calling a user-defined
    function for the integral_cost along the trajectory and then adding the
    value of a user-defined terminal cost at the initial point in the
    trajectory.

    The constraint functions are evaluated at each point on the trajectory
    generated by the proposed estimate and disturbances.  As in the case of
    the cost function, the constraints are evaluated at the estimated
    state, inputs, and measured outputs along each point on the trajectory.
    This information is compared against the constraint upper and lower
    bounds.  The constraint function is processed in the class initializer,
    so that it only needs to be computed once.

    The default values for `minimize_method`, `minimize_options`,
    `minimize_kwargs`, `solve_ivp_method`, and `solve_ivp_options`
    can be set using `config.defaults['optimal.<keyword>']`.

    """
    def __init__(
            self, sys, timepts, integral_cost, terminal_cost=None,
            trajectory_constraints=None, control_indices=None,
            disturbance_indices=None, **kwargs):
        """Set up an optimal control problem."""
        # Save the basic information for use later
        self.system = sys
        self.timepts = timepts
        self.integral_cost = integral_cost
        self.terminal_cost = terminal_cost

        # Process keyword arguments
        self.minimize_kwargs = {}
        self.minimize_kwargs['method'] = kwargs.pop(
            'minimize_method', config.defaults['optimal.minimize_method'])
        self.minimize_kwargs['options'] = kwargs.pop(
            'minimize_options', config.defaults['optimal.minimize_options'])
        self.minimize_kwargs.update(kwargs.pop(
            'minimize_kwargs', config.defaults['optimal.minimize_kwargs']))

        # Save input and disturbance indices (and create input array)
        self.control_indices = control_indices
        self.disturbance_indices = disturbance_indices
        self.ctrl_idx, self.dist_idx = None, None
        self.inputs = np.zeros((sys.ninputs, len(timepts)))

        # Make sure there were no extraneous keywords
        if kwargs:
            raise TypeError("unrecognized keyword(s): ", str(kwargs))

        self.trajectory_constraints = _process_constraints(
            trajectory_constraints, "trajectory")

        #
        # Compute and store constraints
        #
        # While the constraints are evaluated during the execution of the
        # SciPy optimization method itself, we go ahead and pre-compute the
        # `scipy.optimize.NonlinearConstraint` function that will be passed to
        # the optimizer on initialization, since it doesn't change.  This is
        # mainly a matter of computing the lower and upper bound vectors,
        # which we need to "stack" to account for the evaluation at each
        # trajectory time point plus any terminal constraints (in a way that
        # is consistent with the `_constraint_function` that is used at
        # evaluation time.
        #
        # TODO: when using the collocation method, linear constraints on the
        # states and inputs can potentially maintain their linear structure
        # rather than being converted to nonlinear constraints.
        #
        constraint_lb, constraint_ub, eqconst_value = [], [], []

        # Go through each time point and stack the bounds
        for t in self.timepts:
            for type, fun, lb, ub in self.trajectory_constraints:
                if np.all(lb == ub):
                    # Equality constraint
                    eqconst_value.append(lb)
                else:
                    # Inequality constraint
                    constraint_lb.append(lb)
                    constraint_ub.append(ub)

        # Turn constraint vectors into 1D arrays
        self.constraint_lb = np.hstack(constraint_lb) if constraint_lb else []
        self.constraint_ub = np.hstack(constraint_ub) if constraint_ub else []
        self.eqconst_value = np.hstack(eqconst_value) if eqconst_value else []

        # Create the constraints (inequality and equality)
        # TODO: keep track of linear vs nonlinear
        self.constraints = []

        if len(self.constraint_lb) != 0:
            self.constraints.append(sp.optimize.NonlinearConstraint(
                self._constraint_function, self.constraint_lb,
                self.constraint_ub))

        if len(self.eqconst_value) != 0:
            self.constraints.append(sp.optimize.NonlinearConstraint(
                self._eqconst_function, self.eqconst_value,
                self.eqconst_value))

        # Add the collocation constraints
        colloc_zeros = np.zeros(sys.nstates * (self.timepts.size - 1))
        self.colloc_vals = np.zeros((sys.nstates, self.timepts.size - 1))
        self.constraints.append(sp.optimize.NonlinearConstraint(
            self._collocation_constraint, colloc_zeros, colloc_zeros))

        # Initialize run-time statistics
        self._reset_statistics()

    #
    # Cost function
    #
    # We are given the estimated states, applied inputs, and measured
    # outputs at each time point and we use a zero-order hold approximation
    # to compute the integral cost plus the terminal (initial) cost:
    #
    #   cost = sum_{k=1}^{N-1} integral_cost(xhat[k], u[k], v[k], w[k]) * dt
    #          + terminal_cost(xhat[0], x0)
    #
    def _cost_function(self, xvec):
        # Compute the estimated states and disturbance inputs
        xhat, u, v, w = self._compute_states_inputs(xvec)

        # Trajectory cost
        if ct.isctime(self.system):
            # Evaluate the costs
            costs = np.array([self.integral_cost(
                xhat[:, i], u[:, i], v[:, i], w[:, i]) for
                     i in range(self.timepts.size)])

            # Compute the time intervals and integrate the cost (trapezoidal)
            cost = 0.5 * (costs[:-1] + costs[1:]) @ np.diff(self.timepts)

        else:
            # Sum the integral cost over the time (second) indices
            # cost += self.integral_cost(xhat[:, i], u[:, i], v[:, i], w[:, i])
            cost = sum(map(self.integral_cost, xhat.T, u.T, v.T, w.T))

        # Terminal cost
        if self.terminal_cost is not None and self.x0 is not None:
            cost += self.terminal_cost(xhat[:, 0], self.x0)

        # Update statistics
        self.cost_evaluations += 1

        # Return the total cost for this input sequence
        return cost

    #
    # Constraints
    #
    # We are given the constraints along the trajectory and the terminal
    # constraints, which each take inputs [xhat, u, v, w] and evaluate the
    # constraint.  How we handle these depends on the type of constraint:
    #
    # * For linear constraints (LinearConstraint), a combined (hstack'd)
    #   vector of the estimate state and inputs is multiplied by the
    #   polytope A matrix for comparison against the upper and lower
    #   bounds.
    #
    # * For nonlinear constraints (NonlinearConstraint), a user-specific
    #   constraint function having the form
    #
    #      constraint_fun(xhat, u, v, w)
    #
    #   is called at each point along the trajectory and compared against the
    #   upper and lower bounds.
    #
    # * If the upper and lower bound for the constraint are identical, then
    #   we separate out the evaluation into two different constraints, which
    #   allows the SciPy optimizers to be more efficient (and stops them
    #   from generating a warning about mixed constraints).  This is handled
    #   through the use of the `_eqconst_function` and `eqconst_value`
    #   members.
    #
    # In both cases, the constraint is specified at a single point, but we
    # extend this to apply to each point in the trajectory.  This means
    # that for N time points with m trajectory constraints and p terminal
    # constraints we need to compute N*m + p constraints, each of which
    # holds at a specific point in time, and implements the original
    # constraint.
    #
    def _constraint_function(self, xvec):
        # Compute the estimated states and disturbance inputs
        xhat, u, v, w = self._compute_states_inputs(xvec)

        #
        # Evaluate the constraint function along the trajectory
        #
        # TODO: vectorize
        value = []
        for i, t in enumerate(self.timepts):
            for ctype, fun, lb, ub in self.trajectory_constraints:
                if np.all(lb == ub):
                    # Skip equality constraints
                    continue
                elif ctype == opt.LinearConstraint:
                    # `fun` is the A matrix associated with the polytope...
                    value.append(fun @ np.hstack(
                        [xhat[:, i], u[:, i], v[:, i], w[:, i]]))
                elif ctype == opt.NonlinearConstraint:
                    value.append(fun(xhat[:, i], u[:, i], v[:, i], w[:, i]))
                else:      # pragma: no cover
                    # Checked above => we should never get here
                    raise TypeError(f"unknown constraint type {ctype}")

        # Update statistics
        self.constraint_evaluations += 1

        # Return the value of the constraint function
        return np.hstack(value)

    def _eqconst_function(self, xvec):
        # Compute the estimated states and disturbance inputs
        xhat, u, v, w = self._compute_states_inputs(xvec)

        # Evaluate the constraint function along the trajectory
        # TODO: vectorize
        value = []
        for i, t in enumerate(self.timepts):
            for ctype, fun, lb, ub in self.trajectory_constraints:
                if np.any(lb != ub):
                    # Skip inequality constraints
                    continue
                elif ctype == opt.LinearConstraint:
                    # `fun` is the A matrix associated with the polytope...
                    value.append(fun @ np.hstack(
                        [xhat[:, i], u[:, i], v[:, i], w[:, i]]))
                elif ctype == opt.NonlinearConstraint:
                    value.append(fun(xhat[:, i], u[:, i], v[:, i], w[:, i]))
                else:      # pragma: no cover
                    # Checked above => we should never get here
                    raise TypeError(f"unknown constraint type {ctype}")

        # Update statistics
        self.eqconst_evaluations += 1

        # Return the value of the constraint function
        return np.hstack(value)

    def _collocation_constraint(self, xvec):
        # Compute the estimated states and disturbance inputs
        xhat, u, v, w = self._compute_states_inputs(xvec)

        # Create the input vector for the system
        self.inputs.fill(0.)
        self.inputs[self.ctrl_idx, :] = u
        self.inputs[self.dist_idx, :] += v

        if self.system.isctime():
            # Compute the collocation constraints
            # TODO: vectorize
            fk = self.system._rhs(
                self.timepts[0], xhat[:, 0], self.inputs[:, 0])
            for i, t in enumerate(self.timepts[:-1]):
                # From M. Kelly, SIAM Review (2017), equation (3.2), i = k+1
                # x[k+1] - x[k] = 0.5 hk (f(x[k+1], u[k+1] + f(x[k], u[k]))
                fkp1 = self.system._rhs(t, xhat[:, i+1], self.inputs[:, i+1])
                self.colloc_vals[:, i] = xhat[:, i+1] - xhat[:, i] - \
                    0.5 * (self.timepts[i+1] - self.timepts[i]) * (fkp1 + fk)
                fk = fkp1
        else:
            # TODO: vectorize
            for i, t in enumerate(self.timepts[:-1]):
                # x[k+1] = f(x[k], u[k], v[k])
                self.colloc_vals[:, i] = xhat[:, i+1] - \
                    self.system._rhs(t, xhat[:, i], self.inputs[:, i])

        # Return the value of the constraint function
        return self.colloc_vals.reshape(-1)

    #
    # Initial guess processing
    #
    def _process_initial_guess(self, initial_guess):
        if initial_guess is None:
            return np.zeros(
                (self.system.nstates + self.ndisturbances) * self.timepts.size)
        else:
            if initial_guess[0].shape != \
               (self.system.nstates, self.timepts.size):
                raise ValueError(
                    "initial guess for state estimate must have shape "
                    f"{self.system.nstates} x {self.timepts.size}")

            elif initial_guess[1].shape != \
                 (self.ndisturbances, self.timepts.size):
                raise ValueError(
                    "initial guess for disturbances must have shape "
                    f"{self.ndisturbances} x {self.timepts.size}")

            return np.hstack([
                initial_guess[0].reshape(-1),           # estimated states
                initial_guess[1].reshape(-1)])          # disturbances

    #
    # Compute the states and inputs from the optimization parameter vector
    # and the internally stored inputs and measured outputs.
    #
    # The optimization parameter vector has elements (xhat[0], ...,
    # xhat[N-1], v[0], ..., v[N-2]) where N is the number of time
    # points.  The system inputs u and measured outputs y are locally
    # stored in self.u and self.y when compute_estimate() is called.
    #
    def _compute_states_inputs(self, xvec):
        # Extract the state estimate and disturbances
        xhat = xvec[:self.system.nstates * self.timepts.size].reshape(
            self.system.nstates, -1)
        v = xvec[self.system.nstates * self.timepts.size:].reshape(
            self.ndisturbances, -1)

        # Create the input vector for the system
        self.inputs[self.ctrl_idx, :] = self.u
        self.inputs[self.dist_idx, :] = v

        # Compute the estimated output
        yhat = np.vstack([
            self.system._out(self.timepts[i], xhat[:, i], self.inputs[:, i])
            for i in range(self.timepts.size)]).T

        return xhat, self.u, v, self.y - yhat

    #
    # Optimization statistics
    #
    # To allow some insight into where time is being spent, we keep track
    # of the number of times that various functions are called and (TODO)
    # how long we spent inside each function.
    #
    def _reset_statistics(self):
        """Reset counters for keeping track of statistics"""
        self.cost_evaluations, self.cost_process_time = 0, 0
        self.constraint_evaluations, self.constraint_process_time = 0, 0
        self.eqconst_evaluations, self.eqconst_process_time = 0, 0

    def _print_statistics(self, reset=True):
        """Print out summary statistics from last run"""
        print("Summary statistics:")
        print("* Cost function calls:", self.cost_evaluations)
        if self.constraint_evaluations:
            print("* Constraint calls:", self.constraint_evaluations)
        if self.eqconst_evaluations:
            print("* Eqconst calls:", self.eqconst_evaluations)
        if reset:
            self._reset_statistics()

    #
    # Optimal estimate computations
    #
    def compute_estimate(
            self, outputs=None, inputs=None, initial_state=None,
            initial_guess=None, squeeze=None, print_summary=True, **kwargs):
        """Compute the optimal input at state x.

        Parameters
        ----------
        outputs (or Y) : 2D array
            Measured outputs at each time point.
        inputs (or U) : 2D array
            Applied inputs at each time point.
        initial_state (or X0) : 1D array
            Expected initial value of the state.
        initial_guess : 2-tuple of 2D arrays
            A 2-tuple consisting of the estimated states and disturbance
            values to use as a guess for the optimal estimated trajectory.
        squeeze : bool, optional
            If True and if the system has a single disturbance input and
            single measured output, return the system input and output as a
            1D array rather than a 2D array.  If False, return the system
            output as a 2D array even if the system is SISO.  Default value
            set by `config.defaults['control.squeeze_time_response']`.
        print_summary : bool, optional
            If True (default), print a short summary of the computation.

        Returns
        -------
        res : `OptimalEstimationResult`
            Bundle object with the results of the optimal estimation problem.
        res.success : bool
            Boolean flag indicating whether the optimization was successful.
        res.time : array
            Time values of the input (same as self.timepts).
        res.inputs : array
            Estimated disturbance inputs for the system trajectory.
        res.states : array
            Time evolution of the estimated state vector.
        res.outputs : array
            Estimated measurement noise for the system trajectory.

        """
        # Argument and keyword processing
        aliases = _timeresp_aliases | _optimal_aliases
        _process_kwargs(kwargs, aliases)
        Y = _process_param('outputs', outputs, kwargs, aliases)
        U = _process_param('inputs', inputs, kwargs, aliases)
        X0 = _process_param('initial_state', initial_state, kwargs, aliases)

        if kwargs:
            raise TypeError("unrecognized keyword(s): ", str(kwargs))

        # Store the inputs and outputs (for use in _constraint_function)
        self.u = np.atleast_1d(U).reshape(-1, self.timepts.size)
        self.y = np.atleast_1d(Y).reshape(-1, self.timepts.size)
        self.x0 = X0

        # Figure out the number of disturbances
        if self.disturbance_indices is None and self.control_indices is None:
            self.ctrl_idx, self.dist_idx = \
                _process_control_disturbance_indices(
                    self.system, None, self.system.ninputs - self.u.shape[0])
        elif self.ctrl_idx is None or self.dist_idx is None:
            self.ctrl_idx, self.dist_idx = \
                _process_control_disturbance_indices(
                    self.system, self.control_indices,
                    self.disturbance_indices)
        self.ndisturbances = len(self.dist_idx)

        # Make sure the dimensions of the inputs are OK
        if self.u.shape[0] != len(self.ctrl_idx):
            raise ValueError(
                "input vector is incorrect shape; "
                f"should be {len(self.ctrl_idx)} x {self.timepts.size}")
        if self.y.shape[0] != self.system.noutputs:
            raise ValueError(
                "measurements vector is incorrect shape; "
                f"should be {self.system.noutputs} x {self.timepts.size}")

        # Process the initial guess
        initial_guess = self._process_initial_guess(initial_guess)

        # Call SciPy optimizer
        res = sp.optimize.minimize(
            self._cost_function, initial_guess,
            constraints=self.constraints, **self.minimize_kwargs)

        # Process and return the results
        return OptimalEstimationResult(
            self, res, squeeze=squeeze, print_summary=print_summary)

    #
    # Create an input/output system implementing an moving horizon estimator
    #
    # This function creates an input/output system that has internal state
    # xhat, u, v, y for all previous time points.  When the system update
    # function is called,
    #

    def create_mhe_iosystem(
            self, estimate_labels=None, measurement_labels=None,
            control_labels=None, inputs=None, outputs=None, **kwargs):
        """Create an I/O system implementing an MPC controller.

        This function creates an input/output system that implements a
        moving horizon estimator for a an optimal estimation problem.  The
        I/O system takes the system measurements and applied inputs as as
        inputs and outputs the estimated state.

        Parameters
        ----------
        estimate_labels : str or list of str, optional
            Set the name of the signals to use for the estimated state
            (estimator outputs).  If a single string is specified, it
            should be a format string using the variable `i` as an index.
            Otherwise, a list of strings matching the size of the estimated
            state should be used.  Default is "xhat[{i}]".  These settings
            can also be overridden using the `outputs` keyword.
        measurement_labels, control_labels : str or list of str, optional
            Set the names of the measurement and control signal names
            (estimator inputs).  If a single string is specified, it should
            be a format string using the variable `i` as an index.
            Otherwise, a list of strings matching the size of the system
            inputs and outputs should be used.  Default is the signal names
            for the system outputs and control inputs. These settings can
            also be overridden using the `inputs` keyword.
        **kwargs, optional
            Additional keyword arguments to set system, input, and output
            signal names; see `InputOutputSystem`.

        Returns
        -------
        estim : `InputOutputSystem`
            An I/O system taking the measured output and applied input for
            the model system and returning the estimated state of the
            system, as determined by solving the optimal estimation problem.

        Notes
        -----
        The labels for the input signals for the system are determined
        based on the signal names for the system model used in the optimal
        estimation problem.  The system name and signal names can be
        overridden using the `name`, `input`, and `output` keywords, as
        described in `InputOutputSystem`.

        """
        # Check to make sure we are in discrete time
        if self.system.dt == 0:
            raise ct.ControlNotImplemented(
                "MHE for continuous-time systems not implemented")

        # Figure out the location of the disturbances
        self.ctrl_idx, self.dist_idx = \
            _process_control_disturbance_indices(
                self.system, self.control_indices, self.disturbance_indices)

        # Figure out the signal labels to use
        estimate_labels = _process_labels(
            estimate_labels, 'estimate',
            [f'xhat[{i}]' for i in range(self.system.nstates)])
        outputs = estimate_labels if outputs is None else outputs

        measurement_labels = _process_labels(
            measurement_labels, 'measurement', self.system.output_labels)
        control_labels = _process_labels(
            control_labels, 'control',
            [self.system.input_labels[i] for i in self.ctrl_idx])
        inputs = measurement_labels + control_labels if inputs is None \
            else inputs

        nstates = (self.system.nstates + self.system.ninputs
                   + self.system.noutputs) * self.timepts.size
        if kwargs.get('states'):
            raise ValueError("user-specified state signal names not allowed")

        # Utility function to extract elements from MHE state vector
        def _xvec_next(xvec, off, size):
            len_ = size * self.timepts.size
            return (off + len_,
                    xvec[off:off + len_].reshape(-1, self.timepts.size))

        # Update function for the estimator
        def _mhe_update(t, xvec, uvec, params={}):
            # Inputs are the measurements and inputs
            y = uvec[:self.system.noutputs].reshape(-1, 1)
            u = uvec[self.system.noutputs:].reshape(-1, 1)

            # Estimator state = [xhat, v, Y, U]
            off, xhat = _xvec_next(xvec, 0, self.system.nstates)
            off, U = _xvec_next(xvec, off, len(self.ctrl_idx))
            off, V = _xvec_next(xvec, off, len(self.dist_idx))
            off, Y = _xvec_next(xvec, off, self.system.noutputs)

            # Shift the states and add the new measurements and inputs
            # TODO: look for Numpy shift() operator
            xhat = np.hstack([xhat[:, 1:], xhat[:, -1:]])
            U = np.hstack([U[:, 1:], u])
            V = np.hstack([V[:, 1:], V[:, -1:]])
            Y = np.hstack([Y[:, 1:], y])

            # Compute the new states and disturbances
            est = self.compute_estimate(
                Y, U, initial_state=xhat[:, 0], initial_guess=(xhat, V),
                print_summary=False)

            # Restack the new state
            return np.hstack([
                est.states.reshape(-1), U.reshape(-1),
                est.inputs.reshape(-1), Y.reshape(-1)])

        # Output function
        def _mhe_output(t, xvec, uvec, params={}):
            # Get the states and inputs
            off, xhat = _xvec_next(xvec, 0, self.system.nstates)
            off, u_v = _xvec_next(xvec, off, self.system.ninputs)

            # Compute the estimate at the next time point
            return self.system._rhs(t, xhat[:, -1], u_v[:, -1])

        return ct.NonlinearIOSystem(
            _mhe_update, _mhe_output, dt=self.system.dt,
            states=nstates, inputs=inputs, outputs=outputs, **kwargs)


# Optimal estimation result
class OptimalEstimationResult(sp.optimize.OptimizeResult):
    """Result from solving an optimal estimation problem.

    This class is a subclass of `scipy.optimize.OptimizeResult` with
    additional attributes associated with solving optimal estimation
    problems.

    Parameters
    ----------
    oep : OptimalEstimationProblem
        Optimal estimation problem that generated this solution.
    res : scipy.minimize.OptimizeResult
        Result of optimization.
    print_summary : bool, optional
        If True (default), print a short summary of the computation.
    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default
        value set by `config.defaults['control.squeeze_time_response']`.

    Attributes
    ----------
    states : ndarray
        Estimated state trajectory.
    inputs : ndarray
        The disturbances associated with the estimated state trajectory.
    outputs :
        The error between measured outputs and estimated outputs.
    success : bool
        Whether or not the optimizer exited successful.
    problem : OptimalControlProblem
        Optimal control problem that generated this solution.
    cost : float
        Final cost of the return solution.
    system_simulations, {cost, constraint, eqconst}_evaluations : int
        Number of system simulations and evaluations of the cost function,
        (inequality) constraint function, and equality constraint function
        performed during the optimization.
    cost_process_time, constraint_process_time, eqconst_process_time : float
        If logging was enabled, the amount of time spent evaluating the cost
        and constraint functions.

    """
    def __init__(
            self, oep, res, return_states=True, print_summary=False,
            transpose=None, squeeze=None):
        """Create a OptimalControlResult object"""

        # Copy all of the fields we were sent by sp.optimize.minimize()
        for key, val in res.items():
            setattr(self, key, val)

        # Remember the optimal control problem that we solved
        self.problem = oep

        # Parse the optimization variables into states and inputs
        xhat, u, v, w = oep._compute_states_inputs(res.x)

        # See if we got an answer
        if not res.success:
            warnings.warn(
                "unable to solve optimal control problem\n"
                "scipy.optimize.minimize returned " + res.message, UserWarning)

        # Save the final cost
        self.cost = res.fun

        # Optionally print summary information
        if print_summary:
            oep._print_statistics()
            print("* Final cost:", self.cost)

        # Process data as a time response (with "outputs" = inputs)
        response = ct.TimeResponseData(
            oep.timepts, w, xhat, v, issiso=oep.system.issiso(),
            squeeze=squeeze)

        self.time = response.time
        self.inputs = response.inputs
        self.states = response.states
        self.outputs = response.outputs


# Compute the finite horizon estimate for a nonlinear system
def solve_optimal_estimate(
        sys, timepts, outputs=None, inputs=None, integral_cost=None,
        initial_state=None, trajectory_constraints=None, initial_guess=None,
        squeeze=None, print_summary=True, **kwargs):

    """Compute the solution to a finite horizon estimation problem.

    This function computes the maximum likelihood estimate of a system
    state given the input and output over a fixed horizon.  The likelihood
    is evaluated according to a cost function whose value is minimized
    to compute the maximum likelihood estimate.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the optimal input will be computed.
    timepts : 1D array_like
        List of times at which the optimal input should be computed.
    outputs (or Y) : 2D array_like
        Values of the outputs at each time point.
    inputs (or U) : 2D array_like
        Values of the inputs at each time point.
    integral_cost (or cost) : callable
        Function that returns the cost given the current state
        and input.  Called as ``cost(y, u, x0)``.
    initial_state (or X0) : 1D array_like, optional
        Mean value of the initial condition (defaults to 0).
    trajectory_constraints : list of tuples, optional
        List of constraints that should hold at each point in the time
        vector.  See `solve_optimal_trajectory` for more information.
    control_indices : int, slice, or list of int or string, optional
        Specify the indices in the system input vector that correspond to
        the control inputs.  For more information on possible values, see
        `OptimalEstimationProblem`.
    disturbance_indices : int, list of int, or slice, optional
        Specify the indices in the system input vector that correspond to
        the input disturbances.  For more information on possible values, see
        `OptimalEstimationProblem`.
    initial_guess : 2D array_like, optional
        Initial guess for the state estimate at each time point.
    print_summary : bool, optional
        If True (default), print a short summary of the computation.
    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default
        value set by `config.defaults['control.squeeze_time_response']`.

    Returns
    -------
    res : `TimeResponseData`
        Bundle object with the estimated state and noise values.
    res.success : bool
        Boolean flag indicating whether the optimization was successful.
    res.time : array
        Time values of the input.
    res.inputs : array
        Disturbance values corresponding to the estimated state.  If the
        system is SISO and squeeze is not True, the array is 1D (indexed by
        time).  If the system is not SISO or squeeze is False, the array is
        2D (indexed by the output number and time).
    res.states : array
        Estimated state vector over the given time points.
    res.outputs : array
        Noise values corresponding to the estimated state.  If the system
        is SISO and squeeze is not True, the array is 1D (indexed by time).
        If the system is not SISO or squeeze is False, the array is 2D
        (indexed by the output number and time).

    Notes
    -----
    Additional keyword parameters can be used to fine-tune the behavior of
    the underlying optimization and integration functions.  See
    `OptimalControlProblem` for more information.

    """
    aliases = _timeresp_aliases | _optimal_aliases
    _process_kwargs(kwargs, aliases)
    Y = _process_param('outputs', outputs, kwargs, aliases)
    U = _process_param('inputs', inputs, kwargs, aliases)
    X0 = _process_param(
        'initial_state', initial_state, kwargs, aliases)
    trajectory_cost = _process_param(
        'integral_cost', integral_cost, kwargs, aliases)

    # Set up the optimal control problem
    oep = OptimalEstimationProblem(
        sys, timepts, trajectory_cost,
        trajectory_constraints=trajectory_constraints, **kwargs)

    # Solve for the optimal input from the current state
    return oep.compute_estimate(
        Y, U, initial_state=X0, initial_guess=initial_guess,
        squeeze=squeeze, print_summary=print_summary)


#
# Functions to create cost functions (quadratic cost function)
#
# Since a quadratic function is common as a cost function, we provide a
# function that will take a Q and R matrix and return a callable that
# evaluates to associated quadratic cost.  This is compatible with the way that
# the `_cost_function` evaluates the cost at each point in the trajectory.
#
def quadratic_cost(sys, Q, R, x0=0, u0=0):
    """Create quadratic cost function.

    Returns a quadratic cost function that can be used for an optimal control
    problem.  The cost function is of the form

      cost = (x - x0)^T Q (x - x0) + (u - u0)^T R (u - u0)

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the cost function is being defined.
    Q : 2D array_like
        Weighting matrix for state cost.  Dimensions must match system state.
    R : 2D array_like
        Weighting matrix for input cost.  Dimensions must match system input.
    x0 : 1D array
        Nominal value of the system state (for which cost should be zero).
    u0 : 1D array
        Nominal value of the system input (for which cost should be zero).

    Returns
    -------
    cost_fun : callable
        Function that can be used to evaluate the cost at a given state and
        input.  The call signature of the function is cost_fun(x, u).

    """
    # Process the input arguments
    if Q is not None:
        Q = np.atleast_2d(Q)
        if Q.size == 1:         # allow scalar weights
            Q = np.eye(sys.nstates) * Q.item()
        elif Q.shape != (sys.nstates, sys.nstates):
            raise ValueError("Q matrix is the wrong shape")

    if R is not None:
        R = np.atleast_2d(R)
        if R.size == 1:         # allow scalar weights
            R = np.eye(sys.ninputs) * R.item()
        elif R.shape != (sys.ninputs, sys.ninputs):
            raise ValueError("R matrix is the wrong shape")

    if Q is None:
        return lambda x, u: ((u-u0) @ R @ (u-u0)).item()

    if R is None:
        return lambda x, u: ((x-x0) @ Q @ (x-x0)).item()

    # Received both Q and R matrices
    return lambda x, u: ((x-x0) @ Q @ (x-x0) + (u-u0) @ R @ (u-u0)).item()


def gaussian_likelihood_cost(sys, Rv, Rw=None):
    """Create cost function for Gaussian likelihoods.

    Returns a quadratic cost function that can be used for an optimal
    estimation problem.  The cost function is of the form

      cost = v^T R_v^{-1} v + w^T R_w^{-1} w

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the cost function is being defined.
    Rv : 2D array_like
        Covariance matrix for input (or state) disturbances.
    Rw : 2D array_like
        Covariance matrix for sensor noise.

    Returns
    -------
    cost_fun : callable
        Function that can be used to evaluate the cost for a given
        disturbance and sensor noise.  The call signature of the function
        is cost_fun(v, w).

    """
    # Process the input arguments
    if Rv is not None:
        Rv = np.atleast_2d(Rv)

    if Rw is not None:
        Rw = np.atleast_2d(Rw)
        if Rw.size == 1:         # allow scalar weights
            Rw = np.eye(sys.noutputs) * Rw.item()
        elif Rw.shape != (sys.noutputs, sys.noutputs):
            raise ValueError("Rw matrix is the wrong shape")

    if Rv is None:
        return lambda xhat, u, v, w: (w @ np.linalg.inv(Rw) @ w).item()

    if Rw is None:
        return lambda xhat, u, v, w: (v @ np.linalg.inv(Rv) @ v).item()

    # Received both Rv and Rw matrices
    return lambda xhat, u, v, w: \
        (v @ np.linalg.inv(Rv) @ v + w @ np.linalg.inv(Rw) @ w).item()


#
# Functions to create constraints: either polytopes (A x <= b) or ranges
# (lb # <= x <= ub).
#
# As in the cost function evaluation, the main "trick" in creating a
# constraint on the state or input is to properly evaluate the constraint on
# the stacked state and input vector at the current time point.  The
# constraint itself will be called at each point along the trajectory (or the
# endpoint) via the constrain_function() method.
#
# Note that these functions to not actually evaluate the constraint, they
# simply return the information required to do so.  We use the SciPy
# optimization methods LinearConstraint and NonlinearConstraint as "types" to
# keep things consistent with the terminology in scipy.optimize.
#
def state_poly_constraint(sys, A, b):
    """Create state constraint from polytope.

    Creates a linear constraint on the system state of the form A x <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix.
    b : 1D array
        Upper bound for the constraint.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert arguments to arrays and make sure dimensions are right
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)
    if len(A.shape) != 2 or A.shape[1] != sys.nstates:
        raise ValueError("polytope matrix must match number of states")
    elif len(b.shape) != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("number of bounds must match number of constraints")

    # Return a linear constraint object based on the polynomial
    return (opt.LinearConstraint,
            np.hstack([A, np.zeros((A.shape[0], sys.ninputs))]),
            np.full(A.shape[0], -np.inf), b)


def state_range_constraint(sys, lb, ub):
    """Create state constraint from range.

    Creates a linear constraint on the system state that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of 'inf' and '-inf' to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    lb : 1D array
        Lower bound for each of the states.
    ub : 1D array
        Upper bound for each of the states.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert bounds to lists and make sure they are the right dimension
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    if lb.shape != (sys.nstates,) or ub.shape != (sys.nstates,):
        raise ValueError("state bounds must match number of states")

    # Return a linear constraint object based on the polynomial
    return (opt.LinearConstraint,
            np.hstack(
                [np.eye(sys.nstates), np.zeros((sys.nstates, sys.ninputs))]),
            np.array(lb), np.array(ub))


# Create a constraint polytope on the system input
def input_poly_constraint(sys, A, b):
    """Create input constraint from polytope.

    Creates a linear constraint on the system input of the form A u <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix.
    b : 1D array
        Upper bound for the constraint.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert arguments to arrays and make sure dimensions are right
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)
    if len(A.shape) != 2 or A.shape[1] != sys.ninputs:
        raise ValueError("polytope matrix must match number of inputs")
    elif len(b.shape) != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("number of bounds must match number of constraints")

    # Return a linear constraint object based on the polynomial
    return (opt.LinearConstraint,
            np.hstack(
                [np.zeros((A.shape[0], sys.nstates)), A]),
            np.full(A.shape[0], -np.inf), b)


def input_range_constraint(sys, lb, ub):
    """Create input constraint from polytope.

    Creates a linear constraint on the system input that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of 'inf' and '-inf' to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    lb : 1D array
        Lower bound for each of the inputs.
    ub : 1D array
        Upper bound for each of the inputs.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert bounds to lists and make sure they are the right dimension
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    if lb.shape != (sys.ninputs,) or ub.shape != (sys.ninputs,):
        raise ValueError("input bounds must match number of inputs")

    # Return a linear constraint object based on the polynomial
    return (opt.LinearConstraint,
            np.hstack(
                [np.zeros((sys.ninputs, sys.nstates)), np.eye(sys.ninputs)]),
            lb, ub)


#
# Create a constraint polytope/range constraint on the system output
#
# Unlike the state and input constraints, for the output constraint we need
# to do a function evaluation before applying the constraints.
#
# TODO: for the special case of an LTI system, we can avoid the extra
# function call by multiplying the state by the C matrix for the system and
# then imposing a linear constraint:
#
#     np.hstack(
#         [A @ sys.C, np.zeros((A.shape[0], sys.ninputs))])
#
def output_poly_constraint(sys, A, b):
    """Create output constraint from polytope.

    Creates a linear constraint on the system output of the form A y <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix.
    b : 1D array
        Upper bound for the constraint.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert arguments to arrays and make sure dimensions are right
    A = np.atleast_2d(A)
    b = np.atleast_1d(b)
    if len(A.shape) != 2 or A.shape[1] != sys.noutputs:
        raise ValueError("polytope matrix must match number of outputs")
    elif len(b.shape) != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("number of bounds must match number of constraints")

    # Function to create the output
    def _evaluate_output_poly_constraint(x, u):
        return A @ sys._out(0, x, u)

    # Return a nonlinear constraint object based on the polynomial
    return (opt.NonlinearConstraint,
            _evaluate_output_poly_constraint,
            np.full(A.shape[0], -np.inf), b)


def output_range_constraint(sys, lb, ub):
    """Create output constraint from range.

    Creates a linear constraint on the system output that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of 'inf' and '-inf' to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    lb : 1D array
        Lower bound for each of the outputs.
    ub : 1D array
        Upper bound for each of the outputs.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert bounds to lists and make sure they are the right dimension
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    if lb.shape != (sys.noutputs,) or ub.shape != (sys.noutputs,):
        raise ValueError("output bounds must match number of outputs")

    # Function to create the output
    def _evaluate_output_range_constraint(x, u):
        # Separate the constraint into states and inputs
        return sys._out(0, x, u)

    # Return a nonlinear constraint object based on the polynomial
    return (opt.NonlinearConstraint, _evaluate_output_range_constraint, lb, ub)


#
# Create a constraint on the disturbance input
#

def disturbance_range_constraint(sys, lb, ub):
    """Create constraint for bounded disturbances.

    This function computes a constraint that puts a bound on the size of
    input disturbances.  The output of this function can be passed as a
    trajectory constraint for optimal estimation problems.

    Parameters
    ----------
    sys : `InputOutputSystem`
        I/O system for which the constraint is being defined.
    lb : 1D array
        Lower bound for each of the disturbance.
    ub : 1D array
        Upper bound for each of the disturbance.

    Returns
    -------
    constraint : tuple
        A tuple consisting of the constraint type and parameter values.

    """
    # Convert bounds to lists and make sure they are the right dimension
    lb = np.atleast_1d(lb).reshape(-1)
    ub = np.atleast_1d(ub).reshape(-1)
    if lb.shape != ub.shape:
        raise ValueError("upper and lower bound shapes must match")
    ndisturbances = lb.size

    # Generate a linear constraint on the input disturbances
    xvec_len = sys.nstates + sys.ninputs + sys.noutputs
    A = np.zeros((ndisturbances, xvec_len))
    A[:, sys.nstates + sys.ninputs - ndisturbances:-sys.noutputs] = \
        np.eye(ndisturbances)
    return opt.LinearConstraint(A, lb, ub)

#
# Utility functions
#


#
# Process trajectory constraints
#
# Constraints were originally specified as a tuple with the type of
# constraint followed by the arguments.  However, they are now specified
# directly as SciPy constraint objects.
#
# The _process_constraints() function will covert everything to a consistent
# internal representation (currently a tuple with the constraint type as the
# first element.
#

def _process_constraints(clist, name):
    if clist is None:
        clist = []
    elif isinstance(
            clist, (tuple, opt.LinearConstraint, opt.NonlinearConstraint)):
        clist = [clist]
    elif not isinstance(clist, list):
        raise TypeError(f"{name} constraints must be a list")

    # Process individual list elements
    constraint_list = []
    for constraint in clist:
        if isinstance(constraint, tuple):
            # Original style of constraint
            ctype, fun, lb, ub = constraint
            if ctype not in [opt.LinearConstraint, opt.NonlinearConstraint]:
                raise TypeError(f"unknown {name} constraint type {ctype}")
            constraint_list.append(constraint)
        elif isinstance(constraint, opt.LinearConstraint):
            constraint_list.append(
                (opt.LinearConstraint, constraint.A,
                 constraint.lb, constraint.ub))
        elif isinstance(constraint, opt.NonlinearConstraint):
            constraint_list.append(
                (opt.NonlinearConstraint, constraint.fun,
                 constraint.lb, constraint.ub))

    return constraint_list


# Convenience aliases
solve_ocp = solve_optimal_trajectory
solve_oep = solve_optimal_estimate
