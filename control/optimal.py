# optimal.py - optimization based control module
#
# RMM, 11 Feb 2021
#

"""The :mod:`~control.optimal` module provides support for optimization-based
controllers for nonlinear systems with state and input constraints.

"""

import numpy as np
import scipy as sp
import scipy.optimize as opt
import control as ct
import warnings
import logging
import time

from . import config
from .exception import ControlNotImplemented
from .timeresp import TimeResponseData

# Define module default parameter values
_optimal_trajectory_methods = {'shooting', 'collocation'}
_optimal_defaults = {
    'optimal.minimize_method': None,
    'optimal.minimize_options': {},
    'optimal.minimize_kwargs': {},
    'optimal.solve_ivp_method': None,
    'optimal.solve_ivp_options': {},
}


class OptimalControlProblem():
    """Description of a finite horizon, optimal control problem.

    The `OptimalControlProblem` class holds all of the information required to
    specify an optimal control problem: the system dynamics, cost function,
    and constraints.  As much as possible, the information used to specify an
    optimal control problem matches the notation and terminology of the SciPy
    `optimize.minimize` module, with the hope that this makes it easier to
    remember how to describe a problem.

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the optimal input will be computed.
    timepts : 1D array_like
        List of times at which the optimal input should be computed.
    integral_cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as integral_cost(x, u).
    trajectory_constraints : list of constraints, optional
       List of constraints that should hold at each point in the time
       vector.  Each element of the list should be an object of type
       :class:`~scipy.optimize.LinearConstraint` with arguments `(A, lb,
       ub)` or :class:`~scipy.optimize.NonlinearConstraint` with arguments
       `(fun, lb, ub)`.  The constraints will be applied at each time point
       along the trajectory.
    terminal_cost : callable, optional
        Function that returns the terminal cost given the current state
        and input.  Called as terminal_cost(x, u).
    trajectory_method : string, optional
        Method to use for carrying out the optimization. Currently supported
        methods are 'shooting' and 'collocation' (continuous time only). The
        default value is 'shooting' for discrete time systems and
        'collocation' for continuous time systems
    initial_guess : (tuple of) 1D or 2D array_like
        Initial states and/or inputs to use as a guess for the optimal
        trajectory.  For shooting methods, an array of inputs for each time
        point should be specified.  For collocation methods, the initial
        guess is either the input vector or a tuple consisting guesses for
        the state and the input.  Guess should either be a 2D vector of
        shape (ninputs, ntimepts) or a 1D input of shape (ninputs,) that
        will be broadcast by extension of the time axis.
    log : bool, optional
        If `True`, turn on logging messages (using Python logging module).
        Use :py:func:`logging.basicConfig` to enable logging output
        (e.g., to a file).

    Returns
    -------
    ocp : OptimalControlProblem
        Optimal control problem object, to be used in computing optimal
        controllers.

    Other Parameters
    ----------------
    basis : BasisFamily, optional
        Use the given set of basis functions for the inputs instead of
        setting the value of the input at each point in the timepts vector.
    terminal_constraints : list of constraints, optional
        List of constraints that should hold at the terminal point in time,
        in the same form as `trajectory_constraints`.
    solve_ivp_method : str, optional
        Set the method used by :func:`scipy.integrate.solve_ivp`.
    solve_ivp_kwargs : str, optional
        Pass additional keywords to :func:`scipy.integrate.solve_ivp`.
    minimize_method : str, optional
        Set the method used by :func:`scipy.optimize.minimize`.
    minimize_options : str, optional
        Set the options keyword used by :func:`scipy.optimize.minimize`.
    minimize_kwargs : str, optional
        Pass additional keywords to :func:`scipy.optimize.minimize`.

    Notes
    -----
    To describe an optimal control problem we need an input/output system, a
    time horizon, a cost function, and (optionally) a set of constraints on
    the state and/or input, either along the trajectory and at the terminal
    time.  This class sets up an optimization over the inputs at each point in
    time, using the integral and terminal costs as well as the trajectory and
    terminal constraints.  The `compute_trajectory` method sets up an
    optimization problem that can be solved using
    :func:`scipy.optimize.minimize`.

    The `_cost_function` method takes the information computes the cost of the
    trajectory generated by the proposed input.  It does this by calling a
    user-defined function for the integral_cost given the current states and
    inputs at each point along the trajectory and then adding the value of a
    user-defined terminal cost at the final point in the trajectory.

    The `_constraint_function` method evaluates the constraint functions along
    the trajectory generated by the proposed input.  As in the case of the
    cost function, the constraints are evaluated at the state and input along
    each point on the trajectory.  This information is compared against the
    constraint upper and lower bounds.  The constraint function is processed
    in the class initializer, so that it only needs to be computed once.

    If `basis` is specified, then the optimization is done over coefficients
    of the basis elements.  Otherwise, the optimization is performed over the
    values of the input at the specified times (using linear interpolation for
    continuous systems).

    The default values for ``minimize_method``, ``minimize_options``,
    ``minimize_kwargs``, ``solve_ivp_method``, and ``solve_ivp_options`` can
    be set using config.defaults['optimal.<keyword>'].

    """
    def __init__(
            self, sys, timepts, integral_cost, trajectory_constraints=[],
            terminal_cost=None, terminal_constraints=[], initial_guess=None,
            trajectory_method=None, basis=None, log=False, **kwargs):
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
            raise NotImplementedError(f"Unkown method {method}")

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
                    " discrete time systems")

        # Make sure there were no extraneous keywords
        if kwargs:
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
            logging.info("New optimal control problem initailized")

    #
    # Cost function
    #
    # For collocation methods we are given the states and inputs at each
    # time point and we use a trapezoidal approximation to compute the
    # integral cost, then add on the terminal cost.
    #
    # For shooting methods, given the input U = [u[0], ... u[N]] we need to
    # compute the cost of the trajectory generated by that input.  This
    # means we have to simulate the system to get the state trajectory X =
    # [x[0], ..., x[N]] and then compute the cost at each point:
    #
    #   cost = sum_k integral_cost(x[k], u[k]) + terminal_cost(x[N], u[N])
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
            # TODO: vectorize
            cost = 0
            for i in range(self.timepts.size-1):
                # Approximate the integral using trapezoidal rule
                cost += 0.5 * (costs[i] + costs[i+1]) * dt[i]

        else:
            # Sum the integral cost over the time (second) indices
            # cost += self.integral_cost(states[:,i], inputs[:,i])
            cost = sum(map(
                self.integral_cost, np.transpose(states[:, :-1]),
                np.transpose(inputs[:, :-1])))

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
                "collocation not yet implemented for discrete time systems")

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
    # and/or state trajectory (for collocaiton methods).
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
    # collocation points given the ceofficient (optimizer state) vector.
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
                self.last_x = self.x
                self.last_states = states
                self.last_coeffs = coeffs

        return states, inputs

    # Simulate the system dynamis to retrieve the state
    def _simulate_states(self, x0, inputs):
        if self.log:
            logging.debug(
                "calling input_output_response from state\n" + str(x0))
            logging.debug("input =\n" + str(inputs))

        # Simulate the system to get the state
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
        """Compute the optimal input at state x

        Parameters
        ----------
        x : array-like or number, optional
            Initial state for the system.
        return_states : bool, optional
            If True (default), return the values of the state at each time.
        squeeze : bool, optional
            If True and if the system has a single output, return the system
            output as a 1D array rather than a 2D array.  If False, return the
            system output as a 2D array even if the system is SISO.  Default
            value set by config.defaults['control.squeeze_time_response'].
        transpose : bool, optional
            If True, assume that 2D input arrays are transposed from the
            standard format.  Used to convert MATLAB-style inputs to our
            format.

        Returns
        -------
        res : OptimalControlResult
            Bundle object with the results of the optimal control problem.
        res.success: bool
            Boolean flag indicating whether the optimization was successful.
        res.time : array
            Time values of the input.
        res.inputs : array
            Optimal inputs for the system.  If the system is SISO and squeeze
            is not True, the array is 1D (indexed by time).  If the system is
            not SISO or squeeze is False, the array is 2D (indexed by the
            output number and time).
        res.states : array
            Time evolution of the state vector (if return_states=True).

        """
        # Allow 'return_x` as a synonym for 'return_states'
        return_states = ct.config._get_param(
            'optimal', 'return_x', kwargs, return_states, pop=True, last=True)

        # Store the initial state (for use in _constraint_function)
        self.x = x

        # Allow the initial guess to be overriden
        if initial_guess is None:
            initial_guess = self.initial_guess
        else:
            initial_guess = self._process_initial_guess(initial_guess)

        # Call ScipPy optimizer
        res = sp.optimize.minimize(
            self._cost_function, initial_guess,
            constraints=self.constraints, **self.minimize_kwargs)

        # Process and return the results
        return OptimalControlResult(
            self, res, transpose=transpose, return_states=return_states,
            squeeze=squeeze, print_summary=print_summary)

    # Compute the current input to apply from the current state (MPC style)
    def compute_mpc(self, x, squeeze=None):
        """Compute the optimal input at state x

        This function calls the :meth:`compute_trajectory` method and returns
        the input at the first time point.

        Parameters
        ----------
        x: array-like or number, optional
            Initial state for the system.
        squeeze : bool, optional
            If True and if the system has a single output, return the system
            output as a 1D array rather than a 2D array.  If False, return the
            system output as a 2D array even if the system is SISO.  Default
            value set by config.defaults['control.squeeze_time_response'].

        Returns
        -------
        input : array
            Optimal input for the system at the current time.  If the system
            is SISO and squeeze is not True, the array is 1D (indexed by
            time).  If the system is not SISO or squeeze is False, the array
            is 2D (indexed by the output number and time).  Set to `None`
            if the optimization failed.

        """
        res = self.compute_trajectory(x, squeeze=squeeze)
        return res.inputs[:, 0]

    # Create an input/output system implementing an MPC controller
    def create_mpc_iosystem(self):
        """Create an I/O system implementing an MPC controller"""
        # Check to make sure we are in discrete time
        if self.system.dt == 0:
            raise ct.ControlNotImplemented(
                "MPC for continuous time systems not implemented")

        def _update(t, x, u, params={}):
            coeffs = x.reshape((self.system.ninputs, -1))
            if self.basis:
                # Keep the coeffecients unchanged
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

        return ct.NonlinearIOSystem(
            _update, _output, dt=self.system.dt,
            inputs=self.system.nstates, outputs=self.system.ninputs,
            states=self.system.ninputs * \
                (self.timepts.size if self.basis is None else self.basis.N))


# Optimal control result
class OptimalControlResult(sp.optimize.OptimizeResult):
    """Result from solving an optimal control problem.

    This class is a subclass of :class:`scipy.optimize.OptimizeResult` with
    additional attributes associated with solving optimal control problems.

    Attributes
    ----------
    inputs : ndarray
        The optimal inputs associated with the optimal control problem.
    states : ndarray
        If `return_states` was set to true, stores the state trajectory
        associated with the optimal input.
    success : bool
        Whether or not the optimizer exited successful.
    problem : OptimalControlProblem
        Optimal control problem that generated this solution.
    cost : float
        Final cost of the return solution.
    system_simulations, {cost, constraint, eqconst}_evaluations : int
        Number of system simulations and evaluations of the cost function,
        (inequality) constraint function, and equality constraint function
        performed during the optimzation.
    {cost, constraint, eqconst}_process_time : float
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
        response = TimeResponseData(
            ocp.timepts, inputs, states, issiso=ocp.system.issiso(),
            transpose=transpose, return_x=return_states, squeeze=squeeze)

        self.time = response.time
        self.inputs = response.outputs
        self.states = response.states


# Compute the input for a nonlinear, (constrained) optimal control problem
def solve_ocp(
        sys, horizon, X0, cost, trajectory_constraints=None, terminal_cost=None,
        terminal_constraints=[], initial_guess=None, basis=None, squeeze=None,
        transpose=None, return_states=True, print_summary=True, log=False,
        **kwargs):

    """Compute the solution to an optimal control problem

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the optimal input will be computed.

    horizon : 1D array_like
        List of times at which the optimal input should be computed.

    X0: array-like or number, optional
        Initial condition (default = 0).

    cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as `cost(x, u)`.

    trajectory_constraints : list of tuples, optional
        List of constraints that should hold at each point in the time vector.
        Each element of the list should consist of a tuple with first element
        given by :meth:`scipy.optimize.LinearConstraint` or
        :meth:`scipy.optimize.NonlinearConstraint` and the remaining
        elements of the tuple are the arguments that would be passed to those
        functions.  The following tuples are supported:

        * (LinearConstraint, A, lb, ub): The matrix A is multiplied by stacked
          vector of the state and input at each point on the trajectory for
          comparison against the upper and lower bounds.

        * (NonlinearConstraint, fun, lb, ub): a user-specific constraint
          function `fun(x, u)` is called at each point along the trajectory
          and compared against the upper and lower bounds.

        The constraints are applied at each time point along the trajectory.

    terminal_cost : callable, optional
        Function that returns the terminal cost given the current state
        and input.  Called as terminal_cost(x, u).

    terminal_constraints : list of tuples, optional
        List of constraints that should hold at the end of the trajectory.
        Same format as `constraints`.

    initial_guess : 1D or 2D array_like
        Initial inputs to use as a guess for the optimal input.  The inputs
        should either be a 2D vector of shape (ninputs, horizon) or a 1D
        input of shape (ninputs,) that will be broadcast by extension of the
        time axis.

    log : bool, optional
        If `True`, turn on logging messages (using Python logging module).

    print_summary : bool, optional
        If `True` (default), print a short summary of the computation.

    return_states : bool, optional
        If True, return the values of the state at each time (default = True).

    squeeze : bool, optional
        If True and if the system has a single output, return the system
        output as a 1D array rather than a 2D array.  If False, return the
        system output as a 2D array even if the system is SISO.  Default value
        set by config.defaults['control.squeeze_time_response'].

    transpose : bool, optional
        If True, assume that 2D input arrays are transposed from the standard
        format.  Used to convert MATLAB-style inputs to our format.

    Returns
    -------
    res : OptimalControlResult
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
        Time evolution of the state vector (if return_states=True).

    Notes
    -----
    Additional keyword parameters can be used to fine tune the behavior of
    the underlying optimization and integration functions.  See
    :func:`OptimalControlProblem` for more information.

    """
    # Process keyword arguments
    if trajectory_constraints is None:
        # Backwards compatibility
        trajectory_constraints = kwargs.pop('constraints', [])

    # Allow 'return_x` as a synonym for 'return_states'
    return_states = ct.config._get_param(
        'optimal', 'return_x', kwargs, return_states, pop=True)

    # Process (legacy) method keyword
    if kwargs.get('method'):
        method = kwargs.pop('method')
        if method not in optimal_methods:
            if kwargs.get('minimize_method'):
                raise ValueError("'minimize_method' specified more than once")
            warnings.warn(
                "'method' parameter is deprecated; assuming minimize_method",
                DeprecationWarning)
            kwargs['minimize_method'] = method
        else:
            if kwargs.get('trajectory_method'):
                raise ValueError("'trajectory_method' specified more than once")
            warnings.warn(
                "'method' parameter is deprecated; assuming trajectory_method",
                DeprecationWarning)
            kwargs['trajectory_method'] = method

    # Set up the optimal control problem
    ocp = OptimalControlProblem(
        sys, horizon, cost, trajectory_constraints=trajectory_constraints,
        terminal_cost=terminal_cost, terminal_constraints=terminal_constraints,
        initial_guess=initial_guess, basis=basis, log=log, **kwargs)

    # Solve for the optimal input from the current state
    return ocp.compute_trajectory(
        X0, squeeze=squeeze, transpose=transpose, print_summary=print_summary,
        return_states=return_states)


# Create a model predictive controller for an optimal control problem
def create_mpc_iosystem(
        sys, horizon, cost, constraints=[], terminal_cost=None,
        terminal_constraints=[], log=False, **kwargs):
    """Create a model predictive I/O control system

    This function creates an input/output system that implements a model
    predictive control for a system given the time horizon, cost function and
    constraints that define the finite-horizon optimization that should be
    carried out at each state.

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the optimal input will be computed.

    horizon : 1D array_like
        List of times at which the optimal input should be computed.

    cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as cost(x, u).

    constraints : list of tuples, optional
        List of constraints that should hold at each point in the time vector.
        See :func:`~control.optimal.solve_ocp` for more details.

    terminal_cost : callable, optional
        Function that returns the terminal cost given the current state
        and input.  Called as terminal_cost(x, u).

    terminal_constraints : list of tuples, optional
        List of constraints that should hold at the end of the trajectory.
        Same format as `constraints`.

    kwargs : dict, optional
        Additional parameters (passed to :func:`scipy.optimal.minimize`).

    Returns
    -------
    ctrl : InputOutputSystem
        An I/O system taking the current state of the model system and
        returning the current input to be applied that minimizes the cost
        function while satisfying the constraints.

    Notes
    -----
    Additional keyword parameters can be used to fine tune the behavior of
    the underlying optimization and integrations functions.  See
    :func:`OptimalControlProblem` for more information.

    """
    # Set up the optimal control problem
    ocp = OptimalControlProblem(
        sys, horizon, cost, trajectory_constraints=constraints,
        terminal_cost=terminal_cost, terminal_constraints=terminal_constraints,
        log=log, **kwargs)

    # Return an I/O system implementing the model predictive controller
    return ocp.create_mpc_iosystem()


#
# Functions to create cost functions (quadratic cost function)
#
# Since a quadratic function is common as a cost function, we provide a
# function that will take a Q and R matrix and return a callable that
# evaluates to associted quadratic cost.  This is compatible with the way that
# the `_cost_function` evaluates the cost at each point in the trajectory.
#
def quadratic_cost(sys, Q, R, x0=0, u0=0):
    """Create quadratic cost function

    Returns a quadratic cost function that can be used for an optimal control
    problem.  The cost function is of the form

      cost = (x - x0)^T Q (x - x0) + (u - u0)^T R (u - u0)

    Parameters
    ----------
    sys : InputOutputSystem
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


#
# Functions to create constraints: either polytopes (A x <= b) or ranges
# (lb # <= x <= ub).
#
# As in the cost function evaluation, the main "trick" in creating a constrain
# on the state or input is to properly evaluate the constraint on the stacked
# state and input vector at the current time point.  The constraint itself
# will be called at each point along the trajectory (or the endpoint) via the
# constrain_function() method.
#
# Note that these functions to not actually evaluate the constraint, they
# simply return the information required to do so.  We use the SciPy
# optimization methods LinearConstraint and NonlinearConstraint as "types" to
# keep things consistent with the terminology in scipy.optimize.
#
def state_poly_constraint(sys, A, b):
    """Create state constraint from polytope

    Creates a linear constraint on the system state of the form A x <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix
    b : 1D array
        Upper bound for the constraint

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
    """Create state constraint from polytope

    Creates a linear constraint on the system state that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of `inf` and `-inf` to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : InputOutputSystem
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
    """Create input constraint from polytope

    Creates a linear constraint on the system input of the form A u <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix
    b : 1D array
        Upper bound for the constraint

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
    """Create input constraint from polytope

    Creates a linear constraint on the system input that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of `inf` and `-inf` to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : InputOutputSystem
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
# Unlike the state and input constraints, for the output constraint we need to
# do a function evaluation before applying the constraints.
#
# TODO: for the special case of an LTI system, we can avoid the extra function
# call by multiplying the state by the C matrix for the system and then
# imposing a linear constraint:
#
#     np.hstack(
#         [A @ sys.C, np.zeros((A.shape[0], sys.ninputs))])
#
def output_poly_constraint(sys, A, b):
    """Create output constraint from polytope

    Creates a linear constraint on the system output of the form A y <= b that
    can be used as an optimal control constraint (trajectory or terminal).

    Parameters
    ----------
    sys : InputOutputSystem
        I/O system for which the constraint is being defined.
    A : 2D array
        Constraint matrix
    b : 1D array
        Upper bound for the constraint

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
    """Create output constraint from range

    Creates a linear constraint on the system output that bounds the range of
    the individual states to be between `lb` and `ub`.  The upper and lower
    bounds can be set of `inf` and `-inf` to indicate there is no constraint
    or to the same value to describe an equality constraint.

    Parameters
    ----------
    sys : InputOutputSystem
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
    if isinstance(
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
            if not ctype in [opt.LinearConstraint, opt.NonlinearConstraint]:
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
