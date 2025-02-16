# flatsys.py - trajectory generation for differentially flat systems
# RMM, 10 Nov 2012

"""Trajectory generation for differentially flat systems.

"""

import itertools
import warnings

import numpy as np
import scipy as sp
import scipy.optimize

from ..config import _process_kwargs, _process_param
from ..exception import ControlArgument
from ..nlsys import NonlinearIOSystem
from ..optimal import _optimal_aliases
from ..timeresp import _check_convert_array
from .poly import PolyFamily
from .systraj import SystemTrajectory


# Flat system class (for use as a base class)
class FlatSystem(NonlinearIOSystem):
    """Base class for representing a differentially flat system.

    The FlatSystem class is used as a base class to describe differentially
    flat systems for trajectory generation.  The output of the system does
    not need to be the differentially flat output.  Flat systems are
    usually created with the `flatsys` factory function.

    Parameters
    ----------
    forward : callable
        A function to compute the flat flag given the states and input.
    reverse : callable
        A function to compute the states and input given the flat flag.
    dt : None, True or float, optional
        System timebase.

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    input_labels, output_labels, state_labels : list of str
        Names for the input, output, and state variables.
    name : string, optional
        System name.

    See Also
    --------
    flatsys

    Notes
    -----
    The class must implement two functions:

    ``zflag = flatsys.forward(x, u, params)``

        This function computes the flag (derivatives) of the flat output.
        The inputs to this function are the state `x` and inputs `u` (both
        1D arrays).  The output should be a 2D array with the first
        dimension equal to the number of system inputs and the second
        dimension of the length required to represent the full system
        dynamics (typically the number of states)

    ``x, u = flatsys.reverse(zflag, params)``

        This function system state and inputs give the the flag (derivatives)
        of the flat output.  The input to this function is an 2D array whose
        first dimension is equal to the number of system inputs and whose
        second dimension is of length required to represent the full system
        dynamics (typically the number of states).  The output is the state
        `x` and inputs `u` (both 1D arrays).

    A flat system is also an input/output system supporting simulation,
    composition, and linearization.  In the current implementation, the
    update function must be given explicitly, but the output function
    defaults to the flat outputs.  If the output method is given, it is
    used in place of the flat outputs.

    """
    def __init__(self,
                 forward, reverse,              # flat system
                 updfcn=None, outfcn=None,      # nonlinear I/O system
                 **kwargs):                     # I/O system
        """Create a differentially flat I/O system.

        The `FlatSystem` constructor is used to create an input/output system
        object that also represents a differentially flat system.

        """

        # TODO: specify default update and output functions
        if updfcn is None: updfcn = self._flat_updfcn
        if outfcn is None: outfcn = self._flat_outfcn

        # Initialize as an input/output system
        NonlinearIOSystem.__init__(self, updfcn, outfcn, **kwargs)

        # Save the functions to compute forward and reverse conversions
        if forward is not None: self.forward = forward
        if reverse is not None: self.reverse = reverse

        # Save the length of the flat flag
        # TODO: missing

    def __str__(self):
        return f"{NonlinearIOSystem.__str__(self)}\n\n" \
            + f"Forward: {self.forward}\n" \
            + f"Reverse: {self.reverse}"

    def forward(self, x, u, params=None):
        """Compute the flat flag given the states and input.

        Given the states and inputs for a system, compute the flat
        outputs and their derivatives (the flat "flag") for the
        system.

        Parameters
        ----------
        x : list or array
            The state of the  system.
        u : list or array
            The input to the  system.
        params : dict, optional
            Parameter values for the system.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.

        Returns
        -------
        zflag : list of 1D arrays
            For each flat output :math:`z_i`, `zflag[i]` should be an
            ndarray of length :math:`q_i` that contains the flat
            output and its first :math:`q_i` derivatives.

        """
        raise NotImplementedError("internal error; forward method not defined")

    def reverse(self, zflag, params=None):
        """Compute the states and input given the flat flag.

        Parameters
        ----------
        zflag : list of arrays
            For each flat output :math:`z_i`, zflag[i] should be an
            ndarray of length :math:`q_i` that contains the flat
            output and its first :math:`q_i` derivatives.
        params : dict, optional
            Parameter values for the system.  Passed to the evaluation
            functions for the system as default values, overriding internal
            defaults.

        Returns
        -------
        x : 1D array
            The state of the system corresponding to the flat flag.
        u : 1D array
            The input to the system corresponding to the flat flag.

        """
        raise NotImplementedError("internal error; reverse method not defined")

    def _flat_updfcn(self, t, x, u, params=None):
        # TODO: implement state space update using flat coordinates
        raise NotImplementedError("update function for flat system not given")

    def _flat_outfcn(self, t, x, u, params=None):
        # Return the flat output
        zflag = self.forward(x, u, params)
        return np.array([zflag[i][0] for i in range(len(zflag))])


def flatsys(*args, updfcn=None, outfcn=None, **kwargs):
    """flatsys(forward, reverse[, updfcn, outfcn]) \
    flatsys(linsys)

    Create a differentially flat I/O system.

    The flatsys() function is used to create an input/output system object
    that also represents a differentially flat system.  It can be used in a
    variety of forms:

    ``fs.flatsys(forward, reverse)``

        Create a flat system with mappings to/from flat flag.

    ``fs.flatsys(forward, reverse, updfcn[, outfcn])``

        Create a flat system that is also a nonlinear I/O system.

    ``fs.flatsys(linsys)``

        Create a flat system from a linear (`StateSpace`) system.

    Parameters
    ----------
    forward : callable
        A function to compute the flat flag given the states and input.

    reverse : callable
        A function to compute the states and input given the flat flag.

    updfcn : callable, optional
        Function returning the state update function

            ``updfcn(t, x, u[, params]) -> array``

        where `x` is a 1-D array with shape (nstates,), `u` is a 1-D array
        with shape (ninputs,), `t` is a float representing the current
        time, and `params` is an optional dict containing the values of
        parameters used by the function.  If not specified, the state
        space update will be computed using the flat system coordinates.

    outfcn : callable, optional
        Function returning the output at the given state

            ``outfcn(t, x, u[, params]) -> array``

        where the arguments are the same as for `updfcn`.  If not
        specified, the output will be the flat outputs.

    inputs : int, list of str, or None
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.
        If an integer count is specified, the names of the signal will be
        of the form 's[i]' (where 's' is one of 'u', 'y', or 'x').  If
        this parameter is not given or given as None, the relevant
        quantity will be determined when possible based on other
        information provided to functions using the system.

    outputs : int, list of str, or None
        Description of the system outputs.  Same format as `inputs`.

    states : int, list of str, or None
        Description of the system states.  Same format as `inputs`.

    dt : None, True or float, optional
        System timebase.  None (default) indicates continuous time, True
        indicates discrete time with undefined sampling time, positive
        number is discrete time with specified sampling time.

    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation
        functions for the system as default values, overriding internal
        defaults.

    name : string, optional
        System name (used for specifying signals).

    Returns
    -------
    sys : `FlatSystem`
        Flat system.

    Other Parameters
    ----------------
    input_prefix, output_prefix, state_prefix : string, optional
        Set the prefix for input, output, and state signals.  Defaults =
        'u', 'y', 'x'.

    """
    from ..statesp import StateSpace
    from .linflat import LinearFlatSystem

    if len(args) == 1 and isinstance(args[0], StateSpace):
        # We were passed a linear system, so call linflat
        if updfcn is not None or outfcn is not None:
            warnings.warn(
                "update and output functions ignored for linear system")
        return LinearFlatSystem(args[0], **kwargs)

    elif len(args) == 2:
        forward, reverse = args

    elif len(args) == 3:
        if updfcn is not None:
            warnings.warn(
                "update and output functions specified twice; using"
                " positional arguments")
        forward, reverse, updfcn = args

    elif len(args) == 4:
        if updfcn is not None or outfcn is not None:
            warnings.warn(
                "update and output functions specified twice; using"
                " positional arguments")
        forward, reverse, updfcn, outfcn = args

    else:
        raise TypeError("incorrect number or type of arguments")

    # Create the flat system
    return FlatSystem(
        forward, reverse, updfcn=updfcn, outfcn=outfcn, **kwargs)


# Utility function to compute flag matrix given a basis
def _basis_flag_matrix(sys, basis, flag, t):
    """Compute the matrix of basis functions and their derivatives

    This function computes the matrix `M` that is used to solve for the
    coefficients of the basis functions given the state and input.  Each
    column of the matrix corresponds to a basis function and each row is a
    derivative, with the derivatives (flag) for each output stacked on top
    of each other.
l
    """
    flagshape = [len(f) for f in flag]
    M = np.zeros((sum(flagshape),
                  sum([basis.var_ncoefs(i) for i in range(sys.ninputs)])))
    flag_off = 0
    coef_off = 0
    for i, flag_len in enumerate(flagshape):
        coef_len = basis.var_ncoefs(i)
        for j, k in itertools.product(range(coef_len), range(flag_len)):
            M[flag_off + k, coef_off + j] = basis.eval_deriv(j, k, t, var=i)
        flag_off += flag_len
        coef_off += coef_len
    return M


# Solve a point to point trajectory generation problem for a flat system
def point_to_point(
        sys, timepts, initial_state=0, initial_input=0, final_state=0,
        final_input=0, initial_time=0, integral_cost=None, basis=None,
        trajectory_constraints=None, initial_guess=None, params=None, **kwargs):
    """Compute trajectory between an initial and final conditions.

    Compute a feasible trajectory for a differentially flat system between an
    initial condition and a final condition.

    Parameters
    ----------
    sys : `FlatSystem` object
        Description of the differentially flat system.  This object must
        define a function `~FlatSystem.forward` that takes the system state
        and produces the flag of flat outputs and a function
        `~FlatSystem.reverse` that takes the flag of the flat output and
        produces the state and input.
    timepts : float or 1D array_like
        The list of points for evaluating cost and constraints, as well as
        the time horizon.  If given as a float, indicates the final time for
        the trajectory (corresponding to xf)
    initial_state (or x0) : 1D array_like
        Initial state for the system.  Defaults to zero.
    initial_input (or u0) : 1D array_like
        Initial input for the system.  Defaults to zero.
    final_state (or xf) : 1D array_like
        Final state for the system.  Defaults to zero.
    final_input (or uf) : 1D array_like
        Final input for the system.  Defaults to zero.
    initial_time (or T0) : float, optional
        The initial time for the trajectory (corresponding to x0).  If not
        specified, its value is taken to be zero.
    basis : `BasisFamily` object, optional
        The basis functions to use for generating the trajectory.  If not
        specified, the `PolyFamily` basis family
        will be used, with the minimal number of elements required to find a
        feasible trajectory (twice the number of system states)
    integral_cost (or cost) : callable
        Function that returns the integral cost given the current state
        and input.  Called as ``integral_cost(x, u)``.
    trajectory_constraints (or constraints) : list of tuples, optional
        List of constraints that should hold at each point in the time
        vector.  Each element of the list should consist of a tuple with
        first element given by `scipy.optimize.LinearConstraint` or
        `scipy.optimize.NonlinearConstraint` and the remaining elements of
        the tuple are the arguments that would be passed to those
        functions.  The following tuples are supported:

        * (LinearConstraint, A, lb, ub): The matrix A is multiplied by
          stacked vector of the state and input at each point on the
          trajectory for comparison against the upper and lower bounds.

        * (NonlinearConstraint, fun, lb, ub): a user-specific constraint
          function ``fun(x, u)`` is called at each point along the
          trajectory and compared against the upper and lower bounds.

        The constraints are applied at each time point along the trajectory.
    initial_guess : 2D array_like, optional
        Initial guess for the trajectory coefficients (not implemented).
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation
        functions for the system as default values, overriding internal
        defaults.

    Returns
    -------
    traj : `SystemTrajectory` object
        The system trajectory is returned as an object that implements the
        `~SystemTrajectory.eval` function, we can be used to
        compute the value of the state and input and a given time t.

    Other Parameters
    ----------------
    minimize_method : str, optional
        Set the method used by `scipy.optimize.minimize`.
    minimize_options : str, optional
        Set the options keyword used by `scipy.optimize.minimize`.
    minimize_kwargs : str, optional
        Pass additional keywords to `scipy.optimize.minimize`.

    Notes
    -----
    Additional keyword parameters can be used to fine tune the behavior of
    the underlying optimization function.  See `minimize_*` keywords in
    `OptimalControlProblem` for more information.

    """
    # Process parameter and keyword arguments
    _process_kwargs(kwargs, _optimal_aliases)
    x0 = _process_param(
        'initial_state', initial_state, kwargs, _optimal_aliases, sigval=0)
    u0 = _process_param(
        'initial_input', initial_input, kwargs, _optimal_aliases, sigval=0)
    xf = _process_param(
        'final_state', final_state, kwargs, _optimal_aliases, sigval=0)
    uf = _process_param(
        'final_input', final_input, kwargs, _optimal_aliases, sigval=0)
    T0 = _process_param(
        'initial_time', initial_time, kwargs, _optimal_aliases, sigval=0)
    cost = _process_param(
        'integral_cost', integral_cost, kwargs, _optimal_aliases)
    trajectory_constraints = _process_param(
        'trajectory_constraints', trajectory_constraints, kwargs,
        _optimal_aliases)

    #
    # Make sure the problem is one that we can handle
    #
    x0 = _check_convert_array(x0, [(sys.nstates,), (sys.nstates, 1)],
                              'Initial state: ', squeeze=True)
    u0 = _check_convert_array(u0, [(sys.ninputs,), (sys.ninputs, 1)],
                              'Initial input: ', squeeze=True)
    xf = _check_convert_array(xf, [(sys.nstates,), (sys.nstates, 1)],
                              'Final state: ', squeeze=True)
    uf = _check_convert_array(uf, [(sys.ninputs,), (sys.ninputs, 1)],
                              'Final input: ', squeeze=True)

    # Process final time
    timepts = np.atleast_1d(timepts)
    Tf = timepts[-1]
    T0 = timepts[0] if len(timepts) > 1 else T0

    minimize_kwargs = {}
    minimize_kwargs['method'] = kwargs.pop('minimize_method', None)
    minimize_kwargs['options'] = kwargs.pop('minimize_options', {})
    minimize_kwargs.update(kwargs.pop('minimize_kwargs', {}))

    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    #
    # Determine the basis function set to use and make sure it is big enough
    #

    # If no basis set was specified, use a polynomial basis (poor choice...)
    if basis is None:
        basis = PolyFamily(2 * (sys.nstates + sys.ninputs))

    # If a multivariable basis was given, make sure the size is correct
    if basis.nvars is not None and basis.nvars != sys.ninputs:
        raise ValueError("size of basis does not match flat system size")

    # Make sure we have enough basis functions to solve the problem
    ncoefs = sum([basis.var_ncoefs(i) for i in range(sys.ninputs)])
    if ncoefs < 2 * (sys.nstates + sys.ninputs):
        raise ValueError("basis set is too small")
    elif (cost is not None or trajectory_constraints is not None) and \
         ncoefs == 2 * (sys.nstates + sys.ninputs):
        warnings.warn("minimal basis specified; optimization not possible")
        cost = None
        trajectory_constraints = None

    # Figure out the parameters to use, if any
    params = sys.params if params is None else params

    #
    # Map the initial and final conditions to flat output conditions
    #
    # We need to compute the output "flag": [z(t), z'(t), z''(t), ...]
    # and then evaluate this at the initial and final condition.
    #

    zflag_T0 = sys.forward(x0, u0, params)
    zflag_Tf = sys.forward(xf, uf, params)

    #
    # Compute the matrix constraints for initial and final conditions
    #
    # This computation depends on the basis function we are using.  It
    # essentially amounts to evaluating the basis functions and their
    # derivatives at the initial and final conditions.

    # Compute the flags for the initial and final states
    M_T0 = _basis_flag_matrix(sys, basis, zflag_T0, T0)
    M_Tf = _basis_flag_matrix(sys, basis, zflag_Tf, Tf)

    # Stack the initial and final matrix/flag for the point to point problem
    M = np.vstack([M_T0, M_Tf])
    Z = np.hstack([np.hstack(zflag_T0), np.hstack(zflag_Tf)])

    #
    # Solve for the coefficients of the flat outputs
    #
    # At this point, we need to solve the equation M alpha = zflag, where M
    # is the matrix constraints for initial and final conditions and zflag =
    # [zflag_T0; zflag_tf].
    #
    # If there are no constraints, then we just need to solve a linear
    # system of equations => use least squares.  Otherwise, we have a
    # nonlinear optimal control problem with equality constraints => use
    # scipy.optimize.minimize().
    #

    # Start by solving the least squares problem
    # TODO: add warning if rank is too small
    alpha, residuals, rank, s = np.linalg.lstsq(M, Z, rcond=None)
    if rank < Z.size:
        warnings.warn("basis too small; solution may not exist")

    if cost is not None or trajectory_constraints is not None:
        # Make sure that we have enough time points to evaluate
        if timepts.size < 3:
            raise ControlArgument(
                "There must be at least three time points if trajectory"
                " cost or constraints are specified")

        # Search over the null space to minimize cost/satisfy constraints
        N = sp.linalg.null_space(M)

        # Precompute the collocation matrix the defines the flag at timepts
        Mt_list = []
        for t in timepts[1:-1]:
            Mt_list.append(_basis_flag_matrix(sys, basis, zflag_T0, t))

        # Define a function to evaluate the cost along a trajectory
        def traj_cost(null_coeffs):
            # Add this to the existing solution
            coeffs = alpha + N @ null_coeffs

            # Evaluate the costs at the listed time points
            costval = 0
            for i, t in enumerate(timepts[1:-1]):
                M_t = Mt_list[i]

                # Compute flag at this time point
                zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)

                # Find states and inputs at the time points
                x, u = sys.reverse(zflag, params)

                # Evaluate the cost at this time point
                costval += cost(x, u) * (timepts[i+1] - timepts[i])
            return costval

        # If no cost given, override with magnitude of the coefficients
        if cost is None:
            traj_cost = lambda coeffs: coeffs @ coeffs

        # Process the constraints we were given
        traj_constraints = trajectory_constraints
        if traj_constraints is None:
            traj_constraints = []
        elif isinstance(traj_constraints, tuple):
            # TODO: Check to make sure this is really a constraint
            traj_constraints = [traj_constraints]
        elif not isinstance(traj_constraints, list):
            raise TypeError("trajectory constraints must be a list")

        # Process constraints
        minimize_constraints = []
        if len(traj_constraints) > 0:
            # Set up a nonlinear function to evaluate the constraints
            def traj_const(null_coeffs):
                # Add this to the existing solution
                coeffs = alpha + N @ null_coeffs

                # Evaluate the constraints at the listed time points
                values = []
                for i, t in enumerate(timepts[1:-1]):
                    # Calculate the states and inputs for the flat output
                    M_t = Mt_list[i]

                    # Compute flag at this time point
                    zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)

                    # Find states and inputs at the time points
                    states, inputs = sys.reverse(zflag, params)

                    # Evaluate the constraint function along the trajectory
                    for type, fun, lb, ub in traj_constraints:
                        if type == sp.optimize.LinearConstraint:
                            # `fun` is A matrix associated with polytope...
                            values.append(fun @ np.hstack([states, inputs]))
                        elif type == sp.optimize.NonlinearConstraint:
                            values.append(fun(states, inputs))
                        else:
                            raise TypeError(
                                "unknown constraint type %s" % type)
                return np.array(values).flatten()

            # Store upper and lower bounds
            const_lb, const_ub = [], []
            for t in timepts[1:-1]:
                for type, fun, lb, ub in traj_constraints:
                    const_lb.append(lb)
                    const_ub.append(ub)
            const_lb = np.array(const_lb).flatten()
            const_ub = np.array(const_ub).flatten()

            # Store the constraint as a nonlinear constraint
            minimize_constraints = [sp.optimize.NonlinearConstraint(
                traj_const, const_lb, const_ub)]

        # Process the initial condition
        if initial_guess is None:
            initial_guess = np.zeros(M.shape[1] - M.shape[0])
        else:
            raise NotImplementedError("Initial guess not yet implemented.")

        # Find the optimal solution
        res = sp.optimize.minimize(
            traj_cost, initial_guess, constraints=minimize_constraints,
            **minimize_kwargs)
        alpha += N @ res.x

        # See if we got an answer
        if not res.success:
            warnings.warn(
                "unable to solve optimal control problem\n"
                f"scipy.optimize.minimize: '{res.message}'", UserWarning)

    #
    # Transform the trajectory from flat outputs to states and inputs
    #

    # Create a trajectory object to store the result
    systraj = SystemTrajectory(sys, basis, params=params)
    if cost is not None or trajectory_constraints is not None:
        # Store the result of the optimization
        systraj.cost = res.fun
        systraj.success = res.success
        systraj.message = res.message

    # Store the flag lengths and coefficients
    # TODO: make this more pythonic
    coef_off = 0
    for i in range(sys.ninputs):
        # Grab the coefficients corresponding to this flat output
        coef_len = basis.var_ncoefs(i)
        systraj.coeffs.append(alpha[coef_off:coef_off + coef_len])
        coef_off += coef_len

        # Keep track of the length of the flat flag for this output
        systraj.flaglen.append(len(zflag_T0[i]))

    # Return a function that computes inputs and states as a function of time
    return systraj


# Solve a point to point trajectory generation problem for a flat system
def solve_flat_optimal(
        sys, timepts, initial_state=0, initial_input=0, integral_cost=None,
        basis=None, terminal_cost=None, trajectory_constraints=None,
        initial_guess=None, params=None, **kwargs):
    """Compute trajectory between an initial and final conditions.

    Compute an optimal trajectory for a differentially flat system starting
    from an initial state and input value.

    Parameters
    ----------
    sys : `FlatSystem` object
        Description of the differentially flat system.  This object must
        define a function `~FlatSystem.forward` that takes the system state
        and produces the flag of flat outputs and a function
        `~FlatSystem.reverse` that takes the flag of the flat output and
        produces the state and input.
    timepts : float or 1D array_like
        The list of points for evaluating cost and constraints, as well as
        the time horizon.  If given as a float, indicates the final time for
        the trajectory (corresponding to xf)
    initial_state (or x0), input_input (or u0) : 1D arrays
        Define the initial conditions for the system (default = 0).
    initial_input (or u0) : 1D array_like
        Initial input for the system.  Defaults to zero.
    basis : `BasisFamily` object, optional
        The basis functions to use for generating the trajectory.  If not
        specified, the `PolyFamily` basis family
        will be used, with the minimal number of elements required to find a
        feasible trajectory (twice the number of system states)
    integral_cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as ``cost(x, u)``.
    terminal_cost : callable
        Function that returns the terminal cost given the state and input.
        Called as ``cost(x, u)``.
    trajectory_constraints : list of tuples, optional
        List of constraints that should hold at each point in the time
        vector.  Each element of the list should consist of a tuple with
        first element given by `scipy.optimize.LinearConstraint` or
        `scipy.optimize.NonlinearConstraint` and the remaining elements of
        the tuple are the arguments that would be passed to those
        functions.  The following tuples are supported:

        * (LinearConstraint, A, lb, ub): The matrix A is multiplied by
          stacked vector of the state and input at each point on the
          trajectory for comparison against the upper and lower bounds.

        * (NonlinearConstraint, fun, lb, ub): a user-specific constraint
          function ``fun(x, u)`` is called at each point along the
          trajectory and compared against the upper and lower bounds.

        The constraints are applied at each time point along the trajectory.
    initial_guess : 2D array_like, optional
        Initial guess for the optimal trajectory of the flat outputs.
    params : dict, optional
        Parameter values for the system.  Passed to the evaluation
        functions for the system as default values, overriding internal
        defaults.

    Returns
    -------
    traj : `SystemTrajectory`
        The system trajectory is returned as an object that implements the
        `SystemTrajectory.eval` function, we can be used to
        compute the value of the state and input and a given time `t`.

    Other Parameters
    ----------------
    minimize_method : str, optional
        Set the method used by `scipy.optimize.minimize`.

    minimize_options : str, optional
        Set the options keyword used by `scipy.optimize.minimize`.

    minimize_kwargs : str, optional
        Pass additional keywords to `scipy.optimize.minimize`.

    Notes
    -----
    Additional keyword parameters can be used to fine tune the behavior of
    the underlying optimization function.  See `minimize_*` keywords in
    `control.optimal.OptimalControlProblem` for more information.

    The return data structure includes the following additional attributes:

        * `success` : bool indicating whether the optimization succeeded
        * `cost` : computed cost of the returned trajectory
        * `message` : message returned by optimization if success if False

    A common failure in solving optimal control problem is that the default
    initial guess violates the constraints and the optimizer can't find a
    feasible solution.  Using the `initial_guess` parameter can often be
    used to overcome these errors.

    """
    # Process parameter and keyword arguments
    _process_kwargs(kwargs, _optimal_aliases)
    x0 = _process_param(
        'initial_state', initial_state, kwargs, _optimal_aliases, sigval=0)
    u0 = _process_param(
        'initial_input', initial_input, kwargs, _optimal_aliases, sigval=0)
    trajectory_cost = _process_param(
        'integral_cost', integral_cost, kwargs, _optimal_aliases)
    trajectory_constraints = _process_param(
        'trajectory_constraints', trajectory_constraints, kwargs,
        _optimal_aliases)

    #
    # Make sure the problem is one that we can handle
    #
    x0 = _check_convert_array(x0, [(sys.nstates,), (sys.nstates, 1)],
                              'Initial state: ', squeeze=True)
    u0 = _check_convert_array(u0, [(sys.ninputs,), (sys.ninputs, 1)],
                              'Initial input: ', squeeze=True)

    # Process final time
    timepts = np.atleast_1d(timepts)
    T0 = timepts[0] if len(timepts) > 1 else 0

    minimize_kwargs = {}
    minimize_kwargs['method'] = kwargs.pop('minimize_method', None)
    minimize_kwargs['options'] = kwargs.pop('minimize_options', {})
    minimize_kwargs.update(kwargs.pop('minimize_kwargs', {}))

    if trajectory_cost is None and terminal_cost is None:
        raise TypeError("need trajectory and/or terminal cost required")

    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    #
    # Determine the basis function set to use and make sure it is big enough
    #

    # If no basis set was specified, use a polynomial basis (poor choice...)
    if basis is None:
        basis = PolyFamily(2 * (sys.nstates + sys.ninputs))

    # If a multivariable basis was given, make sure the size is correct
    if basis.nvars is not None and basis.nvars != sys.ninputs:
        raise ValueError("size of basis does not match flat system size")

    # Make sure we have enough basis functions to solve the problem
    ncoefs = sum([basis.var_ncoefs(i) for i in range(sys.ninputs)])
    if ncoefs <= sys.nstates + sys.ninputs:
        raise ValueError("basis set is too small")

    # Figure out the parameters to use, if any
    params = sys.params if params is None else params

    #
    # Map the initial and conditions to flat output conditions
    #
    # We need to compute the output "flag": [z(t), z'(t), z''(t), ...]
    # and then evaluate this at the initial and final condition.
    #

    zflag_T0 = sys.forward(x0, u0, params)
    Z_T0 = np.hstack(zflag_T0)

    #
    # Compute the matrix constraints for initial conditions
    #
    # This computation depends on the basis function we are using.  It
    # essentially amounts to evaluating the basis functions and their
    # derivatives at the initial conditions.

    # Compute the flag for the initial state
    M_T0 = _basis_flag_matrix(sys, basis, zflag_T0, T0)

    #
    # Solve for the coefficients of the flat outputs
    #
    # At this point, we need to solve the equation M_T0 alpha = zflag_T0.
    #
    # If there are no additional constraints, then we just need to solve a
    # linear system of equations => use least squares.  Otherwise, we have a
    # nonlinear optimal control problem with equality constraints => use
    # scipy.optimize.minimize().
    #

    # Start by solving the least squares problem
    alpha, residuals, rank, s = np.linalg.lstsq(M_T0, Z_T0, rcond=None)
    if rank < Z_T0.size:
        warnings.warn("basis too small; solution may not exist")

    # Precompute the collocation matrix the defines the flag at timepts
    # TODO: only compute if we have trajectory cost/constraints
    Mt_list = []
    for t in timepts:
        Mt_list.append(_basis_flag_matrix(sys, basis, zflag_T0, t))

    # Search over the null space to minimize cost/satisfy constraints
    N = sp.linalg.null_space(M_T0)

    # Define a function to evaluate the cost along a trajectory
    def traj_cost(null_coeffs):
        # Add this to the existing solution
        coeffs = alpha + N @ null_coeffs
        costval = 0

        # Evaluate the trajectory costs at the listed time points
        if trajectory_cost is not None:
            for i, t in enumerate(timepts[0:-1]):
                M_t = Mt_list[i]

                # Compute flag at this time point
                zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)

                # Find states and inputs at the time points
                x, u = sys.reverse(zflag, params)

                # Evaluate the cost at this time point
                # TODO: make use of time interval
                costval += trajectory_cost(x, u) * (timepts[i+1] - timepts[i])

        # Evaluate the terminal_cost
        if terminal_cost is not None:
            M_t = Mt_list[-1]
            zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)
            x, u = sys.reverse(zflag, params)
            costval += terminal_cost(x, u)

        return costval

    # Process the constraints we were given
    traj_constraints = trajectory_constraints
    if traj_constraints is None:
        traj_constraints = []
    elif isinstance(traj_constraints, tuple):
        # TODO: Check to make sure this is really a constraint
        traj_constraints = [traj_constraints]
    elif not isinstance(traj_constraints, list):
        raise TypeError("trajectory constraints must be a list")

    # Process constraints
    minimize_constraints = []
    if len(traj_constraints) > 0:
        # Set up a nonlinear function to evaluate the constraints
        def traj_const(null_coeffs):
            # Add this to the existing solution
            coeffs = alpha + N @ null_coeffs

            # Evaluate the constraints at the listed time points
            values = []
            for i, t in enumerate(timepts):
                # Calculate the states and inputs for the flat output
                M_t = Mt_list[i]

                # Compute flag at this time point
                zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)

                # Find states and inputs at the time points
                states, inputs = sys.reverse(zflag, params)

                # Evaluate the constraint function along the trajectory
                for type, fun, lb, ub in traj_constraints:
                    if type == sp.optimize.LinearConstraint:
                        # `fun` is A matrix associated with polytope...
                        values.append(fun @ np.hstack([states, inputs]))
                    elif type == sp.optimize.NonlinearConstraint:
                        values.append(fun(states, inputs))
                    else:
                        raise TypeError(
                            "unknown constraint type %s" % type)
            return np.array(values).flatten()

        # Store upper and lower bounds
        const_lb, const_ub = [], []
        for t in timepts:
            for type, fun, lb, ub in traj_constraints:
                const_lb.append(lb)
                const_ub.append(ub)
        const_lb = np.array(const_lb).flatten()
        const_ub = np.array(const_ub).flatten()

        # Store the constraint as a nonlinear constraint
        minimize_constraints = [sp.optimize.NonlinearConstraint(
            traj_const, const_lb, const_ub)]

    # Add initial and terminal constraints
    # minimize_constraints += [sp.optimize.LinearConstraint(M, Z, Z)]

    # Process the initial guess
    if initial_guess is None:
        initial_coefs = np.ones(M_T0.shape[1] - M_T0.shape[0])
    else:
        # Compute the map from coefficients to flat outputs
        initial_coefs = []
        for i in range(sys.ninputs):
            M_z = np.array([
                basis.eval_deriv(j, 0, timepts, var=i)
                for j in range(basis.var_ncoefs(i))]).transpose()

            # Compute the parameters that give the best least squares fit
            coefs, _, _, _ = np.linalg.lstsq(
                M_z, initial_guess[i], rcond=None)
            initial_coefs.append(coefs)
        initial_coefs = np.hstack(initial_coefs)

        # Project the parameters onto the independent variables
        initial_coefs, _, _, _ = np.linalg.lstsq(N, initial_coefs, rcond=None)

    # Find the optimal solution
    res = sp.optimize.minimize(
        traj_cost, initial_coefs, constraints=minimize_constraints,
        **minimize_kwargs)
    alpha += N @ res.x

    # See if we got an answer
    if not res.success:
        warnings.warn(
            "unable to solve optimal control problem\n"
            f"scipy.optimize.minimize: '{res.message}'", UserWarning)

    #
    # Transform the trajectory from flat outputs to states and inputs
    #

    # Create a trajectory object to store the result
    systraj = SystemTrajectory(sys, basis, params=params)
    systraj.cost = res.fun
    systraj.success = res.success
    systraj.message = res.message

    # Store the flag lengths and coefficients
    # TODO: make this more pythonic
    coef_off = 0
    for i in range(sys.ninputs):
        # Grab the coefficients corresponding to this flat output
        coef_len = basis.var_ncoefs(i)
        systraj.coeffs.append(alpha[coef_off:coef_off + coef_len])
        coef_off += coef_len

        # Keep track of the length of the flat flag for this output
        systraj.flaglen.append(len(zflag_T0[i]))

    # Return a function that computes inputs and states as a function of time
    return systraj


# Convenience aliases
solve_flat_ocp = solve_flat_optimal
