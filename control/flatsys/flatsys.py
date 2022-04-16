# flatsys.py - trajectory generation for differentially flat systems
# RMM, 10 Nov 2012
#
# This file contains routines for computing trajectories for differentially
# flat nonlinear systems.  It is (very) loosely based on the NTG software
# package developed by Mark Milam and Kudah Mushambi, but rewritten from
# scratch in python.
#
# Copyright (c) 2012 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import itertools
import numpy as np
import scipy as sp
import scipy.optimize
import warnings
from .poly import PolyFamily
from .systraj import SystemTrajectory
from ..iosys import NonlinearIOSystem
from ..timeresp import _check_convert_array


# Flat system class (for use as a base class)
class FlatSystem(NonlinearIOSystem):
    """Base class for representing a differentially flat system.

    The FlatSystem class is used as a base class to describe differentially
    flat systems for trajectory generation.  The output of the system does not
    need to be the differentially flat output.

    Parameters
    ----------
    forward : callable
        A function to compute the flat flag given the states and input.
    reverse : callable
        A function to compute the states and input given the flat flag.
    updfcn : callable, optional
        Function returning the state update function

            `updfcn(t, x, u[, param]) -> array`

        where `x` is a 1-D array with shape (nstates,), `u` is a 1-D array
        with shape (ninputs,), `t` is a float representing the currrent
        time, and `param` is an optional dict containing the values of
        parameters used by the function.  If not specified, the state
        space update will be computed using the flat system coordinates.
    outfcn : callable
        Function returning the output at the given state

            `outfcn(t, x, u[, param]) -> array`

        where the arguments are the same as for `upfcn`.  If not
        specified, the output will be the flat outputs.
    inputs : int, list of str, or None
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.
        If an integer count is specified, the names of the signal will be
        of the form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If
        this parameter is not given or given as `None`, the relevant
        quantity will be determined when possible based on other
        information provided to functions using the system.
    outputs : int, list of str, or None
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None
        Description of the system states.  Same format as `inputs`.
    dt : None, True or float, optional
        System timebase.  None (default) indicates continuous
        time, True indicates discrete time with undefined sampling
        time, positive number is discrete time with specified
        sampling time.
    params : dict, optional
        Parameter values for the systems.  Passed to the evaluation
        functions for the system as default values, overriding internal
        defaults.
    name : string, optional
        System name (used for specifying signals)

    Notes
    -----
    The class must implement two functions:

    zflag = flatsys.foward(x, u, params)
        This function computes the flag (derivatives) of the flat output.
        The inputs to this function are the state 'x' and inputs 'u' (both
        1D arrays).  The output should be a 2D array with the first
        dimension equal to the number of system inputs and the second
        dimension of the length required to represent the full system
        dynamics (typically the number of states)

    x, u = flatsys.reverse(zflag, params)
        This function system state and inputs give the the flag (derivatives)
        of the flat output.  The input to this function is an 2D array whose
        first dimension is equal to the number of system inputs and whose
        second dimension is of length required to represent the full system
        dynamics (typically the number of states).  The output is the state
        `x` and inputs `u` (both 1D arrays).

    A flat system is also an input/output system supporting simulation,
    composition, and linearization.  If the update and output methods are
    given, they are used in place of the flat coordinates.

    """
    def __init__(self,
                 forward, reverse,              # flat system
                 updfcn=None, outfcn=None,      # I/O system
                 inputs=None, outputs=None,
                 states=None, params={}, dt=None, name=None):
        """Create a differentially flat I/O system.

        The FlatIOSystem constructor is used to create an input/output system
        object that also represents a differentially flat system.

        """

        # TODO: specify default update and output functions
        if updfcn is None: updfcn = self._flat_updfcn
        if outfcn is None: outfcn = self._flat_outfcn

        # Initialize as an input/output system
        NonlinearIOSystem.__init__(
            self, updfcn, outfcn, inputs=inputs, outputs=outputs,
            states=states, params=params, dt=dt, name=name)

        # Save the functions to compute forward and reverse conversions
        if forward is not None: self.forward = forward
        if reverse is not None: self.reverse = reverse

        # Save the length of the flat flag

    def __str__(self):
        return f"{NonlinearIOSystem.__str__(self)}\n\n" \
            + f"Forward: {self.forward}\n" \
            + f"Reverse: {self.reverse}"

    def forward(self, x, u, params={}):

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
            For each flat output :math:`z_i`, zflag[i] should be an
            ndarray of length :math:`q_i` that contains the flat
            output and its first :math:`q_i` derivatives.

        """
        raise NotImplementedError("internal error; forward method not defined")

    def reverse(self, zflag, params={}):
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

    def _flat_updfcn(self, t, x, u, params={}):
        # TODO: implement state space update using flat coordinates
        raise NotImplementedError("update function for flat system not given")

    def _flat_outfcn(self, t, x, u, params={}):
        # Return the flat output
        zflag = self.forward(x, u, params)
        return np.array([zflag[i][0] for i in range(len(zflag))])


# Utility function to compute flag matrix given a basis
def _basis_flag_matrix(sys, basis, flag, t, params={}):
    """Compute the matrix of basis functions and their derivatives

    This function computes the matrix ``M`` that is used to solve for the
    coefficients of the basis functions given the state and input.  Each
    column of the matrix corresponds to a basis function and each row is a
    derivative, with the derivatives (flag) for each output stacked on top
    of each other.

    """
    flagshape = [len(f) for f in flag]
    M = np.zeros((sum(flagshape), basis.N * sys.ninputs))
    flag_off = 0
    coeff_off = 0
    for i, flag_len in enumerate(flagshape):
        for j, k in itertools.product(range(basis.N), range(flag_len)):
            M[flag_off + k, coeff_off + j] = basis.eval_deriv(j, k, t)
        flag_off += flag_len
        coeff_off += basis.N
    return M


# Solve a point to point trajectory generation problem for a flat system
def point_to_point(
        sys, timepts, x0=0, u0=0, xf=0, uf=0, T0=0, cost=None, basis=None,
        trajectory_constraints=None, initial_guess=None, params=None, **kwargs):
    """Compute trajectory between an initial and final conditions.

    Compute a feasible trajectory for a differentially flat system between an
    initial condition and a final condition.

    Parameters
    ----------
    flatsys : FlatSystem object
        Description of the differentially flat system.  This object must
        define a function `flatsys.forward()` that takes the system state and
        produceds the flag of flat outputs and a system `flatsys.reverse()`
        that takes the flag of the flat output and prodes the state and
        input.

    timepts : float or 1D array_like
        The list of points for evaluating cost and constraints, as well as
        the time horizon.  If given as a float, indicates the final time for
        the trajectory (corresponding to xf)

    x0, u0, xf, uf : 1D arrays
        Define the desired initial and final conditions for the system.  If
        any of the values are given as None, they are replaced by a vector of
        zeros of the appropriate dimension.

    T0 : float, optional
        The initial time for the trajectory (corresponding to x0).  If not
        specified, its value is taken to be zero.

    basis : :class:`~control.flatsys.BasisFamily` object, optional
        The basis functions to use for generating the trajectory.  If not
        specified, the :class:`~control.flatsys.PolyFamily` basis family
        will be used, with the minimal number of elements required to find a
        feasible trajectory (twice the number of system states)

    cost : callable
        Function that returns the integral cost given the current state
        and input.  Called as `cost(x, u)`.

    trajectory_constraints : list of tuples, optional
        List of constraints that should hold at each point in the time vector.
        Each element of the list should consist of a tuple with first element
        given by :class:`scipy.optimize.LinearConstraint` or
        :class:`scipy.optimize.NonlinearConstraint` and the remaining
        elements of the tuple are the arguments that would be passed to those
        functions.  The following tuples are supported:

        * (LinearConstraint, A, lb, ub): The matrix A is multiplied by stacked
          vector of the state and input at each point on the trajectory for
          comparison against the upper and lower bounds.

        * (NonlinearConstraint, fun, lb, ub): a user-specific constraint
          function `fun(x, u)` is called at each point along the trajectory
          and compared against the upper and lower bounds.

        The constraints are applied at each time point along the trajectory.

    minimize_kwargs : str, optional
        Pass additional keywords to :func:`scipy.optimize.minimize`.

    Returns
    -------
    traj : :class:`~control.flatsys.SystemTrajectory` object
        The system trajectory is returned as an object that implements the
        `eval()` function, we can be used to compute the value of the state
        and input and a given time t.

    Notes
    -----
    Additional keyword parameters can be used to fine tune the behavior of
    the underlying optimization function.  See `minimize_*` keywords in
    :func:`OptimalControlProblem` for more information.

    """
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

    # Process keyword arguments
    if trajectory_constraints is None:
        # Backwards compatibility
        trajectory_constraints = kwargs.pop('constraints', None)

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

    # Make sure we have enough basis functions to solve the problem
    if basis.N * sys.ninputs < 2 * (sys.nstates + sys.ninputs):
        raise ValueError("basis set is too small")
    elif (cost is not None or trajectory_constraints is not None) and \
         basis.N * sys.ninputs == 2 * (sys.nstates + sys.ninputs):
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
    # is the matrix constrains for initial and final conditions and zflag =
    # [zflag_T0; zflag_tf].
    #
    # If there are no constraints, then we just need to solve a linear
    # system of equations => use least squares.  Otherwise, we have a
    # nonlinear optimal control problem with equality constraints => use
    # scipy.optimize.minimize().
    #

    # Start by solving the least squares problem
    alpha, residuals, rank, s = np.linalg.lstsq(M, Z, rcond=None)

    if cost is not None or trajectory_constraints is not None:
        # Search over the null space to minimize cost/satisfy constraints
        N = sp.linalg.null_space(M)

        # Define a function to evaluate the cost along a trajectory
        def traj_cost(null_coeffs):
            # Add this to the existing solution
            coeffs = alpha + N @ null_coeffs

            # Evaluate the costs at the listed time points
            costval = 0
            for t in timepts:
                M_t = _basis_flag_matrix(sys, basis, zflag_T0, t)

                # Compute flag at this time point
                zflag = (M_t @ coeffs).reshape(sys.ninputs, -1)

                # Find states and inputs at the time points
                x, u = sys.reverse(zflag, params)

                # Evaluate the cost at this time point
                costval += cost(x, u)
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
                for i, t in enumerate(timepts):
                    # Calculate the states and inputs for the flat output
                    M_t = _basis_flag_matrix(sys, basis, zflag_T0, t)

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

        # Process the initial condition
        if initial_guess is None:
            initial_guess = np.zeros(M.shape[1] - M.shape[0])
        else:
            raise NotImplementedError("Initial guess not yet implemented.")

        # Find the optimal solution
        res = sp.optimize.minimize(
            traj_cost, initial_guess, constraints=minimize_constraints,
            **minimize_kwargs)
        if res.success:
            alpha += N @ res.x
        else:
            raise RuntimeError(
                "Unable to solve optimal control problem\n" +
                "scipy.optimize.minimize returned " + res.message)

    #
    # Transform the trajectory from flat outputs to states and inputs
    #

    # Create a trajectory object to store the result
    systraj = SystemTrajectory(sys, basis, params=params)

    # Store the flag lengths and coefficients
    # TODO: make this more pythonic
    coeff_off = 0
    for i in range(sys.ninputs):
        # Grab the coefficients corresponding to this flat output
        systraj.coeffs.append(alpha[coeff_off:coeff_off + basis.N])
        coeff_off += basis.N

        # Keep track of the length of the flat flag for this output
        systraj.flaglen.append(len(zflag_T0[i]))

    # Return a function that computes inputs and states as a function of time
    return systraj
