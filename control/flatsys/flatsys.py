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

import numpy as np
from .poly import PolyFamily
from .systraj import SystemTrajectory
from ..iosys import NonlinearIOSystem


# Flat system class (for use as a base class)
class FlatSystem(NonlinearIOSystem):
    """Base class for representing a differentially flat system.

    The FlatSystem class is used as a base class to describe differentially
    flat systems for trajectory generation.  The class must implement two
    functions:

    zflag = flatsys.foward(x, u)
        This function computes the flag (derivatives) of the flat output.
        The inputs to this function are the state 'x' and inputs 'u' (both
        1D arrays).  The output should be a 2D array with the first
        dimension equal to the number of system inputs and the second
        dimension of the length required to represent the full system
        dynamics (typically the number of states)

    x, u = flatsys.reverse(zflag)
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
        """Create a differentially flat input/output system.

        The FlatIOSystem constructor is used to create an input/output system
        object that also represents a differentially flat system.  The output
        of the system does not need to be the differentially flat output.

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

        Returns
        -------
        InputOutputSystem
            Input/output system object

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
        pass

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
        pass

    def _flat_updfcn(self, t, x, u, params={}):
        # TODO: implement state space update using flat coordinates
        raise NotImplementedError("update function for flat system not given")

    def _flat_outfcn(self, t, x, u, params={}):
        # Return the flat output
        zflag = self.forward(x, u, params)
        return np.array(zflag[:][0])


# Solve a point to point trajectory generation problem for a linear system
def point_to_point(sys, x0, u0, xf, uf, Tf, T0=0, basis=None, cost=None):
    """Compute trajectory between an initial and final conditions.

    Compute a feasible trajectory for a differentially flat system between an
    initial condition and a final condition.

    Parameters
    ----------
    flatsys : FlatSystem object
        Description of the differentially flat system.  This object must
        define a function flatsys.forward() that takes the system state and
        produceds the flag of flat outputs and a system flatsys.reverse()
        that takes the flag of the flat output and prodes the state and
        input.

    x0, u0, xf, uf : 1D arrays
        Define the desired initial and final conditions for the system.  If
        any of the values are given as None, they are replaced by a vector of
        zeros of the appropriate dimension.

    Tf : float
        The final time for the trajectory (corresponding to xf)

    T0 : float (optional)
        The initial time for the trajectory (corresponding to x0).  If not
        specified, its value is taken to be zero.

    basis : BasisFamily object (optional)
        The basis functions to use for generating the trajectory.  If not
        specified, the PolyFamily basis family will be used, with the minimal
        number of elements required to find a feasible trajectory (twice
        the number of system states)

    Returns
    -------
    traj : SystemTrajectory object
        The system trajectory is returned as an object that implements the
        eval() function, we can be used to compute the value of the state
        and input and a given time t.

    """
    #
    # Make sure the problem is one that we can handle
    #
    # TODO: put in tests for flat system input
    # TODO: process initial and final conditions to allow x0 or (x0, u0)
    if x0 is None: x0 = np.zeros(sys.nstates)
    if u0 is None: u0 = np.zeros(sys.ninputs)
    if xf is None: xf = np.zeros(sys.nstates)
    if uf is None: uf = np.zeros(sys.ninputs)

    #
    # Determine the basis function set to use and make sure it is big enough
    #

    # If no basis set was specified, use a polynomial basis (poor choice...)
    if (basis is None): basis = PolyFamily(2*sys.nstates, Tf)

    # Make sure we have enough basis functions to solve the problem
    if (basis.N * sys.ninputs < 2 * (sys.nstates + sys.ninputs)):
        raise ValueError("basis set is too small")

    #
    # Map the initial and final conditions to flat output conditions
    #
    # We need to compute the output "flag": [z(t), z'(t), z''(t), ...]
    # and then evaluate this at the initial and final condition.
    #
    # TODO: should be able to represent flag variables as 1D arrays
    # TODO: need inputs to fully define the flag
    zflag_T0 = sys.forward(x0, u0)
    zflag_Tf = sys.forward(xf, uf)

    #
    # Compute the matrix constraints for initial and final conditions
    #
    # This computation depends on the basis function we are using.  It
    # essentially amounts to evaluating the basis functions and their
    # derivatives at the initial and final conditions.

    # Figure out the size of the problem we are solving
    flag_tot = np.sum([len(zflag_T0[i]) for i in range(sys.ninputs)])

    # Start by creating an empty matrix that we can fill up
    # TODO: allow a different number of basis elements for each flat output
    M = np.zeros((2 * flag_tot, basis.N * sys.ninputs))

    # Now fill in the rows for the initial and final states
    flag_off = 0
    coeff_off = 0
    for i in range(sys.ninputs):
        flag_len = len(zflag_T0[i])
        for j in range(basis.N):
            for k in range(flag_len):
                M[flag_off + k, coeff_off + j] = basis.eval_deriv(j, k, T0)
                M[flag_tot + flag_off + k, coeff_off + j] = \
                    basis.eval_deriv(j, k, Tf)
        flag_off += flag_len
        coeff_off += basis.N

    # Create an empty matrix that we can fill up
    Z = np.zeros(2 * flag_tot)

    # Compute the flag vector to use for the right hand side by
    # stacking up the flags for each input
    # TODO: make this more pythonic
    flag_off = 0
    for i in range(sys.ninputs):
        flag_len = len(zflag_T0[i])
        for j in range(flag_len):
            Z[flag_off + j] = zflag_T0[i][j]
            Z[flag_tot + flag_off + j] = zflag_Tf[i][j]
        flag_off += flag_len

    #
    # Solve for the coefficients of the flat outputs
    #
    # At this point, we need to solve the equation M alpha = zflag, where M
    # is the matrix constrains for initial and final conditions and zflag =
    # [zflag_T0; zflag_tf].  Since everything is linear, just compute the
    # least squares solution for now.
    #
    # TODO: need to allow cost and constraints...
    alpha, residuals, rank, s = np.linalg.lstsq(M, Z, rcond=None)

    #
    # Transform the trajectory from flat outputs to states and inputs
    #
    systraj = SystemTrajectory(sys, basis)

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
