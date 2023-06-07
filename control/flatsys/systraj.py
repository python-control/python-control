# systraj.py - SystemTrajectory class
# RMM, 10 November 2012
#
# The SystemTrajetory class is used to store a feasible trajectory for
# the state and input of a (nonlinear) control system.
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
from ..timeresp import TimeResponseData

class SystemTrajectory:
    """Class representing a trajectory for a flat system.

    The `SystemTrajectory` class is used to represent the
    trajectory of a (differentially flat) system.  Used by the
    :func:`~control.trajsys.point_to_point` function to return a trajectory.

    Parameters
    ----------
    sys : FlatSystem
        Flat system object associated with this trajectory.
    basis : BasisFamily
        Family of basis vectors to use to represent the trajectory.
    coeffs : list of 1D arrays, optional
        For each flat output, define the coefficients of the basis
        functions used to represent the trajectory.  Defaults to an empty
        list.
    flaglen : list of ints, optional
        For each flat output, the number of derivatives of the flat
        output used to define the trajectory.  Defaults to an empty
        list.

    """

    def __init__(self, sys, basis, coeffs=[], flaglen=[], params=None):
        """Initilize a system trajectory object."""
        self.nstates = sys.nstates
        self.ninputs = sys.ninputs
        self.system = sys
        self.basis = basis
        self.coeffs = list(coeffs)
        self.flaglen = list(flaglen)
        self.params = sys.params if params is None else params

    # Evaluate the trajectory over a list of time points
    def eval(self, tlist):
        """Return the state and input for a trajectory at a list of times.

        Evaluate the trajectory at a list of time points, returning the state
        and input vectors for the trajectory:

            x, u = traj.eval(tlist)

        Parameters
        ----------
        tlist : 1D array
            List of times to evaluate the trajectory.

        Returns
        -------
        x : 2D array
            For each state, the values of the state at the given times.
        u : 2D array
            For each input, the values of the input at the given times.

        """
        # Allocate space for the outputs
        xd = np.zeros((self.nstates, len(tlist)))
        ud = np.zeros((self.ninputs, len(tlist)))

        # Go through each time point and compute xd and ud via flat variables
        # TODO: make this more pythonic
        for tind, t in enumerate(tlist):
            zflag = []
            for i in range(self.ninputs):
                flag_len = self.flaglen[i]
                zflag.append(np.zeros(flag_len))
                for j in range(self.basis.var_ncoefs(i)):
                    for k in range(flag_len):
                        #! TODO: rewrite eval_deriv to take in time vector
                        zflag[i][k] += self.coeffs[i][j] * \
                            self.basis.eval_deriv(j, k, t, var=i)

            # Now copy the states and inputs
            # TODO: revisit order of list arguments
            xd[:,tind], ud[:,tind] = \
                self.system.reverse(zflag, self.params)

        return xd, ud

    # Return the system trajectory as a TimeResponseData object
    def response(self, tlist, transpose=False, return_x=False, squeeze=None):
        """Return the trajectory of a system as a TimeResponseData object

        Evaluate the trajectory at a list of time points, returning the state
        and input vectors for the trajectory:

            response = traj.response(tlist)
            time, yd, ud = response.time, response.outputs, response.inputs

        Parameters
        ----------
        tlist : 1D array
            List of times to evaluate the trajectory.

        transpose : bool, optional
            If True, transpose all input and output arrays (for backward
            compatibility with MATLAB and :func:`scipy.signal.lsim`).
            Default value is False.

        return_x : bool, optional
            If True, return the state vector when assigning to a tuple
            (default = False).  See :func:`forced_response` for more details.

        squeeze : bool, optional
            By default, if a system is single-input, single-output (SISO) then
            the output response is returned as a 1D array (indexed by time).
            If squeeze=True, remove single-dimensional entries from the shape
            of the output even if the system is not SISO. If squeeze=False,
            keep the output as a 3D array (indexed by the output, input, and
            time) even if the system is SISO. The default value can be set
            using config.defaults['control.squeeze_time_response'].

        Returns
        -------
        results : TimeResponseData
            Time response represented as a :class:`TimeResponseData` object
            containing the following properties:

            * time (array): Time values of the output.

            * outputs (array): Response of the system.  If the system is SISO
              and squeeze is not True, the array is 1D (indexed by time).  If
              the system is not SISO or ``squeeze`` is False, the array is 3D
              (indexed by the output, trace, and time).

            * states (array): Time evolution of the state vector, represented
              as either a 2D array indexed by state and time (if SISO) or a 3D
              array indexed by state, trace, and time.  Not affected by
              ``squeeze``.

            * inputs (array): Input(s) to the system, indexed in the same
              manner as ``outputs``.

            The return value of the system can also be accessed by assigning
            the function to a tuple of length 2 (time, output) or of length 3
            (time, output, state) if ``return_x`` is ``True``.

        """
        # Compute the state and input response using the eval function
        sys = self.system
        xout, uout = self.eval(tlist)
        yout = np.array([
            sys.output(tlist[i], xout[:, i], uout[:, i])
            for i in range(len(tlist))]).transpose()

        return TimeResponseData(
            tlist, yout, xout, uout, issiso=sys.issiso(),
            input_labels=sys.input_labels, output_labels=sys.output_labels,
            state_labels=sys.state_labels,
            transpose=transpose, return_x=return_x, squeeze=squeeze)
