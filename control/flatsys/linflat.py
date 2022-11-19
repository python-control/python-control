# linflat.py - FlatSystem subclass for linear systems
# RMM, 10 November 2012
#
# This file defines a FlatSystem class for a linear system.
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
import control
from .flatsys import FlatSystem
from ..iosys import LinearIOSystem


class LinearFlatSystem(FlatSystem, LinearIOSystem):
    """Base class for a linear, differentially flat system.

    This class is used to create a differentially flat system representation
    from a linear system.

    Parameters
    ----------
    linsys : StateSpace
        LTI StateSpace system to be converted
    inputs : int, list of str or None, optional
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.
        If an integer count is specified, the names of the signal will be
        of the form `s[i]` (where `s` is one of `u`, `y`, or `x`).  If
        this parameter is not given or given as `None`, the relevant
        quantity will be determined when possible based on other
        information provided to functions using the system.
    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None, optional
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

    """

    def __init__(self, linsys, inputs=None, outputs=None, states=None,
                 name=None):
        """Define a flat system from a SISO LTI system.

        Given a reachable, single-input/single-output, linear time-invariant
        system, create a differentially flat system representation.

        """
        # Make sure we can handle the system
        if (not control.isctime(linsys)):
            raise control.ControlNotImplemented(
                "requires continuous time, linear control system")
        elif (not control.issiso(linsys)):
            raise control.ControlNotImplemented(
                "only single input, single output systems are supported")

        # Initialize the object as a LinearIO system
        LinearIOSystem.__init__(
            self, linsys, inputs=inputs, outputs=outputs, states=states,
            name=name)

        # Find the transformation to chain of integrators form
        # Note: store all array as ndarray, not matrix
        zsys, Tr = control.reachable_form(linsys)
        Tr = np.array(Tr[::-1, ::])     # flip rows

        # Extract the information that we need
        self.F = np.array(zsys.A[0, ::-1])      # input function coeffs
        self.T = Tr                             # state space transformation
        self.Tinv = np.linalg.inv(Tr)           # compute inverse once

        # Compute the flat output variable z = C x
        Cfz = np.zeros(np.shape(linsys.C)); Cfz[0, 0] = 1
        self.Cf = Cfz @ Tr

    # Compute the flat flag from the state (and input)
    def forward(self, x, u, params):
        """Compute the flat flag given the states and input.

        See :func:`control.flatsys.FlatSystem.forward` for more info.

        """
        x = np.reshape(x, (-1, 1))
        u = np.reshape(u, (1, -1))
        zflag = [np.zeros(self.nstates + 1)]
        zflag[0][0] = self.Cf @ x
        H = self.Cf                     # initial state transformation
        for i in range(1, self.nstates + 1):
            zflag[0][i] = H @ (self.A @ x + self.B @ u)
            H = H @ self.A       # derivative for next iteration
        return zflag

    # Compute state and input from flat flag
    def reverse(self, zflag, params):
        """Compute the states and input given the flat flag.

        See :func:`control.flatsys.FlatSystem.reverse` for more info.

        """
        z = zflag[0][0:-1]
        x = self.Tinv @ z
        u = zflag[0][-1] - self.F @ z
        return np.reshape(x, self.nstates), np.reshape(u, self.ninputs)

    # Update function
    def _rhs(self, t, x, u):
        # Use LinearIOSystem._rhs instead of default (MRO) NonlinearIOSystem
        return LinearIOSystem._rhs(self, t, x, u)

    # output function
    def _out(self, t, x, u):
        # Use LinearIOSystem._out instead of default (MRO) NonlinearIOSystem
        return LinearIOSystem._out(self, t, x, u)
