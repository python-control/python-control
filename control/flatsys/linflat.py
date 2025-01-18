# linflat.py - FlatSystem subclass for linear systems
# RMM, 10 November 2012

"""FlatSystem class for a linear system.

"""

import numpy as np

import control

from ..statesp import StateSpace
from .flatsys import FlatSystem


class LinearFlatSystem(FlatSystem, StateSpace):
    """Base class for a linear, differentially flat system.

    This class is used to create a differentially flat system representation
    from a linear system.

    Parameters
    ----------
    linsys : `StateSpace`
        LTI `StateSpace` system to be converted.
    inputs : int, list of str or None, optional
        Description of the system inputs.  This can be given as an integer
        count or as a list of strings that name the individual signals.
        If an integer count is specified, the names of the signal will be
        of the form 's[i]' (where 's' is one of 'u', 'y', or 'x').  If
        this parameter is not given or given as None, the relevant
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
        System name (used for specifying signals).

    """

    def __init__(self, linsys, **kwargs):
        """Define a flat system from a SISO LTI system.

        Given a reachable, single-input/single-output, linear time-invariant
        system, create a differentially flat system representation.

        """
        # Make sure we can handle the system
        if (not control.isctime(linsys)):
            raise control.ControlNotImplemented(
                "requires continuous-time, linear control system")
        elif (not control.issiso(linsys)):
            raise control.ControlNotImplemented(
                "only single input, single output systems are supported")

        # Initialize the object as a StateSpace system
        StateSpace.__init__(self, linsys, **kwargs)

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

        See `FlatSystem.forward` for more info.

        """
        x = np.reshape(x, (-1, 1))
        u = np.reshape(u, (1, -1))
        zflag = [np.zeros(self.nstates + 1)]
        zflag[0][0] = (self.Cf @ x).item()
        H = self.Cf                     # initial state transformation
        for i in range(1, self.nstates + 1):
            zflag[0][i] = (H @ (self.A @ x + self.B @ u)).item()
            H = H @ self.A       # derivative for next iteration
        return zflag

    # Compute state and input from flat flag
    def reverse(self, zflag, params):
        """Compute the states and input given the flat flag.

        See `FlatSystem.reverse` for more info.

        """
        z = zflag[0][0:-1]
        x = self.Tinv @ z
        u = zflag[0][-1] - self.F @ z
        return np.reshape(x, self.nstates), np.reshape(u, self.ninputs)

    # Update function
    def _rhs(self, t, x, u):
        # Use StateSpace._rhs instead of default (MRO) NonlinearIOSystem
        return StateSpace._rhs(self, t, x, u)

    # output function
    def _out(self, t, x, u):
        # Use StateSpace._out instead of default (MRO) NonlinearIOSystem
        return StateSpace._out(self, t, x, u)
