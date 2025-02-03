# systraj.py - SystemTrajectory class
# RMM, 10 November 2012

"""SystemTrajectory class.

The SystemTrajectory class is used to store a feasible trajectory for
the state and input of a (nonlinear) control system.

"""

import numpy as np

from ..timeresp import TimeResponseData


class SystemTrajectory:
    """Trajectory for a differentially flat system.

    The `SystemTrajectory` class is used to represent the trajectory
    of a (differentially flat) system.  Used by the `point_to_point`
    and `solve_flat_optimal` functions to return a trajectory.

    Parameters
    ----------
    sys : `FlatSystem`
        Flat system object associated with this trajectory.
    basis : `BasisFamily`
        Family of basis vectors to use to represent the trajectory.
    coeffs : list of 1D arrays, optional
        For each flat output, define the coefficients of the basis
        functions used to represent the trajectory.  Defaults to an empty
        list.
    flaglen : list of int, optional
        For each flat output, the number of derivatives of the flat
        output used to define the trajectory.  Defaults to an empty
        list.
    params : dict, optional
        Parameter values used for the trajectory.

    """

    def __init__(self, sys, basis, coeffs=[], flaglen=[], params=None):
        """Initialize a system trajectory object."""
        self.nstates = sys.nstates
        self.ninputs = sys.ninputs
        self.system = sys
        self.basis = basis
        self.coeffs = list(coeffs)
        self.flaglen = list(flaglen)
        self.params = sys.params if params is None else params

    # Evaluate the trajectory over a list of time points
    def eval(self, tlist):
        """Compute state and input for a trajectory at a list of times.

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
    def response(self, timepts, transpose=False, return_x=False, squeeze=None):
        """Compute trajectory of a system as a TimeResponseData object.

        Evaluate the trajectory at a list of time points, returning the state
        and input vectors for the trajectory:

            response = traj.response(timepts)
            time, yd, ud = response.time, response.outputs, response.inputs

        Parameters
        ----------
        timepts : 1D array
            List of times to evaluate the trajectory.

        transpose : bool, optional
            If True, transpose all input and output arrays (for backward
            compatibility with MATLAB and `scipy.signal.lsim`).
            Default value is False.

        return_x : bool, optional
            If True, return the state vector when assigning to a tuple
            (default = False).  See `forced_response` for more details.

        squeeze : bool, optional
            By default, if a system is single-input, single-output (SISO)
            then the output response is returned as a 1D array (indexed by
            time).  If `squeeze` = True, remove single-dimensional entries
            from the shape of the output even if the system is not SISO. If
            `squeeze` = False, keep the output as a 3D array (indexed by
            the output, input, and time) even if the system is SISO. The
            default value can be set using
            `config.defaults['control.squeeze_time_response']`.

        Returns
        -------
        response : `TimeResponseData`
            Time response data object representing the input/output response.
            When accessed as a tuple, returns ``(time, outputs)`` or ``(time,
            outputs, states`` if `return_x` is True.  If the input/output
            system signals are named, these names will be used as labels for
            the time response.  If `sys` is a list of systems, returns a
            `TimeResponseList` object.  Results can be plotted using the
            `~TimeResponseData.plot` method.  See `TimeResponseData` for more
            detailed information.
        response.time : array
            Time values of the output.
        response.outputs : array
            Response of the system.  If the system is SISO and `squeeze` is
            not True, the array is 1D (indexed by time).  If the system is not
            SISO or `squeeze` is False, the array is 2D (indexed by output and
            time).
        response.states : array
            Time evolution of the state vector, represented as a 2D array
            indexed by state and time.
        response.inputs : array
            Input(s) to the system, indexed by input and time.

        """
        # Compute the state and input response using the eval function
        sys = self.system
        xout, uout = self.eval(timepts)
        yout = np.array([
            sys.output(timepts[i], xout[:, i], uout[:, i])
            for i in range(len(timepts))]).transpose()

        return TimeResponseData(
            timepts, yout, xout, uout, issiso=sys.issiso(),
            input_labels=sys.input_labels, output_labels=sys.output_labels,
            state_labels=sys.state_labels, sysname=sys.name,
            transpose=transpose, return_x=return_x, squeeze=squeeze)
