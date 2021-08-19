"""lti.py

The lti module contains the LTI parent class to the child classes StateSpace
and TransferFunction.  It is designed for use in the python-control library.

Routines in this module:

LTI.__init__
isdtime()
isctime()
timebase()
common_timebase()
"""

import numpy as np
from numpy import absolute, real, angle, abs
from warnings import warn
from . import config

__all__ = ['issiso', 'timebase', 'common_timebase', 'timebaseEqual',
           'isdtime', 'isctime', 'pole', 'zero', 'damp', 'evalfr',
           'freqresp', 'dcgain']

class LTI:
    """LTI is a parent class to linear time-invariant (LTI) system objects.

    LTI is the parent to the StateSpace and TransferFunction child classes. It
    contains the number of inputs and outputs, and the timebase (dt) for the
    system.  This function is not generally called directly by the user.

    The timebase for the system, dt, is used to specify whether the system
    is operating in continuous or discrete time. It can have the following
    values:

      * dt = None       No timebase specified
      * dt = 0          Continuous time system
      * dt > 0          Discrete time system with sampling time dt
      * dt = True       Discrete time system with unspecified sampling time

    When two LTI systems are combined, their timebases much match.  A system
    with timebase None can be combined with a system having a specified
    timebase, and the result will have the timebase of the latter system.

    """

    def __init__(self, inputs=1, outputs=1, dt=None):
        """Assign the LTI object's numbers of inputs and ouputs."""

        # Data members common to StateSpace and TransferFunction.
        self.ninputs = inputs
        self.noutputs = outputs
        self.dt = dt

    #
    # Getter and setter functions for legacy state attributes
    #
    # For this iteration, generate a deprecation warning whenever the
    # getter/setter is called.  For a future iteration, turn it into a
    # future warning, so that users will see it.
    #

    def _get_inputs(self):
        warn("The LTI `inputs` attribute will be deprecated in a future "
             "release.  Use `ninputs` instead.",
             DeprecationWarning, stacklevel=2)
        return self.ninputs

    def _set_inputs(self, value):
        warn("The LTI `inputs` attribute will be deprecated in a future "
             "release.  Use `ninputs` instead.",
             DeprecationWarning, stacklevel=2)
        self.ninputs = value

    #: Deprecated
    inputs = property(
        _get_inputs, _set_inputs, doc=
        """
        Deprecated attribute; use :attr:`ninputs` instead.

        The ``input`` attribute was used to store the number of system inputs.
        It is no longer used.  If you need access to the number of inputs for
        an LTI system, use :attr:`ninputs`.
        """)

    def _get_outputs(self):
        warn("The LTI `outputs` attribute will be deprecated in a future "
             "release.  Use `noutputs` instead.",
             DeprecationWarning, stacklevel=2)
        return self.noutputs

    def _set_outputs(self, value):
        warn("The LTI `outputs` attribute will be deprecated in a future "
             "release.  Use `noutputs` instead.",
             DeprecationWarning, stacklevel=2)
        self.noutputs = value

    #: Deprecated
    outputs = property(
        _get_outputs, _set_outputs, doc=
        """
        Deprecated attribute; use :attr:`noutputs` instead.

        The ``output`` attribute was used to store the number of system
        outputs.  It is no longer used.  If you need access to the number of
        outputs for an LTI system, use :attr:`noutputs`.
        """)

    def isdtime(self, strict=False):
        """
        Check to see if a system is a discrete-time system

        Parameters
        ----------
        strict: bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.
        """

        # If no timebase is given, answer depends on strict flag
        if self.dt == None:
            return True if not strict else False

        # Look for dt > 0 (also works if dt = True)
        return self.dt > 0

    def isctime(self, strict=False):
        """
        Check to see if a system is a continuous-time system

        Parameters
        ----------
        sys : LTI system
            System to be checked
        strict: bool, optional
            If strict is True, make sure that timebase is not None.  Default
            is False.
        """
        # If no timebase is given, answer depends on strict flag
        if self.dt is None:
            return True if not strict else False
        return self.dt == 0

    def issiso(self):
        '''Check to see if a system is single input, single output'''
        return self.ninputs == 1 and self.noutputs == 1

    def damp(self):
        '''Natural frequency, damping ratio of system poles

        Returns
        -------
        wn : array
            Natural frequencies for each system pole
        zeta : array
            Damping ratio for each system pole
        poles : array
            Array of system poles
        '''
        poles = self.pole()

        if isdtime(self, strict=True):
            splane_poles = np.log(poles.astype(complex))/self.dt
        else:
            splane_poles = poles
        wn = absolute(splane_poles)
        Z = -real(splane_poles)/wn
        return wn, Z, poles

    def frequency_response(self, omega, squeeze=None):
        """Evaluate the linear time-invariant system at an array of angular
        frequencies.

        Reports the frequency response of the system,

             G(j*omega) = mag*exp(j*phase)

        for continuous time systems. For discrete time systems, the response is
        evaluated around the unit circle such that

             G(exp(j*omega*dt)) = mag*exp(j*phase).

        In general the system may be multiple input, multiple output (MIMO),
        where `m = self.ninputs` number of inputs and `p = self.noutputs` number
        of outputs.

        Parameters
        ----------
        omega : float or 1D array_like
            A list, tuple, array, or scalar value of frequencies in
            radians/sec at which the system will be evaluated.
        squeeze : bool, optional
            If squeeze=True, remove single-dimensional entries from the shape
            of the output even if the system is not SISO. If squeeze=False,
            keep all indices (output, input and, if omega is array_like,
            frequency) even if the system is SISO. The default value can be
            set using config.defaults['control.squeeze_frequency_response'].

        Returns
        -------
        mag : ndarray
            The magnitude (absolute value, not dB or log10) of the system
            frequency response.  If the system is SISO and squeeze is not
            True, the array is 1D, indexed by frequency.  If the system is not
            SISO or squeeze is False, the array is 3D, indexed by the output,
            input, and frequency.  If ``squeeze`` is True then
            single-dimensional axes are removed.
        phase : ndarray
            The wrapped phase in radians of the system frequency response.
        omega : ndarray
            The (sorted) frequencies at which the response was evaluated.

        """
        omega = np.sort(np.array(omega, ndmin=1))
        if isdtime(self, strict=True):
            # Convert the frequency to discrete time
            if np.any(omega * self.dt > np.pi):
                warn("__call__: evaluation above Nyquist frequency")
            s = np.exp(1j * omega * self.dt)
        else:
            s = 1j * omega
        response = self.__call__(s, squeeze=squeeze)
        return abs(response), angle(response), omega

    def dcgain(self):
        """Return the zero-frequency gain"""
        raise NotImplementedError("dcgain not implemented for %s objects" %
                                  str(self.__class__))

    def _dcgain(self, warn_infinite):
        zeroresp = self(0 if self.isctime() else 1,
                        warn_infinite=warn_infinite)
        if np.all(np.logical_or(np.isreal(zeroresp), np.isnan(zeroresp.imag))):
            return zeroresp.real
        else:
            return zeroresp

# Test to see if a system is SISO
def issiso(sys, strict=False):
    """
    Check to see if a system is single input, single output

    Parameters
    ----------
    sys : LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, do not treat scalars as SISO
    """
    if isinstance(sys, (int, float, complex, np.number)) and not strict:
        return True
    elif not isinstance(sys, LTI):
        raise ValueError("Object is not an LTI system")

    # Done with the tricky stuff...
    return sys.issiso()

# Return the timebase (with conversion if unspecified)
def timebase(sys, strict=True):
    """Return the timebase for an LTI system

    dt = timebase(sys)

    returns the timebase for a system 'sys'.  If the strict option is
    set to False, dt = True will be returned as 1.
    """
    # System needs to be either a constant or an LTI system
    if isinstance(sys, (int, float, complex, np.number)):
        return None
    elif not isinstance(sys, LTI):
        raise ValueError("Timebase not defined")

    # Return the sample time, with converstion to float if strict is false
    if (sys.dt == None):
        return None
    elif (strict):
        return float(sys.dt)

    return sys.dt

def common_timebase(dt1, dt2):
    """
    Find the common timebase when interconnecting systems

    Parameters
    ----------
    dt1, dt2: number or system with a 'dt' attribute (e.g. TransferFunction
        or StateSpace system)

    Returns
    -------
    dt: number
        The common timebase of dt1 and dt2, as specified in
        :ref:`conventions-ref`.

    Raises
    ------
    ValueError
        when no compatible time base can be found
    """
    # explanation:
    # if either dt is None, they are compatible with anything
    # if either dt is True (discrete with unspecified time base),
    #   use the timebase of the other, if it is also discrete
    # otherwise both dts must be equal
    if hasattr(dt1, 'dt'):
        dt1 = dt1.dt
    if hasattr(dt2, 'dt'):
        dt2 = dt2.dt

    if dt1 is None:
        return dt2
    elif dt2 is None:
        return dt1
    elif dt1 is True:
        if dt2 > 0:
            return dt2
        else:
            raise ValueError("Systems have incompatible timebases")
    elif dt2 is True:
        if dt1 > 0:
            return dt1
        else:
            raise ValueError("Systems have incompatible timebases")
    elif np.isclose(dt1, dt2):
        return dt1
    else:
        raise ValueError("Systems have incompatible timebases")

# Check to see if two timebases are equal
def timebaseEqual(sys1, sys2):
    """
    Check to see if two systems have the same timebase

    timebaseEqual(sys1, sys2)

    returns True if the timebases for the two systems are compatible.  By
    default, systems with timebase 'None' are compatible with either
    discrete or continuous timebase systems.  If two systems have a discrete
    timebase (dt > 0) then their timebases must be equal.
    """
    warn("timebaseEqual will be deprecated in a future release of "
         "python-control; use :func:`common_timebase` instead",
         PendingDeprecationWarning)

    if (type(sys1.dt) == bool or type(sys2.dt) == bool):
        # Make sure both are unspecified discrete timebases
        return type(sys1.dt) == type(sys2.dt) and sys1.dt == sys2.dt
    elif (sys1.dt is None or sys2.dt is None):
        # One or the other is unspecified => the other can be anything
        return True
    else:
        return sys1.dt == sys2.dt


# Check to see if a system is a discrete time system
def isdtime(sys, strict=False):
    """
    Check to see if a system is a discrete time system

    Parameters
    ----------
    sys : LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, make sure that timebase is not None
    """

    # Check to see if this is a constant
    if isinstance(sys, (int, float, complex, np.number)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer function or state-space object
    if isinstance(sys, LTI):
        return sys.isdtime(strict)

    # Check to see if object has a dt object
    if hasattr(sys, 'dt'):
        # If no timebase is given, answer depends on strict flag
        if sys.dt == None:
            return True if not strict else False

        # Look for dt > 0 (also works if dt = True)
        return sys.dt > 0

    # Got passed something we don't recognize
    return False

# Check to see if a system is a continuous time system
def isctime(sys, strict=False):
    """
    Check to see if a system is a continuous-time system

    Parameters
    ----------
    sys : LTI system
        System to be checked
    strict: bool (default = False)
        If strict is True, make sure that timebase is not None
    """

    # Check to see if this is a constant
    if isinstance(sys, (int, float, complex, np.number)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer function or state space object
    if isinstance(sys, LTI):
        return sys.isctime(strict)

    # Check to see if object has a dt object
    if hasattr(sys, 'dt'):
        # If no timebase is given, answer depends on strict flag
        if sys.dt is None:
            return True if not strict else False
        return sys.dt == 0

    # Got passed something we don't recognize
    return False

def pole(sys):
    """
    Compute system poles.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system

    Returns
    -------
    poles: ndarray
        Array that contains the system's poles.

    Raises
    ------
    NotImplementedError
        when called on a TransferFunction object

    See Also
    --------
    zero
    TransferFunction.pole
    StateSpace.pole

    """

    return sys.pole()


def zero(sys):
    """
    Compute system zeros.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system

    Returns
    -------
    zeros: ndarray
        Array that contains the system's zeros.

    Raises
    ------
    NotImplementedError
        when called on a MIMO system

    See Also
    --------
    pole
    StateSpace.zero
    TransferFunction.zero

    """

    return sys.zero()

def damp(sys, doprint=True):
    """
    Compute natural frequency, damping ratio, and poles of a system

    The function takes 1 or 2 parameters

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        A linear system object
    doprint:
        if true, print table with values

    Returns
    -------
    wn: array
        Natural frequencies of the poles
    damping: array
        Damping values
    poles: array
        Pole locations

    Algorithm
    ---------
    If the system is continuous,
        wn = abs(poles)
        Z  = -real(poles)/poles.

    If the system is discrete, the discrete poles are mapped to their
    equivalent location in the s-plane via

        s = log10(poles)/dt

    and

        wn = abs(s)
        Z = -real(s)/wn.

    See Also
    --------
    pole
    """
    wn, damping, poles = sys.damp()
    if doprint:
        print('_____Eigenvalue______ Damping___ Frequency_')
        for p, d, w in zip(poles, damping, wn) :
            if abs(p.imag) < 1e-12:
                print("%10.4g            %10.4g %10.4g" %
                      (p.real, 1.0, -p.real))
            else:
                print("%10.4g%+10.4gj %10.4g %10.4g" %
                      (p.real, p.imag, d, w))
    return wn, damping, poles

def evalfr(sys, x, squeeze=None):
    """Evaluate the transfer function of an LTI system for complex frequency x.

    Returns the complex frequency response `sys(x)` where `x` is `s` for
    continuous-time systems and `z` for discrete-time systems, with
    `m = sys.ninputs` number of inputs and `p = sys.noutputs` number of
    outputs.

    To evaluate at a frequency omega in radians per second, enter
    ``x = omega * 1j`` for continuous-time systems, or
    ``x = exp(1j * omega * dt)`` for discrete-time systems, or use
    ``freqresp(sys, omega)``.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system
    x : complex scalar or 1D array_like
        Complex frequency(s)
    squeeze : bool, optional (default=True)
        If squeeze=True, remove single-dimensional entries from the shape of
        the output even if the system is not SISO. If squeeze=False, keep all
        indices (output, input and, if omega is array_like, frequency) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_frequency_response'].

    Returns
    -------
    fresp : complex ndarray
        The frequency response of the system.  If the system is SISO and
        squeeze is not True, the shape of the array matches the shape of
        omega.  If the system is not SISO or squeeze is False, the first two
        dimensions of the array are indices for the output and input and the
        remaining dimensions match omega.  If ``squeeze`` is True then
        single-dimensional axes are removed.

    See Also
    --------
    freqresp
    bode

    Notes
    -----
    This function is a wrapper for :meth:`StateSpace.__call__` and
    :meth:`TransferFunction.__call__`.

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> evalfr(sys, 1j)
    array([[ 44.8-21.4j]])
    >>> # This is the transfer function matrix evaluated at s = i.

    .. todo:: Add example with MIMO system

    """
    return sys.__call__(x, squeeze=squeeze)

def freqresp(sys, omega, squeeze=None):
    """Frequency response of an LTI system at multiple angular frequencies.

    In general the system may be multiple input, multiple output (MIMO), where
    `m = sys.ninputs` number of inputs and `p = sys.noutputs` number of
    outputs.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system
    omega : float or 1D array_like
        A list of frequencies in radians/sec at which the system should be
        evaluated. The list can be either a python list or a numpy array
        and will be sorted before evaluation.
    squeeze : bool, optional
        If squeeze=True, remove single-dimensional entries from the shape of
        the output even if the system is not SISO. If squeeze=False, keep all
        indices (output, input and, if omega is array_like, frequency) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_frequency_response'].

    Returns
    -------
    mag : ndarray
        The magnitude (absolute value, not dB or log10) of the system
        frequency response.  If the system is SISO and squeeze is not True,
        the array is 1D, indexed by frequency.  If the system is not SISO or
        squeeze is False, the array is 3D, indexed by the output, input, and
        frequency.  If ``squeeze`` is True then single-dimensional axes are
        removed.
    phase : ndarray
        The wrapped phase in radians of the system frequency response.
    omega : ndarray
        The list of sorted frequencies at which the response was
        evaluated.

    See Also
    --------
    evalfr
    bode

    Notes
    -----
    This function is a wrapper for :meth:`StateSpace.frequency_response` and
    :meth:`TransferFunction.frequency_response`.

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = freqresp(sys, [0.1, 1., 10.])
    >>> mag
    array([[[ 58.8576682 ,  49.64876635,  13.40825927]]])
    >>> phase
    array([[[-0.05408304, -0.44563154, -0.66837155]]])

    .. todo::
        Add example with MIMO system

        #>>> sys = rss(3, 2, 2)
        #>>> mag, phase, omega = freqresp(sys, [0.1, 1., 10.])
        #>>> mag[0, 1, :]
        #array([ 55.43747231,  42.47766549,   1.97225895])
        #>>> phase[1, 0, :]
        #array([-0.12611087, -1.14294316,  2.5764547 ])
        #>>> # This is the magnitude of the frequency response from the 2nd
        #>>> # input to the 1st output, and the phase (in radians) of the
        #>>> # frequency response from the 1st input to the 2nd output, for
        #>>> # s = 0.1i, i, 10i.

    """
    return sys.frequency_response(omega, squeeze=squeeze)


def dcgain(sys):
    """Return the zero-frequency (or DC) gain of the given system

    Returns
    -------
    gain : ndarray
        The zero-frequency gain, or (inf + nanj) if the system has a pole at
        the origin, (nan + nanj) if there is a pole/zero cancellation at the
        origin.

    """
    return sys.dcgain()


# Process frequency responses in a uniform way
def _process_frequency_response(sys, omega, out, squeeze=None):
    # Set value of squeeze argument if not set
    if squeeze is None:
        squeeze = config.defaults['control.squeeze_frequency_response']

    if np.asarray(omega).ndim < 1:
        # received a scalar x, squeeze down the array along last dim
        out = np.squeeze(out, axis=2)

    #
    # Get rid of unneeded dimensions
    #
    # There are three possible values for the squeeze keyword at this point:
    #
    #   squeeze=None: squeeze input/output axes iff SISO
    #   squeeze=True: squeeze all single dimensional axes (ala numpy)
    #   squeeze-False: don't squeeze any axes
    #
    if squeeze is True:
        # Squeeze everything that we can if that's what the user wants
        return np.squeeze(out)
    elif squeeze is None and sys.issiso():
        # SISO system output squeezed unless explicitly specified otherwise
        return out[0][0]
    elif squeeze is False or squeeze is None:
        return out
    else:
        raise ValueError("unknown squeeze value")
