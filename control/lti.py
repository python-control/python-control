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
from .namedio import NamedIOSystem, isdtime

__all__ = ['poles', 'zeros', 'damp', 'evalfr', 'frequency_response',
           'freqresp', 'dcgain', 'pole', 'zero']

class LTI(NamedIOSystem):
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

    Note: dt processing has been moved to the NamedIOSystem class.

    """
    def __init__(self, inputs=1, outputs=1, states=None, name=None, **kwargs):
        """Assign the LTI object's numbers of inputs and ouputs."""
        super().__init__(
            name=name, inputs=inputs, outputs=outputs, states=states, **kwargs)

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

        The ``inputs`` attribute was used to store the number of system
        inputs.  It is no longer used.  If you need access to the number
        of inputs for an LTI system, use :attr:`ninputs`.
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

        The ``outputs`` attribute was used to store the number of system
        outputs.  It is no longer used.  If you need access to the number of
        outputs for an LTI system, use :attr:`noutputs`.
        """)

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
        poles = self.poles()

        if self.isdtime(strict=True):
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

             G(j*omega) = mag * exp(j*phase)

        for continuous time systems. For discrete time systems, the response
        is evaluated around the unit circle such that

             G(exp(j*omega*dt)) = mag * exp(j*phase).

        In general the system may be multiple input, multiple output (MIMO),
        where `m = self.ninputs` number of inputs and `p = self.noutputs`
        number of outputs.

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
        response : :class:`FrequencyReponseData`
            Frequency response data object representing the frequency
            response.  This object can be assigned to a tuple using

                mag, phase, omega = response

            where ``mag`` is the magnitude (absolute value, not dB or
            log10) of the system frequency response, ``phase`` is the wrapped
            phase in radians of the system frequency response, and ``omega``
            is the (sorted) frequencies at which the response was evaluated.
            If the system is SISO and squeeze is not True, ``magnitude`` and
            ``phase`` are 1D, indexed by frequency.  If the system is not SISO
            or squeeze is False, the array is 3D, indexed by the output,
            input, and frequency.  If ``squeeze`` is True then
            single-dimensional axes are removed.

        """
        omega = np.sort(np.array(omega, ndmin=1))
        if self.isdtime(strict=True):
            # Convert the frequency to discrete time
            if np.any(omega * self.dt > np.pi):
                warn("__call__: evaluation above Nyquist frequency")
            s = np.exp(1j * omega * self.dt)
        else:
            s = 1j * omega

        # Return the data as a frequency response data object
        from .frdata import FrequencyResponseData
        response = self.__call__(s)
        return FrequencyResponseData(
            response, omega, return_magphase=True, squeeze=squeeze)

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

    #
    # Deprecated functions
    #

    def pole(self):
        warn("pole() will be deprecated; use poles()",
             PendingDeprecationWarning)
        return self.poles()

    def zero(self):
        warn("zero() will be deprecated; use zeros()",
             PendingDeprecationWarning)
        return self.zeros()


def poles(sys):
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

    See Also
    --------
    zeros
    TransferFunction.poles
    StateSpace.poles

    """

    return sys.poles()


def pole(sys):
    warn("pole() will be deprecated; use poles()", PendingDeprecationWarning)
    return poles(sys)


def zeros(sys):
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

    See Also
    --------
    poles
    StateSpace.zeros
    TransferFunction.zeros

    """

    return sys.zeros()


def zero(sys):
    warn("zero() will be deprecated; use zeros()", PendingDeprecationWarning)
    return zeros(sys)


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

def frequency_response(sys, omega, squeeze=None):
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
    response : FrequencyResponseData
        Frequency response data object representing the frequency response.
        This object can be assigned to a tuple using

            mag, phase, omega = response

        where ``mag`` is the magnitude (absolute value, not dB or log10) of
        the system frequency response, ``phase`` is the wrapped phase in
        radians of the system frequency response, and ``omega`` is the
        (sorted) frequencies at which the response was evaluated.  If the
        system is SISO and squeeze is not True, ``magnitude`` and ``phase``
        are 1D, indexed by frequency.  If the system is not SISO or squeeze
        is False, the array is 3D, indexed by the output, input, and
        frequency.  If ``squeeze`` is True then single-dimensional axes are
        removed.

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


# Alternative name (legacy)
freqresp = frequency_response


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
