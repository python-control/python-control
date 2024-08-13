"""lti.py

The lti module contains the LTI parent class to the child classes StateSpace
and TransferFunction.  It is designed for use in the python-control library.
"""

import numpy as np
import math

from numpy import real, angle, abs
from warnings import warn
from . import config
from .iosys import InputOutputSystem

__all__ = ['poles', 'zeros', 'damp', 'evalfr', 'frequency_response',
           'freqresp', 'dcgain', 'bandwidth', 'LTI']


class LTI(InputOutputSystem):
    """LTI is a parent class to linear time-invariant (LTI) system objects.

    LTI is the parent to the StateSpace and TransferFunction child classes. It
    contains the number of inputs and outputs, and the timebase (dt) for the
    system.  This function is not generally called directly by the user.

    When two LTI systems are combined, their timebases much match.  A system
    with timebase None can be combined with a system having a specified
    timebase, and the result will have the timebase of the latter system.

    Note: dt processing has been moved to the InputOutputSystem class.

    """
    def __init__(self, inputs=1, outputs=1, states=None, name=None, **kwargs):
        """Assign the LTI object's numbers of inputs and ouputs."""
        super().__init__(
            name=name, inputs=inputs, outputs=outputs, states=states, **kwargs)

    def damp(self):
        '''Natural frequency, damping ratio of system poles

        Returns
        -------
        wn : array
            Natural frequency for each system pole
        zeta : array
            Damping ratio for each system pole
        poles : array
            System pole locations
        '''
        poles = self.poles()

        if self.isdtime(strict=True):
            splane_poles = np.log(poles.astype(complex))/self.dt
        else:
            splane_poles = poles
        wn = abs(splane_poles)
        zeta = -real(splane_poles)/wn
        return wn, zeta, poles

    def frequency_response(self, omega=None, squeeze=None):
        """Evaluate the linear time-invariant system at an array of angular
        frequencies.

        For continuous time systems, computes the frequency response as

             G(j*omega) = mag * exp(j*phase)

        For discrete time systems, the response is evaluated around the
        unit circle such that

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
        response : :class:`FrequencyResponseData`
            Frequency response data object representing the frequency
            response.  This object can be assigned to a tuple using

                mag, phase, omega = response

            where ``mag`` is the magnitude (absolute value, not dB or log10)
            of the system frequency response, ``phase`` is the wrapped phase
            in radians of the system frequency response, and ``omega`` is
            the (sorted) frequencies at which the response was evaluated.
            If the system is SISO and squeeze is not True, ``magnitude`` and
            ``phase`` are 1D, indexed by frequency.  If the system is not
            SISO or squeeze is False, the array is 3D, indexed by the
            output, input, and, if omega is array_like, frequency.  If
            ``squeeze`` is True then single-dimensional axes are removed.

        """
        from .frdata import FrequencyResponseData

        omega = np.sort(np.array(omega, ndmin=1))
        if self.isdtime(strict=True):
            # Convert the frequency to discrete time
            if np.any(omega * self.dt > np.pi):
                warn("__call__: evaluation above Nyquist frequency")
            s = np.exp(1j * omega * self.dt)
        else:
            s = 1j * omega

        # Return the data as a frequency response data object
        response = self(s)
        return FrequencyResponseData(
            response, omega, return_magphase=True, squeeze=squeeze,
            dt=self.dt, sysname=self.name, inputs=self.input_labels,
            outputs=self.output_labels, plot_type='bode')

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

    def bandwidth(self, dbdrop=-3):
        """Evaluate the bandwidth of the LTI system for a given dB drop.

        Evaluate the first frequency that the response magnitude is lower than
        DC gain by dbdrop dB.

        Parameters
        ----------
        dpdrop : float, optional
            A strictly negative scalar in dB (default = -3) defines the
            amount of gain drop for deciding bandwidth.

        Returns
        -------
        bandwidth : ndarray
            The first frequency (rad/time-unit) where the gain drops below
            dbdrop of the dc gain of the system, or nan if the system has
            infinite dc gain, inf if the gain does not drop for all frequency

        Raises
        ------
        TypeError
            if 'sys' is not an SISO LTI instance
        ValueError
            if 'dbdrop' is not a negative scalar
        """
        # check if system is SISO and dbdrop is a negative scalar
        if not self.issiso():
            raise TypeError("system should be a SISO system")

        if (not np.isscalar(dbdrop)) or dbdrop >= 0:
            raise ValueError("expecting dbdrop be a negative scalar in dB")

        dcgain = self.dcgain()
        if np.isinf(dcgain):
            # infinite dcgain, return np.nan
            return np.nan

        # use frequency range to identify the 0-crossing (dbdrop) bracket
        from control.freqplot import _default_frequency_range
        omega = _default_frequency_range(self)
        mag, phase, omega = self.frequency_response(omega)
        idx_dropped = np.nonzero(mag - dcgain*10**(dbdrop/20) < 0)[0]

        if idx_dropped.shape[0] == 0:
            # no frequency response is dbdrop below the dc gain, return np.inf
            return np.inf
        else:
            # solve for the bandwidth, use scipy.optimize.root_scalar() to
            # solve using bisection
            import scipy
            result = scipy.optimize.root_scalar(
                lambda w: np.abs(self(w*1j)) - np.abs(dcgain)*10**(dbdrop/20),
                bracket=[omega[idx_dropped[0] - 1], omega[idx_dropped[0]]],
                method='bisect')

            # check solution
            if result.converged:
                return np.abs(result.root)
            else:
                raise Exception(result.message)

    def ispassive(self):
        # importing here prevents circular dependancy
        from control.passivity import ispassive
        return ispassive(self)


def poles(sys):
    """
    Compute system poles.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        Linear system

    Returns
    -------
    poles : ndarray
        Array that contains the system's poles.

    See Also
    --------
    zeros
    TransferFunction.poles
    StateSpace.poles

    """

    return sys.poles()


def zeros(sys):
    """
    Compute system zeros.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
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


def damp(sys, doprint=True):
    """
    Compute natural frequencies, damping ratios, and poles of a system.

    Parameters
    ----------
    sys : LTI (StateSpace or TransferFunction)
        A linear system object
    doprint : bool (optional)
        if True, print table with values

    Returns
    -------
    wn : array
        Natural frequency for each system pole
    zeta : array
        Damping ratio for each system pole
    poles : array
        System pole locations

    See Also
    --------
    pole

    Notes
    -----
    If the system is continuous,
        wn = abs(poles)
        zeta  = -real(poles)/poles

    If the system is discrete, the discrete poles are mapped to their
    equivalent location in the s-plane via

        s = log(poles)/dt

    and

        wn = abs(s)
        zeta = -real(s)/wn.

    Examples
    --------
    >>> G = ct.tf([1], [1, 4])
    >>> wn, zeta, poles = ct.damp(G)
        Eigenvalue (pole)       Damping     Frequency
                       -4             1             4

    """
    wn, zeta, poles = sys.damp()
    if doprint:
        print('    Eigenvalue (pole)       Damping     Frequency')
        for p, z, w in zip(poles, zeta, wn):
            if abs(p.imag) < 1e-12:
                print("           %10.4g    %10.4g    %10.4g" %
                      (p.real, 1.0, w))
            else:
                print("%10.4g%+10.4gj    %10.4g    %10.4g" %
                      (p.real, p.imag, z, w))
    return wn, zeta, poles


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
    sys : StateSpace or TransferFunction
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
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> fresp = ct.evalfr(G, 1j)  # evaluate at s = 1j

    .. todo:: Add example with MIMO system

    """
    return sys(x, squeeze=squeeze)


def frequency_response(
        sysdata, omega=None, omega_limits=None, omega_num=None,
        Hz=None, squeeze=None):
    """Frequency response of an LTI system at multiple angular frequencies.

    In general the system may be multiple input, multiple output (MIMO), where
    `m = sys.ninputs` number of inputs and `p = sys.noutputs` number of
    outputs.

    Parameters
    ----------
    sysdata : LTI system or list of LTI systems
        Linear system(s) for which frequency response is computed.
    omega : float or 1D array_like, optional
        Frequencies in radians/sec at which the system should be
        evaluated. Can be a single frequency or array of frequencies, which
        will be sorted before evaluation.  If None (default), a common set
        of frequencies that works across all given systems is computed.
    omega_limits : array_like of two values, optional
        Limits to the range of frequencies, in rad/sec. Specifying
        ``omega`` as a list of two elements is equivalent to providing
        ``omega_limits``.  Ignored if omega is provided.
    omega_num : int, optional
        Number of frequency samples at which to compute the response.
        Defaults to config.defaults['freqplot.number_of_samples'].  Ignored
        if omega is provided.

    Returns
    -------
    response : :class:`FrequencyResponseData`
        Frequency response data object representing the frequency response.
        This object can be assigned to a tuple using

            mag, phase, omega = response

        where ``mag`` is the magnitude (absolute value, not dB or log10) of
        the system frequency response, ``phase`` is the wrapped phase in
        radians of the system frequency response, and ``omega`` is the
        (sorted) frequencies at which the response was evaluated.  If the
        system is SISO and squeeze is not False, ``magnitude`` and ``phase``
        are 1D, indexed by frequency.  If the system is not SISO or squeeze
        is False, the array is 3D, indexed by the output, input, and
        frequency.  If ``squeeze`` is True then single-dimensional axes are
        removed.

        Returns a list of :class:`FrequencyResponseData` objects if sys is
        a list of systems.

    Other Parameters
    ----------------
    Hz : bool, optional
        If True, when computing frequency limits automatically set
        limits to full decades in Hz instead of rad/s. Omega is always
        returned in rad/sec.
    squeeze : bool, optional
        If squeeze=True, remove single-dimensional entries from the shape of
        the output even if the system is not SISO. If squeeze=False, keep all
        indices (output, input and, if omega is array_like, frequency) even if
        the system is SISO. The default value can be set using
        config.defaults['control.squeeze_frequency_response'].

    See Also
    --------
    evalfr
    bode_plot

    Notes
    -----
    1. This function is a wrapper for :meth:`StateSpace.frequency_response`
       and :meth:`TransferFunction.frequency_response`.

    2. You can also use the lower-level methods ``sys(s)`` or ``sys(z)`` to
       generate the frequency response for a single system.

    3. All frequency data should be given in rad/sec.  If frequency limits
       are computed automatically, the `Hz` keyword can be used to ensure
       that limits are in factors of decades in Hz, so that Bode plots with
       `Hz=True` look better.

    4. The frequency response data can be plotted by calling the
       :func:`~control_bode_plot` function or using the `plot` method of
       the :class:`~control.FrequencyResponseData` class.

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> mag, phase, omega = ct.freqresp(G, [0.1, 1., 10.])

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
    from .frdata import FrequencyResponseData
    from .freqplot import _determine_omega_vector

    # Process keyword arguments
    omega_num = config._get_param('freqplot', 'number_of_samples', omega_num)

    # Convert the first argument to a list
    syslist = sysdata if isinstance(sysdata, (list, tuple)) else [sysdata]

    # Get the common set of frequencies to use
    omega_syslist, omega_range_given = _determine_omega_vector(
        syslist, omega, omega_limits, omega_num, Hz=Hz)

    responses = []
    for sys_ in syslist:
        if isinstance(sys_, FrequencyResponseData) and sys_.ifunc is None and \
           not omega_range_given:
            omega_sys = sys_.omega              # use system properties
        else:
            omega_sys = omega_syslist.copy()    # use common omega vector

            # Add the Nyquist frequency for discrete time systems
            if sys_.isdtime(strict=True):
                nyquistfrq = math.pi / sys_.dt
                if not omega_range_given:
                    # Limit up to the Nyquist frequency
                    omega_sys = omega_sys[omega_sys < nyquistfrq]

        # Compute the frequency response
        responses.append(sys_.frequency_response(omega_sys, squeeze=squeeze))

    if isinstance(sysdata, (list, tuple)):
        from .freqplot import FrequencyResponseList
        return FrequencyResponseList(responses)
    else:
        return responses[0]

# Alternative name (legacy)
def freqresp(sys, omega):
    """Legacy version of frequency_response.

    .. deprecated:: 0.9.0
        This function will be removed in a future version of python-control.
        Use `frequency_response` instead.

    """
    warn("freqresp() is deprecated; use frequency_response()", FutureWarning)
    return frequency_response(sys, omega)


def dcgain(sys):
    """Return the zero-frequency (or DC) gain of the given system.

    Parameters
    ----------
    sys : LTI
        System for which the zero-frequency gain is computed.

    Returns
    -------
    gain : ndarray
        The zero-frequency gain, or (inf + nanj) if the system has a pole at
        the origin, (nan + nanj) if there is a pole/zero cancellation at the
        origin.

    Examples
    --------
    >>> G = ct.tf([1], [1, 2])
    >>> ct.dcgain(G)                                            # doctest: +SKIP
    np.float(0.5)

    """
    return sys.dcgain()


def bandwidth(sys, dbdrop=-3):
    """Return the first freqency where the gain drop by dbdrop of the system.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        Linear system for which the bandwidth should be computed.
    dbdrop : float, optional
        By how much the gain drop in dB (default = -3) that defines the
        bandwidth. Should be a negative scalar.

    Returns
    -------
    bandwidth : ndarray
        The first frequency (rad/time-unit) where the gain drops below dbdrop
        of the dc gain of the system, or nan if the system has infinite dc
        gain, inf if the gain does not drop for all frequency

    Raises
    ------
    TypeError
        if 'sys' is not an SISO LTI instance
    ValueError
        if 'dbdrop' is not a negative scalar

    Example
    -------
    >>> G = ct.tf([1], [1, 1])
    >>> ct.bandwidth(G)
    0.9976

    >>> G1 = ct.tf(0.1, [1, 0.1])
    >>> wn2 = 1
    >>> zeta2 = 0.001
    >>> G2 = ct.tf(wn2**2, [1, 2*zeta2*wn2, wn2**2])
    >>> ct.bandwidth(G1*G2)
    0.1018

    """
    if not isinstance(sys, LTI):
        raise TypeError("sys must be a LTI instance.")

    return sys.bandwidth(dbdrop)


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
