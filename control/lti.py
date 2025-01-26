# lti.py - LTI class and functions for linear systems

"""LTI class and functions for linear systems.

This module contains the LTI parent class to the child classes
StateSpace and TransferFunction.

"""

import math
from warnings import warn

import numpy as np
from numpy import abs, real

from . import config
from .iosys import InputOutputSystem

__all__ = ['poles', 'zeros', 'damp', 'evalfr', 'frequency_response',
           'freqresp', 'dcgain', 'bandwidth', 'LTI']


class LTI(InputOutputSystem):
    """Parent class for linear time-invariant system objects.

    LTI is the parent to the `FrequencyResponseData`, `StateSpace`, and
    `TransferFunction` child classes. It contains the number of inputs and
    outputs, and the timebase (dt) for the system.  This class is not
    generally accessed directly by the user.

    See Also
    --------
    InputOutputSystem, StateSpace, TransferFunction, FrequencyResponseData

    """
    def __init__(self, inputs=1, outputs=1, states=None, name=None, **kwargs):
        """Assign the LTI object's numbers of inputs and outputs."""
        super().__init__(
            name=name, inputs=inputs, outputs=outputs, states=states, **kwargs)

    def __call__(self, x, squeeze=None, warn_infinite=True):
        """Evaluate system transfer function at point in complex plane.

        Returns the value of the system's transfer function at a point `x`
        in the complex plane, where `x` is `s` for continuous-time systems
        and `z` for discrete-time systems.

        By default, a (complex) scalar will be returned for SISO systems
        and a p x m array will be return for MIMO systems with m inputs and
        p outputs.  This can be changed using the `squeeze` keyword.

        To evaluate at a frequency `omega` in radians per second,
        enter ``x = omega * 1j`` for continuous-time systems,
        ``x = exp(1j * omega * dt)`` for discrete-time systems, or
        use the `~LTI.frequency_response` method.

        Parameters
        ----------
        x : complex or complex 1D array_like
            Complex value(s) at which transfer function will be evaluated.
        squeeze : bool, optional
            Squeeze output, as described below.  Default value can be set
            using `config.defaults['control.squeeze_frequency_response']`.
        warn_infinite : bool, optional
            If set to False, turn off divide by zero warning.

        Returns
        -------
        fresp : complex ndarray
            The value of the system transfer function at `x`.  If the system
            is SISO and `squeeze` is not True, the shape of the array matches
            the shape of `x`.  If the system is not SISO or `squeeze` is
            False, the first two dimensions of the array are indices for the
            output and input and the remaining dimensions match `x`.  If
            `squeeze` is True then single-dimensional axes are removed.

        Notes
        -----
        See `FrequencyResponseData.__call__`, `StateSpace.__call__`,
        `TransferFunction.__call__` for class-specific details.

        """
        raise NotImplementedError("not implemented in subclass")

    def damp(self):
        """Natural frequency, damping ratio of system poles.

        Returns
        -------
        wn : array
            Natural frequency for each system pole.
        zeta : array
            Damping ratio for each system pole.
        poles : array
            System pole locations.
        """
        poles = self.poles()

        if self.isdtime(strict=True):
            splane_poles = np.log(poles.astype(complex))/self.dt
        else:
            splane_poles = poles
        wn = abs(splane_poles)
        zeta = -real(splane_poles)/wn
        return wn, zeta, poles

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two input/output systems.

        Parameters
        ----------
        other : `InputOutputSystem`
            System in the feedback path.

        sign : float, optional
            Gain to use in feedback path.  Defaults to -1.

        """
        raise NotImplementedError("feedback not implemented in subclass")

    def frequency_response(self, omega=None, squeeze=None):
        """Evaluate LTI system response at an array of frequencies.

        See `frequency_response` for more detailed information.

        """
        from .frdata import FrequencyResponseData

        if omega is None:
            # Use default frequency range
            from .freqplot import _default_frequency_range
            omega = _default_frequency_range(self)

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
        """Return the zero-frequency (DC) gain."""
        raise NotImplementedError("dcgain not defined for subclass")

    def _dcgain(self, warn_infinite):
        zeroresp = self(0 if self.isctime() else 1,
                        warn_infinite=warn_infinite)
        if np.all(np.logical_or(np.isreal(zeroresp), np.isnan(zeroresp.imag))):
            return zeroresp.real
        else:
            return zeroresp

    def bandwidth(self, dbdrop=-3):
        """Evaluate bandwidth of an LTI system for a given dB drop.

        Evaluate the first frequency that the response magnitude is lower than
        DC gain by `dbdrop` dB.

        Parameters
        ----------
        dbdrop : float, optional
            A strictly negative scalar in dB (default = -3) defines the
            amount of gain drop for deciding bandwidth.

        Returns
        -------
        bandwidth : ndarray
            The first frequency (rad/time-unit) where the gain drops below
            `dbdrop` of the dc gain of the system, or nan if the system has
            infinite dc gain, inf if the gain does not drop for all frequency.

        Raises
        ------
        TypeError
            If `sys` is not an SISO LTI instance.
        ValueError
            If `dbdrop` is not a negative scalar.

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
        r"""Indicate if a linear time invariant (LTI) system is passive.

        See `ispassive` for details.

        """
        # importing here prevents circular dependency
        from control.passivity import ispassive
        return ispassive(self)

    #
    # Convenience aliases for conversion functions
    #
    # Allow conversion between state space and transfer function types
    # as methods.  These are just pass throughs to factory functions.
    #
    # Note: in order for docstrings to created, these have to set these up
    # as independent methods, not just assigned to ss() and tf().
    #
    # Imports are done within the function to avoid circular imports.
    #
    def to_ss(self, *args, **kwargs):
        """Convert to state space representation.

        See `ss` for details.
        """
        from .statesp import ss
        return ss(self, *args, **kwargs)

    def to_tf(self, *args, **kwargs):
        """Convert to transfer function representation.

        See `tf` for details.
        """
        from .xferfcn import tf
        return tf(self, *args, **kwargs)

    #
    # Convenience aliases for plotting and response functions
    #
    # Allow standard plots to be generated directly from the system object
    # in addition to standalone plotting and response functions.
    #
    # Note: in order for docstrings to created, these have to set these up as
    # independent methods, not just assigned to plotting/response functions.
    #
    # Imports are done within the function to avoid circular imports.
    #

    def bode_plot(self, *args, **kwargs):
        """Generate a Bode plot for the system.

        See `bode_plot` for more information.
        """
        from .freqplot import bode_plot
        return bode_plot(self, *args, **kwargs)

    def nichols_plot(self, *args, **kwargs):
        """Generate a Nichols plot for the system.

        See `nichols_plot` for more information.
        """
        from .nichols import nichols_plot
        return nichols_plot(self, *args, **kwargs)

    def nyquist_plot(self, *args, **kwargs):
        """Generate a Nyquist plot for the system.

        See `nyquist_plot` for more information.
        """
        from .freqplot import nyquist_plot
        return nyquist_plot(self, *args, **kwargs)

    def forced_response(self, *args, **kwargs):
        """Generate the forced response for the system.

        See `forced_response` for more information.
        """
        from .timeresp import forced_response
        return forced_response(self, *args, **kwargs)

    def impulse_response(self, *args, **kwargs):
        """Generate the impulse response for the system.

        See `impulse_response` for more information.
        """
        from .timeresp import impulse_response
        return impulse_response(self, *args, **kwargs)

    def initial_response(self, *args, **kwargs):
        """Generate the initial response for the system.

        See `initial_response` for more information.
        """
        from .timeresp import initial_response
        return initial_response(self, *args, **kwargs)

    def step_response(self, *args, **kwargs):
        """Generate the step response for the system.

        See `step_response` for more information.
        """
        from .timeresp import step_response
        return step_response(self, *args, **kwargs)


def poles(sys):
    """
    Compute system poles.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear system.

    Returns
    -------
    poles : ndarray
        Array that contains the system's poles.

    See Also
    --------
    zeros, StateSpace.poles, TransferFunction.poles

    """

    return sys.poles()


def zeros(sys):
    """
    Compute system zeros.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear system.

    Returns
    -------
    zeros : ndarray
        Array that contains the system's zeros.

    See Also
    --------
    poles, StateSpace.zeros, TransferFunction.zeros

    """

    return sys.zeros()


def damp(sys, doprint=True):
    """Compute system's natural frequencies, damping ratios, and poles.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        A linear system object.
    doprint : bool (optional)
        If True, print table with values.

    Returns
    -------
    wn : array
        Natural frequency for each system pole.
    zeta : array
        Damping ratio for each system pole.
    poles : array
        System pole locations.

    See Also
    --------
    poles

    Notes
    -----
    If the system is continuous

        | ``wn = abs(poles)``
        | ``zeta  = -real(poles)/poles``

    If the system is discrete, the discrete poles are mapped to their
    equivalent location in the s-plane via

        | ``s = log(poles)/dt``

    and

        | ``wn = abs(s)``
        | ``zeta = -real(s)/wn``

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


# TODO: deprecate this function
def evalfr(sys, x, squeeze=None):
    """Evaluate transfer function of LTI system at complex frequency.

    Returns the complex frequency response ``sys(x)`` where `x` is `s` for
    continuous-time systems and `z` for discrete-time systems, with
    ``m = sys.ninputs`` number of inputs and ``p = sys.noutputs`` number of
    outputs.

    To evaluate at a frequency omega in radians per second, enter
    ``x = omega * 1j`` for continuous-time systems, or
    ``x = exp(1j * omega * dt)`` for discrete-time systems, or use
    ``freqresp(sys, omega)``.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear system.
    x : complex scalar or 1D array_like
        Complex frequency(s).
    squeeze : bool, optional (default=True)
        If `squeeze` = True, remove single-dimensional entries from the
        shape of the output even if the system is not SISO. If
        `squeeze` = False, keep all indices (output, input and, if omega is
        array_like, frequency) even if the system is SISO. The default
        value can be set using
        `config.defaults['control.squeeze_frequency_response']`.

    Returns
    -------
    fresp : complex ndarray
        The frequency response of the system.  If the system is SISO and
        squeeze is not True, the shape of the array matches the shape of
        omega.  If the system is not SISO or squeeze is False, the first two
        dimensions of the array are indices for the output and input and the
        remaining dimensions match omega.  If `squeeze` is True then
        single-dimensional axes are removed.

    See Also
    --------
    LTI.__call__, frequency_response, bode_plot

    Notes
    -----
    This function is a wrapper for `StateSpace.__call__` and
    `TransferFunction.__call__`.

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> fresp = ct.evalfr(G, 1j)  # evaluate at s = 1j

    """
    return sys(x, squeeze=squeeze)


def frequency_response(
        sysdata, omega=None, omega_limits=None, omega_num=None,
        Hz=None, squeeze=None):
    """Frequency response of an LTI system.

    For continuous-time systems with transfer function G, computes the
    frequency response as

         G(j*omega) = mag * exp(j*phase)

    For discrete-time systems, the response is evaluated around the unit
    circle such that

         G(exp(j*omega*dt)) = mag * exp(j*phase).

    In general the system may be multiple input, multiple output (MIMO),
    where ``m = self.ninputs`` number of inputs and ``p = self.noutputs``
    number of outputs.

    Parameters
    ----------
    sysdata : LTI system or list of LTI systems
        Linear system(s) for which frequency response is computed.
    omega : float or 1D array_like, optional
        A list, tuple, array, or scalar value of frequencies in radians/sec
        at which the system will be evaluated.  Can be a single frequency
        or array of frequencies, which will be sorted before evaluation.
        If None (default), a common set of frequencies that works across
        all given systems is computed.
    omega_limits : array_like of two values, optional
        Limits to the range of frequencies, in rad/sec. Specifying
        `omega` as a list of two elements is equivalent to providing
        `omega_limits`.  Ignored if omega is provided.
    omega_num : int, optional
        Number of frequency samples at which to compute the response.
        Defaults to `config.defaults['freqplot.number_of_samples']`.  Ignored
        if omega is provided.

    Returns
    -------
    response : `FrequencyResponseData`
        Frequency response data object representing the frequency
        response.  When accessed as a tuple, returns ``(magnitude,
        phase, omega)``.  If `sysdata` is a list of systems, returns a
        `FrequencyResponseList` object.  Results can be plotted using
        the `~FrequencyResponseData.plot` method.  See
        `FrequencyResponseData` for more detailed information.
    response.magnitude : array
        Magnitude of the frequency response (absolute value, not dB or
        log10).  If the system is SISO and squeeze is not True, the
        array is 1D, indexed by frequency.  If the system is not SISO
        or squeeze is False, the array is 3D, indexed by the output,
        input, and, if omega is array_like, frequency.  If `squeeze` is
        True then single-dimensional axes are removed.
    response.phase : array
        Wrapped phase, in radians, with same shape as `magnitude`.
    response.omega : array
        Sorted list of frequencies at which response was evaluated.

    Other Parameters
    ----------------
    Hz : bool, optional
        If True, when computing frequency limits automatically set
        limits to full decades in Hz instead of rad/s. Omega is always
        returned in rad/sec.
    squeeze : bool, optional
        If `squeeze` = True, remove single-dimensional entries from the
        shape of the output even if the system is not SISO. If
        `squeeze` = False, keep all indices (output, input and, if omega is
        array_like, frequency) even if the system is SISO. The default
        value can be set using
        `config.defaults['control.squeeze_frequency_response']`.

    See Also
    --------
    LTI.__call__, bode_plot

    Notes
    -----
    This function is a wrapper for `StateSpace.frequency_response` and
    `TransferFunction.frequency_response`.  You can also use the
    lower-level methods ``sys(s)`` or ``sys(z)`` to generate the frequency
    response for a single system.

    All frequency data should be given in rad/sec.  If frequency limits are
    computed automatically, the `Hz` keyword can be used to ensure that
    limits are in factors of decades in Hz, so that Bode plots with
    `Hz` = True look better.

    The frequency response data can be plotted by calling the `bode_plot`
    function or using the `plot` method of the `FrequencyResponseData`
    class.

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> mag, phase, omega = ct.frequency_response(G, [0.1, 1., 10.])

    >>> sys = ct.rss(3, 2, 2)
    >>> mag, phase, omega = ct.frequency_response(sys, [0.1, 1., 10.])
    >>> mag[0, 1, :]    # Magnitude of second input to first output
    array([..., ..., ...])
    >>> phase[1, 0, :]  # Phase of first input to second output
    array([..., ..., ...])

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
        if isinstance(sys_, FrequencyResponseData) and sys_._ifunc is None \
           and not omega_range_given:
            omega_sys = sys_.omega              # use system properties
        else:
            omega_sys = omega_syslist.copy()    # use common omega vector

            # Add the Nyquist frequency for discrete-time systems
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
    """Find first frequency where gain drops by 3 dB.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear system for which the bandwidth should be computed.
    dbdrop : float, optional
        By how much the gain drop in dB (default = -3) that defines the
        bandwidth. Should be a negative scalar.

    Returns
    -------
    bandwidth : ndarray
        The first frequency where the gain drops below `dbdrop` of the zero
        frequency (DC) gain of the system, or nan if the system has infinite
        zero frequency gain, inf if the gain does not drop for any frequency.

    Raises
    ------
    TypeError
        If `sys` is not an SISO LTI instance.
    ValueError
        If `dbdrop` is not a negative scalar.

    Examples
    --------
    >>> G = ct.tf([1], [1, 1])
    >>> ct.bandwidth(G)
    np.float64(0.9976283451102316)

    >>> G1 = ct.tf(0.1, [1, 0.1])
    >>> wn2 = 1
    >>> zeta2 = 0.001
    >>> G2 = ct.tf(wn2**2, [1, 2*zeta2*wn2, wn2**2])
    >>> ct.bandwidth(G1*G2)
    np.float64(0.10184838823897456)

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
