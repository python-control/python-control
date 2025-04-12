# wrappers.py - Wrappers for the MATLAB compatibility module.

"""Wrappers for the MATLAB compatibility module.

"""

import warnings
from warnings import warn

import numpy as np
from scipy.signal import zpk2tf

from ..exception import ControlArgument
from ..lti import LTI
from ..statesp import ss
from ..xferfcn import tf

__all__ = ['bode', 'nyquist', 'ngrid', 'rlocus', 'pzmap', 'dcgain', 'connect']

def bode(*args, **kwargs):
    """bode(sys[, omega, dB, Hz, deg, ...])

    Bode plot of the frequency response.

    Plots a bode gain and phase diagram.

    Parameters
    ----------
    sys : LTI, or list of LTI
        System for which the Bode response is plotted and give. Optionally
        a list of systems can be entered, or several systems can be
        specified (i.e. several parameters). The `sys` arguments may also be
        interspersed with format strings. A frequency argument (array_like)
        may also be added (see Examples).
    omega : array
        Range of frequencies in rad/s.
    dB : boolean
        If True, plot result in dB.
    Hz : boolean
        If True, plot frequency in Hz (omega must be provided in rad/sec).
    deg : boolean
        If True, return phase in degrees (else radians).
    plot : boolean
        If True, plot magnitude and phase.

    Returns
    -------
    mag, phase, omega : array
        Magnitude, phase, and frequencies represented in the Bode plot.

    Examples
    --------
    >>> from control.matlab import ss, bode

    >>> sys = ss([[1, -2], [3, -4]], [[5], [7]], [[6, 8]], 9)
    >>> mag, phase, omega = bode(sys)
    >>> bode(sys, w)                  # one system, freq vector # doctest: +SKIP
    >>> bode(sys1, sys2, ..., sysN)   # several systems         # doctest: +SKIP
    >>> bode(sys1, sys2, ..., sysN, w)                          # doctest: +SKIP
    >>> bode(sys1, 'plotstyle1', ..., sysN, 'plotstyleN')       # doctest: +SKIP

    """
    from ..freqplot import bode_plot

    # Use the plot keyword to get legacy behavior
    # TODO: update to call frequency_response and then bode_plot
    kwargs = dict(kwargs)       # make a copy since we modify this
    if 'plot' not in kwargs:
        kwargs['plot'] = True

    # Turn off deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='.* return values of .* is deprecated',
            category=DeprecationWarning)

        # If first argument is a list, assume python-control calling format
        if hasattr(args[0], '__iter__'):
            retval = bode_plot(*args, **kwargs)
        else:
            # Parse input arguments
            syslist, omega, args, other = _parse_freqplot_args(*args)
            kwargs.update(other)

            # Call the bode command
            retval = bode_plot(syslist, omega, *args, **kwargs)

    return retval


def nyquist(*args, plot=True, **kwargs):
    """nyquist(syslist[, omega])

    Nyquist plot of the frequency response.

    Plots a Nyquist plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK).
    omega : array_like
        Set of frequencies to be evaluated, in rad/sec.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`.
    plot : bool
        If False, do not generate a plot.

    Returns
    -------
    real : ndarray (or list of ndarray if len(syslist) > 1))
        Real part of the frequency response array.
    imag : ndarray (or list of ndarray if len(syslist) > 1))
        Imaginary part of the frequency response array.
    omega : ndarray (or list of ndarray if len(syslist) > 1))
        Frequencies in rad/s.

    """
    from ..freqplot import nyquist_plot, nyquist_response

    # If first argument is a list, assume python-control calling format
    if hasattr(args[0], '__iter__'):
        return nyquist_plot(*args, **kwargs)

    # Parse arguments
    syslist, omega, args, other = _parse_freqplot_args(*args)
    kwargs.update(other)

    # Get the Nyquist response (and pop keywords used there)
    response = nyquist_response(
        syslist, omega, *args, omega_limits=kwargs.pop('omega_limits', None))
    contour = response.contour
    if plot:
        # Plot the result
        nyquist_plot(response, *args, **kwargs)

    # Create the MATLAB output arguments
    freqresp = syslist(contour)
    real, imag = freqresp.real, freqresp.imag
    return real, imag, contour.imag


def _parse_freqplot_args(*args):
    """Parse arguments to frequency plot routines (bode, nyquist)"""
    syslist, plotstyle, omega, other = [], [], None, {}
    i = 0
    while i < len(args):
        # Check to see if this is a system of some sort
        if isinstance(args[i], LTI):
            # Append the system to our list of systems
            syslist.append(args[i])
            i += 1

            # See if the next object is a plotsytle (string)
            if (i < len(args) and isinstance(args[i], str)):
                plotstyle.append(args[i])
                i += 1

            # Go on to the next argument
            continue

        # See if this is a frequency list
        elif isinstance(args[i], (list, np.ndarray)):
            omega = args[i]
            i += 1
            break

        # See if this is a frequency range
        elif isinstance(args[i], tuple) and len(args[i]) == 2:
            other['omega_limits'] = args[i]
            i += 1

        else:
            raise ControlArgument("unrecognized argument type")

    # Check to make sure that we processed all arguments
    if (i < len(args)):
        raise ControlArgument("not all arguments processed")

    # Check to make sure we got the same number of plotstyles as systems
    if (len(plotstyle) != 0 and len(syslist) != len(plotstyle)):
        raise ControlArgument(
            "number of systems and plotstyles should be equal")

    # Warn about unimplemented plotstyles
    #! TODO: remove this when plot styles are implemented in bode()
    #! TODO: uncomment unit test code that tests this out
    if (len(plotstyle) != 0):
        warn("Warning (matlab.bode): plot styles not implemented");

    if len(syslist) == 0:
        raise ControlArgument("no systems specified")
    elif len(syslist) == 1:
    # If only one system given, return just that system (not a list)
        syslist = syslist[0]

    return syslist, omega, plotstyle, other


# TODO: rewrite to call root_locus_map, without using legacy plot keyword
def rlocus(*args, **kwargs):
    """rlocus(sys[, gains, xlim, ylim, ...])

    Root locus diagram.

    Calculate the root locus by finding the roots of 1 + k * G(s) where G
    is a linear system with transfer function num(s)/den(s) and each k is
    an element of gains.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems (SISO only, for now).
    gains : array_like, optional
        Gains to use in computing plot of closed-loop poles.
    xlim : tuple or list, optional
        Set limits of x axis (see `matplotlib.axes.Axes.set_xlim`).
    ylim : tuple or list, optional
        Set limits of y axis (see `matplotlib.axes.Axes.set_ylim`).
    plot : bool
        If False, do not generate a plot.

    Returns
    -------
    roots : ndarray
        Closed-loop root locations, arranged in which each row corresponds
        to a gain in gains.
    gains : ndarray
        Gains used.  Same as gains keyword argument if provided.

    Notes
    -----
    This function is a wrapper for `root_locus_plot`,
    with legacy return arguments.

    """
    from ..rlocus import root_locus_plot

    # Use the plot keyword to get legacy behavior
    kwargs = dict(kwargs)       # make a copy since we modify this
    if 'plot' not in kwargs:
        kwargs['plot'] = True

    # Turn off deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='.* return values of .* is deprecated',
            category=DeprecationWarning)
        retval = root_locus_plot(*args, **kwargs)

    return retval


# TODO: rewrite to call pole_zero_map, without using legacy plot keyword
def pzmap(*args, **kwargs):
    """pzmap(sys[, grid, plot])

    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear system for which poles and zeros are computed.
    plot : bool, optional
        If True a graph is generated with matplotlib,
        otherwise the poles and zeros are only computed and returned.
    grid : boolean (default = False)
        If True, plot omega-damping grid.

    Returns
    -------
    poles : array
        The system's poles.
    zeros : array
        The system's zeros.

    Notes
    -----
    This function is a wrapper for `pole_zero_plot`,
    with legacy return arguments.

    """
    from ..pzmap import pole_zero_plot

    # Use the plot keyword to get legacy behavior
    kwargs = dict(kwargs)       # make a copy since we modify this
    if 'plot' not in kwargs:
        kwargs['plot'] = True

    # Turn off deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='.* return values of .* is deprecated',
            category=DeprecationWarning)
        retval = pole_zero_plot(*args, **kwargs)

    return retval


from ..nichols import nichols_grid


def ngrid():
    return nichols_grid()
ngrid.__doc__ = nichols_grid.__doc__


def dcgain(*args):
    """dcgain(sys) \
      dcgain(num, den) \
      dcgain(Z, P, k) \
      dcgain(A, B, C, D)

    Compute the gain of the system in steady state.

    The function takes either 1, 2, 3, or 4 parameters:

      * dcgain(sys)
      * dcgain(num, den)
      * dcgain(Z, P, k)
      * dcgain(A, B, C, D)

    Parameters
    ----------
    A, B, C, D : array_like
        A linear system in state space form.
    Z, P, k : array_like, array_like, number
        A linear system in zero, pole, gain form.
    num, den : array_like
        A linear system in transfer function form.
    sys : `StateSpace` or `TransferFunction`
        A linear system object.

    Returns
    -------
    gain : ndarray
        The gain of each output versus each input:
        :math:`y = gain \\cdot u`.

    Notes
    -----
    This function is only useful for systems with invertible system
    matrix `A`.

    All systems are first converted to state space form. The function then
    computes:

    .. math:: gain = - C \\cdot A^{-1} \\cdot B + D
    """
    #Convert the parameters to state space form
    if len(args) == 4:
        A, B, C, D = args
        return ss(A, B, C, D).dcgain()
    elif len(args) == 3:
        Z, P, k = args
        num, den = zpk2tf(Z, P, k)
        return tf(num, den).dcgain()
    elif len(args) == 2:
        num, den = args
        return tf(num, den).dcgain()
    elif len(args) == 1:
        sys, = args
        return sys.dcgain()
    else:
        raise ValueError("Function `dcgain` needs either 1, 2, 3 or 4 "
                         "arguments.")


from ..bdalg import connect as ct_connect


def connect(*args):
    """connect(sys, Q, inputv, outputv)

    Index-based interconnection of an LTI system.

    The system `sys` is a system typically constructed with `append`, with
    multiple inputs and outputs.  The inputs and outputs are connected
    according to the interconnection matrix `Q`, and then the final inputs
    and outputs are trimmed according to the inputs and outputs listed in
    `inputv` and `outputv`.

    NOTE: Inputs and outputs are indexed starting at 1 and negative values
    correspond to a negative feedback interconnection.

    Parameters
    ----------
    sys : `InputOutputSystem`
        System to be connected.
    Q : 2D array
        Interconnection matrix. First column gives the input to be connected.
        The second column gives the index of an output that is to be fed into
        that input. Each additional column gives the index of an additional
        input that may be optionally added to that input. Negative values
        mean the feedback is negative. A zero value is ignored. Inputs
        and outputs are indexed starting at 1 to communicate sign information.
    inputv : 1D array
        List of final external inputs, indexed starting at 1.
    outputv : 1D array
        List of final external outputs, indexed starting at 1.

    Returns
    -------
    out : `InputOutputSystem`
        Connected and trimmed I/O system.

    See Also
    --------
    append, feedback, connect, negate, parallel, series

    Examples
    --------
    >>> G = ct.rss(7, inputs=2, outputs=2)
    >>> K = [[1, 2], [2, -1]]  # negative feedback interconnection
    >>> T = ct.connect(G, K, [2], [1, 2])
    >>> T.ninputs, T.noutputs, T.nstates
    (1, 2, 7)

    """
    # Turn off the deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="`connect` is deprecated")
        return ct_connect(*args)
