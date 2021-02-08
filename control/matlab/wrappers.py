"""
Wrappers for the MATLAB compatibility module
"""

import numpy as np
from ..statesp import ss
from ..xferfcn import tf
from ..ctrlutil import issys
from ..exception import ControlArgument
from scipy.signal import zpk2tf
from warnings import warn

__all__ = ['bode', 'nyquist', 'ngrid', 'dcgain']

def bode(*args, **kwargs):
    """bode(syslist[, omega, dB, Hz, deg, ...])

    Bode plot of the frequency response

    Plots a bode gain and phase diagram

    Parameters
    ----------
    sys : LTI, or list of LTI
        System for which the Bode response is plotted and give. Optionally
        a list of systems can be entered, or several systems can be
        specified (i.e. several parameters). The sys arguments may also be
        interspersed with format strings. A frequency argument (array_like)
        may also be added, some examples:
        * >>> bode(sys, w)                    # one system, freq vector
        * >>> bode(sys1, sys2, ..., sysN)     # several systems
        * >>> bode(sys1, sys2, ..., sysN, w)
        * >>> bode(sys1, 'plotstyle1', ..., sysN, 'plotstyleN') # + plot formats
    omega: freq_range
        Range of frequencies in rad/s
    dB : boolean
        If True, plot result in dB
    Hz : boolean
        If True, plot frequency in Hz (omega must be provided in rad/sec)
    deg : boolean
        If True, return phase in degrees (else radians)
    plot : boolean
        If True, plot magnitude and phase

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = bode(sys)

    .. todo::

        Document these use cases

        * >>> bode(sys, w)
        * >>> bode(sys1, sys2, ..., sysN)
        * >>> bode(sys1, sys2, ..., sysN, w)
        * >>> bode(sys1, 'plotstyle1', ..., sysN, 'plotstyleN')
    """
    from ..freqplot import bode_plot

    # If first argument is a list, assume python-control calling format
    if hasattr(args[0], '__iter__'):
        return bode_plot(*args, **kwargs)

    # Parse input arguments
    syslist, omega, args, other = _parse_freqplot_args(*args)
    kwargs.update(other)

    # Call the bode command
    return bode_plot(syslist, omega, *args, **kwargs)


def nyquist(*args, **kwargs):
    """nyquist(syslist[, omega])

    Nyquist plot of the frequency response

    Plots a Nyquist plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys1, ..., sysn : list of LTI
        List of linear input/output systems (single system is OK).
    omega : array_like
        Set of frequencies to be evaluated, in rad/sec.

    Returns
    -------
    real : ndarray (or list of ndarray if len(syslist) > 1))
        real part of the frequency response array
    imag : ndarray (or list of ndarray if len(syslist) > 1))
        imaginary part of the frequency response array
    omega : ndarray (or list of ndarray if len(syslist) > 1))
        frequencies in rad/s

    """
    from ..freqplot import nyquist_plot

    # If first argument is a list, assume python-control calling format
    if hasattr(args[0], '__iter__'):
        return nyquist_plot(*args, **kwargs)

    # Parse arguments
    syslist, omega, args, other = _parse_freqplot_args(*args)
    kwargs.update(other)

    # Call the nyquist command
    kwargs['return_contour'] = True
    _, contour = nyquist_plot(syslist, omega, *args, **kwargs)

    # Create the MATLAB output arguments
    freqresp = syslist(contour)
    real, imag = freqresp.real, freqresp.imag
    return real, imag, contour.imag


def _parse_freqplot_args(*args):
    """Parse arguments to frequency plot routines (bode, nyquist)"""
    syslist, plotstyle, omega, other = [], [], None, {}
    i = 0;
    while i < len(args):
        # Check to see if this is a system of some sort
        if issys(args[i]):
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
    # If only one system given, retun just that system (not a list)
        syslist = syslist[0]

    return syslist, omega, plotstyle, other


from ..nichols import nichols_grid
def ngrid():
    return nichols_grid()
ngrid.__doc__ = nichols_grid.__doc__


def dcgain(*args):
    '''
    Compute the gain of the system in steady state.

    The function takes either 1, 2, 3, or 4 parameters:

    Parameters
    ----------
    A, B, C, D: array-like
        A linear system in state space form.
    Z, P, k: array-like, array-like, number
        A linear system in zero, pole, gain form.
    num, den: array-like
        A linear system in transfer function form.
    sys: LTI (StateSpace or TransferFunction)
        A linear system object.

    Returns
    -------
    gain: ndarray
        The gain of each output versus each input:
        :math:`y = gain \\cdot u`

    Notes
    -----
    This function is only useful for systems with invertible system
    matrix ``A``.

    All systems are first converted to state space form. The function then
    computes:

    .. math:: gain = - C \\cdot A^{-1} \\cdot B + D
    '''
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
        raise ValueError("Function ``dcgain`` needs either 1, 2, 3 or 4 "
                         "arguments.")
