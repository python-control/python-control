"""
Plotting routines for the Matlab compatibility module
"""

import numpy as np

__all__ = ['bode', 'ngrid']

def bode(*args, **keywords):
    """Bode plot of the frequency response

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
    Plot : boolean
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

    # If the first argument is a list, then assume python-control calling format
    from ..freqplot import bode as bode_orig
    if (getattr(args[0], '__iter__', False)):
        return bode_orig(*args, **keywords)

    # Otherwise, run through the arguments and collect up arguments
    syslist = []; plotstyle=[]; omega=None;
    i = 0;
    while i < len(args):
        # Check to see if this is a system of some sort
        from ..ctrlutil import issys
        if (issys(args[i])):
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
        elif (isinstance(args[i], (list, np.ndarray))):
            omega = args[i]
            i += 1
            break

        else:
            raise ControlArgument("unrecognized argument type")

    # Check to make sure that we processed all arguments
    if (i < len(args)):
        raise ControlArgument("not all arguments processed")

    # Check to make sure we got the same number of plotstyles as systems
    if (len(plotstyle) != 0 and len(syslist) != len(plotstyle)):
        raise ControlArgument("number of systems and plotstyles should be equal")

    # Warn about unimplemented plotstyles
    #! TODO: remove this when plot styles are implemented in bode()
    #! TODO: uncomment unit test code that tests this out
    if (len(plotstyle) != 0):
        print("Warning (matlab.bode): plot styles not implemented");

    # Call the bode command
    return bode_orig(syslist, omega, **keywords)

from ..nichols import nichols_grid
def ngrid():
    return nichols_grid()
ngrid.__doc__ = nichols_grid.__doc__
