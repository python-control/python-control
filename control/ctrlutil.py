# ctrlutil.py - control system utility functions
#
# Initial author: Richard M. Murray
# Creation date: 24 May 2009
# Use `git shortlog -n -s ctrlutil.py` for full list of contributors

"""Control system utility functions."""

import math
import warnings

import numpy as np

from .lti import LTI

__all__ = ['unwrap', 'issys', 'db2mag', 'mag2db']

# Utility function to unwrap an angle measurement
def unwrap(angle, period=2*math.pi):
    """Unwrap a phase angle to give a continuous curve.

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped.
    period : float, optional
        Period (defaults to 2 pi).

    Returns
    -------
    angle_out : ndarray
        Output array, with jumps of period/2 eliminated.

    Examples
    --------
    >>> # Already continuous
    >>> theta1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0]) * np.pi
    >>> theta2 = ct.unwrap(theta1)
    >>> theta2/np.pi                                            # doctest: +SKIP
    array([1. , 1.5, 2. , 2.5, 3. ])

    >>> # Wrapped, discontinuous
    >>> theta1 = np.array([1.0, 1.5, 0.0, 0.5, 1.0]) * np.pi
    >>> theta2 = ct.unwrap(theta1)
    >>> theta2/np.pi                                            # doctest: +SKIP
    array([1. , 1.5, 2. , 2.5, 3. ])

    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period/2.) % period - period/2.
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    return angle

def issys(obj):
    """Deprecated function to check if an object is an LTI system.

    .. deprecated:: 0.10.0
        Use isinstance(obj, ct.LTI)

    """
    warnings.warn("issys() is deprecated; use isinstance(obj, ct.LTI)",
                  FutureWarning, stacklevel=2)
    return isinstance(obj, LTI)

def db2mag(db):
    """Convert a gain in decibels (dB) to a magnitude.

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    db : float or ndarray
        Input value or array of values, given in decibels.

    Returns
    -------
    mag : float or ndarray
        Corresponding magnitudes.

    Examples
    --------
    >>> ct.db2mag(-40.0)                                        # doctest: +SKIP
    0.01

    >>> ct.db2mag(np.array([0, -20]))                           # doctest: +SKIP
    array([1. , 0.1])

    """
    return 10. ** (db / 20.)

def mag2db(mag):
    """Convert a magnitude to decibels (dB).

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    mag : float or ndarray
        Input magnitude or array of magnitudes.

    Returns
    -------
    db : float or ndarray
        Corresponding values in decibels.

    Examples
    --------
    >>> ct.mag2db(10.0)                                         # doctest: +SKIP
    20.0

    >>> ct.mag2db(np.array([1, 0.01]))                          # doctest: +SKIP
    array([  0., -40.])

    """
    return 20. * np.log10(mag)
