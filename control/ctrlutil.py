# ctrlutil.py - control system utility functions
#
# Author: Richard M. Murray
# Date: 24 May 09
#
# These are some basic utility functions that are used in the control
# systems library and that didn't naturally fit anyplace else.
#
# Copyright (c) 2009 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$

# Packages that we need access to
from . import lti
import numpy as np
import math

__all__ = ['unwrap', 'issys', 'db2mag', 'mag2db']

# Utility function to unwrap an angle measurement
def unwrap(angle, period=2*math.pi):
    """Unwrap a phase angle to give a continuous curve

    Parameters
    ----------
    angle : array_like
        Array of angles to be unwrapped
    period : float, optional
        Period (defaults to `2*pi`)

    Returns
    -------
    angle_out : array_like
        Output array, with jumps of period/2 eliminated

    Examples
    --------
    >>> import numpy as np
    >>> theta = [5.74, 5.97, 6.19, 0.13, 0.35, 0.57]
    >>> unwrap(theta, period=2 * np.pi)
    [5.74, 5.97, 6.19, 6.413185307179586, 6.633185307179586, 6.8531853071795865]

    """
    dangle = np.diff(angle)
    dangle_desired = (dangle + period/2.) % period - period/2.
    correction = np.cumsum(dangle_desired - dangle)
    angle[1:] += correction
    return angle

def issys(obj):
    """Return True if an object is a system, otherwise False"""
    return isinstance(obj, lti.LTI)

def db2mag(db):
    """Convert a gain in decibels (dB) to a magnitude

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    db : float or ndarray
        input value or array of values, given in decibels

    Returns
    -------
    mag : float or ndarray
        corresponding magnitudes

    """
    return 10. ** (db / 20.)

def mag2db(mag):
    """Convert a magnitude to decibels (dB)

    If A is magnitude,

        db = 20 * log10(A)

    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes

    Returns
    -------
    db : float or ndarray
        corresponding values in decibels

    """
    return 20. * np.log10(mag)
