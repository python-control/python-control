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

from scipy import pi

# Utility function to unwrap an angle measurement
def unwrap(angle, period=2*pi):
    """Unwrap a phase angle to give a continuous curve

    Usage: Y = unwrap(X, period=2*pi)
    
    Parameters
    ----------
    X : array_like
        Input array
    period : number
        Input period (usually either 2*pi or 360)

    Returns
    -------
    Y : array_like
        Output array, with jumps of period/2 eliminated
    """
    wrap = 0;
    last = angle[0];
    for k in range(len(angle)):
        # See if we need to account for angle wrapping
        if (angle[k] - last > period/2):
            wrap = -period
        elif (last - angle[k] > period/2):
            wrap = period

        # Update the last value we have sene
        last = angle[k]

        # Add in the wrap angle if nonzer
        if (wrap != 0):
            angle[k] += wrap;

    # return the updated list
    return angle
