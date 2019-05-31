"""dtime.py

Functions for manipulating discrete time systems.

Routines in this module:

sample_system()
"""

"""Copyright (c) 2012 by California Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Richard M. Murray
Date: 6 October 2012

$Id: dtime.py 185 2012-08-30 05:44:32Z murrayrm $

"""

from .lti import isctime
from .statesp import StateSpace, _convertToStateSpace

__all__ = ['sample_system', 'c2d']

# Sample a continuous time system
def sample_system(sysc, Ts, method='zoh', alpha=None):
    """Convert a continuous time system to discrete time

    Creates a discrete time system from a continuous time system by
    sampling.  Multiple methods of conversion are supported.

    Parameters
    ----------
    sysc : linsys
        Continuous time system to be converted
    Ts : real
        Sampling period
    method : string
        Method to use for conversion: 'matched', 'tustin', 'zoh' (default)

    Returns
    -------
    sysd : linsys
        Discrete time system, with sampling rate Ts

    Notes
    -----
    See `TransferFunction.sample` and `StateSpace.sample` for
    further details.

    Examples
    --------
    >>> sysc = TransferFunction([1], [1, 2, 1])
    >>> sysd = sample_system(sysc, 1, method='matched')
    """

    # Make sure we have a continuous time system
    if not isctime(sysc):
        raise ValueError("First argument must be continuous time system")

    return sysc.sample(Ts, method, alpha)


def c2d(sysc, Ts, method='zoh'):
    '''
    Return a discrete-time system

    Parameters
    ----------
    sysc: LTI (StateSpace or TransferFunction), continuous
        System to be converted

    Ts: number
        Sample time for the conversion

    method: string, optional
        Method to be applied,
        'zoh'        Zero-order hold on the inputs (default)
        'foh'        First-order hold, currently not implemented
        'impulse'    Impulse-invariant discretization, currently not implemented
        'tustin'     Bilinear (Tustin) approximation, only SISO
        'matched'    Matched pole-zero method, only SISO
    '''
    #  Call the sample_system() function to do the work
    sysd = sample_system(sysc, Ts, method)

    # TODO: is this check needed?  If sysc is  StateSpace, sysd is too?
    if isinstance(sysc, StateSpace) and not isinstance(sysd, StateSpace):
        return _convertToStateSpace(sysd)       # pragma: no cover

    return sysd
