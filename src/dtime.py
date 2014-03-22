"""dtime.py

Functions for manipulating discrete time systems.

Routines in this module:

sample_system()
_c2dmatched()
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

from scipy.signal import zpk2tf, tf2zpk
import numpy as np
from cmath import exp
from warnings import warn
from control.lti import isctime
from control.statesp import StateSpace, _convertToStateSpace
from control.xferfcn import TransferFunction, _convertToTransferFunction

# Sample a continuous time system
def sample_system(sysc, Ts, method='matched'):
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
        Method to use for conversion: 'matched' (default), 'tustin', 'zoh'

    Returns
    -------
    sysd : linsys
        Discrete time system, with sampling rate Ts

    Notes
    -----
    1. The conversion methods 'tustin' and 'zoh' require the
       cont2discrete() function, including in SciPy 0.10.0 and above.

    2. Additional methods 'foh' and 'impulse' are planned for future
       implementation.

    Examples
    --------
    >>> sysc = TransferFunction([1], [1, 2, 1])
    >>> sysd = sample_system(sysc, 1, method='matched')
    """

    # Make sure we have a continuous time system
    if not isctime(sysc):
        raise ValueError("First argument must be continuous time system")

    # TODO: impelement MIMO version
    if (sysc.inputs != 1 or sysc.outputs != 1):
        raise NotImplementedError("MIMO implementation not available")

    # If we are passed a state space system, convert to transfer function first
    if isinstance(sysc, StateSpace):
        warn("sample_system: converting to transfer function")
        sysc = _convertToTransferFunction(sysc)

    # Decide what to do based on the methods available
    if method == 'matched':
        sysd = _c2dmatched(sysc, Ts)

    elif method == 'tustin':
        try:
            from scipy.signal import cont2discrete
            sys = [sysc.num[0][0], sysc.den[0][0]]
            scipySysD = cont2discrete(sys, Ts, method='bilinear')
            sysd = TransferFunction(scipySysD[0][0], scipySysD[1], Ts)
        except ImportError:
            raise TypeError("cont2discrete not found in scipy.signal; upgrade to v0.10.0+")
        
    elif method == 'zoh':
        try:
            from scipy.signal import cont2discrete
            sys = [sysc.num[0][0], sysc.den[0][0]]
            scipySysD = cont2discrete(sys, Ts, method='zoh')
            sysd = TransferFunction(scipySysD[0][0],scipySysD[1], Ts)
        except ImportError:
            raise TypeError("cont2discrete not found in scipy.signal; upgrade to v0.10.0+")

    elif method == 'foh' or method == 'impulse':
        raise ValueError("Method not developed yet")

    else:
        raise ValueError("Invalid discretization method: %s" % method)

    # TODO: Convert back into the input form
    # Set sampling time
    return sysd

# c2d function contributed by Benjamin White, Oct 2012
def _c2dmatched(sysC, Ts):
    # Pole-zero match method of continuous to discrete time conversion
    szeros, spoles, sgain = tf2zpk(sysC.num[0][0], sysC.den[0][0])
    zzeros = [0] * len(szeros)
    zpoles = [0] * len(spoles)
    pregainnum = [0] * len(szeros)
    pregainden = [0] * len(spoles)
    for idx, s in enumerate(szeros):
        sTs = s*Ts
        z = exp(sTs)
        zzeros[idx] = z
        pregainnum[idx] = 1-z
    for idx, s in enumerate(spoles):
        sTs = s*Ts
        z = exp(sTs)
        zpoles[idx] = z
        pregainden[idx] = 1-z
    zgain = np.multiply.reduce(pregainnum)/np.multiply.reduce(pregainden)
    gain = sgain/zgain
    sysDnum, sysDden = zpk2tf(zzeros, zpoles, gain)
    return TransferFunction(sysDnum, sysDden, Ts)
