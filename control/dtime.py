"""dtime.py

Functions for manipulating discrete time systems.

Routines in this module:

sample_system()
c2d()
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

from .namedio import isctime
from .statesp import StateSpace

__all__ = ['sample_system', 'c2d']

# Sample a continuous time system
def sample_system(sysc, Ts, method='zoh', alpha=None, prewarp_frequency=None,
        name=None, copy_names=True, **kwargs):
    """
    Convert a continuous time system to discrete time by sampling

    Parameters
    ----------
    sysc : LTI (:class:`StateSpace` or :class:`TransferFunction`)
        Continuous time system to be converted
    Ts : float > 0
        Sampling period
    method : string
        Method to use for conversion, e.g. 'bilinear', 'zoh' (default)
    alpha : float within [0, 1]
            The generalized bilinear transformation weighting parameter, which
            should only be specified with method="gbt", and is ignored
            otherwise. See :func:`scipy.signal.cont2discrete`.
    prewarp_frequency : float within [0, infinity)
        The frequency [rad/s] at which to match with the input continuous-
        time system's magnitude and phase (only valid for method='bilinear',
        'tustin', or 'gbt' with alpha=0.5)

    Returns
    -------
    sysd : LTI of the same class (:class:`StateSpace` or :class:`TransferFunction`)
        Discrete time system, with sampling rate Ts

    Other Parameters
    ----------------
    inputs : int, list of str or None, optional
        Description of the system inputs.  If not specified, the origional
        system inputs are used.  See :class:`InputOutputSystem` for more
        information.
    outputs : int, list of str or None, optional
        Description of the system outputs.  Same format as `inputs`.
    states : int, list of str, or None, optional
        Description of the system states.  Same format as `inputs`. Only
        available if the system is :class:`StateSpace`.
    name : string, optional
        Set the name of the sampled system.  If not specified and
        if `copy_names` is `False`, a generic name <sys[id]> is generated
        with a unique integer id.  If `copy_names` is `True`, the new system
        name is determined by adding the prefix and suffix strings in
        config.defaults['namedio.sampled_system_name_prefix'] and
        config.defaults['namedio.sampled_system_name_suffix'], with the
        default being to add the suffix '$sampled'.
    copy_names : bool, Optional
        If True, copy the names of the input signals, output
        signals, and states to the sampled system.

    Notes
    -----
    See :meth:`StateSpace.sample` or :meth:`TransferFunction.sample` for
    further details.

    Examples
    --------
    >>> Gc = ct.tf([1], [1, 2, 1])
    >>> Gc.isdtime()
    False
    >>> Gd = ct.sample_system(Gc, 1, method='bilinear')
    >>> Gd.isdtime()
    True

    """

    # Make sure we have a continuous time system
    if not isctime(sysc):
        raise ValueError("First argument must be continuous time system")

    return sysc.sample(Ts,
        method=method, alpha=alpha, prewarp_frequency=prewarp_frequency,
        name=name, copy_names=copy_names, **kwargs)

c2d = sample_system