"""bdalg.py

This file contains some standard block diagram algebra.

Routines in this module:

append
series
parallel
negate
feedback
connect

"""

"""Copyright (c) 2010 by California Institute of Technology
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
Date: 24 May 09
Revised: Kevin K. Chen, Dec 10

$Id$

"""

import numpy as np
from . import xferfcn
from . import statesp
from . import frdata

__all__ = ['series', 'parallel', 'negate', 'feedback', 'append', 'connect']


def series(sys1, *sysn):
    r"""series(sys1, sys2, [..., sysn])

    Return the series connection (`sysn` \* ...\  \*) `sys2` \* `sys1`.

    Parameters
    ----------
    sys1 : scalar, StateSpace, TransferFunction, or FRD
    *sysn : other scalars, StateSpaces, TransferFunctions, or FRDs

    Returns
    -------
    out : scalar, StateSpace, or TransferFunction

    Raises
    ------
    ValueError
        if `sys2.ninputs` does not equal `sys1.noutputs`
        if `sys1.dt` is not compatible with `sys2.dt`

    See Also
    --------
    parallel
    feedback

    Notes
    -----
    This function is a wrapper for the __mul__ function in the StateSpace and
    TransferFunction classes.  The output type is usually the type of `sys2`.
    If `sys2` is a scalar, then the output type is the type of `sys1`.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> from control import rss, series
    >>> sys_1 = rss(1)
    >>> sys_2 = rss(2)
    >>> sys_3 = series(sys_1, sys_2) # Same as sys3 = sys2 * sys1
    >>> sys_3.ninputs, sys_3.noutputs, sys_3.nstates
    (1, 1, 3)

    >>> sys_4 = rss(3)
    >>> sys_5 = series(sys_1, sys_2, sys_3, sys_4) # More systems
    >>> sys_5.ninputs, sys_5.noutputs, sys_5.nstates
    (1, 1, 9)

    """
    from functools import reduce
    return reduce(lambda x, y:y*x, sysn, sys1)


def parallel(sys1, *sysn):
    r"""parallel(sys1, sys2, [..., sysn])

    Return the parallel connection `sys1` + `sys2` (+ ...\  + `sysn`).

    Parameters
    ----------
    sys1 : scalar, StateSpace, TransferFunction, or FRD
    *sysn : other scalars, StateSpaces, TransferFunctions, or FRDs

    Returns
    -------
    out : scalar, StateSpace, or TransferFunction

    Raises
    ------
    ValueError
        if `sys1` and `sys2` do not have the same numbers of inputs and outputs

    See Also
    --------
    series
    feedback

    Notes
    -----
    This function is a wrapper for the __add__ function in the
    StateSpace and TransferFunction classes.  The output type is usually
    the type of `sys1`.  If `sys1` is a scalar, then the output type is
    the type of `sys2`.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> from control import parallel, rss
    >>> sys_1 = rss(1)
    >>> sys_2 = rss(2)
    >>> sys_3 = parallel(sys_1, sys_2) # Same as sys3 = sys1 + sys2
    >>> sys_3.ninputs, sys_3.noutputs, sys_3.nstates
    (1, 1, 3)

    >>> sys_4 = rss(3)
    >>> sys_5 = parallel(sys_1, sys_2, sys_3, sys_4) # More systems
    >>> sys_5.ninputs, sys_3.noutputs, sys_3.nstates
    (1, 1, 3)

    """
    from functools import reduce
    return reduce(lambda x, y:x+y, sysn, sys1)


def negate(sys):
    """
    Return the negative of a system.

    Parameters
    ----------
    sys : StateSpace, TransferFunction or FRD

    Returns
    -------
    out : StateSpace or TransferFunction

    Notes
    -----
    This function is a wrapper for the __neg__ function in the StateSpace and
    TransferFunction classes.  The output type is the same as the input type.

    Examples
    --------
    >>> from control import negate, rss
    >>> sys_1 = rss(2)
    >>> sys_2 = negate(sys_1)  # Same as sys2 = -sys1.
    >>> sys_2.ninputs, sys_2.noutputs, sys_2.nstates
    (1, 1, 2)

    """
    return -sys

#! TODO: expand to allow sys2 default to work in MIMO case?
def feedback(sys1, sys2=1, sign=-1):
    """
    Feedback interconnection between two I/O systems.

    Parameters
    ----------
    sys1 : scalar, StateSpace, TransferFunction, FRD
        The primary process.
    sys2 : scalar, StateSpace, TransferFunction, FRD
        The feedback process (often a feedback controller).
    sign: scalar
        The sign of feedback.  `sign` = -1 indicates negative feedback, and
        `sign` = 1 indicates positive feedback.  `sign` is an optional
        argument; it assumes a value of -1 if not specified.

    Returns
    -------
    out : StateSpace or TransferFunction

    Raises
    ------
    ValueError
        if `sys1` does not have as many inputs as `sys2` has outputs, or if
        `sys2` does not have as many inputs as `sys1` has outputs
    NotImplementedError
        if an attempt is made to perform a feedback on a MIMO TransferFunction
        object

    See Also
    --------
    series
    parallel

    Notes
    -----
    This function is a wrapper for the feedback function in the StateSpace and
    TransferFunction classes.  It calls TransferFunction.feedback if `sys1` is a
    TransferFunction object, and StateSpace.feedback if `sys1` is a StateSpace
    object.  If `sys1` is a scalar, then it is converted to `sys2`'s type, and
    the corresponding feedback function is used.  If `sys1` and `sys2` are both
    scalars, then TransferFunction.feedback is used.

    """
    # Allow anything with a feedback function to call that function
    try:
        return sys1.feedback(sys2, sign)
    except AttributeError:
        pass

    # Check for correct input types.
    if not isinstance(sys1, (int, float, complex, np.number,
                             xferfcn.TransferFunction, statesp.StateSpace, frdata.FRD)):
        raise TypeError("sys1 must be a TransferFunction, StateSpace " +
                        "or FRD object, or a scalar.")
    if not isinstance(sys2, (int, float, complex, np.number,
                             xferfcn.TransferFunction, statesp.StateSpace, frdata.FRD)):
        raise TypeError("sys2 must be a TransferFunction, StateSpace " +
                        "or FRD object, or a scalar.")

    # If sys1 is a scalar, convert it to the appropriate LTI type so that we can
    # its feedback member function.
    if isinstance(sys1, (int, float, complex, np.number)):
        if isinstance(sys2, xferfcn.TransferFunction):
            sys1 = xferfcn._convert_to_transfer_function(sys1)
        elif isinstance(sys2, statesp.StateSpace):
            sys1 = statesp._convert_to_statespace(sys1)
        elif isinstance(sys2, frdata.FRD):
            sys1 = frdata._convert_to_FRD(sys1, sys2.omega)
        else: # sys2 is a scalar.
            sys1 = xferfcn._convert_to_transfer_function(sys1)
            sys2 = xferfcn._convert_to_transfer_function(sys2)

    return sys1.feedback(sys2, sign)

def append(*sys):
    """append(sys1, sys2, [..., sysn])

    Group models by appending their inputs and outputs.

    Forms an augmented system model, and appends the inputs and
    outputs together. The system type will be the type of the first
    system given; if you mix state-space systems and gain matrices,
    make sure the gain matrices are not first.

    Parameters
    ----------
    sys1, sys2, ..., sysn: StateSpace or TransferFunction
        LTI systems to combine


    Returns
    -------
    sys: LTI system
        Combined LTI system, with input/output vectors consisting of all
        input/output vectors appended

    Examples
    --------
    >>> from control import append, rss
    >>> sys_1 = rss(1)
    >>> sys_2 = rss(2)
    >>> sys_3 = append(sys_1, sys_2)
    >>> sys_3.ninputs, sys_3.noutputs, sys_3.nstates
    (2, 2, 3)

    """
    s1 = statesp._convert_to_statespace(sys[0])
    for s in sys[1:]:
        s1 = s1.append(s)
    return s1

def connect(sys, Q, inputv, outputv):
    """Index-based interconnection of an LTI system.

    The system `sys` is a system typically constructed with `append`, with
    multiple inputs and outputs.  The inputs and outputs are connected
    according to the interconnection matrix `Q`, and then the final inputs and
    outputs are trimmed according to the inputs and outputs listed in `inputv`
    and `outputv`.

    NOTE: Inputs and outputs are indexed starting at 1 and negative values
    correspond to a negative feedback interconnection.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        System to be connected
    Q : 2D array
        Interconnection matrix. First column gives the input to be connected.
        The second column gives the index of an output that is to be fed into
        that input. Each additional column gives the index of an additional
        input that may be optionally added to that input. Negative
        values mean the feedback is negative. A zero value is ignored. Inputs
        and outputs are indexed starting at 1 to communicate sign information.
    inputv : 1D array
        list of final external inputs, indexed starting at 1
    outputv : 1D array
        list of final external outputs, indexed starting at 1

    Returns
    -------
    sys: LTI system
        Connected and trimmed LTI system

    Examples
    --------
    >>> from control import append, connect, rss
    >>> sys_1 = rss(1)
    >>> sys_2 = rss(2)
    >>> sys_3 = append(sys_1, sys_2)
    >>> sys_3.ninputs, sys_3.noutputs, sys_3.nstates
    (2, 2, 3)

    >>> Q = [[1, 2], [2, -1]]  # negative feedback interconnection
    >>> sys_c = connect(sys_3, Q, [2], [1, 2])
    >>> sys_c.ninputs, sys_3.noutputs, sys_3.nstates
    (1, 2, 3)

    Notes
    -----
    The :func:`~control.interconnect` function in the
    :ref:`input/output systems <iosys-module>` module allows the use
    of named signals and provides an alternative method for
    interconnecting multiple systems.

    """
    inputv, outputv, Q = \
        np.atleast_1d(inputv), np.atleast_1d(outputv), np.atleast_1d(Q)
    # check indices
    index_errors = (inputv - 1 > sys.ninputs) | (inputv < 1)
    if np.any(index_errors):
        raise IndexError(
            "inputv index %s out of bounds" % inputv[np.where(index_errors)])
    index_errors = (outputv - 1 > sys.noutputs) | (outputv < 1)
    if np.any(index_errors):
        raise IndexError(
            "outputv index %s out of bounds" % outputv[np.where(index_errors)])
    index_errors = (Q[:,0:1] - 1 > sys.ninputs) | (Q[:,0:1] < 1)
    if np.any(index_errors):
        raise IndexError(
            "Q input index %s out of bounds" % Q[np.where(index_errors)])
    index_errors = (np.abs(Q[:,1:]) - 1 > sys.noutputs)
    if np.any(index_errors):
        raise IndexError(
            "Q output index %s out of bounds" % Q[np.where(index_errors)])

    # first connect
    K = np.zeros((sys.ninputs, sys.noutputs))
    for r in np.array(Q).astype(int):
        inp = r[0]-1
        for outp in r[1:]:
            if outp < 0:
                K[inp,-outp-1] = -1.
            elif outp > 0:
                K[inp,outp-1] = 1.
    sys = sys.feedback(np.array(K), sign=1)

    # now trim
    Ytrim = np.zeros((len(outputv), sys.noutputs))
    Utrim = np.zeros((sys.ninputs, len(inputv)))
    for i,u in enumerate(inputv):
        Utrim[u-1,i] = 1.
    for i,y in enumerate(outputv):
        Ytrim[i,y-1] = 1.

    return Ytrim * sys * Utrim
