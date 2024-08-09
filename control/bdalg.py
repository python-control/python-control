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

from functools import reduce
from warnings import warn

import numpy as np

from . import frdata as frd
from . import statesp as ss
from . import xferfcn as tf
from .iosys import InputOutputSystem

__all__ = ['series', 'parallel', 'negate', 'feedback', 'append', 'connect']


def series(sys1, *sysn, **kwargs):
    r"""series(sys1, sys2, [..., sysn])

    Return the series connection (`sysn` \* ...\  \*) `sys2` \* `sys1`.

    Parameters
    ----------
    sys1, sys2, ..., sysn : scalar, array, or :class:`InputOutputSystem`
        I/O systems to combine.

    Returns
    -------
    out : scalar, array, or :class:`InputOutputSystem`
        Series interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form `s[i]` (where `s` is one of `u`,
        or `y`). See :class:`InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form `x[i]` for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    Raises
    ------
    ValueError
        if `sys2.ninputs` does not equal `sys1.noutputs`
        if `sys1.dt` is not compatible with `sys2.dt`

    See Also
    --------
    append, feedback, interconnect, negate, parallel

    Notes
    -----
    This function is a wrapper for the __mul__ function in the appropriate
    :class:`NonlinearIOSystem`, :class:`StateSpace`,
    :class:`TransferFunction`, or other I/O system class.  The output type
    is the type of `sys1` unless a more general type is required based on
    type type of `sys2`.

    If both systems have a defined timebase (dt = 0 for continuous time,
    dt > 0 for discrete time), then the timebase for both systems must
    match.  If only one of the system has a timebase, the return
    timebase will be set to match it.

    Examples
    --------
    >>> G1 = ct.rss(3)
    >>> G2 = ct.rss(4)
    >>> G = ct.series(G1, G2) # Same as sys3 = sys2 * sys1
    >>> G.ninputs, G.noutputs, G.nstates
    (1, 1, 7)

    >>> G1 = ct.rss(2, inputs=2, outputs=3)
    >>> G2 = ct.rss(3, inputs=3, outputs=1)
    >>> G = ct.series(G1, G2) # Same as sys3 = sys2 * sys1
    >>> G.ninputs, G.noutputs, G.nstates
    (2, 1, 5)

    """
    sys = reduce(lambda x, y: y * x, sysn, sys1)
    sys.update_names(**kwargs)
    return sys


def parallel(sys1, *sysn, **kwargs):
    r"""parallel(sys1, sys2, [..., sysn])

    Return the parallel connection `sys1` + `sys2` (+ ...\  + `sysn`).

    Parameters
    ----------
    sys1, sys2, ..., sysn : scalar, array, or :class:`InputOutputSystem`
        I/O systems to combine.

    Returns
    -------
    out : scalar, array, or :class:`InputOutputSystem`
        Parallel interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form `s[i]` (where `s` is one of `u`,
        or `y`). See :class:`InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form `x[i]` for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    Raises
    ------
    ValueError
        if `sys1` and `sys2` do not have the same numbers of inputs and outputs

    See Also
    --------
    append, feedback, interconnect, negate, series

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
    >>> G1 = ct.rss(3)
    >>> G2 = ct.rss(4)
    >>> G = ct.parallel(G1, G2) # Same as sys3 = sys1 + sys2
    >>> G.ninputs, G.noutputs, G.nstates
    (1, 1, 7)

    >>> G1 = ct.rss(3, inputs=3, outputs=4)
    >>> G2 = ct.rss(4, inputs=3, outputs=4)
    >>> G = ct.parallel(G1, G2)  # Add another system
    >>> G.ninputs, G.noutputs, G.nstates
    (3, 4, 7)

    """
    sys = reduce(lambda x, y: x + y, sysn, sys1)
    sys.update_names(**kwargs)
    return sys

def negate(sys, **kwargs):
    """
    Return the negative of a system.

    Parameters
    ----------
    sys : scalar, array, or :class:`InputOutputSystem`
        I/O systems to negate.

    Returns
    -------
    out : scalar, array, or :class:`InputOutputSystem`
        Negated system.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form `s[i]` (where `s` is one of `u`,
        or `y`). See :class:`InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form `x[i]` for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    See Also
    --------
    append, feedback, interconnect, parallel, series

    Notes
    -----
    This function is a wrapper for the __neg__ function in the StateSpace and
    TransferFunction classes.  The output type is the same as the input type.

    Examples
    --------
    >>> G = ct.tf([2], [1, 1])
    >>> G.dcgain()
    np.float64(2.0)

    >>> Gn = ct.negate(G) # Same as sys2 = -sys1.
    >>> Gn.dcgain()
    np.float64(-2.0)

    """
    sys = -sys
    sys.update_names(**kwargs)
    return sys

#! TODO: expand to allow sys2 default to work in MIMO case?
def feedback(sys1, sys2=1, sign=-1, **kwargs):
    """Feedback interconnection between two I/O systems.

    Parameters
    ----------
    sys1, sys2 : scalar, array, or :class:`InputOutputSystem`
        I/O systems to combine.
    sign : scalar
        The sign of feedback.  `sign` = -1 indicates negative feedback, and
        `sign` = 1 indicates positive feedback.  `sign` is an optional
        argument; it assumes a value of -1 if not specified.

    Returns
    -------
    out : scalar, array, or :class:`InputOutputSystem`
        Feedback interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form `s[i]` (where `s` is one of `u`,
        or `y`). See :class:`InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form `x[i]` for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

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
    append, interconnect, negate, parallel, series

    Notes
    -----
    This function is a wrapper for the `feedback` function in the I/O
    system classes.  It calls sys1.feedback if `sys1` is an I/O system
    object.  If `sys1` is a scalar, then it is converted to `sys2`'s type,
    and the corresponding feedback function is used.

    Examples
    --------
    >>> G = ct.rss(3, inputs=2, outputs=5)
    >>> C = ct.rss(4, inputs=5, outputs=2)
    >>> T = ct.feedback(G, C, sign=1)
    >>> T.ninputs, T.noutputs, T.nstates
    (2, 5, 7)

    """
    # Allow anything with a feedback function to call that function
    # TODO: rewrite to allow __rfeedback__
    try:
        return sys1.feedback(sys2, sign, **kwargs)
    except (AttributeError, TypeError):
        pass

    # Check for correct input types
    if not isinstance(sys1, (int, float, complex, np.number, np.ndarray,
                             InputOutputSystem)):
        raise TypeError("sys1 must be an I/O system, scalar, or array")
    elif not isinstance(sys2, (int, float, complex, np.number, np.ndarray,
                               InputOutputSystem)):
        raise TypeError("sys2 must be an I/O system, scalar, or array")

    # If sys1 is a scalar or ndarray, use the type of sys2 to figure
    # out how to convert sys1, using transfer functions whenever possible.
    if isinstance(sys1, (int, float, complex, np.number, np.ndarray)):
        if isinstance(sys2, (int, float, complex, np.number, np.ndarray,
                             tf.TransferFunction)):
            sys1 = tf._convert_to_transfer_function(sys1)
        elif isinstance(sys2, frd.FrequencyResponseData):
            sys1 = frd._convert_to_frd(sys1, sys2.omega)
        else:
            sys1 = ss._convert_to_statespace(sys1)

    sys = sys1.feedback(sys2, sign)
    sys.update_names(**kwargs)
    return sys

def append(*sys, **kwargs):
    """append(sys1, sys2, [..., sysn])

    Group LTI state space models by appending their inputs and outputs.

    Forms an augmented system model, and appends the inputs and
    outputs together.

    Parameters
    ----------
    sys1, sys2, ..., sysn: scalar, array, or :class:`StateSpace`
        I/O systems to combine.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form `s[i]` (where `s` is one of `u`,
        or `y`). See :class:`InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form `x[i]` for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    Returns
    -------
    out: :class:`StateSpace`
        Combined system, with input/output vectors consisting of all
        input/output vectors appended.

    See Also
    --------
    interconnect, feedback, negate, parallel, series

    Examples
    --------
    >>> G1 = ct.rss(3)
    >>> G2 = ct.rss(4)
    >>> G = ct.append(G1, G2)
    >>> G.ninputs, G.noutputs, G.nstates
    (2, 2, 7)

    >>> G1 = ct.rss(3, inputs=2, outputs=4)
    >>> G2 = ct.rss(4, inputs=1, outputs=4)
    >>> G = ct.append(G1, G2)
    >>> G.ninputs, G.noutputs, G.nstates
    (3, 8, 7)

    """
    s1 = ss._convert_to_statespace(sys[0])
    for s in sys[1:]:
        s1 = s1.append(s)
    s1.update_names(**kwargs)
    return s1

def connect(sys, Q, inputv, outputv):
    """Index-based interconnection of an LTI system.

    .. deprecated:: 0.10.0
        `connect` will be removed in a future version of python-control.
        Use :func:`interconnect` instead, which works with named signals.

    The system `sys` is a system typically constructed with `append`, with
    multiple inputs and outputs.  The inputs and outputs are connected
    according to the interconnection matrix `Q`, and then the final inputs and
    outputs are trimmed according to the inputs and outputs listed in `inputv`
    and `outputv`.

    NOTE: Inputs and outputs are indexed starting at 1 and negative values
    correspond to a negative feedback interconnection.

    Parameters
    ----------
    sys : :class:`InputOutputSystem`
        System to be connected.
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
    out : :class:`InputOutputSystem`
        Connected and trimmed I/O system.

    See Also
    --------
    append, feedback, interconnect, negate, parallel, series

    Notes
    -----
    The :func:`~control.interconnect` function in the :ref:`input/output
    systems <iosys-module>` module allows the use of named signals and
    provides an alternative method for interconnecting multiple systems.

    Examples
    --------
    >>> G = ct.rss(7, inputs=2, outputs=2)
    >>> K = [[1, 2], [2, -1]]  # negative feedback interconnection
    >>> T = ct.connect(G, K, [2], [1, 2])
    >>> T.ninputs, T.noutputs, T.nstates
    (1, 2, 7)

    """
    # TODO: maintain `connect` for use in MATLAB submodule (?)
    warn("connect() is deprecated; use interconnect()", FutureWarning)

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
