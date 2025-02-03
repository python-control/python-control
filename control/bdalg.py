# bdalg.py - block diagram algebra
#
# Initial author: Richard M. Murray
# Creation date: 24 May 09
# Pre-2014 revisions: Kevin K. Chen, Dec 2010
# Use `git shortlog -n -s bdalg.py` for full list of contributors

"""Block diagram algebra.

This module contains some standard block diagram algebra, including
series, parallel, and feedback functions.

"""

from functools import reduce
from warnings import warn

import numpy as np

from . import frdata as frd
from . import statesp as ss
from . import xferfcn as tf
from .iosys import InputOutputSystem

__all__ = ['series', 'parallel', 'negate', 'feedback', 'append', 'connect',
           'combine_tf', 'split_tf']


def series(*sys, **kwargs):
    """series(sys1, sys2[, ..., sysn])

    Series connection of I/O systems.

    Generates a new system ``[sysn * ... *] sys2 * sys1``.

    Parameters
    ----------
    sys1, sys2, ..., sysn : scalar, array, or `InputOutputSystem`
        I/O systems to combine.

    Returns
    -------
    out : `InputOutputSystem`
        Series interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i]' (where 's' is one of 'u,
        or 'y'). See `InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of the form 'x[i]' for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Raises
    ------
    ValueError
        If `sys2.ninputs` does not equal `sys1.noutputs` or if `sys1.dt` is
        not compatible with `sys2.dt`.

    See Also
    --------
    append, feedback, interconnect, negate, parallel

    Notes
    -----
    This function is a wrapper for the __mul__ function in the appropriate
    `NonlinearIOSystem`, `StateSpace`, `TransferFunction`, or other I/O
    system class.  The output type is the type of `sys1` unless a more
    general type is required based on type type of `sys2`.

    If both systems have a defined timebase (`dt` = 0 for continuous time,
    `dt` > 0 for discrete time), then the timebase for both systems must
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
    sys = reduce(lambda x, y: y * x, sys[1:], sys[0])
    sys.update_names(**kwargs)
    return sys


def parallel(*sys, **kwargs):
    r"""parallel(sys1, sys2[, ..., sysn])

    Parallel connection of I/O systems.

    Generates a parallel connection ``sys1 + sys2 [+ ...  + sysn]``.

    Parameters
    ----------
    sys1, sys2, ..., sysn : scalar, array, or `InputOutputSystem`
        I/O systems to combine.

    Returns
    -------
    out : `InputOutputSystem`
        Parallel interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i'` (where 's' is one of 'u',
        or 'y'). See `InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of the form 'x[i]' for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Raises
    ------
    ValueError
        If `sys1` and `sys2` do not have the same numbers of inputs and
        outputs.

    See Also
    --------
    append, feedback, interconnect, negate, series

    Notes
    -----
    This function is a wrapper for the __add__ function in the
    `StateSpace` and `TransferFunction` classes.  The output type is usually
    the type of `sys1`.  If `sys1` is a scalar, then the output type is
    the type of `sys2`.

    If both systems have a defined timebase (`dt` = 0 for continuous time,
    `dt` > 0 for discrete time), then the timebase for both systems must
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
    sys = reduce(lambda x, y: x + y, sys[1:], sys[0])
    sys.update_names(**kwargs)
    return sys

def negate(sys, **kwargs):
    """Return the negative of a system.

    Parameters
    ----------
    sys : scalar, array, or `InputOutputSystem`
        I/O systems to negate.

    Returns
    -------
    out : `InputOutputSystem`
        Negated system.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i]' (where 's' is one of 'u',
        or 'y'). See `InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form 'x[i]' for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    See Also
    --------
    append, feedback, interconnect, parallel, series

    Notes
    -----
    This function is a wrapper for the __neg__ function in the `StateSpace`
    and `TransferFunction` classes.  The output type is the same as the
    input type.

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
    sys1, sys2 : scalar, array, or `InputOutputSystem`
        I/O systems to combine.
    sign : scalar, optional
        The sign of feedback.  `sign=-1` indicates negative feedback
        (default), and `sign=1` indicates positive feedback.

    Returns
    -------
    out : `InputOutputSystem`
        Feedback interconnection of the systems.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i]' (where 's' is one of 'u',
        or 'y'). See `InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form 'x[i]' for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Raises
    ------
    ValueError
        If `sys1` does not have as many inputs as `sys2` has outputs, or if
        `sys2` does not have as many inputs as `sys1` has outputs.
    NotImplementedError
        If an attempt is made to perform a feedback on a MIMO `TransferFunction`
        object.

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
    """append(sys1, sys2[, ..., sysn])

    Group LTI models by appending their inputs and outputs.

    Forms an augmented system model, and appends the inputs and
    outputs together.

    Parameters
    ----------
    sys1, sys2, ..., sysn : scalar, array, or `LTI`
        I/O systems to combine.

    Returns
    -------
    out : `LTI`
        Combined system, with input/output vectors consisting of all
        input/output vectors appended. Specific type returned is the type of
        the first argument.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i]' (where 's' is one of 'u',
        or 'y'). See `InputOutputSystem` for more information.
    states : str, or list of str, optional
        List of names for system states.  If not given, state names will be
        of of the form 'x[i]' for interconnections of linear systems or
        '<subsys_name>.<state_name>' for interconnected nonlinear systems.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

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
    s1 = sys[0]
    for s in sys[1:]:
        s1 = s1.append(s)
    s1.update_names(**kwargs)
    return s1

def connect(sys, Q, inputv, outputv):
    """Index-based interconnection of an LTI system.

    .. deprecated:: 0.10.0
        `connect` will be removed in a future version of python-control.
        Use `interconnect` instead, which works with named signals.

    The system `sys` is a system typically constructed with `append`, with
    multiple inputs and outputs.  The inputs and outputs are connected
    according to the interconnection matrix `Q`, and then the final inputs and
    outputs are trimmed according to the inputs and outputs listed in `inputv`
    and `outputv`.

    NOTE: Inputs and outputs are indexed starting at 1 and negative values
    correspond to a negative feedback interconnection.

    Parameters
    ----------
    sys : `InputOutputSystem`
        System to be connected.
    Q : 2D array
        Interconnection matrix. First column gives the input to be connected.
        The second column gives the index of an output that is to be fed into
        that input. Each additional column gives the index of an additional
        input that may be optionally added to that input. Negative
        values mean the feedback is negative. A zero value is ignored. Inputs
        and outputs are indexed starting at 1 to communicate sign information.
    inputv : 1D array
        List of final external inputs, indexed starting at 1.
    outputv : 1D array
        List of final external outputs, indexed starting at 1.

    Returns
    -------
    out : `InputOutputSystem`
        Connected and trimmed I/O system.

    See Also
    --------
    append, feedback, interconnect, negate, parallel, series

    Notes
    -----
    The `interconnect` function allows the use of named signals and
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

def combine_tf(tf_array, **kwargs):
    """Combine array of transfer functions into MIMO transfer function.

    Parameters
    ----------
    tf_array : list of list of `TransferFunction` or array_like
        Transfer matrix represented as a two-dimensional array or
        list-of-lists containing `TransferFunction` objects. The
        `TransferFunction` objects can have multiple outputs and inputs, as
        long as the dimensions are compatible.

    Returns
    -------
    `TransferFunction`
        Transfer matrix represented as a single MIMO `TransferFunction` object.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals.  If not given,
        signal names will be of the form 's[i]' (where 's' is one of 'u',
        or 'y'). See `InputOutputSystem` for more information.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Raises
    ------
    ValueError
        If timebase of transfer functions do not match.
    ValueError
        If `tf_array` has incorrect dimensions.
    ValueError
        If the transfer functions in a row have mismatched output or input
        dimensions.

    Examples
    --------
    Combine two transfer functions:

    >>> s = ct.tf('s')
    >>> ct.combine_tf(
    ...     [[1 / (s + 1)],
    ...      [s / (s + 2)]],
    ...     name='G'
    ... )
    TransferFunction(
    [[array([1])],
     [array([1, 0])]],
    [[array([1, 1])],
     [array([1, 2])]],
    name='G', outputs=2, inputs=1)

    Combine NumPy arrays with transfer functions:

    >>> ct.combine_tf(
    ...     [[np.eye(2), np.zeros((2, 1))],
    ...      [np.zeros((1, 2)), ct.tf([1], [1, 0])]],
    ...     name='G'
    ... )
    TransferFunction(
    [[array([1.]), array([0.]), array([0.])],
     [array([0.]), array([1.]), array([0.])],
     [array([0.]), array([0.]), array([1])]],
    [[array([1.]), array([1.]), array([1.])],
     [array([1.]), array([1.]), array([1.])],
     [array([1.]), array([1.]), array([1, 0])]],
    name='G', outputs=3, inputs=3)

    """
    # Find common timebase or raise error
    dt_list = []
    try:
        for row in tf_array:
            for tfn in row:
                dt_list.append(getattr(tfn, "dt", None))
    except OSError:
        raise ValueError("`tf_array` has too few dimensions.")
    dt_set = set(dt_list)
    dt_set.discard(None)
    if len(dt_set) > 1:
        raise ValueError("Time steps of transfer functions are "
                         f"mismatched: {dt_set}")
    elif len(dt_set) == 0:
        dt = None
    else:
        dt = dt_set.pop()
    # Convert all entries to transfer function objects
    ensured_tf_array = []
    for row in tf_array:
        ensured_row = []
        for tfn in row:
            ensured_row.append(_ensure_tf(tfn, dt))
        ensured_tf_array.append(ensured_row)
    # Iterate over
    num = []
    den = []
    for row_index, row in enumerate(ensured_tf_array):
        for j_out in range(row[0].noutputs):
            num_row = []
            den_row = []
            for col in row:
                if col.noutputs != row[0].noutputs:
                    raise ValueError(
                        "Mismatched number of transfer function outputs in "
                        f"row {row_index}."
                    )
                for j_in in range(col.ninputs):
                    num_row.append(col.num_array[j_out, j_in])
                    den_row.append(col.den_array[j_out, j_in])
            num.append(num_row)
            den.append(den_row)
    for row_index, row in enumerate(num):
        if len(row) != len(num[0]):
            raise ValueError(
                "Mismatched number transfer function inputs in row "
                f"{row_index} of numerator."
            )
    for row_index, row in enumerate(den):
        if len(row) != len(den[0]):
            raise ValueError(
                "Mismatched number transfer function inputs in row "
                f"{row_index} of denominator."
            )
    return tf.TransferFunction(num, den, dt=dt, **kwargs)



def split_tf(transfer_function):
    """Split MIMO transfer function into SISO transfer functions.

    System and signal names for the array of SISO transfer functions are
    copied from the MIMO system.

    Parameters
    ----------
    transfer_function : `TransferFunction`
        MIMO transfer function to split.

    Returns
    -------
    ndarray
        NumPy array of SISO transfer functions.

    Examples
    --------
    Split a MIMO transfer function:

    >>> G = ct.tf(
    ...     [ [[87.8], [-86.4]],
    ...       [[108.2], [-109.6]] ],
    ...     [ [[1, 1], [1, 1]],
    ...       [[1, 1], [1, 1]],   ],
    ...     name='G'
    ... )
    >>> ct.split_tf(G)
    array([[TransferFunction(
            array([87.8]),
            array([1, 1]),
            name='G', outputs=1, inputs=1), TransferFunction(
                                            array([-86.4]),
                                            array([1, 1]),
                                            name='G', outputs=1, inputs=1)],
           [TransferFunction(
            array([108.2]),
            array([1, 1]),
            name='G', outputs=1, inputs=1), TransferFunction(
                                            array([-109.6]),
                                            array([1, 1]),
                                            name='G', outputs=1, inputs=1)]],
          dtype=object)

    """
    tf_split_lst = []
    for i_out in range(transfer_function.noutputs):
        row = []
        for i_in in range(transfer_function.ninputs):
            row.append(
                tf.TransferFunction(
                    transfer_function.num_array[i_out, i_in],
                    transfer_function.den_array[i_out, i_in],
                    dt=transfer_function.dt,
                    inputs=transfer_function.input_labels[i_in],
                    outputs=transfer_function.output_labels[i_out],
                    name=transfer_function.name
                )
            )
        tf_split_lst.append(row)
    return np.array(tf_split_lst, dtype=object)

def _ensure_tf(arraylike_or_tf, dt=None):
    """Convert an array_like to a transfer function.

    Parameters
    ----------
    arraylike_or_tf : `TransferFunction` or array_like
        Array-like or transfer function.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete
        time). If None, timebase is not validated.

    Returns
    -------
    `TransferFunction`
        Transfer function.

    Raises
    ------
    ValueError
        If input cannot be converted to a transfer function.
    ValueError
        If the timebases do not match.

    """
    # If the input is already a transfer function, return it right away
    if isinstance(arraylike_or_tf, tf.TransferFunction):
        # If timebases don't match, raise an exception
        if (dt is not None) and (arraylike_or_tf.dt != dt):
            raise ValueError(
                f"`arraylike_or_tf.dt={arraylike_or_tf.dt}` does not match "
                f"argument `dt={dt}`."
            )
        return arraylike_or_tf
    if np.ndim(arraylike_or_tf) > 2:
        raise ValueError(
            "Array-like must have less than two dimensions to be converted "
            "into a transfer function."
        )
    # If it's not, then convert it to a transfer function
    arraylike_3d = np.atleast_3d(arraylike_or_tf)
    try:
        tfn = tf.TransferFunction(
            arraylike_3d,
            np.ones_like(arraylike_3d),
            dt,
        )
    except TypeError:
        raise ValueError(
            "`arraylike_or_tf` must only contain array_likes or transfer "
            "functions."
        )
    return tfn
