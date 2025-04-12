# robust.py - tools for robust control
#
# Initial authors: Steve Brunton, Kevin Chen, Lauren Padilla
# Creation date: 24 Dec 2010

"""Robust control synthesis algorithms."""

import warnings

# External packages and modules
import numpy as np

from .exception import ControlSlycot
from .statesp import StateSpace


def h2syn(P, nmeas, ncon):
    """H2 control synthesis for plant P.

    Parameters
    ----------
    P : `StateSpace`
        Partitioned LTI plant (state-space system).
    nmeas : int
        Number of measurements (input to controller).
    ncon : int
        Number of control inputs (output from controller).

    Returns
    -------
    K : `StateSpace`
        Controller to stabilize `P`.

    Raises
    ------
    ImportError
        If slycot routine sb10hd is not loaded.

    See Also
    --------
    StateSpace

    Examples
    --------
    >>> # Unstable first order SISO system
    >>> G = ct.tf([1], [1, -1], inputs=['u'], outputs=['y'])
    >>> all(G.poles() < 0)  # Is G stable?
    False

    >>> # Create partitioned system with trivial unity systems
    >>> P11 = ct.tf([0], [1], inputs=['w'], outputs=['z'])
    >>> P12 = ct.tf([1], [1], inputs=['u'], outputs=['z'])
    >>> P21 = ct.tf([1], [1], inputs=['w'], outputs=['y'])
    >>> P22 = G
    >>> P = ct.interconnect([P11, P12, P21, P22],
    ...                     inplist=['w', 'u'], outlist=['z', 'y'])

    >>> # Synthesize H2 optimal stabilizing controller
    >>> K = ct.h2syn(P, nmeas=1, ncon=1)
    >>> T = ct.feedback(G, K, sign=1)
    >>> all(T.poles() < 0)  # Is T stable?
    True

    """
    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported right now

    try:
        from slycot import sb10hd
    except ImportError:
        raise ControlSlycot("can't find slycot subroutine sb10hd")

    n = np.size(P.A, 0)
    m = np.size(P.B, 1)
    np_ = np.size(P.C, 0)
    out = sb10hd(n, m, np_, ncon, nmeas, P.A, P.B, P.C, P.D)
    Ak = out[0]
    Bk = out[1]
    Ck = out[2]
    Dk = out[3]

    K = StateSpace(Ak, Bk, Ck, Dk)

    return K


def hinfsyn(P, nmeas, ncon):
    # TODO: document significance of rcond
    """H-infinity control synthesis for plant P.

    Parameters
    ----------
    P : `StateSpace`
        Partitioned LTI plant (state-space system).
    nmeas : int
        Number of measurements (input to controller).
    ncon : int
        Number of control inputs (output from controller).

    Returns
    -------
    K : `StateSpace`
        Controller to stabilize `P`.
    CL : `StateSpace`
        Closed loop system.
    gam : float
        Infinity norm of closed loop system.
    rcond : list
        4-vector, reciprocal condition estimates of:
            1: control transformation matrix
            2: measurement transformation matrix
            3: X-Riccati equation
            4: Y-Riccati equation

    Raises
    ------
    ImportError
        If slycot routine sb10ad is not loaded.

    See Also
    --------
    StateSpace

    Examples
    --------
    >>> # Unstable first order SISO system
    >>> G = ct.tf([1], [1,-1], inputs=['u'], outputs=['y'])
    >>> all(G.poles() < 0)
    False

    >>> # Create partitioned system with trivial unity systems
    >>> P11 = ct.tf([0], [1], inputs=['w'], outputs=['z'])
    >>> P12 = ct.tf([1], [1], inputs=['u'], outputs=['z'])
    >>> P21 = ct.tf([1], [1], inputs=['w'], outputs=['y'])
    >>> P22 = G
    >>> P = ct.interconnect([P11, P12, P21, P22], inplist=['w', 'u'], outlist=['z', 'y'])

    >>> # Synthesize Hinf optimal stabilizing controller
    >>> K, CL, gam, rcond = ct.hinfsyn(P, nmeas=1, ncon=1)
    >>> T = ct.feedback(G, K, sign=1)
    >>> all(T.poles() < 0)
    True

    """

    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported right now

    try:
        from slycot import sb10ad
    except ImportError:
        raise ControlSlycot("can't find slycot subroutine sb10ad")

    n = np.size(P.A, 0)
    m = np.size(P.B, 1)
    np_ = np.size(P.C, 0)
    gamma = 1.e100
    out = sb10ad(n, m, np_, ncon, nmeas, gamma, P.A, P.B, P.C, P.D)
    gam = out[0]
    Ak = out[1]
    Bk = out[2]
    Ck = out[3]
    Dk = out[4]
    Ac = out[5]
    Bc = out[6]
    Cc = out[7]
    Dc = out[8]
    rcond = out[9]

    K = StateSpace(Ak, Bk, Ck, Dk)
    CL = StateSpace(Ac, Bc, Cc, Dc)

    return K, CL, gam, rcond


def _size_as_needed(w, wname, n):
    """Convert LTI object to appropriately sized StateSpace object.

    Intended for use in .robust only

    Parameters
    ----------
    w: None, 1x1 LTI object, or mxn LTI object
    wname: name of w, for error message
    n: number of inputs to w

    Returns
    -------
    w_: processed weighting function, a `StateSpace` object:
        - if w is None, empty `StateSpace` object
        - if w is scalar, w_ will be w * eye(n)
        - otherwise, w as `StateSpace` object

    Raises
    ------
    ValueError
        If w is not None or scalar, and does not have n inputs.

    See Also
    --------
    augw

    """
    from . import append, ss
    if w is not None:
        if not isinstance(w, StateSpace):
            w = ss(w)
        if 1 == w.ninputs and 1 == w.noutputs:
            w = append(*(w,) * n)
        else:
            if w.ninputs != n:
                msg = ("{}: weighting function has {} inputs, expected {}".
                       format(wname, w.ninputs, n))
                raise ValueError(msg)
    else:
        w = ss([], [], [], [])
    return w


def augw(g, w1=None, w2=None, w3=None):
    """Augment plant for mixed sensitivity problem.

    If a weighting is None, no augmentation is done for it.  At least
    one weighting must not be None.

    If a weighting w is scalar, it will be replaced by I*w, where I is
    ny-by-ny for `w1` and `w3`, and nu-by-nu for `w2`.

    Parameters
    ----------
    g : LTI object, ny-by-nu
        Plant.
    w1 : None, scalar, or k1-by-ny LTI object
        Weighting on S.
    w2 : None, scalar, or k2-by-nu LTI object
        Weighting on KS.
    w3 : None, scalar, or k3-by-ny LTI object
        Weighting on T.

    Returns
    -------
    p : `StateSpace`
        Plant augmented with weightings, suitable for submission to
        `hinfsyn` or `h2syn`.

    Raises
    ------
    ValueError
        If all weightings are None.

    See Also
    --------
    h2syn, hinfsyn, mixsyn

    """

    from . import append, connect, ss

    if w1 is None and w2 is None and w3 is None:
        raise ValueError("At least one weighting must not be None")
    ny = g.noutputs
    nu = g.ninputs

    w1, w2, w3 = [_size_as_needed(w, wname, n)
                  for w, wname, n in zip((w1, w2, w3),
                                         ('w1', 'w2', 'w3'),
                                         (ny, nu, ny))]

    if not isinstance(g, StateSpace):
        g = ss(g)

    #       w         u
    #  z1 [ w1   |   -w1*g  ]
    #  z2 [ 0    |    w2    ]
    #  z3 [ 0    |    w3*g  ]
    #     [------+--------- ]
    #  v  [ I    |    -g    ]

    # error summer: inputs are -y and r=w
    Ie = ss([], [], [], np.eye(ny))
    # control: needed to "distribute" control input
    Iu = ss([], [], [], np.eye(nu))

    sysall = append(w1, w2, w3, Ie, g, Iu)

    niw1 = w1.ninputs
    niw2 = w2.ninputs
    niw3 = w3.ninputs

    now1 = w1.noutputs
    now2 = w2.noutputs
    now3 = w3.noutputs

    q = np.zeros((niw1 + niw2 + niw3 + ny + nu, 2))
    q[:, 0] = np.arange(1, q.shape[0] + 1)

    # Ie -> w1
    q[:niw1, 1] = np.arange(1 + now1 + now2 + now3,
                            1 + now1 + now2 + now3 + niw1)

    # Iu -> w2
    q[niw1:niw1 + niw2, 1] = np.arange(1 + now1 + now2 + now3 + 2 * ny,
                                       1 + now1 + now2 + now3 + 2 * ny + niw2)

    # y -> w3
    q[niw1 + niw2:niw1 + niw2 + niw3, 1] = np.arange(
        1 + now1 + now2 + now3 + ny, 1 + now1 + now2 + now3 + ny + niw3)

    # -y -> Iy; note the leading -
    q[niw1 + niw2 + niw3:niw1 + niw2 + niw3 + ny, 1] = -np.arange(
        1 + now1 + now2 + now3 + ny, 1 + now1 + now2 + now3 + 2 * ny)

    # Iu -> G
    q[niw1 + niw2 + niw3 + ny:niw1 + niw2 + niw3 + ny + nu, 1] = np.arange(
        1 + now1 + now2 + now3 + 2 * ny,
        1 + now1 + now2 + now3 + 2 * ny + nu)

    # input indices: to Ie and Iu
    ii = np.hstack((np.arange(1 + now1 + now2 + now3,
                              1 + now1 + now2 + now3 + ny),
                    np.arange(1 + now1 + now2 + now3 + ny + nu,
                              1 + now1 + now2 + now3 + ny + nu + nu)))

    # output indices
    oi = np.arange(1, 1 + now1 + now2 + now3 + ny)

    # Filter out known warning due to use of connect
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message="`connect`", category=DeprecationWarning)

        p = connect(sysall, q, ii, oi)

    return p


def mixsyn(g, w1=None, w2=None, w3=None):
    """Mixed-sensitivity H-infinity synthesis.

    mixsyn(g,w1,w2,w3) -> k,cl,info

    Parameters
    ----------
    g : LTI
        The plant for which controller must be synthesized.
    w1 : None, or scalar or k1-by-ny LTI
        Weighting on S = (1+G*K)**-1.
    w2 : None, or scalar or k2-by-nu LTI
        Weighting on K*S.
    w3 : None, or scalar or k3-by-ny LTI
        Weighting on T = G*K*(1+G*K)**-1.

    Returns
    -------
    k : `StateSpace`
        Synthesized controller.
    cl : `StateSpace`
        Closed system mapping evaluation inputs to evaluation outputs.

        Let p be the augmented plant, with::

            [z] = [p11 p12] [w]
            [y]   [p21   g] [u]

        then cl is the system from w -> z with u = -k*y.
    info : tuple
        Two-tuple (`gamma`, `rcond`) containing additional information:
            - `gamma` (scalar): H-infinity norm of cl.
            - `rcond` (array): Estimates of reciprocal condition numbers
               computed during synthesis.  See hinfsyn for details.

    See Also
    --------
    hinfsyn, augw

    Notes
    -----
    If a weighting w is scalar, it will be replaced by I*w, where I is
    ny-by-ny for `w1` and `w3`, and nu-by-nu for `w2`.

    """
    nmeas = g.noutputs
    ncon = g.ninputs
    p = augw(g, w1, w2, w3)

    k, cl, gamma, rcond = hinfsyn(p, nmeas, ncon)
    info = gamma, rcond
    return k, cl, info
