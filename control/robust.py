# robust.py - tools for robust control
#
# Author: Steve Brunton, Kevin Chen, Lauren Padilla
# Date: 24 Dec 2010
#
# This file contains routines for obtaining reduced order models
#
# Copyright (c) 2010 by California Institute of Technology
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

# External packages and modules
import numpy as np
from .exception import *
from .statesp import StateSpace
from .statefbk import *


def h2syn(P, nmeas, ncon):
    """H_2 control synthesis for plant P.

    Parameters
    ----------
    P: partitioned lti plant (State-space sys)
    nmeas: number of measurements (input to controller)
    ncon: number of control inputs (output from controller)

    Returns
    -------
    K: controller to stabilize P (State-space sys)

    Raises
    ------
    ImportError
        if slycot routine sb10hd is not loaded

    See Also
    --------
    StateSpace

    Examples
    --------
    >>> K = h2syn(P,nmeas,ncon)

    """
    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported right now
    # if isCont():
    #    dico = 'C'
    # elif isDisc():
    #    dico = 'D'
    # else:
    dico = 'C'

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
    """H_{inf} control synthesis for plant P.

    Parameters
    ----------
    P: partitioned lti plant
    nmeas: number of measurements (input to controller)
    ncon: number of control inputs (output from controller)

    Returns
    -------
    K: controller to stabilize P (State-space sys)
    CL: closed loop system (State-space sys)
    gam: infinity norm of closed loop system
    rcond: 4-vector, reciprocal condition estimates of:
        1: control transformation matrix
        2: measurement transformation matrix
        3: X-Riccati equation
        4: Y-Riccati equation
    TODO: document significance of rcond

    Raises
    ------
    ImportError
        if slycot routine sb10ad is not loaded

    See Also
    --------
    StateSpace

    Examples
    --------
    >>> K, CL, gam, rcond = hinfsyn(P,nmeas,ncon)

    """

    # Check for ss system object, need a utility for this?

    # TODO: Check for continous or discrete, only continuous supported right now
    # if isCont():
    #    dico = 'C'
    # elif isDisc():
    #    dico = 'D'
    # else:
    dico = 'C'

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
    w_: processed weighting function, a StateSpace object:
        - if w is None, empty StateSpace object
        - if w is scalar, w_ will be w * eye(n)
        - otherwise, w as StateSpace object

    Raises
    ------
    ValueError
        - if w is not None or scalar, and doesn't have n inputs

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

    Parameters
    ----------
    g: LTI object, ny-by-nu
    w1: weighting on S; None, scalar, or k1-by-ny LTI object
    w2: weighting on KS; None, scalar, or k2-by-nu LTI object
    w3: weighting on T; None, scalar, or k3-by-ny LTI object
    p: augmented plant; StateSpace object

    If a weighting is None, no augmentation is done for it.  At least
    one weighting must not be None.

    If a weighting w is scalar, it will be replaced by I*w, where I is
    ny-by-ny for w1 and w3, and nu-by-nu for w2.

    Returns
    -------
    p: plant augmented with weightings, suitable for submission to hinfsyn or h2syn.

    Raises
    ------
    ValueError
        - if all weightings are None

    See Also
    --------
    h2syn, hinfsyn, mixsyn
    """

    from . import append, ss, connect

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
    q[niw1 + niw2:niw1 + niw2 + niw3, 1] = np.arange(1 + now1 + now2 + now3 + ny,
                                                     1 + now1 + now2 + now3 + ny + niw3)

    # -y -> Iy; note the leading -
    q[niw1 + niw2 + niw3:niw1 + niw2 + niw3 + ny, 1] = -np.arange(1 + now1 + now2 + now3 + ny,
                                                                  1 + now1 + now2 + now3 + 2 * ny)

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

    p = connect(sysall, q, ii, oi)

    return p


def mixsyn(g, w1=None, w2=None, w3=None):
    """Mixed-sensitivity H-infinity synthesis.

    mixsyn(g,w1,w2,w3) -> k,cl,info

    Parameters
    ----------
    g: LTI; the plant for which controller must be synthesized
    w1: weighting on s = (1+g*k)**-1; None, or scalar or k1-by-ny LTI
    w2: weighting on k*s; None, or scalar or k2-by-nu LTI
    w3: weighting on t = g*k*(1+g*k)**-1; None, or scalar or k3-by-ny LTI
    At least one of w1, w2, and w3 must not be None.

    Returns
    -------
    k: synthesized controller; StateSpace object
    cl: closed system mapping evaluation inputs to evaluation outputs; if 
    p is the augmented plant, with
        [z] = [p11 p12] [w], 
        [y]   [p21   g] [u]
    then cl is the system from w->z with u=-k*y.  StateSpace object.

    info: tuple with entries, in order,
        - gamma: scalar; H-infinity norm of cl
        - rcond: array; estimates of reciprocal condition numbers
          computed during synthesis.  See hinfsyn for details

    If a weighting w is scalar, it will be replaced by I*w, where I is
    ny-by-ny for w1 and w3, and nu-by-nu for w2.

    See Also
    --------
    hinfsyn, augw
    """
    nmeas = g.noutputs
    ncon = g.ninputs
    p = augw(g, w1, w2, w3)

    k, cl, gamma, rcond = hinfsyn(p, nmeas, ncon)
    info = gamma, rcond
    return k, cl, info
