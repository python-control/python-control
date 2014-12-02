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

def h2syn(P,nmeas,ncon):
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

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
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

    n = np.size(P.A,0)
    m = np.size(P.B,1)
    np = np.size(P.C,0)
    out = sb10hd(n,m,np,ncon,nmeas,P.A,P.B,P.C,P.D)
    Ak = out[0]
    Bk = out[1]
    Ck = out[2]
    Dk = out[3]

    K = StateSpace(Ak, Bk, Ck, Dk)

    return K

def hinfsyn(P,nmeas,ncon):
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
    info: info returned from siycot routine

    Raises
    ------
    ImportError
        if slycot routine sb10ad is not loaded

    See Also
    --------
    StateSpace

    Examples
    --------
    >>> K, CL, gam, info = hinfsyn(P,nmeas,ncon)

    """

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
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

    job = 3
    n = np.size(P.A,0)
    m = np.size(P.B,1)
    np = np.size(P.C,0)
    gamma = 1.e100
    out = sb10ad(job,n,m,np,ncon,nmeas,gamma,P.A,P.B,P.C,P.D)
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
    info = out[10]

    K = StateSpace(Ak, Bk, Ck, Dk)
    CL = StateSpace(Ac, Bc, Cc, Dc)

    return K, CL, gam, info

