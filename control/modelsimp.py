#! TODO: add module docstring
# modelsimp.py - tools for model simplification
#
# Author: Steve Brunton, Kevin Chen, Lauren Padilla
# Date: 30 Nov 2010
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

# Python 3 compatability
from __future__ import print_function

# External packages and modules
import numpy as np
from .exception import ControlSlycot
from .lti import isdtime, isctime
from .statesp import StateSpace
from .statefbk import gram

# Hankel Singular Value Decomposition
#   The following returns the Hankel singular values, which are singular values
#of the matrix formed by multiplying the controllability and observability
#grammians
def hsvd(sys):
    """Calculate the Hankel singular values.

    Parameters
    ----------
    sys : StateSpace
        A state space system

    Returns
    -------
    H : Matrix
        A list of Hankel singular values

    See Also
    --------
    gram

    Notes
    -----
    The Hankel singular values are the singular values of the Hankel operator.
    In practice, we compute the square root of the eigenvalues of the matrix
    formed by taking the product of the observability and controllability
    gramians.  There are other (more efficient) methods based on solving the
    Lyapunov equation in a particular way (more details soon).

    Examples
    --------
    >>> H = hsvd(sys)

    """
    # TODO: implement for discrete time systems
    if (isdtime(sys, strict=True)):
        raise NotImplementedError("Function not implemented in discrete time")

    Wc = gram(sys,'c')
    Wo = gram(sys,'o')
    WoWc = np.dot(Wo, Wc)
    w, v = np.linalg.eig(WoWc)

    hsv = np.sqrt(w)
    hsv = np.matrix(hsv)
    hsv = np.sort(hsv)
    hsv = np.fliplr(hsv)
    # Return the Hankel singular values
    return hsv

def modred(sys, ELIM, method='matchdc'):
    """
    Model reduction of `sys` by eliminating the states in `ELIM` using a given
    method.

    Parameters
    ----------
    sys: StateSpace
        Original system to reduce
    ELIM: array
        Vector of states to eliminate
    method: string
        Method of removing states in `ELIM`: either ``'truncate'`` or
        ``'matchdc'``.

    Returns
    -------
    rsys: StateSpace
        A reduced order model

    Raises
    ------
    ValueError
        * if `method` is not either ``'matchdc'`` or ``'truncate'``
        * if eigenvalues of `sys.A` are not all in left half plane
          (`sys` must be stable)

    Examples
    --------
    >>> rsys = modred(sys, ELIM, method='truncate')
    """

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    if (isctime(sys)):
        dico = 'C'
    else:
        raise NotImplementedError("Function not implemented in discrete time")


    #Check system is stable
    D,V = np.linalg.eig(sys.A)
    for e in D:
        if e.real >= 0:
            raise ValueError("Oops, the system is unstable!")
    ELIM = np.sort(ELIM)
    NELIM = []
    # Create list of elements not to eliminate (NELIM)
    for i in range(0,len(sys.A)):
        if i not in ELIM:
            NELIM.append(i)
    # A1 is a matrix of all columns of sys.A not to eliminate
    A1 = sys.A[:,NELIM[0]]
    for i in NELIM[1:]:
        A1 = np.hstack((A1, sys.A[:,i]))
    A11 = A1[NELIM,:]
    A21 = A1[ELIM,:]
    # A2 is a matrix of all columns of sys.A to eliminate
    A2 = sys.A[:,ELIM[0]]
    for i in ELIM[1:]:
        A2 = np.hstack((A2, sys.A[:,i]))
    A12 = A2[NELIM,:]
    A22 = A2[ELIM,:]

    C1 = sys.C[:,NELIM]
    C2 = sys.C[:,ELIM]
    B1 = sys.B[NELIM,:]
    B2 = sys.B[ELIM,:]

    A22I = np.linalg.inv(A22)

    if method=='matchdc':
        # if matchdc, residualize
        Ar = A11 - A12*A22.I*A21
        Br = B1 - A12*A22.I*B2
        Cr = C1 - C2*A22.I*A21
        Dr = sys.D - C2*A22.I*B2
    elif method=='truncate':
        # if truncate, simply discard state x2
        Ar = A11
        Br = B1
        Cr = C1
        Dr = sys.D
    else:
        raise ValueError("Oops, method is not supported!")

    rsys = StateSpace(Ar,Br,Cr,Dr)
    return rsys

def balred(sys, orders, method='truncate'):
    """
    Balanced reduced order model of sys of a given order.
    States are eliminated based on Hankel singular value.

    Parameters
    ----------
    sys: StateSpace
        Original system to reduce
    orders: integer or array of integer
        Desired order of reduced order model (if a vector, returns a vector
        of systems)
    method: string
        Method of removing states, either ``'truncate'`` or ``'matchdc'``.

    Returns
    -------
    rsys: StateSpace
        A reduced order model

    Raises
    ------
    ValueError
        * if `method` is not ``'truncate'``
        * if eigenvalues of `sys.A` are not all in left half plane
          (`sys` must be stable)
    ImportError
        if slycot routine ab09ad is not found

    Examples
    --------
    >>> rsys = balred(sys, order, method='truncate')

    """

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    dico = 'C'

    #Check system is stable
    D,V = np.linalg.eig(sys.A)
    # print D.shape
    # print D
    for e in D:
        if e.real >= 0:
            raise ValueError("Oops, the system is unstable!")

    if method=='matchdc':
        raise ValueError ("MatchDC not yet supported!")
    elif method=='truncate':
        try:
            from slycot import ab09ad
        except ImportError:
            raise ControlSlycot("can't find slycot subroutine ab09ad")
        job = 'B' # balanced (B) or not (N)
        equil = 'N'  # scale (S) or not (N)
        n = np.size(sys.A,0)
        m = np.size(sys.B,1)
        p = np.size(sys.C,0)
        Nr, Ar, Br, Cr, hsv = ab09ad(dico,job,equil,n,m,p,sys.A,sys.B,sys.C,nr=orders,tol=0.0)

        rsys = StateSpace(Ar, Br, Cr, sys.D)
    else:
        raise ValueError("Oops, method is not supported!")

    return rsys

def minreal(sys, tol=None, verbose=True):
    '''
    Eliminates uncontrollable or unobservable states in state-space
    models or cancelling pole-zero pairs in transfer functions. The
    output sysr has minimal order and the same response
    characteristics as the original model sys.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Original system
    tol: real
        Tolerance
    verbose: bool
        Print results if True

    Returns
    -------
    rsys: StateSpace or TransferFunction
        Cleaned model
    '''
    sysr = sys.minreal(tol)
    if verbose:
        print("{nstates} states have been removed from the model".format(
                nstates=len(sys.pole()) - len(sysr.pole())))
    return sysr

def era(YY, m, n, nin, nout, r):
    """
    Calculate an ERA model of order `r` based on the impulse-response data `YY`.

    .. note:: This function is not implemented yet.

    Parameters
    ----------
    YY: array
        `nout` x `nin` dimensional impulse-response data
    m: integer
        Number of rows in Hankel matrix
    n: integer
        Number of columns in Hankel matrix
    nin: integer
        Number of input variables
    nout: integer
        Number of output variables
    r: integer
        Order of model

    Returns
    -------
    sys: StateSpace
        A reduced order model sys=ss(Ar,Br,Cr,Dr)

    Examples
    --------
    >>> rsys = era(YY, m, n, nin, nout, r)
    """
    raise NotImplementedError('This function is not implemented yet.')

def markov(Y, U, M):
    """
    Calculate the first `M` Markov parameters [D CB CAB ...]
    from input `U`, output `Y`.

    Parameters
    ----------
    Y: array_like
        Output data
    U: array_like
        Input data
    M: integer
        Number of Markov parameters to output

    Returns
    -------
    H: matrix
        First M Markov parameters

    Notes
    -----
    Currently only works for SISO

    Examples
    --------
    >>> H = markov(Y, U, M)
    """

    # Convert input parameters to matrices (if they aren't already)
    Ymat = np.mat(Y)
    Umat = np.mat(U)
    n = np.size(U)

    # Construct a matrix of control inputs to invert
    UU = Umat
    for i in range(1, M-1):
        newCol = np.vstack((0, UU[0:n-1,i-2]))
        UU = np.hstack((UU, newCol))
    Ulast = np.vstack((0, UU[0:n-1,M-2]))
    for i in range(n-1,0,-1):
        Ulast[i] = np.sum(Ulast[0:i-1])
    UU = np.hstack((UU, Ulast))

    # Invert and solve for Markov parameters
    H = UU.I
    H = np.dot(H, Y)

    return H

