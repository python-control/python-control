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

# External packages and modules
import numpy as np
import ctrlutil
from control.exception import *
from statefbk import *

# Hankel Singular Value Decomposition
#   The following returns the Hankel singular values, which are singular values of the matrix formed by multiplying the controllability and observability grammians
def hsvd(sys):
    """Calculate the Hankel singular values

    Usage
    =====
    H = hsvd(sys)

    The Hankel singular values are the singular values of the Hankel operator.  In practice, we compute the square root of the eigenvalues of the matrix formed by taking the product of the observability and controllability gramians.  There are other (more efficient) methods based on solving the Lyapunov equation in a particular way (more details soon).  

    Inputs
    ------
    sys : a state space system 

    Outputs
    -------
    H : a list of Hankel singular values 

    """

    Wc = gram(sys,'c')
    Wo = gram(sys,'o')

    WoWc = np.dot(Wo, Wc)
    w, v = np.linalg.eig(WoWc)

    hsv = np.sqrt(w)
    hsv = np.matrix(hsv)
    # Return the Hankel singular values
    return hsv

def modred(sys,ELIM,method):
    """Model reduction of sys by eliminating the states in ELIM using a given method

    Usage
    =====
    rsys = modred(sys,ELIM,method) 

    Inputs
    ======
    sys : original system to reduce
    ELIM : vector of states to eliminate
    method : method of removing states in ELIM (truncate or matchdc)

    Outputs
    =======
    rsys : a reduced order model 

    """

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    dico = 'C'

    #TODO: Check system is stable, perhaps a utility in ctrlutil.py
        # or a method of the StateSpace class?
    D,V = np.linalg.eig(sys.A)
    for e in D:
        if e.real >= 0:
            raise ValueError, "Oops, the system is unstable!"

    if method=='matchdc':
        print "matchdc"
    elif method=='truncate':
        print "truncate"
    else:
        raise ValueError, "Oops, method is not supported!"

    #Compute rhs using the Slycot routine XXXXXX 
        #make sure Slycot is installed
    #try:
    #    from slycot import XXXXXX
    #except ImportError:
    #    raise ControlSlycot("can't find slycot module 'XXXXXX'")
    rsys = 0.
    return rsys

def balred(sys,orders,elimination,method):
    """Balanced reduced order model of sys of a given order.  States are eliminated based on Hankel singular value.

    Usage
    =====
    rsys = balred(sys,order,elimination,method) 

    Inputs
    ======
    sys : original system to reduce
    orders : desired order of reduced order model (if a vector, returns a vector of systems)
    elimination : if elimination is specified, use 'method'
    method : method of removing states (truncate or matchdc)

    Outputs
    =======
    rsys : a reduced order model 

    """

    #Check for ss system object, need a utility for this?

    #TODO: Check for continous or discrete, only continuous supported right now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    dico = 'C'

    #TODO: Check system is stable, perhaps a utility in ctrlutil.py
        # or a method of the StateSpace class?
    D,V = np.linalg.eig(sys.A)
    for e in D:
        if e.real >= 0:
            raise ValueError, "Oops, the system is unstable!"
   
    if method=='matchdc':
        print "matchdc"
    elif method=='truncate':
        print "truncate"
    else:
        raise ValueError, "Oops, method is not supported!"

    #Compute rhs using the Slycot routine XXXXXX 
        #make sure Slycot is installed
    #try:
    #    from slycot import XXXXXX
    #except ImportError:
    #    raise ControlSlycot("can't find slycot module 'XXXXXX'")
    rsys = 0.
    return rsys


def era(YY,m,n,nin,nout,r):
    """Calculate an ERA model of order r based on the impulse-response data YY

    Usage
    =====
    sys = era(YY,m,n,nin,nout,r)

    Inputs
    ------
    YY : nout x nin dimensional impulse-response data
    m  : number of rows in Hankel matrix
    n  : number of columns in Hankel matrix
    nin : number of input variables
    nout : number of output variables
    r : order of model

    Outputs
    -------
    sys : a reduced order model sys=ss(Ar,Br,Cr,Dr) 

    """
def markov(Y,U,M):
    """Calculate the first M Markov parameters [D CB CAB ...] from input U, output Y

    Usage
    =====
    H = markov(Y,U,M)
    Currently only works for SISO

    Inputs
    ------
    Y : output data 
    U : input data
    M : number of Markov parameters to output

    Outputs
    -------
    H : first M Markov parameters

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
