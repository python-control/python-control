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

# Hankel Singular Value Decomposition
#   The following returns the Hankel singular values, which are singular values of the matrix formed by multiplying the controllability and observability grammians
def hsvd(sys):
    """Calculate the Hankel singular values

    Usage
    =====
    H = hsvd(sys)

    Inputs
    ------
    sys : a state space system 

    Outputs
    -------
    H : a list of Hankel singular values 

    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb01bd'")
 
    H = 1.
    # Return the Hankel singular values
    return H

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

    Inputs
    ------
    Y : output data 
    U : input data
    M : number of Markov parameters to output

    Outputs
    -------
    H : first M Markov parameters

    """

