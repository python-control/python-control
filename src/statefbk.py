# statefbk.py - tools for state feedback control
#
# Author: Richard M. Murray, Roberto Bucher
# Date: 31 May 2010
# 
# This file contains routines for designing state space controllers
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

# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues

    Usage
    =====
    K = place(A, B, p)

    Inputs
    ------
    A : 2-d array (dynamics matrix)
    B : 2-d array (input matrix)
    p : 1-d list of desired eigenvalue locations

    Outputs
    -------
    K : 2-d array with gains such that A - B K has given eigenvalues

    Example
    =======
    A = [[-1, -1], [0, 1]]
    B = [[0], [1]]
    K = place(A, B, [-2, -5])
    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A);
    B_mat = np.array(B);
    if (A_mat.shape[0] != A_mat.shape[1] or 
        A_mat.shape[0] != B_mat.shape[0]):
        raise ControlDimension("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A_mat)[0]
    placed_eigs = np.array(p);

    # SB01BD sets eigenvalues with real part less than alpha
    # We want to place all poles of the system => set alpha to minimum
    alpha = min(system_eigs.real);

    # Call SLICOT routine to place the eigenvalues
    A_z,w,nfp,nap,nup,F,Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], np.size(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, 'C');

    # Return the gain matrix, with MATLAB gain convention
    return -F

def lqr(*args, **keywords):
    """Linear quadratic regulator design

    Usage
    =====
    [K, S, E] = lqr(A, B, Q, R, [N])
    [K, S, E] = lqr(sys, Q, R, [N])

    The lqr() function computes the optimal state feedback controller
    that minimizes the quadratic cost

        J = \int_0^\infty x' Q x + u' R u + 2 x' N u

    Inputs
    ------
    A, B: 2-d arrays with dynamics and input matrices
    sys: linear I/O system 
    Q, R: 2-d array with state and input weight matrices
    N: optional 2-d array with cross weight matrix

    Outputs
    -------
    K: 2-d array with state feedback gains
    S: 2-d array with solution to Riccati equation
    E: 1-d array with eigenvalues of the closed loop system
    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb02md
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md' or 'sb02nt'")

    # 
    # Process the arguments and figure out what inputs we received
    #
    
    # Get the system description
    if (len(args) < 4):
        raise ControlArgument("not enough input arguments")

    elif (ctrlutil.issys(args[0])):
        # We were passed a system as the first argument; extract A and B
        #! TODO: really just need to check for A and B attributes
        A = np.array(args[0].A, ndmin=2, dtype=float);
        B = np.array(args[0].B, ndmin=2, dtype=float);
        index = 1;
    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float);
        B = np.array(args[1], ndmin=2, dtype=float);
        index = 2;

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float);
    R = np.array(args[index+1], ndmin=2, dtype=float);
    if (len(args) > index + 2): 
        N = np.array(args[index+2], ndmin=2, dtype=float);
    else:
        N = np.zeros((Q.shape[0], R.shape[1]));

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs or
          N.shape[0] != nstates or N.shape[1] != ninputs):
        raise ControlDimension("incorrect weighting matrix dimensions")

    # Compute the G matrix required by SB02MD
    A_b,B_b,Q_b,R_b,L_b,ipiv,oufact,G = \
        sb02mt(nstates, ninputs, B, R, A, Q, N, jobl='N');

    # Call the SLICOT function
    X,rcond,w,S,U,A_inv = sb02md(nstates, A_b, G, Q_b, 'C')

    # Now compute the return value
    K = np.dot(np.linalg.inv(R), (np.dot(B.T, X) + N.T));
    S = X;
    E = w[0:nstates];

    return K, S, E

def ctrb(A,B):       
    """Controllabilty matrix

    Usage
    =====
    Wc = ctrb(A, B)

    Inputs
    ------
    A, B: Dynamics and input matrix of the system

    Outputs
    -------
    Wc: Controllability matrix
    """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    bmat = np.mat(B)
    n = np.shape(amat)[0]

    # Construct the controllability matrix
    ctrb = bmat
    for i in range(1, n):
        ctrb = np.vstack((ctrb, amat**i*bmat))
    return ctrb

def obsv(A, C):       
    """Observability matrix

    Usage
    =====
    Wc = obsv(A, C)

    Inputs
    ------
    A, C: Dynamics and output matrix of the system

    Outputs
    -------
    Wc: Observability matrix
    """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    cmat = np.mat(C)
    n = np.shape(amat)[0]

    # Construct the controllability matrix
    obsv = cmat
    for i in range(1, n):
        obsv = np.hstack((obsv, cmat*amat**i))
    return obsv
