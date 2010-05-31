# statefbk.py - tools for state feedback control
#
# Author: Richard M. Murray
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
from control.exception import ControlSlycot, ControlDimension

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
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, 'C');

    # Return the gain matrix, with MATLAB gain convention
    return -F
