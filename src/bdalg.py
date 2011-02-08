# bdalg.py - functions for implmeenting block diagram algebra
#
# Author: Richard M. Murray
# Date: 24 May 09
# 
# This file contains some standard block diagram algebra.  If all
# arguments are SISO transfer functions, the results are a transfer
# function.  Otherwise, the computation is done in state space.
#
#! State space operations are not currently implemented.
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
# $Id: bdalg.py 17 2010-05-29 23:50:52Z murrayrm $

import scipy as sp
import xferfcn as tf
import statesp as ss

def feedback(sys1, sys2, sign=-1):
    """Feedback interconnection between two I/O systems.

    Usage
    =====
    sys = feedback(sys1, sys2)
    sys = feedback(sys1, sys2, sign)

    Compute the system corresponding to a feedback interconnection between
    sys1 and sys2.  When sign is not specified, it assumes a value of -1
    (negative feedback).  A sign of 1 indicates positive feedback.

    Parameters
    ----------
    sys1, sys2: linsys
        Linear input/output systems
    sign: scalar
        Feedback sign.

    Return values
    -------------
    sys: linsys

    Notes
    -----
    1. This function calls calls xferfcn.feedback if sys1 is a TransferFunction
       object and statesp.feedback if sys1 is a StateSpace object.  If sys1 is a
       scalar, then it is converted to sys2's type, and the corresponding
       feedback function is used.  If sys1 and sys2 are both scalars, then use
       xferfcn.feedback."""
  
    # Check for correct input types.
    if not isinstance(sys1, (int, long, float, complex, tf.xTransferFunction,
        ss.StateSpace)):
        raise TypeError("sys1 must be a TransferFunction or StateSpace object, \
or a scalar.")
    if not isinstance(sys2, (int, long, float, complex, tf.xTransferFunction,
        ss.StateSpace)):
        raise TypeError("sys2 must be a TransferFunction or StateSpace object, \
or a scalar.")

    # If sys1 is a scalar, convert it to the appropriate LTI type so that we can
    # its feedback member function.
    if isinstance(sys1, (int, long, float, complex)):
        if isinstance(sys2, tf.xTransferFunction):
            sys1 = tf.convertToTransferFunction(sys1)
        elif isinstance(sys2, ss.StateSpace):
            sys1 = ss.convertToStateSpace(sys1)
        else: # sys2 is a scalar.
            sys1 = tf.convertToTransferFunction(sys1)
            sys2 = tf.convertToTransferFunction(sys2)

    return sys1.feedback(sys2, sign)
