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
# Copyright (c) 2009 by California Institute of Technology
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

import scipy as sp
import xferfcn as tf
import statesp as ss

# Standard interconnections (implemented by objects)
def series(sys1, sys2): return sys1 * sys2
def parallel(sys1, sys2): return sys1 + sys2
def negate(sys): return -sys;

# Feedback interconnection between systems
#! This should be optimized for better performance
#! Needs to be updated to work for state space systems
def feedback(sys1, sys2, **keywords):
    # Grab keyword arguments
    signparm = keywords.pop("sign", -1);

    #
    # Sort out which set of functions to call
    #
    # The main cases we are interested in are those where we use a
    # constant for one of the arguments to the function, in which case
    # we should use the other argument to figure out what type of
    # object we are acting on.  Then call the feedback function for
    # that object.
    #

    if (isinstance(sys1, tf.TransferFunction) and
        (isinstance(sys2, tf.TransferFunction) or
         isinstance(sys2, (int, long, float, complex)))):
        # Use transfer function feedback function
        return sys1.feedback(sys2, sign=signparm)

    elif (isinstance(sys2, tf.TransferFunction) and
          isinstance(sys1, (int, long, float, complex))):
        # Convert sys1 to a transfer function and then perform operation
        sys = tf.convertToTransferFunction(sys1);
        return sys.feedback(sys2, sign=signparm)

    elif (isinstance(sys1, ss.StateSpace) and
          (isinstance(sys2, ss.StateSpace) or
           isinstance(sys2, (int, long, float, complex)))):
        # Use state space feedback function
        return sys1.feedback(sys2, sign=signparm)

    elif (isinstance(sys2, ss.StateSpace) and
          isinstance(sys1, (int, long, float, complex))):
        # Convert sys1 to state space system and then perform operation
        sys = ss.convertToStateSpace(sys1);
        return sys.feedback(sys2, sign=signparm)
    
    else:
        # Assume that the first system has the right member function
        return sys1.feedback(sys2, sign=signparm)
        
        raise TypeError("can't operate on give types")

