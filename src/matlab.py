# matlab.py - MATLAB emulation functions
#
# Author: Richard M. Murray
# Date: 29 May 09
# 
# This file contains a number of functions that emulate some of the
# functionality of MATLAB.  The intent of these functions is to
# provide a simple interface to the python control systems library
# (python-control) for people who are familiar with the MATLAB Control
# Systems Toolbox (tm).  Most of the functions are just calls to
# python-control functions defined elsewhere.  Use 'from
# control.matlab import *' in python to include all of the functions
# defined here.  Functions that are defined in other libraries that
# have the same names as their MATLAB equivalents are automatically
# imported here.
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
# $Id: statesp.py 808 2009-05-26 19:43:01Z murray $

# Libraries that we make use of 
import scipy as sp              # SciPy library (used all over)
import scipy.signal as signal   # Signal processing library

# Import MATLAB-like functions that are defined in other packages
from scipy.signal import zpk2ss, ss2zpk
from scipy.signal import lsim, impulse, step
from scipy import linspace, logspace

# Import MATLAB-like functions that belong elsewhere in python-control
from ctrlutil import unwrap
from freqplot import bode, nyquist, gangof4
from statesp import StateSpace
from xferfcn import TransferFunction
from bdalg import series, parallel, negate, feedback
from pzmap import pzmap

# Create a state space system from appropriate matrices
def ss(A, B, C, D):
    return StateSpace(A, B, C, D)

# Functions for creating a transfer function
def tf(num, den): 
    return TransferFunction(num, den)

# Function for converting state space to transfer function
def ss2tf(*args, **keywords):
    if (len(args) == 4):
        # Assume we were given the A, B, C, D matrix
        return TransferFunction(*args)
    elif (len(args) == 1):
        # Assume we were given a system object (lti or StateSpace)
        sys = args[0]
        return TransferFunction(sys.A, sys.B, sys.C, sys.D)
    else:
        raise ValueError, "Needs 1 or 4 arguments."

# Function for converting transfer function to state space
def tf2ss(*args, **keywords):
    if (len(args) == 2):
        # Assume we were given the num, den
        return TransferFunction(*args)
    elif (len(args) == 1):
        # Assume we were given a system object (lti or TransferFunction)
        sys = args[0]
        #! Should check to make sure object is a transfer function
        return StateSpace(sys.A, sys.B, sys.C, sys.D)
    else:
        raise ValueError, "Needs 1 or 2 arguments."

# Frequency response is handled by the system object
def freqresp(H, omega): return H.freqresp(omega)
