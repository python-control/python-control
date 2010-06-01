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
# $Id$

# Libraries that we make use of 
import scipy as sp              # SciPy library (used all over)
import scipy.signal as signal   # Signal processing library

# Import MATLAB-like functions that are defined in other packages
from scipy.signal import zpk2ss, ss2zpk, tf2zpk, zpk2tf
from scipy.signal import lsim, impulse, step
from scipy import linspace, logspace

# Import MATLAB-like functions that belong elsewhere in python-control
from ctrlutil import unwrap
from freqplot import bode, nyquist, gangof4
from statesp import StateSpace
from xferfcn import TransferFunction
from bdalg import series, parallel, negate, feedback
from pzmap import pzmap
from statefbk import place, lqr

__doc__ = """
The control.matlab module defines functions that are roughly the
equivalents of those in the MATLAB Control Toolbox.  Items marked by a 
'*' are currently implemented; those marked with a '-' are not planned
for implementation.

Creating linear models.
* tf             - create transfer function (TF) models
  zpk            - create zero/pole/gain (ZPK) models.
* ss             - create state-space (SS) models
  dss            - create descriptor state-space models
  delayss        - create state-space models with delayed terms
  frd            - create frequency response data (FRD) models
  lti/exp        - create pure continuous-time delays (TF and ZPK only)
  filt           - specify digital filters
- lti/set        - set/modify properties of LTI models
- setdelaymodel  - specify internal delay model (state space only)
    
Data extraction
   lti/tfdata       - extract numerators and denominators
   lti/zpkdata      - extract zero/pole/gain data
   lti/ssdata       - extract state-space matrices
   lti/dssdata      - descriptor version of SSDATA
   frd/frdata       - extract frequency response data
   lti/get          - access values of LTI model properties
   ss/getDelayModel - access internal delay model (state space only)
 
Conversions
   tf             - conversion to transfer function
   zpk            - conversion to zero/pole/gain
   ss             - conversion to state space
   frd            - conversion to frequency data
   c2d            - continuous to discrete conversion
   d2c            - discrete to continuous conversion
   d2d            - resample discrete-time model
   upsample       - upsample discrete-time LTI systems
   ss2tf          - state space to transfer function
   ss2zpk         - transfer function to zero-pole-gain
   tf2ss          - transfer function to state space
   tf2zpk         - transfer function to zero-pole-gain
   zpk2ss         - zero-pole-gain to state space
   zpk2tf         - zero-pole-gain to transfer function
 
System interconnections
   append         - group LTI models by appending inputs and outputs
*  parallel       - connect LTI models in parallel (see also overloaded +)
*  series         - connect LTI models in series (see also overloaded *)
*  feedback       - connect lti models with a feedback loop
   lti/lft        - generalized feedback interconnection
   lti/connect    - arbitrary interconnection of lti models
   sumblk         - specify summing junction (for use with connect)
   strseq         - builds sequence of indexed strings (for I/O naming)
 
System gain and dynamics
   dcgain         - steady-state (D.C.) gain
   lti/bandwidth  - system bandwidth
   lti/norm       - h2 and Hinfinity norms of LTI models
   lti/pole       - system poles
   lti/zero       - system (transmission) zeros
   lti/order      - model order (number of states)
*  pzmap          - pole-zero map
   lti/iopzmap    - input/output pole-zero map
   damp           - natural frequency and damping of system poles
   esort          - sort continuous poles by real part
   dsort          - sort discrete poles by magnitude
   lti/stabsep    - stable/unstable decomposition
   lti/modsep     - region-based modal decomposition
 
Time-domain analysis
*  step           - step response
   stepinfo       - step response characteristics (rise time, ...)
*  impulse        - impulse response
   initial        - free response with initial conditions
*  lsim           - response to user-defined input signal
   lsiminfo       - linear response characteristics
   gensig         - generate input signal for LSIM
   covar          - covariance of response to white noise
 
Frequency-domain analysis
*  bode           - Bode plot of the frequency response
   lti/bodemag    - Bode magnitude diagram only
   sigma          - singular value frequency plot
*  nyquist        - Nyquist plot
   nichols        - Nichols plot
   margin         - gain and phase margins
   lti/allmargin  - all crossover frequencies and related gain/phase margins
   lti/freqresp   - frequency response over a frequency grid
   lti/evalfr     - evaluate frequency response at given frequency
 
Model simplification
   minreal        - minimal realization and pole/zero cancellation
   ss/sminreal    - structurally minimal realization (state space)
   lti/hsvd       - hankel singular values (state contributions)
   lti/balred     - reduced-order approximations of LTI models
   ss/modred      - model order reduction
 
Compensator design
   rlocus         - evans root locus
   place          - pole placement
   estim          - form estimator given estimator gain
   reg            - form regulator given state-feedback and estimator gains
 
LQR/LQG design
   ss/lqg         - single-step LQG design
   lqr, dlqr      - linear-Quadratic (LQ) state-feedback regulator
   lqry           - lq regulator with output weighting
   lqrd           - discrete LQ regulator for continuous plant
   ss/lqi         - linear-Quadratic-Integral (LQI) controller
   ss/kalman      - Kalman state estimator
   ss/kalmd       - discrete Kalman estimator for continuous plant
   ss/lqgreg      - build LQG regulator from LQ gain and Kalman estimator
   ss/lqgtrack    - build LQG servo-controller
   augstate       - augment output by appending states
 
State-space (SS) models
   rss            - random stable continuous-time state-space models
   drss           - random stable discrete-time state-space models
   ss2ss          - state coordinate transformation
   canon          - canonical forms of state-space models
   ctrb           - controllability matrix
   obsv           - observability matrix
   gram           - controllability and observability gramians
   ss/prescale    - optimal scaling of state-space models.  
   balreal        - gramian-based input/output balancing
   ss/xperm       - reorder states.   
 
Frequency response data (FRD) models
   frd/chgunits   - change frequency vector units
   frd/fcat       - merge frequency responses
   frd/fselect    - select frequency range or subgrid
   frd/fnorm      - peak gain as a function of frequency
   frd/abs        - entrywise magnitude of the frequency response
   frd/real       - real part of the frequency response
   frd/imag       - imaginary part of the frequency response
   frd/interp     - interpolate frequency response data
   mag2db         - convert magnitude to decibels (dB)
   db2mag         - convert decibels (dB) to magnitude
 
Time delays
   lti/hasdelay   - true for models with time delays
   lti/totaldelay - total delay between each input/output pair
   lti/delay2z    - replace delays by poles at z=0 or FRD phase shift
   pade           - pade approximation of time delays
 
Model dimensions and characteristics
   class          - model type ('tf', 'zpk', 'ss', or 'frd')
   isa            - test if model is of given type
   tf/size        - model sizes
   lti/ndims      - number of dimensions
   lti/isempty    - true for empty models
   lti/isct       - true for continuous-time models
   lti/isdt       - true for discrete-time models
   lti/isproper   - true for proper models
   lti/issiso     - true for single-input/single-output models
   lti/isstable   - true for models with stable dynamics
   lti/reshape    - reshape array of linear models
 
Overloaded arithmetic operations
*  + and -        - add and subtract systems (parallel connection)
*  *              - multiply systems (series connection)
*  /              - left divide -- sys1\sys2 means inv(sys1)*sys2
*  /              - right divide -- sys1/sys2 means sys1*inv(sys2)
   ^              - powers of a given system
   '              - pertransposition
   .'             - transposition of input/output map
   .*             - element-by-element multiplication
   [..]           - concatenate models along inputs or outputs
   lti/stack      - stack models/arrays along some array dimension
   lti/inv        - inverse of an LTI system
   lti/conj       - complex conjugation of model coefficients
 
Matrix equation solvers and linear algebra
  lyap, dlyap         - solve Lyapunov equations
  lyapchol, dlyapchol - square-root Lyapunov solvers
  care, dare          - solve algebraic Riccati equations
  gcare, gdare        - generalized Riccati solvers
  bdschur             - block diagonalization of a square matrix

Additional functions
* gangof4       - generate the Gang of 4 sensitivity plots
* linspace      - generate a set of numbers that are linearly spaced
* logspace      - generate a set of numbers that are logarithmically spaced
* unwrap        - unwrap a phase angle to give a continuous curve
"""

# Create a state space system from appropriate matrices
def ss(A, B, C, D):
    """Create a state space system from A, B, C, D"""
    return StateSpace(A, B, C, D)

# Functions for creating a transfer function
def tf(num, den): 
    """Create a SISO transfer function given the numerator and denominator"""
    return TransferFunction(num, den)

# Function for converting state space to transfer function
def ss2tf(*args, **keywords):
    """Transform a state space system to a transfer function
    
    Usage
    =====
    ss2tf(A, B, C, D)
    ss2tf(sys) - sys should have attributes A, B, C, D
    """
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
    """Transform a transfer function to a state space system
    
    Usage
    =====
    tf2ss(num, den)
    ss2tf(sys) - sys should be a system object (lti or TransferFunction)
    """
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
def freqresp(H, omega): 
    """Return the frequency response for an object H at frequency omega"""
    return H.freqresp(omega)
