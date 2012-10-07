# __init__.py - initialization for control systems toolbox
#
# Author: Richard M. Murray
# Date: 24 May 09
#
# This file contains the initialization information from the control package.
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

"""Control System Library

The Python Control System Library (control) provides common functions
for analyzing and designing feedback control systems.

Common functions
----------------
tf      create a transfer function from num, den coefficients
ss      create a state space system from A, B, C, D matrices
pzk     create a transfer function from pole, zero, gain
bode    generate a Bode plot for a linear I/O system
nyquist generate a Nyquist plot for a linear I/O system
lqr     linear quadratic regulator
lqe     linear quadratic estimator
"""

# Import functions from within the control system library
#! Should probably only import the exact functions we use...
from bdalg import series, parallel, negate, feedback
from delay import pade
from dtime import sample_system
from freqplot import bode_plot, nyquist_plot, gangof4_plot
from freqplot import bode, nyquist, gangof4
from lti import timebase, timebaseEqual, isdtime, isctime
from margins import stability_margins, phase_crossover_frequencies
from mateqn import lyap, dlyap, care, dare
from modelsimp import hsvd, modred, balred, era, markov
from nichols import nichols_plot, nichols
from phaseplot import phase_plot, box_grid
from rlocus import root_locus
from statefbk import place, lqr, ctrb, obsv, gram, acker
from statesp import StateSpace
from timeresp import forced_response, initial_response, step_response, \
    impulse_response
from xferfcn import TransferFunction

# Import some of the more common (and benign) MATLAB shortcuts
# By default, don't import conflicting commands here
from matlab import ss, tf, ss2tf, tf2ss, drss
from matlab import pole, zero, evalfr, freqresp, dcgain
from matlab import nichols, rlocus, margin
        # bode and nyquist come directly from freqplot.py
from matlab import step, impulse, initial, lsim
from matlab import ssdata, tfdata
