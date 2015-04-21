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

"""
==============================
Python Control Systems Library
==============================

The Python Control Systems Library (`control`) provides common functions
for analyzing and designing feedback control systems.

System creation
===============
.. autosummary::
    :toctree: generated/

    ss
    tf

Frequency domain plotting
=========================

.. autosummary::
    :toctree: generated/

    bode
    bode_plot
    nyquist
    nyquist_plot
    gangof4
    gangof4_plot
    nichols
    nichols_plot

Time domain simulation
======================

.. autosummary::
    :toctree: generated/

    forced_response
    impulse_response
    initial_response
    step_response
    phase_plot

.. _time-series-convention:

Convention for Time Series
--------------------------

This is a convention for function arguments and return values that
represent time series: sequences of values that change over time. It
is used throughout the library, for example in the functions
:func:`forced_response`, :func:`step_response`, :func:`impulse_response`,
and :func:`initial_response`.

.. note::
    This convention is different from the convention used in the library
    :mod:`scipy.signal`. In Scipy's convention the meaning of rows and columns
    is interchanged.  Thus, all 2D values must be transposed when they are
    used with functions from :mod:`scipy.signal`.

Types:

    * **Arguments** can be **arrays**, **matrices**, or **nested lists**.
    * **Return values** are **arrays** (not matrices).

The time vector is either 1D, or 2D with shape (1, n)::

      T = [[t1,     t2,     t3,     ..., tn    ]]

Input, state, and output all follow the same convention. Columns are different
points in time, rows are different components. When there is only one row, a
1D object is accepted or returned, which adds convenience for SISO systems::

      U = [[u1(t1), u1(t2), u1(t3), ..., u1(tn)]
           [u2(t1), u2(t2), u2(t3), ..., u2(tn)]
           ...
           ...
           [ui(t1), ui(t2), ui(t3), ..., ui(tn)]]

      Same for X, Y

So, U[:,2] is the system's input at the third point in time; and U[1] or U[1,:]
is the sequence of values for the system's second input.

The initial conditions are either 1D, or 2D with shape (j, 1)::

     X0 = [[x1]
           [x2]
           ...
           ...
           [xj]]

As all simulation functions return *arrays*, plotting is convenient::

    t, y = step(sys)
    plot(t, y)

The output of a MIMO system can be plotted like this::

    t, y, x = lsim(sys, u, t)
    plot(t, y[0], label='y_0')
    plot(t, y[1], label='y_1')

The convention also works well with the state space form of linear systems. If
``D`` is the feedthrough *matrix* of a linear system, and ``U`` is its input
(*matrix* or *array*), then the feedthrough part of the system's response,
can be computed like this::

    ft = D * U


Block diagram algebra
=====================
.. autosummary::
    :toctree: generated/

    series
    parallel
    feedback
    negate

Control system analysis
=======================
.. autosummary::
    :toctree: generated/

    dcgain
    evalfr
    freqresp
    margin
    stability_margins
    phase_crossover_frequencies
    pole
    zero
    pzmap
    root_locus

Matrix computations
===================
.. autosummary::
    :toctree: generated/

    care
    dare
    lyap
    dlyap
    ctrb
    obsv
    gram

Control system synthesis
========================
.. autosummary::
    :toctree: generated/

    acker
    lqr
    place

Model simplification tools
==========================
.. autosummary::
    :toctree: generated/

    minreal
    balred
    hsvd
    modred
    era
    markov

Utility functions and conversions
=================================
.. autosummary::
    :toctree: generated/

    unwrap
    db2mag
    mag2db
    drss
    isctime
    isdtime
    issys
    pade
    sample_system
    canonical_form
    reachable_form
    ss2tf
    ssdata
    tf2ss
    tfdata
    timebase
    timebaseEqual

"""

# Import functions from within the control system library
# Should probably only import the exact functions we use...
from .bdalg import series, parallel, negate, feedback
from .delay import pade
from .dtime import sample_system
from .freqplot import bode_plot, nyquist_plot, gangof4_plot
from .freqplot import bode, nyquist, gangof4
from .lti import issiso, timebase, timebaseEqual, isdtime, isctime
from .margins import stability_margins, phase_crossover_frequencies
from .mateqn import lyap, dlyap, care, dare
from .modelsimp import hsvd, modred, balred, era, markov, minreal
from .nichols import nichols_plot, nichols
from .phaseplot import phase_plot, box_grid
from .pzmap import pzmap
from .rlocus import root_locus
from .statefbk import place, lqr, ctrb, obsv, gram, acker
from .statesp import StateSpace
from .timeresp import forced_response, initial_response, step_response, \
    impulse_response
from .xferfcn import TransferFunction
from .ctrlutil import *
from .frdata import FRD
from .canonical import canonical_form, reachable_form

# Exceptions
from .exception import *

# Version information
try:
    from ._version import __version__, __commit__
except ImportError:
    __version__ = "dev"

# Import some of the more common (and benign) MATLAB shortcuts
# By default, don't import conflicting commands here
#! TODO (RMM, 4 Nov 2012): remove MATLAB dependencies from __init__.py
#!
#! Eventually, all functionality should be in modules *other* than matlab.
#! This will allow inclusion of the matlab module to set up a different set
#! of defaults from the main package.  At that point, the matlab module will
#! allow provide compatibility with MATLAB but no package functionality.
#!
from .matlab import ss, tf, ss2tf, tf2ss, drss
from .matlab import pole, zero, evalfr, freqresp, dcgain
from .matlab import nichols, rlocus, margin
        # bode and nyquist come directly from freqplot.py
from .matlab import step, impulse, initial, lsim
from .matlab import ssdata, tfdata

# The following is to use Numpy's testing framework
# Tests go under directory tests/, benchmarks under directory benchmarks/
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
