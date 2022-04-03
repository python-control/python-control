# -*- coding: utf-8 -*-
"""
The :mod:`control.matlab` module contains a number of functions that emulate
some of the functionality of MATLAB.  The intent of these functions is to
provide a simple interface to the python control systems library
(python-control) for people who are familiar with the MATLAB Control Systems
Toolbox (tm).
"""

"""Copyright (c) 2009 by California Institute of Technology
All rights reserved.

Copyright (c) 2011 by Eike Welk


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Richard M. Murray
Date: 29 May 09
Revised: Kevin K. Chen, Dec 10

$Id$

"""

# Import MATLAB-like functions that are defined in other packages
from scipy.signal import zpk2ss, ss2zpk, tf2zpk, zpk2tf
from numpy import linspace, logspace

# If configuration is not yet set, import and use MATLAB defaults
import sys
if not ('.config' in sys.modules):
    from .. import config
    config.use_matlab_defaults()

# Control system library
from ..statesp import *
from ..iosys import ss, rss, drss       # moved from .statesp
from ..xferfcn import *
from ..lti import *
from ..namedio import *
from ..frdata import *
from ..dtime import *
from ..exception import ControlArgument

# Import MATLAB-like functions that can be used as-is
from ..ctrlutil import *
from ..freqplot import gangof4
from ..nichols import nichols
from ..bdalg import *
from ..pzmap import *
from ..statefbk import *
from ..delay import *
from ..modelsimp import *
from ..mateqn import *
from ..margins import margin
from ..rlocus import rlocus
from ..dtime import c2d
from ..sisotool import sisotool

# Functions that are renamed in MATLAB
pole, zero = poles, zeros

# Import functions specific to Matlab compatibility package
from .timeresp import *
from .wrappers import *

# Set up defaults corresponding to MATLAB conventions
from ..config import *
use_matlab_defaults()

r"""
The following tables give an overview of the module ``control.matlab``.
They also show the implementation progress and the planned features of the
module.

The symbols in the first column show the current state of a feature:

* ``*`` : The feature is currently implemented.
* ``-`` : The feature is not planned for implementation.
* ``s`` : A similar feature from another library (Scipy) is imported into
  the module, until the feature is implemented here.


Creating linear models
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`tf`                  create transfer function (TF) models
\   zpk                         create zero/pole/gain (ZPK) models.
\*  :func:`ss`                  create state-space (SS) models
\   dss                         create descriptor state-space models
\   delayss                     create state-space models with delayed terms
\*  :func:`frd`                 create frequency response data (FRD) models
\   lti/exp                     create pure continuous-time delays (TF and
                                ZPK only)
\   filt                        specify digital filters
\-  lti/set                     set/modify properties of LTI models
\-  setdelaymodel               specify internal delay model (state space
                                only)
\*  :func:`rss`                 create a random continuous state space model
\*  :func:`drss`                create a random discrete state space model
==  ==========================  ============================================


Data extraction
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`tfdata`              extract numerators and denominators
\   lti/zpkdata                 extract zero/pole/gain data
\   lti/ssdata                  extract state-space matrices
\   lti/dssdata                 descriptor version of SSDATA
\   frd/frdata                  extract frequency response data
\   lti/get                     access values of LTI model properties
\   ss/getDelayModel            access internal delay model (state space)
==  ==========================  ============================================


Conversions
----------------------------------------------------------------------------

==  ============================   ============================================
\*  :func:`tf`                     conversion to transfer function
\   zpk                            conversion to zero/pole/gain
\*  :func:`ss`                     conversion to state space
\*  :func:`frd`                    conversion to frequency data
\*  :func:`c2d`                    continuous to discrete conversion
\   d2c                            discrete to continuous conversion
\   d2d                            resample discrete-time model
\   upsample                       upsample discrete-time LTI systems
\*  :func:`ss2tf`                  state space to transfer function
\s  :func:`~scipy.signal.ss2zpk`   transfer function to zero-pole-gain
\*  :func:`tf2ss`                  transfer function to state space
\s  :func:`~scipy.signal.tf2zpk`   transfer function to zero-pole-gain
\s  :func:`~scipy.signal.zpk2ss`   zero-pole-gain to state space
\s  :func:`~scipy.signal.zpk2tf`   zero-pole-gain to transfer function
==  ============================   ============================================


System interconnections
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`~control.append`     group LTI models by appending inputs/outputs
\*  :func:`~control.parallel`   connect LTI models in parallel
                                (see also overloaded ``+``)
\*  :func:`~control.series`     connect LTI models in series
                                (see also overloaded ``*``)
\*  :func:`~control.feedback`   connect lti models with a feedback loop
\   lti/lft                     generalized feedback interconnection
\*  :func:`~control.connect`    arbitrary interconnection of lti models
\   sumblk                      summing junction (for use with connect)
\   strseq                      builds sequence of indexed strings
                                (for I/O naming)
==  ==========================  ============================================


System gain and dynamics
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`dcgain`              steady-state (D.C.) gain
\   lti/bandwidth               system bandwidth
\   lti/norm                    h2 and Hinfinity norms of LTI models
\*  :func:`pole`                system poles
\*  :func:`zero`                system (transmission) zeros
\   lti/order                   model order (number of states)
\*  :func:`~control.pzmap`      pole-zero map (TF only)
\   lti/iopzmap                 input/output pole-zero map
\*  :func:`damp`                natural frequency, damping of system poles
\   esort                       sort continuous poles by real part
\   dsort                       sort discrete poles by magnitude
\   lti/stabsep                 stable/unstable decomposition
\   lti/modsep                  region-based modal decomposition
==  ==========================  ============================================


Time-domain analysis
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`step`                step response
\   stepinfo                    step response characteristics
\*  :func:`impulse`             impulse response
\*  :func:`initial`             free response with initial conditions
\*  :func:`lsim`                response to user-defined input signal
\   lsiminfo                    linear response characteristics
\   gensig                      generate input signal for LSIM
\   covar                       covariance of response to white noise
==  ==========================  ============================================


Frequency-domain analysis
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`bode`                Bode plot of the frequency response
\   lti/bodemag                 Bode magnitude diagram only
\   sigma                       singular value frequency plot
\*  :func:`~control.nyquist`    Nyquist plot
\*  :func:`~control.nichols`    Nichols plot
\*  :func:`margin`              gain and phase margins
\   lti/allmargin               all crossover frequencies and margins
\*  :func:`freqresp`            frequency response
\*  :func:`evalfr`              frequency response at complex frequency s
==  ==========================  ============================================


Model simplification
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`~control.minreal`    minimal realization; pole/zero cancellation
\   ss/sminreal                 structurally minimal realization
\*  :func:`~control.hsvd`       hankel singular values (state contributions)
\*  :func:`~control.balred`     reduced-order approximations of LTI models
\*  :func:`~control.modred`     model order reduction
==  ==========================  ============================================


Compensator design
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`rlocus`              evans root locus
\*  :func:`sisotool`            SISO controller design
\*  :func:`~control.place`      pole placement
\   estim                       form estimator given estimator gain
\   reg                         form regulator given state-feedback and
                                estimator gains
==  ==========================  ============================================


LQR/LQG design
----------------------------------------------------------------------------

==  ==========================  ============================================
\   ss/lqg                      single-step LQG design
\*  :func:`~control.lqr`        linear quadratic (LQ) state-fbk regulator
\   dlqr                        discrete-time LQ state-feedback regulator
\   lqry                        LQ regulator with output weighting
\   lqrd                        discrete LQ regulator for continuous plant
\   ss/lqi                      Linear-Quadratic-Integral (LQI) controller
\   ss/kalman                   Kalman state estimator
\   ss/kalmd                    discrete Kalman estimator for cts plant
\   ss/lqgreg                   build LQG regulator from LQ gain and Kalman
                                estimator
\   ss/lqgtrack                 build LQG servo-controller
\   augstate                    augment output by appending states
==  ==========================  ============================================


State-space (SS) models
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`rss`                 random stable cts-time state-space models
\*  :func:`drss`                random stable disc-time state-space models
\   ss2ss                       state coordinate transformation
\   canon                       canonical forms of state-space models
\*  :func:`~control.ctrb`       controllability matrix
\*  :func:`~control.obsv`       observability matrix
\*  :func:`~control.gram`       controllability and observability gramians
\   ss/prescale                 optimal scaling of state-space models.
\   balreal                     gramian-based input/output balancing
\   ss/xperm                    reorder states.
==  ==========================  ============================================


Frequency response data (FRD) models
----------------------------------------------------------------------------

==  ==========================  ============================================
\   frd/chgunits                change frequency vector units
\   frd/fcat                    merge frequency responses
\   frd/fselect                 select frequency range or subgrid
\   frd/fnorm                   peak gain as a function of frequency
\   frd/abs                     entrywise magnitude of frequency response
\   frd/real                    real part of the frequency response
\   frd/imag                    imaginary part of the frequency response
\   frd/interp                  interpolate frequency response data
\*  :func:`~control.mag2db`     convert magnitude to decibels (dB)
\*  :func:`~control.db2mag`     convert decibels (dB) to magnitude
==  ==========================  ============================================


Time delays
----------------------------------------------------------------------------

==  ==========================  ============================================
\   lti/hasdelay                true for models with time delays
\   lti/totaldelay              total delay between each input/output pair
\   lti/delay2z                 replace delays by poles at z=0 or FRD phase
                                shift
\*  :func:`~control.pade`       pade approximation of time delays
==  ==========================  ============================================


Model dimensions and characteristics
----------------------------------------------------------------------------

==  ==========================  ============================================
\   class                       model type ('tf', 'zpk', 'ss', or 'frd')
\   isa                         test if model is of given type
\   tf/size                     model sizes
\   lti/ndims                   number of dimensions
\   lti/isempty                 true for empty models
\   lti/isct                    true for continuous-time models
\   lti/isdt                    true for discrete-time models
\   lti/isproper                true for proper models
\   lti/issiso                  true for single-input/single-output models
\   lti/isstable                true for models with stable dynamics
\   lti/reshape                 reshape array of linear models
==  ==========================  ============================================

Overloaded arithmetic operations
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  \+ and -                    add, subtract systems (parallel connection)
\*  \*                          multiply systems (series connection)
\   /                           right divide -- sys1\*inv(sys2)
\-   \\                         left divide -- inv(sys1)\*sys2
\   ^                           powers of a given system
\   '                           pertransposition
\   .'                          transposition of input/output map
\   .\*                         element-by-element multiplication
\   [..]                        concatenate models along inputs or outputs
\   lti/stack                   stack models/arrays along some dimension
\   lti/inv                     inverse of an LTI system
\   lti/conj                    complex conjugation of model coefficients
==  ==========================  ============================================

Matrix equation solvers and linear algebra
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`~control.lyap`       solve continuous-time Lyapunov equations
\*  :func:`~control.dlyap`      solve discrete-time Lyapunov equations
\   lyapchol, dlyapchol         square-root Lyapunov solvers
\*  :func:`~control.care`       solve continuous-time algebraic Riccati
                                equations
\*  :func:`~control.dare`       solve disc-time algebraic Riccati equations
\   gcare, gdare                generalized Riccati solvers
\   bdschur                     block diagonalization of a square matrix
==  ==========================  ============================================


Additional functions
----------------------------------------------------------------------------

==  ==========================  ============================================
\*  :func:`~control.gangof4`    generate the Gang of 4 sensitivity plots
\*  :func:`~numpy.linspace`     generate a set of numbers that are linearly
                                spaced
\*  :func:`~numpy.logspace`     generate a set of numbers that are
                                logarithmically spaced
\*  :func:`~control.unwrap`     unwrap phase angle to give continuous curve
==  ==========================  ============================================

"""
