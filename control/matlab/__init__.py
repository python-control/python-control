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

# Libraries that we make use of
import scipy as sp              # SciPy library (used all over)
import numpy as np              # NumPy library
import re                       # regular expressions
from copy import deepcopy

# Import MATLAB-like functions that are defined in other packages
from scipy.signal import zpk2ss, ss2zpk, tf2zpk, zpk2tf
from numpy import linspace, logspace

# If configuration is not yet set, import and use MATLAB defaults
#! NOTE (RMM, 4 Nov 2012): MATLAB default initialization commented out for now
#!
#! This code will eventually be used so that import control.matlab will
#! automatically use MATLAB defaults, while import control will use package
#! defaults.  In order for that to work, we need to make sure that
#! __init__.py does not include anything in the MATLAB module.
# import sys
# if not ('.config' in sys.modules):
#     from . import config
#    config.use_matlab()

# Control system library
from .. import ctrlutil
from .. import freqplot
from .. import timeresp
from .. import margins
from ..statesp import *
from ..xferfcn import *
from ..lti import issiso
from ..frdata import FRD
from ..dtime import sample_system
from ..exception import ControlArgument
from .timeresp import *

# Import MATLAB-like functions that can be used as-is
from ..ctrlutil import unwrap
from ..freqplot import nyquist, gangof4
from ..nichols import nichols, nichols_grid
from ..bdalg import series, parallel, negate, feedback, append, connect
from ..pzmap import pzmap
from ..statefbk import ctrb, obsv, gram, place, lqr
from ..delay import pade
from ..modelsimp import hsvd, balred, modred, minreal
from ..mateqn import lyap, dlyap, dare, care

__doc__ += r"""
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
\*  :func:`freqresp`            frequency response over a frequency grid
\*  :func:`evalfr`              frequency response at single frequency
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


def frd(*args):
    '''
    Construct a Frequency Response Data model, or convert a system

    frd models store the (measured) frequency response of a system.

    This function can be called in different ways:

    ``frd(response, freqs)``
        Create an frd model with the given response data, in the form of
        complex response vector, at matching frequency freqs [in rad/s]

    ``frd(sys, freqs)``
        Convert an LTI system into an frd model with data at frequencies
        freqs.

    Parameters
    ----------
    response: array_like, or list
        complex vector with the system response
    freq: array_lik or lis
        vector with frequencies
    sys: LTI (StateSpace or TransferFunction)
        A linear system

    Returns
    -------
    sys: FRD
        New frequency response system

    See Also
    --------
    ss, tf
    '''
    return FRD(*args)



def pole(sys):
    """
    Compute system poles.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system

    Returns
    -------
    poles: ndarray
        Array that contains the system's poles.

    Raises
    ------
    NotImplementedError
        when called on a TransferFunction object

    See Also
    --------
    zero

    Notes
    -----
    This function is a wrapper for StateSpace.pole and
    TransferFunction.pole.

    """

    return sys.pole()

def zero(sys):
    """
    Compute system zeros.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system

    Returns
    -------
    zeros: ndarray
        Array that contains the system's zeros.

    Raises
    ------
    NotImplementedError
        when called on a TransferFunction object or a MIMO StateSpace object

    See Also
    --------
    pole

    Notes
    -----
    This function is a wrapper for StateSpace.zero and
    TransferFunction.zero.

    """

    return sys.zero()

def evalfr(sys, x):
    """
    Evaluate the transfer function of an LTI system for a single complex
    number x.

    To evaluate at a frequency, enter x = omega*j, where omega is the
    frequency in radians

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system
    x: scalar
        Complex number

    Returns
    -------
    fresp: ndarray

    See Also
    --------
    freqresp
    bode

    Notes
    -----
    This function is a wrapper for StateSpace.evalfr and
    TransferFunction.evalfr.

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> evalfr(sys, 1j)
    array([[ 44.8-21.4j]])
    >>> # This is the transfer function matrix evaluated at s = i.

    .. todo:: Add example with MIMO system
    """
    if issiso(sys):
        return sys.horner(x)[0][0]
    return sys.horner(x)


def freqresp(sys, omega):
    """
    Frequency response of an LTI system at multiple angular frequencies.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system
    omega: array_like
        List of frequencies

    Returns
    -------
    mag: ndarray
    phase: ndarray
    omega: list, tuple, or ndarray

    See Also
    --------
    evalfr
    bode

    Notes
    -----
    This function is a wrapper for StateSpace.freqresp and
    TransferFunction.freqresp.  The output omega is a sorted version of the
    input omega.

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = freqresp(sys, [0.1, 1., 10.])
    >>> mag
    array([[[ 58.8576682 ,  49.64876635,  13.40825927]]])
    >>> phase
    array([[[-0.05408304, -0.44563154, -0.66837155]]])

    .. todo::
        Add example with MIMO system

        #>>> sys = rss(3, 2, 2)
        #>>> mag, phase, omega = freqresp(sys, [0.1, 1., 10.])
        #>>> mag[0, 1, :]
        #array([ 55.43747231,  42.47766549,   1.97225895])
        #>>> phase[1, 0, :]
        #array([-0.12611087, -1.14294316,  2.5764547 ])
        #>>> # This is the magnitude of the frequency response from the 2nd
        #>>> # input to the 1st output, and the phase (in radians) of the
        #>>> # frequency response from the 1st input to the 2nd output, for
        #>>> # s = 0.1i, i, 10i.
    """

    return sys.freqresp(omega)

# Bode plots
def bode(*args, **keywords):
    """Bode plot of the frequency response

    Plots a bode gain and phase diagram

    Parameters
    ----------
    sys : LTI, or list of LTI
        System for which the Bode response is plotted and give. Optionally
        a list of systems can be entered, or several systems can be
        specified (i.e. several parameters). The sys arguments may also be
        interspersed with format strings. A frequency argument (array_like)
        may also be added, some examples:
        * >>> bode(sys, w)                    # one system, freq vector
        * >>> bode(sys1, sys2, ..., sysN)     # several systems
        * >>> bode(sys1, sys2, ..., sysN, w)
        * >>> bode(sys1, 'plotstyle1', ..., sysN, 'plotstyleN') # + plot formats
    omega: freq_range
        Range of frequencies in rad/s
    dB : boolean
        If True, plot result in dB
    Hz : boolean
        If True, plot frequency in Hz (omega must be provided in rad/sec)
    deg : boolean
        If True, return phase in degrees (else radians)
    Plot : boolean
        If True, plot magnitude and phase

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = bode(sys)

    .. todo::

        Document these use cases

        * >>> bode(sys, w)
        * >>> bode(sys1, sys2, ..., sysN)
        * >>> bode(sys1, sys2, ..., sysN, w)
        * >>> bode(sys1, 'plotstyle1', ..., sysN, 'plotstyleN')
    """

    # If the first argument is a list, then assume python-control calling format
    if (getattr(args[0], '__iter__', False)):
        return freqplot.bode(*args, **keywords)

    # Otherwise, run through the arguments and collect up arguments
    syslist = []; plotstyle=[]; omega=None;
    i = 0;
    while i < len(args):
        # Check to see if this is a system of some sort
        if (ctrlutil.issys(args[i])):
            # Append the system to our list of systems
            syslist.append(args[i])
            i += 1

            # See if the next object is a plotsytle (string)
            if (i < len(args) and isinstance(args[i], str)):
                plotstyle.append(args[i])
                i += 1

            # Go on to the next argument
            continue

        # See if this is a frequency list
        elif (isinstance(args[i], (list, np.ndarray))):
            omega = args[i]
            i += 1
            break

        else:
            raise ControlArgument("unrecognized argument type")

    # Check to make sure that we processed all arguments
    if (i < len(args)):
        raise ControlArgument("not all arguments processed")

    # Check to make sure we got the same number of plotstyles as systems
    if (len(plotstyle) != 0 and len(syslist) != len(plotstyle)):
        raise ControlArgument("number of systems and plotstyles should be equal")

    # Warn about unimplemented plotstyles
    #! TODO: remove this when plot styles are implemented in bode()
    #! TODO: uncomment unit test code that tests this out
    if (len(plotstyle) != 0):
        print("Warning (matabl.bode): plot styles not implemented");

    # Call the bode command
    return freqplot.bode(syslist, omega, **keywords)

# Nichols chart grid
def ngrid():
    nichols_grid()
ngrid.__doc__ = re.sub('nichols_grid', 'ngrid', nichols_grid.__doc__)

# Root locus plot
def rlocus(sys, klist = None, **keywords):
    """Root locus plot

    The root-locus plot has a callback function that prints pole location,
    gain and damping to the Python consol on mouseclicks on the root-locus
    graph.

    Parameters
    ----------
    sys: StateSpace or TransferFunction
        Linear system
    klist: iterable, optional
        optional list of gains
    xlim : control of x-axis range, normally with tuple, for
        other options, see matplotlib.axes
    ylim : control of y-axis range
    Plot : boolean (default = True)
        If True, plot magnitude and phase
    PrintGain: boolean (default = True)
        If True, report mouse clicks when close to the root-locus branches,
        calculate gain, damping and print

    Returns
    -------
    rlist:
        list of roots for each gain
    klist:
        list of gains used to compute roots
    """
    from ..rlocus import root_locus

    return root_locus(sys, klist, **keywords)

def margin(*args):
    """Calculate gain and phase margins and associated crossover frequencies

    Function ``margin`` takes either 1 or 3 parameters.

    Parameters
    ----------
    sys : StateSpace or TransferFunction
        Linear SISO system
    mag, phase, w : array_like
        Input magnitude, phase (in deg.), and frequencies (rad/sec) from
        bode frequency response data

    Returns
    -------
    gm, pm, Wcg, Wcp : float
        Gain margin gm, phase margin pm (in deg), gain crossover frequency
        (corresponding to phase margin) and phase crossover frequency
        (corresponding to gain margin), in rad/sec of SISO open-loop.
        If more than one crossover frequency is detected, returns the lowest
        corresponding margin.

    Examples
    --------
    >>> sys = tf(1, [1, 2, 1, 0])
    >>> gm, pm, Wcg, Wcp = margin(sys)

    """
    if len(args) == 1:
        sys = args[0]
        margin = margins.stability_margins(sys)
    elif len(args) == 3:
        margin = margins.stability_margins(args)
    else:
        raise ValueError("Margin needs 1 or 3 arguments; received %i."
            % len(args))

    return margin[0], margin[1], margin[4], margin[3]

def dcgain(*args):
    '''
    Compute the gain of the system in steady state.

    The function takes either 1, 2, 3, or 4 parameters:

    Parameters
    ----------
    A, B, C, D: array-like
        A linear system in state space form.
    Z, P, k: array-like, array-like, number
        A linear system in zero, pole, gain form.
    num, den: array-like
        A linear system in transfer function form.
    sys: LTI (StateSpace or TransferFunction)
        A linear system object.

    Returns
    -------
    gain: matrix
        The gain of each output versus each input:
        :math:`y = gain \cdot u`

    Notes
    -----
    This function is only useful for systems with invertible system
    matrix ``A``.

    All systems are first converted to state space form. The function then
    computes:

    .. math:: gain = - C \cdot A^{-1} \cdot B + D
    '''
    #Convert the parameters to state space form
    if len(args) == 4:
        A, B, C, D = args
        sys = ss(A, B, C, D)
    elif len(args) == 3:
        Z, P, k = args
        A, B, C, D = zpk2ss(Z, P, k)
        sys = ss(A, B, C, D)
    elif len(args) == 2:
        num, den = args
        sys = tf2ss(num, den)
    elif len(args) == 1:
        sys, = args
        sys = ss(sys)
    else:
        raise ValueError("Function ``dcgain`` needs either 1, 2, 3 or 4 "
                         "arguments.")
    #gain = - C * A**-1 * B + D
    gain = sys.D - sys.C * sys.A.I * sys.B
    return gain

def damp(sys, doprint=True):
    '''
    Compute natural frequency, damping and poles of a system

    The function takes 1 or 2 parameters

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        A linear system object
    doprint:
        if true, print table with values

    Returns
    -------
    wn: array
        Natural frequencies of the poles
    damping: array
        Damping values
    poles: array
        Pole locations

    See Also
    --------
    pole
    '''
    wn, damping, poles = sys.damp()
    if doprint:
        print('_____Eigenvalue______ Damping___ Frequency_')
        for p, d, w in zip(poles, damping, wn) :
            if abs(p.imag) < 1e-12:
                print("%10.4g            %10.4g %10.4g" %
                      (p.real, 1.0, -p.real))
            else:
                print("%10.4g%+10.4gj %10.4g %10.4g" %
                      (p.real, p.imag, d, w))
    return wn, damping, poles

# Simulation routines
# Call corresponding functions in timeresp, with arguments transposed

# Convert a continuous time system to a discrete time system
def c2d(sysc, Ts, method='zoh'):
    '''
    Return a discrete-time system

    Parameters
    ----------
    sysc: LTI (StateSpace or TransferFunction), continuous
        System to be converted

    Ts: number
        Sample time for the conversion

    method: string, optional
        Method to be applied,
        'zoh'        Zero-order hold on the inputs (default)
        'foh'        First-order hold, currently not implemented
        'impulse'    Impulse-invariant discretization, currently not implemented
        'tustin'     Bilinear (Tustin) approximation, only SISO
        'matched'    Matched pole-zero method, only SISO
    '''
    #  Call the sample_system() function to do the work
    sysd = sample_system(sysc, Ts, method)
    if isinstance(sysc, StateSpace) and not isinstance(sysd, StateSpace):
        return _convertToStateSpace(sysd)
    return sysd
