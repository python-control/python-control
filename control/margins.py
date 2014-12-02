"""margin.py

Functions for computing stability margins and related functions.

Routeins in this module:

margin.stability_margins
margin.phase_crossover_frequencies
"""

# Python 3 compatability (needs to go here)
from __future__ import print_function

"""Copyright (c) 2011 by California Institute of Technology
All rights reserved.

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
Date: 14 July 2011

$Id$

"""

import numpy as np
from . import xferfcn
from .lti import issiso
from . import frdata
import scipy as sp

# helper functions for stability_margins
def _polyimsplit(pol):
    """split a polynomial with (iw) applied into a real and an
       imaginary part with w applied"""
    rpencil = np.zeros_like(pol)
    ipencil = np.zeros_like(pol)
    rpencil[-1::-4] = 1.
    rpencil[-3::-4] = -1.
    ipencil[-2::-4] = 1.
    ipencil[-4::-4] = -1.
    return pol * rpencil, pol*ipencil

def _polysqr(pol):
    """return a polynomial squared"""
    return np.polymul(pol, pol)

# Took the framework for the old function by
# Sawyer B. Fuller <minster@caltech.edu>, removed a lot of the innards
# and replaced with analytical polynomial functions for Lti systems.
#
# idea for the frequency data solution copied/adapted from
# https://github.com/alchemyst/Skogestad-Python/blob/master/BODE.py
# Rene van Paassen <rene.vanpaassen@gmail.com>
#
# RvP, July 8, 2014, corrected to exclude phase=0 crossing for the gain
#                    margin polynomial
def stability_margins(sysdata, deg=True, returnall=False, epsw=1e-10):
    """Calculate gain, phase and stability margins and associated
    crossover frequencies.

    Usage
    -----
    gm, pm, sm, wg, wp, ws = stability_margins(sysdata, deg=True,
                                               returnall=False, epsw=1e-10)

    Parameters
    ----------
    sysdata: linsys or (mag, phase, omega) sequence
        sys : linsys
            Linear SISO system
        mag, phase, omega : sequence of array_like
            Input magnitude, phase, and frequencies (rad/sec) sequence from
            bode frequency response data
    deg=True: boolean
        If true, all input and output phases in degrees, else in radians
    returnall=False: boolean
        If true, return all margins found. Note that for frequency data or
        FRD systems, only one margin is found and returned. 
    epsw=1e-10: float
        frequencies below this value are considered static gain, and not
        returned as margin.

    Returns
    -------
    gm, pm, sm, wg, wp, ws: float or array_like
        Gain margin gm, phase margin pm, stability margin sm, and
        associated crossover
        frequencies wg, wp, and ws of SISO open-loop. If more than
        one crossover frequency is detected, returns the lowest corresponding
        margin.
        When requesting all margins, the return values are array_like,
        and all margins are returns for linear systems not equal to FRD
        """

    try:
        if isinstance(sysdata, frdata.FRD):
            sys = frdata.FRD(sysdata, smooth=True)
        elif isinstance(sysdata, xferfcn.TransferFunction):
            sys = sysdata
        elif getattr(sysdata, '__iter__', False) and len(sysdata) == 3:
            mag, phase, omega = sysdata
            sys = frdata.FRD(mag*np.exp((1j/360.)*phase), omega, smooth=True)
        else:
            sys = xferfcn._convertToTransferFunction(sysdata)
    except Exception as e:
        print (e)
        raise ValueError("Margin sysdata must be either a linear system or "
                         "a 3-sequence of mag, phase, omega.")

    # calculate gain of system
    if isinstance(sys, xferfcn.TransferFunction):

        # check for siso
        if not issiso(sys):
            raise ValueError("Can only do margins for SISO system")

        # real and imaginary part polynomials in omega:
        rnum, inum = _polyimsplit(sys.num[0][0])
        rden, iden = _polyimsplit(sys.den[0][0])

        # test imaginary part of tf == 0, for phase crossover/gain margins
        test_w_180 = np.polyadd(np.polymul(inum, rden), np.polymul(rnum, -iden))
        w_180 = np.roots(test_w_180)

        # first remove imaginary and negative frequencies, epsw removes the
        # "0" frequency for type-2 systems
        w_180 = np.real(w_180[(np.imag(w_180) == 0) * (w_180 >= epsw)])

        # evaluate response at remaining frequencies, to test for phase 180 vs 0
        resp_w_180 = np.real(np.polyval(sys.num[0][0], 1.j*w_180) /
                             np.polyval(sys.den[0][0], 1.j*w_180))

        # only keep frequencies where the negative real axis is crossed
        w_180 = w_180[(resp_w_180 < 0.0)]

        # and sort
        w_180.sort()

        # test magnitude is 1 for gain crossover/phase margins
        test_wc = np.polysub(np.polyadd(_polysqr(rnum), _polysqr(inum)),
                             np.polyadd(_polysqr(rden), _polysqr(iden)))
        wc = np.roots(test_wc)
        wc = np.real(wc[(np.imag(wc) == 0) * (wc > epsw)])
        wc.sort()

        # stability margin was a bitch to elaborate, relies on magnitude to
        # point -1, then take the derivative. Second derivative needs to be >0
        # to have a minimum
        test_wstabn = np.polyadd(_polysqr(rnum), _polysqr(inum))
        test_wstabd = np.polyadd(_polysqr(np.polyadd(rnum,rden)),
                                 _polysqr(np.polyadd(inum,iden)))
        test_wstab = np.polysub(
            np.polymul(np.polyder(test_wstabn),test_wstabd),
            np.polymul(np.polyder(test_wstabd),test_wstabn))

        # find the solutions
        wstab = np.roots(test_wstab)

        # and find the value of the 2nd derivative there, needs to be positive
        wstabplus = np.polyval(np.polyder(test_wstab), wstab)
        wstab = np.real(wstab[(np.imag(wstab) == 0) * (wstab > epsw) *
                              (np.abs(wstabplus) > 0.)])
        wstab.sort()

    else:
        # a bit coarse, have the interpolated frd evaluated again
        def mod(w):
            """to give the function to calculate |G(jw)| = 1"""
            return [np.abs(sys.evalfr(w[0])[0][0]) - 1]

        def arg(w):
            """function to calculate the phase angle at -180 deg"""
            return [np.angle(sys.evalfr(w[0])[0][0]) + np.pi]

        def dstab(w):
            """function to calculate the distance from -1 point"""
            return np.abs(sys.evalfr(w[0])[0][0] + 1.)

        # how to calculate the frequency at which |G(jw)| = 1
        wc = np.array([sp.optimize.fsolve(mod, sys.omega[0])])[0]
        w_180 = np.array([sp.optimize.fsolve(arg, sys.omega[0])])[0]
        wstab = np.real(
            np.array([sp.optimize.fmin(dstab, sys.omega[0], disp=0)])[0])

    # margins, as iterables, converted frdata and xferfcn calculations to
    # vector for this
    PM = np.angle(sys.evalfr(wc)[0][0], deg=True) + 180
    GM = 1/(np.abs(sys.evalfr(w_180)[0][0]))
    SM = np.abs(sys.evalfr(wstab)[0][0]+1)

    if returnall:
        return GM, PM, SM, w_180, wc, wstab
    else:
        return (
            (GM.shape[0] or None) and GM[0],
            (PM.shape[0] or None) and PM[0],
            (SM.shape[0] or None) and SM[0],
            (w_180.shape[0] or None) and w_180[0],
            (wc.shape[0] or None) and wc[0],
            (wstab.shape[0] or None) and wstab[0])


# Contributed by Steffen Waldherr <waldherr@ist.uni-stuttgart.de>
#! TODO - need to add test functions
def phase_crossover_frequencies(sys):
    """
    Compute frequencies and gains at intersections with real axis
    in Nyquist plot.

    Call as:
        omega, gain = phase_crossover_frequencies()

    Returns
    -------
    omega: 1d array of (non-negative) frequencies where Nyquist plot
    intersects the real axis

    gain: 1d array of corresponding gains

    Examples
    --------
    >>> tf = TransferFunction([1], [1, 2, 3, 4])
    >>> PhaseCrossoverFrequenies(tf)
    (array([ 1.73205081,  0.        ]), array([-0.5 ,  0.25]))
    """

    # Convert to a transfer function
    tf = xferfcn._convertToTransferFunction(sys)

    # if not siso, fall back to (0,0) element
    #! TODO: should add a check and warning here
    num = tf.num[0][0]
    den = tf.den[0][0]

    # Compute frequencies that we cross over the real axis
    numj = (1.j)**np.arange(len(num)-1,-1,-1)*num
    denj = (-1.j)**np.arange(len(den)-1,-1,-1)*den
    allfreq = np.roots(np.imag(np.polymul(numj,denj)))
    realfreq = np.real(allfreq[np.isreal(allfreq)])
    realposfreq = realfreq[realfreq >= 0.]

    # using real() to avoid rounding errors and results like 1+0j
    # it would be nice to have a vectorized version of self.evalfr here
    gain = np.real(np.asarray([tf.evalfr(f)[0][0] for f in realposfreq]))

    return realposfreq, gain
