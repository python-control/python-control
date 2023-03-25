"""margins.py

Functions for computing stability margins and related functions.

Routines in this module:

margins.stability_margins
margins.phase_crossover_frequencies
margins.margin
"""

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

import math
from warnings import warn
import numpy as np
import scipy as sp
from . import xferfcn
from .lti import evalfr
from .namedio import issiso
from . import frdata
from . import freqplot
from .exception import ControlMIMONotImplemented

__all__ = ['stability_margins', 'phase_crossover_frequencies', 'margin']


# private helper functions
def _poly_iw(sys):
    """Apply s = iw to G(s)=num(s)/den(s)

    Splits the num and den polynomials with (iw) applied into real and
    imaginary parts with w applied
    """
    num = sys.num[0][0]
    den = sys.den[0][0]
    num_iw = (1J)**np.arange(len(num) - 1, -1, -1) * num
    den_iw = (1J)**np.arange(len(den) - 1, -1, -1) * den
    return num_iw, den_iw


def _poly_iw_sqr(pol_iw):
    return np.real(np.polymul(pol_iw, pol_iw.conj()))


def _poly_iw_real_crossing(num_iw, den_iw, epsw):
    # Return w where imag(H(iw)) == 0
    test_w = np.polysub(np.polymul(num_iw.imag, den_iw.real),
                        np.polymul(num_iw.real, den_iw.imag))
    w = np.roots(test_w)
    w = np.real(w[np.isreal(w)])
    w = w[w >= epsw]
    return w


def _poly_iw_mag1_crossing(num_iw, den_iw, epsw):
    # Return w where |H(iw)| == 1, |num(iw)| - |den(iw)| == 0
    w = np.roots(np.polysub(_poly_iw_sqr(num_iw), _poly_iw_sqr(den_iw)))
    w = np.real(w[np.isreal(w)])
    w = w[w > epsw]
    return w


def _poly_iw_wstab(num_iw, den_iw, epsw):
    # Stability margin: minimum distance to point -1
    # find zero derivative. Second derivative needs to be >0
    # to have a minimum
    test_wstabn = _poly_iw_sqr(np.polyadd(num_iw, den_iw))
    test_wstabd = _poly_iw_sqr(den_iw)
    test_wstab = np.polysub(
        np.polymul(np.polyder(test_wstabn), test_wstabd),
        np.polymul(np.polyder(test_wstabd), test_wstabn))

    # find the solutions, for positive omega, and only real ones
    wstab = np.roots(test_wstab)
    wstab = np.real(wstab[np.isreal(wstab)])
    wstab = wstab[wstab > epsw]

    # and find the value of the 2nd derivative there, needs to be positive
    wstabplus = np.polyval(np.polyder(test_wstab), wstab)
    wstab = wstab[wstabplus > 0.]
    return wstab


def _poly_z_invz(sys):
    num = sys.num[0][0]  # num(z) = a_p * z^p + a_(p-1) * z^(p-1) + ... + a_0
    den = sys.den[0][0]  # num(z) = b_q * z^p + b_(q-1) * z^(q-1) + ... + b_0
    p_q = len(num) - len(den)
    if p_q > 0:
        raise ValueError("Not a proper transfer function: Denominator must "
                         "have equal or higher order than numerator.")
    num_inv_zp = num[::-1]  # num(1/z) * z^p
    den_inv_zq = den[::-1]  # den(1/z) * z^q
    return num, den, num_inv_zp, den_inv_zq, p_q, sys.dt


def _z_filter(z, dt, eps):
    # z = exp(1J w dt)
    # |z| == 1 with some float precision tolerance
    z = z[np.abs(np.abs(z) - 1.) < eps]
    zarg = np.angle(z)
    zidx = (0 <= zarg) * (zarg < np.pi)
    omega = zarg[zidx] / dt
    return z[zidx], omega


def _poly_z_real_crossing(num, den, num_inv_zp, den_inv_zq, p_q, dt, epsw):
    # H(z)==H(1/z), num(z)*den(1/z) == num(1/z)*den(z)
    p1 = np.polymul(num, den_inv_zq)
    p2 = np.polymul(num_inv_zp, den)
    if p_q < 0:
        # * z**(-p_q)
        x = [1] + [0] * (-p_q)
        p2 = np.polymul(p2, x)
    z = np.roots(np.polysub(p1, p2))
    eps = np.finfo(float).eps**(1 / len(p2))
    z, w = _z_filter(z, dt, eps)
    z = z[w >= epsw]
    w = w[w >= epsw]
    return z, w


def _poly_z_mag1_crossing(num, den, num_inv_zp, den_inv_zq, p_q, dt, epsw):
    # |H(z)| = 1, H(z)*H(1/z)=1, num(z)*num(1/z) == den(z)*den(1/z)
    p1 = np.polymul(num, num_inv_zp)
    p2 = np.polymul(den, den_inv_zq)
    if p_q < 0:
        # * z**(-p_q)
        x = [1] + [0] * (-p_q)
        p1 = np.polymul(p1, x)
    z = np.roots(np.polysub(p1, p2))
    eps = np.finfo(float).eps**(1 / len(p2))
    z, w = _z_filter(z, dt, eps)
    z = z[w > epsw]
    w = w[w > epsw]
    return z, w


def _poly_z_wstab(num, den, num_inv_zp, den_inv_zq, p_q, dt, epsw):
    # Stability margin: Minimum distance to -1

    # TODO: Find a way to solve for z or omega analytically with given
    # polynomials
    # d|1 + H(z)|/dz = 0, or d|1 + H(exp(iwdt))|/dw = 0

    # optimization function to minimize
    def fun(wdt):
        with np.errstate(all='ignore'):  # den=0 is okay
            return np.abs(1 + (np.polyval(num, np.exp(1J * wdt)) /
                               np.polyval(den, np.exp(1J * wdt))))

    # find initial guess
    wdt_v = np.geomspace(1e-4, 2 * np.pi, num=100)
    wdt0 = wdt_v[np.argmin(fun(wdt_v))]

    # Use `minimize` instead of univariate `minimize_scalars` because we want
    # to provide some initial value in order to not converge on frequencies
    # with extremely low gradients.
    res = sp.optimize.minimize(
        fun=fun,
        x0=[wdt0],
        bounds=[(0, 2 * np.pi)])
    if res.success:
        wdt = res.x
        z = np.exp(1J * wdt)
        w = wdt / dt
    else:
        z = np.array([])
        w = np.array([])

    return z, w

def _likely_numerical_inaccuracy(sys):
    # crude, conservative check for if
    # num(z)*num(1/z) << den(z)*den(1/z) for DT systems
    num, den, num_inv_zp, den_inv_zq, p_q, dt = _poly_z_invz(sys)
    p1 = np.polymul(num, num_inv_zp)
    p2 = np.polymul(den, den_inv_zq)
    if p_q < 0:
        # * z**(-p_q)
        x = [1] + [0] * (-p_q)
        p1 = np.polymul(p1, x)
    return np.linalg.norm(p1) < 1e-4 * np.linalg.norm(p2)

# Took the framework for the old function by
# Sawyer B. Fuller <minster@uw.edu>, removed a lot of the innards
# and replaced with analytical polynomial functions for LTI systems.
#
# The idea for the frequency data solution copied/adapted from
# https://github.com/alchemyst/Skogestad-Python/blob/master/BODE.py
# Rene van Paassen <rene.vanpaassen@gmail.com>
#
# RvP, July 8, 2014, corrected to exclude phase=0 crossing for the gain
#                    margin polynomial
#
# RvP, July 8, 2015, augmented to calculate all phase/gain crossings with
#                    frd data. Correct to return smallest phase
#                    margin, smallest gain margin and their frequencies
#
# RvP, Jun 10, 2017, modified the inclusion of roots found for phase crossing
#                    to include all >= 0, made subsequent calc insensitive to
#                    div by 0.  Also changed the selection of which crossings
#                    to return on basis of "A note on the Gain and Phase
#                    Margin Concepts" Journal of Control and Systems
#                    Engineering, Yazdan Bavafi-Toosi, Dec 2015, vol 3 issue
#                    1, pp 51-59, closer to Matlab behavior, but not
#                    completely identical in edge cases, which don't cross but
#                    touch gain=1.
#
# BG, Nov 9, 2020,   removed duplicate implementations of the same code
#                    for crossover frequencies and enhanced to handle discrete
#                    systems


def stability_margins(sysdata, returnall=False, epsw=0.0, method='best'):
    """Calculate stability margins and associated crossover frequencies.

    Parameters
    ----------
    sysdata : LTI system or (mag, phase, omega) sequence
        sys : LTI system
            Linear SISO system representing the loop transfer function
        mag, phase, omega : sequence of array_like
            Arrays of magnitudes (absolute values, not dB), phases (degrees),
            and corresponding frequencies. Crossover frequencies returned are
            in the same units as those in `omega` (e.g., rad/sec or Hz).
    returnall : bool, optional
        If true, return all margins found. If False (default), return only the
        minimum stability margins. For frequency data or FRD systems, only
        margins in the given frequency region can be found and returned.
    epsw : float, optional
        Frequencies below this value (default 0.0) are considered static gain,
        and not returned as margin.
    method : string, optional
        Method to use (default is 'best'):
        'poly': use polynomial method if passed a :class:`LTI` system.
        'frd': calculate crossover frequencies using numerical interpolation
        of a :class:`FrequencyResponseData` representation of the system if
        passed a :class:`LTI` system.
        'best': use the 'poly' method if possible, reverting to 'frd' if it is
        detected that numerical inaccuracy is likey to arise in the 'poly'
        method for for discrete-time systems.

    Returns
    -------
    gm : float or array_like
        Gain margin
    pm : float or array_like
        Phase margin
    sm : float or array_like
        Stability margin, the minimum distance from the Nyquist plot to -1
    wpc : float or array_like
        Phase crossover frequency (where phase crosses -180 degrees), which is
        associated with the gain margin.
    wgc : float or array_like
        Gain crossover frequency (where gain crosses 1), which is associated
        with the phase margin.
    wms : float or array_like
        Stability margin frequency (where Nyquist plot is closest to -1)

    Note that the gain margin is determined by the gain of the loop
    transfer function at the phase crossover frequency(s), the phase
    margin is determined by the phase of the loop transfer function at
    the gain crossover frequency(s), and the stability margin is
    determined by the frequency of maximum sensitivity (given by the
    magnitude of 1/(1+L)).
    """
    # TODO: FRD method for cont-time systems doesn't work
    try:
        if isinstance(sysdata, frdata.FRD):
            sys = frdata.FRD(sysdata, smooth=True)
        elif isinstance(sysdata, xferfcn.TransferFunction):
            sys = sysdata
        elif getattr(sysdata, '__iter__', False) and len(sysdata) == 3:
            mag, phase, omega = sysdata
            sys = frdata.FRD(mag * np.exp(1j * phase * math.pi / 180.),
                             omega, smooth=True)
        else:
            sys = xferfcn._convert_to_transfer_function(sysdata)
    except Exception as e:
        print(e)
        raise ValueError("Margin sysdata must be either a linear system or "
                         "a 3-sequence of mag, phase, omega.")

    # check for siso
    if not issiso(sys):
        raise ControlMIMONotImplemented(
            "Can only do margins for SISO system")

    if method == 'frd':
        # convert to FRD if we got a transfer function
        if isinstance(sys, xferfcn.TransferFunction):
            omega_sys = freqplot._default_frequency_range(sys)
            if sys.isctime():
                sys = frdata.FRD(sys, omega_sys)
            else:
                omega_sys = omega_sys[omega_sys < np.pi / sys.dt]
                sys = frdata.FRD(sys, omega_sys, smooth=True)
    elif method == 'best':
        # convert to FRD if anticipated numerical issues
        if isinstance(sys, xferfcn.TransferFunction) and not sys.isctime():
            if _likely_numerical_inaccuracy(sys):
                warn("stability_margins: Falling back to 'frd' method "
                "because of chance of numerical inaccuracy in 'poly' method.",
                stacklevel=2)
                omega_sys = freqplot._default_frequency_range(sys)
                omega_sys = omega_sys[omega_sys < np.pi / sys.dt]
                sys = frdata.FRD(sys, omega_sys, smooth=True)
    elif method != 'poly':
        raise ValueError("method " + method + " unknown")

    if isinstance(sys, xferfcn.TransferFunction):
        if sys.isctime():
            num_iw, den_iw = _poly_iw(sys)
            # frequency for gain margin: phase crosses -180 degrees
            w_180 = _poly_iw_real_crossing(num_iw, den_iw, epsw)
            w180_resp = sys(1J * w_180, warn_infinite=False)  # den=0 is okay

            # frequency for phase margin : gain crosses magnitude 1
            wc = _poly_iw_mag1_crossing(num_iw, den_iw, epsw)
            wc_resp = sys(1J * wc)

            # stability margin
            wstab = _poly_iw_wstab(num_iw, den_iw, epsw)
            ws_resp = sys(1J * wstab)

        else:  # Discrete Time
            zargs = _poly_z_invz(sys)
            # gain margin
            z, w_180 = _poly_z_real_crossing(*zargs, epsw=epsw)
            w180_resp = sys(z)

            # phase margin
            z, wc = _poly_z_mag1_crossing(*zargs, epsw=epsw)
            wc_resp = sys(z)

            # stability margin
            z, wstab = _poly_z_wstab(*zargs, epsw=epsw)
            ws_resp = sys(z)

        # only keep frequencies where the negative real axis is crossed
        w_180 = w_180[w180_resp <= 0.]
        w180_resp = w180_resp[w180_resp <= 0.]

        # sort
        idx = np.argsort(w_180)
        w_180 = w_180[idx]
        w180_resp = w180_resp[idx]

        idx = np.argsort(wc)
        wc = wc[idx]
        wc_resp = wc_resp[idx]

        idx = np.argsort(wstab)
        wstab = wstab[idx]
        ws_resp = ws_resp[idx]

    else:
        # a bit coarse, have the interpolated frd evaluated again
        def _mod(w):
            """Calculate |G(jw)| - 1"""
            return np.abs(sys(1j * w)) - 1

        def _arg(w):
            """Calculate the phase angle at -180 deg"""
            return np.angle(-sys(1j * w))

        def _dstab(w):
            """Calculate the distance from -1 point"""
            return np.abs(sys(1j * w) + 1.)

        # find the phase crossings ang(H(jw) == -180
        widx = np.where(np.diff(np.sign(_arg(sys.omega))))[0]
        widx = widx[np.real(sys(1j * sys.omega[widx])) <= 0]
        w_180 = np.array(
            [sp.optimize.brentq(_arg, sys.omega[i], sys.omega[i+1])
             for i in widx])
        w180_resp = sys(1j * w_180)

        # Find all crossings, note that this depends on omega having
        # a correct range
        widx = np.where(np.diff(np.sign(_mod(sys.omega))))[0]
        wc = np.array(
            [sp.optimize.brentq(_mod, sys.omega[i], sys.omega[i+1])
             for i in widx])
        wc_resp = sys(1j * wc)

        # find all stab margins?
        widx, = np.where(np.diff(np.sign(np.diff(_dstab(sys.omega)))) > 0)
        wstab = np.array(
            [sp.optimize.minimize_scalar(_dstab,
                                         bracket=(sys.omega[i], sys.omega[i+1])
                                         ).x
             for i in widx])
        wstab = wstab[(wstab >= sys.omega[0]) * (wstab <= sys.omega[-1])]
        ws_resp = sys(1j * wstab)

    with np.errstate(all='ignore'):  # |G|=0 is okay and yields inf
        GM = 1. / np.abs(w180_resp)
    PM = np.remainder(np.angle(wc_resp, deg=True), 360.) - 180.
    SM = np.abs(ws_resp + 1.)

    if returnall:
        return GM, PM, SM, w_180, wc, wstab
    else:
        if GM.shape[0] and not np.isinf(GM).all():
            with np.errstate(all='ignore'):
                gmidx = np.where(np.abs(np.log(GM)) ==
                                 np.min(np.abs(np.log(GM))))
        else:
            gmidx = -1
        if PM.shape[0]:
            pmidx = np.where(np.abs(PM) == np.amin(np.abs(PM)))[0]
        return (
            (not gmidx != -1 and float('inf')) or GM[gmidx][0],
            (not PM.shape[0] and float('inf')) or PM[pmidx][0],
            (not SM.shape[0] and float('inf')) or np.amin(SM),
            (not gmidx != -1 and float('nan')) or w_180[gmidx][0],
            (not wc.shape[0] and float('nan')) or wc[pmidx][0],
            (not wstab.shape[0] and float('nan')) or
            wstab[SM == np.amin(SM)][0])


# Contributed by Steffen Waldherr <waldherr@ist.uni-stuttgart.de>
def phase_crossover_frequencies(sys):
    """Compute frequencies and gains at intersections with real axis
    in Nyquist plot.

    Parameters
    ----------
    sys : SISO LTI system

    Returns
    -------
    omega : ndarray
        1d array of (non-negative) frequencies where Nyquist plot
        intersects the real axis
    gain : ndarray
        1d array of corresponding gains

    Examples
    --------
    >>> G = ct.tf([1], [1, 2, 3, 4])
    >>> x_omega, x_gain = ct.phase_crossover_frequencies(G)

    """
    # Convert to a transfer function
    tf = xferfcn._convert_to_transfer_function(sys)

    if not issiso(tf):
        raise ControlMIMONotImplemented(
            "Can only calculate crossovers for SISO system")

    # Compute frequencies that we cross over the real axis
    if sys.isctime():
        num_iw, den_iw = _poly_iw(tf)
        omega = _poly_iw_real_crossing(num_iw, den_iw, 0.)

        # using real() to avoid rounding errors and results like 1+0j
        gain = np.real(evalfr(sys, 1J * omega))
    else:
        zargs = _poly_z_invz(sys)
        z, omega = _poly_z_real_crossing(*zargs, epsw=0.)
        gain = np.real(evalfr(sys, z))

    return omega, gain


def margin(*args):
    """margin(sysdata)

    Calculate gain and phase margins and associated crossover frequencies

    Parameters
    ----------
    sysdata : LTI system or (mag, phase, omega) sequence
        sys : StateSpace or TransferFunction
            Linear SISO system representing the loop transfer function
        mag, phase, omega : sequence of array_like
            Input magnitude, phase (in deg.), and frequencies (rad/sec) from
            bode frequency response data

    Returns
    -------
    gm : float
        Gain margin
    pm : float
        Phase margin (in degrees)
    wcg : float or array_like
        Crossover frequency associated with gain margin (phase crossover
        frequency), where phase crosses below -180 degrees.
    wcp : float or array_like
        Crossover frequency associated with phase margin (gain crossover
        frequency), where gain crosses below 1.

    Margins are calculated for a SISO open-loop system.

    If there is more than one gain crossover, the one at the smallest margin
    (deviation from gain = 1), in absolute sense, is returned. Likewise the
    smallest phase margin (in absolute sense) is returned.

    Examples
    --------
    >>> G = ct.tf(1, [1, 2, 1, 0])
    >>> gm, pm, wcg, wcp = ct.margin(G)

    """
    if len(args) == 1:
        sys = args[0]
        margin = stability_margins(sys)
    elif len(args) == 3:
        margin = stability_margins(args)
    else:
        raise ValueError("Margin needs 1 or 3 arguments; received %i."
                         % len(args))

    return margin[0], margin[1], margin[3], margin[4]
