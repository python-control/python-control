# margins.py - functions for computing stability margins
#
# Initial author: Richard M. Murray
# Creation date: 14 July 2011

"""Functions for computing stability margins and related functions."""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn

import numpy as np
import scipy as sp

from . import frdata, freqplot, xferfcn
from .exception import ControlMIMONotImplemented
from .iosys import issiso
from . import ss
try:
    from slycot import ab13md
except ImportError:
    ab13md = None
try:
    from . import mag2db
except ImportError:
    # Likely due the following circular import issue:
    #
    # ImportError: cannot import name 'mag2db' from partially initialized module
    # 'control' (most likely due to a circular import) (control/__init__.py)
    #
    def mag2db(mag):
        return 20*np.log10(mag)

__all__ = ['stability_margins', 'phase_crossover_frequencies', 'margin', 'disk_margins', 'disk_margin_plot']

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

    # Compute the imaginary part of H = (num.r + j num.i)/(den.r + j den.i)
    test_w = np.polysub(np.polymul(num_iw.imag, den_iw.real),
                        np.polymul(num_iw.real, den_iw.imag))

    # Find the real-valued w > 0 where imag(H(iw)) = 0
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


# TODO: consider handling sysdata similar to margin (via *sysdata?)
def stability_margins(sysdata, returnall=False, epsw=0.0, method='best'):
    """Stability margins and associated crossover frequencies.

    Parameters
    ----------
    sysdata : LTI system or 3-tuple of array_like
        Linear SISO system representing the loop transfer function.
        Alternatively, a three tuple of the form (mag, phase, omega)
        providing the frequency response can be passed.
    returnall : bool, optional
        If true, return all margins found. If False (default), return only the
        minimum stability margins. For frequency data or FRD systems, only
        margins in the given frequency region can be found and returned.
    epsw : float, optional
        Frequencies below this value (default 0.0) are considered static gain,
        and not returned as margin.
    method : string, optional
        Method to use (default is 'best'):

        * 'poly': use polynomial method if passed a `LTI` system.
        * 'frd': calculate crossover frequencies using numerical
          interpolation of a `FrequencyResponseData` representation
          of the system if passed a `LTI` system.
        * 'best': use the 'poly' method if possible, reverting to 'frd' if
          it is detected that numerical inaccuracy is likely to arise in the
          'poly' method for for discrete-time systems.

    Returns
    -------
    gm : float or array_like
        Gain margin.
    pm : float or array_like
        Phase margin.
    sm : float or array_like
        Stability margin, the minimum distance from the Nyquist plot to -1.
    wpc : float or array_like
        Phase crossover frequency (where phase crosses -180 degrees), which is
        associated with the gain margin.
    wgc : float or array_like
        Gain crossover frequency (where gain crosses 1), which is associated
        with the phase margin.
    wms : float or array_like
        Stability margin frequency (where Nyquist plot is closest to -1).

    Notes
    -----
    The gain margin is determined by the gain of the loop transfer function
    at the phase crossover frequency(s), the phase margin is determined by
    the phase of the loop transfer function at the gain crossover
    frequency(s), and the stability margin is determined by the frequency
    of maximum sensitivity (given by the magnitude of 1/(1+L)).

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
            [sp.optimize.minimize_scalar(
                _dstab, bracket=(sys.omega[i], sys.omega[i+1])).x
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
    """Compute Nyquist plot real-axis crossover frequencies and gains.

    Parameters
    ----------
    sys : LTI
        SISO LTI system.

    Returns
    -------
    omega : ndarray
        1d array of (non-negative) frequencies where Nyquist plot
        intersects the real axis.
    gains : ndarray
        1d array of corresponding gains.

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
        gains = np.real(sys(omega * 1j, warn_infinite=False))
    else:
        zargs = _poly_z_invz(sys)
        z, omega = _poly_z_real_crossing(*zargs, epsw=0.)
        gains = np.real(sys(z, warn_infinite=False))

    return omega, gains

def margin(*args):
    """
    margin(sys) \
    margin(mag, phase, omega)

    Gain and phase margins and associated crossover frequencies.

    Can be called as ``margin(sys)`` where `sys` is a SISO LTI system or
    ``margin(mag, phase, omega)``.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        Linear SISO system representing the loop transfer function.
    mag, phase, omega : sequence of array_like
        Input magnitude, phase (in deg.), and frequencies (rad/sec) from
        bode frequency response data.

    Returns
    -------
    gm : float
        Gain margin.
    pm : float
        Phase margin (in degrees).
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

def disk_margins(L, omega, skew = 0.0, returnall = False):
    """Compute disk-based stability margins for SISO or MIMO LTI system.

    Parameters
    ----------
    L : SISO or MIMO LTI system representing the loop transfer function
    omega : ndarray
        1d array of (non-negative) frequencies (rad/s) at which to evaluate
        the disk-based stability margins
    skew : (optional, default = 0) skew parameter for disk margin calculation.
        skew = 0 uses the "balanced" sensitivity function 0.5*(S - T)
        skew = 1 uses the sensitivity function S
        skew = -1 uses the complementary sensitivity function T
    returnall : bool, optional
        If true, return all margins found. If False (default), return only the
        minimum stability margins. Only margins in the given frequency region
        can be found and returned.

    Returns
    -------
    DM : ndarray
        1D array of frequency-dependent disk margins.  DM is the same
        size as "omega" parameter.
    GM : ndarray
        1D array of frequency-dependent disk-based gain margins, in dB.
        GM is the same size as "omega" parameter.
    PM : ndarray
        1D array of frequency-dependent disk-based phase margins, in deg.
        PM is the same size as "omega" parameter.

    Examples
    --------
    >> import control
    >> import numpy as np
    >> import matplotlib
    >> import matplotlib.pyplot as plt
    >>
    >> omega = np.logspace(-1, 3, 1001)
    >>
    >> P = control.ss([[0, 10],[-10, 0]], np.eye(2), [[1, 10], [-10, 1]], [[0, 0],[0, 0]])
    >> K = control.ss([],[],[], [[1, -2], [0, 1]])
    >> L = P*K
    >>
    >> DM, GM, PM = control.disk_margins(L, omega, skew = 0.0, returnall = True) # balanced (S - T)
    >> print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    >> print(f"GM = {GM[np.argmin(DM)]} dB")
    >> print(f"PM = {PM[np.argmin(DM)]} deg\n")
    >>
    >> plt.figure(1)
    >> plt.subplot(3,1,1)
    >> plt.semilogx(omega, DM, label='$\\alpha$')
    >> plt.legend()
    >> plt.title('Disk Margin')
    >> plt.grid()
    >> plt.tight_layout()
    >> plt.xlim([omega[0], omega[-1]])
    >>
    >> plt.figure(1)
    >> plt.subplot(3,1,2)
    >> plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    >> plt.ylabel('Gain Margin (dB)')
    >> plt.legend()
    >> plt.title('Disk-Based Gain Margin')
    >> plt.grid()
    >> plt.ylim([0, 40])
    >> plt.tight_layout()
    >> plt.xlim([omega[0], omega[-1]])
    >>
    >> plt.figure(1)
    >> plt.subplot(3,1,3)
    >> plt.semilogx(omega, PM, label='$\\phi_{m}$')
    >> plt.ylabel('Phase Margin (deg)')
    >> plt.legend()
    >> plt.title('Disk-Based Phase Margin')
    >> plt.grid()
    >> plt.ylim([0, 90])
    >> plt.tight_layout()
    >> plt.xlim([omega[0], omega[-1]])

    References
    ----------
    [1] Blight, James D., R. Lane Dailey, and Dagfinn Gangsaas. “Practical
        Control Law Design for Aircraft Using Multivariable Techniques.”
        International Journal of Control 59, no. 1 (January 1994): 93-137.
        https://doi.org/10.1080/00207179408923071.

    [2] Seiler, Peter, Andrew Packard, and Pascal Gahinet. “An Introduction
        to Disk Margins [Lecture Notes].” IEEE Control Systems Magazine 40,
        no. 5 (October 2020): 78-95.

    [3] P. Benner, V. Mehrmann, V. Sima, S. Van Huffel, and A. Varga, "SLICOT
        - A Subroutine Library in Systems and Control Theory", Applied and
        Computational Control, Signals, and Circuits (Birkhauser), Vol. 1, Ch.
        10, pp. 505-546, 1999.

    [4] S. Van Huffel, V. Sima, A. Varga, S. Hammarling, and F. Delebecque,
        "Development of High Performance Numerical Software for Control", IEEE
        Control Systems Magazine, Vol. 24, Nr. 1, Feb., pp. 60-76, 2004.
    """

    # Check for prerequisites
    if (not L.issiso()) and (ab13md == None):
        raise ControlMIMONotImplemented("Need slycot to compute MIMO disk_margins")

    # Get dimensions of feedback system
    ny,_ = ss(L).C.shape
    I = ss([], [], [], np.eye(ny))

    # Loop sensitivity function
    S = I.feedback(L)

    # Compute frequency response of the "balanced" (according
    # to the skew parameter "sigma") sensitivity function [1-2]
    ST = S + 0.5*(skew - 1)*I
    ST_mag, ST_phase, _ = ST.frequency_response(omega)
    ST_jw = (ST_mag*np.exp(1j*ST_phase))
    if not L.issiso():
        ST_jw = ST_jw.transpose(2,0,1)

    # Frequency-dependent complex disk margin, computed using upper bound of
    # the structured singular value, a.k.a. "mu", of (S + (skew - 1)/2).
    # Uses SLICOT routine AB13MD to compute. [1,3-4].
    DM = np.zeros(omega.shape, np.float64)
    DGM = np.zeros(omega.shape, np.float64)
    DPM = np.zeros(omega.shape, np.float64)
    for ii in range(0,len(omega)):
        # Disk margin (a.k.a. "alpha") vs. frequency
        if L.issiso() and (ab13md == None):
            DM[ii] = np.minimum(1e5,
                1.0/bode(ST_jw, omega = omega[ii], plot = False)[0])
        else:
            DM[ii] = np.minimum(1e5,
                1.0/ab13md(ST_jw[ii], np.array(ny*[1]), np.array(ny*[2]))[0])

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            # Real-axis intercepts with the disk
            gamma_min = (1 - 0.5*DM[ii]*(1 - skew))/(1 + 0.5*DM[ii]*(1 + skew))
            gamma_max = (1 + 0.5*DM[ii]*(1 - skew))/(1 - 0.5*DM[ii]*(1 + skew))

            # Gain margin (dB)
            DGM[ii] = mag2db(np.minimum(1/gamma_min, gamma_max))
            if np.isnan(DGM[ii]):
                DGM[ii] = float('inf')

            # Phase margin (deg)
            if np.isinf(gamma_max):
                DPM[ii] = 90.0
            else:
                DPM[ii] = (1 + gamma_min*gamma_max)/(gamma_min + gamma_max)
                if abs(DPM[ii]) >= 1.0:
                    DPM[ii] = float('Inf')
                else:
                    DPM[ii] = np.rad2deg(np.arccos(DPM[ii]))

    if returnall:
        # Frequency-dependent disk margin, gain margin and phase margin
        return (DM, DGM, DPM)
    else:
        # Worst-case disk margin, gain margin and phase margin
        if DGM.shape[0] and not np.isinf(DGM).all():
            with np.errstate(all='ignore'):
                gmidx = np.where(np.abs(DGM) == np.min(np.abs(DGM)))
        else:
            gmidx = -1
        if DPM.shape[0]:
            pmidx = np.where(np.abs(DPM) == np.amin(np.abs(DPM)))[0]

        return ((not DM.shape[0] and float('inf')) or np.amin(DM),
            (not gmidx != -1 and float('inf')) or DGM[gmidx][0],
            (not DPM.shape[0] and float('inf')) or DPM[pmidx][0])

def disk_margin_plot(alpha_max, skew = 0.0, ax = None):
    """Plot region of allowable gain/phase variation, given worst-case disk margin.

    Parameters
    ----------
    alpha_max : worst-case disk margin(s) across all (relevant) frequencies.
        Note that skew may be a scalar or list.
    skew : (optional, default = 0) skew parameter(s) for disk margin calculation.
        skew = 0 uses the "balanced" sensitivity function 0.5*(S - T)
        skew = 1 uses the sensitivity function S
        skew = -1 uses the complementary sensitivity function T
        Note that skew may be a scalar or list.
    ax : axes to plot bounding curve(s) onto

    Returns
    -------
    DM : ndarray
        1D array of frequency-dependent disk margins.  DM is the same
        size as "omega" parameter.
    GM : ndarray
        1D array of frequency-dependent disk-based gain margins, in dB.
        GM is the same size as "omega" parameter.
    PM : ndarray
        1D array of frequency-dependent disk-based phase margins, in deg.
        PM is the same size as "omega" parameter.

    Examples
    --------
    >> import control
    >> import numpy as np
    >> import matplotlib
    >> import matplotlib.pyplot as plt
    >>
    >> omega = np.logspace(-1, 2, 1001)
    >>
    >> s = control.tf('s') # Laplace variable
    >> L = 6.25*(s + 3)*(s + 5)/(s*(s + 1)**2*(s**2 + 0.18*s + 100)) # loop transfer function
    >>
    >> DM_plot = []
    >> DM_plot.append(control.disk_margins(L, omega, skew = -1.0)[0]) # T-based (T)
    >> DM_plot.append(control.disk_margins(L, omega, skew = 0.0)[0]) # balanced (S - T)
    >> DM_plot.append(control.disk_margins(L, omega, skew = 1.0)[0]) # S-based (S)
    >> plt.figure(1)
    >> control.disk_margin_plot(DM_plot, skew = [-1.0, 0.0, 1.0])
    >> plt.show()

    References
    ----------
    [1] Seiler, Peter, Andrew Packard, and Pascal Gahinet. “An Introduction
        to Disk Margins [Lecture Notes].” IEEE Control Systems Magazine 40,
        no. 5 (October 2020): 78-95.

    """

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Allow scalar or vector arguments (to overlay plots)
    if np.isscalar(alpha_max):
        alpha_max = np.asarray([alpha_max])
    else:
        alpha_max = np.asarray(alpha_max)

    if np.isscalar(skew):
        skew = np.asarray([skew])
    else:
        skew = np.asarray(skew)

    # Add a plot for each (alpha, skew) pair present
    theta = np.linspace(0, np.pi, 500)
    legend_list = []
    for ii in range(0, skew.shape[0]):
        legend_str = "$\\sigma$ = %.1f, $\\alpha_{max}$ = %.2f" %(
            skew[ii], alpha_max[ii])
        legend_list.append(legend_str)

        # Complex bounding curve of stable gain/phase variations
        f = (2 + alpha_max[ii]*(1 - skew[ii])*np.exp(1j*theta))/\
            (2 - alpha_max[ii]*(1 + skew[ii])*np.exp(1j*theta))

        # Allowable combined gain/phase variations
        gamma_dB = mag2db(np.abs(f)) # gain margin (dB)
        phi_deg = np.rad2deg(np.angle(f)) # phase margin (deg)

        # Plot the allowable combined gain/phase variations
        out = ax.plot(gamma_dB, phi_deg, alpha = 0.25,
            label = '_nolegend_')
        ax.fill_between(ax.lines[ii].get_xydata()[:,0],
            ax.lines[ii].get_xydata()[:,1], alpha = 0.25)

    plt.ylabel('Gain Variation (dB)')
    plt.xlabel('Phase Variation (deg)')
    plt.title('Range of Gain and Phase Variations')
    plt.legend(legend_list)
    plt.grid()
    plt.tight_layout()

    return out
