# rlocus.py - code for computing a root locus plot
# Code contributed by Ryan Krauss, 2010
#
# Copyright (c) 2010 by Ryan Krauss
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
# RMM, 17 June 2010: modified to be a standalone piece of code
#   * Added BSD copyright info to file (per Ryan)
#   * Added code to convert (num, den) to poly1d's if they aren't already.
#     This allows Ryan's code to run on a standard signal.ltisys object
#     or a control.TransferFunction object.
#   * Added some comments to make sure I understand the code
#
# RMM, 2 April 2011: modified to work with new LTI structure (see ChangeLog)
#   * Not tested: should still work on signal.ltisys objects
#
# GDM, 16 February 2017: add smart selection of gains based on axis.
#   * Add gains up to a tolerance is achieved
#   * Change some variables and functions names ir order to improve "code style"
#
#
# $Id$

# Packages used by this module
from functools import partial

import numpy as np
import pylab  # plotting routines
import scipy.signal  # signal processing toolbox
from scipy import array, poly1d, row_stack, zeros_like, real, imag

from .exception import ControlMIMONotImplemented
from .xferfcn import _convertToTransferFunction

__all__ = ['root_locus', 'rlocus']


# Main function: compute a root locus diagram
def root_locus(sys, kvect=None, xlim=None, ylim=None, plotstr='-', Plot=True,
               PrintGain=True):
    """Root locus plot

    Calculate the root locus by finding the roots of 1+k*TF(s)
    where TF is self.num(s)/self.den(s) and each k is an element
    of kvect.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems (SISO only, for now)
    kvect : list or ndarray, optional
        List of gains to use in computing diagram
    xlim : tuple or list, optional
        control of x-axis range, normally with tuple (see matplotlib.axes)
    ylim : tuple or list, optional
        control of y-axis range
    Plot : boolean, optional (default = True)
        If True, plot magnitude and phase
    PrintGain: boolean (default = True)
        If True, report mouse clicks when close to the root-locus branches,
        calculate gain, damping and print
    plotstr: string that declare of the rlocus (see matplotlib)

    Returns
    -------
    rlist : ndarray
        Computed root locations, given as a 2d array
    klist : ndarray or list
        Gains used.  Same as klist keyword argument if provided.
    """

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys)

    if kvect is None:
        gvect, mymat, xl, yl = _default_gains(nump, denp, xlim, ylim)
    else:
        gvect = np.asarray(kvect)
        mymat = _find_roots(nump, denp, gvect)
        mymat = _sort_roots(mymat)
        xl = _ax_lim(mymat)
        yl = _ax_lim(mymat * 1j)

    # Create the Plot
    if Plot:
        f = pylab.figure()
        if PrintGain:
            f.canvas.mpl_connect(
                'button_release_event', partial(_feedback_clicks, sys=sys))
        ax = pylab.axes()

        # Plot open loop poles
        poles = array(denp.r)
        ax.plot(real(poles), imag(poles), 'x')

        # Plot open loop zeros
        zeros = array(nump.r)
        if zeros.any():
            ax.plot(real(zeros), imag(zeros), 'o')

        # Now Plot the loci
        for col in mymat.T:
            ax.plot(real(col), imag(col), plotstr)

        # Set up Plot axes and labels
        if xlim is None:
            xlim = xl
        if ylim is None:
            ylim = yl

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')

    return mymat, gvect


def _default_gains(num, den, xlim, ylim):
    """Insert gains up to a tolerance is achieved. This tolerance is a function of figure axis """
    nas = den.order - num.order  # number of asymptotes
    maxk = 0
    olpol = den.roots
    olzer = num.roots
    if nas > 0:
        cas = (sum(den.roots) - sum(num.roots)) / nas
        angles = (2 * np.arange(1, nas + 1) - 1) * np.pi / nas
    else:
        cas = []
        angles = []
        maxk = 100 * den(1) / num(1)

    k_break, real_ax_pts = _break_points(num, den)
    if nas == 0:
        maxk = np.max([1, 2 * maxk])  # get at least some root locus
    else:
        # get distance from breakpoints, poles, and zeros to center of asymptotes
        dmax = 2 * np.max(np.abs(np.concatenate((np.concatenate((olzer, olpol), axis=0),
                                                 real_ax_pts), axis=0) - cas))
        if dmax == 0:
            dmax = 1
        # get gain for dmax along each asymptote, adjust maxk if necessary
        svals = cas + dmax * np.exp(angles * 1j)
        kvals = -den(svals) / num(svals)

        if k_break.size > 0:
            maxk = np.max(np.max(k_break), maxk)

        maxk = np.max([maxk, np.max(np.real(kvals))])

    mink = 0
    ngain = 30
    gvec = np.linspace(mink, maxk, ngain)
    gvec = np.concatenate((gvec, k_break), axis=0)
    gvec.sort()
    done = False

    # Compute out the loci
    mymat = _find_roots(num, den, gvec)
    mymat = _sort_roots(mymat)
    # set smoothing tolerance
    if xlim is None:
        smtolx = 0.01 * (np.max(np.max(np.real(mymat))) - np.min(np.min(np.real(mymat))))
    else:
        smtolx = 0.01 * (xlim[1] - xlim[0])
    if ylim is None:
        smtoly = 0.01 * (np.max(np.max(np.imag(mymat))) - np.min(np.min(np.imag(mymat))))
    else:
        smtoly = 0.01 * (ylim[1] - ylim[0])

    smtol = np.max(np.real([smtolx, smtoly]))
    xl = _ax_lim(mymat)
    yl = _ax_lim(mymat * 1j)

    while ~done & (ngain < 1000):
        done = True
        dp = np.abs(np.diff(mymat, axis=0))
        dp = np.max(dp, axis=1)
        idx = np.where(dp > smtol)

        for ii in np.arange(0, idx[0].size):
            i1 = idx[0][ii]
            g1 = gvec[i1]
            p1 = mymat[i1]

            i2 = idx[0][ii] + 1
            g2 = gvec[i2]
            p2 = mymat[i2]
            # isolate poles in p1, p2
            if np.max(np.abs(p2 - p1)) > smtol:
                newg = np.linspace(g1, g2, 5)
                newmymat = _find_roots(num, den, newg)
                gvec = np.insert(gvec, i1 + 1, newg[1:4])
                mymat = np.insert(mymat, i1 + 1, newmymat[1:4], axis=0)
                mymat = _sort_roots(mymat)
                done = False  # need to process new gains
                ngain = gvec.size

    newg = np.linspace(gvec[-1], gvec[-1] * 200, 5)
    newmymat = _find_roots(num, den, newg)
    gvec = np.append(gvec, newg[1:5])
    mymat = np.concatenate((mymat, newmymat[1:5]), axis=0)
    mymat = _sort_roots(mymat)
    return gvec, mymat, xl, yl


def _break_points(num, den):
    """Extract break points over real axis and the gains give these location"""
    # type: (np.poly1d, np.poly1d) -> (np.array, np.array)
    dnum = num.deriv(m=1)
    dden = den.deriv(m=1)
    brkp = np.poly1d(np.convolve(den, dnum) - np.convolve(num, dden))
    real_ax_pts = brkp.r
    real_ax_pts = real_ax_pts[np.imag(real_ax_pts) == 0]
    real_ax_pts = real_ax_pts[num(real_ax_pts) != 0]  # avoid dividing by zero
    k_break = -den(real_ax_pts) / num(real_ax_pts)
    idx = k_break >= 0
    k_break = k_break[idx]
    real_ax_pts = real_ax_pts[idx]
    return k_break, real_ax_pts


# Utility function to extract numerator and denominator polynomials
def _systopoly1d(sys):
    """Extract numerator and denominator polynomails for a system"""
    # Allow inputs from the signal processing toolbox
    if isinstance(sys, scipy.signal.lti):
        nump = sys.num
        denp = sys.den

    else:
        # Convert to a transfer function, if needed
        sys = _convertToTransferFunction(sys)

        # Make sure we have a SISO system
        if sys.inputs > 1 or sys.outputs > 1:
            raise ControlMIMONotImplemented()

        # Start by extracting the numerator and denominator from system object
        nump = sys.num[0][0]
        denp = sys.den[0][0]

    # Check to see if num, den are already polynomials; otherwise convert
    if not isinstance(nump, poly1d):
        nump = poly1d(nump)
    if not isinstance(denp, poly1d):
        denp = poly1d(denp)
    return nump, denp


def _find_roots(nump, denp, kvect):
    """Find the roots for the root locus."""

    roots = []
    for k in kvect:
        curpoly = denp + k * nump
        curroots = curpoly.r
        curroots.sort()
        roots.append(curroots)
    mymat = row_stack(roots)
    return mymat


def _sort_roots(mymat):
    """Sort the roots from sys._sort_roots, so that the root
    locus doesn't show weird pseudo-branches as roots jump from
    one branch to another."""

    sorted_roots = zeros_like(mymat)
    for n, row in enumerate(mymat):
        if n == 0:
            sorted_roots[n, :] = row
        else:
            # sort the current row by finding the element with the
            # smallest absolute distance to each root in the
            # previous row
            available = list(range(len(prevrow)))
            for elem in row:
                evect = elem - prevrow[available]
                ind1 = abs(evect).argmin()
                ind = available.pop(ind1)
                sorted_roots[n, ind] = elem
        prevrow = sorted_roots[n, :]
    return sorted_roots


def _ax_lim(mymat):
    xmin = np.min(np.min(np.real(mymat)))
    xmax = np.max(np.max(np.real(mymat)))
    if xmax != xmin:
        deltax = (xmax - xmin) * 0.02
    else:
        deltax = np.max([1., xmax / 2])
    xlim = [xmin - deltax, xmax + deltax]
    return xlim


def _feedback_clicks(event, sys):
    """Print root-locus gain feedback for clicks on the root-locus plot
    """
    s = complex(event.xdata, event.ydata)
    k = -1. / sys.horner(s)
    if abs(k.real) > 1e-8 and abs(k.imag / k.real) < 0.04:
        print("Clicked at %10.4g%+10.4gj gain %10.4g damp %10.4g" %
              (s.real, s.imag, k.real, -1 * s.real / abs(s)))


rlocus = root_locus
