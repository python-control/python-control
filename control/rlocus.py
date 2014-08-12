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
# RMM, 2 April 2011: modified to work with new Lti structure (see ChangeLog)
#   * Not tested: should still work on signal.ltisys objects
#
# $Id$

# Packages used by this module
from scipy import array, poly1d, row_stack, zeros_like, real, imag
import scipy.signal             # signal processing toolbox
import pylab                    # plotting routines
from . import xferfcn
from .exception import ControlMIMONotImplemented
from functools import partial


# Main function: compute a root locus diagram
def root_locus(sys, kvect, xlim=None, ylim=None, plotstr='-', Plot=True,
               PrintGain=True):
    """Calculate the root locus by finding the roots of 1+k*TF(s)
    where TF is self.num(s)/self.den(s) and each k is an element
    of kvect.

    Parameters
    ----------
    sys : linsys
        Linear input/output systems (SISO only, for now)
    kvect : gain_range (default = None)
        List of gains to use in computing diagram
    xlim : control of x-axis range, normally with tuple, for
        other options, see matplotlib.axes
    ylim : control of y-axis range
    Plot : boolean (default = True)
        If True, plot magnitude and phase
    PrintGain: boolean (default = True)
        If True, report mouse clicks when close to the root-locus branches,
        calculate gain, damping and print
    Return values
    -------------
    rlist : list of computed root locations
    """

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys)

    # Compute out the loci
    mymat = _RLFindRoots(sys, kvect)
    mymat = _RLSortRoots(sys, mymat)

    # Create the plot
    if (Plot):
        f = pylab.figure()
        if PrintGain:
            f.canvas.mpl_connect(
                'button_release_event', partial(_RLFeedbackClicks, sys=sys))
        ax = pylab.axes()

        # plot open loop poles
        poles = array(denp.r)
        ax.plot(real(poles), imag(poles), 'x')

        # plot open loop zeros
        zeros = array(nump.r)
        if zeros.any():
            ax.plot(real(zeros), imag(zeros), 'o')

        # Now plot the loci
        for col in mymat.T:
            ax.plot(real(col), imag(col), plotstr)

        # Set up plot axes and labels
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')

    return mymat


# Utility function to extract numerator and denominator polynomials
def _systopoly1d(sys):
    """Extract numerator and denominator polynomails for a system"""
    # Allow inputs from the signal processing toolbox
    if (isinstance(sys, scipy.signal.lti)):
        nump = sys.num
        denp = sys.den

    else:
        # Convert to a transfer function, if needed
        sys = xferfcn._convertToTransferFunction(sys)

        # Make sure we have a SISO system
        if (sys.inputs > 1 or sys.outputs > 1):
            raise ControlMIMONotImplemented()

        # Start by extracting the numerator and denominator from system object
        nump = sys.num[0][0]
        denp = sys.den[0][0]

    # Check to see if num, den are already polynomials; otherwise convert
    if (not isinstance(nump, poly1d)):
        nump = poly1d(nump)
    if (not isinstance(denp, poly1d)):
        denp = poly1d(denp)

    return (nump, denp)


def _RLFindRoots(sys, kvect):
    """Find the roots for the root locus."""

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys)

    roots = []
    for k in kvect:
        curpoly = denp + k * nump
        curroots = curpoly.r
        curroots.sort()
        roots.append(curroots)
    mymat = row_stack(roots)
    return mymat


def _RLSortRoots(sys, mymat):
    """Sort the roots from sys._RLFindRoots, so that the root
    locus doesn't show weird pseudo-branches as roots jump from
    one branch to another."""

    sorted = zeros_like(mymat)
    for n, row in enumerate(mymat):
        if n == 0:
            sorted[n, :] = row
        else:
            # sort the current row by finding the element with the
            # smallest absolute distance to each root in the
            # previous row
            available = list(range(len(prevrow)))
            for elem in row:
                evect = elem-prevrow[available]
                ind1 = abs(evect).argmin()
                ind = available.pop(ind1)
                sorted[n, ind] = elem
        prevrow = sorted[n, :]
    return sorted


def _RLFeedbackClicks(event, sys):
    """Print root-locus gain feedback for clicks on the root-locus plot
    """
    s = complex(event.xdata, event.ydata)
    K = -1./sys.horner(s)
    if abs(K.real) > 1e-8 and abs(K.imag/K.real) < 0.04:
        print("Clicked at %10.4g%+10.4gj gain %10.4g damp %10.4g" %
              (s.real, s.imag, K.real, -1*s.real/abs(s)))
