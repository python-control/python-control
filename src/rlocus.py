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
#     or a controls.TransferFunction object.
#   * Added some comments to make sure I understand the code
# 
# $Id: rlocus.py 29 2010-11-06 13:03:55Z murrayrm $

# Packages used by this module
from scipy import *

# Main function: compute a root locus diagram
def RootLocus(sys, kvect, fig=None, fignum=1, \
                  clear=True, xlim=None, ylim=None, plotstr='-'):
    """Calculate the root locus by finding the roots of 1+k*TF(s)
    where TF is self.num(s)/self.den(s) and each k is an element
    of kvect."""

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys);

    # Set up the figure
    if fig is None:
        import pylab
        fig = pylab.figure(fignum)
    if clear:
        fig.clf()
    ax = fig.add_subplot(111)

    # Compute out the loci
    mymat = _RLFindRoots(sys, kvect)
    mymat = _RLSortRoots(sys, mymat)

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

    # Start by extracting the numerator and denominator from system object
    nump = sys.num; denp = sys.den;

    # Check to see if num, den are already polynomials; otherwise convert
    if (not isinstance(nump, poly1d)): nump = poly1d(nump)
    if (not isinstance(denp, poly1d)): denp = poly1d(denp)

    return (nump, denp)

def _RLFindRoots(sys, kvect):
    """Find the roots for the root locus."""

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys);

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
        if n==0:
            sorted[n,:] = row
        else:
            # sort the current row by finding the element with the
            # smallest absolute distance to each root in the
            # previous row
            available = range(len(prevrow))
            for elem in row:
                evect = elem-prevrow[available]
                ind1 = abs(evect).argmin()
                ind = available.pop(ind1)
                sorted[n,ind] = elem
        prevrow = sorted[n,:]
    return sorted
