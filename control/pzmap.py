# pzmap.py - computations involving poles and zeros
#
# Author: Richard M. Murray
# Date: 7 Sep 09
#
# This file contains functions that compute poles, zeros and related
# quantities for a linear system.
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
# $Id:pzmap.py 819 2009-05-29 21:28:07Z murray $

from numpy import real, imag, linspace, pi, exp, cos, sin, sqrt
from .lti import LTI, isdtime, isctime

__all__ = ['pzmap']

def zgrid(plt):
    for zeta in linspace(0, 0.9,10):
        factor = zeta/sqrt(1-zeta**2)
        x = linspace(0, sqrt(1-zeta**2),200)
        ang = pi*x
        mag = exp(-pi*factor*x)
        xret = mag*cos(ang)
        yret = mag*sin(ang)
        plt.plot(xret,yret, 'k:', lw=1)
        xret = mag*cos(-ang)
        yret = mag*sin(-ang)
        plt.plot(xret,yret,'k:', lw=1)
        an_i = int(len(xret)/2.5)
        an_x = xret[an_i]
        an_y = yret[an_i]
        plt.annotate(str(zeta), xy=(an_x, an_y), xytext=(an_x, an_y), size=7)

    for a in linspace(0,1,10):
        x = linspace(-pi/2,pi/2,200)
        ang = pi*a*sin(x)
        mag = exp(-pi*a*cos(x))
        xret = mag*cos(ang)
        yret = mag*sin(ang)
        plt.plot(xret,yret,'k:', lw=1)
        an_i = -1 #int(len(xret)/2.5)
        an_x = xret[an_i]
        an_y = yret[an_i]
        num = '{:1.1f}'.format(a)
        plt.annotate("$\\frac{"+num+"\pi}{T}$", xy=(an_x, an_y), xytext=(an_x, an_y), size=8)


# TODO: Implement more elegant cross-style axes. See:
#    http://matplotlib.sourceforge.net/examples/axes_grid/demo_axisline_style.html
#    http://matplotlib.sourceforge.net/examples/axes_grid/demo_curvelinear_grid.html
def pzmap(sys, Plot=True, title='Pole Zero Map'):
    """
    Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        Linear system for which poles and zeros are computed.
    Plot: bool
        If ``True`` a graph is generated with Matplotlib,
        otherwise the poles and zeros are only computed and returned.

    Returns
    -------
    pole: array
        The systems poles
    zeros: array
        The system's zeros.
    """
    if not isinstance(sys, LTI):
        raise TypeError('Argument ``sys``: must be a linear system.')

    poles = sys.pole()
    zeros = sys.zero()

    if (Plot):
        import matplotlib.pyplot as plt

        if isdtime(sys):
            zgrid(plt)

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            plt.scatter(real(poles), imag(poles), s=50, marker='x', facecolors='k')
        if len(zeros) > 0:
            plt.scatter(real(zeros), imag(zeros), s=50, marker='o',
                        facecolors='none', edgecolors='k')
        # Add axes
        #Somewhat silly workaround
        plt.axhline(y=0, color='black', lw=1)
        plt.axvline(x=0, color='black', lw=1)
        plt.xlabel('Re')
        plt.ylabel('Im')

        plt.title(title)

    # Return locations of poles and zeros as a tuple
    return poles, zeros
