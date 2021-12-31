# pzmap.py - computations involving poles and zeros
#
# Author: Richard M. Murray
# Date: 7 Sep 2009
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

from numpy import real, imag, linspace, exp, cos, sin, sqrt
from math import pi
from .lti import LTI, isdtime, isctime
from .grid import sgrid, zgrid, nogrid
from . import config

__all__ = ['pzmap']


# Define default parameter values for this module
_pzmap_defaults = {
    'pzmap.grid': False,       # Plot omega-damping grid
    'pzmap.plot': True,        # Generate plot using Matplotlib
}


# TODO: Implement more elegant cross-style axes. See:
#    http://matplotlib.sourceforge.net/examples/axes_grid/demo_axisline_style.html
#    http://matplotlib.sourceforge.net/examples/axes_grid/demo_curvelinear_grid.html
def pzmap(sys, plot=None, grid=None, title='Pole Zero Map', **kwargs):
    """Plot a pole/zero map for a linear system.

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        Linear system for which poles and zeros are computed.
    plot: bool, optional
        If ``True`` a graph is generated with Matplotlib,
        otherwise the poles and zeros are only computed and returned.
    grid: boolean (default = False)
        If True plot omega-damping grid.

    Returns
    -------
    poles: array
        The systems poles
    zeros: array
        The system's zeros.

    Notes
    -----
    The pzmap function calls matplotlib.pyplot.axis('equal'), which means
    that trying to reset the axis limits may not behave as expected.  To
    change the axis limits, use matplotlib.pyplot.gca().axis('auto') and
    then set the axis limits to the desired values.

    """
    # Check to see if legacy 'Plot' keyword was used
    if 'Plot' in kwargs:
        import warnings
        warnings.warn("'Plot' keyword is deprecated in pzmap; use 'plot'",
                      FutureWarning)
        plot = kwargs['Plot']

    # Get parameter values
    plot = config._get_param('pzmap', 'plot', plot, True)
    grid = config._get_param('pzmap', 'grid', grid, False)

    if not isinstance(sys, LTI):
        raise TypeError('Argument ``sys``: must be a linear system.')

    poles = sys.pole()
    zeros = sys.zero()

    if (plot):
        import matplotlib.pyplot as plt

        if grid:
            if isdtime(sys, strict=True):
                ax, fig = zgrid()
            else:
                ax, fig = sgrid()
        else:
            ax, fig = nogrid()

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            ax.scatter(real(poles), imag(poles), s=50, marker='x',
                       facecolors='k')
        if len(zeros) > 0:
            ax.scatter(real(zeros), imag(zeros), s=50, marker='o',
                       facecolors='none', edgecolors='k')

        plt.title(title)

    # Return locations of poles and zeros as a tuple
    return poles, zeros
