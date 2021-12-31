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
# Sawyer B. Fuller (minster@uw.edu) 21 May 2020:
#   * added compatibility with discrete-time systems.
#
# $Id$

# Packages used by this module
from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import array, poly1d, row_stack, zeros_like, real, imag
import scipy.signal             # signal processing toolbox
from .lti import isdtime
from .xferfcn import _convert_to_transfer_function
from .exception import ControlMIMONotImplemented
from .sisotool import _SisotoolUpdate
from .grid import sgrid, zgrid
from . import config

__all__ = ['root_locus', 'rlocus']

# Default values for module parameters
_rlocus_defaults = {
    'rlocus.grid': True,
    'rlocus.plotstr': 'b' if int(mpl.__version__[0]) == 1 else 'C0',
    'rlocus.print_gain': True,
    'rlocus.plot': True
}


# Main function: compute a root locus diagram
def root_locus(sys, kvect=None, xlim=None, ylim=None,
               plotstr=None, plot=True, print_gain=None, grid=None, ax=None,
               **kwargs):

    """Root locus plot

    Calculate the root locus by finding the roots of 1+k*TF(s)
    where TF is self.num(s)/self.den(s) and each k is an element
    of kvect.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems (SISO only, for now).
    kvect : list or ndarray, optional
        List of gains to use in computing diagram.
    xlim : tuple or list, optional
        Set limits of x axis, normally with tuple
        (see :doc:`matplotlib:api/axes_api`).
    ylim : tuple or list, optional
        Set limits of y axis, normally with tuple
        (see :doc:`matplotlib:api/axes_api`).
    plotstr : :func:`matplotlib.pyplot.plot` format string, optional
        plotting style specification
    plot : boolean, optional
        If True (default), plot root locus diagram.
    print_gain : bool
        If True (default), report mouse clicks when close to the root locus
        branches, calculate gain, damping and print.
    grid : bool
        If True plot omega-damping grid.  Default is False.
    ax : :class:`matplotlib.axes.Axes`
        Axes on which to create root locus plot

    Returns
    -------
    rlist : ndarray
        Computed root locations, given as a 2D array
    klist : ndarray or list
        Gains used.  Same as klist keyword argument if provided.

    Notes
    -----
    The root_locus function calls matplotlib.pyplot.axis('equal'), which
    means that trying to reset the axis limits may not behave as expected.
    To change the axis limits, use matplotlib.pyplot.gca().axis('auto') and
    then set the axis limits to the desired values.

    """
    # Check to see if legacy 'Plot' keyword was used
    if 'Plot' in kwargs:
        import warnings
        warnings.warn("'Plot' keyword is deprecated in root_locus; "
                      "use 'plot'", FutureWarning)
        # Map 'Plot' keyword to 'plot' keyword
        plot = kwargs.pop('Plot')

    # Check to see if legacy 'PrintGain' keyword was used
    if 'PrintGain' in kwargs:
        import warnings
        warnings.warn("'PrintGain' keyword is deprecated in root_locus; "
                      "use 'print_gain'", FutureWarning)
        # Map 'PrintGain' keyword to 'print_gain' keyword
        print_gain = kwargs.pop('PrintGain')

    # Get parameter values
    plotstr = config._get_param('rlocus', 'plotstr', plotstr, _rlocus_defaults)
    grid = config._get_param('rlocus', 'grid', grid, _rlocus_defaults)
    print_gain = config._get_param(
        'rlocus', 'print_gain', print_gain, _rlocus_defaults)

    sys_loop = sys if sys.issiso() else sys[0, 0]

    # Convert numerator and denominator to polynomials if they aren't
    (nump, denp) = _systopoly1d(sys_loop)

    # if discrete-time system and if xlim and ylim are not given,
    #  that we a view of the unit circle
    if xlim is None and isdtime(sys, strict=True):
        xlim = (-1.2, 1.2)
    if ylim is None and isdtime(sys, strict=True):
        xlim = (-1.3, 1.3)

    if kvect is None:
        start_mat = _RLFindRoots(nump, denp, [1])
        kvect, mymat, xlim, ylim = _default_gains(nump, denp, xlim, ylim)
    else:
        start_mat = _RLFindRoots(nump, denp, [kvect[0]])
        mymat = _RLFindRoots(nump, denp, kvect)
        mymat = _RLSortRoots(mymat)

    # Check for sisotool mode
    sisotool = False if 'sisotool' not in kwargs else True

    # Create the Plot
    if plot:
        if sisotool:
            fig = kwargs['fig']
            ax = fig.axes[1]
        else:
            if ax is None:
                ax = plt.gca()
            fig = ax.figure
            ax.set_title('Root Locus')

        if print_gain and not sisotool:
            fig.canvas.mpl_connect(
                'button_release_event',
                partial(_RLClickDispatcher, sys=sys, fig=fig,
                        ax_rlocus=fig.axes[0], plotstr=plotstr))
        elif sisotool:
            fig.axes[1].plot(
                [root.real for root in start_mat],
                [root.imag for root in start_mat],
                marker='s', markersize=6, zorder=20, color='k', label='gain_point')
            s = start_mat[0][0]
            if isdtime(sys, strict=True):
                zeta = -np.cos(np.angle(np.log(s)))
            else:
                zeta = -1 * s.real / abs(s)
            fig.suptitle(
                "Clicked at: %10.4g%+10.4gj  gain: %10.4g  damp: %10.4g" %
                (s.real, s.imag, kvect[0], zeta),
                fontsize=12 if int(mpl.__version__[0]) == 1 else 10)
            fig.canvas.mpl_connect(
                'button_release_event',
                partial(_RLClickDispatcher, sys=sys, fig=fig,
                        ax_rlocus=fig.axes[1], plotstr=plotstr,
                        sisotool=sisotool,
                        bode_plot_params=kwargs['bode_plot_params'],
                        tvect=kwargs['tvect']))

        # zoom update on xlim/ylim changed, only then data on new limits
        # is available, i.e., cannot combine with _RLClickDispatcher
        dpfun = partial(
            _RLZoomDispatcher, sys=sys, ax_rlocus=ax, plotstr=plotstr)
        # TODO: the next too lines seem to take a long time to execute
        # TODO: is there a way to speed them up?  (RMM, 6 Jun 2019)
        ax.callbacks.connect('xlim_changed', dpfun)
        ax.callbacks.connect('ylim_changed', dpfun)

        # plot open loop poles
        poles = array(denp.r)
        ax.plot(real(poles), imag(poles), 'x')

        # plot open loop zeros
        zeros = array(nump.r)
        if zeros.size > 0:
            ax.plot(real(zeros), imag(zeros), 'o')

        # Now plot the loci
        for index, col in enumerate(mymat.T):
            ax.plot(real(col), imag(col), plotstr, label='rootlocus')

        # Set up plot axes and labels
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')

        # Set up the limits for the plot
        # Note: need to do this before computing grid lines
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Draw the grid
        if grid:
            if isdtime(sys, strict=True):
                zgrid(ax=ax)
            else:
                _sgrid_func(fig=fig if sisotool else None)
        else:
            ax.axhline(0., linestyle=':', color='k', linewidth=.75, zorder=-20)
            ax.axvline(0., linestyle=':', color='k', linewidth=.75, zorder=-20)
            if isdtime(sys, strict=True):
                ax.add_patch(plt.Circle(
                    (0, 0), radius=1.0, linestyle=':', edgecolor='k',
                    linewidth=0.75, fill=False, zorder=-20))

    return mymat, kvect


def _default_gains(num, den, xlim, ylim, zoom_xlim=None, zoom_ylim=None):
    """Unsupervised gains calculation for root locus plot.

    References
    ----------
    Ogata, K. (2002). Modern control engineering (4th ed.). Upper
    Saddle River, NJ : New Delhi: Prentice Hall..

    """
    k_break, real_break = _break_points(num, den)
    kmax = _k_max(num, den, real_break, k_break)
    kvect = np.hstack((np.linspace(0, kmax, 50), np.real(k_break)))
    kvect.sort()

    mymat = _RLFindRoots(num, den, kvect)
    mymat = _RLSortRoots(mymat)
    open_loop_poles = den.roots
    open_loop_zeros = num.roots

    if open_loop_zeros.size != 0 and \
       open_loop_zeros.size < open_loop_poles.size:
        open_loop_zeros_xl = np.append(
            open_loop_zeros,
            np.ones(open_loop_poles.size - open_loop_zeros.size)
            * open_loop_zeros[-1])
        mymat_xl = np.append(mymat, open_loop_zeros_xl)
    else:
        mymat_xl = mymat
    singular_points = np.concatenate((num.roots, den.roots), axis=0)
    important_points = np.concatenate((singular_points, real_break), axis=0)
    important_points = np.concatenate((important_points, np.zeros(2)), axis=0)
    mymat_xl = np.append(mymat_xl, important_points)

    false_gain = float(den.coeffs[0]) / float(num.coeffs[0])
    if false_gain < 0 and not den.order > num.order:
        # TODO: make error message more understandable
        raise ValueError("Not implemented support for 0 degrees root locus "
                         "with equal order of numerator and denominator.")

    if xlim is None and false_gain > 0:
        x_tolerance = 0.05 * (np.max(np.real(mymat_xl))
                              - np.min(np.real(mymat_xl)))
        xlim = _ax_lim(mymat_xl)
    elif xlim is None and false_gain < 0:
        axmin = np.min(np.real(important_points)) \
            - (np.max(np.real(important_points))
               - np.min(np.real(important_points)))
        axmin = np.min(np.array([axmin, np.min(np.real(mymat_xl))]))
        axmax = np.max(np.real(important_points)) \
            + np.max(np.real(important_points)) \
            - np.min(np.real(important_points))
        axmax = np.max(np.array([axmax, np.max(np.real(mymat_xl))]))
        xlim = [axmin, axmax]
        x_tolerance = 0.05 * (axmax - axmin)
    else:
        x_tolerance = 0.05 * (xlim[1] - xlim[0])

    if ylim is None:
        y_tolerance = 0.05 * (np.max(np.imag(mymat_xl))
                              - np.min(np.imag(mymat_xl)))
        ylim = _ax_lim(mymat_xl * 1j)
    else:
        y_tolerance = 0.05 * (ylim[1] - ylim[0])

    # Figure out which points are spaced too far apart
    if x_tolerance == 0:
        # Root locus is on imaginary axis (rare), use just y distance
        tolerance = y_tolerance
    elif y_tolerance == 0:
        # Root locus is on imaginary axis (common), use just x distance
        tolerance = x_tolerance
    else:
        tolerance = np.min([x_tolerance, y_tolerance])
    indexes_too_far = _indexes_filt(mymat, tolerance, zoom_xlim, zoom_ylim)

    # Add more points into the root locus for points that are too far apart
    while len(indexes_too_far) > 0 and kvect.size < 5000:
        for counter, index in enumerate(indexes_too_far):
            index = index + counter*3
            new_gains = np.linspace(kvect[index], kvect[index + 1], 5)
            new_points = _RLFindRoots(num, den, new_gains[1:4])
            kvect = np.insert(kvect, index + 1, new_gains[1:4])
            mymat = np.insert(mymat, index + 1, new_points, axis=0)

        mymat = _RLSortRoots(mymat)
        indexes_too_far = _indexes_filt(mymat, tolerance, zoom_xlim, zoom_ylim)

    new_gains = kvect[-1] * np.hstack((np.logspace(0, 3, 4)))
    new_points = _RLFindRoots(num, den, new_gains[1:4])
    kvect = np.append(kvect, new_gains[1:4])
    mymat = np.concatenate((mymat, new_points), axis=0)
    mymat = _RLSortRoots(mymat)
    return kvect, mymat, xlim, ylim


def _indexes_filt(mymat, tolerance, zoom_xlim=None, zoom_ylim=None):
    """Calculate the distance between points and return the indexes.

    Filter the indexes so only the resolution of points within the xlim and
    ylim is improved when zoom is used.

    """
    distance_points = np.abs(np.diff(mymat, axis=0))
    indexes_too_far = list(np.unique(np.where(distance_points > tolerance)[0]))

    if zoom_xlim is not None and zoom_ylim is not None:
        x_tolerance_zoom = 0.05 * (zoom_xlim[1] - zoom_xlim[0])
        y_tolerance_zoom = 0.05 * (zoom_ylim[1] - zoom_ylim[0])
        tolerance_zoom = np.min([x_tolerance_zoom, y_tolerance_zoom])
        indexes_too_far_zoom = list(
            np.unique(np.where(distance_points > tolerance_zoom)[0]))
        indexes_too_far_filtered = []

        for index in indexes_too_far_zoom:
            for point in mymat[index]:
                if (zoom_xlim[0] <= point.real <= zoom_xlim[1]) and \
                   (zoom_ylim[0] <= point.imag <= zoom_ylim[1]):
                    indexes_too_far_filtered.append(index)
                    break

        # Check if zoom box is not overshot & insert points where neccessary
        if len(indexes_too_far_filtered) == 0 and len(mymat) < 500:
            limits = [zoom_xlim[0], zoom_xlim[1], zoom_ylim[0], zoom_ylim[1]]
            for index, limit in enumerate(limits):
                if index <= 1:
                    asign = np.sign(real(mymat)-limit)
                else:
                    asign = np.sign(imag(mymat) - limit)
                signchange = ((np.roll(asign, 1, axis=0)
                               - asign) != 0).astype(int)
                signchange[0] = np.zeros((len(mymat[0])))
                if len(np.where(signchange == 1)[0]) > 0:
                    indexes_too_far_filtered.append(
                        np.where(signchange == 1)[0][0]-1)

        if len(indexes_too_far_filtered) > 0:
            if indexes_too_far_filtered[0] != 0:
                indexes_too_far_filtered.insert(
                    0, indexes_too_far_filtered[0]-1)
            if not indexes_too_far_filtered[-1] + 1 >= len(mymat) - 2:
                indexes_too_far_filtered.append(
                    indexes_too_far_filtered[-1] + 1)

        indexes_too_far.extend(indexes_too_far_filtered)

    indexes_too_far = list(np.unique(indexes_too_far))
    indexes_too_far.sort()
    return indexes_too_far


def _break_points(num, den):
    """Extract break points over real axis and gains given these locations"""
    # type: (np.poly1d, np.poly1d) -> (np.array, np.array)
    dnum = num.deriv(m=1)
    dden = den.deriv(m=1)
    polynom = den * dnum - num * dden
    real_break_pts = polynom.r
    # don't care about infinite break points
    real_break_pts = real_break_pts[num(real_break_pts) != 0]
    k_break = -den(real_break_pts) / num(real_break_pts)
    idx = k_break >= 0   # only positives gains
    k_break = k_break[idx]
    real_break_pts = real_break_pts[idx]
    if len(k_break) == 0:
        k_break = [0]
        real_break_pts = den.roots
    return k_break, real_break_pts


def _ax_lim(mymat):
    """Utility to get the axis limits"""
    axmin = np.min(np.real(mymat))
    axmax = np.max(np.real(mymat))
    if axmax != axmin:
        deltax = (axmax - axmin) * 0.02
    else:
        deltax = np.max([1., axmax / 2])
    axlim = [axmin - deltax, axmax + deltax]
    return axlim


def _k_max(num, den, real_break_points, k_break_points):
    """"Calculate the maximum gain for the root locus shown in the figure."""
    asymp_number = den.order - num.order
    singular_points = np.concatenate((num.roots, den.roots), axis=0)
    important_points = np.concatenate(
        (singular_points, real_break_points), axis=0)
    false_gain = den.coeffs[0] / num.coeffs[0]

    if asymp_number > 0:
        asymp_center = (np.sum(den.roots) - np.sum(num.roots))/asymp_number
        distance_max = 4 * np.max(np.abs(important_points - asymp_center))
        asymp_angles = (2 * np.arange(0, asymp_number) - 1) \
            * np.pi / asymp_number
        if false_gain > 0:
            # farthest points over asymptotes
            farthest_points = asymp_center \
                + distance_max * np.exp(asymp_angles * 1j)
        else:
            asymp_angles = asymp_angles + np.pi
            # farthest points over asymptotes
            farthest_points = asymp_center \
                + distance_max * np.exp(asymp_angles * 1j)
        kmax_asymp = np.real(np.abs(den(farthest_points)
                                    / num(farthest_points)))
    else:
        kmax_asymp = np.abs([np.abs(den.coeffs[0])
                             / np.abs(num.coeffs[0]) * 3])

    kmax = np.max(np.concatenate((np.real(kmax_asymp),
                                  np.real(k_break_points)), axis=0))
    if np.abs(false_gain) > kmax:
        kmax = np.abs(false_gain)
    return kmax


def _systopoly1d(sys):
    """Extract numerator and denominator polynomails for a system"""
    # Allow inputs from the signal processing toolbox
    if (isinstance(sys, scipy.signal.lti)):
        nump = sys.num
        denp = sys.den

    else:
        # Convert to a transfer function, if needed
        sys = _convert_to_transfer_function(sys)

        # Make sure we have a SISO system
        if not sys.issiso():
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


def _RLFindRoots(nump, denp, kvect):
    """Find the roots for the root locus."""
    # Convert numerator and denominator to polynomials if they aren't
    roots = []
    for k in np.array(kvect, ndmin=1):
        curpoly = denp + k * nump
        curroots = curpoly.r
        if len(curroots) < denp.order:
            # if I have fewer poles than open loop, it is because i have
            # one at infinity
            curroots = np.insert(curroots, len(curroots), np.inf)

        curroots.sort()
        roots.append(curroots)

    mymat = row_stack(roots)
    return mymat


def _RLSortRoots(mymat):
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


def _RLZoomDispatcher(event, sys, ax_rlocus, plotstr):
    """Rootlocus plot zoom dispatcher"""
    sys_loop = sys if sys.issiso() else sys[0,0]

    nump, denp = _systopoly1d(sys_loop)
    xlim, ylim = ax_rlocus.get_xlim(), ax_rlocus.get_ylim()

    kvect, mymat, xlim, ylim = _default_gains(
        nump, denp, xlim=None, ylim=None, zoom_xlim=xlim, zoom_ylim=ylim)
    _removeLine('rootlocus', ax_rlocus)

    for i, col in enumerate(mymat.T):
        ax_rlocus.plot(real(col), imag(col), plotstr, label='rootlocus',
                       scalex=False, scaley=False)


def _RLClickDispatcher(event, sys, fig, ax_rlocus, plotstr, sisotool=False,
                       bode_plot_params=None, tvect=None):
    """Rootlocus plot click dispatcher"""

    # Zoom is handled by specialized callback above, only do gain plot
    if event.inaxes == ax_rlocus.axes and \
       plt.get_current_fig_manager().toolbar.mode not in \
       {'zoom rect', 'pan/zoom'}:

        # if a point is clicked on the rootlocus plot visually emphasize it
        K = _RLFeedbackClicksPoint(event, sys, fig, ax_rlocus, sisotool)
        if sisotool and K is not None:
            _SisotoolUpdate(sys, fig, K, bode_plot_params, tvect)

    # Update the canvas
    fig.canvas.draw()


def _RLFeedbackClicksPoint(event, sys, fig, ax_rlocus, sisotool=False):
    """Display root-locus gain feedback point for clicks on root-locus plot"""
    sys_loop = sys if sys.issiso() else sys[0,0]

    (nump, denp) = _systopoly1d(sys_loop)

    xlim = ax_rlocus.get_xlim()
    ylim = ax_rlocus.get_ylim()
    x_tolerance = 0.1 * abs((xlim[1] - xlim[0]))
    y_tolerance = 0.1 * abs((ylim[1] - ylim[0]))
    gain_tolerance = np.mean([x_tolerance, y_tolerance])*0.1

    # Catch type error when event click is in the figure but not in an axis
    try:
        s = complex(event.xdata, event.ydata)
        K = -1. / sys_loop(s)
        K_xlim = -1. / sys_loop(
            complex(event.xdata + 0.05 * abs(xlim[1] - xlim[0]), event.ydata))
        K_ylim = -1. / sys_loop(
            complex(event.xdata, event.ydata + 0.05 * abs(ylim[1] - ylim[0])))

    except TypeError:
        K = float('inf')
        K_xlim = float('inf')
        K_ylim = float('inf')

    gain_tolerance += 0.1 * max([abs(K_ylim.imag/K_ylim.real),
                                 abs(K_xlim.imag/K_xlim.real)])

    if abs(K.real) > 1e-8 and abs(K.imag / K.real) < gain_tolerance and \
       event.inaxes == ax_rlocus.axes and K.real > 0.:

        if isdtime(sys, strict=True):
            zeta = -np.cos(np.angle(np.log(s)))
        else:
            zeta = -1 * s.real / abs(s)

        # Display the parameters in the output window and figure
        print("Clicked at %10.4g%+10.4gj gain %10.4g damp %10.4g" %
              (s.real, s.imag, K.real, zeta))
        fig.suptitle(
            "Clicked at: %10.4g%+10.4gj  gain: %10.4g  damp: %10.4g" %
            (s.real, s.imag, K.real, zeta),
            fontsize=12 if int(mpl.__version__[0]) == 1 else 10)

        # Remove the previous line
        _removeLine(label='gain_point', ax=ax_rlocus)

        # Visualise clicked point, display all roots for sisotool mode
        if sisotool:
            mymat = _RLFindRoots(nump, denp, K.real)
            ax_rlocus.plot(
                [root.real for root in mymat],
                [root.imag for root in mymat],
                marker='s', markersize=6, zorder=20, label='gain_point', color='k')
        else:
            ax_rlocus.plot(s.real, s.imag, 'k.', marker='s', markersize=8,
                           zorder=20, label='gain_point')

        return K.real


def _removeLine(label, ax):
    """Remove a line from the ax when a label is specified"""
    for line in reversed(ax.lines):
        if line.get_label() == label:
            line.remove()
            del line


def _sgrid_func(fig=None, zeta=None, wn=None):
    if fig is None:
        fig = plt.gcf()
        ax = fig.gca()
    else:
        ax = fig.axes[1]

    # Get locator function for x-axis, y-axis tick marks
    xlocator = ax.get_xaxis().get_major_locator()
    ylocator = ax.get_yaxis().get_major_locator()

    # Decide on the location for the labels (?)
    ylim = ax.get_ylim()
    ytext_pos_lim = ylim[1] - (ylim[1] - ylim[0]) * 0.03
    xlim = ax.get_xlim()
    xtext_pos_lim = xlim[0] + (xlim[1] - xlim[0]) * 0.0

    # Create a list of damping ratios, if needed
    if zeta is None:
        zeta = _default_zetas(xlim, ylim)

    # Figure out the angles for the different damping ratios
    angles = []
    for z in zeta:
        if (z >= 1e-4) and (z <= 1):
            angles.append(np.pi/2 + np.arcsin(z))
        else:
            zeta.remove(z)
    y_over_x = np.tan(angles)

    # zeta-constant lines
    for index, yp in enumerate(y_over_x):
        ax.plot([0, xlocator()[0]], [0, yp * xlocator()[0]], color='gray',
                linestyle='dashed', linewidth=0.5)
        ax.plot([0, xlocator()[0]], [0, -yp * xlocator()[0]], color='gray',
                linestyle='dashed', linewidth=0.5)
        an = "%.2f" % zeta[index]
        if yp < 0:
            xtext_pos = 1/yp * ylim[1]
            ytext_pos = yp * xtext_pos_lim
            if np.abs(xtext_pos) > np.abs(xtext_pos_lim):
                xtext_pos = xtext_pos_lim
            else:
                ytext_pos = ytext_pos_lim
            ax.annotate(an, textcoords='data', xy=[xtext_pos, ytext_pos],
                        fontsize=8)
    ax.plot([0, 0], [ylim[0], ylim[1]],
            color='gray', linestyle='dashed', linewidth=0.5)

    # omega-constant lines
    angles = np.linspace(-90, 90, 20) * np.pi/180
    if wn is None:
        wn = _default_wn(xlocator(), ylocator())

    for om in wn:
        if om < 0:
            # Generate the lines for natural frequency curves
            yp = np.sin(angles) * np.abs(om)
            xp = -np.cos(angles) * np.abs(om)

            # Plot the natural frequency contours
            ax.plot(xp, yp, color='gray', linestyle='dashed', linewidth=0.5)

            # Annotate the natural frequencies by listing on x-axis
            # Note: need to filter values for proper plotting in Jupyter
            if (om > xlim[0]):
                an = "%.2f" % -om
                ax.annotate(an, textcoords='data', xy=[om, 0], fontsize=8)


def _default_zetas(xlim, ylim):
    """Return default list of damping coefficients

    This function computes a list of damping coefficients based on the limits
    of the graph.  A set of 4 damping coefficients are computed for the x-axis
    and a set of three damping coefficients are computed for the y-axis
    (corresponding to the normal 4:3 plot aspect ratio in `matplotlib`?).

    Parameters
    ----------
    xlim : array_like
        List of x-axis limits [min, max]
    ylim : array_like
        List of y-axis limits [min, max]

    Returns
    -------
    zeta : list
        List of default damping coefficients for the plot

    """
    # Damping coefficient lines that intersect the x-axis
    sep1 = -xlim[0] / 4
    ang1 = [np.arctan((sep1*i)/ylim[1]) for i in np.arange(1, 4, 1)]

    # Damping coefficient lines that intersection the y-axis
    sep2 = ylim[1] / 3
    ang2 = [np.arctan(-xlim[0]/(ylim[1]-sep2*i)) for i in np.arange(1, 3, 1)]

    # Put the lines together and add one at -pi/2 (negative real axis)
    angles = np.concatenate((ang1, ang2))
    angles = np.insert(angles, len(angles), np.pi/2)

    # Return the damping coefficients corresponding to these angles
    zeta = np.sin(angles)
    return zeta.tolist()


def _default_wn(xloc, yloc, max_lines=7):
    """Return default wn for root locus plot

    This function computes a list of natural frequencies based on the grid
    parameters of the graph.

    Parameters
    ----------
    xloc : array_like
        List of x-axis tick values
    ylim : array_like
        List of y-axis limits [min, max]
    max_lines : int, optional
        Maximum number of frequencies to generate (default = 7)

    Returns
    -------
    wn : list
        List of default natural frequencies for the plot

    """
    sep = xloc[1]-xloc[0]       # separation between x-ticks

    # Decide whether to use the x or y axis for determining wn
    if yloc[-1] / sep > max_lines*10:
        # y-axis scale >> x-axis scale
        wn = yloc                   # one frequency per y-axis tick mark
    else:
        wn = xloc                   # one frequency per x-axis tick mark

        # Insert additional frequencies to span the y-axis
        while np.abs(wn[0]) < yloc[-1]:
            wn = np.insert(wn, 0, wn[0]-sep)

    # If there are too many values, cut them in half
    while len(wn) > max_lines:
        wn = wn[0:-1:2]

    return wn


rlocus = root_locus
