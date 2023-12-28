# rlocus.py - code for computing a root locus plot
# Code contributed by Ryan Krauss, 2010
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

# Packages used by this module
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, poly1d, row_stack, zeros_like, real, imag
import scipy.signal             # signal processing toolbox
from .iosys import isdtime
from .xferfcn import _convert_to_transfer_function
from .exception import ControlMIMONotImplemented
from .grid import sgrid, zgrid
from . import config
import warnings

__all__ = ['root_locus_map', 'root_locus_plot', 'root_locus', 'rlocus']

# Default values for module parameters
# TODO: merge these with pzmap parameters (?)
_rlocus_defaults = {
    'rlocus.grid': True,
    'rlocus.plotstr': 'C0',     # default color cycle [TODO: not used?]
    'rlocus.print_gain': True,
    'rlocus.plot': True
}


# Root locus map
# TODO: add docstring
def root_locus_map(sysdata, gains=None):
    from .pzmap import PoleZeroData, PoleZeroList

    # Convert the first argument to a list
    syslist = sysdata if isinstance(sysdata, (list, tuple)) else [sysdata]

    responses = []
    for idx, sys in enumerate(syslist):
        if not sys.issiso():
            raise ControlMIMONotImplemented(
                "sys must be single-input single-output (SISO)")

        # Convert numerator and denominator to polynomials if they aren't
        nump, denp = _systopoly1d(sys[0, 0])

        if gains is None:
            kvect, root_array, _, _ = _default_gains(nump, denp, None, None)
        else:
            kvect = np.atleast_1d(gains)
            root_array = _RLFindRoots(nump, denp, kvect)
            root_array = _RLSortRoots(root_array)

        responses.append(PoleZeroData(
            sys.poles(), sys.zeros(), kvect, root_array,
            dt=sys.dt, sysname=sys.name, sys=sys))

    if isinstance(sysdata, (list, tuple)):
        return PoleZeroList(responses)
    else:
        return responses[0]


def root_locus_plot(
        sysdata, kvect=None, grid=None, plot=None, **kwargs):
    """Root locus plot.

    Calculate the root locus by finding the roots of 1 + k * G(s) where G
    is a linear system with transfer function num(s)/den(s) and each k is
    an element of kvect.

    Parameters
    ----------
    sys : LTI object
        Linear input/output systems (SISO only, for now).
    kvect : array_like, optional
        Gains to use in computing plot of closed-loop poles.
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
    initial_gain : float, optional
        Specify the initial gain to use when marking current gain. [TODO: update]

    Returns
    -------
    roots : ndarray
        Closed-loop root locations, arranged in which each row corresponds
        to a gain in gains.
    gains : ndarray
        Gains used.  Same as kvect keyword argument if provided.

    Notes
    -----
    The root_locus function calls matplotlib.pyplot.axis('equal'), which
    means that trying to reset the axis limits may not behave as expected.
    To change the axis limits, use matplotlib.pyplot.gca().axis('auto') and
    then set the axis limits to the desired values.

    """
    from .pzmap import pole_zero_plot

    # Set default parameters
    # TODO: move this to pole_zero_plot()
    grid = config._get_param('rlocus', 'grid', grid, _rlocus_defaults)

    responses = root_locus_map(sysdata, gains=kvect)

    #
    # Process `plot` keyword
    #
    # See bode_plot for a description of how this keyword is handled to
    # support legacy implementatoins of root_locus.
    #
    if plot is not None:
        warnings.warn(
            "`root_locus` return values of roots, gains is deprecated; "
            "use root_locus_map()", DeprecationWarning)

    if plot is False:
        return responses.loci, responses.gains

    # Plot the root loci
    out = responses.plot(grid=grid, **kwargs)

    # Legacy processing: return locations of poles and zeros as a tuple
    if plot is True:
        return responses.loci, responses.gains

    return out


# TODO: get rid of zoom functionality?
def _default_gains(num, den, xlim, ylim, zoom_xlim=None, zoom_ylim=None):
    """Unsupervised gains calculation for root locus plot.

    References
    ----------
    Ogata, K. (2002). Modern control engineering (4th ed.). Upper
    Saddle River, NJ : New Delhi: Prentice Hall..

    """
    # Compute the break points on the real axis for the root locus plot
    k_break, real_break = _break_points(num, den)

    # Decide on the maximum gain to use and create the gain vector
    kmax = _k_max(num, den, real_break, k_break)
    kvect = np.hstack((np.linspace(0, kmax, 50), np.real(k_break)))
    kvect.sort()

    # Find the roots for all of the gains and sort them
    root_array = _RLFindRoots(num, den, kvect)
    root_array = _RLSortRoots(root_array)

    # Keep track of the open loop poles and zeros
    open_loop_poles = den.roots
    open_loop_zeros = num.roots

    # ???
    if open_loop_zeros.size != 0 and \
       open_loop_zeros.size < open_loop_poles.size:
        open_loop_zeros_xl = np.append(
            open_loop_zeros,
            np.ones(open_loop_poles.size - open_loop_zeros.size)
            * open_loop_zeros[-1])
        root_array_xl = np.append(root_array, open_loop_zeros_xl)
    else:
        root_array_xl = root_array
    singular_points = np.concatenate((num.roots, den.roots), axis=0)
    important_points = np.concatenate((singular_points, real_break), axis=0)
    important_points = np.concatenate((important_points, np.zeros(2)), axis=0)
    root_array_xl = np.append(root_array_xl, important_points)

    false_gain = float(den.coeffs[0]) / float(num.coeffs[0])
    if false_gain < 0 and not den.order > num.order:
        # TODO: make error message more understandable
        raise ValueError("Not implemented support for 0 degrees root locus "
                         "with equal order of numerator and denominator.")

    if xlim is None and false_gain > 0:
        x_tolerance = 0.05 * (np.max(np.real(root_array_xl))
                              - np.min(np.real(root_array_xl)))
        xlim = _ax_lim(root_array_xl)
    elif xlim is None and false_gain < 0:
        axmin = np.min(np.real(important_points)) \
            - (np.max(np.real(important_points))
               - np.min(np.real(important_points)))
        axmin = np.min(np.array([axmin, np.min(np.real(root_array_xl))]))
        axmax = np.max(np.real(important_points)) \
            + np.max(np.real(important_points)) \
            - np.min(np.real(important_points))
        axmax = np.max(np.array([axmax, np.max(np.real(root_array_xl))]))
        xlim = [axmin, axmax]
        x_tolerance = 0.05 * (axmax - axmin)
    else:
        x_tolerance = 0.05 * (xlim[1] - xlim[0])

    if ylim is None:
        y_tolerance = 0.05 * (np.max(np.imag(root_array_xl))
                              - np.min(np.imag(root_array_xl)))
        ylim = _ax_lim(root_array_xl * 1j)
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
    indexes_too_far = _indexes_filt(
        root_array, tolerance, zoom_xlim, zoom_ylim)

    # Add more points into the root locus for points that are too far apart
    while len(indexes_too_far) > 0 and kvect.size < 5000:
        for counter, index in enumerate(indexes_too_far):
            index = index + counter*3
            new_gains = np.linspace(kvect[index], kvect[index + 1], 5)
            new_points = _RLFindRoots(num, den, new_gains[1:4])
            kvect = np.insert(kvect, index + 1, new_gains[1:4])
            root_array = np.insert(root_array, index + 1, new_points, axis=0)

        root_array = _RLSortRoots(root_array)
        indexes_too_far = _indexes_filt(
            root_array, tolerance, zoom_xlim, zoom_ylim)

    new_gains = kvect[-1] * np.hstack((np.logspace(0, 3, 4)))
    new_points = _RLFindRoots(num, den, new_gains[1:4])
    kvect = np.append(kvect, new_gains[1:4])
    root_array = np.concatenate((root_array, new_points), axis=0)
    root_array = _RLSortRoots(root_array)
    return kvect, root_array, xlim, ylim


def _indexes_filt(root_array, tolerance, zoom_xlim=None, zoom_ylim=None):
    """Calculate the distance between points and return the indices.

    Filter the indexes so only the resolution of points within the xlim and
    ylim is improved when zoom is used.

    """
    distance_points = np.abs(np.diff(root_array, axis=0))
    indexes_too_far = list(np.unique(np.where(distance_points > tolerance)[0]))

    if zoom_xlim is not None and zoom_ylim is not None:
        x_tolerance_zoom = 0.05 * (zoom_xlim[1] - zoom_xlim[0])
        y_tolerance_zoom = 0.05 * (zoom_ylim[1] - zoom_ylim[0])
        tolerance_zoom = np.min([x_tolerance_zoom, y_tolerance_zoom])
        indexes_too_far_zoom = list(
            np.unique(np.where(distance_points > tolerance_zoom)[0]))
        indexes_too_far_filtered = []

        for index in indexes_too_far_zoom:
            for point in root_array[index]:
                if (zoom_xlim[0] <= point.real <= zoom_xlim[1]) and \
                   (zoom_ylim[0] <= point.imag <= zoom_ylim[1]):
                    indexes_too_far_filtered.append(index)
                    break

        # Check if zoom box is not overshot & insert points where neccessary
        if len(indexes_too_far_filtered) == 0 and len(root_array) < 500:
            limits = [zoom_xlim[0], zoom_xlim[1], zoom_ylim[0], zoom_ylim[1]]
            for index, limit in enumerate(limits):
                if index <= 1:
                    asign = np.sign(real(root_array)-limit)
                else:
                    asign = np.sign(imag(root_array) - limit)
                signchange = ((np.roll(asign, 1, axis=0)
                               - asign) != 0).astype(int)
                signchange[0] = np.zeros((len(root_array[0])))
                if len(np.where(signchange == 1)[0]) > 0:
                    indexes_too_far_filtered.append(
                        np.where(signchange == 1)[0][0]-1)

        if len(indexes_too_far_filtered) > 0:
            if indexes_too_far_filtered[0] != 0:
                indexes_too_far_filtered.insert(
                    0, indexes_too_far_filtered[0]-1)
            if not indexes_too_far_filtered[-1] + 1 >= len(root_array) - 2:
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


def _ax_lim(root_array):
    """Utility to get the axis limits"""
    axmin = np.min(np.real(root_array))
    axmax = np.max(np.real(root_array))
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
    for k in np.atleast_1d(kvect):
        curpoly = denp + k * nump
        curroots = curpoly.r
        if len(curroots) < denp.order:
            # if I have fewer poles than open loop, it is because i have
            # one at infinity
            curroots = np.append(curroots, np.inf)

        curroots.sort()
        roots.append(curroots)

    return row_stack(roots)


def _RLSortRoots(roots):
    """Sort the roots from _RLFindRoots, so that the root
    locus doesn't show weird pseudo-branches as roots jump from
    one branch to another."""

    sorted = zeros_like(roots)
    for n, row in enumerate(roots):
        if n == 0:
            sorted[n, :] = row
        else:
            # sort the current row by finding the element with the
            # smallest absolute distance to each root in the
            # previous row
            available = list(range(len(prevrow)))
            for elem in row:
                evect = elem - prevrow[available]
                ind1 = abs(evect).argmin()
                ind = available.pop(ind1)
                sorted[n, ind] = elem
        prevrow = sorted[n, :]
    return sorted


# Alternative ways to call these functions
root_locus = root_locus_plot
rlocus = root_locus_plot
