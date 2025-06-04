# rlocus.py - code for computing a root locus plot
#
# Initial author: Ryan Krauss
# Creation date: 2010
#
# RMM, 17 June 2010: modified to be a standalone piece of code
#
# RMM, 2 April 2011: modified to work with new LTI structure
#
# Sawyer B. Fuller (minster@uw.edu) 21 May 2020: added compatibility
# with discrete-time systems.

"""Code for computing a root locus plot."""

import warnings

import numpy as np
import scipy.signal  # signal processing toolbox
from numpy import poly1d, vstack, zeros_like

from . import config
from .ctrlplot import ControlPlot
from .exception import ControlMIMONotImplemented
from .lti import LTI
from .xferfcn import _convert_to_transfer_function

__all__ = ['root_locus_map', 'root_locus_plot', 'root_locus', 'rlocus']

# Default values for module parameters
_rlocus_defaults = {
    'rlocus.grid': True,
}


# Root locus map
def root_locus_map(sysdata, gains=None, xlim=None, ylim=None):
    """Compute the root locus map for an LTI system.

    Calculate the root locus by finding the roots of 1 + k * G(s) where G
    is a linear system and k varies over a range of gains.

    Parameters
    ----------
    sysdata : LTI system or list of LTI systems
        Linear input/output systems (SISO only, for now).
    gains : array_like, optional
        Gains to use in computing plot of closed-loop poles.  If not given,
        gains are chosen to include the main features of the root locus map.
    xlim : tuple or list, optional
        Set limits of x axis (see `matplotlib.axes.Axes.set_xlim`).
    ylim : tuple or list, optional
        Set limits of y axis (see `matplotlib.axes.Axes.set_ylim`).

    Returns
    -------
    rldata : `PoleZeroData` or list of `PoleZeroData`
        Root locus data object(s).  The loci of the root locus diagram are
        available in the array `rldata.loci`, indexed by the gain index and
        the locus index, and the gains are in the array `rldata.gains`.

    Notes
    -----
    For backward compatibility, the `rldata` return object can be
    assigned to the tuple ``(roots, gains)``.

    """
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
            kvect, root_array, _, _ = _default_gains(nump, denp, xlim, ylim)
        else:
            kvect = np.atleast_1d(gains)
            root_array = _RLFindRoots(nump, denp, kvect)
            root_array = _RLSortRoots(root_array)

        responses.append(PoleZeroData(
            sys.poles(), sys.zeros(), kvect, root_array, sort_loci=False,
            dt=sys.dt, sysname=sys.name, sys=sys))

    if isinstance(sysdata, (list, tuple)):
        return PoleZeroList(responses)
    else:
        return responses[0]


def root_locus_plot(
        sysdata, gains=None, grid=None, plot=None, **kwargs):

    """Root locus plot.

    Calculate the root locus by finding the roots of 1 + k * G(s) where G
    is a linear system and k varies over a range of gains.

    Parameters
    ----------
    sysdata : PoleZeroMap or LTI object or list
        Linear input/output systems (SISO only, for now).
    gains : array_like, optional
        Gains to use in computing plot of closed-loop poles.  If not given,
        gains are chosen to include the main features of the root locus map.
    xlim : tuple or list, optional
        Set limits of x axis (see `matplotlib.axes.Axes.set_xlim`).
    ylim : tuple or list, optional
        Set limits of y axis (see `matplotlib.axes.Axes.set_ylim`).
    plot : bool, optional
        (legacy) If given, `root_locus_plot` returns the legacy return values
        of roots and gains.  If False, just return the values with no plot.
    grid : bool or str, optional
        If True plot omega-damping grid, if False show imaginary axis
        for continuous-time systems, unit circle for discrete-time systems.
        If 'empty', do not draw any additional lines.  Default value is set
        by `config.defaults['rlocus.grid']`.
    initial_gain : float, optional
        Mark the point on the root locus diagram corresponding to the
        given gain.
    color : matplotlib color spec, optional
        Specify the color of the markers and lines.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : array of list of `matplotlib.lines.Line2D`
        The shape of the array is given by (nsys, 3) where nsys is the number
        of systems or responses passed to the function.  The second index
        specifies the object type:

              - lines[idx, 0]: poles
              - lines[idx, 1]: zeros
              - lines[idx, 2]: loci

    cplt.axes : 2D array of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.
    cplt.legend : 2D array of `matplotlib.legend.Legend`
        Legend object(s) contained in the plot.
    roots, gains : ndarray
        (legacy) If the `plot` keyword is given, returns the closed-loop
        root locations, arranged such that each row corresponds to a gain,
        and the array of gains (same as `gains` keyword argument if provided).

    Other Parameters
    ----------------
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified and
        the current figure has a single axes, that axes is used.
        Otherwise, a new figure is created.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with the given
        label(s).  If sysdata is a list, strings should be specified for each
        system.
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'center right',
        with no legend for a single response.  Use False to suppress legend.
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on the
        plot or `legend_loc` has been specified.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).

    Notes
    -----
    The root_locus_plot function calls matplotlib.pyplot.axis('equal'), which
    means that trying to reset the axis limits may not behave as expected.
    To change the axis limits, use matplotlib.pyplot.gca().axis('auto') and
    then set the axis limits to the desired values.

    """
    # Legacy parameters
    for oldkey in ['kvect', 'k']:
        gains = config._process_legacy_keyword(kwargs, oldkey, 'gains', gains)

    if isinstance(sysdata, list) and all(
            [isinstance(sys, LTI) for sys in sysdata]) or \
            isinstance(sysdata, LTI):
        responses = root_locus_map(sysdata, gains=gains)
    else:
        responses = sysdata

    #
    # Process `plot` keyword
    #
    # See bode_plot for a description of how this keyword is handled to
    # support legacy implementations of root_locus.
    #
    if plot is not None:
        warnings.warn(
            "root_locus() return value of roots, gains is deprecated; "
            "use root_locus_map()", FutureWarning)

    if plot is False:
        return responses.loci, responses.gains

    # Plot the root loci
    cplt = responses.plot(grid=grid, **kwargs)

    # Add a reaction to axis scale changes, if given LTI systems, and
    # there is no set of pre-defined gains
    if gains is None:
        add_loci_recalculate(sysdata, cplt, cplt.axes[0,0])

    # Legacy processing: return locations of poles and zeros as a tuple
    if plot is True:
        return responses.loci, responses.gains

    return ControlPlot(cplt.lines, cplt.axes, cplt.figure)


def add_loci_recalculate(sysdata, cplt, axis):
    """Add a callback to re-calculate the loci data fitting a zoom action.

    Parameters
    ----------
    sysdata: LTI object or list
        Linear input/output systems (SISO only, for now).
    cplt: ControlPlot
        Collection of plot handles.
    axis: matplotlib.axes.Axis
        Axis on which callbacks are installed.
    """

    # if LTI, treat everything as a list of lti
    if isinstance(sysdata, LTI):
        sysdata = [sysdata]

    # check that we can actually recalculate the loci
    if isinstance(sysdata, list) and all(
       [isinstance(sys, LTI) for sys in sysdata]):

        # callback function for axis change (zoom, pan) events
        # captures the sysdata object and cplt
        def _zoom_adapter(_ax):
            newresp = root_locus_map(sysdata, None,
                                     _ax.get_xlim(),
                                     _ax.get_ylim())
            newresp.replot(cplt)

        # connect the callback to axis changes
        axis.callbacks.connect('xlim_changed', _zoom_adapter)
        axis.callbacks.connect('ylim_changed', _zoom_adapter)


def _default_gains(num, den, xlim, ylim):
    """Unsupervised gains calculation for root locus plot.

    References
    ----------
    .. [1] Ogata, K. (2002). Modern control engineering (4th
       ed.). Upper Saddle River, NJ : New Delhi: Prentice Hall..

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
        # Root locus is on real axis (common), use just x distance
        tolerance = x_tolerance
    else:
        tolerance = np.min([x_tolerance, y_tolerance])
    indexes_too_far = _indexes_filt(root_array, tolerance)

    # Add more points into the root locus for points that are too far apart
    while len(indexes_too_far) > 0 and kvect.size < 5000:
        for counter, index in enumerate(indexes_too_far):
            index = index + counter*3
            new_gains = np.linspace(kvect[index], kvect[index + 1], 5)
            new_points = _RLFindRoots(num, den, new_gains[1:4])
            kvect = np.insert(kvect, index + 1, new_gains[1:4])
            root_array = np.insert(root_array, index + 1, new_points, axis=0)

        root_array = _RLSortRoots(root_array)
        indexes_too_far = _indexes_filt(root_array, tolerance)

    new_gains = kvect[-1] * np.hstack((np.logspace(0, 3, 4)))
    new_points = _RLFindRoots(num, den, new_gains[1:4])
    kvect = np.append(kvect, new_gains[1:4])
    root_array = np.concatenate((root_array, new_points), axis=0)
    root_array = _RLSortRoots(root_array)
    return kvect, root_array, xlim, ylim


def _indexes_filt(root_array, tolerance):
    """Calculate the distance between points and return the indices.

    Filter the indexes so only the resolution of points within the xlim and
    ylim is improved when zoom is used.

    """
    distance_points = np.abs(np.diff(root_array, axis=0))
    indexes_too_far = list(np.unique(np.where(distance_points > tolerance)[0]))
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
    """Extract numerator and denominator polynomials for a system"""
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

    return vstack(roots)


def _RLSortRoots(roots):
    """Sort the roots from _RLFindRoots, so that the root
    locus doesn't show weird pseudo-branches as roots jump from
    one branch to another."""

    sorted = zeros_like(roots)
    sorted[0] = roots[0]
    for n, row in enumerate(roots[1:], start=1):
        # sort the current row by finding the element with the
        # smallest absolute distance to each root in the
        # previous row
        prevrow = sorted[n-1]
        available = list(range(len(prevrow)))
        for elem in row:
            evect = elem - prevrow[available]
            ind1 = abs(evect).argmin()
            ind = available.pop(ind1)
            sorted[n, ind] = elem
    return sorted


# Alternative ways to call these functions
root_locus = root_locus_plot
rlocus = root_locus_plot
