# pzmap.py - computations involving poles and zeros
#
# Initial author: Richard M. Murray
# Creation date: 7 Sep 2009

"""Computations involving poles and zeros.

This module contains functions that compute poles, zeros and related
quantities for a linear system, as well as the main functions for
storing and plotting pole/zero and root locus diagrams.  (The actual
computation of root locus diagrams is in rlocus.py.)

"""

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy import imag, real

from . import config
from .config import _process_legacy_keyword
from .ctrlplot import ControlPlot, _get_color, _get_color_offset, \
    _get_line_labels, _process_ax_keyword, _process_legend_keywords, \
    _process_line_labels, _update_plot_title
from .grid import nogrid, sgrid, zgrid
from .iosys import isctime, isdtime
from .statesp import StateSpace
from .xferfcn import TransferFunction

__all__ = ['pole_zero_map', 'pole_zero_plot', 'pzmap', 'PoleZeroData',
           'PoleZeroList']


# Define default parameter values for this module
_pzmap_defaults = {
    'pzmap.grid': False,                 # Plot omega-damping grid
    'pzmap.marker_size': 6,             # Size of the markers
    'pzmap.marker_width': 1.5,          # Width of the markers
    'pzmap.expansion_factor': 1.8,      # Amount to scale plots beyond features
    'pzmap.buffer_factor': 1.05,        # Buffer to leave around plot peaks
}

#
# Classes for keeping track of pzmap plots
#
# The PoleZeroData class keeps track of the information that is on a
# pole/zero plot.
#
# In addition to the locations of poles and zeros, you can also save a set
# of gains and loci for use in generating a root locus plot.  The gain
# variable is a 1D array consisting of a list of increasing gains.  The
# loci variable is a 2D array indexed by [gain_idx, root_idx] that can be
# plotted using the `pole_zero_plot` function.
#
# The PoleZeroList class is used to return a list of pole/zero plots.  It
# is a lightweight wrapper on the built-in list class that includes a
# `plot` method, allowing plotting a set of root locus diagrams.
#
class PoleZeroData:
    """Pole/zero data object.

    This class is used as the return type for computing pole/zero responses
    and root locus diagrams.  It contains information on the location of
    system poles and zeros, as well as the gains and loci for root locus
    diagrams.

    Parameters
    ----------
    poles : ndarray
        1D array of system poles.
    zeros : ndarray
        1D array of system zeros.
    gains : ndarray, optional
        1D array of gains for root locus plots.
    loci : ndarray, optional
        2D array of poles, with each row corresponding to a gain.
    sysname : str, optional
        System name.
    sys : `StateSpace` or `TransferFunction`, optional
        System corresponding to the data.
    dt : None, True or float, optional
        System timebase (used for showing stability boundary).
    sort_loci : bool, optional
        Set to False to turn off sorting of loci into unique branches.

    """
    def __init__(
            self, poles, zeros, gains=None, loci=None, dt=None, sysname=None,
            sys=None, sort_loci=True):
        from .rlocus import _RLSortRoots
        self.poles = poles
        self.zeros = zeros
        self.gains = gains
        if loci is not None and sort_loci:
            self.loci = _RLSortRoots(loci)
        else:
            self.loci = loci
        self.dt = dt
        self.sysname = sysname
        self.sys = sys

    # Implement functions to allow legacy assignment to tuple
    def __iter__(self):
        return iter((self.poles, self.zeros))

    def plot(self, *args, **kwargs):
        """Plot the pole/zero data.

        See `pole_zero_plot` for description of arguments and keywords.

        """
        return pole_zero_plot(self, *args, **kwargs)


class PoleZeroList(list):
    """List of PoleZeroData objects with plotting capability."""
    def plot(self, *args, **kwargs):
        """Plot pole/zero data.

        See `pole_zero_plot` for description of arguments and keywords.

        """
        return pole_zero_plot(self, *args, **kwargs)

    def replot(self, cplt: ControlPlot):
        """Update the pole/zero loci of an existing plot.

        Parameters
        ----------
        cplt : `ControlPlot`
            Graphics handles of the existing plot.
        """
        pole_zero_replot(self, cplt)



# Pole/zero map
def pole_zero_map(sysdata):
    """Compute the pole/zero map for an LTI system.

    Parameters
    ----------
    sysdata : `StateSpace` or `TransferFunction`
        Linear system for which poles and zeros are computed.

    Returns
    -------
    pzmap_data : `PoleZeroMap`
        Pole/zero map containing the poles and zeros of the system.  Use
        ``pzmap_data.plot()`` or ``pole_zero_plot(pzmap_data)`` to plot the
        pole/zero map.

    """
    # Convert the first argument to a list
    syslist = sysdata if isinstance(sysdata, (list, tuple)) else [sysdata]

    responses = []
    for idx, sys in enumerate(syslist):
        responses.append(
            PoleZeroData(
                sys.poles(), sys.zeros(), dt=sys.dt, sysname=sys.name))

    if isinstance(sysdata, (list, tuple)):
        return PoleZeroList(responses)
    else:
        return responses[0]


# TODO: Implement more elegant cross-style axes. See:
#   https://matplotlib.org/2.0.2/examples/axes_grid/demo_axisline_style.html
#   https://matplotlib.org/2.0.2/examples/axes_grid/demo_curvelinear_grid.html
def pole_zero_plot(
        data, plot=None, grid=None, title=None, color=None, marker_size=None,
        marker_width=None, xlim=None, ylim=None, interactive=None, ax=None,
        scaling=None, initial_gain=None, label=None, **kwargs):
    """Plot a pole/zero map for a linear system.

    If the system data include root loci, a root locus diagram for the
    system is plotted.  When the root locus for a single system is plotted,
    clicking on a location on the root locus will mark the gain on all
    branches of the diagram and show the system gain and damping for the
    given pole in the axes title.  Set to False to turn off this behavior.

    Parameters
    ----------
    data : List of `PoleZeroData` objects or `LTI` systems
        List of pole/zero response data objects generated by pzmap_response()
        or root_locus_map() that are to be plotted.  If a list of systems
        is given, the poles and zeros of those systems will be plotted.
    grid : bool or str, optional
        If True plot omega-damping grid, if False show imaginary
        axis for continuous-time systems, unit circle for discrete-time
        systems.  If 'empty', do not draw any additional lines.  Default
        value is set by `config.defaults['pzmap.grid']` or
        `config.defaults['rlocus.grid']`.
    plot : bool, optional
        (legacy) If True a graph is generated with matplotlib,
        otherwise the poles and zeros are only computed and returned.
        If this argument is present, the legacy value of poles and
        zeros is returned.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : array of list of `matplotlib.lines.Line2D`
        The shape of the array is given by (nsys, 2) where nsys is the number
        of systems or responses passed to the function.  The second index
        specifies the pzmap object type:

            - lines[idx, 0]: poles
            - lines[idx, 1]: zeros

    cplt.axes : 2D array of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.
    cplt.legend : 2D array of `matplotlib.legend.Legend`
        Legend object(s) contained in the plot.
    poles, zeros : list of arrays
        (legacy) If the `plot` keyword is given, the system poles and zeros
        are returned.

    Other Parameters
    ----------------
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified and
        the current figure has a single axes, that axes is used.
        Otherwise, a new figure is created.
    color : matplotlib color spec, optional
        Specify the color of the markers and lines.
    initial_gain : float, optional
        If given, the specified system gain will be marked on the plot.
    interactive : bool, optional
        Turn off interactive mode for root locus plots.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with given
        label(s).  If data is a list, strings should be specified for each
        system.
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'upper right',
        with no legend for a single response.  Use False to suppress legend.
    marker_color : str, optional
        Set the color of the markers used for poles and zeros.
    marker_size : int, optional
        Set the size of the markers used for poles and zeros.
    marker_width : int, optional
        Set the line width of the markers used for poles and zeros.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    scaling : str or list, optional
        Set the type of axis scaling.  Can be 'equal' (default), 'auto', or
        a list of the form [xmin, xmax, ymin, ymax].
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on the
        plot or `legend_loc` has been specified.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).
    xlim : list, optional
        Set the limits for the x axis.
    ylim : list, optional
        Set the limits for the y axis.

    Notes
    -----
    By default, the pzmap function calls matplotlib.pyplot.axis('equal'),
    which means that trying to reset the axis limits may not behave as
    expected.  To change the axis limits, use the `scaling` keyword of use
    matplotlib.pyplot.gca().axis('auto') and then set the axis limits to
    the desired values.

    Pole/zero plots that use the continuous-time omega-damping grid do not
    work with the `ax` keyword argument, due to the way that axes grids
    are implemented.  The `grid` argument must be set to False or
    'empty' when using the `ax` keyword argument.

    The limits of the pole/zero plot are set based on the location features
    in the plot, including the location of poles, zeros, and local maxima
    of root locus curves.  The locations of local maxima are expanded by a
    buffer factor set by `config.defaults['phaseplot.buffer_factor']` that is
    applied to the locations of the local maxima.  The final axis limits
    are set to by the largest features in the plot multiplied by an
    expansion factor set by `config.defaults['phaseplot.expansion_factor']`.
    The default value for the buffer factor is 1.05 (5% buffer around local
    maxima) and the default value for the expansion factor is 1.8 (80%
    increase in limits around the most distant features).

    """
    # Get parameter values
    label = _process_line_labels(label)
    marker_size = config._get_param('pzmap', 'marker_size', marker_size, 6)
    marker_width = config._get_param('pzmap', 'marker_width', marker_width, 1.5)
    user_color = _process_legacy_keyword(kwargs, 'marker_color', 'color', color)
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)
    user_ax = ax
    xlim_user, ylim_user = xlim, ylim

    # If argument was a singleton, turn it into a tuple
    if not isinstance(data, (list, tuple)):
        data = [data]

    # If we are passed a list of systems, compute response first
    if all([isinstance(
            sys, (StateSpace, TransferFunction)) for sys in data]):
        # Get the response, popping off keywords used there
        pzmap_responses = pole_zero_map(data)
    elif all([isinstance(d, PoleZeroData) for d in data]):
        pzmap_responses = data
    else:
        raise TypeError("unknown system data type")

    # Decide whether we are plotting any root loci
    rlocus_plot = any([resp.loci is not None for resp in pzmap_responses])

    # Turn on interactive mode by default, if allowed
    if interactive is None and rlocus_plot and len(pzmap_responses) == 1 \
       and pzmap_responses[0].sys is not None:
        interactive = True

    # Legacy return value processing
    if plot is not None:
        warnings.warn(
            "pole_zero_plot() return value of poles, zeros is deprecated; "
            "use pole_zero_map()", FutureWarning)

        # Extract out the values that we will eventually return
        poles = [response.poles for response in pzmap_responses]
        zeros = [response.zeros for response in pzmap_responses]

    if plot is False:
        if len(data) == 1:
            return poles[0], zeros[0]
        else:
            return poles, zeros

    # Initialize the figure
    fig, ax = _process_ax_keyword(
        user_ax, rcParams=rcParams, squeeze=True, create_axes=False)
    legend_loc, _, show_legend = _process_legend_keywords(
        kwargs, None, 'upper right')

    # Make sure there are no remaining keyword arguments
    if kwargs:
        raise TypeError("unrecognized keywords: ", str(kwargs))

    if ax is None:
        # Determine what type of grid to use
        if rlocus_plot:
            from .rlocus import _rlocus_defaults
            grid = config._get_param('rlocus', 'grid', grid, _rlocus_defaults)
        else:
            grid = config._get_param('pzmap', 'grid', grid, _pzmap_defaults)

        # Create the axes with the appropriate grid
        with plt.rc_context(rcParams):
            if grid and grid != 'empty':
                if all([isctime(dt=response.dt) for response in data]):
                    ax, fig = sgrid(scaling=scaling)
                elif all([isdtime(dt=response.dt) for response in data]):
                    ax, fig = zgrid(scaling=scaling)
                else:
                    raise ValueError(
                        "incompatible time bases; don't know how to grid")
                # Store the limits for later use
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
            elif grid == 'empty':
                ax = plt.axes()
                xlim = ylim = [np.inf, -np.inf] # use data to set limits
            else:
                ax, fig = nogrid(data[0].dt, scaling=scaling)
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        # Store the limits for later use
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if grid is not None:
            warnings.warn("axis already exists; grid keyword ignored")

    # Get color offset for the next line to be drawn
    color_offset, color_cycle = _get_color_offset(ax)

    # Create a list of lines for the output
    out = np.empty(
        (len(pzmap_responses), 3 if rlocus_plot else 2), dtype=object)
    for i, j in itertools.product(range(out.shape[0]), range(out.shape[1])):
        out[i, j] = []          # unique list in each element

    # Plot the responses (and keep track of axes limits)
    for idx, response in enumerate(pzmap_responses):
        poles = response.poles
        zeros = response.zeros

        # Get the color to use for this response
        color = _get_color(user_color, offset=color_offset + idx)

        # Plot the locations of the poles and zeros
        if len(poles) > 0:
            if label is None:
                label_ = response.sysname if response.loci is None else None
            else:
                label_ = label[idx]
            out[idx, 0] = ax.plot(
                real(poles), imag(poles), marker='x', linestyle='',
                markeredgecolor=color, markerfacecolor=color,
                markersize=marker_size, markeredgewidth=marker_width,
                color=color, label=label_)
        if len(zeros) > 0:
            out[idx, 1] = ax.plot(
                real(zeros), imag(zeros), marker='o', linestyle='',
                markeredgecolor=color, markerfacecolor='none',
                markersize=marker_size, markeredgewidth=marker_width,
                color=color)

        # Plot the loci, if present
        if response.loci is not None:
            label_ = response.sysname if label is None else label[idx]
            for locus in response.loci.transpose():
                out[idx, 2] += ax.plot(
                    real(locus), imag(locus), color=color, label=label_)

            # Compute the axis limits to use based on the response
            resp_xlim, resp_ylim = _compute_root_locus_limits(response)

            # Keep track of the current limits
            xlim = [min(xlim[0], resp_xlim[0]), max(xlim[1], resp_xlim[1])]
            ylim = [min(ylim[0], resp_ylim[0]), max(ylim[1], resp_ylim[1])]

            # Plot the initial gain, if given
            if initial_gain is not None:
                _mark_root_locus_gain(ax, response.sys, initial_gain)

            # TODO: add arrows to root loci (reuse Nyquist arrow code?)

    # Set the axis limits to something reasonable
    if rlocus_plot:
        # Set up the limits for the plot using information from loci
        ax.set_xlim(xlim if xlim_user is None else xlim_user)
        ax.set_ylim(ylim if ylim_user is None else ylim_user)
    else:
        # No root loci => only set axis limits if users specified them
        if xlim_user is not None:
            ax.set_xlim(xlim_user)
        if ylim_user is not None:
            ax.set_ylim(ylim_user)

    # List of systems that are included in this plot
    lines, labels = _get_line_labels(ax)

    # Add legend if there is more than one system plotted
    if show_legend or len(labels) > 1 and show_legend != False:
        if response.loci is None:
            # Use "x o" for the system label, via matplotlib tuple handler
            from matplotlib.legend_handler import HandlerTuple
            from matplotlib.lines import Line2D

            line_tuples = []
            for pole_line in lines:
                zero_line = Line2D(
                    [0], [0], marker='o', linestyle='',
                    markeredgecolor=pole_line.get_markerfacecolor(),
                    markerfacecolor='none', markersize=marker_size,
                    markeredgewidth=marker_width)
                handle = (pole_line, zero_line)
                line_tuples.append(handle)

            with plt.rc_context(rcParams):
                legend = ax.legend(
                    line_tuples, labels, loc=legend_loc,
                    handler_map={tuple: HandlerTuple(ndivide=None)})
        else:
            # Regular legend, with lines
            with plt.rc_context(rcParams):
                legend = ax.legend(lines, labels, loc=legend_loc)
    else:
        legend = None

    # Add the title
    if title is None:
        title = ("Root locus plot for " if rlocus_plot
                 else "Pole/zero plot for ") + ", ".join(labels)
    if user_ax is None:
        _update_plot_title(
            title, fig, rcParams=rcParams, frame='figure',
            use_existing=False)

    # Add dispatcher to handle choosing a point on the diagram
    if interactive:
        if len(pzmap_responses) > 1:
            raise NotImplementedError(
                "interactive mode only allowed for single system")
        elif pzmap_responses[0].sys == None:
            raise SystemError("missing system information")
        else:
            sys = pzmap_responses[0].sys

        # Define function to handle mouse clicks
        def _click_dispatcher(event):
            # Find the gain corresponding to the clicked point
            K, s = _find_root_locus_gain(event, sys, ax)

            if K is not None:
                # Mark the gain on the root locus diagram
                _mark_root_locus_gain(ax, sys, K)

                # Display the parameters in the axes title
                with plt.rc_context(rcParams):
                    ax.set_title(_create_root_locus_label(sys, K, s))

            ax.figure.canvas.draw()

        fig.canvas.mpl_connect('button_release_event', _click_dispatcher)

    # Legacy processing: return locations of poles and zeros as a tuple
    if plot is True:
        if len(data) == 1:
            return poles, zeros
        else:
            TypeError("system lists not supported with legacy return values")

    return ControlPlot(out, ax, fig, legend=legend)


def pole_zero_replot(pzmap_responses, cplt):
    """Update the loci of a plot after zooming/panning.

    Parameters
    ----------
    pzmap_responses : PoleZeroMap list
        Responses to update.
    cplt : ControlPlot
        Collection of plot handles.
    """

    for idx, response in enumerate(pzmap_responses):

        # remove the old data
        for l in cplt.lines[idx, 2]:
            l.set_data([], [])

        # update the line data
        if response.loci is not None:

            for il, locus in enumerate(response.loci.transpose()):
                try:
                    cplt.lines[idx,2][il].set_data(real(locus), imag(locus))
                except IndexError:
                    # not expected, but more lines apparently needed
                    cplt.lines[idx,2].append(cplt.ax[0,0].plot(
                        real(locus), imag(locus)))


# Utility function to find gain corresponding to a click event
def _find_root_locus_gain(event, sys, ax):
    # Get the current axis limits to set various thresholds
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # Catch type error when event click is in the figure but not on curve
    try:
        s = complex(event.xdata, event.ydata)
        K = -1. / sys(s)
        K_xlim = -1. / sys(
            complex(event.xdata + 0.05 * abs(xlim[1] - xlim[0]), event.ydata))
        K_ylim = -1. / sys(
            complex(event.xdata, event.ydata + 0.05 * abs(ylim[1] - ylim[0])))
    except TypeError:
        K, s = float('inf'), None
        K_xlim = K_ylim = float('inf')

    #
    # Compute tolerances for deciding if we clicked on the root locus
    #
    # This is a bit of black magic that sets some limits for how close we
    # need to be to the root locus in order to consider it a click on the
    # actual curve.  Otherwise, we will just ignore the click.

    x_tolerance = 0.1 * abs((xlim[1] - xlim[0]))
    y_tolerance = 0.1 * abs((ylim[1] - ylim[0]))
    gain_tolerance = np.mean([x_tolerance, y_tolerance]) * 0.1 + \
        0.1 * max([abs(K_ylim.imag/K_ylim.real), abs(K_xlim.imag/K_xlim.real)])

    # Decide whether to pay attention to this event
    if abs(K.real) > 1e-8 and abs(K.imag / K.real) < gain_tolerance and \
       event.inaxes == ax.axes and K.real > 0.:
        return K.real, s
    else:
        return None, None


# Mark points corresponding to a given gain on root locus plot
def _mark_root_locus_gain(ax, sys, K):
    from .rlocus import _RLFindRoots, _systopoly1d

    # Remove any previous gain points
    for line in reversed(ax.lines):
        if line.get_label() == '_gain_point':
            line.remove()
            del line

    # Visualize clicked point, displaying all roots
    # TODO: allow marker parameters to be set
    nump, denp = _systopoly1d(sys)
    root_array = _RLFindRoots(nump, denp, K.real)
    ax.plot(
        [root.real for root in root_array], [root.imag for root in root_array],
        marker='s', markersize=6, zorder=20, label='_gain_point', color='k')


# Return a string identifying a clicked point
def _create_root_locus_label(sys, K, s):
    # Figure out the damping ratio
    if isdtime(sys, strict=True):
        zeta = -np.cos(np.angle(np.log(s)))
    else:
        zeta = -1 * s.real / abs(s)

    return "Clicked at: %.4g%+.4gj   gain = %.4g  damping = %.4g" % \
        (s.real, s.imag, K.real, zeta)


# Utility function to compute limits for root loci
def _compute_root_locus_limits(response):
    loci = response.loci

    # Start with information about zeros, if present
    if response.sys is not None and response.sys.zeros().size > 0:
        xlim = [
            min(0, np.min(response.sys.zeros().real)),
            max(0, np.max(response.sys.zeros().real))
        ]
        ylim = max(0, np.max(response.sys.zeros().imag))
    else:
        xlim, ylim = [np.inf, -np.inf], 0

    # Go through each locus and look for features
    rho = config._get_param('pzmap', 'buffer_factor')
    for locus in loci.transpose():
        # Include all starting points
        xlim = [min(xlim[0], locus[0].real), max(xlim[1], locus[0].real)]
        ylim = max(ylim, locus[0].imag)

        # Find the local maxima of root locus curve
        xpeaks = np.where(
            np.diff(np.abs(locus.real)) < 0, locus.real[0:-1], 0)
        if xpeaks.size > 0:
            xlim = [
                min(xlim[0], np.min(xpeaks) * rho),
                max(xlim[1], np.max(xpeaks) * rho)
            ]

        ypeaks = np.where(
            np.diff(np.abs(locus.imag)) < 0, locus.imag[0:-1], 0)
        if ypeaks.size > 0:
            ylim = max(ylim, np.max(ypeaks) * rho)

    if isctime(dt=response.dt):
        # Adjust the limits to include some space around features
        # TODO: use _k_max and project out to max k for all value?
        rho = config._get_param('pzmap', 'expansion_factor')
        xlim[0] = rho * xlim[0] if xlim[0] < 0 else 0
        xlim[1] = rho * xlim[1] if xlim[1] > 0 else 0
        ylim = rho * ylim if ylim > 0 else np.max(np.abs(xlim))

    # Make sure the limits make sense
    if xlim == [0, 0]:
        xlim = [-1, 1]
    if ylim == 0:
        ylim = 1

    return xlim, [-ylim, ylim]


pzmap = pole_zero_plot
