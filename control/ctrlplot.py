# ctrlplot.py - utility functions for plotting
# RMM, 14 Jun 2024
#

"""Utility functions for plotting.

This module contains a collection of functions that are used by
various plotting functions.

"""

# Code pattern for control system plotting functions:
#
# def name_plot(sysdata, *fmt, plot=None, **kwargs):
#     # Process keywords and set defaults
#     ax = kwargs.pop('ax', None)
#     color = kwargs.pop('color', None)
#     label = kwargs.pop('label', None)
#     rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)
#
#     # Make sure all keyword arguments were processed (if not checked later)
#     if kwargs:
#         raise TypeError("unrecognized keywords: ", str(kwargs))
#
#     # Process the data (including generating responses for systems)
#     sysdata = list(sysdata)
#     if any([isinstance(sys, InputOutputSystem) for sys in sysdata]):
#         data = name_response(sysdata)
#     nrows = max([data.noutputs for data in sysdata])
#     ncols = max([data.ninputs for data in sysdata])
#
#     # Legacy processing of plot keyword
#     if plot is False:
#         return data.x, data.y
#
#     # Figure out the shape of the plot and find/create axes
#     fig, ax_array = _process_ax_keyword(ax, (nrows, ncols), rcParams)
#     legend_loc, legend_map, show_legend = _process_legend_keywords(
#         kwargs, (nrows, ncols), 'center right')
#
#     # Customize axes (curvilinear grids, shared axes, etc)
#
#     # Plot the data
#     lines = np.empty(ax_array.shape, dtype=object)
#     for i in range(ax_array.shape[0]):
#         for j in range(ax_array.shape[1]):
#             lines[i, j] = []
#     line_labels = _process_line_labels(label, ntraces, nrows, ncols)
#     color_offset, color_cycle = _get_color_offset(ax)
#     for i, j in itertools.product(range(nrows), range(ncols)):
#         ax = ax_array[i, j]
#         for k in range(ntraces):
#             if color is None:
#                 color = _get_color(
#                     color, fmt=fmt, offset=k, color_cycle=color_cycle)
#             label = line_labels[k, i, j]
#             lines[i, j] += ax.plot(data.x, data.y, color=color, label=label)
#
#     # Customize and label the axes
#     for i, j in itertools.product(range(nrows), range(ncols)):
#         ax_array[i, j].set_xlabel("x label")
#         ax_array[i, j].set_ylabel("y label")
#
#     # Create legends
#     if show_legend != False:
#         legend_array = np.full(ax_array.shape, None, dtype=object)
#         for i, j in itertools.product(range(nrows), range(ncols)):
#             if legend_map[i, j] is not None:
#                 lines = ax_array[i, j].get_lines()
#                 labels = _make_legend_labels(lines)
#                 if len(labels) > 1:
#                     legend_array[i, j] = ax.legend(
#                         lines, labels, loc=legend_map[i, j])
#     else:
#         legend_array = None
#
#     # Update the plot title (only if ax was not given)
#     sysnames = [response.sysname for response in data]
#     if ax is None and title is None:
#         title = "Name plot for " + ", ".join(sysnames)
#         _update_plot_title(title, fig, rcParams=rcParams)
#     elif ax == None:
#         _update_plot_title(title, fig, rcParams=rcParams, use_existing=False)
#
#     # Legacy processing of plot keyword
#     if plot is True:
#         return data
#
#     return ControlPlot(lines, ax_array, fig, legend=legend_map)

import itertools
import warnings
from os.path import commonprefix

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

__all__ = [
    'ControlPlot', 'suptitle', 'get_plot_axes', 'pole_zero_subplots',
    'rcParams', 'reset_rcParams']

#
# Style parameters
#

rcParams_default = {
    'axes.labelsize': 'small',
    'axes.titlesize': 'small',
    'figure.titlesize': 'medium',
    'legend.fontsize': 'x-small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
}
_ctrlplot_rcParams = rcParams_default.copy()    # provide access inside module
rcParams = _ctrlplot_rcParams                   # provide access outside module

_ctrlplot_defaults = {'ctrlplot.rcParams': _ctrlplot_rcParams}


#
# Control figure
#

class ControlPlot():
    """Return class for control platting functions.

    This class is used as the return type for control plotting functions.
    It contains the information required to access portions of the plot
    that the user might want to adjust, as well as providing methods to
    modify some of the properties of the plot.

    A control figure consists of a `matplotlib.figure.Figure` with
    an array of `matplotlib.axes.Axes`.  Each axes in the figure has
    a number of lines that represent the data for the plot.  There may also
    be a legend present in one or more of the axes.

    Parameters
    ----------
    lines : array of list of `matplotlib.lines.Line2D`
        Array of Line2D objects for each line in the plot.  Generally, the
        shape of the array matches the subplots shape and the value of the
        array is a list of Line2D objects in that subplot.  Some plotting
        functions will return variants of this structure, as described in
        the individual documentation for the functions.
    axes : 2D array of `matplotlib.axes.Axes`
        Array of Axes objects for each subplot in the plot.
    figure : `matplotlib.figure.Figure`
        Figure on which the Axes are drawn.
    legend : `matplotlib.legend.Legend` (instance or ndarray)
        Legend object(s) for the plot.  If more than one legend is
        included, this will be an array with each entry being either None
        (for no legend) or a legend object.

    """
    def __init__(self, lines, axes=None, figure=None, legend=None):
        self.lines = lines
        if axes is None:
            _get_axes = np.vectorize(lambda lines: lines[0].axes)
            axes = _get_axes(lines)
        self.axes = np.atleast_2d(axes)
        if figure is None:
            figure = self.axes[0, 0].figure
        self.figure = figure
        self.legend = legend

    # Implement methods and properties to allow legacy interface (np.array)
    __iter__ = lambda self: self.lines
    __len__ = lambda self: len(self.lines)
    def __getitem__(self, item):
        warnings.warn(
            "return of Line2D objects from plot function is deprecated in "
            "favor of ControlPlot; use out.lines to access Line2D objects",
            category=FutureWarning)
        return self.lines[item]
    def __setitem__(self, item, val):
        self.lines[item] = val
    shape = property(lambda self: self.lines.shape, None)
    def reshape(self, *args):
        """Reshape lines array (legacy)."""
        return self.lines.reshape(*args)

    def set_plot_title(self, title, frame='axes'):
        """Set the title for a control plot.

        This is a wrapper for the matplotlib `suptitle` function, but by
        setting `frame` to 'axes' (default) then the title is centered on
        the midpoint of the axes in the figure, rather than the center of
        the figure.  This usually looks better (particularly with
        multi-panel plots), though it takes longer to render.

        Parameters
        ----------
        title : str
            Title text.
        fig : Figure, optional
            Matplotlib figure.  Defaults to current figure.
        frame : str, optional
            Coordinate frame for centering: 'axes' (default) or 'figure'.
        **kwargs : `matplotlib.pyplot.suptitle` keywords, optional
            Additional keywords (passed to matplotlib).

        """
        _update_plot_title(
            title, fig=self.figure, frame=frame, use_existing=False)

#
# User functions
#
# The functions below can be used by users to modify control plots or get
# information about them.
#

def suptitle(
        title, fig=None, frame='axes', **kwargs):
    """Add a centered title to a figure.

    .. deprecated:: 0.10.1
        Use `ControlPlot.set_plot_title`.

    """
    warnings.warn(
        "suptitle() is deprecated; use cplt.set_plot_title()", FutureWarning)
    _update_plot_title(
        title, fig=fig, frame=frame, use_existing=False, **kwargs)


# Create vectorized function to find axes from lines
def get_plot_axes(line_array):
    """Get a list of axes from an array of lines.

    .. deprecated:: 0.10.1
        This function will be removed in a future version of python-control.
        Use `cplt.axes` to obtain axes for an instance of `ControlPlot`.

    This function can be used to return the set of axes corresponding
    to the line array that is returned by `time_response_plot`.  This
    is useful for generating an axes array that can be passed to
    subsequent plotting calls.

    Parameters
    ----------
    line_array : array of list of `matplotlib.lines.Line2D`
        A 2D array with elements corresponding to a list of lines appearing
        in an axes, matching the return type of a time response data plot.

    Returns
    -------
    axes_array : array of list of `matplotlib.axes.Axes`
        A 2D array with elements corresponding to the Axes associated with
        the lines in `line_array`.

    Notes
    -----
    Only the first element of each array entry is used to determine the axes.

    """
    warnings.warn(
        "get_plot_axes() is deprecated; use cplt.axes()", FutureWarning)
    _get_axes = np.vectorize(lambda lines: lines[0].axes)
    if isinstance(line_array, ControlPlot):
        return _get_axes(line_array.lines)
    else:
        return _get_axes(line_array)


def pole_zero_subplots(
        nrows, ncols, grid=None, dt=None, fig=None, scaling=None,
        rcParams=None):
    """Create axes for pole/zero plot.

    Parameters
    ----------
    nrows, ncols : int
        Number of rows and columns.
    grid : True, False, or 'empty', optional
        Grid style to use.  Can also be a list, in which case each subplot
        will have a different style (columns then rows).
    dt : timebase, option
        Timebase for each subplot (or a list of timebases).
    scaling : 'auto', 'equal', or None
        Scaling to apply to the subplots.
    fig : `matplotlib.figure.Figure`
        Figure to use for creating subplots.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.

    Returns
    -------
    ax_array : ndarray
        2D array of axes.

    """
    from .grid import nogrid, sgrid, zgrid
    from .iosys import isctime

    if fig is None:
        fig = plt.gcf()
    rcParams = config._get_param('ctrlplot', 'rcParams', rcParams)

    if not isinstance(grid, list):
        grid = [grid] * nrows * ncols
    if not isinstance(dt, list):
        dt = [dt] * nrows * ncols

    ax_array = np.full((nrows, ncols), None)
    index = 0
    with plt.rc_context(rcParams):
        for row, col in itertools.product(range(nrows), range(ncols)):
            match grid[index], isctime(dt=dt[index]):
                case 'empty', _:        # empty grid
                    ax_array[row, col] = fig.add_subplot(nrows, ncols, index+1)

                case True, True:        # continuous-time grid
                    ax_array[row, col], _ = sgrid(
                        (nrows, ncols, index+1), scaling=scaling)

                case True, False:       # discrete-time grid
                    ax_array[row, col] = fig.add_subplot(nrows, ncols, index+1)
                    zgrid(ax=ax_array[row, col], scaling=scaling)

                case False | None, _:   # no grid (just stability boundaries)
                    ax_array[row, col] = fig.add_subplot(nrows, ncols, index+1)
                    nogrid(
                        ax=ax_array[row, col], dt=dt[index], scaling=scaling)
            index += 1
    return ax_array


def reset_rcParams():
    """Reset rcParams to default values for control plots."""
    _ctrlplot_rcParams.update(rcParams_default)


#
# Utility functions
#
# These functions are used by plotting routines to provide a consistent way
# of processing and displaying information.
#

def _process_ax_keyword(
        axs, shape=(1, 1), rcParams=None, squeeze=False, clear_text=False,
        create_axes=True, sharex=False, sharey=False):
    """Process ax keyword to plotting commands.

    This function processes the `ax` keyword to plotting commands.  If no
    ax keyword is passed, the current figure is checked to see if it has
    the correct shape.  If the shape matches the desired shape, then the
    current figure and axes are returned.  Otherwise a new figure is
    created with axes of the desired shape.

    If `create_axes` is False and a new/empty figure is returned, then `axs`
    is an array of the proper shape but None for each element.  This allows
    the calling function to do the actual axis creation (needed for
    curvilinear grids that use the AxisArtist module).

    Legacy behavior: some of the older plotting commands use an axes label
    to identify the proper axes for plotting.  This behavior is supported
    through the use of the label keyword, but will only work if shape ==
    (1, 1) and squeeze == True.

    """
    if axs is None:
        fig = plt.gcf()         # get current figure (or create new one)
        axs = fig.get_axes()

        # Check to see if axes are the right shape; if not, create new figure
        # Note: can't actually check the shape, just the total number of axes
        if len(axs) != np.prod(shape):
            with plt.rc_context(rcParams):
                if len(axs) != 0 and create_axes:
                    # Create a new figure
                    fig, axs = plt.subplots(
                        *shape, sharex=sharex, sharey=sharey, squeeze=False)
                elif create_axes:
                    # Create new axes on (empty) figure
                    axs = fig.subplots(
                        *shape, sharex=sharex, sharey=sharey, squeeze=False)
                else:
                    # Create an empty array and let user create axes
                    axs = np.full(shape, None)
            if create_axes:     # if not creating axes, leave these to caller
                fig.set_layout_engine('tight')
                fig.align_labels()

        else:
            # Use the existing axes, properly reshaped
            axs = np.asarray(axs).reshape(*shape)

            if clear_text:
                # Clear out any old text from the current figure
                for text in fig.texts:
                    text.set_visible(False)     # turn off the text
                    del text                    # get rid of it completely
    else:
        axs = np.atleast_1d(axs)
        try:
            axs = axs.reshape(shape)
        except ValueError:
            raise ValueError(
                "specified axes are not the right shape; "
                f"got {axs.shape} but expecting {shape}")
        fig = axs[0, 0].figure

    # Process the squeeze keyword
    if squeeze and shape == (1, 1):
        axs = axs[0, 0]         # Just return the single axes object
    elif squeeze:
        axs = axs.squeeze()

    return fig, axs


# Turn label keyword into array indexed by trace, output, input
# TODO: move to ctrlutil.py and update parameter names to reflect general use
def _process_line_labels(label, ntraces=1, ninputs=0, noutputs=0):
    if label is None:
        return None

    if isinstance(label, str):
        label = [label] * ntraces          # single label for all traces

    # Convert to an ndarray, if not done already
    try:
        line_labels = np.asarray(label)
    except ValueError:
        raise ValueError("label must be a string or array_like")

    # Turn the data into a 3D array of appropriate shape
    # TODO: allow more sophisticated broadcasting (and error checking)
    try:
        if ninputs > 0 and noutputs > 0:
            if line_labels.ndim == 1 and line_labels.size == ntraces:
                line_labels = line_labels.reshape(ntraces, 1, 1)
                line_labels = np.broadcast_to(
                    line_labels, (ntraces, ninputs, noutputs))
            else:
                line_labels = line_labels.reshape(ntraces, ninputs, noutputs)
    except ValueError:
        if line_labels.shape[0] != ntraces:
            raise ValueError("number of labels must match number of traces")
        else:
            raise ValueError("labels must be given for each input/output pair")

    return line_labels


# Get labels for all lines in an axes
def _get_line_labels(ax, use_color=True):
    labels_colors, lines = [], []
    last_color, counter = None, 0       # label unknown systems
    for i, line in enumerate(ax.get_lines()):
        label = line.get_label()
        color = line.get_color()
        if use_color and label.startswith("Unknown"):
            label = f"Unknown-{counter}"
            if last_color != color:
                counter += 1
            last_color = color
        elif label[0] == '_':
            continue

        if (label, color) not in labels_colors:
            lines.append(line)
            labels_colors.append((label, color))

    return lines, [label for label, color in labels_colors]


def _process_legend_keywords(
        kwargs, shape=None, default_loc='center right'):
    legend_loc = kwargs.pop('legend_loc', None)
    if shape is None and 'legend_map' in kwargs:
        raise TypeError("unexpected keyword argument 'legend_map'")
    else:
        legend_map = kwargs.pop('legend_map', None)
    show_legend = kwargs.pop('show_legend', None)

    # If legend_loc or legend_map were given, always show the legend
    if legend_loc is False or legend_map is False:
        if show_legend is True:
            warnings.warn(
                "show_legend ignored; legend_loc or legend_map was given")
        show_legend = False
        legend_loc = legend_map = None
    elif legend_loc is not None or legend_map is not None:
        if show_legend is False:
            warnings.warn(
                "show_legend ignored; legend_loc or legend_map was given")
        show_legend = True

    if legend_loc is None:
        legend_loc = default_loc
    elif not isinstance(legend_loc, (int, str)):
        raise ValueError("legend_loc must be string or int")

    # Make sure the legend map is the right size
    if legend_map is not None:
        legend_map = np.atleast_2d(legend_map)
        if legend_map.shape != shape:
            raise ValueError("legend_map shape just match axes shape")

    return legend_loc, legend_map, show_legend


# Utility function to make legend labels
def _make_legend_labels(labels, ignore_common=False):
    if len(labels) == 1:
        return labels

    # Look for a common prefix (up to a space)
    common_prefix = commonprefix(labels)
    last_space = common_prefix.rfind(', ')
    if last_space < 0 or ignore_common:
        common_prefix = ''
    elif last_space > 0:
        common_prefix = common_prefix[:last_space + 2]
    prefix_len = len(common_prefix)

    # Look for a common suffix (up to a space)
    common_suffix = commonprefix(
        [label[::-1] for label in labels])[::-1]
    suffix_len = len(common_suffix)
    # Only chop things off after a comma or space
    while suffix_len > 0 and common_suffix[-suffix_len] != ',':
        suffix_len -= 1

    # Strip the labels of common information
    if suffix_len > 0 and not ignore_common:
        labels = [label[prefix_len:-suffix_len] for label in labels]
    else:
        labels = [label[prefix_len:] for label in labels]

    return labels


def _update_plot_title(
        title, fig=None, frame='axes', use_existing=True, **kwargs):
    if title is False or title is None:
        return
    if fig is None:
        fig = plt.gcf()
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    if use_existing:
        # Get the current title, if it exists
        old_title = None if fig._suptitle is None else fig._suptitle._text

        if old_title is not None:
            # Find the common part of the titles
            common_prefix = commonprefix([old_title, title])

            # Back up to the last space
            last_space = common_prefix.rfind(' ')
            if last_space > 0:
                common_prefix = common_prefix[:last_space]
            common_len = len(common_prefix)

            # Add the new part of the title (usually the system name)
            if old_title[common_len:] != title[common_len:]:
                separator = ',' if len(common_prefix) > 0 else ';'
                title = old_title + separator + title[common_len:]

    if frame == 'figure':
        with plt.rc_context(rcParams):
            fig.suptitle(title, **kwargs)

    elif frame == 'axes':
        with plt.rc_context(rcParams):
            fig.suptitle(title, **kwargs)           # Place title in center
            plt.tight_layout()                      # Put everything into place
            xc, _ = _find_axes_center(fig, fig.get_axes())
            fig.suptitle(title, x=xc, **kwargs)     # Redraw title, centered

    else:
        raise ValueError(f"unknown frame '{frame}'")


def _find_axes_center(fig, axs):
    """Find the midpoint between axes in display coordinates.

    This function finds the middle of a plot as defined by a set of axes.

    """
    inv_transform = fig.transFigure.inverted()
    xlim = ylim = [1, 0]
    for ax in axs:
        ll = inv_transform.transform(ax.transAxes.transform((0, 0)))
        ur = inv_transform.transform(ax.transAxes.transform((1, 1)))

        xlim = [min(ll[0], xlim[0]), max(ur[0], xlim[1])]
        ylim = [min(ll[1], ylim[0]), max(ur[1], ylim[1])]

    return (np.sum(xlim)/2, np.sum(ylim)/2)


# Internal function to add arrows to a curve
def _add_arrows_to_line2D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', arrowsize=1, dir=1):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters
    ----------
    axes: Axes object as returned by axes command (or gca)
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow

    Returns
    -------
    arrows : list of arrows

    Notes
    -----
    Based on https://stackoverflow.com/questions/26911898/

    """
    # Get the coordinates of the line, in plot coordinates
    if not isinstance(line, mpl.lines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    # Determine the arrow properties
    arrow_kw = {"arrowstyle": arrowstyle}

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multi-color lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multi-width lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    # Figure out the size of the axes (length of diagonal)
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    ul, lr = np.array([xlim[0], ylim[0]]), np.array([xlim[1], ylim[1]])
    diag = np.linalg.norm(ul - lr)

    # Compute the arc length along the curve
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Truncate the number of arrows if the curve is short
    # TODO: figure out a smarter way to do this
    frac = min(s[-1] / diag, 1)
    if len(arrow_locs) and frac < 0.05:
        arrow_locs = []         # too short; no arrows at all
    elif len(arrow_locs) and frac < 0.2:
        arrow_locs = [0.5]      # single arrow in the middle

    # Plot the arrows (and return list if patches)
    arrows = []
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)

        if dir == 1 and n == 0:
            # Move the arrow forward by one if it is at start of a segment
            n = 1

        # Place the head of the arrow at the desired location
        arrow_head = [x[n], y[n]]
        arrow_tail = [x[n - dir], y[n - dir]]

        p = mpl.patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=axes.transData, lw=0,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


def _get_color_offset(ax, color_cycle=None):
    """Get color offset based on current lines.

    This function determines that the current offset is for the next color
    to use based on current colors in a plot.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes containing already plotted lines.
    color_cycle : list of matplotlib color specs, optional
        Colors to use in plotting lines.  Defaults to matplotlib rcParams
        color cycle.

    Returns
    -------
    color_offset : matplotlib color spec
        Starting color for next line to be drawn.
    color_cycle : list of matplotlib color specs
        Color cycle used to determine colors.

    """
    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    color_offset = 0
    if len(ax.lines) > 0:
        last_color = ax.lines[-1].get_color()
        if last_color in color_cycle:
            color_offset = color_cycle.index(last_color) + 1

    return color_offset % len(color_cycle), color_cycle


def _get_color(
        colorspec, offset=None, fmt=None, ax=None, lines=None,
        color_cycle=None):
    """Get color to use for plotting line.

    This function returns the color to be used for the line to be drawn (or
    None if the default color cycle for the axes should be used).

    Parameters
    ----------
    colorspec : matplotlib color specification
        User-specified color (or None).
    offset : int, optional
        Offset into the color cycle (for multi-trace plots).
    fmt : str, optional
        Format string passed to plotting command.
    ax : `matplotlib.axes.Axes`, optional
        Axes containing already plotted lines.
    lines : list of matplotlib.lines.Line2D, optional
        List of plotted lines.  If not given, use ax.get_lines().
    color_cycle : list of matplotlib color specs, optional
        Colors to use in plotting lines.  Defaults to matplotlib rcParams
        color cycle.

    Returns
    -------
    color : matplotlib color spec
        Color to use for this line (or None for matplotlib default).

    """
    # See if the color was explicitly specified by the user
    if isinstance(colorspec, dict):
        if 'color' in colorspec:
            return colorspec.pop('color')
    elif fmt is not None and \
         [isinstance(arg, str) and
          any([c in arg for c in "bgrcmykw#"]) for arg in fmt]:
        return None             # *fmt will set the color
    elif colorspec != None:
        return colorspec

    # Figure out what color cycle to use, if not given by caller
    if color_cycle == None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Find the lines that we should pay attention to
    if lines is None and ax is not None:
        lines = ax.lines

    # If we were passed a set of lines, try to increment color from previous
    if offset is not None:
        return color_cycle[offset]
    elif lines is not None:
        color_offset = 0
        if len(ax.lines) > 0:
            last_color = ax.lines[-1].get_color()
            if last_color in color_cycle:
                color_offset = color_cycle.index(last_color) + 1
        color_offset = color_offset % len(color_cycle)
        return color_cycle[color_offset]
    else:
        return None
