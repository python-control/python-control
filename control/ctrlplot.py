# ctrlplot.py - utility functions for plotting
# Richard M. Murray, 14 Jun 2024
#
# Collection of functions that are used by various plotting functions.

import warnings
from os.path import commonprefix

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

__all__ = ['ControlPlot', 'suptitle', 'get_plot_axes']

#
# Style parameters
#

_ctrlplot_rcParams = mpl.rcParams.copy()
_ctrlplot_rcParams.update({
    'axes.labelsize': 'small',
    'axes.titlesize': 'small',
    'figure.titlesize': 'medium',
    'legend.fontsize': 'x-small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
})


#
# Control figure
#

class ControlPlot(object):
    """A class for returning control figures.

    This class is used as the return type for control plotting functions.
    It contains the information required to access portions of the plot
    that the user might want to adjust, as well as providing methods to
    modify some of the properties of the plot.

    A control figure consists of a :class:`matplotlib.figure.Figure` with
    an array of :class:`matplotlib.axes.Axes`.  Each axes in the figure has
    a number of lines that represent the data for the plot.  There may also
    be a legend present in one or more of the axes.

    Attributes
    ----------
    lines : array of list of :class:`matplotlib:Line2D`
        Array of Line2D objects for each line in the plot.  Generally, the
        shape of the array matches the subplots shape and the value of the
        array is a list of Line2D objects in that subplot.  Some plotting
        functions will return variants of this structure, as described in
        the individual documentation for the functions.
    axes : 2D array of :class:`matplotlib:Axes`
        Array of Axes objects for each subplot in the plot.
    figure : :class:`matplotlib:Figure`
        Figure on which the Axes are drawn.
    legend : :class:`matplotlib:.legend.Legend` (instance or ndarray)
        Legend object(s) for the plat.  If more than one legend is
        included, this will be an array with each entry being either None
        (for no legend) or a legend object.

    """
    def __init__(self, lines, axes=None, figure=None, legend=None):
        self.lines = lines
        if axes is None:
            axes = get_plot_axes(lines)
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
        return self.lines.reshape(*args)


#
# User functions
#
# The functions below can be used by users to modify control plots or get
# information about them.
#


def suptitle(
        title, fig=None, frame='axes', **kwargs):
    """Add a centered title to a figure.

    This is a wrapper for the matplotlib `suptitle` function, but by
    setting ``frame`` to 'axes' (default) then the title is centered on the
    midpoint of the axes in the figure, rather than the center of the
    figure.  This usually looks better (particularly with multi-panel
    plots), though it takes longer to render.

    Parameters
    ----------
    title : str
        Title text.
    fig : Figure, optional
        Matplotlib figure.  Defaults to current figure.
    frame : str, optional
        Coordinate frame to use for centering: 'axes' (default) or 'figure'.
    **kwargs : :func:`matplotlib.pyplot.suptitle` keywords, optional
        Additional keywords (passed to matplotlib).

    """
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    if fig is None:
        fig = plt.gcf()

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


# Create vectorized function to find axes from lines
def get_plot_axes(line_array):
    """Get a list of axes from an array of lines.

    This function can be used to return the set of axes corresponding
    to the line array that is returned by `time_response_plot`.  This
    is useful for generating an axes array that can be passed to
    subsequent plotting calls.

    Parameters
    ----------
    line_array : array of list of Line2D
        A 2D array with elements corresponding to a list of lines appearing
        in an axes, matching the return type of a time response data plot.

    Returns
    -------
    axes_array : array of list of Axes
        A 2D array with elements corresponding to the Axes associated with
        the lines in `line_array`.

    Notes
    -----
    Only the first element of each array entry is used to determine the axes.

    """
    _get_axes = np.vectorize(lambda lines: lines[0].axes)
    if isinstance(line_array, ControlPlot):
        return _get_axes(line_array.lines)
    else:
        return _get_axes(line_array)

#
# Utility functions
#
# These functions are used by plotting routines to provide a consistent way
# of processing and displaying information.
#


def _process_ax_keyword(
        axs, shape=(1, 1), rcParams=None, squeeze=False, clear_text=False):
    """Utility function to process ax keyword to plotting commands.

    This function processes the `ax` keyword to plotting commands.  If no
    ax keyword is passed, the current figure is checked to see if it has
    the correct shape.  If the shape matches the desired shape, then the
    current figure and axes are returned.  Otherwise a new figure is
    created with axes of the desired shape.

    Legacy behavior: some of the older plotting commands use a axes label
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
                if len(axs) != 0:
                    # Create a new figure
                    fig, axs = plt.subplots(*shape, squeeze=False)
                else:
                    # Create new axes on (empty) figure
                    axs = fig.subplots(*shape, squeeze=False)
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
        try:
            axs = np.asarray(axs).reshape(shape)
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
def _process_line_labels(label, ntraces, ninputs=0, noutputs=0):
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


# Utility function to make legend labels
def _make_legend_labels(labels, ignore_common=False):

    # Look for a common prefix (up to a space)
    common_prefix = commonprefix(labels)
    last_space = common_prefix.rfind(', ')
    if last_space < 0 or ignore_common:
        common_prefix = ''
    elif last_space > 0:
        common_prefix = common_prefix[:last_space]
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


def _update_suptitle(fig, title, rcParams=None, frame='axes'):
    if fig is not None and isinstance(title, str):
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

        # Add the title
        suptitle(title, fig=fig, rcParams=rcParams, frame=frame)


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

    Parameters:
    -----------
    axes: Axes object as returned by axes command (or gca)
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow

    Returns:
    --------
    arrows: list of arrows

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
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
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
