# ctrlplot.py - utility functions for plotting
# Richard M. Murray, 14 Jun 2024
#
# Collection of functions that are used by various plotting functions.

from os.path import commonprefix

import matplotlib.pyplot as plt
import numpy as np

from . import config

__all__ = ['suptitle', 'get_plot_axes']


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
    rcParams = config._get_param('freqplot', 'rcParams', kwargs, pop=True)

    if fig is None:
        fig = plt.gcf()

    if frame == 'figure':
        with plt.rc_context(rcParams):
            fig.suptitle(title, **kwargs)

    elif frame == 'axes':
        # TODO: move common plotting params to 'ctrlplot'
        rcParams = config._get_param('freqplot', 'rcParams', rcParams)
        with plt.rc_context(rcParams):
            plt.tight_layout()          # Put the figure into proper layout
            xc, _ = _find_axes_center(fig, fig.get_axes())

            fig.suptitle(title, x=xc, **kwargs)
            plt.tight_layout()          # Update the layout

    else:
        raise ValueError(f"unknown frame '{frame}'")


# Create vectorized function to find axes from lines
def get_plot_axes(line_array):
    """Get a list of axes from an array of lines.

    This function can be used to return the set of axes corresponding to
    the line array that is returned by `time_response_plot`.  This is useful for
    generating an axes array that can be passed to subsequent plotting
    calls.

    Parameters
    ----------
    line_array : array of list of Line2D
        A 2D array with elements corresponding to a list of lines appearing
        in an axes, matching the return type of a time response data plot.

    Returns
    -------
    axes_array : array of list of Axes
        A 2D array with elements corresponding to the Axes assocated with
        the lines in `line_array`.

    Notes
    -----
    Only the first element of each array entry is used to determine the axes.

    """
    _get_axes = np.vectorize(lambda lines: lines[0].axes)
    return _get_axes(line_array)

#
# Utility functions
#


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

    # Look for a common suffice (up to a space)
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
