# plotutil.py - utility functions for plotting
# Richard M. Murray, 14 Jun 2024
#
# Collection of functions that are used by various plotting functions.

import matplotlib.pyplot as plt
import numpy as np

from . import config

__all__ = ['suptitle']


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
