"""ss_plot.py

Heat map plots of a state space or a matrix.

"""

import numpy as np
import matplotlib.pylab as plt
from . import StateSpace

__all__ = ['ss_plot', 'matrix_plot']


def ss_plot(sys, separated_figures=True, abs_values=True):
    """State space plot for a system

    Plots the state space plot for the system given as input.

    Parameters
    ----------
    sys : LTI
        Linear input/output systems (single system is OK)
    separated_figures : boolean, optional
        True if each state space matrix has its own figure
    abs_values : boolean, optional
        True if the plot should show absolute values of the matrices in cells.
        False if the plot should show relative [-100..100] values of the matrices in cells.

    Returns
    -------
    None
    """

    if not isinstance(sys, StateSpace):
        raise TypeError('ss_plot expects State Space object as input.')

    if separated_figures is True:
        plt.figure()
        matrix_plot(sys.A, 'A', abs_values=abs_values)
        plt.figure()
        matrix_plot(sys.B, 'B', abs_values=abs_values)
        plt.figure()
        matrix_plot(sys.C, 'C', abs_values=abs_values)
        plt.figure()
        matrix_plot(sys.D, 'D', abs_values=abs_values)
    else:
        plt.figure()
        plt.subplot(2, 2, 1)
        matrix_plot(sys.A, 'A', abs_values=abs_values)
        plt.subplot(2, 2, 2)
        matrix_plot(sys.B, 'B', abs_values=abs_values)
        plt.subplot(2, 2, 3)
        matrix_plot(sys.C, 'C', abs_values=abs_values)
        plt.subplot(2, 2, 4)
        matrix_plot(sys.D, 'D', abs_values=abs_values)


def matrix_plot(matrix, name='matrix', abs_values=True):
    """ Heat map plot of a matrix

    Parameters
    ----------
    matrix : numpy array-like matrix
    name : string
        Matrix name used to give a title to the figure
    abs_values : boolean, optional
        True if the plot should show absolute values of the matrices in cells.
        False if the plot should show relative [-100..100] values of the matrices in cells.

    Returns
    -------
    None
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix_plot expects numpy.ndarray object as input.')

    # Scale matrix in case relative values are asked
    max_abs_value = np.max(np.abs(matrix))
    if abs_values is False:
        if max_abs_value != 0.:
            matrix = matrix / max_abs_value * 100.
            max_abs_value = 100.

    # Get the current axis
    fig = plt.gcf()
    ax = plt.gca()
    ax.clear()

    # Determine font size through minimum cell size
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    min_cell_size = min(height / matrix.shape[0], width / matrix.shape[1])
    font_size = min_cell_size / 3

    ax.set_title(name + ' [' + str(matrix.shape[0]) + 'x' + str(matrix.shape[1]) + ']',
                 size=font_size)

    # Set orthonormal view
    ax.set_aspect('equal')

    # Show A matrix interpolating color as a function of cell value, while keeping 0 value as gray
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.coolwarm,
               vmin=-max_abs_value, vmax=max_abs_value)
    color_bar = plt.colorbar(ax=ax)
    color_bar.solids.set_edgecolor('face')
    color_bar.ax.tick_params(labelsize=font_size)

    # Separate cells with white line
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)

    # Show numerated axis
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.xaxis.set_tick_params(labelsize=font_size)
    ax.yaxis.set_tick_params(labelsize=font_size)

    # Show cell values of the matrix
    for (i, j), z in np.ndenumerate(matrix):
        if z != -0.:
            ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', size=font_size,
                    color='black')
        else:
            ax.text(j, i, '.', ha='center', va='center', size=font_size,
                    color='black')
