# nichols.py - Nichols plot
#
# Initial author: Allan McInnes <Allan.McInnes@canterbury.ac.nz>

"""Functions for plotting Black-Nichols charts."""

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

from . import config
from .ctrlplot import ControlPlot, _get_line_labels, _process_ax_keyword, \
    _process_legend_keywords, _process_line_labels, _update_plot_title
from .ctrlutil import unwrap
from .lti import frequency_response
from .statesp import StateSpace
from .xferfcn import TransferFunction

__all__ = ['nichols_plot', 'nichols', 'nichols_grid']

# Default parameters values for the nichols module
_nichols_defaults = {
    'nichols.grid': True,
}


def nichols_plot(
        data, omega=None, *fmt, grid=None, title=None, ax=None,
        label=None, **kwargs):
    """Nichols plot for a system.

    Plots a Nichols plot for the system over a (optional) frequency range.

    Parameters
    ----------
    data : list of `FrequencyResponseData` or `LTI`
        List of LTI systems or `FrequencyResponseData` objects.  A
        single system or frequency response can also be passed.
    omega : array_like
        Range of frequencies (list or bounds) in rad/sec.
    *fmt : `matplotlib.pyplot.plot` format string, optional
        Passed to `matplotlib` as the format string for all lines in the plot.
        The `omega` parameter must be present (use omega=None if needed).
    grid : boolean, optional
        True if the plot should include a Nichols-chart grid. Default is
        True and can be set using `config.defaults['nichols.grid']`.
    **kwargs : `matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords passed to `matplotlib` to specify line properties.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : Array of `matplotlib.lines.Line2D` objects
        Array containing information on each line in the plot.  The shape
        of the array matches the subplots shape and the value of the array
        is a list of Line2D objects in that subplot.
    cplt.axes : 2D ndarray of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.
    cplt.legend : 2D array of `matplotlib.legend.Legend`
        Legend object(s) contained in the plot.

    Other Parameters
    ----------------
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified and
        the current figure has a single axes, that axes is used.
        Otherwise, a new figure is created.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with given
        label(s).  If sysdata is a list, strings should be specified for each
        system.
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'upper left',
        with no legend for a single response.  Use False to suppress legend.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on the
        plot or `legend_loc` has been specified.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).

    """
    # Get parameter values
    grid = config._get_param('nichols', 'grid', grid, True)
    label = _process_line_labels(label)
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)

    # If argument was a singleton, turn it into a list
    if not isinstance(data, (tuple, list)):
        data = [data]

    # If we were passed a list of systems, convert to data
    if all([isinstance(
            sys, (StateSpace, TransferFunction)) for sys in data]):
        data = frequency_response(data, omega=omega)

    # Make sure that all systems are SISO
    if any([resp.ninputs > 1 or resp.noutputs > 1 for resp in data]):
        raise NotImplementedError("MIMO Nichols plots not implemented")

    fig, ax_nichols = _process_ax_keyword(ax, rcParams=rcParams, squeeze=True)
    legend_loc, _, show_legend = _process_legend_keywords(
        kwargs, None, 'upper left')

    # Create a list of lines for the output
    out = np.empty(len(data), dtype=object)

    for idx, response in enumerate(data):
        # Get the magnitude and phase of the system
        mag = np.squeeze(response.magnitude)
        phase = np.squeeze(response.phase)
        omega = response.omega

        # Convert to Nichols-plot format (phase in degrees,
        # and magnitude in dB)
        x = unwrap(np.degrees(phase), 360)
        y = 20*np.log10(mag)

        # Decide on the system name and label
        sysname = response.sysname if response.sysname is not None \
            else f"Unknown-sys_{idx}"
        label_ = sysname if label is None else label[idx]

        # Generate the plot
        out[idx] = ax_nichols.plot(x, y, *fmt, label=label_, **kwargs)

    # Label the plot axes
    ax_nichols.set_xlabel('Phase [deg]')
    ax_nichols.set_ylabel('Magnitude [dB]')

    # Mark the -180 point
    ax_nichols.plot([-180], [0], 'r+')

    # Add grid
    if grid:
        nichols_grid(ax=ax_nichols)

    # List of systems that are included in this plot
    lines, labels = _get_line_labels(ax_nichols)

    # Add legend if there is more than one system plotted
    if show_legend == True or (show_legend != False and len(labels) > 1):
        with plt.rc_context(rcParams):
            legend = ax_nichols.legend(lines, labels, loc=legend_loc)
    else:
        legend = None

    # Add the title
    if ax is None:
        if title is None:
            title = "Nichols plot for " + ", ".join(labels)
        _update_plot_title(
            title, fig=fig, rcParams=rcParams, use_existing=False)

    return ControlPlot(out, ax_nichols, fig, legend=legend)


def _inner_extents(ax):
    # intersection of data and view extents
    # if intersection empty, return view extents
    _inner = matplotlib.transforms.Bbox.intersection(ax.viewLim, ax.dataLim)
    if _inner is None:
        return ax.ViewLim.extents
    else:
        return _inner.extents


def nichols_grid(cl_mags=None, cl_phases=None, line_style='dotted', ax=None,
                 label_cl_phases=True):
    """Plot Nichols chart grid.

    Plots a Nichols chart grid on the current axes, or creates a new chart
    if no plot already exists.

    Parameters
    ----------
    cl_mags : array_like (dB), optional
        Array of closed-loop magnitudes defining the iso-gain lines on a
        custom Nichols chart.
    cl_phases : array_like (degrees), optional
        Array of closed-loop phases defining the iso-phase lines on a custom
        Nichols chart. Must be in the range -360 < cl_phases < 0
    line_style : string, optional
        :doc:`Matplotlib linestyle \
            <matplotlib:gallery/lines_bars_and_markers/linestyles>`.
    ax : `matplotlib.axes.Axes`, optional
        Axes to add grid to.  If None, use `matplotlib.pyplot.gca`.
    label_cl_phases : bool, optional
        If True, closed-loop phase lines will be labeled.

    Returns
    -------
    cl_mag_lines : list of `matplotlib.line.Line2D`
      The constant closed-loop gain contours.
    cl_phase_lines : list of `matplotlib.line.Line2D`
      The constant closed-loop phase contours.
    cl_mag_labels : list of `matplotlib.text.Text`
      Magnitude contour labels; each entry corresponds to the respective
      entry in `cl_mag_lines`.
    cl_phase_labels : list of `matplotlib.text.Text`
      Phase contour labels; each entry corresponds to the respective entry
      in `cl_phase_lines`.

    """
    if ax is None:
        ax = plt.gca()

    # Default chart size
    ol_phase_min = -359.99
    ol_phase_max = 0.0
    ol_mag_min = -40.0
    ol_mag_max = default_ol_mag_max = 50.0

    if ax.has_data():
        # Find extent of intersection the current dataset or view
        ol_phase_min, ol_mag_min, ol_phase_max, ol_mag_max = _inner_extents(ax)

    # M-circle magnitudes.
    if cl_mags is None:
        # Default chart magnitudes
        # The key set of magnitudes are always generated, since this
        # guarantees a recognizable Nichols chart grid.
        key_cl_mags = np.array([-40.0, -20.0, -12.0, -6.0, -3.0, -1.0, -0.5,
                                0.0, 0.25, 0.5, 1.0, 3.0, 6.0, 12.0])

        # Extend the range of magnitudes if necessary. The extended arange
        # will end up empty if no extension is required. Assumes that
        # closed-loop magnitudes are approximately aligned with open-loop
        # magnitudes beyond the value of np.min(key_cl_mags)
        cl_mag_step = -20.0  # dB
        extended_cl_mags = np.arange(np.min(key_cl_mags),
                                     ol_mag_min + cl_mag_step, cl_mag_step)
        cl_mags = np.concatenate((extended_cl_mags, key_cl_mags))

    # a minimum 360deg extent containing the phases
    phase_round_max = 360.0*np.ceil(ol_phase_max/360.0)
    phase_round_min = min(phase_round_max-360,
                          360.0*np.floor(ol_phase_min/360.0))

    # N-circle phases (should be in the range -360 to 0)
    if cl_phases is None:
        # aim for 9 lines, but always show (-360+eps, -180, -eps)
        # smallest spacing is 45, biggest is 180
        phase_span = phase_round_max - phase_round_min
        spacing = np.clip(round(phase_span / 8 / 45) * 45, 45, 180)
        key_cl_phases = np.array([-0.25, -359.75])
        other_cl_phases = np.arange(-spacing, -360.0, -spacing)
        cl_phases = np.unique(np.concatenate((key_cl_phases, other_cl_phases)))
    elif not ((-360 < np.min(cl_phases)) and (np.max(cl_phases) < 0.0)):
        raise ValueError('cl_phases must between -360 and 0, exclusive')

    # Find the M-contours
    m = m_circles(cl_mags, phase_min=np.min(cl_phases),
                  phase_max=np.max(cl_phases))
    m_mag = 20*np.log10(np.abs(m))
    m_phase = np.mod(np.degrees(np.angle(m)), -360.0)  # Unwrap

    # Find the N-contours
    n = n_circles(cl_phases, mag_min=np.min(cl_mags), mag_max=np.max(cl_mags))
    n_mag = 20*np.log10(np.abs(n))
    n_phase = np.mod(np.degrees(np.angle(n)), -360.0)  # Unwrap

    # Plot the contours behind other plot elements.
    # The "phase offset" is used to produce copies of the chart that cover
    # the entire range of the plotted data, starting from a base chart computed
    # over the range -360 < phase < 0. Given the range
    # the base chart is computed over, the phase offset should be 0
    # for -360 < ol_phase_min < 0.
    phase_offsets = 360 + np.arange(phase_round_min, phase_round_max, 360.0)

    cl_mag_lines = []
    cl_phase_lines = []
    cl_mag_labels = []
    cl_phase_labels = []

    for phase_offset in phase_offsets:
        # Draw M and N contours
        cl_mag_lines.extend(
            ax.plot(m_phase + phase_offset, m_mag, color='lightgray',
                    linestyle=line_style, zorder=0))
        cl_phase_lines.extend(
            ax.plot(n_phase + phase_offset, n_mag, color='lightgray',
                    linestyle=line_style, zorder=0))

        # Add magnitude labels
        for x, y, m in zip(m_phase[:][-1] + phase_offset, m_mag[:][-1],
                           cl_mags):
            align = 'right' if m < 0.0 else 'left'
            cl_mag_labels.append(
                ax.text(x, y, str(m) + ' dB', size='small', ha=align,
                        color='gray', clip_on=True))

        # phase labels
        if label_cl_phases:
            for x, y, p in zip(n_phase[:][0] + phase_offset,
                               n_mag[:][0],
                               cl_phases):
                if p > -175:
                    align = 'right'
                elif p > -185:
                    align = 'center'
                else:
                    align = 'left'
                cl_phase_labels.append(
                    ax.text(x, y, f'{round(p)}\N{DEGREE SIGN}',
                            size='small',
                            ha=align,
                            va='bottom',
                            color='gray',
                            clip_on=True))


    # Fit axes to generated chart
    ax.axis([phase_round_min,
             phase_round_max,
             np.min(np.concatenate([cl_mags,[ol_mag_min]])),
             np.max([ol_mag_max, default_ol_mag_max])])

    return cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating Nichols plots
#


def closed_loop_contours(Gcl_mags, Gcl_phases):
    """Contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    Gcl_mags : array_like
        Array of magnitudes of the contours
    Gcl_phases : array_like
        Array of phases in radians of the contours

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.

    """
    # Compute the contours in Gcl-space. Since we're given closed-loop
    # magnitudes and phases, this is just a case of converting them into
    # a complex number.
    Gcl = Gcl_mags*np.exp(1.j*Gcl_phases)

    # Invert Gcl = Gol/(1+Gol) to map the contours into the open-loop space
    return Gcl/(1.0 - Gcl)


def m_circles(mags, phase_min=-359.75, phase_max=-0.25):
    """Constant-magnitude contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    mags : array_like
        Array of magnitudes in dB of the M-circles
    phase_min : degrees
        Minimum phase in degrees of the N-circles
    phase_max : degrees
        Maximum phase in degrees of the N-circles

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.

    """
    # Convert magnitudes and phase range into a grid suitable for
    # building contours
    phases = np.radians(np.linspace(phase_min, phase_max, 2000))
    Gcl_mags, Gcl_phases = np.meshgrid(10.0**(mags/20.0), phases)
    return closed_loop_contours(Gcl_mags, Gcl_phases)


def n_circles(phases, mag_min=-40.0, mag_max=12.0):
    """Constant-phase contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Parameters
    ----------
    phases : array_like
        Array of phases in degrees of the N-circles
    mag_min : dB
        Minimum magnitude in dB of the N-circles
    mag_max : dB
        Maximum magnitude in dB of the N-circles

    Returns
    -------
    contours : complex array
        Array of complex numbers corresponding to the contours.

    """
    # Convert phases and magnitude range into a grid suitable for
    # building contours
    mags = np.linspace(10**(mag_min/20.0), 10**(mag_max/20.0), 2000)
    Gcl_phases, Gcl_mags = np.meshgrid(np.radians(phases), mags)
    return closed_loop_contours(Gcl_mags, Gcl_phases)


# Function aliases
nichols = nichols_plot
