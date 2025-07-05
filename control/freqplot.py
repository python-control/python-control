# freqplot.py - frequency domain plots for control systems
#
# Initial author: Richard M. Murray
# Creation date: 24 May 2009

"""Frequency domain plots for control systems.

This module contains some standard control system plots: Bode plots,
Nyquist plots and other frequency response plots.  The code for
Nichols charts is in nichols.py.  The code for pole-zero diagrams is
in pzmap.py and rlocus.py.

"""

import itertools
import math
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .bdalg import feedback
from .ctrlplot import ControlPlot, _add_arrows_to_line2D, _find_axes_center, \
    _get_color, _get_color_offset, _get_line_labels, _make_legend_labels, \
    _process_ax_keyword, _process_legend_keywords, _process_line_labels, \
    _update_plot_title
from .ctrlutil import unwrap
from .exception import ControlMIMONotImplemented
from .frdata import FrequencyResponseData
from .lti import LTI, _process_frequency_response, frequency_response
from .margins import stability_margins
from .statesp import StateSpace
from .xferfcn import TransferFunction

__all__ = ['bode_plot', 'NyquistResponseData', 'nyquist_response',
           'nyquist_plot', 'singular_values_response',
           'singular_values_plot', 'gangof4_plot', 'gangof4_response',
           'bode', 'nyquist', 'gangof4', 'FrequencyResponseList',
           'NyquistResponseList']

# Default values for module parameter variables
_freqplot_defaults = {
    'freqplot.feature_periphery_decades': 1,
    'freqplot.number_of_samples': 1000,
    'freqplot.dB': False,  # Plot gain in dB
    'freqplot.deg': True,  # Plot phase in degrees
    'freqplot.Hz': False,  # Plot frequency in Hertz
    'freqplot.grid': True,  # Turn on grid for gain and phase
    'freqplot.wrap_phase': False,  # Wrap the phase plot at a given value
    'freqplot.freq_label': "Frequency [{units}]",
    'freqplot.magnitude_label': "Magnitude",
    'freqplot.share_magnitude': 'row',
    'freqplot.share_phase': 'row',
    'freqplot.share_frequency': 'col',
    'freqplot.title_frame': 'axes',
}

#
# Frequency response data list class
#
# This class is a subclass of list that adds a plot() method, enabling
# direct plotting from routines returning a list of FrequencyResponseData
# objects.
#

class FrequencyResponseList(list):
    """List of FrequencyResponseData objects with plotting capability.

    This class consists of a list of `FrequencyResponseData` objects.
    It is a subclass of the Python `list` class, with a `plot` method that
    plots the individual `FrequencyResponseData` objects.

    """
    def plot(self, *args, plot_type=None, **kwargs):
        """Plot a list of frequency responses.

        See `FrequencyResponseData.plot` for details.

        """
        if plot_type == None:
            for response in self:
                if plot_type is not None and response.plot_type != plot_type:
                    raise TypeError(
                        "inconsistent plot_types in data; set plot_type "
                        "to 'bode', 'nichols', or 'svplot'")
                plot_type = response.plot_type

        # Use FRD plot method, which can handle lists via plot functions
        return FrequencyResponseData.plot(
            self, plot_type=plot_type, *args, **kwargs)

#
# Bode plot
#
# This is the default method for plotting frequency responses.  There are
# lots of options available for tuning the format of the plot, (hopefully)
# covering most of the common use cases.
#

def bode_plot(
        data, omega=None, *fmt, ax=None, omega_limits=None, omega_num=None,
        plot=None, plot_magnitude=True, plot_phase=None,
        overlay_outputs=None, overlay_inputs=None, phase_label=None,
        magnitude_label=None, label=None, display_margins=None,
        margins_method='best', title=None, sharex=None, sharey=None, **kwargs):
    """Bode plot for a system.

    Plot the magnitude and phase of the frequency response over a
    (optional) frequency range.

    Parameters
    ----------
    data : list of `FrequencyResponseData` or `LTI`
        List of LTI systems or `FrequencyResponseData` objects.  A
        single system or frequency response can also be passed.
    omega : array_like, optional
        Set of frequencies in rad/sec to plot over.  If not specified, this
        will be determined from the properties of the systems.  Ignored if
        `data` is not a list of systems.
    *fmt : `matplotlib.pyplot.plot` format string, optional
        Passed to `matplotlib` as the format string for all lines in the plot.
        The `omega` parameter must be present (use omega=None if needed).
    dB : bool
        If True, plot result in dB.  Default is False.
    Hz : bool
        If True, plot frequency in Hz (omega must be provided in rad/sec).
        Default value (False) set by `config.defaults['freqplot.Hz']`.
    deg : bool
        If True, plot phase in degrees (else radians).  Default
        value (True) set by `config.defaults['freqplot.deg']`.
    display_margins : bool or str
        If True, draw gain and phase margin lines on the magnitude and phase
        graphs and display the margins at the top of the graph.  If set to
        'overlay', the values for the gain and phase margin are placed on
        the graph.  Setting `display_margins` turns off the axes grid, unless
        `grid` is explicitly set to True.
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
    ax : array of `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified, the
        axes for the current figure are used or, if there is no current
        figure with the correct number and shape of axes, a new figure is
        created.  The shape of the array must match the shape of the
        plotted data.
    freq_label, magnitude_label, phase_label : str, optional
        Labels to use for the frequency, magnitude, and phase axes.
        Defaults are set by `config.defaults['freqplot.<keyword>']`.
    grid : bool, optional
        If True, plot grid lines on gain and phase plots.  Default is set by
        `config.defaults['freqplot.grid']`.
    initial_phase : float, optional
        Set the reference phase to use for the lowest frequency.  If set, the
        initial phase of the Bode plot will be set to the value closest to the
        value specified.  Units are in either degrees or radians, depending on
        the `deg` parameter. Default is -180 if wrap_phase is False, 0 if
        wrap_phase is True.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with the given
        label(s).  If sysdata is a list, strings should be specified for each
        system.  If MIMO, strings required for each system, output, and input.
    legend_map : array of str, optional
        Location of the legend for multi-axes plots.  Specifies an array
        of legend location strings matching the shape of the subplots, with
        each entry being either None (for no legend) or a legend location
        string (see `~matplotlib.pyplot.legend`).
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'center right',
        with no legend for a single response.  Use False to suppress legend.
    margins_method : str, optional
        Method to use in computing margins (see `stability_margins`).
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`. Ignored if
        data is not a list of systems.
    omega_num : int
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.  Ignored if data is
        not a list of systems.
    overlay_inputs, overlay_outputs : bool, optional
        If set to True, combine input and/or output signals onto a single
        plot and use line colors, labels, and a legend to distinguish them.
    plot : bool, optional
        (legacy) If given, `bode_plot` returns the legacy return values
        of magnitude, phase, and frequency.  If False, just return the
        values with no plot.
    plot_magnitude, plot_phase : bool, optional
        If set to False, do not plot the magnitude or phase, respectively.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    share_frequency, share_magnitude, share_phase : str or bool, optional
        Determine whether and how axis limits are shared between the
        indicated variables.  Can be set set to 'row' to share across all
        subplots in a row, 'col' to set across all subplots in a column, or
        False to allow independent limits.  Note: if `sharex` is given,
        it sets the value of `share_frequency`; if `sharey` is given, it
        sets the value of both `share_magnitude` and `share_phase`.
        Default values are 'row' for `share_magnitude` and `share_phase`,
        'col', for `share_frequency`, and can be set using
        `config.defaults['freqplot.share_<axis>']`.
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on an
        axis or `legend_loc` or `legend_map` has been specified.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).
    title_frame : str, optional
        Set the frame of reference used to center the plot title. If set to
        'axes' (default), the horizontal position of the title will be
        centered relative to the axes.  If set to 'figure', it will be
        centered with respect to the figure (faster execution).  The default
        value can be set using `config.defaults['freqplot.title_frame']`.
    wrap_phase : bool or float
        If wrap_phase is False (default), then the phase will be unwrapped
        so that it is continuously increasing or decreasing.  If wrap_phase is
        True the phase will be restricted to the range [-180, 180) (or
        [:math:`-\\pi`, :math:`\\pi`) radians). If `wrap_phase` is specified
        as a float, the phase will be offset by 360 degrees if it falls below
        the specified value. Default value is False and can be set using
        `config.defaults['freqplot.wrap_phase']`.

    See Also
    --------
    frequency_response

    Notes
    -----
    Starting with python-control version 0.10, `bode_plot` returns a
    `ControlPlot` object instead of magnitude, phase, and
    frequency. To recover the old behavior, call `bode_plot` with
    `plot` = True, which will force the legacy values (mag, phase, omega) to
    be returned (with a warning).  To obtain just the frequency response of
    a system (or list of systems) without plotting, use the
    `frequency_response` command.

    If a discrete-time model is given, the frequency response is plotted
    along the upper branch of the unit circle, using the mapping ``z =
    exp(1j * omega * dt)`` where `omega` ranges from 0 to pi/`dt` and `dt`
    is the discrete timebase.  If timebase not specified (`dt` = True),
    `dt` is set to 1.

    The default values for Bode plot configuration parameters can be reset
    using the `config.defaults` dictionary, with module name 'bode'.

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> out = ct.bode_plot(G)

    """
    #
    # Process keywords and set defaults
    #

    # Make a copy of the kwargs dictionary since we will modify it
    kwargs = dict(kwargs)

    # Legacy keywords for margins
    display_margins = config._process_legacy_keyword(
        kwargs, 'margins', 'display_margins', display_margins)
    if kwargs.pop('margin_info', False):
        warnings.warn(
            "keyword 'margin_info' is deprecated; "
            "use 'display_margins='overlay'")
        if display_margins is False:
            raise ValueError(
                "conflicting_keywords: `display_margins` and `margin_info`")

    # Turn off grid if display margins, unless explicitly overridden
    if display_margins and 'grid' not in kwargs:
        kwargs['grid'] = False

    margins_method = config._process_legacy_keyword(
        kwargs, 'method', 'margins_method', margins_method)

    # Get values for params (and pop from list to allow keyword use in plot)
    dB = config._get_param(
        'freqplot', 'dB', kwargs, _freqplot_defaults, pop=True)
    deg = config._get_param(
        'freqplot', 'deg', kwargs, _freqplot_defaults, pop=True)
    Hz = config._get_param(
        'freqplot', 'Hz', kwargs, _freqplot_defaults, pop=True)
    grid = config._get_param(
        'freqplot', 'grid', kwargs, _freqplot_defaults, pop=True)
    wrap_phase = config._get_param(
        'freqplot', 'wrap_phase', kwargs, _freqplot_defaults, pop=True)
    initial_phase = config._get_param(
        'freqplot', 'initial_phase', kwargs, None, pop=True)
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)
    title_frame = config._get_param(
        'freqplot', 'title_frame', kwargs, _freqplot_defaults, pop=True)

    # Set the default labels
    freq_label = config._get_param(
        'freqplot', 'freq_label', kwargs, _freqplot_defaults, pop=True)
    if magnitude_label is None:
        magnitude_label = config._get_param(
            'freqplot', 'magnitude_label', kwargs,
            _freqplot_defaults, pop=True) + (" [dB]" if dB else "")
    if phase_label is None:
        phase_label = "Phase [deg]" if deg else "Phase [rad]"

    # Use sharex and sharey as proxies for share_{magnitude, phase, frequency}
    if sharey is not None:
        if 'share_magnitude' in kwargs or 'share_phase' in kwargs:
            ValueError(
                "sharey cannot be present with share_magnitude/share_phase")
        kwargs['share_magnitude'] = sharey
        kwargs['share_phase'] = sharey
    if sharex is not None:
        if 'share_frequency' in kwargs:
            ValueError(
                "sharex cannot be present with share_frequency")
        kwargs['share_frequency'] = sharex

    if not isinstance(data, (list, tuple)):
        data = [data]

    #
    # Pre-process the data to be plotted (unwrap phase, limit frequencies)
    #
    # To maintain compatibility with legacy uses of bode_plot(), we do some
    # initial processing on the data, specifically phase unwrapping and
    # setting the initial value of the phase.  If bode_plot is called with
    # plot == False, then these values are returned to the user (instead of
    # the list of lines created, which is the new output for _plot functions.
    #

    # If we were passed a list of systems, convert to data
    if any([isinstance(
            sys, (StateSpace, TransferFunction)) for sys in data]):
        data = frequency_response(
            data, omega=omega, omega_limits=omega_limits,
            omega_num=omega_num, Hz=Hz)
    else:
        # Generate warnings if frequency keywords were given
        if omega_num is not None:
            warnings.warn("`omega_num` ignored when passed response data")
        elif omega is not None:
            warnings.warn("`omega` ignored when passed response data")

        # Check to make sure omega_limits is sensible
        if omega_limits is not None and \
           (len(omega_limits) != 2 or omega_limits[1] <= omega_limits[0]):
            raise ValueError(f"invalid limits: {omega_limits=}")

    # If plot_phase is not specified, check the data first, otherwise true
    if plot_phase is None:
        plot_phase = True if data[0].plot_phase is None else data[0].plot_phase

    if not plot_magnitude and not plot_phase:
        raise ValueError(
            "plot_magnitude and plot_phase both False; no data to plot")

    mag_data, phase_data, omega_data = [], [], []
    for response in data:
        noutputs, ninputs = response.noutputs, response.ninputs

        if initial_phase is None:
            # Start phase in the range 0 to -360 w/ initial phase = 0
            # TODO: change this to 0 to 270 (?)
            # If wrap_phase is true, use 0 instead (phase \in (-pi, pi])
            initial_phase_value = -math.pi if wrap_phase is not True else 0
        elif isinstance(initial_phase, (int, float)):
            # Allow the user to override the default calculation
            if deg:
                initial_phase_value = initial_phase/180. * math.pi
            else:
                initial_phase_value = initial_phase
        else:
            raise ValueError("initial_phase must be a number.")

        # Shift and wrap the phase
        phase = np.angle(response.frdata)               # 3D array
        for i, j in itertools.product(range(noutputs), range(ninputs)):
            # Shift the phase if needed
            if abs(phase[i, j, 0] - initial_phase_value) > math.pi:
                phase[i, j] -= 2*math.pi * round(
                    (phase[i, j, 0] - initial_phase_value) / (2*math.pi))

            # Phase wrapping
            if wrap_phase is False:
                phase[i, j] = unwrap(phase[i, j]) # unwrap the phase
            elif wrap_phase is True:
                pass                                    # default calc OK
            elif isinstance(wrap_phase, (int, float)):
                phase[i, j] = unwrap(phase[i, j]) # unwrap phase first
                if deg:
                    wrap_phase *= math.pi/180.

                # Shift the phase if it is below the wrap_phase
                phase[i, j] += 2*math.pi * np.maximum(
                    0, np.ceil((wrap_phase - phase[i, j])/(2*math.pi)))
            else:
                raise ValueError("wrap_phase must be bool or float.")

        # Save the data for later use
        mag_data.append(np.abs(response.frdata))
        phase_data.append(phase)
        omega_data.append(response.omega)

    #
    # Process `plot` keyword
    #
    # We use the `plot` keyword to track legacy usage of `bode_plot`.
    # Prior to v0.10, the `bode_plot` command returned mag, phase, and
    # omega.  Post v0.10, we return an array with the same shape as the
    # axes we use for plotting, with each array element containing a list
    # of lines drawn on that axes.
    #
    # There are three possibilities at this stage in the code:
    #
    # * plot == True: set explicitly by the user. Return mag, phase, omega,
    #   with a warning.
    #
    # * plot == False: set explicitly by the user. Return mag, phase,
    #   omega, with a warning.
    #
    # * plot == None: this is the new default setting.  Return an array of
    #   lines that were drawn.
    #
    # If `bode_plot` was called with no `plot` argument and the return
    # values were used, the new code will cause problems (you get an array
    # of lines instead of magnitude, phase, and frequency).  To recover the
    # old behavior, call `bode_plot` with `plot=True`.
    #
    # All of this should be removed in v0.11+ when we get rid of deprecated
    # code.
    #

    if plot is not None:
        warnings.warn(
            "bode_plot() return value of mag, phase, omega is deprecated; "
            "use frequency_response()", FutureWarning)

    if plot is False:
        # Process the data to match what we were sent
        for i in range(len(mag_data)):
            mag_data[i] = _process_frequency_response(
                data[i], omega_data[i], mag_data[i], squeeze=data[i].squeeze)
            phase_data[i] = _process_frequency_response(
                data[i], omega_data[i], phase_data[i], squeeze=data[i].squeeze)

        if len(data) == 1:
            return mag_data[0], phase_data[0], omega_data[0]
        else:
            return mag_data, phase_data, omega_data
    #
    # Find/create axes
    #
    # Data are plotted in a standard subplots array, whose size depends on
    # which signals are being plotted and how they are combined.  The
    # baseline layout for data is to plot everything separately, with
    # the magnitude and phase for each output making up the rows and the
    # columns corresponding to the different inputs.
    #
    #      Input 0                 Input m
    # +---------------+       +---------------+
    # |  mag H_y0,u0  |  ...  |  mag H_y0,um  |
    # +---------------+       +---------------+
    # +---------------+       +---------------+
    # | phase H_y0,u0 |  ...  | phase H_y0,um |
    # +---------------+       +---------------+
    #         :                       :
    # +---------------+       +---------------+
    # |  mag H_yp,u0  |  ...  |  mag H_yp,um  |
    # +---------------+       +---------------+
    # +---------------+       +---------------+
    # | phase H_yp,u0 |  ...  | phase H_yp,um |
    # +---------------+       +---------------+
    #
    # Several operations are available that change this layout.
    #
    # * Omitting: either the magnitude or the phase plots can be omitted
    #   using the plot_magnitude and plot_phase keywords.
    #
    # * Overlay: inputs and/or outputs can be combined onto a single set of
    #   axes using the overlay_inputs and overlay_outputs keywords.  This
    #   basically collapses data along either the rows or columns, and a
    #   legend is generated.
    #

    # Decide on the maximum number of inputs and outputs
    ninputs, noutputs = 0, 0
    for response in data:       # TODO: make more pythonic/numpic
        ninputs = max(ninputs, response.ninputs)
        noutputs = max(noutputs, response.noutputs)

    # Figure how how many rows and columns to use + offsets for inputs/outputs
    if overlay_outputs and overlay_inputs:
        nrows = plot_magnitude + plot_phase
        ncols = 1
    elif overlay_outputs:
        nrows = plot_magnitude + plot_phase
        ncols = ninputs
    elif overlay_inputs:
        nrows = (noutputs if plot_magnitude else 0) + \
            (noutputs if plot_phase else 0)
        ncols = 1
    else:
        nrows = (noutputs if plot_magnitude else 0) + \
            (noutputs if plot_phase else 0)
        ncols = ninputs

    if ax is None:
        # Set up default sharing of axis limits if not specified
        for kw in ['share_magnitude', 'share_phase', 'share_frequency']:
            if kw not in kwargs or kwargs[kw] is None:
                kwargs[kw] = config.defaults['freqplot.' + kw]

    fig, ax_array = _process_ax_keyword(
        ax, (nrows, ncols), squeeze=False, rcParams=rcParams, clear_text=True)
    legend_loc, legend_map, show_legend = _process_legend_keywords(
        kwargs, (nrows,ncols), 'center right')

    # Get the values for sharing axes limits
    share_magnitude = kwargs.pop('share_magnitude', None)
    share_phase = kwargs.pop('share_phase', None)
    share_frequency = kwargs.pop('share_frequency', None)

    # Set up axes variables for easier access below
    if plot_magnitude and not plot_phase:
        mag_map = np.empty((noutputs, ninputs), dtype=tuple)
        for i in range(noutputs):
            for j in range(ninputs):
                if overlay_outputs and overlay_inputs:
                    mag_map[i, j] = (0, 0)
                elif overlay_outputs:
                    mag_map[i, j] = (0, j)
                elif overlay_inputs:
                    mag_map[i, j] = (i, 0)
                else:
                    mag_map[i, j] = (i, j)
        phase_map = np.full((noutputs, ninputs), None)
        share_phase = False

    elif plot_phase and not plot_magnitude:
        phase_map = np.empty((noutputs, ninputs), dtype=tuple)
        for i in range(noutputs):
            for j in range(ninputs):
                if overlay_outputs and overlay_inputs:
                    phase_map[i, j] = (0, 0)
                elif overlay_outputs:
                    phase_map[i, j] = (0, j)
                elif overlay_inputs:
                    phase_map[i, j] = (i, 0)
                else:
                    phase_map[i, j] = (i, j)
        mag_map = np.full((noutputs, ninputs), None)
        share_magnitude = False

    else:
        mag_map = np.empty((noutputs, ninputs), dtype=tuple)
        phase_map = np.empty((noutputs, ninputs), dtype=tuple)
        for i in range(noutputs):
            for j in range(ninputs):
                if overlay_outputs and overlay_inputs:
                    mag_map[i, j] = (0, 0)
                    phase_map[i, j] = (1, 0)
                elif overlay_outputs:
                    mag_map[i, j] = (0, j)
                    phase_map[i, j] = (1, j)
                elif overlay_inputs:
                    mag_map[i, j] = (i*2, 0)
                    phase_map[i, j] = (i*2 + 1, 0)
                else:
                    mag_map[i, j] = (i*2, j)
                    phase_map[i, j] = (i*2 + 1, j)

    # Identity map needed for setting up shared axes
    ax_map = np.empty((nrows, ncols), dtype=tuple)
    for i, j in itertools.product(range(nrows), range(ncols)):
        ax_map[i, j] = (i, j)

    #
    # Set up axes limit sharing
    #
    # This code uses the share_magnitude, share_phase, and share_frequency
    # keywords to decide which axes have shared limits and what ticklabels
    # to include.  The sharing code needs to come before the plots are
    # generated, but additional code for removing tick labels needs to come
    # *during* and *after* the plots are generated (see below).
    #
    # Note: if the various share_* keywords are None then a previous set of
    # axes are available and no updates should be made.
    #

    # Utility function to turn on sharing
    def _share_axes(ref, share_map, axis):
        ref_ax = ax_array[ref]
        for index in np.nditer(share_map, flags=["refs_ok"]):
            if index.item() == ref:
                continue
            if axis == 'x':
                ax_array[index.item()].sharex(ref_ax)
            elif axis == 'y':
                ax_array[index.item()].sharey(ref_ax)
            else:
                raise ValueError("axis must be 'x' or 'y'")

    # Process magnitude, phase, and frequency axes
    for name, value, map, axis in zip(
            ['share_magnitude', 'share_phase', 'share_frequency'],
            [ share_magnitude,   share_phase,   share_frequency],
            [ mag_map,           phase_map,     ax_map],
            [ 'y',               'y',           'x']):
        if value in [True, 'all']:
            _share_axes(map[0 if axis == 'y' else -1, 0], map, axis)
        elif axis == 'y' and value in ['row']:
            for i in range(noutputs if not overlay_outputs else 1):
                _share_axes(map[i, 0], map[i], 'y')
        elif axis == 'x' and value in ['col']:
            for j in range(ncols):
                _share_axes(map[-1, j], map[:, j], 'x')
        elif value in [False, 'none']:
            # TODO: turn off any sharing that is on
            pass
        elif value is not None:
            raise ValueError(
                f"unknown value for `{name}`: '{value}'")

    #
    # Plot the data
    #
    # The mag_map and phase_map arrays have the indices axes needed for
    # making the plots.  Labels are used on each axes for later creation of
    # legends.  The generic labels if of the form:
    #
    #     To output label, From input label, system name
    #
    # The input and output labels are omitted if overlay_inputs or
    # overlay_outputs is False, respectively.  The system name is always
    # included, since multiple calls to plot() will require a legend that
    # distinguishes which system signals are plotted.  The system name is
    # stripped off later (in the legend-handling code) if it is not needed.
    #
    # Note: if we are building on top of an existing plot, tick labels
    # should be preserved from the existing axes.  For log scale axes the
    # tick labels seem to appear no matter what => we have to detect if
    # they are present at the start and, it not, remove them after calling
    # loglog or semilogx.
    #

    # Create a list of lines for the output
    out = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            out[i, j] = []      # unique list in each element

    # Process label keyword
    line_labels = _process_line_labels(label, len(data), ninputs, noutputs)

    # Utility function for creating line label
    def _make_line_label(response, output_index, input_index):
        label = ""              # start with an empty label

        # Add the output name if it won't appear as an axes label
        if noutputs > 1 and overlay_outputs:
            label += response.output_labels[output_index]

        # Add the input name if it won't appear as a column label
        if ninputs > 1 and overlay_inputs:
            label += ", " if label != "" else ""
            label += response.input_labels[input_index]

        # Add the system name (will strip off later if redundant)
        label += ", " if label != "" else ""
        label += f"{response.sysname}"

        return label

    for index, response in enumerate(data):
        # Get the (pre-processed) data in fully indexed form
        mag = mag_data[index]
        phase = phase_data[index]
        omega_sys, sysname = omega_data[index], response.sysname

        for i, j in itertools.product(range(noutputs), range(ninputs)):
            # Get the axes to use for magnitude and phase
            ax_mag = ax_array[mag_map[i, j]]
            ax_phase = ax_array[phase_map[i, j]]

            # Get the frequencies and convert to Hz, if needed
            omega_plot = omega_sys / (2 * math.pi) if Hz else omega_sys
            if response.isdtime(strict=True):
                nyq_freq = (0.5/response.dt) if Hz else (math.pi/response.dt)

            # Save the magnitude and phase to plot
            mag_plot = 20 * np.log10(mag[i, j]) if dB else mag[i, j]
            phase_plot = phase[i, j] * 180. / math.pi if deg else phase[i, j]

            # Generate a label
            if line_labels is None:
                label = _make_line_label(response, i, j)
            else:
                label = line_labels[index, i, j]

            # Magnitude
            if plot_magnitude:
                pltfcn = ax_mag.semilogx if dB else ax_mag.loglog

                # Plot the main data
                lines = pltfcn(
                    omega_plot, mag_plot, *fmt, label=label, **kwargs)
                out[mag_map[i, j]] += lines

                # Save the information needed for the Nyquist line
                if response.isdtime(strict=True):
                    ax_mag.axvline(
                        nyq_freq, color=lines[0].get_color(), linestyle='--',
                        label='_nyq_mag_' + sysname)

                # Add a grid to the plot
                ax_mag.grid(grid, which='both')

            # Phase
            if plot_phase:
                lines = ax_phase.semilogx(
                    omega_plot, phase_plot, *fmt, label=label, **kwargs)
                out[phase_map[i, j]] += lines

                # Save the information needed for the Nyquist line
                if response.isdtime(strict=True):
                    ax_phase.axvline(
                        nyq_freq, color=lines[0].get_color(), linestyle='--',
                        label='_nyq_phase_' + sysname)

                # Add a grid to the plot
                ax_phase.grid(grid, which='both')

        #
        # Display gain and phase margins (SISO only)
        #

        if display_margins:
            if ninputs > 1 or noutputs > 1:
                raise NotImplementedError(
                    "margins are not available for MIMO systems")

            if display_margins == 'overlay' and len(data) > 1:
                raise NotImplementedError(
                    f"{display_margins=} not supported for multi-trace plots")

            # Compute stability margins for the system
            margins = stability_margins(response, method=margins_method)
            gm, pm, Wcg, Wcp = (margins[i] for i in [0, 1, 3, 4])

            # Figure out sign of the phase at the first gain crossing
            # (needed if phase_wrap is True)
            phase_at_cp = phase[
                0, 0, (np.abs(omega_data[0] - Wcp)).argmin()]
            if phase_at_cp >= 0.:
                phase_limit = 180.
            else:
                phase_limit = -180.

            if Hz:
                Wcg, Wcp = Wcg/(2*math.pi), Wcp/(2*math.pi)

            # Draw lines at gain and phase limits
            if plot_magnitude:
                ax_mag.axhline(y=0 if dB else 1, color='k', linestyle=':',
                               zorder=-20)

            if plot_phase:
                ax_phase.axhline(y=phase_limit if deg else
                                 math.radians(phase_limit),
                                 color='k', linestyle=':', zorder=-20)

            # Annotate the phase margin (if it exists)
            if plot_phase and pm != float('inf') and Wcp != float('nan'):
                # Draw dotted lines marking the gain crossover frequencies
                if plot_magnitude:
                    ax_mag.axvline(Wcp, color='k', linestyle=':', zorder=-30)
                ax_phase.axvline(Wcp, color='k', linestyle=':', zorder=-30)

                # Draw solid segments indicating the margins
                if deg:
                    ax_phase.semilogx(
                        [Wcp, Wcp], [phase_limit + pm, phase_limit],
                        color='k', zorder=-20)
                else:
                    ax_phase.semilogx(
                        [Wcp, Wcp], [math.radians(phase_limit) +
                                     math.radians(pm),
                                     math.radians(phase_limit)],
                        color='k', zorder=-20)

            # Annotate the gain margin (if it exists)
            if plot_magnitude and gm != float('inf') and \
               Wcg != float('nan'):
                # Draw dotted lines marking the phase crossover frequencies
                ax_mag.axvline(Wcg, color='k', linestyle=':', zorder=-30)
                if plot_phase:
                    ax_phase.axvline(Wcg, color='k', linestyle=':', zorder=-30)

                # Draw solid segments indicating the margins
                if dB:
                    ax_mag.semilogx(
                        [Wcg, Wcg], [0, -20*np.log10(gm)],
                        color='k', zorder=-20)
                else:
                    ax_mag.loglog(
                        [Wcg, Wcg], [1., 1./gm], color='k', zorder=-20)

            if display_margins == 'overlay':
                # TODO: figure out how to handle case of multiple lines
                # Put the margin information in the lower left corner
                if plot_magnitude:
                    ax_mag.text(
                        0.04, 0.06,
                        'G.M.: %.2f %s\nFreq: %.2f %s' %
                        (20*np.log10(gm) if dB else gm,
                         'dB ' if dB else '',
                         Wcg, 'Hz' if Hz else 'rad/s'),
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax_mag.transAxes,
                        fontsize=8 if int(mpl.__version__[0]) == 1 else 6)

                if plot_phase:
                    ax_phase.text(
                        0.04, 0.06,
                        'P.M.: %.2f %s\nFreq: %.2f %s' %
                        (pm if deg else math.radians(pm),
                         'deg' if deg else 'rad',
                         Wcp, 'Hz' if Hz else 'rad/s'),
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax_phase.transAxes,
                        fontsize=8 if int(mpl.__version__[0]) == 1 else 6)

            else:
                # Put the title underneath the suptitle (one line per system)
                ax_ = ax_mag if ax_mag else ax_phase
                axes_title = ax_.get_title()
                if axes_title is not None and axes_title != "":
                    axes_title += "\n"
                with plt.rc_context(rcParams):
                    ax_.set_title(
                        axes_title + f"{sysname}: "
                        "Gm = %.2f %s(at %.2f %s), "
                        "Pm = %.2f %s (at %.2f %s)" %
                        (20*np.log10(gm) if dB else gm,
                         'dB ' if dB else '',
                         Wcg, 'Hz' if Hz else 'rad/s',
                         pm if deg else math.radians(pm),
                         'deg' if deg else 'rad',
                         Wcp, 'Hz' if Hz else 'rad/s'))

    #
    # Finishing handling axes limit sharing
    #
    # This code handles labels on Bode plots and also removes tick labels
    # on shared axes.  It needs to come *after* the plots are generated,
    # in order to handle two things:
    #
    # * manually generated labels and grids need to reflect the limits for
    #   shared axes, which we don't know until we have plotted everything;
    #
    # * the loglog and semilog functions regenerate the labels (not quite
    #   sure why, since using sharex and sharey in subplots does not have
    #   this behavior).
    #
    # Note: as before, if the various share_* keywords are None then a
    # previous set of axes are available and no updates are made. (TODO: true?)
    #

    for i in range(noutputs):
        for j in range(ninputs):
            # Utility function to generate phase labels
            def gen_zero_centered_series(val_min, val_max, period):
                v1 = np.ceil(val_min / period - 0.2)
                v2 = np.floor(val_max / period + 0.2)
                return np.arange(v1, v2 + 1) * period

            # Label the phase axes using multiples of 45 degrees
            if plot_phase:
                ax_phase = ax_array[phase_map[i, j]]

                # Set the labels
                if deg:
                    ylim = ax_phase.get_ylim()
                    num = np.floor((ylim[1] - ylim[0]) / 45)
                    factor = max(1, np.round(num / (32 / nrows)) * 2)
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], 45 * factor))
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], 15 * factor), minor=True)
                else:
                    ylim = ax_phase.get_ylim()
                    num = np.ceil((ylim[1] - ylim[0]) / (math.pi/4))
                    factor = max(1, np.round(num / (36 / nrows)) * 2)
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], math.pi / 4. * factor))
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], math.pi / 12. * factor), minor=True)

    # Turn off y tick labels for shared axes
    for i in range(0, noutputs):
        for j in range(1, ncols):
            if share_magnitude in [True, 'all', 'row']:
                ax_array[mag_map[i, j]].tick_params(labelleft=False)
            if share_phase in [True, 'all', 'row']:
                ax_array[phase_map[i, j]].tick_params(labelleft=False)

    # Turn off x tick labels for shared axes
    for i in range(0, nrows-1):
        for j in range(0, ncols):
            if share_frequency in [True, 'all', 'col']:
                ax_array[i, j].tick_params(labelbottom=False)

    # If specific omega_limits were given, use them
    if omega_limits is not None:
        for i, j in itertools.product(range(nrows), range(ncols)):
            ax_array[i, j].set_xlim(omega_limits)

    #
    # Label the axes (including header labels)
    #
    # Once the data are plotted, we label the axes.  The horizontal axes is
    # always frequency and this is labeled only on the bottom most row.  The
    # vertical axes can consist either of a single signal or a combination
    # of signals (when overlay_inputs or overlay_outputs is True)
    #
    # Input/output signals are give at the top of columns and left of rows
    # when these are individually plotted.
    #

    # Label the columns (do this first to get row labels in the right spot)
    for j in range(ncols):
        # If we have more than one column, label the individual responses
        if (noutputs > 1 and not overlay_outputs or ninputs > 1) \
           and not overlay_inputs:
            with plt.rc_context(rcParams):
                ax_array[0, j].set_title(f"From {data[0].input_labels[j]}")

        # Label the frequency axis
        ax_array[-1, j].set_xlabel(
            freq_label.format(units="Hz" if Hz else "rad/s"))

    # Label the rows
    for i in range(noutputs if not overlay_outputs else 1):
        if plot_magnitude:
            ax_mag = ax_array[mag_map[i, 0]]
            ax_mag.set_ylabel(magnitude_label)
        if plot_phase:
            ax_phase = ax_array[phase_map[i, 0]]
            ax_phase.set_ylabel(phase_label)

        if (noutputs > 1 or ninputs > 1) and not overlay_outputs:
            if plot_magnitude and plot_phase:
                # Get existing ylabel for left column and add a blank line
                ax_mag.set_ylabel("\n" + ax_mag.get_ylabel())
                ax_phase.set_ylabel("\n" + ax_phase.get_ylabel())

                # Find the midpoint between the row axes (+ tight_layout)
                _, ypos = _find_axes_center(fig, [ax_mag, ax_phase])

                # Get the bounding box including the labels
                inv_transform = fig.transFigure.inverted()
                mag_bbox = inv_transform.transform(
                    ax_mag.get_tightbbox(fig.canvas.get_renderer()))

                # Figure out location for text (center left in figure frame)
                xpos = mag_bbox[0, 0]               # left edge

                # Put a centered label as text outside the box
                fig.text(
                    0.8 * xpos, ypos, f"To {data[0].output_labels[i]}\n",
                    rotation=90, ha='left', va='center',
                    fontsize=rcParams['axes.titlesize'])
            else:
                # Only a single axes => add label to the left
                ax_array[i, 0].set_ylabel(
                    f"To {data[0].output_labels[i]}\n" +
                    ax_array[i, 0].get_ylabel())

    #
    # Update the plot title (= figure suptitle)
    #
    # If plots are built up by multiple calls to plot() and the title is
    # not given, then the title is updated to provide a list of unique text
    # items in each successive title.  For data generated by the frequency
    # response function this will generate a common prefix followed by a
    # list of systems (e.g., "Step response for sys[1], sys[2]").
    #

    # Set initial title for the data (unique system names, preserving order)
    seen = set()
    sysnames = [response.sysname for response in data if not
                (response.sysname in seen or seen.add(response.sysname))]

    if ax is None and title is None:
        if data[0].title is None:
            title = "Bode plot for " + ", ".join(sysnames)
        else:
            # Allow data to set the title (used by gangof4)
            title = data[0].title
        _update_plot_title(title, fig, rcParams=rcParams, frame=title_frame)
    elif ax is None:
        _update_plot_title(
            title, fig=fig, rcParams=rcParams, frame=title_frame,
            use_existing=False)

    #
    # Create legends
    #
    # Legends can be placed manually by passing a legend_map array that
    # matches the shape of the sublots, with each item being a string
    # indicating the location of the legend for that axes (or None for no
    # legend).
    #
    # If no legend spec is passed, a minimal number of legends are used so
    # that each line in each axis can be uniquely identified.  The details
    # depends on the various plotting parameters, but the general rule is
    # to place legends in the top row and right column.
    #
    # Because plots can be built up by multiple calls to plot(), the legend
    # strings are created from the line labels manually.  Thus an initial
    # call to plot() may not generate any legends (e.g., if no signals are
    # overlaid), but subsequent calls to plot() will need a legend for each
    # different response (system).
    #

    # Create axis legends
    if show_legend != False:
        # Figure out where to put legends
        if legend_map is None:
            legend_map = np.full(ax_array.shape, None, dtype=object)
            legend_map[0, -1] = legend_loc

        legend_array = np.full(ax_array.shape, None, dtype=object)
        for i, j in itertools.product(range(nrows), range(ncols)):
            if legend_map[i, j] is None:
                continue
            ax = ax_array[i, j]

            # Get the labels to use, removing common strings
            lines = [line for line in ax.get_lines()
                     if line.get_label()[0] != '_']
            labels = _make_legend_labels(
                [line.get_label() for line in lines],
                ignore_common=line_labels is not None)

            # Generate the label, if needed
            if show_legend == True or len(labels) > 1:
                with plt.rc_context(rcParams):
                    legend_array[i, j] = ax.legend(
                        lines, labels, loc=legend_map[i, j])
    else:
        legend_array = None

    #
    # Legacy return processing
    #
    if plot is True:            # legacy usage; remove in future release
        # Process the data to match what we were sent
        for i in range(len(mag_data)):
            mag_data[i] = _process_frequency_response(
                data[i], omega_data[i], mag_data[i], squeeze=data[i].squeeze)
            phase_data[i] = _process_frequency_response(
                data[i], omega_data[i], phase_data[i], squeeze=data[i].squeeze)

        if len(data) == 1:
            return mag_data[0], phase_data[0], omega_data[0]
        else:
            return mag_data, phase_data, omega_data

    return ControlPlot(out, ax_array, fig, legend=legend_array)


#
# Nyquist plot
#

# Default values for module parameter variables
_nyquist_defaults = {
    'nyquist.primary_style': ['-', '-.'],       # style for primary curve
    'nyquist.mirror_style': ['--', ':'],        # style for mirror curve
    'nyquist.arrows': 3,                        # number of arrows around curve
    'nyquist.arrow_size': 8,                    # pixel size for arrows
    'nyquist.encirclement_threshold': 0.05,     # warning threshold
    'nyquist.indent_radius': 1e-4,              # indentation radius
    'nyquist.indent_direction': 'right',        # indentation direction
    'nyquist.indent_points': 200,               # number of points to insert
    'nyquist.max_curve_magnitude': 15,          # rescale large values
    'nyquist.blend_fraction': 0.15,             # when to start scaling
    'nyquist.max_curve_offset': 0.02,           # offset of primary/mirror
    'nyquist.start_marker': 'o',                # marker at start of curve
    'nyquist.start_marker_size': 4,             # size of the marker
    'nyquist.circle_style':                     # style for unit circles
      {'color': 'black', 'linestyle': 'dashed', 'linewidth': 1}
}


class NyquistResponseData:
    """Nyquist response data object.

    Nyquist contour analysis allows the stability and robustness of a
    closed loop linear system to be evaluated using the open loop response
    of the loop transfer function.  The NyquistResponseData class is used
    by the `nyquist_response` function to return the
    response of a linear system along the Nyquist 'D' contour.  The
    response object can be used to obtain information about the Nyquist
    response or to generate a Nyquist plot.

    Parameters
    ----------
    count : integer
        Number of encirclements of the -1 point by the Nyquist curve for
        a system evaluated along the Nyquist contour.
    contour : complex array
        The Nyquist 'D' contour, with appropriate indentations to avoid
        open loop poles and zeros near/on the imaginary axis.
    response : complex array
        The value of the linear system under study along the Nyquist contour.
    dt : None or float
        The system timebase.
    sysname : str
        The name of the system being analyzed.
    return_contour : bool
        If True, when the object is accessed as an iterable return two
        elements: `count` (number of encirclements) and `contour`.  If
        False (default), then return only `count`.

    """
    def __init__(
            self, count, contour, response, dt, sysname=None,
            return_contour=False):
        self.count = count
        self.contour = contour
        self.response = response
        self.dt = dt
        self.sysname = sysname
        self.return_contour = return_contour

    # Implement iter to allow assigning to a tuple
    def __iter__(self):
        if self.return_contour:
            return iter((self.count, self.contour))
        else:
            return iter((self.count, ))

    # Implement (thin) getitem to allow access via legacy indexing
    def __getitem__(self, index):
        return list(self.__iter__())[index]

    # Implement (thin) len to emulate legacy testing interface
    def __len__(self):
        return 2 if self.return_contour else 1

    def plot(self, *args, **kwargs):
        """Plot a list of Nyquist responses.

        See `nyquist_plot` for details.

        """
        return nyquist_plot(self, *args, **kwargs)


class NyquistResponseList(list):
    """List of NyquistResponseData objects with plotting capability.

    This class consists of a list of `NyquistResponseData` objects.
    It is a subclass of the Python `list` class, with a `plot` method that
    plots the individual `NyquistResponseData` objects.

    """
    def plot(self, *args, **kwargs):
        """Plot a list of Nyquist responses.

        See `nyquist_plot` for details.

        """
        return nyquist_plot(self, *args, **kwargs)


def nyquist_response(
        sysdata, omega=None, omega_limits=None, omega_num=None,
        return_contour=False, warn_encirclements=True, warn_nyquist=True,
        _kwargs=None, _check_kwargs=True, **kwargs):
    """Nyquist response for a system.

    Computes a Nyquist contour for the system over a (optional) frequency
    range and evaluates the number of net encirclements.  The curve is
    computed by evaluating the Nyquist segment along the positive imaginary
    axis, with a mirror image generated to reflect the negative imaginary
    axis.  Poles on or near the imaginary axis are avoided using a small
    indentation.  The portion of the Nyquist contour at infinity is not
    explicitly computed (since it maps to a constant value for any system
    with a proper transfer function).

    Parameters
    ----------
    sysdata : LTI or list of LTI
        List of linear input/output systems (single system is OK). Nyquist
        curves for each system are plotted on the same graph.
    omega : array_like, optional
        Set of frequencies to be evaluated, in rad/sec.

    Returns
    -------
    responses : list of `NyquistResponseData`
        For each system, a Nyquist response data object is returned.  If
        `sysdata` is a single system, a single element is returned (not a
        list).
    response.count : int
        Number of encirclements of the point -1 by the Nyquist curve.  If
        multiple systems are given, an array of counts is returned.
    response.contour : ndarray
        The contour used to create the primary Nyquist curve segment.  To
        obtain the Nyquist curve values, evaluate system(s) along contour.

    Other Parameters
    ----------------
    encirclement_threshold : float, optional
        Define the threshold for generating a warning if the number of net
        encirclements is a non-integer value.  Default value is 0.05 and can
        be set using `config.defaults['nyquist.encirclement_threshold']`.
    indent_direction : str, optional
        For poles on the imaginary axis, set the direction of indentation to
        be 'right' (default), 'left', or 'none'.  The default value can
        be set using `config.defaults['nyquist.indent_direction']`.
    indent_points : int, optional
        Number of points to insert in the Nyquist contour around poles that
        are at or near the imaginary axis.
    indent_radius : float, optional
        Amount to indent the Nyquist contour around poles on or near the
        imaginary axis. Portions of the Nyquist plot corresponding to
        indented portions of the contour are plotted using a different line
        style. The default value can be set using
        `config.defaults['nyquist.indent_radius']`.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`.
    omega_num : int, optional
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.
    warn_nyquist : bool, optional
        If set to False, turn off warnings about frequencies above Nyquist.
    warn_encirclements : bool, optional
        If set to False, turn off warnings about number of encirclements not
        meeting the Nyquist criterion.

    Notes
    -----
    If a discrete-time model is given, the frequency response is computed
    along the upper branch of the unit circle, using the mapping ``z =
    exp(1j * omega * dt)`` where `omega` ranges from 0 to pi/`dt` and
    `dt` is the discrete timebase.  If timebase not specified
    (`dt` = True), `dt` is set to 1.

    If a continuous-time system contains poles on or near the imaginary
    axis, a small indentation will be used to avoid the pole.  The radius
    of the indentation is given by `indent_radius` and it is taken to the
    right of stable poles and the left of unstable poles.  If a pole is
    exactly on the imaginary axis, the `indent_direction` parameter can be
    used to set the direction of indentation.  Setting `indent_direction`
    to 'none' will turn off indentation.

    For those portions of the Nyquist plot in which the contour is indented
    to avoid poles, resulting in a scaling of the Nyquist plot, the line
    styles are according to the settings of the `primary_style` and
    `mirror_style` keywords.  By default the scaled portions of the primary
    curve use a dotted line style and the scaled portion of the mirror
    image use a dashdot line style.

    If the legacy keyword `return_contour` is specified as True, the
    response object can be iterated over to return ``(count, contour)``.
    This behavior is deprecated and will be removed in a future release.

    See Also
    --------
    nyquist_plot

    Examples
    --------
    >>> G = ct.zpk([], [-1, -2, -3], gain=100)
    >>> response = ct.nyquist_response(G)
    >>> count = response.count
    >>> cplt = response.plot()

    """
    # Create unified list of keyword arguments
    if _kwargs is None:
        _kwargs = kwargs
    else:
        # Use existing dictionary, to keep track of processed keywords
        _kwargs |= kwargs

    # Get values for params
    omega_num_given = omega_num is not None
    omega_num = config._get_param('freqplot', 'number_of_samples', omega_num)
    indent_radius = config._get_param(
        'nyquist', 'indent_radius', _kwargs, _nyquist_defaults, pop=True)
    encirclement_threshold = config._get_param(
        'nyquist', 'encirclement_threshold', _kwargs,
        _nyquist_defaults, pop=True)
    indent_direction = config._get_param(
        'nyquist', 'indent_direction', _kwargs, _nyquist_defaults, pop=True)
    indent_points = config._get_param(
        'nyquist', 'indent_points', _kwargs, _nyquist_defaults, pop=True)

    if _check_kwargs and _kwargs:
        raise TypeError("unrecognized keywords: ", str(_kwargs))

    # Convert the first argument to a list
    syslist = sysdata if isinstance(sysdata, (list, tuple)) else [sysdata]

    # Determine the range of frequencies to use, based on args/features
    omega, omega_range_given = _determine_omega_vector(
        syslist, omega, omega_limits, omega_num, feature_periphery_decades=2)

    # If omega was not specified explicitly, start at omega = 0
    if not omega_range_given:
        if omega_num_given:
            # Just reset the starting point
            omega[0] = 0.0
        else:
            # Insert points between the origin and the first frequency point
            omega = np.concatenate((
                np.linspace(0, omega[0], indent_points), omega[1:]))

    # Go through each system and keep track of the results
    responses = []
    for idx, sys in enumerate(syslist):
        if not sys.issiso():
            # TODO: Add MIMO nyquist plots.
            raise ControlMIMONotImplemented(
                "Nyquist plot currently only supports SISO systems.")

        # Figure out the frequency range
        if isinstance(sys, FrequencyResponseData) and sys._ifunc is None \
           and not omega_range_given:
            omega_sys = sys.omega               # use system frequencies
        else:
            omega_sys = np.asarray(omega)       # use common omega vector

        # Determine the contour used to evaluate the Nyquist curve
        if sys.isdtime(strict=True):
            # Restrict frequencies for discrete-time systems
            nyq_freq = math.pi / sys.dt
            if not omega_range_given:
                # limit up to and including Nyquist frequency
                omega_sys = np.hstack((
                    omega_sys[omega_sys < nyq_freq], nyq_freq))

            # Issue a warning if we are sampling above Nyquist
            if np.any(omega_sys * sys.dt > np.pi) and warn_nyquist:
                warnings.warn("evaluation above Nyquist frequency")

        # do indentations in s-plane where it is more convenient
        splane_contour = 1j * omega_sys

        # Bend the contour around any poles on/near the imaginary axis
        if isinstance(sys, (StateSpace, TransferFunction)) \
                and indent_direction != 'none':
            if sys.isctime():
                splane_poles = sys.poles()
                splane_cl_poles = sys.feedback().poles()
            else:
                # map z-plane poles to s-plane. We ignore any at the origin
                # to avoid numerical warnings because we know we
                # don't need to indent for them
                zplane_poles = sys.poles()
                zplane_poles = zplane_poles[~np.isclose(abs(zplane_poles), 0.)]
                splane_poles = np.log(zplane_poles) / sys.dt

                zplane_cl_poles = sys.feedback().poles()
                # eliminate z-plane poles at the origin to avoid warnings
                zplane_cl_poles = zplane_cl_poles[
                    ~np.isclose(abs(zplane_cl_poles), 0.)]
                splane_cl_poles = np.log(zplane_cl_poles) / sys.dt

            #
            # Check to make sure indent radius is small enough
            #
            # If there is a closed loop pole that is near the imaginary axis
            # at a point that is near an open loop pole, it is possible that
            # indentation might skip or create an extraneous encirclement.
            # We check for that situation here and generate a warning if that
            # could happen.
            #
            for p_cl in splane_cl_poles:
                # See if any closed loop poles are near the imaginary axis
                if abs(p_cl.real) <= indent_radius:
                    # See if any open loop poles are close to closed loop poles
                    if len(splane_poles) > 0:
                        p_ol = splane_poles[
                            (np.abs(splane_poles - p_cl)).argmin()]

                        if abs(p_ol - p_cl) <= indent_radius and \
                                warn_encirclements:
                            warnings.warn(
                                "indented contour may miss closed loop pole; "
                                "consider reducing indent_radius to below "
                                f"{abs(p_ol - p_cl):5.2g}", stacklevel=2)

            #
            # See if we should add some frequency points near imaginary poles
            #
            for p in splane_poles:
                # See if we need to process this pole (skip if on the negative
                # imaginary axis or not near imaginary axis + user override)
                if p.imag < 0 or abs(p.real) > indent_radius or \
                   omega_range_given:
                    continue

                # Find the frequencies before the pole frequency
                below_points = np.argwhere(
                    splane_contour.imag - abs(p.imag) < -indent_radius)
                if below_points.size > 0:
                    first_point = below_points[-1].item()
                    start_freq = p.imag - indent_radius
                else:
                    # Add the points starting at the beginning of the contour
                    assert splane_contour[0] == 0
                    first_point = 0
                    start_freq = 0

                # Find the frequencies after the pole frequency
                above_points = np.argwhere(
                    splane_contour.imag - abs(p.imag) > indent_radius)
                last_point = above_points[0].item()

                # Add points for half/quarter circle around pole frequency
                # (these will get indented left or right below)
                splane_contour = np.concatenate((
                    splane_contour[0:first_point+1],
                    (1j * np.linspace(
                        start_freq, p.imag + indent_radius, indent_points)),
                    splane_contour[last_point:]))

            # Indent points that are too close to a pole
            if len(splane_poles) > 0: # accommodate no splane poles if dtime sys
                for i, s in enumerate(splane_contour):
                    # Find the nearest pole
                    p = splane_poles[(np.abs(splane_poles - s)).argmin()]

                    # See if we need to indent around it
                    if abs(s - p) < indent_radius:
                        # Figure out how much to offset (simple trigonometry)
                        offset = np.sqrt(
                            indent_radius ** 2 - (s - p).imag ** 2) \
                            - (s - p).real

                        # Figure out which way to offset the contour point
                        if p.real < 0 or (p.real == 0 and
                                        indent_direction == 'right'):
                            # Indent to the right
                            splane_contour[i] += offset

                        elif p.real > 0 or (p.real == 0 and
                                            indent_direction == 'left'):
                            # Indent to the left
                            splane_contour[i] -= offset

                        else:
                            raise ValueError(
                                "unknown value for indent_direction")

        # change contour to z-plane if necessary
        if sys.isctime():
            contour = splane_contour
        else:
            contour = np.exp(splane_contour * sys.dt)

        # Make sure we don't try to evaluate at a pole
        if isinstance(sys, (StateSpace, TransferFunction)):
            if any([pole in contour for pole in sys.poles()]):
                raise RuntimeError(
                    "attempt to evaluate at a pole; indent required")

        # Compute the primary curve
        resp = sys(contour)

        # Compute CW encirclements of -1 by integrating the (unwrapped) angle
        phase = -unwrap(np.angle(resp + 1))
        encirclements = np.sum(np.diff(phase)) / np.pi
        count = int(np.round(encirclements, 0))

        # Let the user know if the count might not make sense
        if abs(encirclements - count) > encirclement_threshold and \
           warn_encirclements:
            warnings.warn(
                "number of encirclements was a non-integer value; this can"
                " happen is contour is not closed, possibly based on a"
                " frequency range that does not include zero.")

        #
        # Make sure that the encirclements match the Nyquist criterion
        #
        # If the user specifies the frequency points to use, it is possible
        # to miss encirclements, so we check here to make sure that the
        # Nyquist criterion is actually satisfied.
        #
        if isinstance(sys, (StateSpace, TransferFunction)):
            # Count the number of open/closed loop RHP poles
            if sys.isctime():
                if indent_direction == 'right':
                    P = (sys.poles().real > 0).sum()
                else:
                    P = (sys.poles().real >= 0).sum()
                Z = (sys.feedback().poles().real >= 0).sum()
            else:
                if indent_direction == 'right':
                    P = (np.abs(sys.poles()) > 1).sum()
                else:
                    P = (np.abs(sys.poles()) >= 1).sum()
                Z = (np.abs(sys.feedback().poles()) >= 1).sum()

            # Check to make sure the results make sense; warn if not
            if Z != count + P and warn_encirclements:
                warnings.warn(
                    "number of encirclements does not match Nyquist criterion;"
                    " check frequency range and indent radius/direction",
                    UserWarning, stacklevel=2)
            elif indent_direction == 'none' and any(sys.poles().real == 0) \
                 and warn_encirclements:
                warnings.warn(
                    "system has pure imaginary poles but indentation is"
                    " turned off; results may be meaningless",
                    RuntimeWarning, stacklevel=2)

        # Decide on system name
        sysname = sys.name if sys.name is not None else f"Unknown-{idx}"

        responses.append(NyquistResponseData(
            count, contour, resp, sys.dt, sysname=sysname,
            return_contour=return_contour))

    if isinstance(sysdata, (list, tuple)):
        return NyquistResponseList(responses)
    else:
        return responses[0]


def nyquist_plot(
        data, omega=None, plot=None, label_freq=0, color=None, label=None,
        return_contour=None, title=None, ax=None,
        unit_circle=False, mt_circles=None, ms_circles=None, **kwargs):
    """Nyquist plot for a system.

    Generates a Nyquist plot for the system over a (optional) frequency
    range.  The curve is computed by evaluating the Nyquist segment along
    the positive imaginary axis, with a mirror image generated to reflect
    the negative imaginary axis.  Poles on or near the imaginary axis are
    avoided using a small indentation.  The portion of the Nyquist contour
    at infinity is not explicitly computed (since it maps to a constant
    value for any system with a proper transfer function).

    Parameters
    ----------
    data : list of `LTI` or `NyquistResponseData`
        List of linear input/output systems (single system is OK) or
        Nyquist responses (computed using `nyquist_response`).
        Nyquist curves for each system are plotted on the same graph.
    omega : array_like, optional
        Set of frequencies to be evaluated, in rad/sec. Specifying
        `omega` as a list of two elements is equivalent to providing
        `omega_limits`.
    unit_circle : bool, optional
        If True, display the unit circle, to read gain crossover
        frequency.  The circle style is determined by
        `config.defaults['nyquist.circle_style']`.
    mt_circles : array_like, optional
        Draw circles corresponding to the given magnitudes of sensitivity.
    ms_circles : array_like, optional
        Draw circles corresponding to the given magnitudes of complementary
        sensitivity.
    **kwargs : `matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords passed to `matplotlib` to specify line properties.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : 2D array of `matplotlib.lines.Line2D`
        Array containing information on each line in the plot.  The shape
        of the array is given by (nsys, 4) where nsys is the number of
        systems or Nyquist responses passed to the function.  The second
        index specifies the segment type:

            - lines[idx, 0]: unscaled portion of the primary curve
            - lines[idx, 1]: scaled portion of the primary curve
            - lines[idx, 2]: unscaled portion of the mirror curve
            - lines[idx, 3]: scaled portion of the mirror curve

    cplt.axes : 2D array of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.
    cplt.legend : 2D array of `matplotlib.legend.Legend`
        Legend object(s) contained in the plot.

    Other Parameters
    ----------------
    arrows : int or 1D/2D array of floats, optional
        Specify the number of arrows to plot on the Nyquist curve.  If an
        integer is passed. that number of equally spaced arrows will be
        plotted on each of the primary segment and the mirror image.  If a
        1D array is passed, it should consist of a sorted list of floats
        between 0 and 1, indicating the location along the curve to plot an
        arrow.  If a 2D array is passed, the first row will be used to
        specify arrow locations for the primary curve and the second row
        will be used for the mirror image.  Default value is 2 and can be
        set using `config.defaults['nyquist.arrows']`.
    arrow_size : float, optional
        Arrowhead width and length (in display coordinates).  Default value is
        8 and can be set using `config.defaults['nyquist.arrow_size']`.
    arrow_style : matplotlib.patches.ArrowStyle, optional
        Define style used for Nyquist curve arrows (overrides `arrow_size`).
    ax : `matplotlib.axes.Axes`, optional
        The matplotlib axes to draw the figure on.  If not specified and
        the current figure has a single axes, that axes is used.
        Otherwise, a new figure is created.
    blend_fraction : float, optional
        For portions of the Nyquist curve that are scaled to have a maximum
        magnitude of `max_curve_magnitude`, begin a smooth rescaling at
        this fraction of `max_curve_magnitude`. Default value is 0.15.
    encirclement_threshold : float, optional
        Define the threshold for generating a warning if the number of net
        encirclements is a non-integer value.  Default value is 0.05 and can
        be set using `config.defaults['nyquist.encirclement_threshold']`.
    indent_direction : str, optional
        For poles on the imaginary axis, set the direction of indentation to
        be 'right' (default), 'left', or 'none'.
    indent_points : int, optional
        Number of points to insert in the Nyquist contour around poles that
        are at or near the imaginary axis.
    indent_radius : float, optional
        Amount to indent the Nyquist contour around poles on or near the
        imaginary axis. Portions of the Nyquist plot corresponding to indented
        portions of the contour are plotted using a different line style.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with the given
        label(s).  If `data` is a list, strings should be specified for each
        system.
    label_freq : int, optional
        Label every nth frequency on the plot.  If not specified, no labels
        are generated.
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'upper right',
        with no legend for a single response.  Use False to suppress legend.
    max_curve_magnitude : float, optional
        Restrict the maximum magnitude of the Nyquist plot to this value.
        Portions of the Nyquist plot whose magnitude is restricted are
        plotted using a different line style.  The default value is 20 and
        can be set using `config.defaults['nyquist.max_curve_magnitude']`.
    max_curve_offset : float, optional
        When plotting scaled portion of the Nyquist plot, increase/decrease
        the magnitude by this fraction of the max_curve_magnitude to allow
        any overlaps between the primary and mirror curves to be avoided.
        The default value is 0.02 and can be set using
        `config.defaults['nyquist.max_curve_magnitude']`.
    mirror_style : [str, str] or False
        Linestyles for mirror image of the Nyquist curve.  The first element
        is used for unscaled portions of the Nyquist curve, the second element
        is used for portions that are scaled (using max_curve_magnitude).  If
        False then omit completely.  Default linestyle (['--', ':']) is
        determined by `config.defaults['nyquist.mirror_style']`.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`.
    omega_num : int, optional
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.  Ignored if `data`
        is not a system or list of systems.
    plot : bool, optional
        (legacy) If given, `nyquist_plot` returns the legacy return values
        of (counts, contours).  If False, return the values with no plot.
    primary_style : [str, str], optional
        Linestyles for primary image of the Nyquist curve.  The first
        element is used for unscaled portions of the Nyquist curve,
        the second element is used for portions that are scaled (using
        max_curve_magnitude).  Default linestyle (['-', '-.']) is
        determined by `config.defaults['nyquist.mirror_style']`.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set by `config.defaults['ctrlplot.rcParams']`.
    return_contour : bool, optional
        (legacy) If True, return the encirclement count and Nyquist
        contour used to generate the Nyquist plot.
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on the
        plot or `legend_loc` has been specified.
    start_marker : str, optional
        Matplotlib marker to use to mark the starting point of the Nyquist
        plot.  Defaults value is 'o' and can be set using
        `config.defaults['nyquist.start_marker']`.
    start_marker_size : float, optional
        Start marker size (in display coordinates).  Default value is
        4 and can be set using `config.defaults['nyquist.start_marker_size']`.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).
    title_frame : str, optional
        Set the frame of reference used to center the plot title. If set to
        'axes' (default), the horizontal position of the title will
        centered relative to the axes.  If set to 'figure', it will be
        centered with respect to the figure (faster execution).
    warn_nyquist : bool, optional
        If set to False, turn off warnings about frequencies above Nyquist.
    warn_encirclements : bool, optional
        If set to False, turn off warnings about number of encirclements not
        meeting the Nyquist criterion.

    See Also
    --------
    nyquist_response

    Notes
    -----
    If a discrete-time model is given, the frequency response is computed
    along the upper branch of the unit circle, using the mapping ``z =
    exp(1j * omega * dt)`` where `omega` ranges from 0 to pi/`dt` and
    `dt` is the discrete timebase.  If timebase not specified
    (`dt` = True), `dt` is set to 1.

    If a continuous-time system contains poles on or near the imaginary
    axis, a small indentation will be used to avoid the pole.  The radius
    of the indentation is given by `indent_radius` and it is taken to the
    right of stable poles and the left of unstable poles.  If a pole is
    exactly on the imaginary axis, the `indent_direction` parameter can be
    used to set the direction of indentation.  Setting `indent_direction`
    to 'none' will turn off indentation.  If `return_contour` is True,
    the exact contour used for evaluation is returned.

    For those portions of the Nyquist plot in which the contour is indented
    to avoid poles, resulting in a scaling of the Nyquist plot, the line
    styles are according to the settings of the `primary_style` and
    `mirror_style` keywords.  By default the scaled portions of the primary
    curve use a dashdot line style and the scaled portions of the mirror
    image use a dotted line style.

    Examples
    --------
    >>> G = ct.zpk([], [-1, -2, -3], gain=100)
    >>> out = ct.nyquist_plot(G)

    """
    #
    # Keyword processing
    #
    # Keywords for the nyquist_plot function can either be keywords that
    # are unique to this function, keywords that are intended for use by
    # nyquist_response (if data is a list of systems), or keywords that
    # are intended for the plotting commands.
    #
    # We first pop off all keywords that are used directly by this
    # function.  If data is a list of systems, when then pop off keywords
    # that correspond to nyquist_response() keywords.  The remaining
    # keywords are passed to matplotlib (and will generate an error if
    # unrecognized).
    #

    # Get values for params (and pop from list to allow keyword use in plot)
    arrows = config._get_param(
        'nyquist', 'arrows', kwargs, _nyquist_defaults, pop=True)
    arrow_size = config._get_param(
        'nyquist', 'arrow_size', kwargs, _nyquist_defaults, pop=True)
    arrow_style = config._get_param('nyquist', 'arrow_style', kwargs, None)
    ax_user = ax
    max_curve_magnitude = config._get_param(
        'nyquist', 'max_curve_magnitude', kwargs, _nyquist_defaults, pop=True)
    blend_fraction = config._get_param(
        'nyquist', 'blend_fraction', kwargs, _nyquist_defaults, pop=True)
    max_curve_offset = config._get_param(
        'nyquist', 'max_curve_offset', kwargs, _nyquist_defaults, pop=True)
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)
    start_marker = config._get_param(
        'nyquist', 'start_marker', kwargs, _nyquist_defaults, pop=True)
    start_marker_size = config._get_param(
        'nyquist', 'start_marker_size', kwargs, _nyquist_defaults, pop=True)
    title_frame = config._get_param(
        'freqplot', 'title_frame', kwargs, _freqplot_defaults, pop=True)

    # Set line styles for the curves
    def _parse_linestyle(style_name, allow_false=False):
        style = config._get_param(
            'nyquist', style_name, kwargs, _nyquist_defaults, pop=True)
        if isinstance(style, str):
            # Only one style provided, use the default for the other
            style = [style, _nyquist_defaults['nyquist.' + style_name][1]]
            warnings.warn(
                "use of a single string for linestyle will be deprecated "
                " in a future release", PendingDeprecationWarning)
        if (allow_false and style is False) or \
           (isinstance(style, list) and len(style) == 2):
            return style
        else:
            raise ValueError(f"invalid '{style_name}': {style}")

    primary_style = _parse_linestyle('primary_style')
    mirror_style = _parse_linestyle('mirror_style', allow_false=True)

    # Parse the arrows keyword
    if not arrows:
        arrow_pos = []
    elif isinstance(arrows, int):
        N = arrows
        # Space arrows out, starting midway along each "region"
        arrow_pos = np.linspace(0.5/N, 1 + 0.5/N, N, endpoint=False)
    elif isinstance(arrows, (list, np.ndarray)):
        arrow_pos = np.sort(np.atleast_1d(arrows))
    else:
        raise ValueError("unknown or unsupported arrow location")

    # Set the arrow style
    if arrow_style is None:
        arrow_style = mpl.patches.ArrowStyle(
            'simple', head_width=arrow_size, head_length=arrow_size)

    # If argument was a singleton, turn it into a tuple
    if not isinstance(data, (list, tuple)):
        data = [data]

    # Process label keyword
    line_labels = _process_line_labels(label, len(data))

    # If we are passed a list of systems, compute response first
    if all([isinstance(
            sys, (StateSpace, TransferFunction, FrequencyResponseData))
            for sys in data]):
        # Get the response; pop explicit keywords here, kwargs in _response()
        nyquist_responses = nyquist_response(
            data, omega=omega, return_contour=return_contour,
            omega_limits=kwargs.pop('omega_limits', None),
            omega_num=kwargs.pop('omega_num', None),
            warn_encirclements=kwargs.pop('warn_encirclements', True),
            warn_nyquist=kwargs.pop('warn_nyquist', True),
            _kwargs=kwargs, _check_kwargs=False)
    else:
        nyquist_responses = data

    # Legacy return value processing
    if plot is not None or return_contour is not None:
        warnings.warn(
            "nyquist_plot() return value of count[, contour] is deprecated; "
            "use nyquist_response()", FutureWarning)

        # Extract out the values that we will eventually return
        counts = [response.count for response in nyquist_responses]
        contours = [response.contour for response in nyquist_responses]

    if plot is False:
        # Make sure we used all of the keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

        if len(data) == 1:
            counts, contours = counts[0], contours[0]

        # Return counts and (optionally) the contour we used
        return (counts, contours) if return_contour else counts

    fig, ax = _process_ax_keyword(
        ax_user, shape=(1, 1), squeeze=True, rcParams=rcParams)
    legend_loc, _, show_legend = _process_legend_keywords(
        kwargs, None, 'upper right')

    # Figure out where the blended curve should start
    if blend_fraction < 0 or blend_fraction > 1:
        raise ValueError("blend_fraction must be between 0 and 1")
    blend_curve_start = (1 - blend_fraction) * max_curve_magnitude

    # Create a list of lines for the output
    out = np.empty((len(nyquist_responses), 4), dtype=object)
    for i in range(len(nyquist_responses)):
        for j in range(4):
            out[i, j] = []      # unique list in each element

    for idx, response in enumerate(nyquist_responses):
        resp = response.response
        if response.dt in [0, None]:
            splane_contour = response.contour
        else:
            splane_contour = np.log(response.contour) / response.dt

        # Find the different portions of the curve (with scaled pts marked)
        reg_mask = np.logical_or(
            np.abs(resp) > blend_curve_start,
            np.logical_not(np.isclose(splane_contour.real, 0)))

        scale_mask = ~reg_mask \
            & np.concatenate((~reg_mask[1:], ~reg_mask[-1:])) \
            & np.concatenate((~reg_mask[0:1], ~reg_mask[:-1]))

        # Rescale the points with large magnitude
        rescale_idx = (np.abs(resp) > blend_curve_start)

        if np.any(rescale_idx):  # Only process if rescaling is needed
            subset = resp[rescale_idx]
            abs_subset = np.abs(subset)
            unit_vectors = subset / abs_subset  # Preserve phase/direction

            if blend_curve_start == max_curve_magnitude:
                # Clip at max_curve_magnitude
                resp[rescale_idx] = max_curve_magnitude * unit_vectors
            else:
                # Logistic scaling
                newmag = blend_curve_start + \
                    (max_curve_magnitude - blend_curve_start) * \
                    (abs_subset - blend_curve_start) / \
                    (abs_subset + max_curve_magnitude - 2 * blend_curve_start)
                resp[rescale_idx] = newmag * unit_vectors

        # Get the label to use for the line
        label = response.sysname if line_labels is None else line_labels[idx]

        # Plot the regular portions of the curve (and grab the color)
        x_reg = np.ma.masked_where(reg_mask, resp.real)
        y_reg = np.ma.masked_where(reg_mask, resp.imag)
        p = ax.plot(
            x_reg, y_reg, primary_style[0], color=color, label=label, **kwargs)
        c = p[0].get_color()
        out[idx, 0] += p

        # Figure out how much to offset the curve: the offset goes from
        # zero at the start of the scaled section to max_curve_offset as
        # we move along the curve
        curve_offset = _compute_curve_offset(
            resp, scale_mask, max_curve_offset)

        # Plot the scaled sections of the curve (changing linestyle)
        x_scl = np.ma.masked_where(scale_mask, resp.real)
        y_scl = np.ma.masked_where(scale_mask, resp.imag)
        if x_scl.count() >= 1 and y_scl.count() >= 1:
            out[idx, 1] += ax.plot(
                x_scl * (1 + curve_offset),
                y_scl * (1 + curve_offset),
                primary_style[1], color=c, **kwargs)
        else:
            out[idx, 1] += [None]

        # Plot the primary curve (invisible) for setting arrows
        x, y = resp.real.copy(), resp.imag.copy()
        x[reg_mask] *= (1 + curve_offset[reg_mask])
        y[reg_mask] *= (1 + curve_offset[reg_mask])
        p = ax.plot(x, y, linestyle='None', color=c)

        # Add arrows
        _add_arrows_to_line2D(
            ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=1)

        # Plot the mirror image
        if mirror_style is not False:
            # Plot the regular and scaled segments
            out[idx, 2] += ax.plot(
                x_reg, -y_reg, mirror_style[0], color=c, **kwargs)
            if x_scl.count() >= 1 and y_scl.count() >= 1:
                out[idx, 3] += ax.plot(
                    x_scl * (1 - curve_offset),
                    -y_scl * (1 - curve_offset),
                    mirror_style[1], color=c, **kwargs)
            else:
                out[idx, 3] += [None]

            # Add the arrows (on top of an invisible contour)
            x, y = resp.real.copy(), resp.imag.copy()
            x[reg_mask] *= (1 - curve_offset[reg_mask])
            y[reg_mask] *= (1 - curve_offset[reg_mask])
            p = ax.plot(x, -y, linestyle='None', color=c, **kwargs)
            _add_arrows_to_line2D(
                ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=-1)
        else:
            out[idx, 2] += [None]
            out[idx, 3] += [None]

        # Mark the start of the curve
        if start_marker:
            segment = 0 if 0 in rescale_idx else 1      # regular vs scaled
            out[idx, segment] += ax.plot(
                resp[0].real, resp[0].imag, start_marker,
                color=c, markersize=start_marker_size)

        # Mark the -1 point
        ax.plot([-1], [0], 'r+')

        #
        # Draw circles for gain crossover and sensitivity functions
        #
        theta = np.linspace(0, 2*np.pi, 100)
        cos = np.cos(theta)
        sin = np.sin(theta)
        label_pos = 15

        # Display the unit circle, to read gain crossover frequency
        if unit_circle:
            ax.plot(cos, sin, **config.defaults['nyquist.circle_style'])

        # Draw circles for given magnitudes of sensitivity
        if ms_circles is not None:
            for ms in ms_circles:
                pos_x = -1 + (1/ms)*cos
                pos_y = (1/ms)*sin
                ax.plot(
                    pos_x, pos_y, **config.defaults['nyquist.circle_style'])
                ax.text(pos_x[label_pos], pos_y[label_pos], ms)

        # Draw circles for given magnitudes of complementary sensitivity
        if mt_circles is not None:
            for mt in mt_circles:
                if mt != 1:
                    ct = -mt**2/(mt**2-1)  # Mt center
                    rt = mt/(mt**2-1)  # Mt radius
                    pos_x = ct+rt*cos
                    pos_y = rt*sin
                    ax.plot(
                        pos_x, pos_y,
                        **config.defaults['nyquist.circle_style'])
                    ax.text(pos_x[label_pos], pos_y[label_pos], mt)
                else:
                    _, _, ymin, ymax = ax.axis()
                    pos_y = np.linspace(ymin, ymax, 100)
                    ax.vlines(
                        -0.5, ymin=ymin, ymax=ymax,
                        **config.defaults['nyquist.circle_style'])
                    ax.text(-0.5, pos_y[label_pos], 1)

        # Label the frequencies of the points on the Nyquist curve
        if label_freq:
            ind = slice(None, None, label_freq)
            omega_sys = np.imag(splane_contour[np.real(splane_contour) == 0])
            for xpt, ypt, omegapt in zip(x[ind], y[ind], omega_sys[ind]):
                # Convert to Hz
                f = omegapt / (2 * np.pi)

                # Factor out multiples of 1000 and limit the
                # result to the range [-8, 8].
                pow1000 = max(min(get_pow1000(f), 8), -8)

                # Get the SI prefix.
                prefix = gen_prefix(pow1000)

                # Apply the text. (Use a space before the text to
                # prevent overlap with the data.)
                #
                # np.round() is used because 0.99... appears
                # instead of 1.0, and this would otherwise be
                # truncated to 0.
                ax.text(xpt, ypt, ' ' +
                         str(int(np.round(f / 1000 ** pow1000, 0))) + ' ' +
                         prefix + 'Hz')

    # Label the axes
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")
    ax.grid(color="lightgray")

    # List of systems that are included in this plot
    lines, labels = _get_line_labels(ax)

    # Add legend if there is more than one system plotted
    if show_legend == True or (show_legend != False and len(labels) > 1):
        with plt.rc_context(rcParams):
            legend = ax.legend(lines, labels, loc=legend_loc)
    else:
        legend = None

    # Add the title
    sysnames = [response.sysname for response in nyquist_responses]
    if ax_user is None and title is None:
        title = "Nyquist plot for " + ", ".join(sysnames)
        _update_plot_title(
            title, fig=fig, rcParams=rcParams, frame=title_frame)
    elif ax_user is None:
        _update_plot_title(
            title, fig=fig, rcParams=rcParams, frame=title_frame,
            use_existing=False)

    # Legacy return processing
    if plot is True or return_contour is not None:
        if len(data) == 1:
            counts, contours = counts[0], contours[0]

        # Return counts and (optionally) the contour we used
        return (counts, contours) if return_contour else counts

    return ControlPlot(out, ax, fig, legend=legend)


#
# Function to compute Nyquist curve offsets
#
# This function computes a smoothly varying offset that starts and ends at
# zero at the ends of a scaled segment.
#
def _compute_curve_offset(resp, mask, max_offset):
    # Compute the arc length along the curve
    s_curve = np.cumsum(
        np.sqrt(np.diff(resp.real) ** 2 + np.diff(resp.imag) ** 2))

    # Initialize the offset
    offset = np.zeros(resp.size)
    arclen = np.zeros(resp.size)

    # Walk through the response and keep track of each continuous component
    i, nsegs = 0, 0
    while i < resp.size:
        # Skip the regular segment
        while i < resp.size and mask[i]:
            i += 1              # Increment the counter
            if i == resp.size:
                break
            # Keep track of the arclength
            arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

        nsegs += 0.5
        if i == resp.size:
            break

        # Save the starting offset of this segment
        seg_start = i

        # Walk through the scaled segment
        while i < resp.size and not mask[i]:
            i += 1
            if i == resp.size:  # See if we are done with this segment
                break
            # Keep track of the arclength
            arclen[i] = arclen[i-1] + np.abs(resp[i] - resp[i-1])

        nsegs += 0.5
        if i == resp.size:
            break

        # Save the ending offset of this segment
        seg_end = i

        # Now compute the scaling for this segment
        s_segment = arclen[seg_end-1] - arclen[seg_start]
        offset[seg_start:seg_end] = max_offset * s_segment/s_curve[-1] * \
            np.sin(np.pi * (arclen[seg_start:seg_end]
                            - arclen[seg_start])/s_segment)

    return offset


#
# Gang of Four plot
#
def gangof4_response(
        P, C, omega=None, omega_limits=None, omega_num=None, Hz=False):
    """Compute response of "Gang of 4" transfer functions.

    Generates a 2x2 frequency response for the "Gang of 4" sensitivity
    functions [T, PS; CS, S].

    Parameters
    ----------
    P, C : LTI
        Linear input/output systems (process and control).
    omega : array
        Range of frequencies (list or bounds) in rad/sec.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`. Ignored if
        data is not a list of systems.
    omega_num : int
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.  Ignored if data is
        not a list of systems.
    Hz : bool, optional
        If True, when computing frequency limits automatically set
        limits to full decades in Hz instead of rad/s.

    Returns
    -------
    response : `FrequencyResponseData`
        Frequency response with inputs 'r' and 'd' and outputs 'y', and 'u'
        representing the 2x2 matrix of transfer functions in the Gang of 4.

    Examples
    --------
    >>> P = ct.tf([1], [1, 1])
    >>> C = ct.tf([2], [1])
    >>> response = ct.gangof4_response(P, C)
    >>> cplt = response.plot()

    """
    if not P.issiso() or not C.issiso():
        # TODO: Add MIMO go4 plots.
        raise ControlMIMONotImplemented(
            "Gang of four is currently only implemented for SISO systems.")

    # Compute the sensitivity functions
    L = P * C
    S = feedback(1, L)
    T = L * S

    # Select a default range if none is provided
    # TODO: This needs to be made more intelligent
    omega, _ = _determine_omega_vector(
        [P, C, S], omega, omega_limits, omega_num, Hz=Hz)

    #
    # bode_plot based implementation
    #

    # Compute the response of the Gang of 4
    resp_T = T(1j * omega)
    resp_PS = (P * S)(1j * omega)
    resp_CS = (C * S)(1j * omega)
    resp_S = S(1j * omega)

    # Create a single frequency response data object with the underlying data
    data = np.empty((2, 2, omega.size), dtype=complex)
    data[0, 0, :] = resp_T
    data[0, 1, :] = resp_PS
    data[1, 0, :] = resp_CS
    data[1, 1, :] = resp_S

    return FrequencyResponseData(
        data, omega, outputs=['y', 'u'], inputs=['r', 'd'],
        title=f"Gang of Four for P={P.name}, C={C.name}",
        sysname=f"P={P.name}, C={C.name}", plot_phase=False)


def gangof4_plot(
        *args, omega=None, omega_limits=None, omega_num=None,
        Hz=False, **kwargs):
    """gangof4_plot(response) \
    gangof4_plot(P, C, omega)

    Plot response of "Gang of 4" transfer functions.

    Plots a 2x2 frequency response for the "Gang of 4" sensitivity
    functions [T, PS; CS, S].  Can be called in one of two ways:

        gangof4_plot(response[, ...])
        gangof4_plot(P, C[, ...])

    Parameters
    ----------
    response : FrequencyPlotData
        Gang of 4 frequency response from `gangof4_response`.
    P, C : LTI
        Linear input/output systems (process and control).
    omega : array
        Range of frequencies (list or bounds) in rad/sec.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`. Ignored if
        data is not a list of systems.
    omega_num : int
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.  Ignored if data is
        not a list of systems.
    Hz : bool, optional
        If True, when computing frequency limits automatically set
        limits to full decades in Hz instead of rad/s.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : 2x2 array of `matplotlib.lines.Line2D`
        Array containing information on each line in the plot.  The value
        of each array entry is a list of Line2D objects in that subplot.
    cplt.axes : 2D array of `matplotlib.axes.Axes`
        Axes for each subplot.
    cplt.figure : `matplotlib.figure.Figure`
        Figure containing the plot.
    cplt.legend : 2D array of `matplotlib.legend.Legend`
        Legend object(s) contained in the plot.

    """
    if len(args) == 1 and isinstance(args[0], FrequencyResponseData):
        if any([kw is not None
                for kw in [omega, omega_limits, omega_num, Hz]]):
            raise ValueError(
                "omega, omega_limits, omega_num, Hz not allowed when "
                "given a Gang of 4 response as first argument")
        return args[0].plot(kwargs)
    else:
        if len(args) > 3:
            raise TypeError(
                f"expecting 2 or 3 positional arguments; received {len(args)}")
        omega = omega if len(args) < 3 else args[2]
        args = args[0:2]
        return gangof4_response(
            *args, omega=omega, omega_limits=omega_limits,
            omega_num=omega_num, Hz=Hz).plot(**kwargs)


#
# Singular values plot
#
def singular_values_response(
        sysdata, omega=None, omega_limits=None, omega_num=None, Hz=False):
    """Singular value response for a system.

    Computes the singular values for a system or list of systems over
    a (optional) frequency range.

    Parameters
    ----------
    sysdata : LTI or list of LTI
        List of linear input/output systems (single system is OK).
    omega : array_like
        List of frequencies in rad/sec to be used for frequency response.
    Hz : bool, optional
        If True, when computing frequency limits automatically set
        limits to full decades in Hz instead of rad/s.

    Returns
    -------
    response : `FrequencyResponseData`
        Frequency response with the number of outputs equal to the
        number of singular values in the response, and a single input.

    Other Parameters
    ----------------
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`.
    omega_num : int, optional
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.

    See Also
    --------
    singular_values_plot

    Examples
    --------
    >>> omegas = np.logspace(-4, 1, 1000)
    >>> den = [75, 1]
    >>> G = ct.tf([[[87.8], [-86.4]], [[108.2], [-109.6]]],
    ...           [[den, den], [den, den]])
    >>> response = ct.singular_values_response(G, omega=omegas)

    """
    # Convert the first argument to a list
    syslist = sysdata if isinstance(sysdata, (list, tuple)) else [sysdata]

    if any([not isinstance(sys, LTI) for sys in syslist]):
        ValueError("singular values can only be computed for LTI systems")

    # Compute the frequency responses for the systems
    responses = frequency_response(
        syslist, omega=omega, omega_limits=omega_limits,
        omega_num=omega_num, Hz=Hz, squeeze=False)

    # Calculate the singular values for each system in the list
    svd_responses = []
    for response in responses:
        # Compute the singular values (permute indices to make things work)
        fresp_permuted = response.frdata.transpose((2, 0, 1))
        sigma = np.linalg.svd(fresp_permuted, compute_uv=False).transpose()
        sigma_fresp = sigma.reshape(sigma.shape[0], 1, sigma.shape[1])

        # Save the singular values as an FRD object
        svd_responses.append(
            FrequencyResponseData(
                sigma_fresp, response.omega, _return_singvals=True,
                outputs=[f'$\\sigma_{{{k+1}}}$' for k in range(sigma.shape[0])],
                inputs='inputs', dt=response.dt, plot_phase=False,
                sysname=response.sysname, plot_type='svplot',
                title=f"Singular values for {response.sysname}"))

    if isinstance(sysdata, (list, tuple)):
        return FrequencyResponseList(svd_responses)
    else:
        return svd_responses[0]


def singular_values_plot(
        data, omega=None, *fmt, plot=None, omega_limits=None, omega_num=None,
        ax=None, label=None, title=None, **kwargs):
    """Plot the singular values for a system.

    Plot the singular values as a function of frequency for a system or
    list of systems.  If multiple systems are plotted, each system in the
    list is plotted in a different color.

    Parameters
    ----------
    data : list of `FrequencyResponseData`
        List of `FrequencyResponseData` objects.  For backward
        compatibility, a list of LTI systems can also be given.
    omega : array_like
        List of frequencies in rad/sec over to plot over.
    *fmt : `matplotlib.pyplot.plot` format string, optional
        Passed to `matplotlib` as the format string for all lines in the plot.
        The `omega` parameter must be present (use omega=None if needed).
    dB : bool
        If True, plot result in dB.  Default is False.
    Hz : bool
        If True, plot frequency in Hz (omega must be provided in rad/sec).
        Default value (False) set by `config.defaults['freqplot.Hz']`.
    **kwargs : `matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords passed to `matplotlib` to specify line properties.

    Returns
    -------
    cplt : `ControlPlot` object
        Object containing the data that were plotted.  See `ControlPlot`
        for more detailed information.
    cplt.lines : array of `matplotlib.lines.Line2D`
        Array containing information on each line in the plot.  The size of
        the array matches the number of systems and the value of the array
        is a list of Line2D objects for that system.
    cplt.axes : 2D array of `matplotlib.axes.Axes`
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
    color : matplotlib color spec
        Color to use for singular values (or None for matplotlib default).
    grid : bool
        If True, plot grid lines on gain and phase plots.  Default is
        set by `config.defaults['freqplot.grid']`.
    label : str or array_like of str, optional
        If present, replace automatically generated label(s) with the given
        label(s).  If sysdata is a list, strings should be specified for each
        system.
    legend_loc : int or str, optional
        Include a legend in the given location. Default is 'center right',
        with no legend for a single response.  Use False to suppress legend.
    omega_limits : array_like of two values
        Set limits for plotted frequency range. If Hz=True the limits are
        in Hz otherwise in rad/s.  Specifying `omega` as a list of two
        elements is equivalent to providing `omega_limits`.
    omega_num : int, optional
        Number of samples to use for the frequency range.  Defaults to
        `config.defaults['freqplot.number_of_samples']`.  Ignored if data is
        not a list of systems.
    plot : bool, optional
        (legacy) If given, `singular_values_plot` returns the legacy return
        values of magnitude, phase, and frequency.  If False, just return
        the values with no plot.
    rcParams : dict
        Override the default parameters used for generating plots.
        Default is set up `config.defaults['ctrlplot.rcParams']`.
    show_legend : bool, optional
        Force legend to be shown if True or hidden if False.  If
        None, then show legend when there is more than one line on an
        axis or `legend_loc` or `legend_map` has been specified.
    title : str, optional
        Set the title of the plot.  Defaults to plot type and system name(s).
    title_frame : str, optional
        Set the frame of reference used to center the plot title. If set to
        'axes' (default), the horizontal position of the title will
        centered relative to the axes.  If set to 'figure', it will be
        centered with respect to the figure (faster execution).

    See Also
    --------
    singular_values_response

    Notes
    -----
    If `plot` = False, the following legacy values are returned:
       * `mag` : ndarray (or list of ndarray if len(data) > 1))
           Magnitude of the response (deprecated).
       * `phase` : ndarray (or list of ndarray if len(data) > 1))
           Phase in radians of the response (deprecated).
       * `omega` : ndarray (or list of ndarray if len(data) > 1))
           Frequency in rad/sec (deprecated).

    """
    # Keyword processing
    color = kwargs.pop('color', None)
    dB = config._get_param(
        'freqplot', 'dB', kwargs, _freqplot_defaults, pop=True)
    Hz = config._get_param(
        'freqplot', 'Hz', kwargs, _freqplot_defaults, pop=True)
    grid = config._get_param(
        'freqplot', 'grid', kwargs, _freqplot_defaults, pop=True)
    rcParams = config._get_param('ctrlplot', 'rcParams', kwargs, pop=True)
    title_frame = config._get_param(
        'freqplot', 'title_frame', kwargs, _freqplot_defaults, pop=True)

    # If argument was a singleton, turn it into a tuple
    data = data if isinstance(data, (list, tuple)) else (data,)

    # Convert systems into frequency responses
    if any([isinstance(response, (StateSpace, TransferFunction))
            for response in data]):
        responses = singular_values_response(
                    data, omega=omega, omega_limits=omega_limits,
                    omega_num=omega_num)
    else:
        # Generate warnings if frequency keywords were given
        if omega_num is not None:
            warnings.warn("`omega_num` ignored when passed response data")
        elif omega is not None:
            warnings.warn("`omega` ignored when passed response data")

        # Check to make sure omega_limits is sensible
        if omega_limits is not None and \
           (len(omega_limits) != 2 or omega_limits[1] <= omega_limits[0]):
            raise ValueError(f"invalid limits: {omega_limits=}")

        responses = data

    # Process label keyword
    line_labels = _process_line_labels(label, len(data))

    # Process (legacy) plot keyword
    if plot is not None:
        warnings.warn(
            "`singular_values_plot` return values of sigma, omega is "
            "deprecated; use singular_values_response()", FutureWarning)

    # Warn the user if we got past something that is not real-valued
    if any([not np.allclose(np.imag(response.frdata[:, 0, :]), 0)
            for response in responses]):
        warnings.warn("data has non-zero imaginary component")

    # Extract the data we need for plotting
    sigmas = [np.real(response.frdata[:, 0, :]) for response in responses]
    omegas = [response.omega for response in responses]

    # Legacy processing for no plotting case
    if plot is False:
        if len(data) == 1:
            return sigmas[0], omegas[0]
        else:
            return sigmas, omegas

    fig, ax_sigma = _process_ax_keyword(
        ax, shape=(1, 1), squeeze=True, rcParams=rcParams)
    ax_sigma.set_label('control-sigma')         # TODO: deprecate?
    legend_loc, _, show_legend = _process_legend_keywords(
        kwargs, None, 'center right')

    # Get color offset for first (new) line to be drawn
    color_offset, color_cycle = _get_color_offset(ax_sigma)

    # Create a list of lines for the output
    out = np.empty(len(data), dtype=object)

    # Plot the singular values for each response
    for idx_sys, response in enumerate(responses):
        sigma = sigmas[idx_sys].transpose()     # frequency first for plotting
        omega = omegas[idx_sys] / (2 * math.pi) if Hz else  omegas[idx_sys]

        if response.isdtime(strict=True):
            nyq_freq = (0.5/response.dt) if Hz else (math.pi/response.dt)
        else:
            nyq_freq = None

        # Determine the color to use for this response
        current_color = _get_color(
            color, fmt=fmt, offset=color_offset + idx_sys,
            color_cycle=color_cycle)

        # To avoid conflict with *fmt, only pass color kw if non-None
        color_arg = {} if current_color is None else {'color': current_color}

        # Decide on the system name
        sysname = response.sysname if response.sysname is not None \
            else f"Unknown-{idx_sys}"

        # Get the label to use for the line
        label = sysname if line_labels is None else line_labels[idx_sys]

        # Plot the data
        if dB:
            out[idx_sys] = ax_sigma.semilogx(
                omega, 20 * np.log10(sigma), *fmt,
                label=label, **color_arg, **kwargs)
        else:
            out[idx_sys] = ax_sigma.loglog(
                omega, sigma, label=label, *fmt, **color_arg, **kwargs)

        # Plot the Nyquist frequency
        if nyq_freq is not None:
            ax_sigma.axvline(
                nyq_freq, linestyle='--', label='_nyq_freq_' + sysname,
                **color_arg)

    # If specific omega_limits were given, use them
    if omega_limits is not None:
        ax_sigma.set_xlim(omega_limits)

    # Add a grid to the plot + labeling
    if grid:
        ax_sigma.grid(grid, which='both')

    ax_sigma.set_ylabel(
        "Singular Values [dB]" if dB else "Singular Values")
    ax_sigma.set_xlabel("Frequency [Hz]" if Hz else "Frequency [rad/sec]")

    # List of systems that are included in this plot
    lines, labels = _get_line_labels(ax_sigma)

    # Add legend if there is more than one system plotted
    if show_legend == True or (show_legend != False and len(labels) > 1):
        with plt.rc_context(rcParams):
            legend = ax_sigma.legend(lines, labels, loc=legend_loc)
    else:
        legend = None

    # Add the title
    if ax is None:
        if title is None:
            title = "Singular values for " + ", ".join(labels)
        _update_plot_title(
            title, fig=fig, rcParams=rcParams, frame=title_frame,
            use_existing=False)

    # Legacy return processing
    if plot is not None:
        if len(responses) == 1:
            return sigmas[0], omegas[0]
        else:
            return sigmas, omegas

    return ControlPlot(out, ax_sigma, fig, legend=legend)

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating frequency domain plots.
#


# Determine the frequency range to be used
def _determine_omega_vector(syslist, omega_in, omega_limits, omega_num,
                            Hz=None, feature_periphery_decades=None):
    """Determine the frequency range for a frequency-domain plot
    according to a standard logic.

    If `omega_in` and `omega_limits` are both None, then `omega_out` is
    computed on `omega_num` points according to a default logic defined by
    `_default_frequency_range` and tailored for the list of systems
    syslist, and `omega_range_given` is set to False.

    If `omega_in` is None but `omega_limits` is a tuple of 2 elements, then
    `omega_out` is computed with the function `numpy.logspace` on
    `omega_num` points within the interval ``[min, max] = [omega_limits[0],
    omega_limits[1]]``, and `omega_range_given` is set to True.

    If `omega_in` is a tuple of length 2, it is interpreted as a range and
    handled like `omega_limits`.  If `omega_in` is a tuple of length 3, it
    is interpreted a range plus number of points and handled like
    `omega_limits` and `omega_num`.

    If `omega_in` is an array or a list/tuple of length greater than two,
    then `omega_out` is set to `omega_in` (as an array), and
    `omega_range_given` is set to True

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK).
    omega_in : 1D array_like or None
        Frequency range specified by the user.
    omega_limits : 1D array_like or None
        Frequency limits specified by the user.
    omega_num : int
        Number of points to be used for the frequency range (if the
        frequency range is not user-specified).
    Hz : bool, optional
        If True, the limits (first and last value) of the frequencies
        are set to full decades in Hz so it fits plotting with logarithmic
        scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.

    Returns
    -------
    omega_out : 1D array
        Frequency range to be used.
    omega_range_given : bool
        True if the frequency range was specified by the user, either through
        omega_in or through omega_limits. False if both omega_in
        and omega_limits are None.

    """
    # Handle the special case of a range of frequencies
    if omega_in is not None and omega_limits is not None:
        warnings.warn(
            "omega and omega_limits both specified; ignoring limits")
    elif isinstance(omega_in, (list, tuple)) and len(omega_in) == 2:
        omega_limits = omega_in
        omega_in = None

    omega_range_given = True
    if omega_in is None:
        if omega_limits is None:
            omega_range_given = False
            # Select a default range if none is provided
            omega_out = _default_frequency_range(
                syslist, number_of_samples=omega_num, Hz=Hz,
                feature_periphery_decades=feature_periphery_decades)
        else:
            omega_limits = np.asarray(omega_limits)
            if len(omega_limits) != 2:
                raise ValueError("len(omega_limits) must be 2")
            omega_out = np.logspace(np.log10(omega_limits[0]),
                                    np.log10(omega_limits[1]),
                                    num=omega_num, endpoint=True)
    else:
        omega_out = np.copy(omega_in)

    return omega_out, omega_range_given


# Compute reasonable defaults for axes
def _default_frequency_range(syslist, Hz=None, number_of_samples=None,
                             feature_periphery_decades=None):
    """Compute a default frequency range for frequency domain plots.

    This code looks at the poles and zeros of all of the systems that
    we are plotting and sets the frequency range to be one decade above
    and below the min and max feature frequencies, rounded to the nearest
    integer.  If no features are found, it returns logspace(-1, 1)

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    Hz : bool, optional
        If True, the limits (first and last value) of the frequencies
        are set to full decades in Hz so it fits plotting with logarithmic
        scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.
    number_of_samples : int, optional
        Number of samples to generate.  The default value is read from
        `config.defaults['freqplot.number_of_samples']`.  If None,
        then the default from `numpy.logspace` is used.
    feature_periphery_decades : float, optional
        Defines how many decades shall be included in the frequency range on
        both sides of features (poles, zeros).  The default value is read from
        `config.defaults['freqplot.feature_periphery_decades']`.

    Returns
    -------
    omega : array
        Range of frequencies in rad/sec

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> omega = ct._default_frequency_range(G)
    >>> omega.min(), omega.max()
    (0.1, 100.0)

    """
    # Set default values for options
    number_of_samples = config._get_param(
        'freqplot', 'number_of_samples', number_of_samples)
    feature_periphery_decades = config._get_param(
        'freqplot', 'feature_periphery_decades', feature_periphery_decades, 1)

    # Find the list of all poles and zeros in the systems
    features = np.array(())
    freq_interesting = []

    # detect if single sys passed by checking if it is sequence-like
    if not hasattr(syslist, '__iter__'):
        syslist = (syslist,)

    for sys in syslist:
        # For FRD systems, just use the response frequencies
        if isinstance(sys, FrequencyResponseData):
            # Add the min and max frequency, minus periphery decades
            # (keeps frequency ranges from artificially expanding)
            features = np.concatenate([features, np.array([
                np.min(sys.omega) * 10**feature_periphery_decades,
                np.max(sys.omega) / 10**feature_periphery_decades])])
            continue

        try:
            # Add new features to the list
            if sys.isctime():
                features_ = np.concatenate(
                    (np.abs(sys.poles()), np.abs(sys.zeros())))
                # Get rid of poles and zeros at the origin
                toreplace = np.isclose(features_, 0.0)
                if np.any(toreplace):
                    features_ = features_[~toreplace]
            elif sys.isdtime(strict=True):
                fn = math.pi / sys.dt
                # TODO: What distance to the Nyquist frequency is appropriate?
                freq_interesting.append(fn * 0.9)

                features_ = np.concatenate(
                    (np.abs(sys.poles()), np.abs(sys.zeros())))
                # Get rid of poles and zeros on the real axis (imag==0)
                # * origin and real < 0
                # * at 1.: would result in omega=0. (logarithmic plot!)
                toreplace = np.isclose(features_.imag, 0.0) & (
                                    (features_.real <= 0.) |
                                    (np.abs(features_.real - 1.0) < 1.e-10))
                if np.any(toreplace):
                    features_ = features_[~toreplace]
                # TODO: improve (mapping pack to continuous time)
                features_ = np.abs(np.log(features_) / (1.j * sys.dt))
            else:
                # TODO
                raise NotImplementedError(
                    "type of system in not implemented now")
            features = np.concatenate([features, features_])
        except NotImplementedError:
            # Don't add any features for anything we don't understand
            pass

    # Make sure there is at least one point in the range
    if features.shape[0] == 0:
        features = np.array([1.])

    if Hz:
        features /= 2. * math.pi
    features = np.log10(features)
    lsp_min = np.rint(np.min(features) - feature_periphery_decades)
    lsp_max = np.rint(np.max(features) + feature_periphery_decades)
    if Hz:
        lsp_min += np.log10(2. * math.pi)
        lsp_max += np.log10(2. * math.pi)

    if freq_interesting:
        lsp_min = min(lsp_min, np.log10(min(freq_interesting)))
        lsp_max = max(lsp_max, np.log10(max(freq_interesting)))

    # TODO: Add a check in discrete case to make sure we don't get aliasing
    # (Attention: there is a list of system but only one omega vector)

    # Set the range to be an order of magnitude beyond any features
    if number_of_samples:
        omega = np.logspace(
            lsp_min, lsp_max, num=number_of_samples, endpoint=True)
    else:
        omega = np.logspace(lsp_min, lsp_max, endpoint=True)
    return omega


#
# Utility functions to create nice looking labels (KLD 5/23/11)
#

def get_pow1000(num):
    """Determine exponent for which significance of a number is within the
    range [1, 1000).
    """
    # Based on algorithm from http://www.mail-archive.com/
    # matplotlib-users@lists.sourceforge.net/msg14433.html, accessed 2010/11/7
    # by Jason Heeris 2009/11/18
    from decimal import Decimal
    from math import floor
    dnum = Decimal(str(num))
    if dnum == 0:
        return 0
    elif dnum < 0:
        dnum = -dnum
    return int(floor(dnum.log10() / 3))


def gen_prefix(pow1000):
    """Return the SI prefix for a power of 1000.
    """
    # Prefixes according to Table 5 of [BIPM 2006] (excluding hecto,
    # deca, deci, and centi).
    if pow1000 < -8 or pow1000 > 8:
        raise ValueError(
            "Value is out of the range covered by the SI prefixes.")
    return ['Y',  # yotta (10^24)
            'Z',  # zetta (10^21)
            'E',  # exa (10^18)
            'P',  # peta (10^15)
            'T',  # tera (10^12)
            'G',  # giga (10^9)
            'M',  # mega (10^6)
            'k',  # kilo (10^3)
            '',  # (10^0)
            'm',  # milli (10^-3)
            r'$\mu$',  # micro (10^-6)
            'n',  # nano (10^-9)
            'p',  # pico (10^-12)
            'f',  # femto (10^-15)
            'a',  # atto (10^-18)
            'z',  # zepto (10^-21)
            'y'][8 - pow1000]  # yocto (10^-24)


# Function aliases
bode = bode_plot
nyquist = nyquist_plot
gangof4 = gangof4_plot
