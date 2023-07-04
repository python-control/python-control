# freqplot.py - frequency domain plots for control systems
#
# Author: Richard M. Murray
# Date: 24 May 09
#
# Functionality to add
# [ ] Get rid of this long header (need some common, documented convention)
# [ ] Add mechanisms for storing/plotting margins? (currently forces FRD)
# [ ] Allow line colors/styles to be set in plot() command (also time plots)
# [ ] Allow bode or nyquist style plots from plot()
# [ ] Allow nyquist_curve() to generate the response curve (?)
# [ ] Allow MIMO frequency plots (w/ mag/phase subplots a la MATLAB)
# [ ] Update sisotool to use ax=
# [ ] Create __main__ in freqplot_test to view results (a la timeplot_test)
# [ ] Get sisotool working in iPython and document how to make it work

#
# This file contains some standard control system plots: Bode plots,
# Nyquist plots and pole-zero diagrams.  The code for Nichols charts
# is in nichols.py.
#
# Copyright (c) 2010 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$

# TODO: clean up imports
import math
from os.path import commonprefix

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings
from math import nan
import itertools

from .ctrlutil import unwrap
from .bdalg import feedback
from .margins import stability_margins
from .exception import ControlMIMONotImplemented
from .statesp import StateSpace
from .lti import frequency_response, _process_frequency_response
from .xferfcn import TransferFunction
from .frdata import FrequencyResponseData
from .timeplot import _make_legend_labels
from . import config

__all__ = ['bode_plot', 'nyquist_plot', 'gangof4_plot', 'singular_values_plot',
           'bode', 'nyquist', 'gangof4']

# Default font dictionary
_freqplot_rcParams = mpl.rcParams.copy()
_freqplot_rcParams.update({
    'axes.labelsize': 'small',
    'axes.titlesize': 'small',
    'figure.titlesize': 'medium',
    'legend.fontsize': 'x-small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
})

# Default values for module parameter variables
_freqplot_defaults = {
    'freqplot.rcParams': _freqplot_rcParams,
    'freqplot.feature_periphery_decades': 1,
    'freqplot.number_of_samples': 1000,
    'freqplot.dB': False,  # Plot gain in dB
    'freqplot.deg': True,  # Plot phase in degrees
    'freqplot.Hz': False,  # Plot frequency in Hertz
    'freqplot.grid': True,  # Turn on grid for gain and phase
    'freqplot.wrap_phase': False,  # Wrap the phase plot at a given value
    'freqplot.freq_label': "Frequency [%s]",

    # deprecations
    'deprecated.bode.dB': 'freqplot.dB',
    'deprecated.bode.deg': 'freqplot.deg',
    'deprecated.bode.Hz': 'freqplot.Hz',
    'deprecated.bode.grid': 'freqplot.grid',
    'deprecated.bode.wrap_phase': 'freqplot.wrap_phase',
}


#
# Main plotting functions
#
# This section of the code contains the functions for generating
# frequency domain plots
#

#
# Bode plot
#

def bode_plot(
        data, omega=None, *fmt, ax=None, omega_limits=None, omega_num=None,
        plot=None, plot_magnitude=True, plot_phase=True, margins=None,
        margin_info=False, method='best', legend_map=None, legend_loc=None,
        title=None, relabel=True, **kwargs):
    """Bode plot for a system.

    Bode plot of a frequency response over a (optional) frequency range.

    Parameters
    ----------
    data : list of `FrequencyResponseData`
        List of :class:`FrequencyResponseData` objects.  For backward
        compatibility, a list of LTI systems can also be given.
    omega : array_like
        List of frequencies in rad/sec over to plot over.
    dB : bool
        If True, plot result in dB.  Default is False.
    Hz : bool
        If True, plot frequency in Hz (omega must be provided in rad/sec).
        Default value (False) set by config.defaults['freqplot.Hz'].
    deg : bool
        If True, plot phase in degrees (else radians).  Default value (True)
        set by config.defaults['freqplot.deg'].
    margins : bool
        If True, plot gain and phase margin.  (TODO: merge with margin_info)
    margin_info : bool
        If True, plot information about gain and phase margin.
    method : str, optional
        Method to use in computing margins (see :func:`stability_margins`).
    *fmt : :func:`matplotlib.pyplot.plot` format string, optional
        Passed to `matplotlib` as the format string for all lines in the plot.
    **kwargs : :func:`matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords passed to `matplotlib` to specify line properties.

    Returns
    -------
    out : array of Line2D
        Array of Line2D objects for each line in the plot.  The shape of
        the array matches the subplots shape and the value of the array is a
        list of Line2D objects in that subplot.
    mag : ndarray (or list of ndarray if len(data) > 1))
        If plot=False, magnitude of the respone (deprecated).
    phase : ndarray (or list of ndarray if len(data) > 1))
        If plot=False, phase in radians of the respone (deprecated).
    omega : ndarray (or list of ndarray if len(data) > 1))
        If plot=False, frequency in rad/sec (deprecated).

    Other Parameters
    ----------------
    plot : bool
        If True (default), plot magnitude and phase.
    omega_limits : array_like of two values
        Limits of the to generate frequency vector. If Hz=True the limits
        are in Hz otherwise in rad/s.
    omega_num : int
        Number of samples to plot.  Defaults to
        config.defaults['freqplot.number_of_samples'].
    grid : bool
        If True, plot grid lines on gain and phase plots.  Default is set by
        `config.defaults['freqplot.grid']`.
    initial_phase : float
        Set the reference phase to use for the lowest frequency.  If set, the
        initial phase of the Bode plot will be set to the value closest to the
        value specified.  Units are in either degrees or radians, depending on
        the `deg` parameter. Default is -180 if wrap_phase is False, 0 if
        wrap_phase is True.
    wrap_phase : bool or float
        If wrap_phase is `False` (default), then the phase will be unwrapped
        so that it is continuously increasing or decreasing.  If wrap_phase is
        `True` the phase will be restricted to the range [-180, 180) (or
        [:math:`-\\pi`, :math:`\\pi`) radians). If `wrap_phase` is specified
        as a float, the phase will be offset by 360 degrees if it falls below
        the specified value. Default value is `False` and can be set using
        config.defaults['freqplot.wrap_phase'].

    The default values for Bode plot configuration parameters can be reset
    using the `config.defaults` dictionary, with module name 'bode'.

    Notes
    -----
    1. Alternatively, you may use the lower-level methods
       :meth:`LTI.frequency_response` or ``sys(s)`` or ``sys(z)`` or to
       generate the frequency response for a single system.

    2. If a discrete time model is given, the frequency response is plotted
       along the upper branch of the unit circle, using the mapping ``z =
       exp(1j * omega * dt)`` where `omega` ranges from 0 to `pi/dt` and `dt`
       is the discrete timebase.  If timebase not specified (``dt=True``),
       `dt` is set to 1.

    3. The legacy version of this function is invoked if instead of passing
       frequency response data, a system (or list of systems) is passed as
       the first argument, or if the (deprecated) keyword `plot` is set to
       True or False. The return value is then given as `mag`, `phase`,
       `omega` for the plotted frequency response (SISO only).

    Examples
    --------
    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
    >>> Gmag, Gphase, Gomega = ct.bode_plot(G)

    """
    #
    # Process keywords and set defaults
    #

    # Make a copy of the kwargs dictionary since we will modify it
    kwargs = dict(kwargs)

    # Get values for params (and pop from list to allow keyword use in plot)
    dB = config._get_param(
        'freqplot', 'dB', kwargs, _freqplot_defaults, pop=True)
    deg = config._get_param(
        'freqplot', 'deg', kwargs, _freqplot_defaults, pop=True)
    Hz = config._get_param(
        'freqplot', 'Hz', kwargs, _freqplot_defaults, pop=True)
    grid = config._get_param(
        'freqplot', 'grid', kwargs, _freqplot_defaults, pop=True)
    plot = config._get_param('freqplot', 'plot', plot, True)
    margins = config._get_param(
        'freqplot', 'margins', margins, False)
    wrap_phase = config._get_param(
        'freqplot', 'wrap_phase', kwargs, _freqplot_defaults, pop=True)
    initial_phase = config._get_param(
        'freqplot', 'initial_phase', kwargs, None, pop=True)
    omega_num = config._get_param('freqplot', 'number_of_samples', omega_num)
    freqplot_rcParams = config._get_param(
        'freqplot', 'rcParams', kwargs, _freqplot_defaults, pop=True)
    freq_label = config._get_param(
        'freqplot', 'freq_label', kwargs, _freqplot_defaults, pop=True)

    if not isinstance(data, (list, tuple)):
        data = [data]

    # For backwards compatibility, allow systems in the data list
    if all([isinstance(
            sys, (StateSpace, TransferFunction)) for sys in data]):
        data = frequency_response(
            data, omega=omega, omega_limits=omega_limits,
            omega_num=omega_num)
        warnings.warn(
            "passing systems to `bode_plot` is deprecated; "
            "use `frequency_response()`", DeprecationWarning)
        if plot is None:
            plot = True         # Keep track of legacy usage (see notes below)

    #
    # Process the data to be plotted
    #
    # To maintain compatibility with legacy uses of bode_plot(), we do some
    # initial processing on the data, specifically phase unwrapping and
    # setting the initial value of the phase.  If bode_plot is called with
    # plot == False, then these values are returned to the user (instead of
    # the list of lines created, which is the new output for _plot functions.
    #
    # TODO: update to match timpelot inputs/outputs structure

    if not plot_magnitude and not plot_phase:
        raise ValueError(
            "plot_magnitude and plot_phase both False; no data to plot")

    mags, phases, omegas, nyquistfrqs = [], [], [], []
    sysnames = []
    for response in data:
        mag, phase, omega_sys = response
        mag = np.atleast_1d(mag)
        phase = np.atleast_1d(phase)
        # mag, phase = response.magnitude, response.phase       # TODO: use this
        noutputs, ninputs = response.noutputs, response.ninputs
        omega_sys = response.omega
        nyquistfrq = None if response.isctime() else math.pi / response.dt

        ###
        ### Code below can go into plotting section, but may need to
        ### duplicate in frequency_response() ??
        ###

        #
        # Post-process the phase to handle initial value and wrapping
        #

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

        # TODO: hack to make sure that SISO case still works properly
        my_phase = phase.reshape((noutputs, ninputs, -1))  # TODO: remove
        for i, j in itertools.product(range(noutputs), range(ninputs)):
            # Shift the phase if needed
            if abs(my_phase[i, j, 0] - initial_phase_value) > math.pi:
                my_phase[i, j] -= 2*math.pi * round(
                    (my_phase[i, j, 0] - initial_phase_value) / (2*math.pi))

            # Phase wrapping
            if wrap_phase is False:
                my_phase[i, j] = unwrap(my_phase[i, j]) # unwrap the phase
            elif wrap_phase is True:
                pass                                    # default calc OK
            elif isinstance(wrap_phase, (int, float)):
                my_phase[i, j] = unwrap(my_phase[i, j]) # unwrap phase first
                if deg:
                    wrap_phase *= math.pi/180.

                # Shift the phase if it is below the wrap_phase
                my_phase[i, j] += 2*math.pi * np.maximum(
                    0, np.ceil((wrap_phase - my_phase[i, j])/(2*math.pi)))
            else:
                raise ValueError("wrap_phase must be bool or float.")
        phase = my_phase.reshape(phase.shape)           # TODO: remove

        mags.append(mag)
        phases.append(phase)
        omegas.append(omega_sys)
        nyquistfrqs.append(nyquistfrq)

        # Save the system names for later
        sysnames.append(response.sysname)

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
    # * plot == True: either set explicitly by the user or we were passed a
    #   non-FRD system instead of data.  Return mag, phase, omega, with a
    #   warning.
    #
    # * plot == False: set explicitly by the user. Return mag, phase,
    #   omega, with a warning.
    #
    # * plot == None: this is the new default setting and if it hasn't been
    #   changed, then we use the v0.10+ standard of returning an array of
    #   lines that were drawn.
    #
    # The one case that can cause problems is that a user called
    # `bode_plot` with an FRD system, didn't set the plot keyword
    # explicitly, and expected mag, phase, omega as a return value.  This
    # is hopefully a rare case (it wasn't in any of our unit tests nor
    # examples at the time of v0.10.0).
    #
    # All of this should be removed in v0.11+ when we get rid of deprecated
    # code.
    #

    if plot is True or plot is False:
        warnings.warn(
            "`bode_plot` return values of mag, phase, omega is deprecated; "
            "use frequency_response()", DeprecationWarning)

    if plot is False:
        if len(data) == 1:
            return mags[0], phases[0], omegas[0]
        else:
            return mags, phases, omegas
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

    # Decide on the number of inputs and outputs
    ninputs, noutputs = 0, 0
    for response in data:       # TODO: make more pythonic/numpic
        ninputs = max(ninputs, response.ninputs)
        noutputs = max(noutputs, response.noutputs)
    ntraces = 1                 # TODO: assume 1 trace per response for now

    # Figure how how many rows and columns to use + offsets for inputs/outputs
    nrows = (noutputs if plot_magnitude else 0) + \
        (noutputs if plot_phase else 0)
    ncols = ninputs

    # See if we can use the current figure axes
    fig = plt.gcf()         # get current figure (or create new one)
    if ax is None and plt.get_fignums():
        ax = fig.get_axes()
        if len(ax) == nrows * ncols:
            # Assume that the shape is right (no easy way to infer this)
            ax = np.array(ax).reshape(nrows, ncols)
        elif len(ax) != 0:
            # Need to generate a new figure
            fig, ax = plt.figure(), None
        else:
            # Blank figure, just need to recreate axes
            ax = None

    # Clear out any old text from the current figure
    for text in fig.texts:
        text.set_visible(False)         # turn off the text
        del text                        # get rid of it completely

    # Create new axes, if needed, and customize them
    if ax is None:
        with plt.rc_context(_freqplot_rcParams):
            ax_array = fig.subplots(
                nrows, ncols, sharex='col', sharey='row', squeeze=False)
            fig.set_tight_layout(True)
            fig.align_labels()

    else:
        # Make sure the axes are the right shape
        if ax.shape != (nrows, ncols):
            raise ValueError(
                "specified axes are not the right shape; "
                f"got {ax.shape} but expecting ({nrows}, {ncols})")
        ax_array = ax

    #
    # Plot the data
    #
    # The ax_magnitude and ax_phase arrays have the axes needed for making the
    # plots.  Labels are used on each axes for later creation of legends.
    # The generic labels if of the form:
    #
    #     To output label, From input label, system name
    #
    # The input and output labels are omitted if overlay_inputs or
    # overlay_outputs is False, respectively.  The system name is always
    # included, since multiple calls to plot() will require a legend that
    # distinguishes which system signals are plotted.  The system name is
    # stripped off later (in the legend-handling code) if it is not needed.
    #

    for mag, phase, omega_sys, nyquistfrq, sysname in \
        zip(mags, phases, omegas, nyquistfrqs, sysnames):

        # TODO: hack to handle MIMO while not breaking SISO
        my_mag = mag.reshape((noutputs, ninputs, -1))
        my_phase = phase.reshape((noutputs, ninputs, -1))
        for i, j in itertools.product(
                range(my_mag.shape[0]), range(my_mag.shape[1])):
            if plot_magnitude:
                ax_mag = ax_array[i*2 if plot_phase else i, j]
            if plot_phase:
                ax_phase = ax_array[i*2 + 1 if plot_magnitude else i, j]

            nyquistfrq_plot = None
            if Hz:
                omega_plot = omega_sys / (2. * math.pi)
                if nyquistfrq:
                    nyquistfrq_plot = nyquistfrq / (2. * math.pi)
            else:
                omega_plot = omega_sys
                if nyquistfrq:
                    nyquistfrq_plot = nyquistfrq

            phase_plot = my_phase[i, j] * 180. / math.pi if deg \
                else my_phase[i, j]
            mag_plot = my_mag[i, j]

            if nyquistfrq_plot:
                # append data for vertical nyquist freq indicator line.
                # if this extra nyquist line is is plotted in a single plot
                # command then line order is preserved when
                # creating a legend eg. legend(('sys1', 'sys2'))
                omega_nyq_line = np.array(
                    (np.nan, nyquistfrq_plot, nyquistfrq_plot))
                omega_plot = np.hstack((omega_plot, omega_nyq_line))
                mag_nyq_line = np.array((
                    np.nan, 0.7*min(mag_plot), 1.3*max(mag_plot)))
                mag_plot = np.hstack((mag_plot, mag_nyq_line))
                phase_range = max(phase_plot) - min(phase_plot)
                phase_nyq_line = np.array(
                    (np.nan,
                     min(phase_plot) - 0.2 * phase_range,
                     max(phase_plot) + 0.2 * phase_range))
                phase_plot = np.hstack((phase_plot, phase_nyq_line))

            # Magnitude
            if plot_magnitude:
                if dB:
                    ax_mag.semilogx(
                        omega_plot, 20 * np.log10(mag_plot), *fmt,
                        label=sysname, **kwargs)
                else:
                    ax_mag.loglog(
                        omega_plot, mag_plot, *fmt, label=sysname, **kwargs)

                # Add a grid to the plot + labeling
                ax_mag.grid(grid and not margins, which='both')

            # Phase
            if plot_phase:
                ax_phase.semilogx(
                    omega_plot, phase_plot, *fmt, label=sysname, **kwargs)

            #
            # Plot gain and phase margins
            #

            # Show the phase and gain margins in the plot
            if margins:
                # Compute stability margins for the system
                margin = stability_margins(response, method=method)
                gm, pm, Wcg, Wcp = (margin[i] for i in [0, 1, 3, 4])

                # Figure out sign of the phase at the first gain crossing
                # (needed if phase_wrap is True)
                phase_at_cp = phases[0][(np.abs(omegas[0] - Wcp)).argmin()]
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
                    mag_ylim = ax_mag.get_ylim()

                if plot_phase:
                    ax_phase.axhline(y=phase_limit if deg else
                                     math.radians(phase_limit),
                                     color='k', linestyle=':', zorder=-20)
                    phase_ylim = ax_phase.get_ylim()

                # Annotate the phase margin (if it exists)
                if plot_phase and pm != float('inf') and Wcp != float('nan'):
                    if dB:
                        ax_mag.semilogx(
                            [Wcp, Wcp], [0., -1e5],
                            color='k', linestyle=':', zorder=-20)
                    else:
                        ax_mag.loglog(
                            [Wcp, Wcp], [1., 1e-8],
                            color='k', linestyle=':', zorder=-20)

                    if deg:
                        ax_phase.semilogx(
                            [Wcp, Wcp], [1e5, phase_limit + pm],
                            color='k', linestyle=':', zorder=-20)
                        ax_phase.semilogx(
                            [Wcp, Wcp], [phase_limit + pm, phase_limit],
                            color='k', zorder=-20)
                    else:
                        ax_phase.semilogx(
                            [Wcp, Wcp], [1e5, math.radians(phase_limit) +
                                         math.radians(pm)],
                            color='k', linestyle=':', zorder=-20)
                        ax_phase.semilogx(
                            [Wcp, Wcp], [math.radians(phase_limit) +
                                         math.radians(pm),
                                         math.radians(phase_limit)],
                            color='k', zorder=-20)

                    ax_phase.set_ylim(phase_ylim)

                # Annotate the gain margin (if it exists)
                if plot_magnitude and gm != float('inf') and \
                   Wcg != float('nan'):
                    if dB:
                        ax_mag.semilogx(
                            [Wcg, Wcg], [-20.*np.log10(gm), -1e5],
                            color='k', linestyle=':', zorder=-20)
                        ax_mag.semilogx(
                            [Wcg, Wcg], [0, -20*np.log10(gm)],
                            color='k', zorder=-20)
                    else:
                        ax_mag.loglog(
                            [Wcg, Wcg], [1./gm, 1e-8], color='k',
                            linestyle=':', zorder=-20)
                        ax_mag.loglog(
                            [Wcg, Wcg], [1., 1./gm], color='k', zorder=-20)

                    if plot_phase:
                        if deg:
                            ax_phase.semilogx(
                                [Wcg, Wcg], [0, phase_limit],
                                color='k', linestyle=':', zorder=-20)
                        else:
                            ax_phase.semilogx(
                                [Wcg, Wcg], [0, math.radians(phase_limit)],
                                color='k', linestyle=':', zorder=-20)

                    ax_mag.set_ylim(mag_ylim)
                    ax_phase.set_ylim(phase_ylim)

                if margin_info:
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
                    # TODO: gets overwritten below
                    plt.suptitle(
                    "Gm = %.2f %s(at %.2f %s), "
                        "Pm = %.2f %s (at %.2f %s)" %
                        (20*np.log10(gm) if dB else gm,
                         'dB ' if dB else '',
                         Wcg, 'Hz' if Hz else 'rad/s',
                         pm if deg else math.radians(pm),
                         'deg' if deg else 'rad',
                         Wcp, 'Hz' if Hz else 'rad/s'))

            def gen_zero_centered_series(val_min, val_max, period):
                v1 = np.ceil(val_min / period - 0.2)
                v2 = np.floor(val_max / period + 0.2)
                return np.arange(v1, v2 + 1) * period

            # TODO: what is going on here
            # TODO: fix to use less dense labels, when needed
            if plot_phase:
                if deg:
                    ylim = ax_phase.get_ylim()
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], 45.))
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], 15.), minor=True)
                else:
                    ylim = ax_phase.get_ylim()
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], math.pi / 4.))
                    ax_phase.set_yticks(gen_zero_centered_series(
                        ylim[0], ylim[1], math.pi / 12.), minor=True)

                ax_phase.grid(grid and not margins, which='both')

    #
    # Update the plot title (= figure suptitle)
    #
    # If plots are built up by multiple calls to plot() and the title is
    # not given, then the title is updated to provide a list of unique text
    # items in each successive title.  For data generated by the frequency
    # response function this will generate a common prefix followed by a
    # list of systems (e.g., "Step response for sys[1], sys[2]").
    #

    # Set the initial title for the data
    sysnames = list(set(sysnames))      # get rid of duplicates
    if title is None:
        title = "Bode plot for " + ", ".join(sysnames)

    if fig is not None and title is not None:
        # Get the current title, if it exists
        old_title = None if fig._suptitle is None else fig._suptitle._text
        new_title = title

        if old_title is not None:
            # Find the common part of the titles
            common_prefix = commonprefix([old_title, new_title])

            # Back up to the last space
            last_space = common_prefix.rfind(' ')
            if last_space > 0:
                common_prefix = common_prefix[:last_space]
            common_len = len(common_prefix)

            # Add the new part of the title (usually the system name)
            if old_title[common_len:] != new_title[common_len:]:
                separator = ',' if len(common_prefix) > 0 else ';'
                new_title = old_title + separator + new_title[common_len:]

        # Add the title
        with plt.rc_context(freqplot_rcParams):
            fig.suptitle(new_title)

    #
    # Label the axes (including trace labels)
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
    for j in range(ninputs):
        # If we have more than one column, label the individual responses
        if noutputs > 1 or ninputs > 1:
            with plt.rc_context(_freqplot_rcParams):
                ax_array[0, j].set_title(f"From {data[0].input_labels[j]}")

        # Label the frequency axis
        ax_array[-1, j].set_xlabel(freq_label % ("Hz" if Hz else "rad/s",))

    # Label the rows
    for i in range(noutputs):
        if plot_magnitude:
            ax_mag = ax_array[i*2 if plot_phase else i, 0]
            ax_mag.set_ylabel("Magnitude (dB)" if dB else "Magnitude")
        if plot_phase:
            ax_phase = ax_array[i*2 + 1 if plot_magnitude else i, 0]
            ax_phase.set_ylabel("Phase (deg)" if deg else "Phase (rad)")

        if noutputs > 1 or ninputs > 1:
            if plot_magnitude and plot_phase:
                # Get existing ylabel for left column and add a blank line
                ax_mag.set_ylabel("\n" + ax_mag.get_ylabel())
                ax_phase.set_ylabel("\n" + ax_phase.get_ylabel())

                # TODO: remove?
                # Redraw the figure to get the proper locations for everything
                # fig.tight_layout()

                # Get the bounding box including the labels
                inv_transform = fig.transFigure.inverted()
                mag_bbox = inv_transform.transform(
                    ax_mag.get_tightbbox(fig.canvas.get_renderer()))
                phase_bbox = inv_transform.transform(
                    ax_phase.get_tightbbox(fig.canvas.get_renderer()))

                # Get the axes limits without labels for use in the y position
                mag_bot = inv_transform.transform(
                    ax_mag.transAxes.transform((0, 0)))[1]
                phase_top = inv_transform.transform(
                    ax_phase.transAxes.transform((0, 1)))[1]

                # Figure out location for the text (center left in figure frame)
                xpos = mag_bbox[0, 0]               # left edge
                ypos = (mag_bot + phase_top) / 2    # centered between axes

                # Put a centered label as text outside the box
                fig.text(
                    0.8 * xpos, ypos, f"To {data[0].output_labels[i]}\n",
                    rotation=90, ha='left', va='center',
                    fontsize=_freqplot_rcParams['axes.titlesize'])
            else:
                # Only a single axes => add label to the left
                ax_array[i, 0].set_ylabel(
                    f"To {data[0].output_labels[i]}\n" +
                    ax_array[i, 0].get_ylabel())

    #
    # Create legends
    #
    # Legends can be placed manually by passing a legend_map array that
    # matches the shape of the suplots, with each item being a string
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
    # call to plot() may not generate any legends (eg, if no signals are
    # overlaid), but subsequent calls to plot() will need a legend for each
    # different response (system).
    #

    # Figure out where to put legends
    if legend_map is None:
        legend_map = np.full(ax_array.shape, None, dtype=object)
        if legend_loc == None:
            legend_loc = 'center right'

        # TODO: add in additional processing later

        # Put legend in the upper right
        legend_map[0, -1] = legend_loc

    # Create axis legends
    for i in range(nrows):
        for j in range(ncols):
            ax = ax_array[i, j]
            # Get the labels to use, removing common strings
            labels = _make_legend_labels(
                [line.get_label() for line in ax.get_lines()])

            # Generate the label, if needed
            if len(labels) > 1 and legend_map[i, j] != None:
                with plt.rc_context(freqplot_rcParams):
                    ax.legend(labels, loc=legend_map[i, j])

    #
    # Legacy return pocessing
    #
    if plot is True:            # legacy usage; remove in future release
        if len(data) == 1:
            return mags[0], phases[0], omegas[0]
        else:
            return mags, phases, omegas

    return None                 # TODO: replace with ax


#
# Nyquist plot
#

# Default values for module parameter variables
_nyquist_defaults = {
    'nyquist.primary_style': ['-', '-.'],       # style for primary curve
    'nyquist.mirror_style': ['--', ':'],        # style for mirror curve
    'nyquist.arrows': 2,                        # number of arrors around curve
    'nyquist.arrow_size': 8,                    # pixel size for arrows
    'nyquist.encirclement_threshold': 0.05,     # warning threshold
    'nyquist.indent_radius': 1e-4,              # indentation radius
    'nyquist.indent_direction': 'right',        # indentation direction
    'nyquist.indent_points': 50,                # number of points to insert
    'nyquist.max_curve_magnitude': 20,          # clip large values
    'nyquist.max_curve_offset': 0.02,           # offset of primary/mirror
    'nyquist.start_marker': 'o',                # marker at start of curve
    'nyquist.start_marker_size': 4,             # size of the maker
}


def nyquist_plot(
        syslist, omega=None, plot=True, omega_limits=None, omega_num=None,
        label_freq=0, color=None, return_contour=False,
        warn_encirclements=True, warn_nyquist=True, **kwargs):
    """Nyquist plot for a system.

    Plots a Nyquist plot for the system over a (optional) frequency range.
    The curve is computed by evaluating the Nyqist segment along the positive
    imaginary axis, with a mirror image generated to reflect the negative
    imaginary axis.  Poles on or near the imaginary axis are avoided using a
    small indentation.  The portion of the Nyquist contour at infinity is not
    explicitly computed (since it maps to a constant value for any system with
    a proper transfer function).

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK). Nyquist
        curves for each system are plotted on the same graph.

    omega : array_like, optional
        Set of frequencies to be evaluated, in rad/sec.

    omega_limits : array_like of two values, optional
        Limits to the range of frequencies. Ignored if omega is provided, and
        auto-generated if omitted.

    omega_num : int, optional
        Number of frequency samples to plot.  Defaults to
        config.defaults['freqplot.number_of_samples'].

    plot : boolean, optional
        If True (default), plot the Nyquist plot.

    color : string, optional
        Used to specify the color of the line and arrowhead.

    return_contour : bool, optional
        If 'True', return the contour used to evaluate the Nyquist plot.

    **kwargs : :func:`matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords (passed to `matplotlib`)

    Returns
    -------
    count : int (or list of int if len(syslist) > 1)
        Number of encirclements of the point -1 by the Nyquist curve.  If
        multiple systems are given, an array of counts is returned.

    contour : ndarray (or list of ndarray if len(syslist) > 1)), optional
        The contour used to create the primary Nyquist curve segment, returned
        if `return_contour` is Tue.  To obtain the Nyquist curve values,
        evaluate system(s) along contour.

    Other Parameters
    ----------------
    arrows : int or 1D/2D array of floats, optional
        Specify the number of arrows to plot on the Nyquist curve.  If an
        integer is passed. that number of equally spaced arrows will be
        plotted on each of the primary segment and the mirror image.  If a 1D
        array is passed, it should consist of a sorted list of floats between
        0 and 1, indicating the location along the curve to plot an arrow.  If
        a 2D array is passed, the first row will be used to specify arrow
        locations for the primary curve and the second row will be used for
        the mirror image.

    arrow_size : float, optional
        Arrowhead width and length (in display coordinates).  Default value is
        8 and can be set using config.defaults['nyquist.arrow_size'].

    arrow_style : matplotlib.patches.ArrowStyle, optional
        Define style used for Nyquist curve arrows (overrides `arrow_size`).

    encirclement_threshold : float, optional
        Define the threshold for generating a warning if the number of net
        encirclements is a non-integer value.  Default value is 0.05 and can
        be set using config.defaults['nyquist.encirclement_threshold'].

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

    label_freq : int, optiona
        Label every nth frequency on the plot.  If not specified, no labels
        are generated.

    max_curve_magnitude : float, optional
        Restrict the maximum magnitude of the Nyquist plot to this value.
        Portions of the Nyquist plot whose magnitude is restricted are
        plotted using a different line style.

    max_curve_offset : float, optional
        When plotting scaled portion of the Nyquist plot, increase/decrease
        the magnitude by this fraction of the max_curve_magnitude to allow
        any overlaps between the primary and mirror curves to be avoided.

    mirror_style : [str, str] or False
        Linestyles for mirror image of the Nyquist curve.  The first element
        is used for unscaled portions of the Nyquist curve, the second element
        is used for portions that are scaled (using max_curve_magnitude).  If
        `False` then omit completely.  Default linestyle (['--', ':']) is
        determined by config.defaults['nyquist.mirror_style'].

    primary_style : [str, str], optional
        Linestyles for primary image of the Nyquist curve.  The first
        element is used for unscaled portions of the Nyquist curve,
        the second element is used for portions that are scaled (using
        max_curve_magnitude).  Default linestyle (['-', '-.']) is
        determined by config.defaults['nyquist.mirror_style'].

    start_marker : str, optional
        Matplotlib marker to use to mark the starting point of the Nyquist
        plot.  Defaults value is 'o' and can be set using
        config.defaults['nyquist.start_marker'].

    start_marker_size : float, optional
        Start marker size (in display coordinates).  Default value is
        4 and can be set using config.defaults['nyquist.start_marker_size'].

    warn_nyquist : bool, optional
        If set to 'False', turn off warnings about frequencies above Nyquist.

    warn_encirclements : bool, optional
        If set to 'False', turn off warnings about number of encirclements not
        meeting the Nyquist criterion.

    Notes
    -----
    1. If a discrete time model is given, the frequency response is computed
       along the upper branch of the unit circle, using the mapping ``z =
       exp(1j * omega * dt)`` where `omega` ranges from 0 to `pi/dt` and `dt`
       is the discrete timebase.  If timebase not specified (``dt=True``),
       `dt` is set to 1.

    2. If a continuous-time system contains poles on or near the imaginary
       axis, a small indentation will be used to avoid the pole.  The radius
       of the indentation is given by `indent_radius` and it is taken to the
       right of stable poles and the left of unstable poles.  If a pole is
       exactly on the imaginary axis, the `indent_direction` parameter can be
       used to set the direction of indentation.  Setting `indent_direction`
       to `none` will turn off indentation.  If `return_contour` is True, the
       exact contour used for evaluation is returned.

    3. For those portions of the Nyquist plot in which the contour is
       indented to avoid poles, resuling in a scaling of the Nyquist plot,
       the line styles are according to the settings of the `primary_style`
       and `mirror_style` keywords.  By default the scaled portions of the
       primary curve use a dotted line style and the scaled portion of the
       mirror image use a dashdot line style.

    Examples
    --------
    >>> G = ct.zpk([], [-1, -2, -3], gain=100)
    >>> ct.nyquist_plot(G)
    2

    """
    # Check to see if legacy 'Plot' keyword was used
    if 'Plot' in kwargs:
        warnings.warn("'Plot' keyword is deprecated in nyquist_plot; "
                      "use 'plot'", FutureWarning)
        # Map 'Plot' keyword to 'plot' keyword
        plot = kwargs.pop('Plot')

    # Check to see if legacy 'labelFreq' keyword was used
    if 'labelFreq' in kwargs:
        warnings.warn("'labelFreq' keyword is deprecated in nyquist_plot; "
                      "use 'label_freq'", FutureWarning)
        # Map 'labelFreq' keyword to 'label_freq' keyword
        label_freq = kwargs.pop('labelFreq')

    # Check to see if legacy 'arrow_width' or 'arrow_length' were used
    if 'arrow_width' in kwargs or 'arrow_length' in kwargs:
        warnings.warn(
            "'arrow_width' and 'arrow_length' keywords are deprecated in "
            "nyquist_plot; use `arrow_size` instead", FutureWarning)
        kwargs['arrow_size'] = \
            (kwargs.get('arrow_width', 0) + kwargs.get('arrow_length', 0)) / 2
        kwargs.pop('arrow_width', False)
        kwargs.pop('arrow_length', False)

    # Get values for params (and pop from list to allow keyword use in plot)
    omega_num_given = omega_num is not None
    omega_num = config._get_param('freqplot', 'number_of_samples', omega_num)
    arrows = config._get_param(
        'nyquist', 'arrows', kwargs, _nyquist_defaults, pop=True)
    arrow_size = config._get_param(
        'nyquist', 'arrow_size', kwargs, _nyquist_defaults, pop=True)
    arrow_style = config._get_param('nyquist', 'arrow_style', kwargs, None)
    indent_radius = config._get_param(
        'nyquist', 'indent_radius', kwargs, _nyquist_defaults, pop=True)
    encirclement_threshold = config._get_param(
        'nyquist', 'encirclement_threshold', kwargs,
        _nyquist_defaults, pop=True)
    indent_direction = config._get_param(
        'nyquist', 'indent_direction', kwargs, _nyquist_defaults, pop=True)
    indent_points = config._get_param(
        'nyquist', 'indent_points', kwargs, _nyquist_defaults, pop=True)
    max_curve_magnitude = config._get_param(
        'nyquist', 'max_curve_magnitude', kwargs, _nyquist_defaults, pop=True)
    max_curve_offset = config._get_param(
        'nyquist', 'max_curve_offset', kwargs, _nyquist_defaults, pop=True)
    start_marker = config._get_param(
        'nyquist', 'start_marker', kwargs, _nyquist_defaults, pop=True)
    start_marker_size = config._get_param(
        'nyquist', 'start_marker_size', kwargs, _nyquist_defaults, pop=True)

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

    # If argument was a singleton, turn it into a tuple
    if not isinstance(syslist, (list, tuple)):
        syslist = (syslist,)

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
    counts, contours = [], []
    for sys in syslist:
        if not sys.issiso():
            # TODO: Add MIMO nyquist plots.
            raise ControlMIMONotImplemented(
                "Nyquist plot currently only supports SISO systems.")

        # Figure out the frequency range
        omega_sys = np.asarray(omega)

        # Determine the contour used to evaluate the Nyquist curve
        if sys.isdtime(strict=True):
            # Restrict frequencies for discrete-time systems
            nyquistfrq = math.pi / sys.dt
            if not omega_range_given:
                # limit up to and including nyquist frequency
                omega_sys = np.hstack((
                    omega_sys[omega_sys < nyquistfrq], nyquistfrq))

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
            if len(splane_poles) > 0: # accomodate no splane poles if dtime sys
                for i, s in enumerate(splane_contour):
                    # Find the nearest pole
                    p = splane_poles[(np.abs(splane_poles - s)).argmin()]

                    # See if we need to indent around it
                    if abs(s - p) < indent_radius:
                        # Figure out how much to offset (simple trigonometry)
                        offset = np.sqrt(indent_radius ** 2 - (s - p).imag ** 2) \
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
                            raise ValueError("unknown value for indent_direction")

        # change contour to z-plane if necessary
        if sys.isctime():
            contour = splane_contour
        else:
            contour = np.exp(splane_contour * sys.dt)

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
        # Make sure that the enciriclements match the Nyquist criterion
        #
        # If the user specifies the frequency points to use, it is possible
        # to miss enciriclements, so we check here to make sure that the
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
            elif indent_direction == 'none' and any(sys.poles().real == 0) and \
                 warn_encirclements:
                warnings.warn(
                    "system has pure imaginary poles but indentation is"
                    " turned off; results may be meaningless",
                    RuntimeWarning, stacklevel=2)

        counts.append(count)
        contours.append(contour)

        if plot:
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

            # Find the different portions of the curve (with scaled pts marked)
            reg_mask = np.logical_or(
                np.abs(resp) > max_curve_magnitude,
                splane_contour.real != 0)
            # reg_mask = np.logical_or(
            #     np.abs(resp.real) > max_curve_magnitude,
            #     np.abs(resp.imag) > max_curve_magnitude)

            scale_mask = ~reg_mask \
                & np.concatenate((~reg_mask[1:], ~reg_mask[-1:])) \
                & np.concatenate((~reg_mask[0:1], ~reg_mask[:-1]))

            # Rescale the points with large magnitude
            rescale = np.logical_and(
                reg_mask, abs(resp) > max_curve_magnitude)
            resp[rescale] *= max_curve_magnitude / abs(resp[rescale])

            # Plot the regular portions of the curve (and grab the color)
            x_reg = np.ma.masked_where(reg_mask, resp.real)
            y_reg = np.ma.masked_where(reg_mask, resp.imag)
            p = plt.plot(
                x_reg, y_reg, primary_style[0], color=color, **kwargs)
            c = p[0].get_color()

            # Figure out how much to offset the curve: the offset goes from
            # zero at the start of the scaled section to max_curve_offset as
            # we move along the curve
            curve_offset = _compute_curve_offset(
                resp, scale_mask, max_curve_offset)

            # Plot the scaled sections of the curve (changing linestyle)
            x_scl = np.ma.masked_where(scale_mask, resp.real)
            y_scl = np.ma.masked_where(scale_mask, resp.imag)
            if x_scl.count() >= 1 and y_scl.count() >= 1:
                plt.plot(
                    x_scl * (1 + curve_offset),
                    y_scl * (1 + curve_offset),
                    primary_style[1], color=c, **kwargs)

            # Plot the primary curve (invisible) for setting arrows
            x, y = resp.real.copy(), resp.imag.copy()
            x[reg_mask] *= (1 + curve_offset[reg_mask])
            y[reg_mask] *= (1 + curve_offset[reg_mask])
            p = plt.plot(x, y, linestyle='None', color=c, **kwargs)

            # Add arrows
            ax = plt.gca()
            _add_arrows_to_line2D(
                ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=1)

            # Plot the mirror image
            if mirror_style is not False:
                # Plot the regular and scaled segments
                plt.plot(
                    x_reg, -y_reg, mirror_style[0], color=c, **kwargs)
                if x_scl.count() >= 1 and y_scl.count() >= 1:
                    plt.plot(
                        x_scl * (1 - curve_offset),
                        -y_scl * (1 - curve_offset),
                        mirror_style[1], color=c, **kwargs)

                # Add the arrows (on top of an invisible contour)
                x, y = resp.real.copy(), resp.imag.copy()
                x[reg_mask] *= (1 - curve_offset[reg_mask])
                y[reg_mask] *= (1 - curve_offset[reg_mask])
                p = plt.plot(x, -y, linestyle='None', color=c, **kwargs)
                _add_arrows_to_line2D(
                    ax, p[0], arrow_pos, arrowstyle=arrow_style, dir=-1)

            # Mark the start of the curve
            if start_marker:
                plt.plot(resp[0].real, resp[0].imag, start_marker,
                         color=c, markersize=start_marker_size)

            # Mark the -1 point
            plt.plot([-1], [0], 'r+')

            # Label the frequencies of the points
            if label_freq:
                ind = slice(None, None, label_freq)
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
                    plt.text(xpt, ypt, ' ' +
                             str(int(np.round(f / 1000 ** pow1000, 0))) + ' ' +
                             prefix + 'Hz')

    if plot:
        ax = plt.gca()
        ax.set_xlabel("Real axis")
        ax.set_ylabel("Imaginary axis")
        ax.grid(color="lightgray")

    # "Squeeze" the results
    if len(syslist) == 1:
        counts, contours = counts[0], contours[0]

    # Return counts and (optionally) the contour we used
    return (counts, contours) if return_contour else counts


# Internal function to add arrows to a curve
def _add_arrows_to_line2D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', arrowsize=1, dir=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: Axes object as returned by axes command (or gca)
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows

    Based on https://stackoverflow.com/questions/26911898/

    """
    if not isinstance(line, mpl.lines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
    }

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

    if transform is None:
        transform = axes.transData

    # Compute the arc length along the curve
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    arrows = []
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)

        # Figure out what direction to paint the arrow
        if dir == 1:
            arrow_tail = (x[n], y[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        elif dir == -1:
            # Orient the arrow in the other direction on the segment
            arrow_tail = (x[n + 1], y[n + 1])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        else:
            raise ValueError("unknown value for keyword 'dir'")

        p = mpl.patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, lw=0,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


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

    # Walk through the response and keep track of each continous component
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
# TODO: think about how (and whether) to handle lists of systems
def gangof4_plot(P, C, omega=None, **kwargs):
    """Plot the "Gang of 4" transfer functions for a system.

    Generates a 2x2 plot showing the "Gang of 4" sensitivity functions
    [T, PS; CS, S]

    Parameters
    ----------
    P, C : LTI
        Linear input/output systems (process and control)
    omega : array
        Range of frequencies (list or bounds) in rad/sec
    **kwargs : :func:`matplotlib.pyplot.plot` keyword properties, optional
        Additional keywords (passed to `matplotlib`)

    Returns
    -------
    None

    Examples
    --------
    >>> P = ct.tf([1], [1, 1])
    >>> C = ct.tf([2], [1])
    >>> ct.gangof4_plot(P, C)

    """
    if not P.issiso() or not C.issiso():
        # TODO: Add MIMO go4 plots.
        raise ControlMIMONotImplemented(
            "Gang of four is currently only implemented for SISO systems.")

    # Get the default parameter values
    dB = config._get_param(
        'freqplot', 'dB', kwargs, _freqplot_defaults, pop=True)
    Hz = config._get_param(
        'freqplot', 'Hz', kwargs, _freqplot_defaults, pop=True)
    grid = config._get_param(
        'freqplot', 'grid', kwargs, _freqplot_defaults, pop=True)

    # Compute the senstivity functions
    L = P * C
    S = feedback(1, L)
    T = L * S

    # Select a default range if none is provided
    # TODO: This needs to be made more intelligent
    if omega is None:
        omega = _default_frequency_range((P, C, S), Hz=Hz)

    # Set up the axes with labels so that multiple calls to
    # gangof4_plot will superimpose the data.  See details in bode_plot.
    plot_axes = {'t': None, 's': None, 'ps': None, 'cs': None}
    for ax in plt.gcf().axes:
        label = ax.get_label()
        if label.startswith('control-gangof4-'):
            key = label[len('control-gangof4-'):]
            if key not in plot_axes:
                raise RuntimeError(
                    "unknown gangof4 axis type '{}'".format(label))
            plot_axes[key] = ax

    # if any of the axes are missing, start from scratch
    if any((ax is None for ax in plot_axes.values())):
        plt.clf()
        plot_axes = {'s': plt.subplot(221, label='control-gangof4-s'),
                     'ps': plt.subplot(222, label='control-gangof4-ps'),
                     'cs': plt.subplot(223, label='control-gangof4-cs'),
                     't': plt.subplot(224, label='control-gangof4-t')}

    #
    # Plot the four sensitivity functions
    #
    omega_plot = omega / (2. * math.pi) if Hz else omega

    # TODO: Need to add in the mag = 1 lines
    mag_tmp, phase_tmp, omega = S.frequency_response(omega)
    mag = np.squeeze(mag_tmp)
    if dB:
        plot_axes['s'].semilogx(omega_plot, 20 * np.log10(mag), **kwargs)
    else:
        plot_axes['s'].loglog(omega_plot, mag, **kwargs)
    plot_axes['s'].set_ylabel("$|S|$" + " (dB)" if dB else "")
    plot_axes['s'].tick_params(labelbottom=False)
    plot_axes['s'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = (P * S).frequency_response(omega)
    mag = np.squeeze(mag_tmp)
    if dB:
        plot_axes['ps'].semilogx(omega_plot, 20 * np.log10(mag), **kwargs)
    else:
        plot_axes['ps'].loglog(omega_plot, mag, **kwargs)
    plot_axes['ps'].tick_params(labelbottom=False)
    plot_axes['ps'].set_ylabel("$|PS|$" + " (dB)" if dB else "")
    plot_axes['ps'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = (C * S).frequency_response(omega)
    mag = np.squeeze(mag_tmp)
    if dB:
        plot_axes['cs'].semilogx(omega_plot, 20 * np.log10(mag), **kwargs)
    else:
        plot_axes['cs'].loglog(omega_plot, mag, **kwargs)
    plot_axes['cs'].set_xlabel(
        "Frequency (Hz)" if Hz else "Frequency (rad/sec)")
    plot_axes['cs'].set_ylabel("$|CS|$" + " (dB)" if dB else "")
    plot_axes['cs'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = T.frequency_response(omega)
    mag = np.squeeze(mag_tmp)
    if dB:
        plot_axes['t'].semilogx(omega_plot, 20 * np.log10(mag), **kwargs)
    else:
        plot_axes['t'].loglog(omega_plot, mag, **kwargs)
    plot_axes['t'].set_xlabel(
        "Frequency (Hz)" if Hz else "Frequency (rad/sec)")
    plot_axes['t'].set_ylabel("$|T|$" + " (dB)" if dB else "")
    plot_axes['t'].grid(grid, which='both')

    plt.tight_layout()

#
# Singular values plot
#


def singular_values_plot(syslist, omega=None,
                         plot=True, omega_limits=None, omega_num=None,
                         *args, **kwargs):
    """Singular value plot for a system

    Plots a singular value plot for the system over a (optional) frequency
    range.

    Parameters
    ----------
    syslist : linsys
        List of linear systems (single system is OK).
    omega : array_like
        List of frequencies in rad/sec to be used for frequency response.
    plot : bool
        If True (default), generate the singular values plot.
    omega_limits : array_like of two values
        Limits of the frequency vector to generate.
        If Hz=True the limits are in Hz otherwise in rad/s.
    omega_num : int
        Number of samples to plot. Default value (1000) set by
        config.defaults['freqplot.number_of_samples'].
    dB : bool
        If True, plot result in dB. Default value (False) set by
        config.defaults['freqplot.dB'].
    Hz : bool
        If True, plot frequency in Hz (omega must be provided in rad/sec).
        Default value (False) set by config.defaults['freqplot.Hz']

    Returns
    -------
    sigma : ndarray (or list of ndarray if len(syslist) > 1))
        singular values
    omega : ndarray (or list of ndarray if len(syslist) > 1))
        frequency in rad/sec

    Other Parameters
    ----------------
    grid : bool
        If True, plot grid lines on gain and phase plots.  Default is set by
        `config.defaults['freqplot.grid']`.

    Examples
    --------
    >>> omegas = np.logspace(-4, 1, 1000)
    >>> den = [75, 1]
    >>> G = ct.tf([[[87.8], [-86.4]], [[108.2], [-109.6]]],
    ...           [[den, den], [den, den]])
    >>> sigmas, omegas = ct.singular_values_plot(G, omega=omegas, plot=False)

    >>> sigmas, omegas = ct.singular_values_plot(G, 0.0, plot=False)

    """

    # Make a copy of the kwargs dictionary since we will modify it
    kwargs = dict(kwargs)

    # Get values for params (and pop from list to allow keyword use in plot)
    dB = config._get_param(
        'freqplot', 'dB', kwargs, _freqplot_defaults, pop=True)
    Hz = config._get_param(
        'freqplot', 'Hz', kwargs, _freqplot_defaults, pop=True)
    grid = config._get_param(
        'freqplot', 'grid', kwargs, _freqplot_defaults, pop=True)
    plot = config._get_param(
        'freqplot', 'plot', plot, True)
    omega_num = config._get_param('freqplot', 'number_of_samples', omega_num)

    # If argument was a singleton, turn it into a tuple
    if not isinstance(syslist, (list, tuple)):
        syslist = (syslist,)

    omega, omega_range_given = _determine_omega_vector(
        syslist, omega, omega_limits, omega_num, Hz=Hz)

    omega = np.atleast_1d(omega)

    if plot:
        fig = plt.gcf()
        ax_sigma = None

        # Get the current axes if they already exist
        for ax in fig.axes:
            if ax.get_label() == 'control-sigma':
                ax_sigma = ax

        # If no axes present, create them from scratch
        if ax_sigma is None:
            plt.clf()
            ax_sigma = plt.subplot(111, label='control-sigma')

        # color cycle handled manually as all singular values
        # of the same systems are expected to be of the same color
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_offset = 0
        if len(ax_sigma.lines) > 0:
            last_color = ax_sigma.lines[-1].get_color()
            if last_color in color_cycle:
                color_offset = color_cycle.index(last_color) + 1

    sigmas, omegas, nyquistfrqs = [], [], []
    for idx_sys, sys in enumerate(syslist):
        omega_sys = np.asarray(omega)
        if sys.isdtime(strict=True):
            nyquistfrq = math.pi / sys.dt
            if not omega_range_given:
                # limit up to and including nyquist frequency
                omega_sys = np.hstack((
                    omega_sys[omega_sys < nyquistfrq], nyquistfrq))

            omega_complex = np.exp(1j * omega_sys * sys.dt)
        else:
            nyquistfrq = None
            omega_complex = 1j*omega_sys

        fresp = sys(omega_complex, squeeze=False)

        fresp = fresp.transpose((2, 0, 1))
        sigma = np.linalg.svd(fresp, compute_uv=False)

        sigmas.append(sigma.transpose())  # return shape is "channel first"
        omegas.append(omega_sys)
        nyquistfrqs.append(nyquistfrq)

        if plot:
            color = color_cycle[(idx_sys + color_offset) % len(color_cycle)]
            color = kwargs.pop('color', color)

            nyquistfrq_plot = None
            if Hz:
                omega_plot = omega_sys / (2. * math.pi)
                if nyquistfrq:
                    nyquistfrq_plot = nyquistfrq / (2. * math.pi)
            else:
                omega_plot = omega_sys
                if nyquistfrq:
                    nyquistfrq_plot = nyquistfrq
            sigma_plot = sigma

            if dB:
                ax_sigma.semilogx(omega_plot, 20 * np.log10(sigma_plot),
                                  color=color, *args, **kwargs)
            else:
                ax_sigma.loglog(omega_plot, sigma_plot,
                                color=color, *args, **kwargs)

            if nyquistfrq_plot is not None:
                ax_sigma.axvline(x=nyquistfrq_plot, color=color)

    # Add a grid to the plot + labeling
    if plot:
        ax_sigma.grid(grid, which='both')
        ax_sigma.set_ylabel(
            "Singular Values (dB)" if dB else "Singular Values")
        ax_sigma.set_xlabel("Frequency (Hz)" if Hz else "Frequency (rad/sec)")

    if len(syslist) == 1:
        return sigmas[0], omegas[0]
    else:
        return sigmas, omegas
#
# Utility functions
#
# This section of the code contains some utility functions for
# generating frequency domain plots
#


# Determine the frequency range to be used
def _determine_omega_vector(syslist, omega_in, omega_limits, omega_num,
                            Hz=None, feature_periphery_decades=None):
    """Determine the frequency range for a frequency-domain plot
    according to a standard logic.

    If omega_in and omega_limits are both None, then omega_out is computed
    on omega_num points according to a default logic defined by
    _default_frequency_range and tailored for the list of systems syslist, and
    omega_range_given is set to False.
    If omega_in is None but omega_limits is an array-like of 2 elements, then
    omega_out is computed with the function np.logspace on omega_num points
    within the interval [min, max] =  [omega_limits[0], omega_limits[1]], and
    omega_range_given is set to True.
    If omega_in is not None, then omega_out is set to omega_in,
    and omega_range_given is set to True

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    omega_in : 1D array_like or None
        Frequency range specified by the user
    omega_limits : 1D array_like or None
        Frequency limits specified by the user
    omega_num : int
        Number of points to be used for the frequency
        range (if the frequency range is not user-specified)
    Hz : bool, optional
        If True, the limits (first and last value) of the frequencies
        are set to full decades in Hz so it fits plotting with logarithmic
        scale in Hz otherwise in rad/s. Omega is always returned in rad/sec.

    Returns
    -------
    omega_out : 1D array
        Frequency range to be used
    omega_range_given : bool
        True if the frequency range was specified by the user, either through
        omega_in or through omega_limits. False if both omega_in
        and omega_limits are None.
    """
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
        ``config.defaults['freqplot.number_of_samples'].  If None, then the
        default from `numpy.logspace` is used.
    feature_periphery_decades : float, optional
        Defines how many decades shall be included in the frequency range on
        both sides of features (poles, zeros).  The default value is read from
        ``config.defaults['freqplot.feature_periphery_decades']``.

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
                fn = math.pi * 1. / sys.dt
                # TODO: What distance to the Nyquist frequency is appropriate?
                freq_interesting.append(fn * 0.9)

                features_ = np.concatenate((sys.poles(), sys.zeros()))
                # Get rid of poles and zeros on the real axis (imag==0)
               # * origin and real < 0
                # * at 1.: would result in omega=0. (logaritmic plot!)
                toreplace = np.isclose(features_.imag, 0.0) & (
                                    (features_.real <= 0.) |
                                    (np.abs(features_.real - 1.0) < 1.e-10))
                if np.any(toreplace):
                    features_ = features_[~toreplace]
                # TODO: improve
                features_ = np.abs(np.log(features_) / (1.j * sys.dt))
            else:
                # TODO
                raise NotImplementedError(
                    "type of system in not implemented now")
            features = np.concatenate((features, features_))
        except NotImplementedError:
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
    """Determine exponent for which significand of a number is within the
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
