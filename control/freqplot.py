# freqplot.py - frequency domain plots for control systems
#
# Author: Richard M. Murray
# Date: 24 May 09
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import math
from .ctrlutil import unwrap
from .bdalg import feedback
from .margins import stability_margins
from . import config

__all__ = ['bode_plot', 'nyquist_plot', 'gangof4_plot',
           'bode', 'nyquist', 'gangof4']

# Default values for module parameter variables
_freqplot_defaults = {
    'freqplot.feature_periphery_decades': 1,
    'freqplot.number_of_samples': None,
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

# Default values for Bode plot configuration variables
_bode_defaults = {
    'bode.dB': False,           # Plot gain in dB
    'bode.deg': True,           # Plot phase in degrees
    'bode.Hz': False,           # Plot frequency in Hertz
    'bode.grid': True,          # Turn on grid for gain and phase
}


def bode_plot(syslist, omega=None,
              Plot=True, omega_limits=None, omega_num=None,
              margins=None, *args, **kwargs):
    """Bode plot for a system

    Plots a Bode plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : linsys
        List of linear input/output systems (single system is OK)
    omega : list
        List of frequencies in rad/sec to be used for frequency response
    dB : bool
        If True, plot result in dB.  Default is false.
    Hz : bool
        If True, plot frequency in Hz (omega must be provided in rad/sec).
        Default value (False) set by config.defaults['bode.Hz']
    deg : bool
        If True, plot phase in degrees (else radians).  Default value (True)
        config.defaults['bode.deg']
    Plot : bool
        If True (default), plot magnitude and phase
    omega_limits: tuple, list, ... of two values
        Limits of the to generate frequency vector.
        If Hz=True the limits are in Hz otherwise in rad/s.
    omega_num: int
        Number of samples to plot.  Defaults to
        config.defaults['freqplot.number_of_samples'].
    margins : bool
        If True, plot gain and phase margin.
    *args
        Additional arguments for :func:`matplotlib.plot` (color, linestyle, etc)
    **kwargs:
        Additional keywords (passed to `matplotlib`)

    Returns
    -------
    mag : array (list if len(syslist) > 1)
        magnitude
    phase : array (list if len(syslist) > 1)
        phase in radians
    omega : array (list if len(syslist) > 1)
        frequency in rad/sec

    Other Parameters
    ----------------
    grid : bool
        If True, plot grid lines on gain and phase plots.  Default is set by
        config.defaults['bode.grid'].

    The default values for Bode plot configuration parameters can be reset
    using the `config.defaults` dictionary, with module name 'bode'.

    Notes
    -----
    1. Alternatively, you may use the lower-level method (mag, phase, freq)
    = sys.freqresp(freq) to generate the frequency response for a system,
    but it returns a MIMO response.

    2. If a discrete time model is given, the frequency response is plotted
    along the upper branch of the unit circle, using the mapping z = exp(j
    \\omega dt) where omega ranges from 0 to pi/dt and dt is the discrete
    timebase.  If not timebase is specified (dt = True), dt is set to 1.

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = bode(sys)

    """
    # Make a copy of the kwargs dictonary since we will modify it
    kwargs = dict(kwargs)

    # Get values for params (and pop from list to allow keyword use in plot)
    dB = config._get_param('bode', 'dB', kwargs, _bode_defaults, pop=True)
    deg = config._get_param('bode', 'deg', kwargs, _bode_defaults, pop=True)
    Hz = config._get_param('bode', 'Hz', kwargs, _bode_defaults, pop=True)
    grid = config._get_param('bode', 'grid', kwargs, _bode_defaults, pop=True)
    Plot = config._get_param('bode', 'grid', Plot, True)
    margins = config._get_param('bode', 'margins', margins, False)

    # If argument was a singleton, turn it into a list
    if not getattr(syslist, '__iter__', False):
        syslist = (syslist,)

    if omega is None:
        if omega_limits is None:
            # Select a default range if none is provided
            omega = default_frequency_range(syslist, Hz=Hz,
                                            number_of_samples=omega_num)
        else:
            omega_limits = np.array(omega_limits)
            if Hz:
                omega_limits *= 2. * math.pi
            if omega_num:
                omega = sp.logspace(np.log10(omega_limits[0]),
                                    np.log10(omega_limits[1]),
                                    num=omega_num,
                                    endpoint=True)
            else:
                omega = sp.logspace(np.log10(omega_limits[0]),
                                    np.log10(omega_limits[1]),
                                    endpoint=True)

    mags, phases, omegas, nyquistfrqs = [], [], [], []
    for sys in syslist:
        if sys.inputs > 1 or sys.outputs > 1:
            # TODO: Add MIMO bode plots.
            raise NotImplementedError(
                "Bode is currently only implemented for SISO systems.")
        else:
            omega_sys = np.array(omega)
            if sys.isdtime(True):
                nyquistfrq = 2. * math.pi * 1. / sys.dt / 2.
                omega_sys = omega_sys[omega_sys < nyquistfrq]
                # TODO: What distance to the Nyquist frequency is appropriate?
            else:
                nyquistfrq = None
            # Get the magnitude and phase of the system
            mag_tmp, phase_tmp, omega_sys = sys.freqresp(omega_sys)
            mag = np.atleast_1d(np.squeeze(mag_tmp))
            phase = np.atleast_1d(np.squeeze(phase_tmp))
            phase = unwrap(phase)

            mags.append(mag)
            phases.append(phase)
            omegas.append(omega_sys)
            nyquistfrqs.append(nyquistfrq)
            # Get the dimensions of the current axis, which we will divide up
            # TODO: Not current implemented; just use subplot for now

            if Plot:
                nyquistfrq_plot = None
                if Hz:
                    omega_plot = omega_sys / (2. * math.pi)
                    if nyquistfrq:
                        nyquistfrq_plot = nyquistfrq / (2. * math.pi)
                else:
                    omega_plot = omega_sys
                    if nyquistfrq:
                        nyquistfrq_plot = nyquistfrq

                # Set up the axes with labels so that multiple calls to
                # bode_plot will superimpose the data.  This was implicit
                # before matplotlib 2.1, but changed after that (See
                # https://github.com/matplotlib/matplotlib/issues/9024).
                # The code below should work on all cases.

                # Get the current figure

                if 'sisotool' in kwargs:
                    fig = kwargs['fig']
                    ax_mag = fig.axes[0]
                    ax_phase = fig.axes[2]
                    sisotool = kwargs['sisotool']
                    del kwargs['fig']
                    del kwargs['sisotool']
                else:
                    fig = plt.gcf()
                    ax_mag = None
                    ax_phase = None
                    sisotool = False

                    # Get the current axes if they already exist
                    for ax in fig.axes:
                        if ax.get_label() == 'control-bode-magnitude':
                            ax_mag = ax
                        elif ax.get_label() == 'control-bode-phase':
                            ax_phase = ax

                    # If no axes present, create them from scratch
                    if ax_mag is None or ax_phase is None:
                        plt.clf()
                        ax_mag = plt.subplot(211,
                                             label='control-bode-magnitude')
                        ax_phase = plt.subplot(212,
                                               label='control-bode-phase',
                                               sharex=ax_mag)

                # Magnitude plot
                if dB:
                    pltline = ax_mag.semilogx(omega_plot, 20 * np.log10(mag),
                                              *args, **kwargs)
                else:
                    pltline = ax_mag.loglog(omega_plot, mag, *args, **kwargs)

                if nyquistfrq_plot:
                    ax_mag.axvline(nyquistfrq_plot,
                                   color=pltline[0].get_color())

                # Add a grid to the plot + labeling
                ax_mag.grid(grid and not margins, which='both')
                ax_mag.set_ylabel("Magnitude (dB)" if dB else "Magnitude")

                # Phase plot
                if deg:
                    phase_plot = phase * 180. / math.pi
                else:
                    phase_plot = phase
                ax_phase.semilogx(omega_plot, phase_plot, *args, **kwargs)

                # Show the phase and gain margins in the plot
                if margins:
                    margin = stability_margins(sys)
                    gm, pm, Wcg, Wcp = \
                        margin[0], margin[1], margin[3], margin[4]
                    # TODO: add some documentation describing why this is here
                    phase_at_cp = phases[0][(np.abs(omegas[0] - Wcp)).argmin()]
                    if phase_at_cp >= 0.:
                        phase_limit = 180.
                    else:
                        phase_limit = -180.

                    if Hz:
                        Wcg, Wcp = Wcg/(2*math.pi), Wcp/(2*math.pi)

                    ax_mag.axhline(y=0 if dB else 1, color='k', linestyle=':',
                                   zorder=-20)
                    ax_phase.axhline(y=phase_limit if deg else
                                     math.radians(phase_limit),
                                     color='k', linestyle=':', zorder=-20)
                    mag_ylim = ax_mag.get_ylim()
                    phase_ylim = ax_phase.get_ylim()

                    if pm != float('inf') and Wcp != float('nan'):
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
                                [Wcp, Wcp], [1e5, phase_limit+pm],
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

                    if gm != float('inf') and Wcg != float('nan'):
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

                        if deg:
                            ax_phase.semilogx(
                                [Wcg, Wcg], [1e-8, phase_limit],
                                color='k', linestyle=':', zorder=-20)
                        else:
                            ax_phase.semilogx(
                                [Wcg, Wcg], [1e-8, math.radians(phase_limit)],
                                color='k', linestyle=':', zorder=-20)

                    ax_mag.set_ylim(mag_ylim)
                    ax_phase.set_ylim(phase_ylim)

                    if sisotool:
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
                        plt.suptitle(
                            "Gm = %.2f %s(at %.2f %s), "
                            "Pm = %.2f %s (at %.2f %s)" %
                            (20*np.log10(gm) if dB else gm,
                             'dB ' if dB else '\b',
                             Wcg, 'Hz' if Hz else 'rad/s',
                             pm if deg else math.radians(pm),
                             'deg' if deg else 'rad',
                             Wcp, 'Hz' if Hz else 'rad/s'))

                if nyquistfrq_plot:
                    ax_phase.axvline(
                        nyquistfrq_plot, color=pltline[0].get_color())

                # Add a grid to the plot + labeling
                ax_phase.set_ylabel("Phase (deg)" if deg else "Phase (rad)")

                def gen_zero_centered_series(val_min, val_max, period):
                    v1 = np.ceil(val_min / period - 0.2)
                    v2 = np.floor(val_max / period + 0.2)
                    return np.arange(v1, v2 + 1) * period
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
                # ax_mag.grid(which='minor', alpha=0.3)
                # ax_mag.grid(which='major', alpha=0.9)
                # ax_phase.grid(which='minor', alpha=0.3)
                # ax_phase.grid(which='major', alpha=0.9)

                # Label the frequency axis
                ax_phase.set_xlabel("Frequency (Hz)" if Hz
                                    else "Frequency (rad/sec)")

    if len(syslist) == 1:
        return mags[0], phases[0], omegas[0]
    else:
        return mags, phases, omegas

#
# Nyquist plot
#

def nyquist_plot(syslist, omega=None, Plot=True, color=None,
                 labelFreq=0, *args, **kwargs):
    """
    Nyquist plot for a system

    Plots a Nyquist plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec
    Plot : boolean
        If True, plot magnitude
    color : string
        Used to specify the color of the plot
    labelFreq : int
        Label every nth frequency on the plot
    *args
        Additional arguments for :func:`matplotlib.plot` (color, linestyle, etc)
    **kwargs:
        Additional keywords (passed to `matplotlib`)

    Returns
    -------
    real : array
        real part of the frequency response array
    imag : array
        imaginary part of the frequency response array
    freq : array
        frequencies

    Examples
    --------
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> real, imag, freq = nyquist_plot(sys)

    """
    # If argument was a singleton, turn it into a list
    if not getattr(syslist, '__iter__', False):
        syslist = (syslist,)

    # Select a default range if none is provided
    if omega is None:
        omega = default_frequency_range(syslist)

    # Interpolate between wmin and wmax if a tuple or list are provided
    elif isinstance(omega, list) or isinstance(omega, tuple):
        # Only accept tuple or list of length 2
        if len(omega) != 2:
            raise ValueError("Supported frequency arguments are (wmin,wmax)"
                             "tuple or list, or frequency vector. ")
        omega = np.logspace(np.log10(omega[0]), np.log10(omega[1]),
                            num=50, endpoint=True, base=10.0)

    for sys in syslist:
        if sys.inputs > 1 or sys.outputs > 1:
            # TODO: Add MIMO nyquist plots.
            raise NotImplementedError(
                "Nyquist is currently only implemented for SISO systems.")
        else:
            # Get the magnitude and phase of the system
            mag_tmp, phase_tmp, omega = sys.freqresp(omega)
            mag = np.squeeze(mag_tmp)
            phase = np.squeeze(phase_tmp)

            # Compute the primary curve
            x = sp.multiply(mag, sp.cos(phase))
            y = sp.multiply(mag, sp.sin(phase))

            if Plot:
                # Plot the primary curve and mirror image
                p = plt.plot(x, y, '-', color=color, *args, **kwargs)
                c = p[0].get_color()
                ax = plt.gca()
                # Plot arrow to indicate Nyquist encirclement orientation
                ax.arrow(x[0], y[0], (x[1]-x[0])/2, (y[1]-y[0])/2, fc=c, ec=c,
                         head_width=0.2, head_length=0.2)

                plt.plot(x, -y, '-', color=c, *args, **kwargs)
                ax.arrow(
                    x[-1], -y[-1], (x[-1]-x[-2])/2, (y[-1]-y[-2])/2,
                    fc=c, ec=c, head_width=0.2, head_length=0.2)

                # Mark the -1 point
                plt.plot([-1], [0], 'r+')

            # Label the frequencies of the points
            if labelFreq:
                ind = slice(None, None, labelFreq)
                for xpt, ypt, omegapt in zip(x[ind], y[ind], omega[ind]):
                    # Convert to Hz
                    f = omegapt / (2 * sp.pi)

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

    if Plot:
        ax = plt.gca()
        ax.set_xlabel("Real axis")
        ax.set_ylabel("Imaginary axis")
        ax.grid(color="lightgray")

    return x, y, omega

#
# Gang of Four plot
#

# TODO: think about how (and whether) to handle lists of systems
def gangof4_plot(P, C, omega=None, **kwargs):
    """Plot the "Gang of 4" transfer functions for a system

    Generates a 2x2 plot showing the "Gang of 4" sensitivity functions
    [T, PS; CS, S]

    Parameters
    ----------
    P, C : LTI
        Linear input/output systems (process and control)
    omega : array
        Range of frequencies (list or bounds) in rad/sec

    Returns
    -------
    None
    """
    if P.inputs > 1 or P.outputs > 1 or C.inputs > 1 or C.outputs > 1:
        # TODO: Add MIMO go4 plots.
        raise NotImplementedError(
            "Gang of four is currently only implemented for SISO systems.")

    # Get the default parameter values
    dB = config._get_param('bode', 'dB', kwargs, _bode_defaults, pop=True)
    Hz = config._get_param('bode', 'Hz', kwargs, _bode_defaults, pop=True)
    grid = config._get_param('bode', 'grid', kwargs, _bode_defaults, pop=True)

    # Select a default range if none is provided
    # TODO: This needs to be made more intelligent
    if omega is None:
        omega = default_frequency_range((P, C))

    # Compute the senstivity functions
    L = P * C
    S = feedback(1, L)
    T = L * S

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
    mag_tmp, phase_tmp, omega = S.freqresp(omega)
    mag = np.squeeze(mag_tmp)
    plot_axes['s'].loglog(omega_plot, 20 * np.log10(mag) if dB else mag)
    plot_axes['s'].set_ylabel("$|S|$")
    plot_axes['s'].tick_params(labelbottom=False)
    plot_axes['s'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = (P * S).freqresp(omega)
    mag = np.squeeze(mag_tmp)
    plot_axes['ps'].loglog(omega_plot, 20 * np.log10(mag) if dB else mag)
    plot_axes['ps'].tick_params(labelbottom=False)
    plot_axes['ps'].set_ylabel("$|PS|$")
    plot_axes['ps'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = (C * S).freqresp(omega)
    mag = np.squeeze(mag_tmp)
    plot_axes['cs'].loglog(omega_plot, 20 * np.log10(mag) if dB else mag)
    plot_axes['cs'].set_xlabel(
        "Frequency (Hz)" if Hz else "Frequency (rad/sec)")
    plot_axes['cs'].set_ylabel("$|CS|$")
    plot_axes['cs'].grid(grid, which='both')

    mag_tmp, phase_tmp, omega = T.freqresp(omega)
    mag = np.squeeze(mag_tmp)
    plot_axes['t'].loglog(omega_plot, 20 * np.log10(mag) if dB else mag)
    plot_axes['t'].set_xlabel(
        "Frequency (Hz)" if Hz else "Frequency (rad/sec)")
    plot_axes['t'].set_ylabel("$|T|$")
    plot_axes['t'].grid(grid, which='both')

    plt.tight_layout()

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating frequency domain plots
#

# Compute reasonable defaults for axes
def default_frequency_range(syslist, Hz=None, number_of_samples=None,
                            feature_periphery_decades=None):
    """Compute a reasonable default frequency range for frequency
    domain plots.

    Finds a reasonable default frequency range by examining the features
    (poles and zeros) of the systems in syslist.

    Parameters
    ----------
    syslist : list of LTI
        List of linear input/output systems (single system is OK)
    Hz : bool
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
    >>> from matlab import ss
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> omega = default_frequency_range(sys)

    """
    # This code looks at the poles and zeros of all of the systems that
    # we are plotting and sets the frequency range to be one decade above
    # and below the min and max feature frequencies, rounded to the nearest
    # integer.  It excludes poles and zeros at the origin.  If no features
    # are found, it turns logspace(-1, 1)

    # Set default values for options
    number_of_samples = config._get_param(
        'freqplot', 'number_of_samples', number_of_samples)
    feature_periphery_decades = config._get_param(
        'freqplot', 'feature_periphery_decades', feature_periphery_decades, 1)

    # Find the list of all poles and zeros in the systems
    features = np.array(())
    freq_interesting = []

    # detect if single sys passed by checking if it is sequence-like
    if not getattr(syslist, '__iter__', False):
        syslist = (syslist,)

    for sys in syslist:
        try:
            # Add new features to the list
            if sys.isctime():
                features_ = np.concatenate((np.abs(sys.pole()),
                                            np.abs(sys.zero())))
                # Get rid of poles and zeros at the origin
                features_ = features_[features_ != 0.0]
                features = np.concatenate((features, features_))
            elif sys.isdtime(strict=True):
                fn = math.pi * 1. / sys.dt
                # TODO: What distance to the Nyquist frequency is appropriate?
                freq_interesting.append(fn * 0.9)

                features_ = np.concatenate((sys.pole(),
                                            sys.zero()))
                # Get rid of poles and zeros
                # * at the origin and real <= 0 & imag==0: log!
                # * at 1.: would result in omega=0. (logaritmic plot!)
                features_ = features_[
                    (features_.imag != 0.0) | (features_.real > 0.)]
                features_ = features_[
                    np.bitwise_not((features_.imag == 0.0) &
                                   (np.abs(features_.real - 1.0) < 1.e-10))]
                # TODO: improve
                features__ = np.abs(np.log(features_) / (1.j * sys.dt))
                features = np.concatenate((features, features__))
            else:
                # TODO
                raise NotImplementedError(
                    "type of system in not implemented now")
        except:
            pass

    # Make sure there is at least one point in the range
    if features.shape[0] == 0:
        features = np.array([1.])

    if Hz:
        features /= 2. * math.pi
        features = np.log10(features)
        lsp_min = np.floor(np.min(features) - feature_periphery_decades)
        lsp_max = np.ceil(np.max(features) + feature_periphery_decades)
        lsp_min += np.log10(2. * math.pi)
        lsp_max += np.log10(2. * math.pi)
    else:
        features = np.log10(features)
        lsp_min = np.floor(np.min(features) - feature_periphery_decades)
        lsp_max = np.ceil(np.max(features) + feature_periphery_decades)
    if freq_interesting:
        lsp_min = min(lsp_min, np.log10(min(freq_interesting)))
        lsp_max = max(lsp_max, np.log10(max(freq_interesting)))

    # TODO: Add a check in discrete case to make sure we don't get aliasing
    # (Attention: there is a list of system but only one omega vector)

    # Set the range to be an order of magnitude beyond any features
    if number_of_samples:
        omega = sp.logspace(
            lsp_min, lsp_max, num=number_of_samples, endpoint=True)
    else:
        omega = sp.logspace(lsp_min, lsp_max, endpoint=True)
    return omega

#
# KLD 5/23/11: Two functions to create nice looking labels
#

def get_pow1000(num):
    """Determine exponent for which significand of a number is within the
    range [1, 1000).
    """
    # Based on algorithm from http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14433.html, accessed 2010/11/7
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


def find_nearest_omega(omega_list, omega):
    omega_list = np.asarray(omega_list)
    return omega_list[(np.abs(omega_list - omega)).argmin()]


# Function aliases
bode = bode_plot
nyquist = nyquist_plot
gangof4 = gangof4_plot
