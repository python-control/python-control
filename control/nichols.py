"""nichols.py

Functions for plotting Black-Nichols charts.

Routines in this module:

nichols.nichols_plot aliased as nichols.nichols
nichols.nichols_grid
"""

# nichols.py - Nichols plot
#
# Contributed by Allan McInnes <Allan.McInnes@canterbury.ac.nz>
#
# This file contains some standard control system plots: Bode plots,
# Nyquist plots, Nichols plots and pole-zero diagrams
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
# $Id: freqplot.py 139 2011-03-30 16:19:59Z murrayrm $

import numpy as np
import matplotlib.pyplot as plt
from .ctrlutil import unwrap
from .freqplot import _default_frequency_range
from . import config

__all__ = ['nichols_plot', 'nichols', 'nichols_grid']

# Default parameters values for the nichols module
_nichols_defaults = {
    'nichols.grid': True,
}


def nichols_plot(sys_list, omega=None, grid=None):
    """Nichols plot for a system

    Plots a Nichols plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys_list : list of LTI, or LTI
        List of linear input/output systems (single system is OK)
    omega : array_like
        Range of frequencies (list or bounds) in rad/sec
    grid : boolean, optional
        True if the plot should include a Nichols-chart grid. Default is True.

    Returns
    -------
    None
    """
    # Get parameter values
    grid = config._get_param('nichols', 'grid', grid, True)


    # If argument was a singleton, turn it into a list
    if not getattr(sys_list, '__iter__', False):
        sys_list = (sys_list,)

    # Select a default range if none is provided
    if omega is None:
        omega = _default_frequency_range(sys_list)

    for sys in sys_list:
        # Get the magnitude and phase of the system
        mag_tmp, phase_tmp, omega = sys.frequency_response(omega)
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)

        # Convert to Nichols-plot format (phase in degrees,
        # and magnitude in dB)
        x = unwrap(np.degrees(phase), 360)
        y = 20*np.log10(mag)

        # Generate the plot
        plt.plot(x, y)

    plt.xlabel('Phase (deg)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Nichols Plot')

    # Mark the -180 point
    plt.plot([-180], [0], 'r+')

    # Add grid
    if grid:
        nichols_grid()


def nichols_grid(cl_mags=None, cl_phases=None, line_style='dotted'):
    """Nichols chart grid

    Plots a Nichols chart grid on the current axis, or creates a new chart
    if no plot already exists.

    Parameters
    ----------
    cl_mags : array-like (dB), optional
        Array of closed-loop magnitudes defining the iso-gain lines on a
        custom Nichols chart.
    cl_phases : array-like (degrees), optional
        Array of closed-loop phases defining the iso-phase lines on a custom
        Nichols chart. Must be in the range -360 < cl_phases < 0
    line_style : string, optional
        :doc:`Matplotlib linestyle \
            <matplotlib:gallery/lines_bars_and_markers/linestyles>`

    """
    # Default chart size
    ol_phase_min = -359.99
    ol_phase_max = 0.0
    ol_mag_min = -40.0
    ol_mag_max = default_ol_mag_max = 50.0

    # Find bounds of the current dataset, if there is one.
    if plt.gcf().gca().has_data():
        ol_phase_min, ol_phase_max, ol_mag_min, ol_mag_max = plt.axis()

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

    # N-circle phases (should be in the range -360 to 0)
    if cl_phases is None:
        # Choose a reasonable set of default phases (denser if the open-loop
        # data is restricted to a relatively small range of phases).
        key_cl_phases = np.array([-0.25, -45.0, -90.0, -180.0, -270.0,
                                  -325.0, -359.75])
        if np.abs(ol_phase_max - ol_phase_min) < 90.0:
            other_cl_phases = np.arange(-10.0, -360.0, -10.0)
        else:
            other_cl_phases = np.arange(-10.0, -360.0, -20.0)
        cl_phases = np.concatenate((key_cl_phases, other_cl_phases))
    else:
        assert ((-360.0 < np.min(cl_phases)) and (np.max(cl_phases) < 0.0))

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
    phase_offset_min = 360.0*np.ceil(ol_phase_min/360.0)
    phase_offset_max = 360.0*np.ceil(ol_phase_max/360.0) + 360.0
    phase_offsets = np.arange(phase_offset_min, phase_offset_max, 360.0)

    for phase_offset in phase_offsets:
        # Draw M and N contours
        plt.plot(m_phase + phase_offset, m_mag, color='lightgray',
                 linestyle=line_style, zorder=0)
        plt.plot(n_phase + phase_offset, n_mag, color='lightgray',
                 linestyle=line_style, zorder=0)

        # Add magnitude labels
        for x, y, m in zip(m_phase[:][-1] + phase_offset, m_mag[:][-1],
                           cl_mags):
            align = 'right' if m < 0.0 else 'left'
            plt.text(x, y, str(m) + ' dB', size='small', ha=align,
                     color='gray')

    # Fit axes to generated chart
    plt.axis([phase_offset_min - 360.0, phase_offset_max - 360.0,
              np.min(cl_mags), np.max([ol_mag_max, default_ol_mag_max])])

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
    Gcl_mags : array-like
        Array of magnitudes of the contours
    Gcl_phases : array-like
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
    mags : array-like
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
    phases : array-like
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
