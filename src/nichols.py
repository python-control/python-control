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

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from ctrlutil import unwrap
from freqplot import default_frequency_range

def nichols(syslist, omega=None):
    """Nichols plot for a system

    Usage
    =====
    magh = nichols(sys, omega=None)

    Plots a Nichols plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : linsys
        List of linear input/output systems (single system is OK)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec

    Return values
    -------------
    None
    """

    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)

    # Select a default range if none is provided
    if (omega == None):
        omega = default_frequency_range(syslist)

    for sys in syslist:
        # Get the magnitude and phase of the system
        mag_tmp, phase_tmp, omega = sys.freqresp(omega)
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)
    
        # Convert to Nichols-plot format (phase in degrees, 
        # and magnitude in dB)
        x = unwrap(sp.degrees(phase), 360)
        y = 20*sp.log10(mag)
    
        # Generate the plot
        plt.plot(x, y)
        
    plt.xlabel('Phase (deg)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Nichols Plot')

    # Mark the -180 point
    plt.plot([-180], [0], 'r+')
    
# Nichols grid
def nichols_grid():
    """Nichols chart grid
    
    Usage
    =====
    nichols_grid()

    Plots a Nichols chart grid on the current axis.

    Parameters
    ----------
    None

    Return values
    -------------
    None    
    """
    mag_min_default = -40.0 # dB
    mag_step = 20.0         # dB

    # Chart defaults
    phase_min, phase_max, mag_min, mag_max = -360.0, 0.0, mag_min_default, 40.0

    # Set actual chart bounds based on current plot
    if plt.gcf().gca().has_data():
        phase_min, phase_max, mag_min, mag_max = plt.axis()
        
    # M-circle magnitudes.
    # The "fixed" set are always generated, since this guarantees a recognizable
    # Nichols chart grid.
    mags_fixed = np.array([-40.0, -20.0, -12.0, -6.0, -3.0, -1.0, -0.5, 0.0,
                                 0.25, 0.5, 1.0, 3.0, 6.0, 12.0])

    if mag_min < mag_min_default:
        # Outside the "fixed" set of magnitudes, the generated M-circles
        # are extended in steps of 'mag_step' dB to cover anything made
        # visible by the range of the existing plot
        mags_adjust = np.arange(mag_step*np.floor(mag_min/mag_step),
                                mag_min_default, mag_step)
        mags = np.concatenate((mags_adjust, mags_fixed))
    else:
        mags = mags_fixed
                     
    # N-circle phases (should be in the range -360 to 0)
    phases = np.array([-0.25, -10.0, -20.0, -30.0, -45.0, -60.0, -90.0,
                       -120.0, -150.0, -180.0, -210.0, -240.0, -270.0,
                       -310.0, -325.0, -340.0, -350.0, -359.75])

    # Find the M-contours
    m = m_circles(mags, phase_min=np.min(phases), phase_max=np.max(phases))
    m_mag = 20*sp.log10(np.abs(m))
    m_phase = sp.mod(sp.degrees(sp.angle(m)), -360.0) # Unwrap

    # Find the N-contours
    n = n_circles(phases, mag_min=np.min(mags), mag_max=np.max(mags))
    n_mag = 20*sp.log10(np.abs(n))
    n_phase = sp.mod(sp.degrees(sp.angle(n)), -360.0) # Unwrap

    # Plot the contours behind other plot elements.
    # The "phase offset" is used to produce copies of the chart that cover
    # the entire range of the plotted data, starting from a base chart computed
    # over the range -360 < phase < 0 (see above). Given the range 
    # the base chart is computed over, the phase offset should be 0
    # for -360 < phase_min < 0.
    phase_offset_min = 360.0*np.ceil(phase_min/360.0)
    phase_offset_max = 360.0*np.ceil(phase_max/360.0) + 360.0
    phase_offsets = np.arange(phase_offset_min, phase_offset_max, 360.0)
    for phase_offset in phase_offsets:
        plt.plot(m_phase + phase_offset, m_mag, color='gray',
                 linestyle='dashed', zorder=0)
        plt.plot(n_phase + phase_offset, n_mag, color='gray',
                 linestyle='dashed', zorder=0)

    # Add magnitude labels
    for x, y, m in zip(m_phase[:][-1], m_mag[:][-1], mags):
        align = 'right' if m < 0.0 else 'left'
        plt.text(x, y, str(m) + ' dB', size='small', ha=align)

    # Make sure axes conform to any pre-existing plot.
    plt.axis([phase_min, phase_max, mag_min, mag_max])

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating Nichols plots
#
   
# Compute contours of a closed-loop transfer function
def closed_loop_contours(Hmag, Hphase):
    """Contours of the function H = G/(1+G).

    Usage
    =====
    contours = closed_loop_contours(mags, phases)

    Parameters
    ----------
    mags : array-like
        Meshgrid array of magnitudes of the contours
    phases : array-like
        Meshgrid array of phases in radians of the contours

    Return values
    -------------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Compute the contours in H-space
    H = Hmag*sp.exp(1.j*Hphase)

    # Invert H = G/(1+G) to get an expression for the contours in G-space
    return H/(1.0 - H)

# M-circle
def m_circles(mags, phase_min=-359.75, phase_max=-0.25):
    """Constant-magnitude contours of the function H = G/(1+G).

    Usage
    =====
    contours = m_circles(mags)

    Parameters
    ----------
    mags : array-like
        Array of magnitudes in dB of the M-circles
    phase_min : degrees
        Minimum phase in degrees of the N-circles
    phase_max : degrees
        Maximum phase in degrees of the N-circles

    Return values
    -------------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Convert magnitudes and phase range into a grid suitable for
    # building contours
    phases = sp.radians(sp.linspace(phase_min, phase_max, 500))
    Hmag, Hphase = sp.meshgrid(10.0**(mags/20.0), phases)
    return closed_loop_contours(Hmag, Hphase)

# N-circle
def n_circles(phases, mag_min=-40.0, mag_max=12.0):
    """Constant-phase contours of the function H = G/(1+G).

    Usage
    ===== 
    contour = n_circles(angles)

    Parameters
    ----------
    phases : array-like
        Array of phases in degrees of the N-circles
    mag_min : dB
        Minimum magnitude in dB of the N-circles
    mag_max : dB
        Maximum magnitude in dB of the N-circles

    Return values
    -------------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Convert phases and magnitude range into a grid suitable for
    # building contours    
    mags = sp.linspace(10**(mag_min/20.0), 10**(mag_max/20.0), 2000)
    Hphase, Hmag = sp.meshgrid(sp.radians(phases), mags)
    return closed_loop_contours(Hmag, Hphase)
