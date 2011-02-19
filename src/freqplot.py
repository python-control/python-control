# freqplot.py - frequency domain plots for control systems
#
# Author: Richard M. Murray
# Date: 24 May 09
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
# $Id$

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from ctrlutil import unwrap
from bdalg import feedback

#
# Main plotting functions
#
# This section of the code contains the functions for generating
# frequency domain plots
#
   
# Bode plot
def bode(syslist, omega=None, dB=False, Hz=False):
    """Bode plot for a system

    Usage
    =====
    (magh, phaseh) = bode(syslist, omega=None, dB=False, Hz=False)

    Plots a Bode plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : linsys
        List of linear input/output systems (single system is OK)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec
    dB : boolean
        If True, plot result in dB
    Hz : boolean
        If True, plot frequency in Hz (omega must be provided in rad/sec)

    Return values
    -------------
    magh : graphics handle to magnitude plot (for rescaling, etc)
    phaseh : graphics handle to phase plot

    Notes
    -----
    1. Use (mag, phase, freq) = sys.freqresp(freq) to generate the 
       frequency response for a system.
    """
    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)

    # Select a default range if none is provided
    if (omega == None):
        omega = default_frequency_range(syslist)

    for sys in syslist:
        # Get the magnitude and phase of the system
        mag, phase, omega = sys.freqresp(omega)
        if Hz: omega = omega/(2*sp.pi)
        if dB: mag = 20*sp.log10(mag)
        phase = unwrap(phase*180/sp.pi, 360)

        # Get the dimensions of the current axis, which we will divide up
        #! TODO: Not current implemented; just use subplot for now

        # Magnitude plot
        plt.subplot(211); 
        if dB:
            plt.semilogx(omega, mag)
            plt.ylabel("Magnitude (dB)")
        else:
            plt.loglog(omega, mag)
            plt.ylabel("Magnitude")

        # Add a grid to the plot
        plt.grid(True)
        plt.grid(True, which='minor')
        plt.hold(True);

        # Phase plot
        plt.subplot(212);
        plt.semilogx(omega, phase)
        plt.hold(True)

        # Add a grid to the plot
        plt.grid(True)
        plt.grid(True, which='minor')
        plt.ylabel("Phase (deg)")

        # Label the frequency axis
        if Hz:
            plt.xlabel("Frequency (Hz)")
        else:
            plt.xlabel("Frequency (rad/sec)")

    return (211, 212)

# Nyquist plot
def nyquist(syslist, omega=None):
    """Nyquist plot for a system

    Usage
    =====
    magh = nyquist(sys, omega=None)

    Plots a Nyquist plot for the system over a (optional) frequency range.

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
        mag, phase, omega = sys.freqresp(omega)
    
        # Compute the primary curve
        x = sp.multiply(mag, sp.cos(phase));
        y = sp.multiply(mag, sp.sin(phase));
    
        # Plot the primary curve and mirror image
        plt.plot(x, y, '-');
        plt.plot(x, -y, '--');

    # Mark the -1 point
    plt.plot([-1], [0], 'r+')

# Nyquist grid
#! TODO: Consider making linestyle configurable
def nyquist_grid(cl_mags=None, cl_phases=None):
    """Nyquist plot grid of M-circles and N-circles (aka "Hall chart")

    Usage
    =====
    nyquist_grid()

    Plots a grid of M-circles and N-circles on the current axis, or
    creates a default grid if no plot already exists.

    Parameters
    ----------
    cl_mags : array-like (dB)
        Array of closed-loop magnitudes defining a custom set of
        M-circle iso-gain lines.
    cl_phases : array-like (degrees)
        Array of closed-loop phases defining a custom set of
        N-circle iso-phase lines. Must be in the range -180.0 < cl_phases < 180.0

    Return values
    -------------
    None
    """
    # Default chart size
    re_min = -4.0
    re_max = 3.0
    im_min = -2.0
    im_max = 2.0

    # Find bounds of the current dataset, if there is one.
    if plt.gcf().gca().has_data():
        re_min, re_max, im_min, im_max = plt.axis()

    # M-circle magnitudes.
    if cl_mags is None:
        cl_mags = np.array([-20.0, -10.0, -6.0, -4.0, -2.0, 0.0,
                            2.0, 4.0, 6.0, 10.0, 20.0])

    # N-circle phases (should be in the range -180.0 to 180.0)
    if cl_phases is None:
        cl_phases = np.array([-90.0, -60.0, -45.0, -30.0, -15.0,
                              15.0, 30.0, 45.0, 60.0, 90.0])
    else:
        assert ((-180.0 < np.min(cl_phases)) and (np.max(cl_phases) < 180.0))

    # Find the M-contours and N-contours
    m = m_circles(cl_mags, phase_min=0.0, phase_max=359.99)
    n = n_circles(cl_phases, mag_min=-40.0, mag_max=40.0)

    # Draw contours
    plt.plot(np.real(m), np.imag(m), color='gray', linestyle='dotted', zorder=0)
    plt.plot(np.real(n), np.imag(n), color='gray', linestyle='dotted', zorder=0)

    # Add magnitude labels
    for i in range(0, len(cl_mags)):
        if not cl_mags[i] == 0.0:
            mag = 10.0**(cl_mags[i]/20.0)
            x = -mag**2.0/(mag**2.0 - 1.0)   # Center of the M-circle
            y = np.abs(mag/(mag**2.0 - 1.0)) # Maximum point
        else:
            x, y = -0.5, im_max
        plt.text(x, y, str(cl_mags[i]) + ' dB', size='small', color='gray')

    # Add phase labels
    for i in range(0, len(cl_phases)):
        y = np.sign(cl_phases[i])*np.max(np.abs(np.imag(n)[:,i]))
        p = str(cl_phases[i])
        plt.text(-0.5, y, p + '$^\circ$', size='small', color='gray')
        
    # Fit axes to original plot
    plt.axis([re_min, re_max, im_min, im_max])

# Make an alias
hall_grid = nyquist_grid

# Nichols plot
# Contributed by Allan McInnes <Allan.McInnes@canterbury.ac.nz>
#! TODO: need unit test code
def nichols(syslist, omega=None, grid=True):
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
    grid : boolean
        True if the plot should include a Nichols-chart grid

    Return values
    -------------
    None
    """

    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)

    # Select a default range if none is provided
    if omega is None:
        omega = default_frequency_range(syslist)

    for sys in syslist:
        # Get the magnitude and phase of the system
        mag, phase, omega = sys.freqresp(omega)
    
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

    # Add grid
    if grid:
        nichols_grid()
    
# Nichols grid
#! TODO: Consider making linestyle configurable
def nichols_grid(cl_mags=None, cl_phases=None):
    """Nichols chart grid
    
    Usage
    =====
    nichols_grid()

    Plots a Nichols chart grid on the current axis, or creates a new chart
    if no plot already exists.

    Parameters
    ----------
    cl_mags : array-like (dB)
        Array of closed-loop magnitudes defining the iso-gain lines on a
        custom Nichols chart.
    cl_phases : array-like (degrees)
        Array of closed-loop phases defining the iso-phase lines on a custom
        Nichols chart. Must be in the range -360 < cl_phases < 0

    Return values
    -------------
    None    
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
        key_cl_mags = np.array([-40.0, -20.0, -12.0, -6.0, -3.0, -1.0, -0.5, 0.0,
                                     0.25, 0.5, 1.0, 3.0, 6.0, 12.0])
        # Extend the range of magnitudes if necessary. The extended arange
        # will end up empty if no extension is required. Assumes that closed-loop
        # magnitudes are approximately aligned with open-loop magnitudes beyond
        # the value of np.min(key_cl_mags)
        cl_mag_step = -20.0 # dB
        extended_cl_mags = np.arange(np.min(key_cl_mags),
                                     ol_mag_min + cl_mag_step, cl_mag_step)
        cl_mags = np.concatenate((extended_cl_mags, key_cl_mags))
                     
    # N-circle phases (should be in the range -360 to 0)
    if cl_phases is None:
        # Choose a reasonable set of default phases (denser if the open-loop
        # data is restricted to a relatively small range of phases).
        key_cl_phases = np.array([-0.25, -45.0, -90.0, -180.0, -270.0, -325.0, -359.75])
        if np.abs(ol_phase_max - ol_phase_min) < 90.0:
            other_cl_phases = np.arange(-10.0, -360.0, -10.0)
        else:
            other_cl_phases = np.arange(-10.0, -360.0, -20.0)
        cl_phases = np.concatenate((key_cl_phases, other_cl_phases))
    else:
        assert ((-360.0 < np.min(cl_phases)) and (np.max(cl_phases) < 0.0))

    # Find the M-contours
    m = m_circles(cl_mags, phase_min=np.min(cl_phases), phase_max=np.max(cl_phases))
    m_mag = 20*sp.log10(np.abs(m))
    m_phase = sp.mod(sp.degrees(sp.angle(m)), -360.0) # Unwrap

    # Find the N-contours
    n = n_circles(cl_phases, mag_min=np.min(cl_mags), mag_max=np.max(cl_mags))
    n_mag = 20*sp.log10(np.abs(n))
    n_phase = sp.mod(sp.degrees(sp.angle(n)), -360.0) # Unwrap

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
        plt.plot(m_phase + phase_offset, m_mag, color='gray',
                 linestyle='dotted', zorder=0)
        plt.plot(n_phase + phase_offset, n_mag, color='gray',
                 linestyle='dotted', zorder=0)

        # Add magnitude labels
        for x, y, m in zip(m_phase[:][-1] + phase_offset, m_mag[:][-1], cl_mags):
            align = 'right' if m < 0.0 else 'left'
            plt.text(x, y, str(m) + ' dB', size='small', ha=align, color='gray')

    # Fit axes to generated chart
    plt.axis([phase_offset_min - 360.0, phase_offset_max - 360.0,
              np.min(cl_mags), np.max([ol_mag_max, default_ol_mag_max])])

# Gang of Four
#! TODO: think about how (and whether) to handle lists of systems
def gangof4(P, C, omega=None):
    """Plot the "Gang of 4" transfer functions for a system

    Usage
    =====
    gangof4(P, C, omega=None)

    Generates a 2x2 plot showing the "Gang of 4" sensitivity functions
    [T, PS; CS, S]

    Parameters
    ----------
    P, C : linsys
        Linear input/output systems (process and control)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec

    Return values
    -------------
    None
    """

    # Select a default range if none is provided
    #! TODO: This needs to be made more intelligent
    if (omega == None):
        omega = default_frequency_range((P,C))

    # Compute the senstivity functions
    L = P*C;
    S = feedback(1, L);
    T = L * S;

    # Plot the four sensitivity functions
    #! TODO: Need to add in the mag = 1 lines
    mag, phase, omega = T.freqresp(omega);
    plt.subplot(221); plt.loglog(omega, mag);

    mag, phase, omega = (P*S).freqresp(omega);
    plt.subplot(222); plt.loglog(omega, mag);

    mag, phase, omega = (C*S).freqresp(omega);
    plt.subplot(223); plt.loglog(omega, mag);

    mag, phase, omega = S.freqresp(omega);
    plt.subplot(224); plt.loglog(omega, mag);

#
# Utility functions
#
# This section of the code contains some utility functions for
# generating frequency domain plots
#
   
# Compute reasonable defaults for axes
def default_frequency_range(syslist):
    """Compute a reasonable default frequency range for frequency
    domain plots.

    Usage
    =====
    omega = default_frequency_range(syslist)

    Finds a reasonable default frequency range by examining the features
    (poles and zeros) of the systems in syslist.

    Parameters
    ----------
    syslist : linsys
        List of linear input/output systems (single system is OK)

    Return values
    -------------
    omega : freq_range
        Range of frequencies in rad/sec
    """
    # This code looks at the poles and zeros of all of the systems that
    # we are plotting and sets the frequency range to be one decade above
    # and below the min and max feature frequencies, rounded to the nearest
    # integer.  It excludes poles and zeros at the origin.  If no features
    # are found, it turns logspace(-1, 1)
    
    # Find the list of all poles and zeros in the systems
    features = np.array(())
    for sys in syslist:
        # Add new features to the list
        features = np.concatenate((features, np.abs(sys.poles)))
        features = np.concatenate((features, np.abs(sys.zeros)))

    # Get rid of poles and zeros at the origin
    features = features[features != 0];

    # Make sure there is at least one point in the range
    if (features.shape[0] == 0): features = [1];

    # Take the log of the features
    features = np.log10(features)

    # Set the range to be an order of magnitude beyond any features
    omega = sp.logspace(np.floor(np.min(features))-1, 
                        np.ceil(np.max(features))+1)   
                        
    return omega

# Compute contours of a closed-loop transfer function
def closed_loop_contours(Gcl_mags, Gcl_phases):
    """Contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Usage
    =====
    contours = closed_loop_contours(Gcl_mags, Gcl_phases)

    Parameters
    ----------
    Gcl_mags : array-like
        Array of magnitudes of the contours
    Gcl_phases : array-like
        Array of phases in radians of the contours

    Return values
    -------------
    contours : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Compute the contours in Gcl-space. Since we're given closed-loop
    # magnitudes and phases, this is just a case of converting them into
    # a complex number.
    Gcl = Gcl_mags*sp.exp(1.j*Gcl_phases)

    # Invert Gcl = Gol/(1+Gol) to map the contours into the open-loop space
    return Gcl/(1.0 - Gcl)

# M-circle
def m_circles(mags, phase_min=-359.75, phase_max=-0.25):
    """Constant-magnitude contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Usage
    =====
    contours = m_circles(mags, phase_min, phase_max)

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
    phases = sp.radians(sp.linspace(phase_min, phase_max, 2000))
    Gcl_mags, Gcl_phases = sp.meshgrid(10.0**(mags/20.0), phases)
    return closed_loop_contours(Gcl_mags, Gcl_phases)

# N-circle
def n_circles(phases, mag_min=-40.0, mag_max=12.0):
    """Constant-phase contours of the function Gcl = Gol/(1+Gol), where
    Gol is an open-loop transfer function, and Gcl is a corresponding
    closed-loop transfer function.

    Usage
    ===== 
    contours = n_circles(phases, mag_min, mag_max)

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
    Gcl_phases, Gcl_mags = sp.meshgrid(sp.radians(phases), mags)
    return closed_loop_contours(Gcl_mags, Gcl_phases)
