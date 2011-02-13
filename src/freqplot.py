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

# Nichols plot
# Contributed by Allan McInnes <Allan.McInnes@canterbury.ac.nz>
#! TODO: need unit test code
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
    gmin_default = -40.0 # dB
    gain_step = 20.0     # dB

    if plt.gcf().gca().has_data():
        pmin, pmax, gmin, gmax = plt.axis()
    else:
        pmin, pmax = -360.0, 0.0
        gmin, gmax = gmin_default, 40.0

    # Determine the bounds of the chart
    mags_low_end = np.min([gmin_default, gain_step*np.floor(gmin/gain_step)])
    phases_low_end = 360.0*np.ceil(pmin/360.0)
    phases_high_end = 360.0*(1.0 + np.ceil(pmax/360.0))

    # M-circle magnitudes - we adjust the lower end of the range to match
    # any existing data
    mags_fixed = np.array([-20.0, -12.0, -6.0, -3.0, -1.0, -0.5, 0.0,
                                 0.25, 0.5, 1.0, 3.0, 6.0, 12.0])
    mags_adjustable = np.arange(mags_low_end, np.min(mags_fixed), gain_step)
    mags = np.concatenate((mags_adjustable, mags_fixed))
                     
    # N-circle phases (should be in the range -360 to 0)
    phases = np.array([-0.25, -10.0, -20.0, -30.0, -45.0, -60.0, -90.0,
                       -120.0, -150.0, -180.0, -210.0, -240.0, -270.0,
                       -310.0, -325.0, -340.0, -350.0, -359.75])
    # Find the M-contours
    mcs = m_circles(mags, pmin=np.min(phases), pmax=np.max(phases))
    mag_m = 20*sp.log10(np.abs(mcs))
    phase_m = sp.mod(sp.degrees(sp.angle(mcs)), -360.0) # Unwrap

    # Find the N-contours
    ncs = n_circles(phases, gmin=np.min(mags), gmax=np.max(mags))
    mag_n = 20*sp.log10(np.abs(ncs))
    phase_n = sp.mod(sp.degrees(sp.angle(ncs)), -360.0) # Unwrap

    # Plot the contours
    for phase_offset in np.arange(phases_low_end, phases_high_end, 360.0):
        plt.plot(phase_m + phase_offset, mag_m, color='gray',
                 linestyle='dashed', zorder=0)
        plt.plot(phase_n + phase_offset, mag_n, color='gray',
                 linestyle='dashed', zorder=0)

    # Add magnitude labels
    for x, y, m in zip(phase_m[:][-1], mag_m[:][-1], mags):
        align = 'right' if m < 0.0 else 'left'
        plt.text(x, y, str(m) + ' dB', size='small', ha=align)

    # Make sure axes conform to any pre-existing plot.
    plt.axis([pmin, pmax, gmin, gmax])

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

# M-circle
def m_circles(mags, pmin=-359.75, pmax=-0.25):
    """Constant-magnitude contours of the function H = G/(1+G).

    Usage
    =====
    contour = m_circle(mags)

    Parameters
    ----------
    mags : array-like
        Array of magnitudes in dB of the M-circles

    Return values
    -------------
    contour : complex array
        Array of complex numbers corresponding to the contour.
    """
    # Compute the contours in H-space
    phases = sp.radians(sp.linspace(pmin, pmax, 500))
    Hmag, Hphase = sp.meshgrid(10.0**(mags/20.0), phases)
    H = Hmag*sp.exp(1.j*Hphase)
    
    # Invert H = G/(1+G) to get an expression for the contour in G-space
    return H/(1.0 - H)

# N-circle
def n_circles(phases, gmin=-40.0, gmax=12.0):
    """Constant-phase contours of the function H = G/(1+G).

    Usage
    ===== 
    contour = n_circle(angles)

    Parameters
    ----------
    phases : array-like
        Array of phases in degrees of the N-circle

    Return values
    -------------
    contour : complex array
        Array of complex numbers corresponding to the contours.
    """
    # Compute the contours in H-space
    mags = sp.linspace(10**(gmin/20.0), 10**(gmax/20.0), 2000)
    Hphase, Hmag = sp.meshgrid(sp.radians(phases), mags)
    H = Hmag*sp.exp(1.j*Hphase)

    # Invert H = G/(1+G) to get an expression for the contours in G-space
    return H/(1.0 - H)
