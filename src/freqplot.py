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
def bode(syslist, omega=None, dB=False, Hz=False, color=None, Plot=True):
    """Bode plot for a system

    Usage
    =====
    (magh, phaseh, omega) = bode(syslist, omega=None, dB=False, Hz=False, color=None, Plot=True)

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
    Plot : boolean
        If True, plot magnitude and phase

    Return values
    -------------
    magh : magnitude array
    phaseh : phase array
    omega : frequency array

    Notes
    -----
    1. Alternatively, may use (mag, phase, freq) = sys.freqresp(freq) to generate the frequency response for a system.
    """
    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)

    for sys in syslist:
        if (sys.inputs > 1 or sys.outputs > 1):
            #TODO: Add MIMO bode plots. 
            raise NotImplementedError("Bode is currently only implemented for SISO systems.")
        else:
            # Select a default range if none is provided
            if (omega == None):
                omega = default_frequency_range(syslist)

            # Get the magnitude and phase of the system
            mag_tmp, phase_tmp, omega = sys.freqresp(omega)
            mag = np.squeeze(mag_tmp)
            phase = np.squeeze(phase_tmp)
            if Hz: omega = omega/(2*sp.pi)
            if dB: mag = 20*sp.log10(mag)
            phase = unwrap(phase*180/sp.pi, 360)

            # Get the dimensions of the current axis, which we will divide up
            #! TODO: Not current implemented; just use subplot for now

            if (Plot):
                # Magnitude plot
                plt.subplot(211); 
                if dB:
                    if color==None:
                        plt.semilogx(omega, mag)
                    else:
                        plt.semilogx(omega, mag, color=color)
                    plt.ylabel("Magnitude (dB)")
                else:
                    if color==None:
                        plt.loglog(omega, mag)
                    else: 
                        plt.loglog(omega, mag, color=color) 
                    plt.ylabel("Magnitude")

                # Add a grid to the plot
                plt.grid(True)
                plt.grid(True, which='minor')
                plt.hold(True);

                # Phase plot
                plt.subplot(212);
                if color==None:
                    plt.semilogx(omega, phase)
                else:
                    plt.semilogx(omega, phase, color=color)
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

    return mag, phase, omega

# Nyquist plot
def nyquist(syslist, omega=None, Plot=True):
    """Nyquist plot for a system

    Usage
    =====
    real, imag, freq = nyquist(sys, omega=None, Plot=True)

    Plots a Nyquist plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : linsys
        List of linear input/output systems (single system is OK)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec
    Plot : boolean
        if True, plot magnitude

    Return values
    -------------
    real : real part of the frequency response array
    imag : imaginary part of the frequency response array
    freq : frequencies
    """
    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)
        
    # Select a default range if none is provided
    if (omega == None):
        omega = default_frequency_range(syslist)
    # Interpolate between wmin and wmax if a tuple or list are provided
    elif (isinstance(omega,list) | isinstance(omega,tuple)):
        # Only accept tuple or list of length 2
        if (len(omega) != 2):
            raise ValueError("Supported frequency arguments are (wmin,wmax) tuple or list, or frequency vector. ")
        omega = np.logspace(np.log10(omega[0]),np.log10(omega[1]),num=50,endpoint=True,base=10.0)
    for sys in syslist:
        if (sys.inputs > 1 or sys.outputs > 1):
            #TODO: Add MIMO nyquist plots. 
            raise NotImplementedError("Nyquist is currently only implemented for SISO systems.")
        else:
            # Get the magnitude and phase of the system
            mag_tmp, phase_tmp, omega = sys.freqresp(omega)
            mag = np.squeeze(mag_tmp)
            phase = np.squeeze(phase_tmp)
 
            # Compute the primary curve
            x = sp.multiply(mag, sp.cos(phase));
            y = sp.multiply(mag, sp.sin(phase));

            if (Plot):    
                # Plot the primary curve and mirror image
                plt.plot(x, y, '-');
                plt.plot(x, -y, '--');
                # Mark the -1 point
                plt.plot([-1], [0], 'r+')

        return x, y, omega

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
    if (P.inputs > 1 or P.outputs > 1 or C.inputs > 1 or C.outputs >1):
        #TODO: Add MIMO go4 plots. 
        raise NotImplementedError("Gang of four is currently only implemented for SISO systems.")
    else:
 
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
        mag_tmp, phase_tmp, omega = T.freqresp(omega);
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)
        plt.subplot(221); plt.loglog(omega, mag);

        mag_tmp, phase_tmp, omega = (P*S).freqresp(omega);
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)
        plt.subplot(222); plt.loglog(omega, mag);

        mag_tmp, phase_tmp, omega = (C*S).freqresp(omega);
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)
        plt.subplot(223); plt.loglog(omega, mag);

        mag_tmp, phase_tmp, omega = S.freqresp(omega);
        mag = np.squeeze(mag_tmp)
        phase = np.squeeze(phase_tmp)
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
        features = np.concatenate((features, np.abs(sys.pole())))
        features = np.concatenate((features, np.abs(sys.zero())))

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

