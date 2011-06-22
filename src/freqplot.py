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
def bode(syslist, omega=None, dB=False, Hz=False, deg=True, 
        color=None, Plot=True):
    """Bode plot for a system

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
    color : matplotlib color
        Color of line in bode plot
    deg : boolean
        If True, return phase in degrees (else radians)
    Plot : boolean
        If True, plot magnitude and phase

    Returns
    -------
    mag : array (list if len(syslist) > 1)
        magnitude
    phase : array (list if len(syslist) > 1)
        phase
    omega : array (list if len(syslist) > 1)
        frequency
    
    Notes
    -----
    1. Alternatively, you may use the lower-level method 
    (mag, phase, freq) = sys.freqresp(freq) to generate the frequency 
    response for a system, but it returns a MIMO response.

    Examples
    --------
    >>> from matlab import ss
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> mag, phase, omega = bode(sys)
    """
    # If argument was a singleton, turn it into a list
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)

    mags, phases, omegas = [], [], []
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
            phase = unwrap(phase)
            if Hz: omega = omega/(2*sp.pi)
            if dB: mag = 20*sp.log10(mag)
            if deg: phase = phase * 180 / sp.pi
            
            mags.append(mag)
            phases.append(phase)
            omegas.append(omega)
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
                if deg: 
                    phase_deg = phase
                else:
                    phase_deg = phase * 180 / sp.pi
                if color==None:
                    plt.semilogx(omega, phase_deg)
                else:
                    plt.semilogx(omega, phase_deg, color=color)
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

    if len(syslist) == 1:
        return mags[0], phases[0], omegas[0]
    else:
        return mags, phases, omegas

# Nyquist plot
def nyquist(syslist, omega=None, Plot=True):
    """Nyquist plot for a system

    Plots a Nyquist plot for the system over a (optional) frequency range.

    Parameters
    ----------
    syslist : list of Lti
        List of linear input/output systems (single system is OK)
    omega : freq_range
        Range of frequencies (list or bounds) in rad/sec
    Plot : boolean
        if True, plot magnitude

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
    >>> from matlab import ss
    >>> sys = ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> real, imag, freq = nyquist(sys)
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

    Generates a 2x2 plot showing the "Gang of 4" sensitivity functions
    [T, PS; CS, S]

    Parameters
    ----------
    P, C : Lti
        Linear input/output systems (process and control)
    omega : array
        Range of frequencies (list or bounds) in rad/sec

    Returns
    -------
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


# gain and phase margins
# contributed by Sawyer B. Fuller <minster@caltech.edu>
def margin(sysdata, deg=True):
    """Calculate gain and phase margins and associated crossover frequencies

    Usage:
    
    gm, pm, sm, wg, wp, ws = margin(sysdata, deg=True)
    
    Parameters
    ----------
    sysdata: linsys or (mag, phase, omega) sequence 
        sys : linsys
            Linear SISO system
        mag, phase, omega : sequence of array_like
            Input magnitude, phase, and frequencies (rad/sec) sequence from 
            bode frequency response data 
    deg=True: boolean  
        If true, all input and output phases in degrees, else in radians
        
    Returns
    -------
    gm, pm, sm, wg, wp, ws: float
        Gain margin gm, phase margin pm, stability margin sm, and 
        associated crossover
        frequencies wg, wp, and ws of SISO open-loop. If more than
        one crossover frequency is detected, returns the lowest corresponding
        margin. 
    """
    #TODO do this precisely without the effects of discretization of frequencies?
    #TODO assumes SISO
    #TODO unit tests, margin plot

    if (not getattr(sysdata, '__iter__', False)):
        sys = sysdata
        mag, phase, omega = bode(sys, deg=deg, Plot=False)
    elif len(sysdata) == 3:
        mag, phase, omega = sysdata
    else: 
        raise ValueError("Margin sysdata must be either a linear system or a 3-sequence of mag, phase, omega.")
        
    if deg:
        cycle = 360. 
        crossover = 180. 
    else:
        cycle = 2 * np.pi
        crossover = np.pi
        
    wrapped_phase = -np.mod(phase, cycle)
    
    # phase margin from minimum phase among all gain crossovers
    neg_mag_crossings_i = np.nonzero(np.diff(mag < 1) > 0)[0]
    mag_crossings_p = wrapped_phase[neg_mag_crossings_i]
    if len(neg_mag_crossings_i) == 0:
        if mag[0] < 1: # gain always less than one
            wp = np.nan
            pm = np.inf
        else: # gain always greater than one
            print "margin: no magnitude crossings found"
            wp = np.nan
            pm = np.nan
    else:
        min_mag_crossing_i = neg_mag_crossings_i[np.argmin(mag_crossings_p)]
        wp = omega[min_mag_crossing_i]
        pm = crossover + phase[min_mag_crossing_i] 
        if pm < 0:
            print "warning: system unstable: negative phase margin"
    
    # gain margin from minimum gain margin among all phase crossovers
    neg_phase_crossings_i = np.nonzero(np.diff(wrapped_phase < -crossover) > 0)[0]
    neg_phase_crossings_g = mag[neg_phase_crossings_i]
    if len(neg_phase_crossings_i) == 0:
        wg = np.nan
        gm = np.inf
    else:
        min_phase_crossing_i = neg_phase_crossings_i[
            np.argmax(neg_phase_crossings_g)]
        wg = omega[min_phase_crossing_i]
        gm = abs(1/mag[min_phase_crossing_i])
        if gm < 1: 
            print "warning: system unstable: gain margin < 1"    

    # stability margin from minimum abs distance from -1 point
    if deg:
        phase_rad = phase * np.pi / 180.
    else:
        phase_rad = phase
    L = mag * np.exp(1j * phase_rad) # complex loop response to -1 pt
    min_Lplus1_i = np.argmin(np.abs(L + 1))
    sm = np.abs(L[min_Lplus1_i] + 1)
    ws = phase[min_Lplus1_i]

    return gm, pm, sm, wg, wp, ws 

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

    Finds a reasonable default frequency range by examining the features
    (poles and zeros) of the systems in syslist.

    Parameters
    ----------
    syslist : list of Lti
        List of linear input/output systems (single system is OK)

    Return
    ------
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
    
    # Find the list of all poles and zeros in the systems
    features = np.array(())
    
    # detect if single sys passed by checking if it is sequence-like
    if (not getattr(syslist, '__iter__', False)):
        syslist = (syslist,)
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

