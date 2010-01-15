# freqplot.py - frequency domain plots for control systems
#
# Author: Richard M. Murray
# Date: 24 May 09
# 
# This file contains some standard control system plots: Bode plots,
# Nyquist plots and pole-zero diagrams
#
# Copyright (c) 2009 by California Institute of Technology
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
from ctrlutil import unwrap
from bdalg import feedback

# Bode plot
def bode(sys, omega=None, dB=False, Hz=False):
    """Bode plot for a system

    Usage
    =====
    (magh, phaseh) = bode(sys, omega=None, dB=False, Hz=False)

    Plots a Bode plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys : linsys
        Linear input/output system
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
    # Select a default range if none is provided
    if (omega == None):
        omega = sp.logspace(-2, 2);

    # Get the magnitude and phase of the system
    mag, phase, omega = sys.freqresp(omega)
    if Hz:
        omega = omega/(2*sp.pi)
    if dB:
        mag = 20*sp.log10(mag)
    phase = unwrap(phase*180/sp.pi, 360)

    # Get the dimensions of the current axis, which we will divide up
    #! Not current implemented; just use subplot for now

    # Magnitude plot
    plt.subplot(211);
    if dB:
        plt.semilogx(omega, mag)
    else:
        plt.loglog(omega, mag)
    plt.grid(True)
    plt.grid(True, which='minor')
    if dB:
        plt.ylabel("Magnitude (dB)")
    else:
        plt.ylabel("Magnitude")

    # Phase plot
    plt.subplot(212);
    plt.semilogx(omega, phase)
    plt.grid(True)
    plt.grid(True, which='minor')
    plt.ylabel("Phase (deg)")
    if Hz:
        plt.xlabel("Frequency (Hz)")
    else:
        plt.xlabel("Frequency (rad/sec)")

    return (211, 212)

# Nyquist plot
def nyquist(sys, omega=None):
    # Select a default range if none is provided
    #! This needs to be made more intelligent
    if (omega == None):
        omega = sp.logspace(-2, 2);

    # Get the magnitude and phase of the system
    mag, phase, omega = sys.freqresp(omega)

    # Compute the primary curve
    x = sp.multiply(mag, sp.cos(phase));
    y = sp.multiply(mag, sp.sin(phase));

    # Plot the primary curve and mirror image
    plt.plot(x, y, '-');
    plt.plot(x, -y, '--');

    # Mark the -1 point
    plt.plot([-1], [0], '+k')

# Gang of Four
def gangof4(P, C, omega=None):
    # Select a default range if none is provided
    #! This needs to be made more intelligent
    if (omega == None):
        omega = sp.logspace(-2, 2);

    # Compute the senstivity functions
    L = P*C;

    #! Replace with feedback when written
    S = feedback(1, L);
    T = L * S;

    # Plot the four sensitivity functions
    #! Need to add in the mag = 1 lines
    mag, phase, omega = T.freqresp(omega);
    plt.subplot(221); plt.loglog(omega, mag);

    mag, phase, omega = (P*S).freqresp(omega);
    plt.subplot(222); plt.loglog(omega, mag);

    mag, phase, omega = (C*S).freqresp(omega);
    plt.subplot(223); plt.loglog(omega, mag);

    mag, phase, omega = S.freqresp(omega);
    plt.subplot(224); plt.loglog(omega, mag);
