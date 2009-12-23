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
# $Id: freqplot.py 819 2009-05-29 21:28:07Z murray $

import matplotlib.pyplot as plt
import scipy as sp
from ctrlutil import unwrap

# Bode plot
def bode(sys, omega=None):
    """Bode plot for a system

    Usage
    =====
    bode(sys, omega=None)

    Plots a Bode plot for the system over a (optional) frequency range.

    Parameters
    ----------
    sys : linsys
        Linear input/output system
    omega : freq_range
        Range of frequencies (list or bounds)
    """
    # Select a default range if none is provided
    if (omega == None):
        omega = sp.logspace(-2, 2);

    # Get the magnitude and phase of the system
    mag, phase, omega = sys.freqresp(omega)
    phase = unwrap(phase*180/sp.pi, 360)

    # Get the dimensions of the current axis, which we will divide up
    #! Not current implemented; just use subplot for now

    # Magnitude plot
    plt.subplot(211);
    plt.loglog(omega, mag)

    # Phase plot
    plt.subplot(212);
    plt.semilogx(omega, phase)

# Nyquist plot
def nyquist(sys, omega=None):
    # Select a default range if none is provided
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

