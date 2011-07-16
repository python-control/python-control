"""margin.py

Functions for computing stability margins and related functions.

Routeins in this module:

margin.StabilityMargins
margin.PhaseCrossoverFrequencies
"""

"""Copyright (c) 2011 by California Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Richard M. Murray
Date: 14 July 2011

$Id: xferfcn.py 165 2011-06-26 02:44:09Z murrayrm $

"""

import xferfcn
from freqplot import bode
import numpy as np

# gain and phase margins
# contributed by Sawyer B. Fuller <minster@caltech.edu>
#! TODO - need to add unit test functions
def StabilityMargins(sysdata, deg=True):
    """Calculate gain, phase and stability margins and associated
    crossover frequencies.

    Usage:
    
    gm, pm, sm, wg, wp, ws = StabilityMargins(sysdata, deg=True)
    
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

# Contributed by Steffen Waldherr <waldherr@ist.uni-stuttgart.de>
#! TODO - need to add test functions
def PhaseCrossoverFrequencies(sys):
    """
    Compute frequencies and gains at intersections with real axis
    in Nyquist plot.

    Call as:
        omega, gain = PhaseCrossoverFrequencies()

    Returns
    -------
    omega: 1d array of (non-negative) frequencies where Nyquist plot
    intersects the real axis

    gain: 1d array of corresponding gains
        
    Examples
    --------
    >>> tf = TransferFunction([1], [1, 2, 3, 4])
    >>> PhaseCrossoverFrequenies(tf)
    (array([ 1.73205081,  0.        ]), array([-0.5 ,  0.25]))
    """

    # Convert to a transfer function
    tf = xferfcn._convertToTransferFunction(sys)

    # if not siso, fall back to (0,0) element
    #! TODO: should add a check and warning here
    num = tf.num[0][0]
    den = tf.den[0][0]

    # Compute frequencies that we cross over the real axis
    numj = (1.j)**np.arange(len(num)-1,-1,-1)*num
    denj = (-1.j)**np.arange(len(den)-1,-1,-1)*den
    allfreq = np.roots(np.imag(np.polymul(numj,denj)))
    realfreq = np.real(allfreq[np.isreal(allfreq)])
    realposfreq = realfreq[realfreq >= 0.]

    # using real() to avoid rounding errors and results like 1+0j
    # it would be nice to have a vectorized version of self.evalfr here
    gain = np.real(np.asarray([tf.evalfr(f)[0][0] for f in realposfreq]))

    return realposfreq, gain
