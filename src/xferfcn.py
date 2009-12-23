# xferfcn.py - transfer function class and functions
#
# Author: Richard M. Murray
# Date: 24 May 09
#
# This file contains the TransferFunction class and also functions
# that operate on transfer functions.
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
# $Id: xferfcn.py 813 2009-05-29 18:08:08Z murray $

# External function declarations
import scipy as sp
import scipy.signal as signal
import bdalg as bd
from ctrlutil import unwrap

# Functions for creating a transfer function
def tf(num, den): 
    return TransferFunction(num, den)

def ss2tf(A, B, C, D):
    A, B, C, D = signal.abcd_normalize(A, B, C, D)

    # Save the size of the system
    num_states = A.shape[0]
    nout, nin = D.shape

    # Compute the denominator from the A matrix
    den = sp.poly(A)

    # Compute the numerator based on zeros
    #! Assumes single input/single output
    num = sp.poly(A - sp.dot(B, C)) + (D[0] - 1) * den
    
    # Now construct the transfer function
    return TransferFunction(num, den)

class TransferFunction(object):
    # Initialization with optional arguments
    def __init__ (self, num, den):
        # Save the numerator and denominator polynomials, for future use
        self.num = sp.poly1d(num, variable='s');
        self.den = sp.poly1d(den, variable='s');

    # Style to use for printing (similar to MATLAB)
    def __str__(self):
        # Convert the numerator and denominator polynomials to strings
        numstr = _tfpolyToString(self.num);
        denstr = _tfpolyToString(self.den);

        # Figure out the length of the separating line
        dashcount = max(len(numstr), len(denstr))
        dashes = '-' * dashcount

        # Center the numerator or denominator
        if (len(numstr) < dashcount):
            numstr = ' ' * int(round((dashcount - len(numstr))/2)) + numstr
        if (len(denstr) < dashcount): 
            denstr = ' ' * int(round((dashcount - len(denstr))/2)) + denstr

        # Put it all together
        return numstr + "\n" + dashes + "\n" + denstr

    # Negation of a transfer function
    def __neg__(self):
        return bd.negate(self)

    # Addition of two transfer functions (parallel interconnection)
    def __add__(self, other):
        return bd.parallel(self, other)

    # Addition of two transfer functions (parallel interconnection)
    def __sub__(self, other):
        return bd.parallel(self, other.__neg__())

    # Multiplication of two transfer functions (series interconnection)
    def __mul__(self, other):
        return bd.series(self, other)

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        # Generate the frequency response at each frequency
        fresp = map(self.evalfr, omega);
        mag = sp.sqrt(sp.multiply(fresp, sp.conjugate(fresp)));
        phase = unwrap(sp.angle(fresp)) * 180 / sp.pi;

        return mag, phase, omega

    # Method for evaluating a transfer function at one frequency
    def evalfr(self, freq):
        return self.num(freq*1j) / self.den(freq*1j)

# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tfpolyToString(poly):
    thestr = "0"
    var = poly.variable

    # Compute the number of coefficients
    coeffs = poly.coeffs
    N = len(coeffs)-1

    for k in range(len(coeffs)):
        coefstr ='%.4g' % abs(coeffs[k])
        if coefstr[-4:] == '0000':
            coefstr = coefstr[:-5]
        power = (N-k)
        if power == 0:
            if coefstr != '0':
                newstr = '%s' % (coefstr,)
            else:
                if k == 0:
                    newstr = '0'
                else:
                    newstr = ''
        elif power == 1:
            if coefstr == '0':
                newstr = ''
            elif coefstr == '1':
                newstr = var
            else:
                newstr = '%s %s' % (coefstr, var)
        else:
            if coefstr == '0':
                newstr = ''
            elif coefstr == '1':
                newstr = '%s^%d' % (var, power,)
            else:
                newstr = '%s %s^%d' % (coefstr, var, power)

        if k > 0:
            if newstr != '':
                if coeffs[k] < 0:
                    thestr = "%s - %s" % (thestr, newstr)
                else:
                    thestr = "%s + %s" % (thestr, newstr)
        elif (k == 0) and (newstr != '') and (coeffs[k] < 0):
            thestr = "-%s" % (newstr,)
        else:
            thestr = newstr
    return thestr
