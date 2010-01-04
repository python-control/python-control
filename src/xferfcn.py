# xferfcn.py - transfer function class and functions
#
# Author: Richard M. Murray
# Date: 24 May 09
#
# This file contains the TransferFunction class and also functions
# that operate on transfer functions.  This class extends the
# signal.lti class by adding some additional useful functions like
# block diagram algebra.
#
# NOTE: Transfer function in this module are restricted to be SISO
# systems.  To perform calcualtiosn on MIMO systems, you need to use
# the state space module (statesp.py).
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

# External function declarations
import scipy as sp
import scipy.signal as signal
import bdalg as bd
import statesp

class TransferFunction(signal.lti):
    """The TransferFunction class is used to represent linear
    input/output systems via its transfer function.
    """
    # Constructor
    def __init__(self, *args, **keywords):
        # First initialize the parent object
        signal.lti.__init__(self, *args, **keywords)

        # Make sure that this is only a SISO function
        if (self.inputs != 1 or self.outputs != 1):
            raise NotImplementedError("MIMO transfer functions not supported")

        # Now add a few more attributes
        self.variable = 's'
        
    # Style to use for printing (similar to MATLAB)
    def __str__(self):
        labstr = ""
        outstr = ""
        for i in range(self.inputs):
            for j in range(self.outputs):
                # Create a label for the transfer function and extract
                # numerator polynomial (depends on number of inputs/outputs)
                if (self.inputs > 1 and self.outputs > 1):
                    labstr = "H[] = "; lablen = 7;
                    numstr = _tfpolyToString(self.num[i,j], self.variable);
                elif (self.inputs > 1):
                    labstr = "H[] = "; lablen = 7;
                    numstr = _tfpolyToString(self.num[i], self.variable);
                elif (self.outputs > 1):
                    labstr = "H[] = "; lablen = 7;
                    numstr = _tfpolyToString(self.num[j], self.variable);
                else:
                    labstr = ""; lablen = 0;
                    numstr = _tfpolyToString(self.num, self.variable);

                # Convert the (common) denominator polynomials to strings
                denstr = _tfpolyToString(self.den, self.variable);

                # Figure out the length of the separating line
                dashcount = max(len(numstr), len(denstr))
                dashes = labstr + '-' * dashcount

                # Center the numerator or denominator
                if (len(numstr) < dashcount):
                    numstr = ' ' * \
                        int(round((dashcount - len(numstr))/2) + lablen) + \
                        numstr
                if (len(denstr) < dashcount): 
                    denstr = ' ' * \
                        int(round((dashcount - len(denstr))/2) + lablen) + \
                        denstr

                outstr += "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"
        return outstr

    # Negation of a transfer function
    def __neg__(self):
        return TransferFunction(-self.num, self.den)

    # Subtraction (use addition)
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)

    # Addition of two transfer functions (parallel interconnection)
    def __add__(self, sys):
        # Convert the second argument to a transfer function
        other = convertToTransferFunction(sys)

        # Compute the numerator and denominator of the sum
        den = sp.polymul(self.den, other.den)
        num = sp.polyadd(sp.polymul(self.num, other.den), \
                         sp.polymul(other.num, self.den))

        return TransferFunction(num, den)

    # Reverse addition - just switch the order
    def __radd__(self, other): return self + other;

    # Difference of two transfer functions
    def __sub__(self, other): return __add__(self, -other)
    def __rsub__(self, other): return __add__(other, -self)

    # Multiplication of two transfer functions (series interconnection)
    def __mul__(self, sys):
        # Make sure we have a transfer function (or convert to one)
        other = convertToTransferFunction(sys)

        # Compute the product of the transfer functions
        num = sp.polymul(self.num, other.num)
        den = sp.polymul(self.den, other.den)
        return TransferFunction(num, den)

    # Reverse multiplication - switch order (works for SISO)
    def __rmul__(self, other): return self * other

    # Division between transfer functions
    def __div__(self, sys):
        other = convertToTransferFunction(sys);
        return TransferFunction(sp.polymul(self.num, other.den),
                                sp.polymul(self.den, other.num));

    # Reverse division 
    def __rdiv__(self, sys):
        other = convertToTransferFunction(sys);
        return TransferFunction(sp.polymul(other.num, self.den),
                                sp.polymul(other.den, self.num));

    # Method for evaluating a transfer function at one frequency
    def evalfr(self, freq):
        return sp.polyval(self.num, freq*1j) / sp.polyval(self.den, freq*1j)

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        # Convert numerator and denomintator to 1D polynomials
        num = sp.poly1d(self.num)
        den = sp.poly1d(self.den)

        # Generate the frequency response at each frequency
        fresp = map(lambda w: num(w*1j) / den(w*1j), omega)

        mag = sp.sqrt(sp.multiply(fresp, sp.conjugate(fresp)));
        phase = sp.angle(fresp)

        return mag, phase, omega

    # Feedback around a trasnfer function
    def feedback(sys1, sys2, sign=-1): 
        # Get the numerator and denominator of the first system
        if (isinstance(sys1, (int, long, float, complex))):
            num1 = sys1; den1 = 1;
        elif (isinstance(sys1, TransferFunction)):
            num1 = sys1.num; den1 = sys1.den;
        else:
            raise TypeError

        # Get the numerator and denominator of the second system
        if (isinstance(sys2, (int, long, float, complex))):
            num2 = sys2; den2 = 1;
        elif (isinstance(sys2, TransferFunction)):
            num2 = sys2.num; den2 = sys2.den;
        else:
            raise TypeError

        # Compute sys1/(1 - sign*sys1*sys2)
        num = sp.polymul(num1, den2);
        den = sp.polysub(sp.polymul(den1, den2), sign * sp.polymul(num1, num2))

        # Return the result as a transfer function
        return TransferFunction(num, den)

# Function to create a transfer function from another type
def convertToTransferFunction(sys):
    if (isinstance(sys, TransferFunction)):
        # Already a transfer function; just return it
        return sys

    elif (isinstance(sys, statesp.StateSpace)):
        # State space system, convert using signal.lti
        return TransferFunction(sys.A, sys.B, sys.C, sys.D)

    elif (isinstance(sys, (int, long, float, complex))):
        # Convert a number into a transfer function
        return TransferFunction(sys, 1)

    else:
        raise TypeError("can't convert given type to TransferFunction")

# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tfpolyToString(coeffs, var='s'):
    thestr = "0"

    # Compute the number of coefficients
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
