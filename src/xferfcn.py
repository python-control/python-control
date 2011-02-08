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
# $Id: xferfcn.py 21 2010-06-06 17:29:42Z murrayrm $

# External function declarations
import scipy as sp
import scipy.signal as signal
import bdalg as bd
import statesp
from lti2 import Lti2

class xTransferFunction(Lti2):
    """The TransferFunction class is derived from the Lti2 parent class.  The
    main data members are 'num' and 'den', which are 2-D lists of arrays
    containing MIMO numerator and denominator coefficients.  For example,

    >>> num[2][5] = [1., 4., 8.]
    
    means that the numerator of the transfer function from the 6th input to the
    3rd output is set to s^2 + 4s + 8."""
    
    def __init__(self, num=1, den=1):
        """This is the constructor.  The default transfer function is 1 (unit
        gain direct feedthrough)."""

        # Make num and den into 3-d numpy arrays, if necessary.
        data = [num, den]
        for i in range(len(data)):
            if isinstance(data[i], (int, float, long, complex)):
                # Convert scalar to list of list of array.
                data[i] = [[[data[i]]]]
            elif isinstance(data[i], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0], (int, float, long, complex)):
                # Convert array to list of list of array.
                data[i] = [[data[i]]]
            elif isinstance(data[i], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0][0], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0][0][0], (int, float, long, complex)):
                # We already have the right format.
                pass
            else:
                # If the user passed in anything else, then it's unclear what
                # the meaning is.
                raise ValueError("The numerator and denominator inputs must be \
scalars or arrays (for\nSISO), or lists of lists of arrays (for MIMO).")
        [num, den] = data
        
        inputs = len(num[0])
        outputs = len(num)
        
        if inputs != len(den[0]):
            raise ValueError("The numerator and denominator matrices must have \
the same column\n(input) size.")
        if outputs != len(den):
            raise ValueError("The numerator and denominator matrices must have \
the same row\n(output) size.")
        for i in range(1, outputs):
            if len(num[i]) != inputs:
                raise ValueError("Each row of the numerator matrix must have \
the same number of\nelements.")
            if len(den[i]) != inputs:
                raise ValueError("Each row of the denominator matrix must have \
the same number of\nelements.")

        self.num = num
        self.den = den
        Lti2.__init__(self, inputs, outputs)
        
        self.truncatecoeff()
        
    def __str__(self):
        """String representation of the transfer function."""
    
        pass
    
    def truncatecoeff(self):
        """Remove extraneous zero coefficients from polynomials in numerator and
        denominator matrices."""

        data = [self.num, self.den]
        
        for p in range(len(data)):
            for i in range(self.outputs):
                for j in range(self.inputs):
                    # Find the first nontrivial coefficient.
                    k = 0
                    while not data[p][i][j][k]:
                        k += 1
                    
                    # Now truncate the trivial coefficients.
                    data[p][i][j] = data[p][i][j][k:]
                    
        [self.num, self.den] = data
    
    def __neg__(self):
        """Negate a transfer function."""
        
        return xTransferFunction(-self.num, self.den)
        
    def __add__(self, other):
        """Add two transfer functions (parallel connection)."""
        
        # Convert the second argument to a transfer function.
        if not isinstance(other, xTransferFunction):
            other = ss2tf(other)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.inputs:
            raise ValueError("The two systems to be added must have the same \
input size.")
        if self.outputs != other.outputs:
            raise ValueError("The two systems to be added must have the same \
output size.")

        # Preallocate the numerator and denominator of the sum.
        num = [[[] for j in range(self.inputs)] for i in range(self.outputs)]
        den = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j], den[i][j] = addSISO(self.num[i][j], self.den[i][j],
                    other.num[i][j], other.den[i][j])

        return xTransferFunction(num, den)
 
    def __radd__(self, other): 
        """Add two transfer functions (parallel connection)"""
        
        return self + other;
        
    def __sub__(self, other): 
        """Subtract two transfer functions."""
        
        return self + (-other)
        
    def __rsub__(self, other): 
        """Subtract two transfer functions."""
        
        return other + (-self)

    def __mul__(self, other):
        """Multiply two transfer functions (serial connection)"""
        
        # Convert the second argument to a transfer function.
        if not isinstance(other, xTransferFunction):
            other = ss2tf(other)
            
        # Check that the input-output sizes are consistent.
        if self.inputs != other.outputs:
            raise ValueError("C = A * B: A must have the same number of \
columns (inputs) as B has\nrows (outputs).")

        inputs = other.inputs
        outputs = self.outputs
        
        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[0] for j in range(inputs)] for i in range(outputs)]
        
        # Temporary storage for the summands needed to find the (i, j)th element
        # of the product.
        num_summand = [[] for k in range(self.inputs)]
        den_summand = [[] for k in range(self.inputs)]
        
        for i in range(outputs): # Iterate through rows of product.
            for j in range(inputs): # Iterate through columns of product.
                for k in range(self.inputs): # Multiply & add.
                    num_summand[k] = sp.polymul(self.num[i][k], other.num[k][j])
                    den_summand[k] = sp.polymul(self.den[i][k], other.den[k][j])
                    num[i][j], den[i][j] = addSISO(num[i][j], den[i][j],
                        num_summand[k], den_summand[k])
        
        return xTransferFunction(num, den)

    def __rmul__(self, other): 
        """Multiply two transfer functions (serial connection)"""
        
        return self * other

    def __div__(self, sys):
        """Divide two transfer functions"""
        
        pass
        
    def __rdiv__(self, sys):
        """Reverse divide two transfer functions"""
        
        pass
        
    def evalfr(self, freq):
        """Evaluate a transfer function at a single frequency"""

        # return sp.polyval(self.num, freq*1j) / sp.polyval(self.den, freq*1j)
        pass

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        """Evaluate a transfer function at a list of frequencies"""
        
        pass

    def poles(self):
        """Compute poles of a transfer function."""
        
        pass
        
    def zeros(self): 
        """Compute zeros of a transfer function."""
        
        pass

    def feedback(sys1, sys2, sign=-1): 
        """Feedback interconnection between two transfer functions"""
        
        pass

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
        """Negate a transfer function"""
        return TransferFunction(-self.num, self.den)

    # Subtraction (use addition)
    def __sub__(self, other): 
        """Subtract two transfer functions"""
        return self + (-other)
    def __rsub__(self, other): 
        """Subtract two transfer functions"""
        return other + (-self)

    # Addition of two transfer functions (parallel interconnection)
    def __add__(self, sys):
        """Add two transfer functions (parallel connection)"""
        # Convert the second argument to a transfer function
        other = convertToTransferFunction(sys)

        # Compute the numerator and denominator of the sum
        den = sp.polymul(self.den, other.den)
        num = sp.polyadd(sp.polymul(self.num, other.den), \
                         sp.polymul(other.num, self.den))

        return TransferFunction(num, den)

    # Reverse addition - just switch the order
    def __radd__(self, other): 
        """Add two transfer functions (parallel connection)"""
        return self + other;

    # Multiplication of two transfer functions (series interconnection)
    def __mul__(self, sys):
        """Multiply two transfer functions (serial connection)"""
        # Make sure we have a transfer function (or convert to one)
        other = convertToTransferFunction(sys)

        # Compute the product of the transfer functions
        num = sp.polymul(self.num, other.num)
        den = sp.polymul(self.den, other.den)
        return TransferFunction(num, den)

    # Reverse multiplication - switch order (works for SISO)
    def __rmul__(self, other): 
        """Multiply two transfer functions (serial connection)"""
        return self * other

    # Division between transfer functions
    def __div__(self, sys):
        """Divide two transfer functions"""
        other = convertToTransferFunction(sys);
        return TransferFunction(sp.polymul(self.num, other.den),
                                sp.polymul(self.den, other.num));

    # Reverse division 
    def __rdiv__(self, sys):
        """Divide two transfer functions"""
        other = convertToTransferFunction(sys);
        return TransferFunction(sp.polymul(other.num, self.den),
                                sp.polymul(other.den, self.num));

    # Method for evaluating a transfer function at one frequency
    def evalfr(self, freq):
        """Evaluate a transfer function at a single frequency"""
        return sp.polyval(self.num, freq*1j) / sp.polyval(self.den, freq*1j)

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        """Evaluate a transfer function at a list of frequencies"""
        # Convert numerator and denomintator to 1D polynomials
        num = sp.poly1d(self.num)
        den = sp.poly1d(self.den)

        # Generate the frequency response at each frequency
        fresp = map(lambda w: num(w*1j) / den(w*1j), omega)

        mag = sp.sqrt(sp.multiply(fresp, sp.conjugate(fresp)));
        phase = sp.angle(fresp)

        return mag, phase, omega

    # Compute poles and zeros
    def poles(self): return sp.roots(self.den)
    def zeros(self): return sp.roots(self.num)

    # Feedback around a transfer function
    def feedback(sys1, sys2, sign=-1): 
        """Feedback interconnection between two transfer functions"""
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

# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tfpolyToString(coeffs, var='s'):
    """Convert a transfer function polynomial to a string"""
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
    
def addSISO(num1, den1, num2, den2):
    """Return num/den = num1/den1 + num2/den2, where each numerator and
    denominator is a list of polynomial coefficients."""
    
    num = sp.polyadd(sp.polymul(num1, den2), sp.polymul(num2, den1))
    den = sp.polymul(den1, den2)
    
    return num, den
    
def ss2tf(sys):
    """Convert a state space object to a transfer function object."""

    pass