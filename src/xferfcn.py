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
from lti2 import Lti2

class TransferFunction(Lti2):
    """The TransferFunction class is derived from the Lti2 parent class.  The
    main data members are 'num' and 'den', which are 2-D lists of arrays
    containing MIMO numerator and denominator coefficients.  For example,

    >>> num[2][5] = numpy.array([1., 4., 8.])
    
    means that the numerator of the transfer function from the 6th input to the
    3rd output is set to s^2 + 4s + 8."""
    
    def __init__(self, num=1, den=1):
        """This is the constructor.  The default transfer function is 1 (unit
        gain direct feedthrough)."""
 
        # Make num and den into lists of lists of arrays, if necessary.  Beware:
        # this is a shallow copy!  This should be okay, but be careful.
        data = [num, den]
        for i in range(len(data)):
            if isinstance(data[i], (int, float, long, complex)):
                # Convert scalar to list of list of array.
                data[i] = [[sp.array([data[i]])]]
            elif isinstance(data[i], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0], (int, float, long, complex)):
                # Convert array to list of list of array.
                data[i] = [[sp.array(data[i])]]
            elif isinstance(data[i], list) and \
                isinstance(data[i][0], list) and \
                isinstance(data[i][0][0], (list, tuple, sp.ndarray)) and \
                isinstance(data[i][0][0][0], (int, float, long, complex)):
                # We might already have the right format.  Convert the
                # coefficient vectors to arrays, if necessary.
                for j in range(len(data[i])):
                    for k in range(len(data[i][j])):
                        data[i][j][k] = sp.array(data[i][j][k])
            else:
                # If the user passed in anything else, then it's unclear what
                # the meaning is.
                raise TypeError("The numerator and denominator inputs must be \
scalars or vectors (for\nSISO), or lists of lists of vectors (for SISO or \
MIMO).")
        [num, den] = data
        
        inputs = len(num[0])
        outputs = len(num)
        
        # Make sure the numerator and denominator matrices have consistent
        # sizes.
        if inputs != len(den[0]):
            raise ValueError("The numerator has %i input(s), but the \
denominator has %i\ninput(s)." % (inputs, len(den[0])))
        if outputs != len(den):
            raise ValueError("The numerator has %i output(s), but the \
denominator has %i\noutput(s)." % (outputs, len(den)))
        
        for i in range(outputs):
            # Make sure that each row has the same number of columns.
            if len(num[i]) != inputs:
                raise ValueError("Row 0 of the numerator matrix has %i \
elements, but row %i\nhas %i." % (inputs, i, len(num[i])))
            if len(den[i]) != inputs:
                raise ValueError("Row 0 of the denominator matrix has %i \
elements, but row %i\nhas %i." % (inputs, i, len(den[i])))
            
            # TODO: Right now these checks are only done during construction.
            # It might be worthwhile to think of a way to perform checks if the
            # user modifies the transfer function after construction.
            for j in range(inputs):
                # Check that we don't have any zero denominators.
                zeroden = True
                for k in den[i][j]:
                    if k:
                        zeroden = False
                        break
                if zeroden:
                    raise ValueError("Input %i, output %i has a zero \
denominator." % (j + 1, i + 1))

                # If we have zero numerators, set the denominator to 1.
                zeronum = True
                for k in num[i][j]:
                    if k:
                        zeronum = False
                        break
                if zeronum:
                    den[i][j] = sp.ones(1)

        self.num = num
        self.den = den
        Lti2.__init__(self, inputs, outputs)
        
        self._truncatecoeff()
        
    def __str__(self):
        """String representation of the transfer function."""
        
        mimo = self.inputs > 1 or self.outputs > 1  
        outstr = ""
        
        for i in range(self.inputs):
            for j in range(self.outputs):
                if mimo:
                    outstr += "\nInput %i to output %i:" % (i + 1, j + 1)
                    
                # Convert the numerator and denominator polynomials to strings.
                numstr = _tfpolyToString(self.num[j][i]);
                denstr = _tfpolyToString(self.den[j][i]);

                # Figure out the length of the separating line
                dashcount = max(len(numstr), len(denstr))
                dashes = '-' * dashcount

                # Center the numerator or denominator
                if (len(numstr) < dashcount):
                    numstr = ' ' * \
                        int(round((dashcount - len(numstr))/2)) + \
                        numstr
                if (len(denstr) < dashcount): 
                    denstr = ' ' * \
                        int(round((dashcount - len(denstr))/2)) + \
                        denstr

                outstr += "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"
        return outstr
    
    def _truncatecoeff(self):
        """Remove extraneous zero coefficients from polynomials in numerator and
        denominator matrices."""

        # Beware: this is a shallow copy.  This should be okay.
        data = [self.num, self.den]
        for p in range(len(data)):
            for i in range(self.outputs):
                for j in range(self.inputs):
                    # Find the first nontrivial coefficient.
                    nonzero = None
                    for k in range(data[p][i][j].size):
                        if data[p][i][j][k]:
                            nonzero = k
                            break
                            
                    if nonzero is None:
                        # The array is all zeros.
                        data[p][i][j] = sp.zeros(1)
                    else:
                        # Truncate the trivial coefficients.
                        data[p][i][j] = data[p][i][j][nonzero:]        
        [self.num, self.den] = data
    
    def __neg__(self):
        """Negate a transfer function."""
        
        import copy

        num = copy.deepcopy(self.num)
        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j] *= -1
        
        return TransferFunction(num, self.den)
        
    def __add__(self, other):
        """Add two transfer functions (parallel connection)."""
        
        # Convert the second argument to a transfer function.
        if not isinstance(other, TransferFunction):
            other = convertToTransferFunction(other, self.inputs, self.outputs)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.inputs:
            raise ValueError("The first summand has %i input(s), but the \
second has %i." % (self.inputs, other.inputs))
        if self.outputs != other.outputs:
            raise ValueError("The first summand has %i output(s), but the \
second has %i." % (self.outputs, other.outputs))

        # Preallocate the numerator and denominator of the sum.
        num = [[[] for j in range(self.inputs)] for i in range(self.outputs)]
        den = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j], den[i][j] = _addSISO(self.num[i][j], self.den[i][j],
                    other.num[i][j], other.den[i][j])

        return TransferFunction(num, den)
 
    def __radd__(self, other): 
        """Add two transfer functions (parallel connection)."""
        
        return self + other;
        
    def __sub__(self, other): 
        """Subtract two transfer functions."""
        
        return self + (-other)
        
    def __rsub__(self, other): 
        """Subtract two transfer functions."""
        
        return other + (-self)

    def __mul__(self, other):
        """Multiply two transfer functions (serial connection)."""
        
        # Convert the second argument to a transfer function.
        if not isinstance(other, TransferFunction):
            other = convertToTransferFunction(other, self.inputs, self.inputs)
            
        # Check that the input-output sizes are consistent.
        if self.inputs != other.outputs:
            raise ValueError("C = A * B: A has %i column(s) (input(s)), but B \
has %i row(s)\n(output(s))." % (self.inputs, other.outputs))

        inputs = other.inputs
        outputs = self.outputs
        
        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        
        # Temporary storage for the summands needed to find the (i, j)th element
        # of the product.
        num_summand = [[] for k in range(self.inputs)]
        den_summand = [[] for k in range(self.inputs)]
        
        for i in range(outputs): # Iterate through rows of product.
            for j in range(inputs): # Iterate through columns of product.
                for k in range(self.inputs): # Multiply & add.
                    num_summand[k] = sp.polymul(self.num[i][k], other.num[k][j])
                    den_summand[k] = sp.polymul(self.den[i][k], other.den[k][j])
                    num[i][j], den[i][j] = _addSISO(num[i][j], den[i][j],
                        num_summand[k], den_summand[k])
        
        return TransferFunction(num, den)

    def __rmul__(self, other): 
        """Multiply two transfer functions (serial connection)."""
        
        return self * other

    # TODO: Division of MIMO transfer function objects is quite difficult.
    def __div__(self, other):
        """Divide two transfer functions."""
        
        if self.inputs > 1 or self.outputs > 1 or \
            other.inputs > 1 or other.outputs > 1:
            raise NotImplementedError("TransferFunction.__div__ is currently \
implemented only for SISO systems.")

        # Convert the second argument to a transfer function.
        if not isinstance(other, TransferFunction):
            other = convertToTransferFunction(other, 1, 1)

        num = sp.polymul(self.num[0][0], other.den[0][0])
        den = sp.polymul(self.den[0][0], other.num[0][0])
        
        return TransferFunction(num, den)
       
    # TODO: Division of MIMO transfer function objects is quite difficult.
    def __rdiv__(self, other):
        """Reverse divide two transfer functions."""
        
        if self.inputs > 1 or self.outputs > 1 or \
            other.inputs > 1 or other.outputs > 1:
            raise NotImplementedError("TransferFunction.__rdiv__ is currently \
implemented only for SISO systems.")

        return other / self
        
    def evalfr(self, freq):
        """Evaluate a transfer function at a single frequency."""

        # Preallocate the output.
        out = sp.empty((self.outputs, self.inputs), dtype=complex)

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = sp.polyval(self.num[i][j], freq * 1.j) / \
                    sp.polyval(self.den[i][j], freq * 1.j)

        return out

    # Method for generating the frequency response of the system
    def freqresp(self, omega=None):
        """Evaluate a transfer function at a list of frequencies."""
        
        # Preallocate outputs.
        numfreq = len(omega)
        mag = sp.empty((self.outputs, self.inputs, numfreq))
        phase = sp.empty((self.outputs, self.inputs, numfreq))

        for i in range(self.outputs):
            for j in range(self.inputs):
                fresp = map(lambda w: sp.polyval(self.num[i][j], w * 1.j) / \
                    sp.polyval(self.den[i][j], w * 1.j), omega)
                fresp = sp.array(fresp)

                mag[i, j] = abs(fresp)
                phase[i, j] = sp.angle(fresp)

        return mag, phase, omega

    def poles(self):
        """Compute poles of a transfer function."""
        
        pass
        
    def zeros(self): 
        """Compute zeros of a transfer function."""
        
        pass

    def feedback(self, other, sign=-1): 
        """Feedback interconnection between two transfer functions."""
        
        other = convertToTransferFunction(other)

        if self.inputs > 1 or self.outputs > 1 or \
            other.inputs > 1 or other.outputs > 1:
            raise NotImplementedError("TransferFunction.feedback is currently \
only implemented for SISO functions.")

        num1 = self.num[0][0]
        den1 = self.den[0][0]
        num2 = other.num[0][0]
        den2 = other.den[0][0]

        num = sp.polymul(num1, den2)
        den = sp.polyadd(sp.polymul(den2, den1), -sign * sp.polymul(num2, num1))

        return TransferFunction(num, den)

        # For MIMO or SISO systems, the analytic expression is
        #     self / (1 - sign * other * self)
        # But this does not work correctly because the state size will be too
        # large.

    def returnScipySignalLti(self):
        """Return a list of a list of scipy.signal.lti objects for a MIMO
        system.  For instance,
        
        >>> out = ssobject.returnSignalScipyLti()
        >>> out[3][5]
            
        is a signal.scipy.lti object corresponding to the transfer function from
        the 6th input to the 4th output."""

        from scipy.signal import lti

        # Preallocate the output.
        out = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = lti(self.num[i][j], self.den[i][j])
            
        return out 

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
    
def _addSISO(num1, den1, num2, den2):
    """Return num/den = num1/den1 + num2/den2, where each numerator and
    denominator is a list of polynomial coefficients."""
    
    num = sp.polyadd(sp.polymul(num1, den2), sp.polymul(num2, den1))
    den = sp.polymul(den1, den2)
    
    return num, den

def convertToTransferFunction(sys, inputs=1, outputs=1):
    """Convert a system to transfer function form (if needed).  If sys is a
    scalar, then the number of inputs and outputs can be specified manually."""
    
    import statesp

    if isinstance(sys, TransferFunction):
        return sys
    elif isinstance(sys, statesp.StateSpace):
        raise NotImplementedError("State space to transfer function conversion \
is not implemented yet.")
    elif isinstance(sys, (int, long, float, complex)):
        # Make an identity system.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        for i in range(min(inputs, outputs)):
            num[i][i] = [sys]
        
        return TransferFunction(num, den)
    else:
        raise TypeError("Can't convert given type to TransferFunction system.")
