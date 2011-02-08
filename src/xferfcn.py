"""xferfcn.py

Transfer function representation and functions.

This file contains the TransferFunction class and also functions
that operate on transfer functions.  This is the primary representation
for the python-control library.
     
Routines in this module:

TransferFunction.__init__
TransferFunction._truncatecoeff
TransferFunction.__str__
TransferFunction.__neg__
TransferFunction.__add__
TransferFunction.__radd__
TransferFunction.__sub__
TransferFunction.__rsub__
TransferFunction.__mul__
TransferFunction.__rmul__
TransferFunction.__div__
TransferFunction.__rdiv__
TransferFunction.evalfr
TransferFunction.freqresp
TransferFunction.pole
TransferFunction.zero
TransferFunction.feedback
TransferFunction.returnScipySignalLti
TransferFunction._common_den
_tfpolyToString
_addSISO
convertToTransferFunction

Copyright (c) 2010 by California Institute of Technology
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
Date: 24 May 09
Revised: Kevin K. Chewn, Dec 10

$Id: xferfcn.py 21 2010-06-06 17:29:42Z murrayrm $

"""

# External function declarations
from numpy import angle, array, empty, ndarray, ones, polyadd, polymul, \
    polyval, roots, zeros
from scipy.signal import lti
from copy import deepcopy
from slycot import tb04ad
from lti import Lti
import statesp

class TransferFunction(Lti):

    """The TransferFunction class represents TF instances and functions.
    
    The TransferFunction class is derived from the Lti parent class.  It is used
    throught the python-control library to represent systems in transfer
    function form. 
    
    The main data members are 'num' and 'den', which are 2-D lists of arrays
    containing MIMO numerator and denominator coefficients.  For example,

    >>> num[2][5] = numpy.array([1., 4., 8.])
    
    means that the numerator of the transfer function from the 6th input to the
    3rd output is set to s^2 + 4s + 8.
    
    """
    
    def __init__(self, num=1, den=1):
        """Construct a transfer function.  The default is unit static gain."""

        # Make num and den into lists of lists of arrays, if necessary.  Beware:
        # this is a shallow copy!  This should be okay, but be careful.
        data = [num, den]
        for i in range(len(data)):
            if isinstance(data[i], (int, float, long, complex)):
                # Convert scalar to list of list of array.
                data[i] = [[array([data[i]])]]
            elif (isinstance(data[i], (list, tuple, ndarray)) and 
                isinstance(data[i][0], (int, float, long, complex))):
                # Convert array to list of list of array.
                data[i] = [[array(data[i])]]
            elif (isinstance(data[i], list) and 
                isinstance(data[i][0], list) and 
                isinstance(data[i][0][0], (list, tuple, ndarray)) and 
                isinstance(data[i][0][0][0], (int, float, long, complex))):
                # We might already have the right format.  Convert the
                # coefficient vectors to arrays, if necessary.
                for j in range(len(data[i])):
                    for k in range(len(data[i][j])):
                        data[i][j][k] = array(data[i][j][k])
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
                    den[i][j] = ones(1)

        self.num = num
        self.den = den
        Lti.__init__(self, inputs, outputs)
        
        self._truncatecoeff()
        
    def _truncatecoeff(self):
        """Remove extraneous zero coefficients from num and den.

        Check every element of the numerator and denominator matrices, and
        truncate leading zeros.  For instance, running self._truncatecoeff()
        will reduce self.num = [[[0, 0, 1, 2]]] to [[[1, 2]]].
        
        """

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
                        data[p][i][j] = zeros(1)
                    else:
                        # Truncate the trivial coefficients.
                        data[p][i][j] = data[p][i][j][nonzero:]        
        [self.num, self.den] = data
    
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
                if len(numstr) < dashcount:
                    numstr = (' ' * int(round((dashcount - len(numstr))/2)) + 
                        numstr)
                if len(denstr) < dashcount: 
                    denstr = (' ' * int(round((dashcount - len(denstr))/2)) + 
                        denstr)

                outstr += "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"
        return outstr
    
    def __neg__(self):
        """Negate a transfer function."""
        
        num = deepcopy(self.num)
        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j] *= -1
        
        return TransferFunction(num, self.den)
        
    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""
        
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
        """Reverse add two LTI objects (parallel connection)."""
        
        return self + other;
        
    def __sub__(self, other): 
        """Subtract two LTI objects."""
        
        return self + (-other)
        
    def __rsub__(self, other): 
        """Reverse subtract two LTI objects."""
        
        return other + (-self)

    def __mul__(self, other):
        """Multiply two LTI objects (serial connection)."""
        
        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, long, complex)):
            other = convertToTransferFunction(other, self.inputs, self.inputs)
        else:
            other = convertToTransferFunction(other)
            
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
                    num_summand[k] = polymul(self.num[i][k], other.num[k][j])
                    den_summand[k] = polymul(self.den[i][k], other.den[k][j])
                    num[i][j], den[i][j] = _addSISO(num[i][j], den[i][j],
                        num_summand[k], den_summand[k])
        
        return TransferFunction(num, den)

    def __rmul__(self, other): 
        """Reverse multiply two LTI objects (serial connection)."""
        
        return self * other

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __div__(self, other):
        """Divide two LTI objects."""
        
        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError("TransferFunction.__div__ is currently \
implemented only for SISO systems.")

        # Convert the second argument to a transfer function.
        other = convertToTransferFunction(other)

        num = polymul(self.num[0][0], other.den[0][0])
        den = polymul(self.den[0][0], other.num[0][0])
        
        return TransferFunction(num, den)
       
    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rdiv__(self, other):
        """Reverse divide two LTI objects."""
        
        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError("TransferFunction.__rdiv__ is currently \
implemented only for SISO systems.")

        return other / self
        
    def evalfr(self, omega):
        """Evaluate a transfer function at a single angular frequency.
        
        self.evalfr(omega) returns the value of the transfer function matrix with
        input value s = i * omega.

        """

        # Preallocate the output.
        out = empty((self.outputs, self.inputs), dtype=complex)

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = (polyval(self.num[i][j], omega * 1.j) / 
                    polyval(self.den[i][j], omega * 1.j))

        return out

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        """Evaluate a transfer function at a list of angular frequencies.

        mag, phase, omega = self.freqresp(omega)

        reports the value of the magnitude, phase, and angular frequency of the 
        transfer function matrix evaluated at s = i * omega, where omega is a
        list of angular frequencies.

        """
        
        # Preallocate outputs.
        numfreq = len(omega)
        mag = empty((self.outputs, self.inputs, numfreq))
        phase = empty((self.outputs, self.inputs, numfreq))

        for i in range(self.outputs):
            for j in range(self.inputs):
                fresp = map(lambda w: (polyval(self.num[i][j], w * 1.j) / 
                    polyval(self.den[i][j], w * 1.j)), omega)
                fresp = array(fresp)

                mag[i, j, :] = abs(fresp)
                phase[i, j, :] = angle(fresp)

        return mag, phase, omega

    def pole(self):
        """Compute the poles of a transfer function."""
        
        num, den = self._common_den()
        return roots(den) 

    def zero(self): 
        """Compute the zeros of a transfer function."""
        
        raise NotImplementedError("TransferFunction.zero is not implemented \
yet.")

    def feedback(self, other, sign=-1): 
        """Feedback interconnection between two LTI objects."""
        
        other = convertToTransferFunction(other)

        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            # TODO: MIMO feedback
            raise NotImplementedError("TransferFunction.feedback is currently \
only implemented for SISO functions.")

        num1 = self.num[0][0]
        den1 = self.den[0][0]
        num2 = other.num[0][0]
        den2 = other.den[0][0]

        num = polymul(num1, den2)
        den = polyadd(polymul(den2, den1), -sign * polymul(num2, num1))

        return TransferFunction(num, den)

        # For MIMO or SISO systems, the analytic expression is
        #     self / (1 - sign * other * self)
        # But this does not work correctly because the state size will be too
        # large.

    def returnScipySignalLti(self):
        """Return a list of a list of scipy.signal.lti objects.
        
        For instance,
        
        >>> out = tfobject.returnScipySignalLti()
        >>> out[3][5]
            
        is a signal.scipy.lti object corresponding to the transfer function from
        the 6th input to the 4th output.
        
        """

        # Preallocate the output.
        out = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = lti(self.num[i][j], self.den[i][j])
            
        return out 

    def _common_den(self):
        """Compute MIMO common denominator; return it and an adjusted numerator.
        
        >>> n, d = sys._common_den()
        
        computes the single denominator containing all the poles of sys.den, and
        reports it as the array d.

        The output numerator array n is modified to use the common denominator.
        It is an sys.outputs-by-sys.inputs-by-[something] array.

        """
         
        # Preallocate some variables.  Start by figuring out the maximum number
        # of numerator coefficients.
        numcoeffs = 0
        for i in range(self.outputs):
            for j in range(self.inputs):
                numcoeffs = max(numcoeffs, len(self.num[i][j]))
        # The output 3-D adjusted numerator array.
        num = empty((i, j, numcoeffs))
        # A list to keep track of roots found as we scan self.den.
        poles = []
        # A 3-D list to keep track of common denominator roots not present in
        # the self.den[i][j].
        missingpoles = [[[] for j in range(self.inputs)] for i in
            range(self.outputs)]

        for i in range(sys.outputs):
            for j in range(sys.inputs):
                currentpoles = roots(self.den[i][j])
                #TODO: finish this

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
    """Return num/den = num1/den1 + num2/den2.
    
    Each numerator and denominator is a list of polynomial coefficients.
    
    """
    
    num = polyadd(polymul(num1, den2), polymul(num2, den1))
    den = polymul(den1, den2)
    
    return num, den

def convertToTransferFunction(sys, inputs=1, outputs=1):
    """Convert a system to transfer function form (if needed).
    
    If sys is already a transfer function, then it is returned.  If sys is a
    state space object, then it is converted to a transfer function and
    returned.  If sys is a scalar, then the number of inputs and outputs can be
    specified manually.
    
    """
    
    if isinstance(sys, TransferFunction):
        return sys
    elif isinstance(sys, statesp.StateSpace):
        # Use Slycot to make the transformation.
        tfout = tb04ad(sys.states, sys.inputs, sys.outputs, sys.A, sys.B, sys.C,
            sys.D, sys.outputs, sys.outputs, sys.inputs)

        # Preallocate outputs.
        num = [[[] for j in range(sys.inputs)] for i in range(sys.outputs)]
        den = [[[] for j in range(sys.inputs)] for i in range(sys.outputs)]

        for i in range(sys.outputs):
            for j in range(sys.inputs):
                num[i][j] = list(tfout[6][i, j, :])
                # Each transfer function matrix row has a common denominator.
                den[i][j] = list(tfout[5][i, :])

        return TransferFunction(num, den)
    elif isinstance(sys, (int, long, float, complex)):
        num = [[[sys] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        
        return TransferFunction(num, den)
    else:
        raise TypeError("Can't convert given type to TransferFunction system.")
