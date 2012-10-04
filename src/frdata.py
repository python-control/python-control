"""frd.py

Frequency response data representation and functions.

This file contains the FRD class and also functions
that operate on transfer functions.  This is the primary representation
for the python-control library.
     
Routines in this module:

FRD.__init__
FRD._truncatecoeff
FRD.copy
FRD.__str__
FRD.__neg__
FRD.__add__
FRD.__radd__
FRD.__sub__
FRD.__rsub__
FRD.__mul__
FRD.__rmul__
FRD.__div__
FRD.__rdiv__
FRD.evalfr
FRD.freqresp
FRD.pole
FRD.zero
FRD.feedback
FRD.returnScipySignalLti
FRD._common_den
_tfpolyToString
_addSISO
_convertToFRD

"""

"""Copyright (c) 2010 by California Institute of Technology
   and 2012 Delft University of Technology
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

Author: M.M. (Rene) van Paassen
Date: 02 Oct 12
Revised: 

$Id: frd.py 185 2012-08-30 05:44:32Z murrayrm $

"""

# External function declarations
from numpy import angle, any, array, empty, finfo, insert, ndarray, ones, \
    polyadd, polymul, polyval, roots, sort, sqrt, zeros, squeeze
from scipy.signal import lti
from copy import deepcopy
from lti import Lti
import statesp

class FRD(Lti):

    """The FRD class represents (measured?) frequency response 
    TF instances and functions.
    
    The FRD class is derived from the Lti parent class.  It is used
    throughout the python-control library to represent systems in frequency
    response data form. 
    
    The main data members are 'omega' and 'frdata'. omega is a single 
    array with the frequency points of the response. frdata is a list of arrays
    containing frequency points (in rad/s) and gain data as a complex number. 
    For example,

    >>> frdata[2][5] = numpy.array([1., 0.8-0.2j, 0.2-0.8j])
    
    means that the frequency response from the 6th input to the
    3rd output at the frequencies defined in omega is set the array above.
    
    """
    
    def __init__(self, *args):
        """Construct a transfer function.
        
        The default constructor is FRD(w, d), where w is an iterable of 
        frequency points, and d is the matching frequency data. 
        
        If d is a single list, 1d array, or tuple, a SISO system description 
        is assumed. d can also be 

        To call the copy constructor, call FRD(sys), where sys is a
        FRD object.

        """

        if len(args) == 2:
            if not isinstance(args[0], FRD) and isinstance(args[0], Lti):
                # not an FRD, but still a system, second argument should be
                # the frequency range
                otherlti = args[0]
                
                self.omega = array(args[1].sort(), dtype=float)

                # calculate frequency response at my points
                self.fresp = empty(
                    (otherlti.outputs, otherlti.inputs, numfreq), 
                    dtype=complex)
                for k, w in enumerate(omega):
                    self.fresp[:, :, k] = otherlti.evalfr(w)

            else:
                # The user provided a response and a freq vector
                self.fresp = array(args[0], dtype=complex)
                if len(self.fresp.shape) == 1:
                    self.fresp.reshape(1, 1, len(args[0]))
                self.omega = array(args[1])
                if len(self.fresp.shape) != 3 or \
                        self.fresp.shape[-1] != self.omega.shape[-1] or \
                        len(self.omega.shape) != 1:
                    raise TypeError(
                        "The frequency data constructor needs a 1-d or 3-d"
                        " response data array and a matching frequency vector"
                        " size")

        elif len(args) == 1:
            # Use the copy constructor.
            if not isinstance(args[0], FRD):
                raise TypeError(
                    "The one-argument constructor can only take in"
                    " an FRD object.  Received %s." % type(args[0]))
            self.omega = args[0].omega
            self.fresp = args[0].fresp
        else:
            raise ValueError("Needs 1 or 2 arguments; receivd %i." % len(args))

        Lti.__init__(self, self.fresp.shape[1], self.fresp.shape[0])
        
    def __str__(self):
        """String representation of the transfer function."""
        
        mimo = self.inputs > 1 or self.outputs > 1  
        outstr = [ 'frequency response data ' ]
        
        for i in range(self.inputs):
            for j in range(self.outputs):
                if mimo:
                    outstr.append("Input %i to output %i:" % (i + 1, j + 1))
                outstr.append('Freq [rad/s]  Magnitude   Phase')
                outstr.extend(
                    [ '%f %f %f' % (w, m, p*180.0)
                      for m, p, w in self.freqresp(self.omega) ])

        return '\n'.join(outstr)
    
    def __neg__(self):
        """Negate a transfer function."""
        
        return FRD(self.omega, -self.fresp)
    
    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""
        
        if isinstance(other, FRD):
            # verify that the frequencies match
            if (other.omega != self.omega).any():
                print("Warning: frequency points do not match; expect"
                      " truncation and interpolation")
                
        # Convert the second argument to a frequency response function.
        # or re-base the frd to the current omega (if needed)
        other = _convertToFRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.inputs:
            raise ValueError("The first summand has %i input(s), but the \
second has %i." % (self.inputs, other.inputs))
        if self.outputs != other.outputs:
            raise ValueError("The first summand has %i output(s), but the \
second has %i." % (self.outputs, other.outputs))

        return FRD(other.omega, self.frd + other.frd)
 
    def __radd__(self, other): 
        """Right add two LTI objects (parallel connection)."""
        
        return self + other;
        
    def __sub__(self, other): 
        """Subtract two LTI objects."""
        
        return self + (-other)
        
    def __rsub__(self, other): 
        """Right subtract two LTI objects."""
        
        return other + (-self)

    def __mul__(self, other):
        """Multiply two LTI objects (serial connection)."""
        
        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, long, complex)):
            other = _convertToFRD(other, inputs=self.inputs, 
                outputs=self.inputs)
        else:
            other = _convertToFRD(other)
            
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
        
        return FRD(num, den)

    def __rmul__(self, other): 
        """Right multiply two LTI objects (serial connection)."""
        
        return self * other

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __div__(self, other):
        """Divide two LTI objects."""
        
        if isinstance(other, (int, float, long, complex)):
            other = _convertToFRD(other, inputs=self.inputs, 
                outputs=self.inputs)
        else:
            other = _convertToFRD(other)


        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError("FRD.__div__ is currently \
implemented only for SISO systems.")

        num = polymul(self.num[0][0], other.den[0][0])
        den = polymul(self.den[0][0], other.num[0][0])
        
        return FRD(num, den)
       
    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rdiv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, long, complex)):
            other = _convertToFRD(other, inputs=self.inputs, 
                outputs=self.inputs)
        else:
            other = _convertToFRD(other)
        
        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError("FRD.__rdiv__ is currently \
implemented only for SISO systems.")

        return other / self
    def __pow__(self,other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return FRD([1],[1]) #unity
        if other > 0:
            return self * (self**(other-1))
        if other < 0:
            return (FRD([1],[1]) / self) * (self**(other+1))
            
        
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
        list of angular frequencies, and is a sorted version of the input omega.

        """
        
        # Preallocate outputs.
        numfreq = len(omega)
        mag = empty((self.outputs, self.inputs, numfreq))
        phase = empty((self.outputs, self.inputs, numfreq))

        omega.sort()

        for i in range(self.outputs):
            for j in range(self.inputs):
                fresp = map(lambda w: (polyval(self.num[i][j], w * 1.j) / 
                    polyval(self.den[i][j], w * 1.j)), omega)
                fresp = array(fresp)

                mag[i, j, :] = abs(fresp)
                phase[i, j, :] = angle(fresp)

        return mag, phase, omega

    def feedback(self, other, sign=-1): 
        """Feedback interconnection between two LTI objects."""
        
        other = _convertToFRD(other)

        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            # TODO: MIMO feedback
            raise NotImplementedError("FRD.feedback is currently \
only implemented for SISO functions.")

        num1 = self.num[0][0]
        den1 = self.den[0][0]
        num2 = other.num[0][0]
        den2 = other.den[0][0]

        num = polymul(num1, den2)
        den = polyadd(polymul(den2, den1), -sign * polymul(num2, num1))

        return FRD(num, den)

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

 
def _convertToFRD(sys, omega):
    """Convert a system to transfer function form (if needed).
    
    If sys is already a transfer function, then it is returned.  If sys is a
    state space object, then it is converted to a transfer function and
    returned.  If sys is a scalar, then the number of inputs and outputs can be
    specified manually, as in:

    >>> sys = _convertToFRD(3.) # Assumes inputs = outputs = 1
    >>> sys = _convertToFRD(1., inputs=3, outputs=2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].
    
    """
    
    if isinstance(sys, FRD):
        
        if (abs(omega - sys.omega) < eps).all():
            # frequencies match, and system was already frd; simply use
            return sys
        
        # omega becomes lowest common range
        omega = omega[omega >= min(sys.omega)]
        omega = omega[omega <= max(sys.omega)]
        if not omega:
            raise ValueError("Frequency ranges of FRD do not overlap")
        return FRDsys

    elif isinstance(sys, statesp.StateSpace):
        try:
            from slycot import tb04ad
            if len(kw):
                raise TypeError("If sys is a StateSpace, \
                        _convertToFRD cannot take keywords.")

            # Use Slycot to make the transformation
            # Make sure to convert system matrices to numpy arrays
            tfout = tb04ad(sys.states, sys.inputs, sys.outputs, array(sys.A),
                           array(sys.B), array(sys.C), array(sys.D), tol1=0.0)

            # Preallocate outputs.
            num = [[[] for j in range(sys.inputs)] for i in range(sys.outputs)]
            den = [[[] for j in range(sys.inputs)] for i in range(sys.outputs)]

            for i in range(sys.outputs):
                for j in range(sys.inputs):
                    num[i][j] = list(tfout[6][i, j, :])
                    # Each transfer function matrix row has a common denominator.
                    den[i][j] = list(tfout[5][i, :])
            # print num
            # print den
        except ImportError:
            # If slycot is not available, use signal.lti (SISO only)
            if (sys.inputs != 1 or sys.outputs != 1):
                raise TypeError("No support for MIMO without slycot")

            lti_sys = lti(sys.A, sys.B, sys.C, sys.D)
            num = squeeze(lti_sys.num)
            den = squeeze(lti_sys.den)
            print num
            print den

        return FRD(num, den)
    elif isinstance(sys, (int, long, float, complex)):
        if "inputs" in kw:
            inputs = kw["inputs"]
        else:
            inputs = 1
        if "outputs" in kw:
            outputs = kw["outputs"]
        else:
            outputs = 1

        num = [[[sys] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        
        return FRD(num, den)
    else:
        raise TypeError("Can't convert given type to FRD system.")
