from __future__ import division
"""frdata.py

Frequency response data representation and functions.

This file contains the FRD class and also functions that operate on
FRD data.
     
Routines in this module:

FRD.__init__
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
FRD.__truediv__
FRD.__rtruediv__
FRD.evalfr
FRD.freqresp
FRD.pole
FRD.zero
FRD.feedback
FRD._common_den
_convertToFRD

"""

"""Copyright (c) 2010 by California Institute of Technology
   Copyright (c) 2012 by Delft University of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of the California Institute of Technology nor
   the Delft University of Technology nor
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

Author: M.M. (Rene) van Paassen (using xferfcn.py as basis)
Date: 02 Oct 12
Revised: 

$Id: frd.py 185 2012-08-30 05:44:32Z murrayrm $

"""

# External function declarations
from numpy import angle, array, empty, ones, \
    real, imag, matrix, absolute, eye, linalg, where, dot
from scipy.interpolate import splprep, splev
from control.lti import Lti

class FRD(Lti):
    """The FRD class represents (measured?) frequency response 
    TF instances and functions.
    
    The FRD class is derived from the Lti parent class.  It is used
    throughout the python-control library to represent systems in frequency
    response data form. 
    
    The main data members are 'omega' and 'fresp'. omega is a 1D 
    array with the frequency points of the response. fresp is a 3D array,
    with the first dimension corresponding to the outputs of the FRD, 
    the second dimension corresponding to the inputs, and the 3rd dimension 
    corresponding to the frequency points in omega.
    For example,

    >>> frdata[2,5,:] = numpy.array([1., 0.8-0.2j, 0.2-0.8j])
    
    means that the frequency response from the 6th input to the 3rd
    output at the frequencies defined in omega is set to the array
    above, i.e. the rows represent the outputs and the columns
    represent the inputs.
    
    """
    
    epsw = 1e-8

    def __init__(self, *args, **kwargs):
        """Construct a transfer function.
        
        The default constructor is FRD(d, w), where w is an iterable of 
        frequency points, and d is the matching frequency data. 
        
        If d is a single list, 1d array, or tuple, a SISO system description 
        is assumed. d can also be 

        To call the copy constructor, call FRD(sys), where sys is a
        FRD object.

        To construct frequency response data for an existing Lti
        object, other than an FRD, call FRD(sys, omega)

        

        """
        smooth = kwargs.get('smooth', False)

        if len(args) == 2:
            if not isinstance(args[0], FRD) and isinstance(args[0], Lti):
                # not an FRD, but still a system, second argument should be
                # the frequency range
                otherlti = args[0]
                self.omega = array(args[1], dtype=float)
                self.omega.sort()
                numfreq = len(self.omega)

                # calculate frequency response at my points
                self.fresp = empty(
                    (otherlti.outputs, otherlti.inputs, numfreq), 
                    dtype=complex)
                for k, w in enumerate(self.omega):
                    self.fresp[:, :, k] = otherlti.evalfr(w)

            else:
                # The user provided a response and a freq vector
                self.fresp = array(args[0], dtype=complex)
                if len(self.fresp.shape) == 1:
                    self.fresp = self.fresp.reshape(1, 1, len(args[0]))
                self.omega = array(args[1], dtype=float)
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

        # create interpolation functions
        if smooth:
            self.ifunc = empty((self.fresp.shape[0], self.fresp.shape[1]), 
                               dtype=tuple)
            for i in range(self.fresp.shape[0]):
                for j in range(self.fresp.shape[1]):
                    self.ifunc[i,j],u = splprep(
                        u=self.omega, x=[real(self.fresp[i, j, :]), 
                                         imag(self.fresp[i, j, :])], 
                        w=1.0/(absolute(self.fresp[i, j, :])+0.001), s=0.0)
        else:
            self.ifunc = None
        Lti.__init__(self, self.fresp.shape[1], self.fresp.shape[0])
        
    def __str__(self):
        """String representation of the transfer function."""
        
        mimo = self.inputs > 1 or self.outputs > 1  
        outstr = [ 'frequency response data ' ]
        
        mt, pt, wt = self.freqresp(self.omega)
        for i in range(self.inputs):
            for j in range(self.outputs):
                if mimo:
                    outstr.append("Input %i to output %i:" % (i + 1, j + 1))
                outstr.append('Freq [rad/s]  Response   ')
                outstr.append('------------  ---------------------')
                outstr.extend(
                    [ '%12.3f  %10.4g%+10.4gj' % (w, m, p)
                      for m, p, w in zip(real(self.fresp[j,i,:]), imag(self.fresp[j,i,:]), wt) ])


        return '\n'.join(outstr)
    
    def __neg__(self):
        """Negate a transfer function."""
        
        return FRD(-self.fresp, self.omega)
    
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

        return FRD(self.fresp + other.fresp, other.omega)
 
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
        if isinstance(other, (int, float, complex)):
            return FRD(self.fresp * other, self.omega)
        else:
            other = _convertToFRD(other, omega=self.omega)
            
        # Check that the input-output sizes are consistent.
        if self.inputs != other.outputs:
            raise ValueError("H = G1*G2: input-output size mismatch"
                " G1 has %i input(s), G2 has %i output(s)." %
                (self.inputs, other.outputs))

        inputs = other.inputs
        outputs = self.outputs
        fresp = empty((outputs, inputs, len(self.omega)), 
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):        
            fresp[:,:,i] = dot(self.fresp[:,:,i], other.fresp[:,:,i])
        return FRD(fresp, self.omega)

    def __rmul__(self, other): 
        """Right Multiply two LTI objects (serial connection)."""
        
        # Convert the second argument to an frd function.
        if isinstance(other, (int, float, complex)):
            return FRD(self.fresp * other, self.omega)
        else:
            other = _convertToFRD(other, omega=self.omega)
            
        # Check that the input-output sizes are consistent.
        if self.outputs != other.inputs:
            raise ValueError("H = G1*G2: input-output size mismatch"
                " G1 has %i input(s), G2 has %i output(s)." %
                (other.inputs, self.outputs))

        inputs = self.inputs
        outputs = other.outputs
        
        fresp = empty((outputs, inputs, len(self.omega)), 
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):        
            fresp[:,:,i] = dot(other.fresp[:,:,i], self.fresp[:,:,i])
        return FRD(fresp, self.omega)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""
        
        if isinstance(other, (int, float, complex)):
            return FRD(self.fresp * (1/other), self.omega)
        else:
            other = _convertToFRD(other, omega=self.omega)


        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError(
                "FRD.__truediv__ is currently implemented only for SISO systems.")
        
        return FRD(self.fresp/other.fresp, self.omega)

    # TODO: Remove when transition to python3 complete
    def __div__(self, other):
        return self.__truediv__(other)
       
    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rtruediv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, complex)):
            return FRD(other / self.fresp, self.omega)
        else:
            other = _convertToFRD(other, omega=self.omega)
        
        if (self.inputs > 1 or self.outputs > 1 or 
            other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError(
                "FRD.__rtruediv__ is currently implemented only for SISO systems.")

        return other / self

    # TODO: Remove when transition to python3 complete
    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __pow__(self,other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return FRD(ones(self.fresp.shape),self.omega) #unity
        if other > 0:
            return self * (self**(other-1))
        if other < 0:
            return (FRD(ones(self.fresp.shape), self.omega) / self) * \
                (self**(other+1))
            
        
    def evalfr(self, omega):
        """Evaluate a transfer function at a single angular frequency.
        
        self.evalfr(omega) returns the value of the frequency response
        at frequency omega.

        Note that a "normal" FRD only returns values for which there is an
        entry in the omega vector. An interpolating FRD can return 
        intermediate values.

        """

        # Preallocate the output.
        if getattr(omega, '__iter__', False):
            out = empty((self.outputs, self.inputs, len(omega)), dtype=complex)
        else:
            out = empty((self.outputs, self.inputs), dtype=complex)

        if self.ifunc is None:
            try:
                out = self.fresp[:, :, where(self.omega == omega)[0][0]]
            except:
                raise ValueError(
                    "Frequency %f not in frequency list, try an interpolating"
                    " FRD if you want additional points")
        else:
            if getattr(omega, '__iter__', False):
                for i in range(self.outputs):
                    for j in range(self.inputs):
                        for k,w in enumerate(omega): 
                            frraw = splev(w, self.ifunc[i,j], der=0)
                            out[i,j,k] = frraw[0] + 1.0j*frraw[1]
            else:
                for i in range(self.outputs):
                    for j in range(self.inputs):
                        frraw = splev(omega, self.ifunc[i,j], der=0)
                        out[i,j] = frraw[0] + 1.0j*frraw[1]
                
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

        for k, w in enumerate(omega):
            fresp = self.evalfr(w)
            mag[:, :, k] = abs(fresp)
            phase[:, :, k] = angle(fresp)

        return mag, phase, omega

    def feedback(self, other=1, sign=-1): 
        """Feedback interconnection between two FRD objects."""
        
        other = _convertToFRD(other, omega=self.omega)

        if (self.outputs != other.inputs or 
            self.inputs != other.outputs):
            raise ValueError(
                "FRD.feedback, inputs/outputs mismatch")
        fresp = empty((self.outputs, self.inputs, len(other.omega)), 
                      dtype=complex)
        # TODO: vectorize this
        # TODO: handle omega re-mapping
        for k, w in enumerate(other.omega):
            fresp[:, :, k] = self.fresp[:, :, k].view(type=matrix)* \
                linalg.solve(
                eye(self.inputs) +
                other.fresp[:, :, k].view(type=matrix) * 
                self.fresp[:, :, k].view(type=matrix),
                eye(self.inputs))
            
        return FRD(fresp, other.omega)
 
def _convertToFRD(sys, omega, inputs=1, outputs=1):
    """Convert a system to frequency response data form (if needed).
    
    If sys is already an frd, and its frequency range matches or
    overlaps the range given in omega then it is returned.  If sys is
    another Lti object or a transfer function, then it is converted to
    a frequency response data at the specified omega. If sys is a
    scalar, then the number of inputs and outputs can be specified
    manually, as in:

    >>> sys = _convertToFRD(3.) # Assumes inputs = outputs = 1
    >>> sys = _convertToFRD(1., inputs=3, outputs=2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].
    
    """
    
    if isinstance(sys, FRD):
        omega.sort()
        if (abs(omega - sys.omega) < FRD.epsw).all():
            # frequencies match, and system was already frd; simply use
            return sys
        
        raise NotImplementedError(
            "Frequency ranges of FRD do not match, conversion not implemented")
                         
    elif isinstance(sys, Lti):
        omega.sort()
        fresp = empty((sys.outputs, sys.inputs, len(omega)), dtype=complex)
        for k, w in enumerate(omega):
            fresp[:, :, k] = sys.evalfr(w)
        
        return FRD(fresp, omega)

    elif isinstance(sys, (int, float, complex)):
        fresp = ones((outputs, inputs, len(omega)), dtype=float)*sys
        return FRD(fresp, omega)

    # try converting constant matrices
    try:
        sys = array(sys)
        outputs,inputs = sys.shape
        fresp = empty((outputs, inputs, len(omega)), dtype=float)
        for i in range(outputs):
            for j in range(inputs):
                fresp[i,j,:] = sys[i,j]
        return FRD(fresp, omega)
    except:
        pass

    raise TypeError('''Can't convert given type "%s" to FRD system.''' %
                    sys.__class__)
