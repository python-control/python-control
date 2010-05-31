# stateSpace.py - state space class for control systems library
#
# Author: Richard M. Murray
# Date: 24 May 09
# 
# This file contains the StateSpace class, which is used to represent
# linear systems in state space.  This is the primary representation
# for the control system library.
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
# $Id$

import scipy as sp
import scipy.signal as signal
import xferfcn
from scipy import concatenate, zeros

#
# StateSpace class
#
# The StateSpace class is used throughout the control systems library to
# represent systems in state space form.  This class is derived from
# the ltisys class defined in the scipy.signal package, allowing many
# of the functions that already existing in that package to be used
# directly.
#
class StateSpace(signal.lti):
    """The StateSpace class is used to represent linear input/output systems.
    """
    # Initialization 
    def __init__(self, *args, **keywords):
        # First initialize the parent object
        signal.lti.__init__(self, *args, **keywords)

    # Style to use for printing
    def __str__(self):
        str =  "A = " + self.A.__str__() + "\n\n"
        str += "B = " + self.B.__str__() + "\n\n"
        str += "C = " + self.C.__str__() + "\n\n"
        str += "D = " + self.D.__str__() + "\n"
        return str

    # Method for generating the frequency response of the system
    def freqresp(self, omega=None):
        """Compute the response of a system to a list of frequencies"""
        # Generate and save a transfer function matrix
        #! TODO: This is currently limited to SISO systems
        nout, nin = self.D.shape

        # Compute the denominator from the A matrix
        den = sp.poly1d(sp.poly(self.A))

        # Compute the numerator based on zeros
        #! TODO: This is currently limited to SISO systems
        num = sp.poly1d(\
            sp.poly(self.A - sp.dot(self.B, self.C)) + (self.D[0] - 1) * den)

        # Generate the frequency response at each frequency
        fresp = map(lambda w: num(w*1j) / den(w*1j), omega)
        mag = sp.sqrt(sp.multiply(fresp, sp.conjugate(fresp)))
        phase = sp.angle(fresp)

        return mag, phase, omega

    # Method for evaluating a system at one frequency
    def evalfr(self, freq):
        #! TODO: Not implemented
        return None

    # Negation of a system
    def __neg__(self):
        """Negate a state space system"""
        return StateSpace(self.A, self.B, -self.C, -self.D)

    # Addition of two transfer functions (parallel interconnection)
    def __add__(self, other):
        """Add two state space systems"""
        # Check for a couple of special cases
        if (isinstance(other, (int, long, float, complex))):
            # Just adding a scalar; put it in the D matrix
            A, B, C = self.A, self.B, self.C;
            D = self.D + other;

        else:
            # Check to make sure the dimensions are OK
            if ((self.inputs != other.inputs) or \
                    (self.outputs != other.outputs)):
                raise ValueError, "Systems have different shapes."

            # Concatenate the various arrays
            A = concatenate((
                concatenate((self.A, zeros((self.A.shape[0],
                                           other.A.shape[-1]))),axis=1),
                concatenate((zeros((other.A.shape[0], self.A.shape[-1])),
                                other.A),axis=1)
                            ),axis=0)
            B = concatenate((self.B, other.B), axis=0)
            C = concatenate((self.C, other.C), axis=1)
            D = self.D + other.D
        return StateSpace(A, B, C, D)

    # Reverse addition - just switch the arguments
    def __radd__(self, other): 
        """Add two state space systems"""
        return self.__add__(other)

    # Subtraction of two transfer functions (parallel interconnection)
    def __sub__(self, other):
        """Subtract two state space systems"""
        return __add__(self, other.__neg__())

    # Multiplication of two transfer functions (series interconnection)
    def __mul__(self, other):
        """Serial interconnection between two state space systems"""
        # Check for a couple of special cases
        if (isinstance(other, (int, long, float, complex))):
            # Just multiplying by a scalar; change the output
            A, B = self.A, self.B;
            C = self.C * other;
            D = self.D * other;

        else:
           # Check to make sure the dimensions are OK
           if (self.outputs != other.inputs):
               raise ValueError, "Number of first's outputs must match number \
                                    of second's inputs."

           # Concatenate the various arrays
           A = concatenate((
                   concatenate(( self.A, zeros((self.A.shape[0],
                                            other.A.shape[-1]))   ),axis=1),
                   concatenate(( other.B*self.C,  other.A  ),axis=1),
                   ),axis=0)
           B = concatenate( (self.B, other.B*self.D), axis=0 )
           C = concatenate( (other.D*self.C, other.C), axis=1 )
           D = other.D*self.D
        return StateSpace(A, B, C, D)

    # Reverse multiplication of two transfer functions (series interconnection)
    # Just need to convert LH argument to a state space object
    def __rmul__(self, other):
        """Serial interconnection between two state space systems"""
        # Check for a couple of special cases
        if (isinstance(other, (int, long, float, complex))):
            # Just multiplying by a scalar; change the input
            A, C = self.A, self.C;
            B = self.B * other;
            D = self.D * other;
            return StateSpace(A, B, C, D)

        else:
            raise TypeError("can't interconnect systems")

    # Feedback around a state space system
    def feedback(self, other, sign=-1):
        """Feedback interconnection between two state space systems"""
        # Check for special cases
        if (isinstance(other, (int, long, float, complex))):
            # Scalar feedback, create state space system that is this case
            other = StateSpace([[0]], [[0]], [[0]], [[ other ]])

        # Check to make sure the dimensions are OK
        if ((self.inputs != other.outputs) or (self.outputs != other.inputs)):
                raise ValueError, "State space systems don't have compatible inputs/outputs for feedback."

        # note that if there is an algebraic loop then this matrix inversion won't work
        # (I-D1 D2) or (I-D2 D1) will be singular
        # the easiest way to get this is to have D1 = I, D2 = I
        #! TODO: trap this error and report algebraic loop
        #! TODO: use determinant instead of inverse??
        from scipy.linalg import inv
        from numpy import eye
        E21 = inv(eye(self.outputs)+sign*self.D*other.D)
        E12 = inv(eye(self.inputs)+sign*other.D*self.D)
        
        A = concatenate((
                concatenate(( self.A-sign*self.B*E12*other.D*self.C, -sign*self.B*E12*other.C ),axis=1),
                concatenate(( other.B*E21*self.C, other.A-sign*other.B*E21*self.D*other.C ),axis=1),
               ),axis=0)
        B = concatenate( (self.B*E12, other.B*E21*self.D), axis=0 )
        C = concatenate( (E21*self.C, -sign*E21*self.D*other.C), axis=1 )
        D = E21*self.D

        return StateSpace(A, B, C, D)

#
# convertToStateSpace - create a state space system from another type
#
# To allow scalar constants to be used in a simple way (k*P, 1+L), this
# function allows the dimension of the input/output system to be specified
# in the case of a scalar system
#
def convertToStateSpace(sys, inputs=1, outputs=1):
    """Convert a system to state space form (if needed)"""
    if (isinstance(sys, StateSpace) or
        isinstance(sys, xferfcn.TransferFunction)):
        # Already a state space system; just return it
        return sys

    elif (isinstance(sys, (int, long, float, complex))):
        # Generate a simple state space system of the desired dimension
        # The following Doesn't work due to inconsistencies in ltisys:
        #   return StateSpace([[]], [[]], [[]], sp.eye(outputs, inputs))
        return StateSpace(-1, 0, 0, sp.eye(outputs, inputs))

    else:
        raise TypeError("can't convert given type to StateSpace system")
