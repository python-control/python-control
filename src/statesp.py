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
# $Id: statesp.py 21 2010-06-06 17:29:42Z murrayrm $

import scipy as sp
from scipy import concatenate, zeros
from numpy.linalg import solve
import xferfcn
from lti2 import Lti2

class StateSpace(Lti2):
    """The StateSpace class is used throughout the python-control library to
    represent systems in state space form.  This class is derived from the Lti2
    base class."""

    def __init__(self, A=0, B=0, C=0, D=1): 
        """StateSpace constructor.  The default constructor is the unit gain
        direct feedthrough system."""
        
        # Here we're going to convert inputs to matrices, if the user gave a
        # non-matrix type.
        matrices = [A, B, C, D] 
        for i in range(len(matrices)):
            # Convert to matrix first, if necessary.
            matrices[i] = sp.matrix(matrices[i])     
        [A, B, C, D] = matrices

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.states = A.shape[0]
        Lti2.__init__(self, B.shape[1], C.shape[0])
        
        # Check that the matrix sizes are consistent.
        if self.states != A.shape[1]:
            raise ValueError("A must be square.")
        if self.states != B.shape[0]:
            raise ValueError("B must have the same row size as A.")
        if self.states != C.shape[1]:
            raise ValueError("C must have the same column size as A.")
        if self.inputs != D.shape[1]:
            raise ValueError("D must have the same column size as B.")
        if self.outputs != D.shape[0]:
            raise ValueError("D must have the same row size as C.")

    def __str__(self):
        """Style to use for printing."""

        str =  "A = " + self.A.__str__() + "\n\n"
        str += "B = " + self.B.__str__() + "\n\n"
        str += "C = " + self.C.__str__() + "\n\n"
        str += "D = " + self.D.__str__() + "\n"
        return str

    def evalfr(self, freq):
        """Method for evaluating a system at one frequency."""
        
        fresp = self.C * solve(freq * 1.j * sp.eye(self.states) - self.A,
            self.B) + self.D
        return fresp

    # Method for generating the frequency response of the system
    def freqresp(self, omega=None):
        """Compute the response of a system to a list of frequencies."""
        
        # Preallocate outputs.
        numfreq = len(omega)
        mag = sp.empty((self.outputs, self.inputs, numfreq))
        phase = sp.empty((self.outputs, self.inputs, numfreq))
        fresp = sp.empty((self.outputs, self.inputs, numfreq), dtype=complex)

        for k in range(numfreq):
            fresp[:, :, k] = self.evalfr(omega[k])

        mag = abs(fresp)
        phase = sp.angle(fresp)

        return mag, phase, omega

    # Compute poles and zeros
    def poles(self):
        return sp.roots(sp.poly(self.A))

    def zeros(self): 
        den = sp.poly1d(sp.poly(self.A))

        # Compute the numerator based on zeros
        #! TODO: This is currently limited to SISO systems
        num = sp.poly1d(\
            sp.poly(self.A - sp.dot(self.B, self.C)) + (self.D[0] - 1) * den)

        return (sp.roots(num))

    # Negation of a system
    def __neg__(self):
        """Negate a state space system."""
        
        return StateSpace(self.A, self.B, -self.C, -self.D)

    # Addition of two transfer functions (parallel interconnection)
    def __add__(self, other):
        """Add two state space systems."""
        
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
        """Add two state space systems."""
        
        return self.__add__(other)

    # Subtraction of two transfer functions (parallel interconnection)
    def __sub__(self, other):
        """Subtract two state space systems."""
        
        return __add__(self, other.__neg__())

    # Multiplication of two transfer functions (series interconnection)
    def __mul__(self, other):
        """Serial interconnection between two state space systems."""
        
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
        """Feedback interconnection between two state space systems."""
        
        # Check for special cases
        if (isinstance(other, (int, long, float, complex))):
            # Scalar feedback, create state space system that is this case
            other = StateSpace([[0]], [[0]], [[0]], [[ other ]])

        # Check to make sure the dimensions are OK
        if ((self.inputs != other.outputs) or (self.outputs != other.inputs)):
                raise ValueError, "State space systems don't have compatible \
inputs/outputs for feedback."

        # note that if there is an algebraic loop then this matrix inversion
        # won't work
        # (I-D1 D2) or (I-D2 D1) will be singular
        # the easiest way to get this is to have D1 = I, D2 = I
        #! TODO: trap this error and report algebraic loop
        #! TODO: use determinant instead of inverse??
        from scipy.linalg import inv
        from numpy import eye
        E21 = inv(eye(self.outputs)+sign*self.D*other.D)
        E12 = inv(eye(self.inputs)+sign*other.D*self.D)
        
        A = concatenate((
                concatenate(( self.A-sign*self.B*E12*other.D*self.C,
                    -sign*self.B*E12*other.C ),axis=1),
                concatenate(( other.B*E21*self.C,
                    other.A-sign*other.B*E21*self.D*other.C ),axis=1),),
                axis=0)
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
    """Convert a system to state space form (if needed)."""
    
    if isinstance(sys, StateSpace):
        # Already a state space system; just return it
        return sys
    elif isinstance(sys, xferfcn.TransferFunction):
        pass # TODO: convert SS to TF
    elif (isinstance(sys, (int, long, float, complex))):
        # Generate a simple state space system of the desired dimension
        # The following Doesn't work due to inconsistencies in ltisys:
        #   return StateSpace([[]], [[]], [[]], sp.eye(outputs, inputs))
        return StateSpace(0, zeros((1, inputs)), zeros((outputs, 1)), 
            sys * sp.eye(outputs, inputs))
    else:
        raise TypeError("Can't convert given type to StateSpace system.")
    
def rss_generate(states, inputs, outputs, type):
    """This does the actual random state space generation expected from rss and
    drss.  type is 'c' for continuous systems and 'd' for discrete systems."""

    import numpy
    from numpy.random import rand, randn
    
    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Make some poles for A.  Preallocate a complex array.
    poles = numpy.zeros(states) + numpy.zeros(states) * 0.j
    i = 0

    while i < states:
        if rand() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i-1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i-1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i:i+2] = poles[i-2:i]
                i += 2
        elif rand() < pReal or i == states - 1:
            # No-oscillation pole.
            if type == 'c':
                poles[i] = -sp.exp(randn()) + 0.j
            elif type == 'd':
                poles[i] = 2. * rand() - 1.
            i += 1
        else:
            # Complex conjugate pair of oscillating poles.
            if type == 'c':
                poles[i] = complex(-sp.exp(randn()), 3. * sp.exp(randn()))
            elif type == 'd':
                mag = rand()
                phase = 2. * numpy.pi * rand()
                poles[i] = complex(mag * numpy.cos(phase), 
                    mag * numpy.sin(phase))
            poles[i+1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = numpy.zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i+1, i+1] = poles[i].real
            A[i, i+1] = poles[i].imag
            A[i+1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = randn(states, states)
        try:
            A = numpy.dot(numpy.linalg.solve(T, A), T) # A = T \ A * T
            break
        except numpy.linalg.linalg.LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask 
        if sp.any(Bmask): # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if sp.any(Cmask): # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = numpy.zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask

    return StateSpace(A, B, C, D)
