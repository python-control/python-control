"""statesp.py

State space representation and functions.

This file contains the StateSpace class, which is used to represent linear
systems in state space.  This is the primary representation for the
python-control library.

Routines in this module:

StateSpace.__init__
StateSpace._remove_useless_states
StateSpace.copy
StateSpace.__str__
StateSpace.__repr__
StateSpace.__neg__
StateSpace.__add__
StateSpace.__radd__
StateSpace.__sub__
StateSpace.__rsub__
StateSpace.__mul__
StateSpace.__rmul__
StateSpace.__div__
StateSpace.__rdiv__
StateSpace.evalfr
StateSpace.freqresp
StateSpace.pole
StateSpace.zero
StateSpace.feedback
StateSpace.returnScipySignalLti
StateSpace.append
StateSpace.__getitem__
_convertToStateSpace
_rss_generate

"""

# Python 3 compatability (needs to go here)
from __future__ import print_function

"""Copyright (c) 2010 by California Institute of Technology
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
Revised: Kevin K. Chen, Dec 10

$Id$
"""

from numpy import all, angle, any, array, asarray, concatenate, cos, delete, \
    dot, empty, exp, eye, matrix, ones, pi, poly, poly1d, roots, shape, sin, \
    zeros, squeeze
from numpy.random import rand, randn
from numpy.linalg import inv, det, solve
from numpy.linalg.linalg import LinAlgError
from scipy.signal import lti
# from exceptions import Exception
import warnings
from .lti import Lti, timebase, timebaseEqual, isdtime

class StateSpace(Lti):
    """The StateSpace class represents state space instances and functions.

    The StateSpace class is used throughout the python-control library to
    represent systems in state space form.  This class is derived from the Lti
    base class.

    The main data members are the A, B, C, and D matrices.  The class also
    keeps track of the number of states (i.e., the size of A).

    Discrete time state space system are implemented by using the 'dt' class
    variable and setting it to the sampling period.  If 'dt' is not None,
    then it must match whenever two state space systems are combined.
    Setting dt = 0 specifies a continuous system, while leaving dt = None
    means the system timebase is not specified.  If 'dt' is set to True, the
    system will be treated as a discrete time system with unspecified
    sampling time.
    """

    def __init__(self, *args):
        """Construct a state space object.

        The default constructor is StateSpace(A, B, C, D), where A, B, C, D are
        matrices or equivalent objects.  To call the copy constructor, call
        StateSpace(sys), where sys is a StateSpace object.

        """

        if len(args) == 4:
            # The user provided A, B, C, and D matrices.
            (A, B, C, D) = args
            dt = None;
        elif len(args) == 5:
            # Discrete time system
            (A, B, C, D, dt) = args
        elif len(args) == 1:
            # Use the copy constructor.
            if not isinstance(args[0], StateSpace):
                raise TypeError("The one-argument constructor can only take in \
a StateSpace object.  Recived %s." % type(args[0]))
            A = args[0].A
            B = args[0].B
            C = args[0].C
            D = args[0].D
            try:
                dt = args[0].dt
            except NameError:
                dt = None;
        else:
            raise ValueError("Needs 1 or 4 arguments; received %i." % len(args))

        # Here we're going to convert inputs to matrices, if the user gave a
        # non-matrix type.
        #! TODO: [A, B, C, D] = map(matrix, [A, B, C, D])?
        matrices = [A, B, C, D]
        for i in range(len(matrices)):
            # Convert to matrix first, if necessary.
            matrices[i] = matrix(matrices[i])
        [A, B, C, D] = matrices

        Lti.__init__(self, B.shape[1], C.shape[0], dt)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.states = A.shape[0]

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

        # Check for states that don't do anything, and remove them.
        self._remove_useless_states()

    def _remove_useless_states(self):
        """Check for states that don't do anything, and remove them.

        Scan the A, B, and C matrices for rows or columns of zeros.  If the
        zeros are such that a particular state has no effect on the input-output
        dynamics, then remove that state from the A, B, and C matrices.

        """

        # Indices of useless states.
        useless = []

        # Search for useless states.
        for i in range(self.states):
            if (all(self.A[i, :] == zeros((1, self.states))) and
                all(self.B[i, :] == zeros((1, self.inputs)))):
                useless.append(i)
                # To avoid duplucate indices in useless, jump to the next
                # iteration.
                continue
            if (all(self.A[:, i] == zeros((self.states, 1))) and
                all(self.C[:, i] == zeros((self.outputs, 1)))):
                useless.append(i)

        # Remove the useless states.
        if all(useless == range(self.states)):
            # All the states were useless.
            self.A = zeros((1, 1))
            self.B = zeros((1, self.inputs))
            self.C = zeros((self.outputs, 1))
        else:
            # A more typical scenario.
            self.A = delete(self.A, useless, 0)
            self.A = delete(self.A, useless, 1)
            self.B = delete(self.B, useless, 0)
            self.C = delete(self.C, useless, 1)

        self.states = self.A.shape[0]
        self.inputs = self.B.shape[1]
        self.outputs = self.C.shape[0]

    def __str__(self):
        """String representation of the state space."""

        str =  "A = " + self.A.__str__() + "\n\n"
        str += "B = " + self.B.__str__() + "\n\n"
        str += "C = " + self.C.__str__() + "\n\n"
        str += "D = " + self.D.__str__() + "\n"
        #! TODO: replace with standard calls to lti functions
        if (type(self.dt) == bool and self.dt == True):
            str += "\ndt unspecified\n"
        elif (not (self.dt is None) and type(self.dt) != bool and self.dt > 0):
            str += "\ndt = " + self.dt.__str__() + "\n"
        return str

    # represent as string, makes display work for IPython
    __repr__ = __str__

    # Negation of a system
    def __neg__(self):
        """Negate a state space system."""

        return StateSpace(self.A, self.B, -self.C, -self.D, self.dt)

    # Addition of two state space systems (parallel interconnection)
    def __add__(self, other):
        """Add two LTI systems (parallel connection)."""

        # Check for a couple of special cases
        if (isinstance(other, (int, float, complex))):
            # Just adding a scalar; put it in the D matrix
            A, B, C = self.A, self.B, self.C;
            D = self.D + other;
            dt = self.dt
        else:
            other = _convertToStateSpace(other)

            # Check to make sure the dimensions are OK
            if ((self.inputs != other.inputs) or
                    (self.outputs != other.outputs)):
                raise ValueError("Systems have different shapes.")

            # Figure out the sampling time to use
            if (self.dt == None and other.dt != None):
                dt = other.dt       # use dt from second argument
            elif (other.dt == None and self.dt != None) or \
                    (timebaseEqual(self, other)):
                dt = self.dt        # use dt from first argument
            else:
                raise ValueError("Systems have different sampling times")

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

        return StateSpace(A, B, C, D, dt)

    # Right addition - just switch the arguments
    def __radd__(self, other):
        """Right add two LTI systems (parallel connection)."""

        return self + other

    # Subtraction of two state space systems (parallel interconnection)
    def __sub__(self, other):
        """Subtract two LTI systems."""

        return self + (-other)

    def __rsub__(self, other):
        """Right subtract two LTI systems."""

        return other + (-self)

    # Multiplication of two state space systems (series interconnection)
    def __mul__(self, other):
        """Multiply two LTI objects (serial connection)."""

        # Check for a couple of special cases
        if isinstance(other, (int, float, complex)):
            # Just multiplying by a scalar; change the output
            A, B = self.A, self.B
            C = self.C * other
            D = self.D * other
            dt = self.dt
        else:
            other = _convertToStateSpace(other)

            # Check to make sure the dimensions are OK
            if self.inputs != other.outputs:
                raise ValueError("C = A * B: A has %i column(s) (input(s)), \
but B has %i row(s)\n(output(s))." % (self.inputs, other.outputs))

            # Figure out the sampling time to use
            if (self.dt == None and other.dt != None):
                dt = other.dt       # use dt from second argument
            elif (other.dt == None and self.dt != None) or \
                    (timebaseEqual(self, other)):
                dt = self.dt        # use dt from first argument
            else:
                raise ValueError("Systems have different sampling times")

            # Concatenate the various arrays
            A = concatenate(
                (concatenate((other.A, zeros((other.A.shape[0], self.A.shape[1]))),
                 axis=1),
                concatenate((self.B * other.C, self.A), axis=1)), axis=0)
            B = concatenate((other.B, self.B * other.D), axis=0)
            C = concatenate((self.D * other.C, self.C),axis=1)
            D = self.D * other.D

        return StateSpace(A, B, C, D, dt)

    # Right multiplication of two state space systems (series interconnection)
    # Just need to convert LH argument to a state space object
    # TODO: __rmul__ only works for special cases (??)
    def __rmul__(self, other):
        """Right multiply two LTI objects (serial connection)."""

        # Check for a couple of special cases
        if isinstance(other, (int, float, complex)):
            # Just multiplying by a scalar; change the input
            A, C = self.A, self.C;
            B = self.B * other;
            D = self.D * other;
            return StateSpace(A, B, C, D, self.dt)

        # is lti, and convertible?
        if isinstance(other, Lti):
            return _convertToStateSpace(other) * self

        # try to treat this as a matrix
        try:
            X = matrix(other)
            C = X * self.C
            D = X * self.D
            return StateSpace(self.A, self.B, C, D, self.dt)

        except Exception as e:
            print(e)
            pass
        raise TypeError("can't interconnect systems")

    # TODO: __div__ and __rdiv__ are not written yet.
    def __div__(self, other):
        """Divide two LTI systems."""

        raise NotImplementedError("StateSpace.__div__ is not implemented yet.")

    def __rdiv__(self, other):
        """Right divide two LTI systems."""

        raise NotImplementedError("StateSpace.__rdiv__ is not implemented yet.")

    # TODO: add discrete time check
    def evalfr(self, omega):
        """Evaluate a SS system's transfer function at a single frequency.

        self.evalfr(omega) returns the value of the transfer function matrix with
        input value s = i * omega.

        """
        # Figure out the point to evaluate the transfer function
        if isdtime(self, strict=True):
            dt = timebase(self)
            s = exp(1.j * omega * dt)
            if (omega * dt > pi):
                warnings.warn("evalfr: frequency evaluation above Nyquist frequency")
        else:
            s = omega * 1.j

        return self.horner(s)

    def horner(self, s):
        '''Evaluate the systems's transfer function for a complex variable

        Returns a matrix of values evaluated at complex variable s.
        '''
        resp = self.C * solve(s * eye(self.states) - self.A,
                              self.B) + self.D
        return array(resp)

    # Method for generating the frequency response of the system
    # TODO: add discrete time check
    def freqresp(self, omega):
        """Evaluate the system's transfer func. at a list of ang. frequencies.

        mag, phase, omega = self.freqresp(omega)

        reports the value of the magnitude, phase, and angular frequency of the
        system's transfer function matrix evaluated at s = i * omega, where
        omega is a list of angular frequencies, and is a sorted version of the
        input omega.

        """
        # Preallocate outputs.
        numfreq = len(omega)
        mag = empty((self.outputs, self.inputs, numfreq))
        phase = empty((self.outputs, self.inputs, numfreq))
        fresp = empty((self.outputs, self.inputs, numfreq), dtype=complex)

        omega.sort()

        # Evaluate response at each frequency
        for k in range(numfreq):
            fresp[:, :, k] = self.evalfr(omega[k])

        mag = abs(fresp)
        phase = angle(fresp)

        return mag, phase, omega

    # Compute poles and zeros
    def pole(self):
        """Compute the poles of a state space system."""

        return roots(poly(self.A))

    def zero(self):
        """Compute the zeros of a state space system."""

        if self.inputs > 1 or self.outputs > 1:
            raise NotImplementedError("StateSpace.zeros is currently \
implemented only for SISO systems.")

        den = poly1d(poly(self.A))
        # Compute the numerator based on zeros
        #! TODO: This is currently limited to SISO systems
        num = poly1d(poly(self.A - dot(self.B, self.C)) + ((self.D[0, 0] - 1) *
            den))

        return roots(num)

    # Feedback around a state space system
    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two LTI systems."""

        other = _convertToStateSpace(other)

        # Check to make sure the dimensions are OK
        if ((self.inputs != other.outputs) or (self.outputs != other.inputs)):
                raise ValueError("State space systems don't have compatible \
inputs/outputs for feedback.")

        # Figure out the sampling time to use
        if (self.dt == None and other.dt != None):
            dt = other.dt       # use dt from second argument
        elif (other.dt == None and self.dt != None) or \
                timebaseEqual(self, other):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        A1 = self.A
        B1 = self.B
        C1 = self.C
        D1 = self.D
        A2 = other.A
        B2 = other.B
        C2 = other.C
        D2 = other.D

        F = eye(self.inputs) - sign * D2 * D1
        if abs(det(F)) < 1.e-6:
            raise ValueError("I - sign * D2 * D1 is singular.")

        E = inv(F)
        T1 = eye(self.outputs) + sign * D1 * E * D2
        T2 = eye(self.inputs) + sign * E * D2 * D1

        A = concatenate(
            (concatenate(
                (A1 + sign * B1 * E * D2 * C1, sign * B1 * E * C2), axis=1),
            concatenate(
                (B2 * T1 * C1, A2 + sign * B2 * D1 * E * C2), axis=1)),
            axis=0)
        B = concatenate((B1 * T2, B2 * D1 * T2), axis=0)
        C = concatenate((T1 * C1, sign * D1 * E * C2), axis=1)
        D = D1 * T2

        return StateSpace(A, B, C, D, dt)

    def minreal(self, tol=0.0):
        """Calculate a minimal realization, removes unobservable and
        uncontrollable states"""
        try:
            from slycot import tb01pd
            B = empty((self.states, max(self.inputs, self.outputs)))
            B[:,:self.inputs] = self.B
            C = empty((max(self.outputs, self.inputs), self.states))
            C[:self.outputs,:] = self.C
            A, B, C, nr = tb01pd(self.states, self.inputs, self.outputs,
                                    self.A, B, C, tol=tol)
            return StateSpace(A[:nr,:nr], B[:nr,:self.inputs],
                              C[:self.outputs,:nr], self.D)
        except ImportError:
            raise TypeError("minreal requires slycot tb01pd")

    # TODO: add discrete time check
    def returnScipySignalLti(self):
        """Return a list of a list of scipy.signal.lti objects.

        For instance,

        >>> out = ssobject.returnScipySignalLti()
        >>> out[3][5]

        is a signal.scipy.lti object corresponding to the transfer function from
        the 6th input to the 4th output."""

        # Preallocate the output.
        out = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = lti(asarray(self.A), asarray(self.B[:, j]),
                                asarray(self.C[i, :]), asarray(self.D[i, j]))

        return out

    def append(self, other):
        """Append a second model to the present model. The second
        model is converted to state-space if necessary, inputs and
        outputs are appended and their order is preserved"""
        if not isinstance(other, StateSpace):
            other = _convertToStateSpace(other)

        if self.dt != other.dt:
            raise ValueError("Systems must have the same time step")

        n = self.states + other.states
        m = self.inputs + other.inputs
        p = self.outputs + other.outputs
        A = zeros( (n, n) )
        B = zeros( (n, m) )
        C = zeros( (p, n) )
        D = zeros( (p, m) )
        A[:self.states,:self.states] = self.A
        A[self.states:,self.states:] = other.A
        B[:self.states,:self.inputs] = self.B
        B[self.states:,self.inputs:] = other.B
        C[:self.outputs,:self.states] = self.C
        C[self.outputs:,self.states:] = other.C
        D[:self.outputs,:self.inputs] = self.D
        D[self.outputs:,self.inputs:] = other.D
        return StateSpace(A, B, C, D, self.dt)

    def __getitem__(self, indices):
        """Array style acces"""
        if len(indices) != 2:
            raise IOError('must provide indices of length 2 for state space')
        i = indices[0]
        j = indices[1]
        return StateSpace(self.A,
                self.B[:,j],
                self.C[i,:],
                self.D[i,j], self.dt)

# TODO: add discrete time check
def _convertToStateSpace(sys, **kw):
    """Convert a system to state space form (if needed).

    If sys is already a state space, then it is returned.  If sys is a transfer
    function object, then it is converted to a state space and returned.  If sys
    is a scalar, then the number of inputs and outputs can be specified
    manually, as in:

    >>> sys = _convertToStateSpace(3.) # Assumes inputs = outputs = 1
    >>> sys = _convertToStateSpace(1., inputs=3, outputs=2)

    In the latter example, A = B = C = 0 and D = [[1., 1., 1.]
                                                  [1., 1., 1.]].

    """

    from .xferfcn import TransferFunction
    if isinstance(sys, StateSpace):
        if len(kw):
            raise TypeError("If sys is a StateSpace, _convertToStateSpace \
cannot take keywords.")

        # Already a state space system; just return it
        return sys
    elif isinstance(sys, TransferFunction):
        try:
            from slycot import td04ad
            if len(kw):
                raise TypeError("If sys is a TransferFunction, _convertToStateSpace \
    cannot take keywords.")

            # Change the numerator and denominator arrays so that the transfer
            # function matrix has a common denominator.
            num, den = sys._common_den()
            # Make a list of the orders of the denominator polynomials.
            index = [len(den) - 1 for i in range(sys.outputs)]
            # Repeat the common denominator along the rows.
            den = array([den for i in range(sys.outputs)])
            #! TODO: transfer function to state space conversion is still buggy!
            #print num
            #print shape(num)
            ssout = td04ad('R',sys.inputs, sys.outputs, index, den, num,tol=0.0)

            states = ssout[0]
            return StateSpace(ssout[1][:states, :states],
                ssout[2][:states, :sys.inputs],
                ssout[3][:sys.outputs, :states],
                ssout[4], sys.dt)
        except ImportError:
            # TODO: do we want to squeeze first and check dimenations?
            # I think this will fail if num and den aren't 1-D after
            # the squeeze
            lti_sys = lti(squeeze(sys.num), squeeze(sys.den))
            return StateSpace(lti_sys.A, lti_sys.B, lti_sys.C, lti_sys.D,
                              sys.dt)

    elif isinstance(sys, (int, float, complex)):
        if "inputs" in kw:
            inputs = kw["inputs"]
        else:
            inputs = 1
        if "outputs" in kw:
            outputs = kw["outputs"]
        else:
            outputs = 1

        # Generate a simple state space system of the desired dimension
        # The following Doesn't work due to inconsistencies in ltisys:
        #   return StateSpace([[]], [[]], [[]], eye(outputs, inputs))
        return StateSpace(0., zeros((1, inputs)), zeros((outputs, 1)),
            sys * ones((outputs, inputs)))

    # If this is a matrix, try to create a constant feedthrough
    try:
        D = matrix(sys)
        outputs, inputs = D.shape

        return StateSpace(0., zeros((1, inputs)), zeros((outputs, 1)), D)
    except Exception(e):
        print("Failure to assume argument is matrix-like in" \
            " _convertToStateSpace, result %s" % e)

    raise TypeError("Can't convert given type to StateSpace system.")

# TODO: add discrete time option
def _rss_generate(states, inputs, outputs, type):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  type is 'c' for continuous systems and 'd' for discrete systems.

    """

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

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." %
            states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." %
            inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." %
            outputs)

    # Make some poles for A.  Preallocate a complex array.
    poles = zeros(states) + zeros(states) * 0.j
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
                poles[i] = -exp(randn()) + 0.j
            elif type == 'd':
                poles[i] = 2. * rand() - 1.
            i += 1
        else:
            # Complex conjugate pair of oscillating poles.
            if type == 'c':
                poles[i] = complex(-exp(randn()), 3. * exp(randn()))
            elif type == 'd':
                mag = rand()
                phase = 2. * pi * rand()
                poles[i] = complex(mag * cos(phase),
                    mag * sin(phase))
            poles[i+1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = zeros((states, states))
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
            A = dot(solve(T, A), T) # A = T \ A * T
            break
        except LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask
        if any(Bmask): # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if any(Cmask): # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask

    return StateSpace(A, B, C, D)

# Convert a MIMO system to a SISO system
# TODO: add discrete time check
def _mimo2siso(sys, input, output, warn_conversion=False):
    #pylint: disable=W0622
    """
    Convert a MIMO system to a SISO system. (Convert a system with multiple
    inputs and/or outputs, to a system with a single input and output.)

    The input and output that are used in the SISO system can be selected
    with the parameters ``input`` and ``output``. All other inputs are set
    to 0, all other outputs are ignored.

    If ``sys`` is already a SISO system, it will be returned unaltered.

    Parameters
    ----------
    sys: StateSpace
        Linear (MIMO) system that should be converted.
    input: int
        Index of the input that will become the SISO system's only input.
    output: int
        Index of the output that will become the SISO system's only output.
    warn_conversion: bool
        If True: print a warning message when sys is a MIMO system.
        Warn that a conversion will take place.

    Returns:

    sys: StateSpace
        The converted (SISO) system.
    """
    if not (isinstance(input, int) and isinstance(output, int)):
        raise TypeError("Parameters ``input`` and ``output`` must both "
                        "be integer numbers.")
    if not (0 <= input < sys.inputs):
        raise ValueError("Selected input does not exist. "
                         "Selected input: {sel}, "
                         "number of system inputs: {ext}."
                         .format(sel=input, ext=sys.inputs))
    if not (0 <= output < sys.outputs):
        raise ValueError("Selected output does not exist. "
                         "Selected output: {sel}, "
                         "number of system outputs: {ext}."
                         .format(sel=output, ext=sys.outputs))
    #Convert sys to SISO if necessary
    if sys.inputs > 1 or sys.outputs > 1:
        if warn_conversion:
            warnings.warn("Converting MIMO system to SISO system. "
                          "Only input {i} and output {o} are used."
                          .format(i=input, o=output))
        # $X = A*X + B*U
        #  Y = C*X + D*U
        new_B = sys.B[:, input]
        new_C = sys.C[output, :]
        new_D = sys.D[output, input]
        sys = StateSpace(sys.A, new_B, new_C, new_D, sys.dt)

    return sys

def _mimo2simo(sys, input, warn_conversion=False):
    #pylint: disable=W0622
    """
    Convert a MIMO system to a SIMO system. (Convert a system with multiple
    inputs and/or outputs, to a system with a single input but possibly
    multiple outputs.)

    The input that is used in the SIMO system can be selected with the
    parameter ``input``. All other inputs are set to 0, all other
    outputs are ignored.

    If ``sys`` is already a SIMO system, it will be returned unaltered.

    Parameters
    ----------
    sys: StateSpace
        Linear (MIMO) system that should be converted.
    input: int
        Index of the input that will become the SIMO system's only input.
    warn_conversion: bool
        If True: print a warning message when sys is a MIMO system.
        Warn that a conversion will take place.

    Returns:
    --------
    sys: StateSpace
        The converted (SIMO) system.
    """
    if not (isinstance(input, int)):
        raise TypeError("Parameter ``input`` be an integer number.")
    if not (0 <= input < sys.inputs):
        raise ValueError("Selected input does not exist. "
                         "Selected input: {sel}, "
                         "number of system inputs: {ext}."
                         .format(sel=input, ext=sys.inputs))
    #Convert sys to SISO if necessary
    if sys.inputs > 1:
        if warn_conversion:
            warnings.warn("Converting MIMO system to SIMO system. "
                          "Only input {i} is used."
                          .format(i=input))
        # $X = A*X + B*U
        #  Y = C*X + D*U
        new_B = sys.B[:, input]
        new_D = sys.D[:, input]
        sys = StateSpace(sys.A, new_B, sys.C, new_D, sys.dt)

    return sys
