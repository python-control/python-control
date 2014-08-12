"""xferfcn.py

Transfer function representation and functions.

This file contains the TransferFunction class and also functions
that operate on transfer functions.  This is the primary representation
for the python-control library.

Routines in this module:

TransferFunction.__init__
TransferFunction._truncatecoeff
TransferFunction.copy
TransferFunction.__str__
TransferFunction.__repr__
TransferFunction.__neg__
TransferFunction.__add__
TransferFunction.__radd__
TransferFunction.__sub__
TransferFunction.__rsub__
TransferFunction.__mul__
TransferFunction.__rmul__
TransferFunction.__div__
TransferFunction.__rdiv__
TransferFunction.__truediv__
TransferFunction.__rtruediv__
TransferFunction.evalfr
TransferFunction.freqresp
TransferFunction.pole
TransferFunction.zero
TransferFunction.feedback
TransferFunction.minreal
TransferFunction.returnScipySignalLti
TransferFunction._common_den
_tfpolyToString
_addSISO
_convertToTransferFunction

"""

# Python 3 compatability (needs to go here)
from __future__ import print_function
from __future__ import division

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
Revised: Kevin K. Chewn, Dec 10

$Id$

"""

# External function declarations
from numpy import angle, any, array, empty, finfo, insert, ndarray, ones, \
    polyadd, polymul, polyval, roots, sort, sqrt, zeros, squeeze, exp, pi, \
    where, delete, real, poly, poly1d
from scipy.signal import lti
from copy import deepcopy
from warnings import warn
from .lti import Lti, timebaseEqual, timebase, isdtime


class TransferFunction(Lti):

    """The TransferFunction class represents TF instances and functions.

    The TransferFunction class is derived from the Lti parent class.  It
    is used throught the python-control library to represent systems in
    transfer function form.

    The main data members are 'num' and 'den', which are 2-D lists of arrays
    containing MIMO numerator and denominator coefficients.  For example,

    >>> num[2][5] = numpy.array([1., 4., 8.])

    means that the numerator of the transfer function from the 6th input to the
    3rd output is set to s^2 + 4s + 8.

    Discrete time transfer functions are implemented by using the 'dt' class
    variable and setting it to something other than 'None'.  If 'dt' has a
    non-zero value, then it must match whenever two transfer functions are
    combined.  If 'dt' is set to True, the system will be treated as a
    discrete time system with unspecified sampling time.
    """

    def __init__(self, *args):
        """Construct a transfer function.

        The default constructor is TransferFunction(num, den), where num and
        den are lists of lists of arrays containing polynomial coefficients.
        To crete a discrete time transfer funtion, use TransferFunction(num,
        den, dt).  To call the copy constructor, call TransferFunction(sys),
        where sys is a TransferFunction object (continuous or discrete).

        """

        if len(args) == 2:
            # The user provided a numerator and a denominator.
            (num, den) = args
            dt = None
        elif len(args) == 3:
            # Discrete time transfer function
            (num, den, dt) = args
        elif len(args) == 1:
            # Use the copy constructor.
            if not isinstance(args[0], TransferFunction):
                raise TypeError("The one-argument constructor can only take \
                        in a TransferFunction object.  Received %s."
                                % type(args[0]))
            num = args[0].num
            den = args[0].den
            try:
                dt = args[0].dt
            except NameError:
                dt = None
        else:
            raise ValueError("Needs 1, 2 or 3 arguments; received %i."
                             % len(args))

        # Make num and den into lists of lists of arrays, if necessary.
        # Beware: this is a shallow copy! This should be okay,
        # but be careful.
        data = [num, den]
        for i in range(len(data)):
            if isinstance(data[i], (int, float, complex)):
                # Convert scalar to list of list of array.
                if (isinstance(data[i], int)):
                    # Convert integers to floats at this point
                    data[i] = [[array([data[i]], dtype=float)]]
                else:
                    data[i] = [[array([data[i]])]]
            elif (isinstance(data[i], (list, tuple, ndarray)) and
                    isinstance(data[i][0], (int, float, complex))):
                # Convert array to list of list of array.
                if (isinstance(data[i][0], int)):
                    # Convert integers to floats at this point
                    #! Not sure this covers all cases correctly
                    data[i] = [[array(data[i], dtype=float)]]
                else:
                    data[i] = [[array(data[i])]]
            elif (isinstance(data[i], list) and
                    isinstance(data[i][0], list) and
                    isinstance(data[i][0][0], (list, tuple, ndarray)) and
                    isinstance(data[i][0][0][0], (int, float, complex))):
                # We might already have the right format.  Convert the
                # coefficient vectors to arrays, if necessary.
                for j in range(len(data[i])):
                    for k in range(len(data[i][j])):
                        if (isinstance(data[i][j][k], int)):
                            data[i][j][k] = array(data[i][j][k], dtype=float)
                        else:
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

        Lti.__init__(self, inputs, outputs, dt)
        self.num = num
        self.den = den

        self._truncatecoeff()

    def __call__(self, s):
        """Evaluate the system's transfer function for a complex vairable

        For a SISO transfer function, returns the value of the
        transfer function.  For a MIMO transfer fuction, returns a
        matrix of values evaluated at complex variable s."""

        if (self.inputs > 1 or self.outputs > 1):
            # MIMO transfer function, return a matrix
            return self.horner(s)
        else:
            # SISO transfer function, return a scalar
            return self.horner(s)[0][0]

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
                        if (data[p][i][j][k]):
                            nonzero = k
                            break

                    if nonzero is None:
                        # The array is all zeros.
                        data[p][i][j] = zeros(1)
                    else:
                        # Truncate the trivial coefficients.
                        data[p][i][j] = data[p][i][j][nonzero:]
        [self.num, self.den] = data

    def __str__(self, var=None):
        """String representation of the transfer function."""

        mimo = self.inputs > 1 or self.outputs > 1
        if (var is None):
            #! TODO: replace with standard calls to lti functions
            var = 's' if self.dt is None or self.dt == 0 else 'z'
        outstr = ""

        for i in range(self.inputs):
            for j in range(self.outputs):
                if mimo:
                    outstr += "\nInput %i to output %i:" % (i + 1, j + 1)

                # Convert the numerator and denominator polynomials to strings.
                numstr = _tfpolyToString(self.num[j][i], var=var)
                denstr = _tfpolyToString(self.den[j][i], var=var)

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

        # See if this is a discrete time system with specific sampling time
        if (not (self.dt is None) and type(self.dt) != bool and self.dt > 0):
            #! TODO: replace with standard calls to lti functions
            outstr += "\ndt = " + self.dt.__str__() + "\n"

        return outstr

    # represent as string, makes display work for IPython
    __repr__ = __str__

    def __neg__(self):
        """Negate a transfer function."""

        num = deepcopy(self.num)
        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j] *= -1

        return TransferFunction(num, self.den, self.dt)

    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""
        from .statesp import StateSpace

        # Convert the second argument to a transfer function.
        if (isinstance(other, StateSpace)):
            other = _convertToTransferFunction(other)
        elif not isinstance(other, TransferFunction):
            other = _convertToTransferFunction(other, inputs=self.inputs,
                                               outputs=self.outputs)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.inputs:
            raise ValueError("The first summand has %i input(s), but the \
second has %i." % (self.inputs, other.inputs))
        if self.outputs != other.outputs:
            raise ValueError("The first summand has %i output(s), but the \
second has %i." % (self.outputs, other.outputs))

        # Figure out the sampling time to use
        if (self.dt is None and other.dt is not None):
            dt = other.dt       # use dt from second argument
        elif (other.dt is None and self.dt is not None) or \
                (timebaseEqual(self, other)):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        # Preallocate the numerator and denominator of the sum.
        num = [[[] for j in range(self.inputs)] for i in range(self.outputs)]
        den = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                num[i][j], den[i][j] = _addSISO(self.num[i][j], self.den[i][j],
                                                other.num[i][j],
                                                other.den[i][j])

        return TransferFunction(num, den, dt)

    def __radd__(self, other):
        """Right add two LTI objects (parallel connection)."""
        return self + other

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
            other = _convertToTransferFunction(other, inputs=self.inputs,
                                               outputs=self.inputs)
        else:
            other = _convertToTransferFunction(other)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.outputs:
            raise ValueError("C = A * B: A has %i column(s) (input(s)), but B \
has %i row(s)\n(output(s))." % (self.inputs, other.outputs))

        inputs = other.inputs
        outputs = self.outputs

        # Figure out the sampling time to use
        if (self.dt is None and other.dt is not None):
            dt = other.dt       # use dt from second argument
        elif (other.dt is None and self.dt is not None) or \
                (self.dt == other.dt):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]

        # Temporary storage for the summands needed to
        # find the (i, j)th element of the product.
        num_summand = [[] for k in range(self.inputs)]
        den_summand = [[] for k in range(self.inputs)]

        for i in range(outputs):  # Iterate through rows of product.
            for j in range(inputs):  # Iterate through columns of product.
                for k in range(self.inputs):  # Multiply & add.
                    num_summand[k] = polymul(self.num[i][k], other.num[k][j])
                    den_summand[k] = polymul(self.den[i][k], other.den[k][j])
                    num[i][j], den[i][j] = _addSISO(
                        num[i][j], den[i][j],
                        num_summand[k], den_summand[k])

        return TransferFunction(num, den, dt)

    def __rmul__(self, other):
        """Right multiply two LTI objects (serial connection)."""

        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, complex)):
            other = _convertToTransferFunction(other, inputs=self.inputs,
                                               outputs=self.inputs)
        else:
            other = _convertToTransferFunction(other)

        # Check that the input-output sizes are consistent.
        if other.inputs != self.outputs:
            raise ValueError("C = A * B: A has %i column(s) (input(s)), but B \
has %i row(s)\n(output(s))." % (other.inputs, self.outputs))

        inputs = self.inputs
        outputs = other.outputs

        # Figure out the sampling time to use
        if (self.dt is None and other.dt is not None):
            dt = other.dt       # use dt from second argument
        elif (other.dt is None and self.dt is not None) \
                or (self.dt == other.dt):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]

        # Temporary storage for the summands needed to find the
        # (i, j)th element
        # of the product.
        num_summand = [[] for k in range(other.inputs)]
        den_summand = [[] for k in range(other.inputs)]

        for i in range(outputs):  # Iterate through rows of product.
            for j in range(inputs):  # Iterate through columns of product.
                for k in range(other.inputs):  # Multiply & add.
                    num_summand[k] = polymul(other.num[i][k], self.num[k][j])
                    den_summand[k] = polymul(other.den[i][k], self.den[k][j])
                    num[i][j], den[i][j] = _addSISO(
                        num[i][j], den[i][j],
                        num_summand[k], den_summand[k])

        return TransferFunction(num, den, dt)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""

        if isinstance(other, (int, float, complex)):
            other = _convertToTransferFunction(
                other, inputs=self.inputs,
                outputs=self.inputs)
        else:
            other = _convertToTransferFunction(other)

        if (self.inputs > 1 or self.outputs > 1 or
                other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError(
                "TransferFunction.__truediv__ is currently \
                implemented only for SISO systems.")

        # Figure out the sampling time to use
        if (self.dt is None and other.dt is not None):
            dt = other.dt       # use dt from second argument
        elif (other.dt is None and self.dt is not None)\
                or (self.dt == other.dt):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        num = polymul(self.num[0][0], other.den[0][0])
        den = polymul(self.den[0][0], other.num[0][0])

        return TransferFunction(num, den, dt)

    # TODO: Remove when transition to python3 complete
    def __div__(self, other):
        return TransferFunction.__truediv__(self, other)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rtruediv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, complex)):
            other = _convertToTransferFunction(
                other, inputs=self.inputs,
                outputs=self.inputs)
        else:
            other = _convertToTransferFunction(other)

        if (self.inputs > 1 or self.outputs > 1 or
                other.inputs > 1 or other.outputs > 1):
            raise NotImplementedError(
                "TransferFunction.__rtruediv__ is currently \
                implemented only for SISO systems.")

        return other / self

    # TODO: Remove when transition to python3 complete
    def __rdiv__(self, other):
        return TransferFunction.__rtruediv__(self, other)

    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return TransferFunction([1], [1])  # unity
        if other > 0:
            return self * (self**(other-1))
        if other < 0:
            return (TransferFunction([1], [1]) / self) * (self**(other+1))

    def evalfr(self, omega):
        """Evaluate a transfer function at a single angular frequency.

        self.evalfr(omega) returns the value of the
        transfer function matrix with
        input value s = i * omega.

        """

        # TODO: implement for discrete time systems
        if isdtime(self, strict=True):
            # Convert the frequency to discrete time
            dt = timebase(self)
            s = exp(1.j * omega * dt)
            if (omega * dt > pi):
                warn("evalfr: frequency evaluation above Nyquist frequency")
        else:
            s = 1.j * omega

        return self.horner(s)

    def horner(self, s):
        """Evaluate the systems's transfer function for a complex variable

        Returns a matrix of values evaluated at complex variable s.
        """

        # Preallocate the output.
        if getattr(s, '__iter__', False):
            out = empty((self.outputs, self.inputs, len(s)), dtype=complex)
        else:
            out = empty((self.outputs, self.inputs), dtype=complex)

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = (polyval(self.num[i][j], s) /
                             polyval(self.den[i][j], s))

        return out

    # Method for generating the frequency response of the system
    def freqresp(self, omega):
        """Evaluate a transfer function at a list of angular frequencies.

        mag, phase, omega = self.freqresp(omega)

        reports the value of the magnitude, phase, and angular frequency of the
        transfer function matrix evaluated at s = i * omega, where omega is a
        list of angular frequencies, and is a sorted
        version of the input omega.

        """

        # Preallocate outputs.
        numfreq = len(omega)
        mag = empty((self.outputs, self.inputs, numfreq))
        phase = empty((self.outputs, self.inputs, numfreq))

        # Figure out the frequencies
        omega.sort()
        if isdtime(self, strict=True):
            dt = timebase(self)
            slist = map(lambda w: exp(1.j * w * dt), omega)
            if (max(omega) * dt > pi):
                warn("evalfr: frequency evaluation above Nyquist frequency")
        else:
            slist = map(lambda w: 1.j * w, omega)

        # Compute frequency response for each input/output pair
        for i in range(self.outputs):
            for j in range(self.inputs):
                fresp = map(lambda s: (polyval(self.num[i][j], s) /
                            polyval(self.den[i][j], s)), slist)
                fresp = array(list(fresp))

                mag[i, j, :] = abs(fresp)
                phase[i, j, :] = angle(fresp)

        return mag, phase, omega

    def pole(self):
        """Compute the poles of a transfer function."""
        num, den = self._common_den()
        return roots(den)

    def zero(self):
        """Compute the zeros of a transfer function."""
        if self.inputs > 1 or self.outputs > 1:
            raise NotImplementedError("TransferFunction.zero is currently \
only implemented for SISO systems.")
        else:
            #for now, just give zeros of a SISO tf
            return roots(self.num[0][0])

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two LTI objects."""
        other = _convertToTransferFunction(other)

        if (self.inputs > 1 or self.outputs > 1 or
                other.inputs > 1 or other.outputs > 1):
            # TODO: MIMO feedback
            raise NotImplementedError("TransferFunction.feedback is currently \
only implemented for SISO functions.")

        # Figure out the sampling time to use
        if (self.dt is None and other.dt is not None):
            dt = other.dt       # use dt from second argument
        elif (other.dt is None and self.dt is not None) \
                or (self.dt == other.dt):
            dt = self.dt        # use dt from first argument
        else:
            raise ValueError("Systems have different sampling times")

        num1 = self.num[0][0]
        den1 = self.den[0][0]
        num2 = other.num[0][0]
        den2 = other.den[0][0]

        num = polymul(num1, den2)
        den = polyadd(polymul(den2, den1), -sign * polymul(num2, num1))

        return TransferFunction(num, den, dt)

        # For MIMO or SISO systems, the analytic expression is
        #     self / (1 - sign * other * self)
        # But this does not work correctly because the state size will be too
        # large.

    def minreal(self, tol=None):
        """Remove cancelling pole/zero pairs from a transfer function"""
        # based on octave minreal

        # default accuracy
        from sys import float_info
        sqrt_eps = sqrt(float_info.epsilon)

        # pre-allocate arrays
        num = [[[] for j in range(self.inputs)] for i in range(self.outputs)]
        den = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):

                # split up in zeros, poles and gain
                newzeros = []
                zeros = roots(self.num[i][j])
                poles = roots(self.den[i][j])
                gain = self.num[i][j][0] / self.den[i][j][0]

                # check all zeros
                for z in zeros:
                    t = tol or \
                        1000 * max(float_info.epsilon, abs(z) * sqrt_eps)
                    idx = where(abs(z - poles) < t)[0]
                    if len(idx):
                        # cancel this zero against one of the poles
                        poles = delete(poles, idx[0])
                    else:
                        # keep this zero
                        newzeros.append(z)

                # keep result
                if len(newzeros):
                    num[i][j] = gain * real(poly(newzeros))
                else:
                    num[i][j] = array([gain])
                den[i][j] = real(poly(poles))

        # end result
        return TransferFunction(num, den)

    def returnScipySignalLti(self):
        """Return a list of a list of scipy.signal.lti objects.

        For instance,

        >>> out = tfobject.returnScipySignalLti()
        >>> out[3][5]

        is a signal.scipy.lti object corresponding to the
        transfer function from the 6th input to the 4th output.

        """

        # TODO: implement for discrete time systems
        if (self.dt != 0 and self.dt is not None):
            raise NotImplementedError("Function not \
                    implemented in discrete time")

        # Preallocate the output.
        out = [[[] for j in range(self.inputs)] for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                out[i][j] = lti(self.num[i][j], self.den[i][j])

        return out

    def _common_den(self, imag_tol=None):
        """
        Compute MIMO common denominator; return it and an adjusted numerator.

        This function computes the single denominator containing all
        the poles of sys.den, and reports it as the array d.  The
        output numerator array n is modified to use the common
        denominator; the coefficient arrays are also padded with zeros
        to be the same size as d.  n is an sys.outputs by sys.inputs
        by len(d) array.

        Parameters
        ----------
        imag_tol: float
            Threshold for the imaginary part of a root to use in detecting
            complex poles

        Returns
        -------
        num: array
            Multi-dimensional array of numerator coefficients. num[i][j]
            gives the numerator coefficient array for the ith input and jth
            output

        den: array
            Array of coefficients for common denominator polynomial

        Examples
        --------
        >>> n, d = sys._common_den()

        """

        # Machine precision for floats.
        eps = finfo(float).eps

        # Decide on the tolerance to use in deciding of a pole is complex
        if (imag_tol is None):
            imag_tol = 1e-8     # TODO: figure out the right number to use

        # A sorted list to keep track of cumulative poles found as we scan
        # self.den.
        poles = []

        # A 3-D list to keep track of common denominator poles not present in
        # the self.den[i][j].
        missingpoles = [[[] for j in range(self.inputs)]
                        for i in range(self.outputs)]

        for i in range(self.outputs):
            for j in range(self.inputs):
                # A sorted array of the poles of this SISO denominator.
                currentpoles = sort(roots(self.den[i][j]))

                cp_ind = 0  # Index in currentpoles.
                p_ind = 0  # Index in poles.

                # Crawl along the list of current poles and the list of
                # cumulative poles, until one of them reaches the end.  Keep in
                # mind that both lists are always sorted.
                while cp_ind < len(currentpoles) and p_ind < len(poles):
                    if abs(currentpoles[cp_ind] - poles[p_ind]) < (10 * eps):
                        # If the current element of both
                        # lists match, then we're
                        # good.  Move to the next pair of elements.
                        cp_ind += 1
                    elif currentpoles[cp_ind] < poles[p_ind]:
                        # We found a pole in this transfer function that's not
                        # in the list of cumulative poles.  Add it to the list.
                        poles.insert(p_ind, currentpoles[cp_ind])
                        # Now mark this pole as "missing" in all previous
                        # denominators.
                        for k in range(i):
                            for m in range(self.inputs):
                                # All previous rows.
                                missingpoles[k][m].append(currentpoles[cp_ind])
                        for m in range(j):
                            # This row only.
                            missingpoles[i][m].append(currentpoles[cp_ind])
                        cp_ind += 1
                    else:
                        # There is a pole in the cumulative list of poles that
                        # is not in our transfer function denominator.  Mark
                        # this pole as "missing", and do not increment cp_ind.
                        missingpoles[i][j].append(poles[p_ind])
                    p_ind += 1

                if cp_ind == len(currentpoles) and p_ind < len(poles):
                    # If we finished scanning currentpoles first, then all the
                    # remaining cumulative poles are missing poles.
                    missingpoles[i][j].extend(poles[p_ind:])
                elif cp_ind < len(currentpoles) and p_ind == len(poles):
                    # If we finished scanning the cumulative poles first, then
                    # all the reamining currentpoles need to be added to poles.
                    poles.extend(currentpoles[cp_ind:])
                    # Now mark these poles as "missing" in previous
                    # denominators.
                    for k in range(i):
                        for m in range(self.inputs):
                            # All previous rows.
                            missingpoles[k][m].extend(currentpoles[cp_ind:])
                    for m in range(j):
                        # This row only.
                        missingpoles[i][m].extend(currentpoles[cp_ind:])

        # Construct the common denominator.
        den = 1.
        n = 0
        while n < len(poles):
            if abs(poles[n].imag) > 10 * eps:
                # To prevent buildup of imaginary part error, handle complex
                # pole pairs together.
                #
                # Because we might have repeated real parts of poles
                # and the fact that we are using lexigraphical
                # ordering, we can't just combine adjacent poles.
                # Instead, we have to figure out the multiplicity
                # first, then multiple the pairs from the outside in.

                # Figure out the multiplicity
                m = 1          # multiplicity count
                while (n+m < len(poles) and
                       poles[n].real == poles[n+m].real and
                       poles[n].imag * poles[n+m].imag > 0):
                    m += 1

                if (m > 1):
                    print("Found pole with multiplicity %d" % m)
                    # print("Poles = ", poles)

                # Multiple pairs from the outside in
                for i in range(m):
                    quad = polymul([1., -poles[n]], [1., -poles[n+2*(m-i)-1]])
                    assert all(quad.imag < 10 * eps), \
                        "Quadratic has a nontrivial imaginary part: %g" \
                        % quad.imag.max()

                    den = polymul(den, quad.real)
                    n += 1      # move to next pair
                n += m          # skip past conjugate pairs
            else:
                den = polymul(den, [1., -poles[n].real])
                n += 1

        # Modify the numerators so that they each take the common denominator.
        num = deepcopy(self.num)
        if isinstance(den, float):
            den = array([den])

        for i in range(self.outputs):
            for j in range(self.inputs):
                # The common denominator has leading coefficient 1.  Scale out
                # the existing denominator's leading coefficient.
                assert self.den[i][j][0], "The i = %i, j = %i denominator has \
a zero leading coefficient." % (i, j)
                num[i][j] = num[i][j] / self.den[i][j][0]

                # Multiply in the missing poles.
                for p in missingpoles[i][j]:
                    num[i][j] = polymul(num[i][j], [1., -p])

        # Pad all numerator polynomials with zeros so that the numerator arrays
        # are the same size as the denominator.
        for i in range(self.outputs):
            for j in range(self.inputs):
                pad = len(den) - len(num[i][j])
                if (pad > 0):
                    num[i][j] = insert(
                        num[i][j], zeros(pad, dtype=int),
                        zeros(pad))

        # Finally, convert the numerator to a 3-D array.
        num = array(num)
        # Remove trivial imaginary parts.
        # Check for nontrivial imaginary parts.
        if any(abs(num.imag) > sqrt(eps)):
            print ("Warning: The numerator has a nontrivial imaginary part: %g"
                   % abs(num.imag).max())
        num = num.real

        return num, den


# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tfpolyToString(coeffs, var='s'):
    """Convert a transfer function polynomial to a string"""

    thestr = "0"

    # Compute the number of coefficients
    N = len(coeffs)-1

    for k in range(len(coeffs)):
        coefstr = '%.4g' % abs(coeffs[k])
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


def _convertToTransferFunction(sys, **kw):
    """Convert a system to transfer function form (if needed).

    If sys is already a transfer function, then it is returned.  If sys is a
    state space object, then it is converted to a transfer function and
    returned.  If sys is a scalar, then the number of inputs and outputs can be
    specified manually, as in:

    >>> sys = _convertToTransferFunction(3.) # Assumes inputs = outputs = 1
    >>> sys = _convertToTransferFunction(1., inputs=3, outputs=2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].

    If sys is an array-like type, then it is converted to a constant-gain
    transfer function.

    >>> sys = _convertToTransferFunction([[1. 0.], [2. 3.]])

    In this example, the numerator matrix will be
       [[[1.0], [0.0]], [[2.0], [3.0]]]
    and the denominator matrix [[[1.0], [1.0]], [[1.0], [1.0]]]

    """
    from .statesp import StateSpace

    if isinstance(sys, TransferFunction):
        if len(kw):
            raise TypeError("If sys is a TransferFunction, " +
                            "_convertToTransferFunction cannot take keywords.")

        return sys
    elif isinstance(sys, StateSpace):
        try:
            from slycot import tb04ad
            if len(kw):
                raise TypeError(
                    "If sys is a StateSpace, " +
                    "_convertToTransferFunction cannot take keywords.")

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
                    # Each transfer function matrix row
                    # has a common denominator.
                    den[i][j] = list(tfout[5][i, :])
            # print(num)
            # print(den)
        except ImportError:
            # If slycot is not available, use signal.lti (SISO only)
            if (sys.inputs != 1 or sys.outputs != 1):
                raise TypeError("No support for MIMO without slycot")

            lti_sys = lti(sys.A, sys.B, sys.C, sys.D)
            num = squeeze(lti_sys.num)
            den = squeeze(lti_sys.den)
            # print(num)
            # print(den)

        return TransferFunction(num, den, sys.dt)

    elif isinstance(sys, (int, float, complex)):
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

        return TransferFunction(num, den)

    # If this is array-like, try to create a constant feedthrough
    try:
        D = array(sys)
        outputs, inputs = D.shape
        num = [[[D[i, j]] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        return TransferFunction(num, den)
    except Exception as e:
        print("Failure to assume argument is matrix-like in"
              " _convertToTransferFunction, result %s" % e)

    raise TypeError("Can't convert given type to TransferFunction system.")
