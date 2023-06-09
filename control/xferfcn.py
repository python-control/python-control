"""xferfcn.py

Transfer function representation and functions.

This file contains the TransferFunction class and also functions
that operate on transfer functions.  This is the primary representation
for the python-control library.
"""

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

# External function declarations
import numpy as np
from numpy import angle, array, empty, finfo, ndarray, ones, \
    polyadd, polymul, polyval, roots, sqrt, zeros, squeeze, exp, pi, \
    where, delete, real, poly, nonzero
import scipy as sp
from scipy.signal import tf2zpk, zpk2tf, cont2discrete
from scipy.signal import TransferFunction as signalTransferFunction
from copy import deepcopy
from warnings import warn
from itertools import chain
from re import sub
from .lti import LTI, _process_frequency_response
from .namedio import common_timebase, isdtime, _process_namedio_keywords
from .exception import ControlMIMONotImplemented
from .frdata import FrequencyResponseData
from . import config

__all__ = ['TransferFunction', 'tf', 'zpk', 'ss2tf', 'tfdata']


# Define module default parameter values
_xferfcn_defaults = {
    'xferfcn.display_format': 'poly',
    'xferfcn.floating_point_format': '.4g'
}


def _float2str(value):
    _num_format = config.defaults.get('xferfcn.floating_point_format', ':.4g')
    return f"{value:{_num_format}}"


class TransferFunction(LTI):
    """TransferFunction(num, den[, dt])

    A class for representing transfer functions.

    The TransferFunction class is used to represent systems in transfer
    function form.

    Parameters
    ----------
    num : array_like, or list of list of array_like
        Polynomial coefficients of the numerator
    den : array_like, or list of list of array_like
        Polynomial coefficients of the denominator
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous
        time, True indicates discrete time with unspecified sampling
        time, positive number is discrete time with specified
        sampling time, None indicates unspecified timebase (either
        continuous or discrete time).
    display_format: None, 'poly' or 'zpk'
        Set the display format used in printing the TransferFunction object.
        Default behavior is polynomial display and can be changed by
        changing config.defaults['xferfcn.display_format'].

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables.
    num, den : 2D list of array
        Polynomial coefficients of the numerator and denominator.
    dt : None, True or float
        System timebase. 0 (default) indicates continuous time, True indicates
        discrete time with unspecified sampling time, positive number is
        discrete time with specified sampling time, None indicates unspecified
        timebase (either continuous or discrete time).

    Notes
    -----
    The attribues 'num' and 'den' are 2-D lists of arrays containing MIMO
    numerator and denominator coefficients.  For example,

    >>> num[2][5] = numpy.array([1., 4., 8.])                   # doctest: +SKIP

    means that the numerator of the transfer function from the 6th input to
    the 3rd output is set to s^2 + 4s + 8.

    A discrete time transfer function is created by specifying a nonzero
    'timebase' dt when the system is constructed:

    * dt = 0: continuous time system (default)
    * dt > 0: discrete time system with sampling period 'dt'
    * dt = True: discrete time with unspecified sampling period
    * dt = None: no timebase specified

    Systems must have compatible timebases in order to be combined. A discrete
    time system with unspecified sampling time (`dt = True`) can be combined
    with a system having a specified sampling time; the result will be a
    discrete time system with the sample time of the latter system. Similarly,
    a system with timebase `None` can be combined with a system having any
    timebase; the result will have the timebase of the latter system.
    The default value of dt can be changed by changing the value of
    ``control.config.defaults['control.default_dt']``.

    A transfer function is callable and returns the value of the transfer
    function evaluated at a point in the complex plane.  See
    :meth:`~control.TransferFunction.__call__` for a more detailed description.

    The TransferFunction class defines two constants ``s`` and ``z`` that
    represent the differentiation and delay operators in continuous and
    discrete time.  These can be used to create variables that allow algebraic
    creation of transfer functions.  For example,

    >>> s = ct.TransferFunction.s
    >>> G = (s + 1)/(s**2 + 2*s + 1)

    """

    # Give TransferFunction._rmul_() priority for ndarray * TransferFunction
    __array_priority__ = 11     # override ndarray and matrix types

    def __init__(self, *args, **kwargs):
        """TransferFunction(num, den[, dt])

        Construct a transfer function.

        The default constructor is TransferFunction(num, den), where num and
        den are lists of lists of arrays containing polynomial coefficients.
        To create a discrete time transfer funtion, use TransferFunction(num,
        den, dt) where 'dt' is the sampling time (or True for unspecified
        sampling time).  To call the copy constructor, call
        TransferFunction(sys), where sys is a TransferFunction object
        (continuous or discrete).

        """
        #
        # Process positional arguments
        #
        if len(args) == 2:
            # The user provided a numerator and a denominator.
            num, den = args

        elif len(args) == 3:
            # Discrete time transfer function
            num, den, dt = args
            if 'dt' in kwargs:
                warn("received multiple dt arguments, "
                     "using positional arg dt = %s" % dt)
            kwargs['dt'] = dt
            args = args[:-1]

        elif len(args) == 1:
            # Use the copy constructor.
            if not isinstance(args[0], TransferFunction):
                raise TypeError("The one-argument constructor can only take \
                        in a TransferFunction object.  Received %s."
                                % type(args[0]))
            num = args[0].num
            den = args[0].den

        else:
            raise TypeError("Needs 1, 2 or 3 arguments; received %i."
                             % len(args))

        num = _clean_part(num)
        den = _clean_part(den)

        #
        # Process keyword arguments
        #
        # During module init, TransferFunction.s and TransferFunction.z
        # get initialized when defaults are not fully initialized yet.
        # Use 'poly' in these cases.

        self.display_format = kwargs.pop(
            'display_format',
            config.defaults.get('xferfcn.display_format', 'poly'))

        if self.display_format not in ('poly', 'zpk'):
            raise ValueError("display_format must be 'poly' or 'zpk',"
                             " got '%s'" % self.display_format)

        # Determine if the transfer function is static (needed for dt)
        static = True
        for col in num + den:
            for poly in col:
                if len(poly) > 1:
                    static = False

        defaults = args[0] if len(args) == 1 else \
            {'inputs': len(num[0]), 'outputs': len(num)}

        name, inputs, outputs, states, dt = _process_namedio_keywords(
                kwargs, defaults, static=static, end=True)
        if states:
            raise TypeError(
                "states keyword not allowed for transfer functions")

        # Initialize LTI (NamedIOSystem) object
        super().__init__(
            name=name, inputs=inputs, outputs=outputs, dt=dt)

        #
        # Check to make sure everything is consistent
        #
        # Make sure numerator and denominator matrices have consistent sizes
        if self.ninputs != len(den[0]):
            raise ValueError(
                "The numerator has %i input(s), but the denominator has "
                "%i input(s)." % (self.ninputs, len(den[0])))
        if self.noutputs != len(den):
            raise ValueError(
                "The numerator has %i output(s), but the denominator has "
                "%i output(s)." % (self.noutputs, len(den)))

        # Additional checks/updates on structure of the transfer function
        for i in range(self.noutputs):
            # Make sure that each row has the same number of columns
            if len(num[i]) != self.ninputs:
                raise ValueError(
                    "Row 0 of the numerator matrix has %i elements, but row "
                    "%i has %i." % (self.ninputs, i, len(num[i])))
            if len(den[i]) != self.ninputs:
                raise ValueError(
                    "Row 0 of the denominator matrix has %i elements, but row "
                    "%i has %i." % (self.ninputs, i, len(den[i])))

            # Check for zeros in numerator or denominator
            # TODO: Right now these checks are only done during construction.
            # It might be worthwhile to think of a way to perform checks if the
            # user modifies the transfer function after construction.
            for j in range(self.ninputs):
                # Check that we don't have any zero denominators.
                zeroden = True
                for k in den[i][j]:
                    if k:
                        zeroden = False
                        break
                if zeroden:
                    raise ValueError(
                        "Input %i, output %i has a zero denominator."
                        % (j + 1, i + 1))

                # If we have zero numerators, set the denominator to 1.
                zeronum = True
                for k in num[i][j]:
                    if k:
                        zeronum = False
                        break
                if zeronum:
                    den[i][j] = ones(1)

        # Store the numerator and denominator
        self.num = num
        self.den = den

        #
        # Final processing
        #
        # Truncate leading zeros
        self._truncatecoeff()

    #
    # Class attributes
    #
    # These attributes are defined as class attributes so that they are
    # documented properly.  They are "overwritten" in __init__.
    #

    #: Number of system inputs.
    #:
    #: :meta hide-value:
    ninputs = 1

    #: Number of system outputs.
    #:
    #: :meta hide-value:
    noutputs = 1

    #: Transfer function numerator polynomial (array)
    #:
    #: The numerator of the transfer function is stored as an 2D list of
    #: arrays containing MIMO numerator coefficients, indexed by outputs and
    #: inputs.  For example, ``num[2][5]`` is the array of coefficients for
    #: the numerator of the transfer function from the sixth input to the
    #: third output.
    #:
    #: :meta hide-value:
    num = [[0]]

    #: Transfer function denominator polynomial (array)
    #:
    #: The numerator of the transfer function is store as an 2D list of
    #: arrays containing MIMO numerator coefficients, indexed by outputs and
    #: inputs.  For example, ``den[2][5]`` is the array of coefficients for
    #: the denominator of the transfer function from the sixth input to the
    #: third output.
    #:
    #: :meta hide-value:
    den = [[0]]

    def __call__(self, x, squeeze=None, warn_infinite=True):
        """Evaluate system's transfer function at complex frequencies.

        Returns the complex frequency response `sys(x)` where `x` is `s` for
        continuous-time systems and `z` for discrete-time systems.

        In general the system may be multiple input, multiple output
        (MIMO), where `m = self.ninputs` number of inputs and `p =
        self.noutputs` number of outputs.

        To evaluate at a frequency omega in radians per second, enter
        ``x = omega * 1j``, for continuous-time systems, or
        ``x = exp(1j * omega * dt)`` for discrete-time systems. Or use
        :meth:`TransferFunction.frequency_response`.

        Parameters
        ----------
        x : complex or complex 1D array_like
            Complex frequencies
        squeeze : bool, optional
            If squeeze=True, remove single-dimensional entries from the shape
            of the output even if the system is not SISO. If squeeze=False,
            keep all indices (output, input and, if omega is array_like,
            frequency) even if the system is SISO. The default value can be
            set using config.defaults['control.squeeze_frequency_response'].
            If True and the system is single-input single-output (SISO),
            return a 1D array rather than a 3D array.  Default value (True)
            set by config.defaults['control.squeeze_frequency_response'].
        warn_infinite : bool, optional
            If set to `False`, turn off divide by zero warning.

        Returns
        -------
        fresp : complex ndarray
            The frequency response of the system.  If the system is SISO and
            squeeze is not True, the shape of the array matches the shape of
            omega.  If the system is not SISO or squeeze is False, the first
            two dimensions of the array are indices for the output and input
            and the remaining dimensions match omega.  If ``squeeze`` is True
            then single-dimensional axes are removed.

        """
        out = self.horner(x, warn_infinite=warn_infinite)
        return _process_frequency_response(self, x, out, squeeze=squeeze)

    def horner(self, x, warn_infinite=True):
        """Evaluate system's transfer function at complex frequency
        using Horner's method.

        Evaluates `sys(x)` where `x` is `s` for continuous-time systems and `z`
        for discrete-time systems.

        Expects inputs and outputs to be formatted correctly. Use ``sys(x)``
        for a more user-friendly interface.

        Parameters
        ----------
        x : complex array_like or complex scalar
            Complex frequencies

        Returns
        -------
        output : (self.noutputs, self.ninputs, len(x)) complex ndarray
            Frequency response

        """
        # Make sure the argument is a 1D array of complex numbers
        x_arr = np.atleast_1d(x).astype(complex, copy=False)

        # Make sure that we are operating on a simple list
        if len(x_arr.shape) > 1:
            raise ValueError("input list must be 1D")

        # Initialize the output matrix in the proper shape
        out = empty((self.noutputs, self.ninputs, len(x_arr)), dtype=complex)

        # Set up error processing based on warn_infinite flag
        with np.errstate(all='warn' if warn_infinite else 'ignore'):
            for i in range(self.noutputs):
                for j in range(self.ninputs):
                    out[i][j] = (polyval(self.num[i][j], x_arr) /
                                 polyval(self.den[i][j], x_arr))
        return out

    def _truncatecoeff(self):
        """Remove extraneous zero coefficients from num and den.

        Check every element of the numerator and denominator matrices, and
        truncate leading zeros.  For instance, running self._truncatecoeff()
        will reduce self.num = [[[0, 0, 1, 2]]] to [[[1, 2]]].

        """

        # Beware: this is a shallow copy.  This should be okay.
        data = [self.num, self.den]
        for p in range(len(data)):
            for i in range(self.noutputs):
                for j in range(self.ninputs):
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

    def __str__(self, var=None):
        """String representation of the transfer function.

        Based on the display_format property, the output will be formatted as
        either polynomials or in zpk form.
        """
        mimo = not self.issiso()
        if var is None:
            var = 's' if self.isctime() else 'z'
        outstr = ""

        for ni in range(self.ninputs):
            for no in range(self.noutputs):
                if mimo:
                    outstr += "\nInput %i to output %i:" % (ni + 1, no + 1)

                # Convert the numerator and denominator polynomials to strings.
                if self.display_format == 'poly':
                    numstr = _tf_polynomial_to_string(self.num[no][ni], var=var)
                    denstr = _tf_polynomial_to_string(self.den[no][ni], var=var)
                elif self.display_format == 'zpk':
                    z, p, k = tf2zpk(self.num[no][ni], self.den[no][ni])
                    numstr = _tf_factorized_polynomial_to_string(
                        z, gain=k, var=var)
                    denstr = _tf_factorized_polynomial_to_string(p, var=var)

                # Figure out the length of the separating line
                dashcount = max(len(numstr), len(denstr))
                dashes = '-' * dashcount

                # Center the numerator or denominator
                if len(numstr) < dashcount:
                    numstr = ' ' * ((dashcount - len(numstr)) // 2) + numstr
                if len(denstr) < dashcount:
                    denstr = ' ' * ((dashcount - len(denstr)) // 2) + denstr

                outstr += "\n" + numstr + "\n" + dashes + "\n" + denstr + "\n"

        # If this is a strict discrete time system, print the sampling time
        if type(self.dt) != bool and self.isdtime(strict=True):
            outstr += "\ndt = " + str(self.dt) + "\n"

        return outstr

    # represent to implement a re-loadable version
    def __repr__(self):
        """Print transfer function in loadable form"""
        if self.issiso():
            return "TransferFunction({num}, {den}{dt})".format(
                num=self.num[0][0].__repr__(), den=self.den[0][0].__repr__(),
                dt=', {}'.format(self.dt) if isdtime(self, strict=True)
                else '')
        else:
            return "TransferFunction({num}, {den}{dt})".format(
                num=self.num.__repr__(), den=self.den.__repr__(),
                dt=', {}'.format(self.dt) if isdtime(self, strict=True)
                else '')

    def _repr_latex_(self, var=None):
        """LaTeX representation of transfer function, for Jupyter notebook"""

        mimo = not self.issiso()

        if var is None:
            # ! TODO: replace with standard calls to lti functions
            var = 's' if self.dt is None or self.dt == 0 else 'z'

        out = ['$$']

        if mimo:
            out.append(r"\begin{bmatrix}")

        for no in range(self.noutputs):
            for ni in range(self.ninputs):
                # Convert the numerator and denominator polynomials to strings.
                if self.display_format == 'poly':
                    numstr = _tf_polynomial_to_string(self.num[no][ni], var=var)
                    denstr = _tf_polynomial_to_string(self.den[no][ni], var=var)
                elif self.display_format == 'zpk':
                    z, p, k = tf2zpk(self.num[no][ni], self.den[no][ni])
                    numstr = _tf_factorized_polynomial_to_string(
                        z, gain=k, var=var)
                    denstr = _tf_factorized_polynomial_to_string(p, var=var)

                numstr = _tf_string_to_latex(numstr, var=var)
                denstr = _tf_string_to_latex(denstr, var=var)

                out += [r"\frac{", numstr, "}{", denstr, "}"]

                if mimo and ni < self.ninputs - 1:
                    out.append("&")

            if mimo:
                out.append(r"\\")

        if mimo:
            out.append(r" \end{bmatrix}")

        # See if this is a discrete time system with specific sampling time
        if not (self.dt is None) and type(self.dt) != bool and self.dt > 0:
            out += [r"\quad dt = ", str(self.dt)]

        out.append("$$")

        return ''.join(out)

    def __neg__(self):
        """Negate a transfer function."""

        num = deepcopy(self.num)
        for i in range(self.noutputs):
            for j in range(self.ninputs):
                num[i][j] *= -1

        return TransferFunction(num, self.den, self.dt)

    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""
        from .statesp import StateSpace

        # Check to see if the right operator has priority
        if getattr(other, '__array_priority__', None) and \
           getattr(self, '__array_priority__', None) and \
           other.__array_priority__ > self.__array_priority__:
            return other.__radd__(self)

        # Convert the second argument to a transfer function.
        if isinstance(other, StateSpace):
            other = _convert_to_transfer_function(other)
        elif not isinstance(other, TransferFunction):
            other = _convert_to_transfer_function(other, inputs=self.ninputs,
                                                  outputs=self.noutputs)

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.ninputs:
            raise ValueError(
                "The first summand has %i input(s), but the second has %i."
                % (self.ninputs, other.ninputs))
        if self.noutputs != other.noutputs:
            raise ValueError(
                "The first summand has %i output(s), but the second has %i."
                % (self.noutputs, other.noutputs))

        dt = common_timebase(self.dt, other.dt)

        # Preallocate the numerator and denominator of the sum.
        num = [[[] for j in range(self.ninputs)] for i in range(self.noutputs)]
        den = [[[] for j in range(self.ninputs)] for i in range(self.noutputs)]

        for i in range(self.noutputs):
            for j in range(self.ninputs):
                num[i][j], den[i][j] = _add_siso(
                    self.num[i][j], self.den[i][j],
                    other.num[i][j], other.den[i][j])

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
        # Check to see if the right operator has priority
        if getattr(other, '__array_priority__', None) and \
           getattr(self, '__array_priority__', None) and \
           other.__array_priority__ > self.__array_priority__:
            return other.__rmul__(self)

        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, complex, np.number)):
            other = _convert_to_transfer_function(other, inputs=self.ninputs,
                                                  outputs=self.ninputs)
        else:
            other = _convert_to_transfer_function(other)

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.noutputs:
            raise ValueError(
                "C = A * B: A has %i column(s) (input(s)), but B has %i "
                "row(s)\n(output(s))." % (self.ninputs, other.noutputs))

        inputs = other.ninputs
        outputs = self.noutputs

        dt = common_timebase(self.dt, other.dt)

        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]

        # Temporary storage for the summands needed to find the (i, j)th
        # element of the product.
        num_summand = [[] for k in range(self.ninputs)]
        den_summand = [[] for k in range(self.ninputs)]

        # Multiply & add.
        for row in range(outputs):
            for col in range(inputs):
                for k in range(self.ninputs):
                    num_summand[k] = polymul(
                        self.num[row][k], other.num[k][col])
                    den_summand[k] = polymul(
                        self.den[row][k], other.den[k][col])
                    num[row][col], den[row][col] = _add_siso(
                        num[row][col], den[row][col],
                        num_summand[k], den_summand[k])

        return TransferFunction(num, den, dt)

    def __rmul__(self, other):
        """Right multiply two LTI objects (serial connection)."""

        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, complex, np.number)):
            other = _convert_to_transfer_function(other, inputs=self.ninputs,
                                                  outputs=self.ninputs)
        else:
            other = _convert_to_transfer_function(other)

        # Check that the input-output sizes are consistent.
        if other.ninputs != self.noutputs:
            raise ValueError(
                "C = A * B: A has %i column(s) (input(s)), but B has %i "
                "row(s)\n(output(s))." % (other.ninputs, self.noutputs))

        inputs = self.ninputs
        outputs = other.noutputs

        dt = common_timebase(self.dt, other.dt)

        # Preallocate the numerator and denominator of the sum.
        num = [[[0] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]

        # Temporary storage for the summands needed to find the
        # (i, j)th element
        # of the product.
        num_summand = [[] for k in range(other.ninputs)]
        den_summand = [[] for k in range(other.ninputs)]

        for i in range(outputs):  # Iterate through rows of product.
            for j in range(inputs):  # Iterate through columns of product.
                for k in range(other.ninputs):  # Multiply & add.
                    num_summand[k] = polymul(other.num[i][k], self.num[k][j])
                    den_summand[k] = polymul(other.den[i][k], self.den[k][j])
                    num[i][j], den[i][j] = _add_siso(
                        num[i][j], den[i][j],
                        num_summand[k], den_summand[k])

        return TransferFunction(num, den, dt)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""

        if isinstance(other, (int, float, complex, np.number)):
            other = _convert_to_transfer_function(
                other, inputs=self.ninputs,
                outputs=self.ninputs)
        else:
            other = _convert_to_transfer_function(other)

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            raise NotImplementedError(
                "TransferFunction.__truediv__ is currently \
                implemented only for SISO systems.")

        dt = common_timebase(self.dt, other.dt)

        num = polymul(self.num[0][0], other.den[0][0])
        den = polymul(self.den[0][0], other.num[0][0])

        return TransferFunction(num, den, dt)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rtruediv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, complex, np.number)):
            other = _convert_to_transfer_function(
                other, inputs=self.ninputs,
                outputs=self.ninputs)
        else:
            other = _convert_to_transfer_function(other)

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            raise NotImplementedError(
                "TransferFunction.__rtruediv__ is currently implemented only "
                "for SISO systems.")

        return other / self

    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return TransferFunction([1], [1])  # unity
        if other > 0:
            return self * (self**(other - 1))
        if other < 0:
            return (TransferFunction([1], [1]) / self) * (self**(other + 1))

    def __getitem__(self, key):
        key1, key2 = key

        # pre-process
        if isinstance(key1, int):
            key1 = slice(key1, key1 + 1, 1)
        if isinstance(key2, int):
            key2 = slice(key2, key2 + 1, 1)
        # dim1
        start1, stop1, step1 = key1.start, key1.stop, key1.step
        if step1 is None:
            step1 = 1
        if start1 is None:
            start1 = 0
        if stop1 is None:
            stop1 = len(self.num)
        # dim1
        start2, stop2, step2 = key2.start, key2.stop, key2.step
        if step2 is None:
            step2 = 1
        if start2 is None:
            start2 = 0
        if stop2 is None:
            stop2 = len(self.num[0])

        num = []
        den = []
        for i in range(start1, stop1, step1):
            num_i = []
            den_i = []
            for j in range(start2, stop2, step2):
                num_i.append(self.num[i][j])
                den_i.append(self.den[i][j])
            num.append(num_i)
            den.append(den_i)
        if self.isctime():
            return TransferFunction(num, den)
        else:
            return TransferFunction(num, den, self.dt)

    def freqresp(self, omega):
        """(deprecated) Evaluate transfer function at complex frequencies.

        .. deprecated::0.9.0
            Method has been given the more pythonic name
            :meth:`TransferFunction.frequency_response`. Or use
            :func:`freqresp` in the MATLAB compatibility module.
        """
        warn("TransferFunction.freqresp(omega) will be removed in a "
             "future release of python-control; use "
             "sys.frequency_response(omega), or freqresp(sys, omega) in the "
             "MATLAB compatibility module instead", DeprecationWarning)
        return self.frequency_response(omega)

    def poles(self):
        """Compute the poles of a transfer function."""
        _, den, denorder = self._common_den(allow_nonproper=True)
        rts = []
        for d, o in zip(den, denorder):
            rts.extend(roots(d[:o + 1]))
        return np.array(rts).astype(complex)

    def zeros(self):
        """Compute the zeros of a transfer function."""
        if self.ninputs > 1 or self.noutputs > 1:
            raise NotImplementedError(
                "TransferFunction.zeros is currently only implemented "
                "for SISO systems.")
        else:
            # for now, just give zeros of a SISO tf
            return roots(self.num[0][0]).astype(complex)

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two LTI objects."""
        other = _convert_to_transfer_function(other)

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            # TODO: MIMO feedback
            raise ControlMIMONotImplemented(
                "TransferFunction.feedback is currently not implemented for "
                "MIMO systems.")
        dt = common_timebase(self.dt, other.dt)

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
        num = [[[] for j in range(self.ninputs)] for i in range(self.noutputs)]
        den = [[[] for j in range(self.ninputs)] for i in range(self.noutputs)]

        for i in range(self.noutputs):
            for j in range(self.ninputs):

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

                # poly([]) returns a scalar, but we always want a 1d array
                num[i][j] = np.atleast_1d(gain * real(poly(newzeros)))
                den[i][j] = np.atleast_1d(real(poly(poles)))

        # end result
        return TransferFunction(num, den, self.dt)

    def returnScipySignalLTI(self, strict=True):
        """Return a list of a list of :class:`scipy.signal.lti` objects.

        For instance,

        >>> out = tfobject.returnScipySignalLTI()               # doctest: +SKIP
        >>> out[3][5]                                           # doctest: +SKIP

        is a :class:`scipy.signal.lti` object corresponding to the
        transfer function from the 6th input to the 4th output.

        Parameters
        ----------
        strict : bool, optional
            True (default):
                The timebase `tfobject.dt` cannot be None; it must be
                continuous (0) or discrete (True or > 0).
            False:
                if `tfobject.dt` is None, continuous time
                :class:`scipy.signal.lti` objects are returned

        Returns
        -------
        out : list of list of :class:`scipy.signal.TransferFunction`
            continuous time (inheriting from :class:`scipy.signal.lti`)
            or discrete time (inheriting from :class:`scipy.signal.dlti`)
            SISO objects
        """
        if strict and self.dt is None:
            raise ValueError("with strict=True, dt cannot be None")

        if self.dt:
            kwdt = {'dt': self.dt}
        else:
            # scipy convention for continuous time lti systems: call without
            # dt keyword argument
            kwdt = {}

        # Preallocate the output.
        out = [[[] for j in range(self.ninputs)] for i in range(self.noutputs)]

        for i in range(self.noutputs):
            for j in range(self.ninputs):
                out[i][j] = signalTransferFunction(self.num[i][j],
                                                   self.den[i][j],
                                                   **kwdt)

        return out

    def _common_den(self, imag_tol=None, allow_nonproper=False):
        """
        Compute MIMO common denominators; return them and adjusted numerators.

        This function computes the denominators per input containing all
        the poles of sys.den, and reports it as the array den.  The
        output numerator array num is modified to use the common
        denominator for this input/column; the coefficient arrays are also
        padded with zeros to be the same size for all num/den.

        Parameters
        ----------
        imag_tol: float
            Threshold for the imaginary part of a root to use in detecting
            complex poles

        allow_nonproper : boolean
            Do not enforce proper transfer functions

        Returns
        -------
        num: array
            n by n by kd where n = max(sys.noutputs,sys.ninputs)
                              kd = max(denorder)+1
            Multi-dimensional array of numerator coefficients. num[i,j]
            gives the numerator coefficient array for the ith output and jth
            input; padded for use in td04ad ('C' option); matches the
            denorder order; highest coefficient starts on the left.
            If allow_nonproper=True and the order of a numerator exceeds the
            order of the common denominator, num will be returned as None

        den: array
            sys.ninputs by kd
            Multi-dimensional array of coefficients for common denominator
            polynomial, one row per input. The array is prepared for use in
            slycot td04ad, the first element is the highest-order polynomial
            coefficient of s, matching the order in denorder. If denorder <
            number of columns in den, the den is padded with zeros.

        denorder: array of int, orders of den, one per input



        Examples
        --------
        >>> num, den, denorder = sys._common_den()              # doctest: +SKIP

        """

        # Machine precision for floats.
        eps = finfo(float).eps
        real_tol = sqrt(eps * self.ninputs * self.noutputs)

        # The tolerance to use in deciding if a pole is complex
        if (imag_tol is None):
            imag_tol = 2 * real_tol

        # A list to keep track of cumulative poles found as we scan
        # self.den[..][..]
        poles = [[] for j in range(self.ninputs)]

        # RvP, new implementation 180526, issue #194
        # BG, modification, issue #343, PR #354

        # pre-calculate the poles for all num, den
        # has zeros, poles, gain, list for pole indices not in den,
        # number of poles known at the time analyzed

        # do not calculate minreal. Rory's hint .minreal()
        poleset = []
        for i in range(self.noutputs):
            poleset.append([])
            for j in range(self.ninputs):
                if abs(self.num[i][j]).max() <= eps:
                    poleset[-1].append([array([], dtype=float),
                                        roots(self.den[i][j]), 0.0, [], 0])
                else:
                    z, p, k = tf2zpk(self.num[i][j], self.den[i][j])
                    poleset[-1].append([z, p, k, [], 0])

        # collect all individual poles
        for j in range(self.ninputs):
            for i in range(self.noutputs):
                currentpoles = poleset[i][j][1]
                nothave = ones(currentpoles.shape, dtype=bool)
                for ip, p in enumerate(poles[j]):
                    collect = (np.isclose(currentpoles.real, p.real,
                                          atol=real_tol) &
                               np.isclose(currentpoles.imag, p.imag,
                                          atol=imag_tol) &
                               nothave)
                    if np.any(collect):
                        # mark first found pole as already collected
                        nothave[nonzero(collect)[0][0]] = False
                    else:
                        # remember id of pole not in tf
                        poleset[i][j][3].append(ip)
                for h, c in zip(nothave, currentpoles):
                    if h:
                        if abs(c.imag) < imag_tol:
                            c = c.real
                        poles[j].append(c)
                # remember how many poles now known
                poleset[i][j][4] = len(poles[j])

        # figure out maximum number of poles, for sizing the den
        maxindex = max([len(p) for p in poles])
        den = zeros((self.ninputs, maxindex + 1), dtype=float)
        num = zeros((max(1, self.noutputs, self.ninputs),
                     max(1, self.noutputs, self.ninputs),
                     maxindex + 1),
                    dtype=float)
        denorder = zeros((self.ninputs,), dtype=int)

        havenonproper = False

        for j in range(self.ninputs):
            if not len(poles[j]):
                # no poles matching this input; only one or more gains
                den[j, 0] = 1.0
                for i in range(self.noutputs):
                    num[i, j, 0] = poleset[i][j][2]
            else:
                # create the denominator matching this input
                # coefficients should be padded on right, ending at maxindex
                maxindex = len(poles[j])
                den[j, :maxindex+1] = poly(poles[j])
                denorder[j] = maxindex

                # now create the numerator, also padded on the right
                for i in range(self.noutputs):
                    # start with the current set of zeros for this output
                    nwzeros = list(poleset[i][j][0])
                    # add all poles not found in the original denominator,
                    # and the ones later added from other denominators
                    for ip in chain(poleset[i][j][3],
                                    range(poleset[i][j][4], maxindex)):
                        nwzeros.append(poles[j][ip])

                    numpoly = poleset[i][j][2] * np.atleast_1d(poly(nwzeros))

                    # td04ad expects a proper transfer function. If the
                    # numerater has a higher order than the denominator, the
                    # padding will fail
                    if len(numpoly) > maxindex + 1:
                        if allow_nonproper:
                            havenonproper = True
                            break
                        raise ValueError(
                            self.__str__() +
                            "is not a proper transfer function. "
                            "The degree of the numerators must not exceed "
                            "the degree of the denominators.")

                    # numerator polynomial should be padded on left and right
                    #   ending at maxindex to line up with what td04ad expects.
                    num[i, j, maxindex+1-len(numpoly):maxindex+1] = numpoly
                    # print(num[i, j])

        if havenonproper:
            num = None

        return num, den, denorder

    def sample(self, Ts, method='zoh', alpha=None, prewarp_frequency=None,
               name=None, copy_names=True, **kwargs):
        """Convert a continuous-time system to discrete time

        Creates a discrete-time system from a continuous-time system by
        sampling.  Multiple methods of conversion are supported.

        Parameters
        ----------
        Ts : float
            Sampling period
        method : {"gbt", "bilinear", "euler", "backward_diff",
                  "zoh", "matched"}
            Method to use for sampling:

            * gbt: generalized bilinear transformation
            * bilinear or tustin: Tustin's approximation ("gbt" with alpha=0.5)
            * euler: Euler (or forward difference) method ("gbt" with alpha=0)
            * backward_diff: Backwards difference ("gbt" with alpha=1.0)
            * zoh: zero-order hold (default)
        alpha : float within [0, 1]
            The generalized bilinear transformation weighting parameter, which
            should only be specified with method="gbt", and is ignored
            otherwise. See :func:`scipy.signal.cont2discrete`.
        prewarp_frequency : float within [0, infinity)
            The frequency [rad/s] at which to match with the input continuous-
            time system's magnitude and phase (the gain=1 crossover frequency,
            for example). Should only be specified with method='bilinear' or
            'gbt' with alpha=0.5 and ignored otherwise.
        name : string, optional
            Set the name of the sampled system.  If not specified and
            if `copy_names` is `False`, a generic name <sys[id]> is generated
            with a unique integer id.  If `copy_names` is `True`, the new system
            name is determined by adding the prefix and suffix strings in
            config.defaults['namedio.sampled_system_name_prefix'] and
            config.defaults['namedio.sampled_system_name_suffix'], with the
            default being to add the suffix '$sampled'.
        copy_names : bool, Optional
            If True, copy the names of the input signals, output
            signals, and states to the sampled system.

        Returns
        -------
        sysd : TransferFunction system
            Discrete-time system, with sample period Ts

        Other Parameters
        ----------------
        inputs : int, list of str or None, optional
            Description of the system inputs.  If not specified, the origional
            system inputs are used.  See :class:`InputOutputSystem` for more
            information.
        outputs : int, list of str or None, optional
            Description of the system outputs.  Same format as `inputs`.

        Notes
        -----
        1. Available only for SISO systems

        2. Uses :func:`scipy.signal.cont2discrete`

        Examples
        --------
        >>> sys = ct.tf(1, [1, 1])
        >>> sysd = sys.sample(0.5, method='bilinear')

        """
        if not self.isctime():
            raise ValueError("System must be continuous time system")
        if not self.issiso():
            raise ControlMIMONotImplemented("Not implemented for MIMO systems")
        if method == "matched":
            return _c2d_matched(self, Ts)
        sys = (self.num[0][0], self.den[0][0])
        if prewarp_frequency is not None:
            if method in ('bilinear', 'tustin') or \
                    (method == 'gbt' and alpha == 0.5):
                Twarp = 2*np.tan(prewarp_frequency*Ts/2)/prewarp_frequency
            else:
                warn('prewarp_frequency ignored: incompatible conversion')
                Twarp = Ts
        else:
            Twarp = Ts
        numd, dend, _ = cont2discrete(sys, Twarp, method, alpha)

        sysd = TransferFunction(numd[0, :], dend, Ts)
        # copy over the system name, inputs, outputs, and states
        if copy_names:
            sysd._copy_names(self, prefix_suffix_name='sampled')
            if name is not None:
                sysd.name = name
        # pass desired signal names if names were provided
        return TransferFunction(sysd, name=name, **kwargs)

    def dcgain(self, warn_infinite=False):
        """Return the zero-frequency (or DC) gain

        For a continous-time transfer function G(s), the DC gain is G(0)
        For a discrete-time transfer function G(z), the DC gain is G(1)

        Parameters
        ----------
        warn_infinite : bool, optional
            By default, don't issue a warning message if the zero-frequency
            gain is infinite.  Setting `warn_infinite` to generate the warning
            message.

        Returns
        -------
        gain : (noutputs, ninputs) ndarray or scalar
            Array or scalar value for SISO systems, depending on
            config.defaults['control.squeeze_frequency_response'].
            The value of the array elements or the scalar is either the
            zero-frequency (or DC) gain, or `inf`, if the frequency response
            is singular.

            For real valued systems, the empty imaginary part of the
            complex zero-frequency response is discarded and a real array or
            scalar is returned.

        Examples
        --------
        >>> G = ct.tf([1], [1, 4])
        >>> G.dcgain()
        0.25

        """
        return self._dcgain(warn_infinite)

    def _isstatic(self):
        """returns True if and only if all of the numerator and denominator
        polynomials of the (possibly MIMO) transfer function are zeroth order,
        that is, if the system has no dynamics. """
        for list_of_polys in self.num, self.den:
            for row in list_of_polys:
                for poly in row:
                    if len(poly) > 1:
                        return False
        return True

    # Attributes for differentiation and delay
    #
    # These attributes are created here with sphinx docstrings so that the
    # autodoc generated documentation has a description.  The actual values of
    # the class attributes are set at the bottom of the file to avoid problems
    # with recursive calls.

    #: Differentation operator (continuous time)
    #:
    #: The ``s`` constant can be used to create continuous time transfer
    #: functions using algebraic expressions.
    #:
    #: Example
    #: -------
    #: >>> s = TransferFunction.s                               # doctest: +SKIP
    #: >>> G  = (s + 1)/(s**2 + 2*s + 1)                        # doctest: +SKIP
    #:
    #: :meta hide-value:
    s = None

    #: Delay operator (discrete time)
    #:
    #: The ``z`` constant can be used to create discrete time transfer
    #: functions using algebraic expressions.
    #:
    #: Example
    #: -------
    #: >>> z = TransferFunction.z                               # doctest: +SKIP
    #: >>> G  = 2 * z / (4 * z**3 + 3*z - 1)                    # doctest: +SKIP
    #:
    #: :meta hide-value:
    z = None


# c2d function contributed by Benjamin White, Oct 2012
def _c2d_matched(sysC, Ts):
    # Pole-zero match method of continuous to discrete time conversion
    szeros, spoles, sgain = tf2zpk(sysC.num[0][0], sysC.den[0][0])
    zzeros = [0] * len(szeros)
    zpoles = [0] * len(spoles)
    pregainnum = [0] * len(szeros)
    pregainden = [0] * len(spoles)
    for idx, s in enumerate(szeros):
        sTs = s * Ts
        z = exp(sTs)
        zzeros[idx] = z
        pregainnum[idx] = 1 - z
    for idx, s in enumerate(spoles):
        sTs = s * Ts
        z = exp(sTs)
        zpoles[idx] = z
        pregainden[idx] = 1 - z
    zgain = np.multiply.reduce(pregainnum) / np.multiply.reduce(pregainden)
    gain = sgain / zgain
    sysDnum, sysDden = zpk2tf(zzeros, zpoles, gain)
    return TransferFunction(sysDnum, sysDden, Ts)


# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tf_polynomial_to_string(coeffs, var='s'):
    """Convert a transfer function polynomial to a string"""

    thestr = "0"

    # Compute the number of coefficients
    N = len(coeffs) - 1

    for k in range(len(coeffs)):
        coefstr = _float2str(abs(coeffs[k]))
        power = (N - k)
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


def _tf_factorized_polynomial_to_string(roots, gain=1, var='s'):
    """Convert a factorized polynomial to a string"""

    if roots.size == 0:
        return _float2str(gain)

    factors = []
    for root in sorted(roots, reverse=True):
        if np.isreal(root):
            if root == 0:
                factor = f"{var}"
                factors.append(factor)
            elif root > 0:
                factor = f"{var} - {_float2str(np.abs(root))}"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}"
                factors.append(factor)
        elif np.isreal(root * 1j):
            if root.imag > 0:
                factor = f"{var} - {_float2str(np.abs(root))}j"
                factors.append(factor)
            else:
                factor = f"{var} + {_float2str(np.abs(root))}j"
                factors.append(factor)
        else:
            if root.real > 0:
                factor = f"{var} - ({_float2str(root)})"
                factors.append(factor)
            else:
                factor = f"{var} + ({_float2str(-root)})"
                factors.append(factor)

    multiplier = ''
    if round(gain, 4) != 1.0:
        multiplier = _float2str(gain) + " "

    if len(factors) > 1 or multiplier:
        factors = [f"({factor})" for factor in factors]

    return multiplier + " ".join(factors)


def _tf_string_to_latex(thestr, var='s'):
    """ make sure to superscript all digits in a polynomial string
        and convert float coefficients in scientific notation
        to prettier LaTeX representation """
    # TODO: make the multiplication sign configurable
    expmul = r' \\times'
    thestr = sub(var + r'\^(\d{2,})', var + r'^{\1}', thestr)
    thestr = sub(r'[eE]\+0*(\d+)', expmul + r' 10^{\1}', thestr)
    thestr = sub(r'[eE]\-0*(\d+)', expmul + r' 10^{-\1}', thestr)
    return thestr


def _add_siso(num1, den1, num2, den2):
    """Return num/den = num1/den1 + num2/den2.

    Each numerator and denominator is a list of polynomial coefficients.

    """

    num = polyadd(polymul(num1, den2), polymul(num2, den1))
    den = polymul(den1, den2)

    return num, den


def _convert_to_transfer_function(
        sys, inputs=1, outputs=1, use_prefix_suffix=False):
    """Convert a system to transfer function form (if needed).

    If sys is already a transfer function, then it is returned.  If sys is a
    state space object, then it is converted to a transfer function and
    returned.  If sys is a scalar, then the number of inputs and outputs can be
    specified manually, as in:

    >>> sys = _convert_to_transfer_function(3.) # Assumes inputs = outputs = 1
    >>> sys = _convert_to_transfer_function(1., inputs=3, outputs=2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].

    If sys is an array-like type, then it is converted to a constant-gain
    transfer function.

    Note: no renaming of inputs and outputs is performed; this should be done
    by the calling function.

    >>> sys = _convert_to_transfer_function([[1., 0.], [2., 3.]])

    In this example, the numerator matrix will be
       [[[1.0], [0.0]], [[2.0], [3.0]]]
    and the denominator matrix [[[1.0], [1.0]], [[1.0], [1.0]]]

    """
    from .statesp import StateSpace
    kwargs = {}

    if isinstance(sys, TransferFunction):
        return sys

    elif isinstance(sys, StateSpace):
        if 0 == sys.nstates:
            # Slycot doesn't like static SS->TF conversion, so handle
            # it first.  Can't join this with the no-Slycot branch,
            # since that doesn't handle general MIMO systems
            num = [[[sys.D[i, j]] for j in range(sys.ninputs)]
                   for i in range(sys.noutputs)]
            den = [[[1.] for j in range(sys.ninputs)]
                   for i in range(sys.noutputs)]
        else:
            try:
                # Use Slycot to make the transformation
                # Make sure to convert system matrices to numpy arrays
                from slycot import tb04ad
                tfout = tb04ad(
                    sys.nstates, sys.ninputs, sys.noutputs, array(sys.A),
                    array(sys.B), array(sys.C), array(sys.D), tol1=0.0)

                # Preallocate outputs.
                num = [[[] for j in range(sys.ninputs)]
                       for i in range(sys.noutputs)]
                den = [[[] for j in range(sys.ninputs)]
                       for i in range(sys.noutputs)]

                for i in range(sys.noutputs):
                    for j in range(sys.ninputs):
                        num[i][j] = list(tfout[6][i, j, :])
                        # Each transfer function matrix row
                        # has a common denominator.
                        den[i][j] = list(tfout[5][i, :])

            except ImportError:
                # If slycot is not available, use signal.lti (SISO only)
                if sys.ninputs != 1 or sys.noutputs != 1:
                    raise ControlMIMONotImplemented("Not implemented for " +
                        "MIMO systems without slycot.")

                # Do the conversion using sp.signal.ss2tf
                # Note that this returns a 2D array for the numerator
                num, den = sp.signal.ss2tf(sys.A, sys.B, sys.C, sys.D)
                num = squeeze(num)  # Convert to 1D array
                den = squeeze(den)  # Probably not needed

        newsys = TransferFunction(num, den, sys.dt)
        if use_prefix_suffix:
            newsys._copy_names(sys, prefix_suffix_name='converted')
        return newsys

    elif isinstance(sys, (int, float, complex, np.number)):
        num = [[[sys] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]

        return TransferFunction(num, den)

    elif isinstance(sys, FrequencyResponseData):
        raise TypeError("Can't convert given FRD to TransferFunction system.")

    # If this is array-like, try to create a constant feedthrough
    try:
        D = array(sys, ndmin=2)
        outputs, inputs = D.shape
        num = [[[D[i, j]] for j in range(inputs)] for i in range(outputs)]
        den = [[[1] for j in range(inputs)] for i in range(outputs)]
        return TransferFunction(num, den)

    except Exception:
        raise TypeError("Can't convert given type to TransferFunction system.")


def tf(*args, **kwargs):
    """tf(num, den[, dt])

    Create a transfer function system. Can create MIMO systems.

    The function accepts either 1, 2, or 3 parameters:

    ``tf(sys)``
        Convert a linear system into transfer function form. Always creates
        a new system, even if sys is already a TransferFunction object.

    ``tf(num, den)``
        Create a transfer function system from its numerator and denominator
        polynomial coefficients.

        If `num` and `den` are 1D array_like objects, the function creates a
        SISO system.

        To create a MIMO system, `num` and `den` need to be 2D nested lists
        of array_like objects. (A 3 dimensional data structure in total.)
        (For details see note below.)

    ``tf(num, den, dt)``
        Create a discrete time transfer function system; dt can either be a
        positive number indicating the sampling time or 'True' if no
        specific timebase is given.

    ``tf('s')`` or ``tf('z')``
        Create a transfer function representing the differential operator
        ('s') or delay operator ('z').

    Parameters
    ----------
    sys: LTI (StateSpace or TransferFunction)
        A linear system
    num: array_like, or list of list of array_like
        Polynomial coefficients of the numerator
    den: array_like, or list of list of array_like
        Polynomial coefficients of the denominator
    display_format: None, 'poly' or 'zpk'
        Set the display format used in printing the TransferFunction object.
        Default behavior is polynomial display and can be changed by
        changing config.defaults['xferfcn.display_format']..

    Returns
    -------
    out: :class:`TransferFunction`
        The new linear system

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    name : string, optional
        System name. If unspecified, a generic name <sys[id]> is generated
        with a unique integer id.

    Raises
    ------
    ValueError
        if `num` and `den` have invalid or unequal dimensions
    TypeError
        if `num` or `den` are of incorrect type

    See Also
    --------
    TransferFunction
    ss
    ss2tf
    tf2ss

    Notes
    -----
    ``num[i][j]`` contains the polynomial coefficients of the numerator
    for the transfer function from the (j+1)st input to the (i+1)st output.
    ``den[i][j]`` works the same way.

    The list ``[2, 3, 4]`` denotes the polynomial :math:`2s^2 + 3s + 4`.

    The special forms ``tf('s')`` and ``tf('z')`` can be used to create
    transfer functions for differentiation and unit delays.

    Examples
    --------
    >>> # Create a MIMO transfer function object
    >>> # The transfer function from the 2nd input to the 1st output is
    >>> # (3s + 4) / (6s^2 + 5s + 4).
    >>> num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    >>> den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
    >>> sys1 = ct.tf(num, den)

    >>> # Create a variable 's' to allow algebra operations for SISO systems
    >>> s = ct.tf('s')
    >>> G  = (s + 1)/(s**2 + 2*s + 1)

    >>> # Convert a StateSpace to a TransferFunction object.
    >>> sys_ss = ct.ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
    >>> sys2 = ct.tf(sys1)

    """

    if len(args) == 2 or len(args) == 3:
        return TransferFunction(*args, **kwargs)

    elif len(args) == 1 and isinstance(args[0], str):
        # Make sure there were no extraneous keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

        # Look for special cases defining differential/delay operator
        if args[0] == 's':
            return TransferFunction.s
        elif args[0] == 'z':
            return TransferFunction.z

    elif len(args) == 1:
        from .statesp import StateSpace
        sys = args[0]
        if isinstance(sys, StateSpace):
            return ss2tf(sys, **kwargs)
        elif isinstance(sys, TransferFunction):
            # Use copy constructor
            return TransferFunction(sys, **kwargs)
        else:
            raise TypeError("tf(sys): sys must be a StateSpace or "
                            "TransferFunction object.   It is %s." % type(sys))
    else:
        raise ValueError("Needs 1 or 2 arguments; received %i." % len(args))


def zpk(zeros, poles, gain, *args, **kwargs):
    """zpk(zeros, poles, gain[, dt])

    Create a transfer function from zeros, poles, gain.

    Given a list of zeros z_i, poles p_j, and gain k, return the transfer
    function:

    .. math::
      H(s) = k \\frac{(s - z_1) (s - z_2) \\cdots (s - z_m)}
                     {(s - p_1) (s - p_2) \\cdots (s - p_n)}

    Parameters
    ----------
    zeros : array_like
        Array containing the location of zeros.
    poles : array_like
        Array containing the location of zeros.
    gain : float
        System gain
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous
        time, True indicates discrete time with unspecified sampling
        time, positive number is discrete time with specified
        sampling time, None indicates unspecified timebase (either
        continuous or discrete time).
    inputs, outputs, states : str, or list of str, optional
        List of strings that name the individual signals.  If this parameter
        is not given or given as `None`, the signal names will be of the
        form `s[i]` (where `s` is one of `u`, `y`, or `x`). See
        :class:`InputOutputSystem` for more information.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.
    display_format: None, 'poly' or 'zpk'
        Set the display format used in printing the TransferFunction object.
        Default behavior is polynomial display and can be changed by
        changing config.defaults['xferfcn.display_format'].

    Returns
    -------
    out: :class:`TransferFunction`
        Transfer function with given zeros, poles, and gain.

    Examples
    --------
    >>> G = ct.zpk([1], [2, 3], gain=1, display_format='zpk')
    >>> print(G)                                                # doctest: +SKIP

         s - 1
    ---------------
    (s - 2) (s - 3)

    """
    num, den = zpk2tf(zeros, poles, gain)
    return TransferFunction(num, den, *args, **kwargs)


def ss2tf(*args, **kwargs):

    """ss2tf(sys)

    Transform a state space system to a transfer function.

    The function accepts either 1 or 4 parameters:

    ``ss2tf(sys)``
        Convert a linear system from state space into transfer function
        form. Always creates a new system.

    ``ss2tf(A, B, C, D)``
        Create a transfer function system from the matrices of its state and
        output equations.

        For details see: :func:`tf`

    Parameters
    ----------
    sys: StateSpace
        A linear system
    A: array_like or string
        System matrix
    B: array_like or string
        Control matrix
    C: array_like or string
        Output matrix
    D: array_like or string
        Feedthrough matrix

    Returns
    -------
    out: TransferFunction
        New linear system in transfer function form

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    name : string, optional
        System name. If unspecified, a generic name <sys[id]> is generated
        with a unique integer id.

    Raises
    ------
    ValueError
        if matrix sizes are not self-consistent, or if an invalid number of
        arguments is passed in
    TypeError
        if `sys` is not a StateSpace object

    See Also
    --------
    tf
    ss
    tf2ss

    Examples
    --------
    >>> A = [[-1, -2], [3, -4]]
    >>> B = [[5], [6]]
    >>> C = [[7, 8]]
    >>> D = [[9]]
    >>> sys1 = ct.ss2tf(A, B, C, D)

    >>> sys_ss = ct.ss(A, B, C, D)
    >>> sys2 = ct.ss2tf(sys_ss)

    """

    from .statesp import StateSpace
    if len(args) == 4 or len(args) == 5:
        # Assume we were given the A, B, C, D matrix and (optional) dt
        return _convert_to_transfer_function(StateSpace(*args, **kwargs))

    if len(args) == 1:
        sys = args[0]
        if isinstance(sys, StateSpace):
            kwargs = kwargs.copy()
            if not kwargs.get('inputs'):
                kwargs['inputs'] = sys.input_labels
            if not kwargs.get('outputs'):
                kwargs['outputs'] = sys.output_labels
            return TransferFunction(
                _convert_to_transfer_function(
                    sys, use_prefix_suffix=not sys._generic_name_check()),
                **kwargs)
        else:
            raise TypeError(
                "ss2tf(sys): sys must be a StateSpace object.  It is %s."
                % type(sys))
    else:
        raise ValueError("Needs 1 or 4 arguments; received %i." % len(args))


def tfdata(sys):
    """
    Return transfer function data objects for a system

    Parameters
    ----------
    sys: LTI (StateSpace, or TransferFunction)
        LTI system whose data will be returned

    Returns
    -------
    (num, den): numerator and denominator arrays
        Transfer function coefficients (SISO only)
    """
    tf = _convert_to_transfer_function(sys)

    return tf.num, tf.den


def _clean_part(data):
    """
    Return a valid, cleaned up numerator or denominator
    for the TransferFunction class.

    Parameters
    ----------
    data: numerator or denominator of a transfer function.

    Returns
    -------
    data: list of lists of ndarrays, with int converted to float
    """
    valid_types = (int, float, complex, np.number)
    valid_collection = (list, tuple, ndarray)

    if (isinstance(data, valid_types) or
            (isinstance(data, ndarray) and data.ndim == 0)):
        # Data is a scalar (including 0d ndarray)
        data = [[array([data])]]
    elif (isinstance(data, ndarray) and data.ndim == 3 and
          isinstance(data[0, 0, 0], valid_types)):
        data = [[array(data[i, j])
                 for j in range(data.shape[1])]
                for i in range(data.shape[0])]
    elif (isinstance(data, valid_collection) and
            all([isinstance(d, valid_types) for d in data])):
        data = [[array(data)]]
    elif (isinstance(data, (list, tuple)) and
          isinstance(data[0], (list, tuple)) and
          (isinstance(data[0][0], valid_collection) and
           all([isinstance(d, valid_types) for d in data[0][0]]))):
        data = list(data)
        for j in range(len(data)):
            data[j] = list(data[j])
            for k in range(len(data[j])):
                data[j][k] = array(data[j][k])
    else:
        # If the user passed in anything else, then it's unclear what
        # the meaning is.
        raise TypeError(
            "The numerator and denominator inputs must be scalars or vectors "
            "(for\nSISO), or lists of lists of vectors (for SISO or MIMO).")

    # Check for coefficients that are ints and convert to floats
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if isinstance(data[i][j][k], (int, np.int32, np.int64)):
                    data[i][j][k] = float(data[i][j][k])

    return data


# Define constants to represent differentiation, unit delay
TransferFunction.s = TransferFunction([1, 0], [1], 0)
TransferFunction.z = TransferFunction([1, 0], [1], True)
