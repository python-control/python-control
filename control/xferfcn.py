# xferfcn.py - transfer function class and related functions
#
# Initial author: Richard M. Murray
# Creation date: 24 May 2009
# Pre-2014 revisions: Kevin K. Chen, Dec 2010
# Use `git shortlog -n -s xferfcn.py` for full list of contributors

"""Transfer function class and related functions.

This module contains the `TransferFunction` class and also functions
that operate on transfer functions.

"""

import sys
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain, product
from re import sub
from warnings import warn

import numpy as np
import scipy as sp
# float64 needed in eval() call
from numpy import float64  # noqa: F401
from numpy import array, delete, empty, exp, finfo, ndarray, nonzero, ones, \
    poly, polyadd, polymul, polyval, real, roots, sqrt, where, zeros
from scipy.signal import TransferFunction as signalTransferFunction
from scipy.signal import cont2discrete, tf2zpk, zpk2tf

from . import bdalg, config
from .exception import ControlMIMONotImplemented
from .frdata import FrequencyResponseData
from .iosys import InputOutputSystem, NamedSignal, _process_iosys_keywords, \
    _process_subsys_index, common_timebase
from .lti import LTI, _process_frequency_response

__all__ = ['TransferFunction', 'tf', 'zpk', 'ss2tf', 'tfdata']


# Define module default parameter values
_xferfcn_defaults = {
    'xferfcn.display_format': 'poly',
    'xferfcn.floating_point_format': '.4g'
}


class TransferFunction(LTI):
    """TransferFunction(num, den[, dt])

    Transfer function representation for LTI input/output systems.

    The TransferFunction class is used to represent systems in transfer
    function form.  Transfer functions are usually created with the
    `tf` factory function.

    Parameters
    ----------
    num : 2D list of coefficient arrays
        Polynomial coefficients of the numerator.
    den : 2D list of coefficient arrays
        Polynomial coefficients of the denominator.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None indicates
        unspecified timebase (either continuous or discrete time).

    Attributes
    ----------
    ninputs, noutputs : int
        Number of input and output signals.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    input_labels, output_labels : list of str
        Names for the input and output signals.
    name : string, optional
        System name.
    num_array, den_array : 2D array of lists of float
        Numerator and denominator polynomial coefficients as 2D array
        of 1D array objects (of varying length).
    num_list, den_list : 2D list of 1D array
        Numerator and denominator polynomial coefficients as 2D lists
        of 1D array objects (of varying length).
    display_format : None, 'poly' or 'zpk'
        Display format used in printing the TransferFunction object.
        Default behavior is polynomial display and can be changed by
        changing `config.defaults['xferfcn.display_format']`.
    s : `TransferFunction`
        Represents the continuous-time differential operator.
    z : `TransferFunction`
        Represents the discrete-time delay operator.

    See Also
    --------
    tf, InputOutputSystem, FrequencyResponseData

    Notes
    -----
    The numerator and denominator polynomials are stored as 2D arrays
    with each element containing a 1D array of coefficients.  These data
    structures can be retrieved using `num_array` and `den_array`.  For
    example,

    >>> sys.num_array[2, 5]         # doctest: +SKIP

    gives the numerator of the transfer function from the 6th input to the
    3rd output. (Note: a single 3D array structure cannot be used because
    the numerators and denominators can have different numbers of
    coefficients in each entry.)

    The attributes `num_list` and `den_list` are properties that return
    2D nested lists containing MIMO numerator and denominator coefficients.
    For example,

    >>> sys.num_list[2][5]          # doctest: +SKIP

    For legacy purposes, this list-based representation can also be
    obtained using `num` and `den`.

    A discrete-time transfer function is created by specifying a nonzero
    'timebase' dt when the system is constructed:

    * `dt` = 0: continuous-time system (default)
    * `dt` > 0: discrete-time system with sampling period `dt`
    * `dt` = True: discrete time with unspecified sampling period
    * `dt` = None: no timebase specified

    Systems must have compatible timebases in order to be combined. A
    discrete-time system with unspecified sampling time (`dt` = True) can
    be combined with a system having a specified sampling time; the result
    will be a discrete-time system with the sample time of the other
    system. Similarly, a system with timebase None can be combined with a
    system having any timebase; the result will have the timebase of the
    other system.  The default value of dt can be changed by changing the
    value of `config.defaults['control.default_dt']`.

    A transfer function is callable and returns the value of the transfer
    function evaluated at a point in the complex plane.  See
    `TransferFunction.__call__` for a more detailed description.

    Subsystems corresponding to selected input/output pairs can be
    created by indexing the transfer function::

        subsys = sys[output_spec, input_spec]

    The input and output specifications can be single integers, lists of
    integers, or slices.  In addition, the strings representing the names
    of the signals can be used and will be replaced with the equivalent
    signal offsets.

    The TransferFunction class defines two constants `s` and `z` that
    represent the differentiation and delay operators in continuous and
    discrete time.  These can be used to create variables that allow
    algebraic creation of transfer functions.  For example,

    >>> s = ct.TransferFunction.s  # or ct.tf('s')
    >>> G = (s + 1)/(s**2 + 2*s + 1)

    """
    def __init__(self, *args, **kwargs):
        """TransferFunction(num, den[, dt])

        Construct a transfer function.

        The default constructor is TransferFunction(num, den), where num
        and den are 2D arrays of arrays containing polynomial coefficients.
        To create a discrete-time transfer function, use
        ``TransferFunction(num, den, dt)`` where `dt` is the sampling time
        (or True for unspecified sampling time).  To call the copy
        constructor, call ``TransferFunction(sys)``, where `sys` is a
        TransferFunction object (continuous or discrete).

        See `TransferFunction` and `tf` for more information.

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

        num = _clean_part(num, "numerator")
        den = _clean_part(den, "denominator")

        #
        # Process keyword arguments
        #
        # During module init, TransferFunction.s and TransferFunction.z
        # get initialized when defaults are not fully initialized yet.
        # Use 'poly' in these cases.

        self.display_format = kwargs.pop('display_format', None)
        if self.display_format not in (None, 'poly', 'zpk'):
            raise ValueError("display_format must be 'poly' or 'zpk',"
                             " got '%s'" % self.display_format)

        #
        # Determine if the transfer function is static (memoryless)
        #
        # True if and only if all of the numerator and denominator
        # polynomials of the (MIMO) transfer function are zeroth order.
        #
        static = True
        for arr in [num, den]:
            # Iterate using refs_OK since num and den are ndarrays of ndarrays
            for poly_ in np.nditer(arr, flags=['refs_ok']):
                if poly_.item().size > 1:
                    static = False
                    break
            if not static:
                break
        self._static = static           # retain for later usage

        defaults = args[0] if len(args) == 1 else \
            {'inputs': num.shape[1], 'outputs': num.shape[0]}

        name, inputs, outputs, states, dt = _process_iosys_keywords(
                kwargs, defaults, static=static)
        if states:
            raise TypeError(
                "states keyword not allowed for transfer functions")

        # Initialize LTI (InputOutputSystem) object
        super().__init__(
            name=name, inputs=inputs, outputs=outputs, dt=dt, **kwargs)

        #
        # Check to make sure everything is consistent
        #
        # Make sure numerator and denominator matrices have consistent sizes
        if self.ninputs != den.shape[1]:
            raise ValueError(
                "The numerator has %i input(s), but the denominator has "
                "%i input(s)." % (self.ninputs, den.shape[1]))
        if self.noutputs != den.shape[0]:
            raise ValueError(
                "The numerator has %i output(s), but the denominator has "
                "%i output(s)." % (self.noutputs, den.shape[0]))

        # Additional checks/updates on structure of the transfer function
        for i in range(self.noutputs):
            # Check for zeros in numerator or denominator
            # TODO: Right now these checks are only done during construction.
            # It might be worthwhile to think of a way to perform checks if the
            # user modifies the transfer function after construction.
            for j in range(self.ninputs):
                # Check that we don't have any zero denominators.
                zeroden = True
                for k in den[i, j]:
                    if np.any(k):
                        zeroden = False
                        break
                if zeroden:
                    raise ValueError(
                        "Input %i, output %i has a zero denominator."
                        % (j + 1, i + 1))

                # If we have zero numerators, set the denominator to 1.
                zeronum = True
                for k in num[i, j]:
                    if np.any(k):
                        zeronum = False
                        break
                if zeronum:
                    den[i][j] = ones(1)

        # Store the numerator and denominator
        self.num_array = num
        self.den_array = den

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

    #: Numerator polynomial coefficients as a 2D array of 1D coefficients.
    #:
    #: :meta hide-value:
    num_array = None

    #: Denominator polynomial coefficients as a 2D array of 1D coefficients.
    #:
    #: :meta hide-value:
    den_array = None

    # Numerator and denominator as lists of lists of lists
    @property
    def num_list(self):
        """Numerator polynomial (as 2D nested list of 1D arrays)."""
        return self.num_array.tolist()

    @property
    def den_list(self):
        """Denominator polynomial (as 2D nested lists of 1D arrays)."""
        return self.den_array.tolist()

    # Legacy versions (TODO: add DeprecationWarning in a later release?)
    num, den = num_list, den_list

    def __call__(self, x, squeeze=None, warn_infinite=True):
        """Evaluate system transfer function at point in complex plane.

        Returns the value of the system's transfer function at a point `x`
        in the complex plane, where `x` is `s` for continuous-time systems
        and `z` for discrete-time systems.

        See `LTI.__call__` for details.

        """
        out = self.horner(x, warn_infinite=warn_infinite)
        return _process_frequency_response(self, x, out, squeeze=squeeze)

    def horner(self, x, warn_infinite=True):
        """Evaluate value of transfer function using Horner's method.

        Evaluates ``sys(x)`` where `x` is a complex number `s` for
        continuous-time systems and `z` for discrete-time systems.  Expects
        inputs and outputs to be formatted correctly. Use ``sys(x)`` for a
        more user-friendly interface.

        Parameters
        ----------
        x : complex
            Complex frequency at which the transfer function is evaluated.

        warn_infinite : bool, optional
            If True (default), generate a warning if `x` is a pole.

        Returns
        -------
        complex

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
                    out[i][j] = (polyval(self.num_array[i, j], x_arr) /
                                 polyval(self.den_array[i, j], x_arr))
        return out

    def _truncatecoeff(self):
        """Remove extraneous zero coefficients from num and den.

        Check every element of the numerator and denominator matrices, and
        truncate leading zeros.  For instance, running self._truncatecoeff()
        will reduce self.num = [[[0, 0, 1, 2]]] to [[[1, 2]]].

        """

        # Beware: this is a shallow copy.  This should be okay.
        data = [self.num_array, self.den_array]
        for p in range(len(data)):
            for i in range(self.noutputs):
                for j in range(self.ninputs):
                    # Find the first nontrivial coefficient.
                    nonzero = None
                    for k in range(data[p][i, j].size):
                        if data[p][i, j][k]:
                            nonzero = k
                            break

                    if nonzero is None:
                        # The array is all zeros.
                        data[p][i][j] = zeros(1)
                    else:
                        # Truncate the trivial coefficients.
                        data[p][i][j] = data[p][i][j][nonzero:]
        [self.num_array, self.den_array] = data

    def __str__(self, var=None):
        """String representation of the transfer function.

        Based on the display_format property, the output will be formatted as
        either polynomials or in zpk form.
        """
        display_format = config.defaults['xferfcn.display_format'] if \
            self.display_format is None else self.display_format
        mimo = not self.issiso()
        if var is None:
            var = 's' if self.isctime() else 'z'
        outstr = f"{InputOutputSystem.__str__(self)}"

        for ni in range(self.ninputs):
            for no in range(self.noutputs):
                outstr += "\n"
                if mimo:
                    outstr += "\nInput %i to output %i:\n" % (ni + 1, no + 1)

                # Convert the numerator and denominator polynomials to strings.
                if display_format == 'poly':
                    numstr = _tf_polynomial_to_string(
                        self.num_array[no, ni], var=var)
                    denstr = _tf_polynomial_to_string(
                        self.den_array[no, ni], var=var)
                elif display_format == 'zpk':
                    num = self.num_array[no, ni]
                    if num.size == 1 and num.item() == 0:
                        # Catch a special case that SciPy doesn't handle
                        z, p, k = tf2zpk([1.], self.den_array[no, ni])
                        k = 0
                    else:
                        z, p, k = tf2zpk(
                            self.num[no][ni], self.den_array[no, ni])
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

                outstr += "\n  " + numstr + "\n  " + dashes + "\n  " + denstr

        return outstr

    def _repr_eval_(self):
        # Loadable format
        if self.issiso():
            out = "TransferFunction(\n{num},\n{den}".format(
                num=self.num_array[0, 0].__repr__(),
                den=self.den_array[0, 0].__repr__())
        else:
            out = "TransferFunction(\n["
            for entry in [self.num_array, self.den_array]:
                for i in range(self.noutputs):
                    out += "[" if i == 0 else "\n ["
                    linelen = 0
                    for j in range(self.ninputs):
                        out += ", " if j != 0 else ""
                        numstr = np.array_repr(entry[i, j])
                        if linelen + len(numstr) > 72:
                            out += "\n  "
                            linelen = 0
                        out += numstr
                        linelen += len(numstr)
                    out += "]," if i < self.noutputs - 1 else "]"
                out += "],\n[" if entry is self.num_array else "]"

        out += super()._dt_repr(separator=",\n", space="")
        if len(labels := self._label_repr()) > 0:
            out += ",\n" + labels

        out += ")"
        return out

    def _repr_html_(self, var=None):
        """HTML/LaTeX representation of xferfcn, for Jupyter notebook."""
        display_format = config.defaults['xferfcn.display_format'] if \
            self.display_format is None else self.display_format
        mimo = not self.issiso()
        if var is None:
            var = 's' if self.isctime() else 'z'
        out = [super()._repr_info_(html=True), '\n$$']

        if mimo:
            out.append(r"\begin{bmatrix}")

        for no in range(self.noutputs):
            for ni in range(self.ninputs):
                # Convert the numerator and denominator polynomials to strings.
                if display_format == 'poly':
                    numstr = _tf_polynomial_to_string(
                        self.num_array[no, ni], var=var)
                    denstr = _tf_polynomial_to_string(
                        self.den_array[no, ni], var=var)
                elif display_format == 'zpk':
                    z, p, k = tf2zpk(
                        self.num_array[no, ni], self.den_array[no, ni])
                    numstr = _tf_factorized_polynomial_to_string(
                        z, gain=k, var=var)
                    denstr = _tf_factorized_polynomial_to_string(p, var=var)

                numstr = _tf_string_to_latex(numstr, var=var)
                denstr = _tf_string_to_latex(denstr, var=var)

                out += [r"\dfrac{", numstr, "}{", denstr, "}"]

                if mimo and ni < self.ninputs - 1:
                    out.append("&")

            if mimo:
                out.append(r"\\")

        if mimo:
            out.append(r" \end{bmatrix}")

        out.append("$$")

        return ''.join(out)

    def __neg__(self):
        """Negate a transfer function."""
        num = deepcopy(self.num_array)
        for i in range(self.noutputs):
            for j in range(self.ninputs):
                num[i, j] *= -1
        return TransferFunction(num, self.den, self.dt)

    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""
        from .statesp import StateSpace

        # Convert the second argument to a transfer function.
        if isinstance(other, StateSpace):
            other = _convert_to_transfer_function(other)
        elif isinstance(other, (int, float, complex, np.number, np.ndarray)):
            other = _convert_to_transfer_function(other, inputs=self.ninputs,
                                                  outputs=self.noutputs)

        if not isinstance(other, TransferFunction):
            return NotImplemented

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = np.ones((other.noutputs, other.ninputs)) * self
        elif not self.issiso() and other.issiso():
            other = np.ones((self.noutputs, self.ninputs)) * other

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
        num = _create_poly_array((self.noutputs, self.ninputs))
        den = _create_poly_array((self.noutputs, self.ninputs))

        for i in range(self.noutputs):
            for j in range(self.ninputs):
                num[i, j], den[i, j] = _add_siso(
                    self.num_array[i, j], self.den_array[i, j],
                    other.num_array[i, j], other.den_array[i, j])

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
        from .statesp import StateSpace

        # Convert the second argument to a transfer function.
        if isinstance(other, (StateSpace, np.ndarray)):
            other = _convert_to_transfer_function(other)
        elif isinstance(other, (int, float, complex, np.number)):
            # Multiply by a scaled identity matrix (transfer function)
            other = _convert_to_transfer_function(np.eye(self.ninputs) * other)
        if not isinstance(other, TransferFunction):
            return NotImplemented

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self] * other.noutputs))
        elif not self.issiso() and other.issiso():
            other = bdalg.append(*([other] * self.ninputs))

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.noutputs:
            raise ValueError(
                "C = A * B: A has %i column(s) (input(s)), but B has %i "
                "row(s)\n(output(s))." % (self.ninputs, other.noutputs))

        ninputs = other.ninputs
        noutputs = self.noutputs

        dt = common_timebase(self.dt, other.dt)

        # Preallocate the numerator and denominator of the sum.
        num = _create_poly_array((noutputs, ninputs), [0])
        den = _create_poly_array((noutputs, ninputs), [1])

        # Temporary storage for the summands needed to find the (i, j)th
        # element of the product.
        num_summand = [[] for k in range(self.ninputs)]
        den_summand = [[] for k in range(self.ninputs)]

        # Multiply & add.
        for row in range(noutputs):
            for col in range(ninputs):
                for k in range(self.ninputs):
                    num_summand[k] = polymul(
                        self.num_array[row, k], other.num_array[k, col])
                    den_summand[k] = polymul(
                        self.den_array[row, k], other.den_array[k, col])
                    num[row, col], den[row, col] = _add_siso(
                        num[row, col], den[row, col],
                        num_summand[k], den_summand[k])
        return TransferFunction(num, den, dt)

    def __rmul__(self, other):
        """Right multiply two LTI objects (serial connection)."""

        # Convert the second argument to a transfer function.
        if isinstance(other, (int, float, complex, np.number)):
            # Multiply by a scaled identity matrix (transfer function)
            other = _convert_to_transfer_function(np.eye(self.noutputs) * other)
        else:
            other = _convert_to_transfer_function(other)

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self] * other.ninputs))
        elif not self.issiso() and other.issiso():
            other = bdalg.append(*([other] * self.noutputs))

        # Check that the input-output sizes are consistent.
        if other.ninputs != self.noutputs:
            raise ValueError(
                "C = A * B: A has %i column(s) (input(s)), but B has %i "
                "row(s)\n(output(s))." % (other.ninputs, self.noutputs))

        ninputs = self.ninputs
        noutputs = other.noutputs

        dt = common_timebase(self.dt, other.dt)

        # Preallocate the numerator and denominator of the sum.
        num = _create_poly_array((noutputs, ninputs), [0])
        den = _create_poly_array((noutputs, ninputs), [1])

        # Temporary storage for the summands needed to find the
        # (i, j)th element
        # of the product.
        num_summand = [[] for k in range(other.ninputs)]
        den_summand = [[] for k in range(other.ninputs)]

        for i in range(noutputs):  # Iterate through rows of product.
            for j in range(ninputs):  # Iterate through columns of product.
                for k in range(other.ninputs):  # Multiply & add.
                    num_summand[k] = polymul(
                        other.num_array[i, k], self.num_array[k, j])
                    den_summand[k] = polymul(
                        other.den_array[i, k], self.den_array[k, j])
                    num[i][j], den[i][j] = _add_siso(
                        num[i, j], den[i, j],
                        num_summand[k], den_summand[k])

        return TransferFunction(num, den, dt)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""

        if isinstance(other, (int, float, complex, np.number)):
            # Multiply by a scaled identity matrix (transfer function)
            other = _convert_to_transfer_function(np.eye(self.ninputs) * other)
        else:
            other = _convert_to_transfer_function(other)

        # Special case for SISO ``other``
        if not self.issiso() and other.issiso():
            other = bdalg.append(*([other**-1] * self.noutputs))
            return self * other

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            # TransferFunction.__truediv__ is currently implemented only for
            # SISO systems.
            return NotImplemented
        dt = common_timebase(self.dt, other.dt)

        num = polymul(self.num_array[0, 0], other.den_array[0, 0])
        den = polymul(self.den_array[0, 0], other.num_array[0, 0])

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

        # Special case for SISO ``self``
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self**-1] * other.ninputs))
            return other * self

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            # TransferFunction.__rtruediv__ is currently implemented only for
            # SISO systems
            return NotImplemented

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
        if not isinstance(key, Iterable) or len(key) != 2:
            raise IOError(
                "must provide indices of length 2 for transfer functions")

        # Convert signal names to integer offsets (via NamedSignal object)
        iomap = NamedSignal(
            np.empty((self.noutputs, self.ninputs)),
            self.output_labels, self.input_labels)
        indices = iomap._parse_key(key, level=1)  # ignore index checks
        outdx, outputs = _process_subsys_index(
            indices[0], self.output_labels, slice_to_list=True)
        inpdx, inputs = _process_subsys_index(
            indices[1], self.input_labels, slice_to_list=True)

        # Construct the transfer function for the subsystem
        num = _create_poly_array((len(outputs), len(inputs)))
        den = _create_poly_array(num.shape)
        for row, i in enumerate(outdx):
            for col, j in enumerate(inpdx):
                num[row, col] = self.num_array[i, j]
                den[row, col] = self.den_array[i, j]
                col += 1
            row += 1

        # Create the system name
        sysname = config.defaults['iosys.indexed_system_name_prefix'] + \
            self.name + config.defaults['iosys.indexed_system_name_suffix']

        return TransferFunction(
            num, den, self.dt, inputs=inputs, outputs=outputs, name=sysname)

    def freqresp(self, omega):
        """Evaluate transfer function at complex frequencies.

        .. deprecated::0.9.0
            Method has been given the more Pythonic name
            `TransferFunction.frequency_response`. Or use
            `freqresp` in the MATLAB compatibility module.
        """
        warn("TransferFunction.freqresp(omega) will be removed in a "
             "future release of python-control; use "
             "sys.frequency_response(omega), or freqresp(sys, omega) in the "
             "MATLAB compatibility module instead", FutureWarning)
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
            return roots(self.num_array[0, 0]).astype(complex)

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two LTI objects.

        Parameters
        ----------
        other : `InputOutputSystem`
            System in the feedback path.

        sign : float, optional
            Gain to use in feedback path.  Defaults to -1.

        """
        other = _convert_to_transfer_function(other)

        if (self.ninputs > 1 or self.noutputs > 1 or
                other.ninputs > 1 or other.noutputs > 1):
            # TODO: MIMO feedback
            raise ControlMIMONotImplemented(
                "TransferFunction.feedback is currently not implemented for "
                "MIMO systems.")
        dt = common_timebase(self.dt, other.dt)

        num1 = self.num_array[0, 0]
        den1 = self.den_array[0, 0]
        num2 = other.num_array[0, 0]
        den2 = other.den_array[0, 0]

        num = polymul(num1, den2)
        den = polyadd(polymul(den2, den1), -sign * polymul(num2, num1))

        return TransferFunction(num, den, dt)

        # For MIMO or SISO systems, the analytic expression is
        #     self / (1 - sign * other * self)
        # But this does not work correctly because the state size will be too
        # large.

    def append(self, other):
        """Append a second model to the present model.

        The second model is converted to a transfer function if necessary,
        inputs and outputs are appended and their order is preserved.

        Parameters
        ----------
        other : `StateSpace` or `TransferFunction`
            System to be appended.

        Returns
        -------
        sys : `TransferFunction`
            System model with `other` appended to `self`.

        """
        other = _convert_to_transfer_function(other)

        new_tf = bdalg.combine_tf([
            [self, np.zeros((self.noutputs, other.ninputs))],
            [np.zeros((other.noutputs, self.ninputs)), other],
        ])

        return new_tf

    def minreal(self, tol=None):
        """Remove canceling pole/zero pairs from a transfer function.

        Parameters
        ----------
        tol : float
            Tolerance for determining whether poles and zeros overlap.

        """
        # based on octave minreal

        # default accuracy
        from sys import float_info
        sqrt_eps = sqrt(float_info.epsilon)

        # pre-allocate arrays
        num = _create_poly_array((self.noutputs, self.ninputs))
        den = _create_poly_array((self.noutputs, self.ninputs))

        for i in range(self.noutputs):
            for j in range(self.ninputs):

                # split up in zeros, poles and gain
                newzeros = []
                zeros = roots(self.num_array[i, j])
                poles = roots(self.den_array[i, j])
                gain = self.num_array[i, j][0] / self.den_array[i, j][0]

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
                num[i, j] = np.atleast_1d(gain * real(poly(newzeros)))
                den[i, j] = np.atleast_1d(real(poly(poles)))

        # end result
        return TransferFunction(num, den, self.dt)

    def returnScipySignalLTI(self, strict=True):
        """Return a 2D array of `scipy.signal.lti` objects.

        For instance,

        >>> out = tfobject.returnScipySignalLTI()               # doctest: +SKIP
        >>> out[3, 5]                                           # doctest: +SKIP

        is a `scipy.signal.lti` object corresponding to the
        transfer function from the 6th input to the 4th output.

        Parameters
        ----------
        strict : bool, optional
            True (default):
                The timebase `tfobject.dt` cannot be None; it must be
                continuous (0) or discrete (True or > 0).
            False:
                if `tfobject.dt` is None, continuous-time
                `scipy.signal.lti` objects are returned

        Returns
        -------
        out : list of list of `scipy.signal.TransferFunction`
            Continuous time (inheriting from `scipy.signal.lti`)
            or discrete time (inheriting from `scipy.signal.dlti`)
            SISO objects.
        """
        if strict and self.dt is None:
            raise ValueError("with strict=True, dt cannot be None")

        if self.dt:
            kwdt = {'dt': self.dt}
        else:
            # scipy convention for continuous-time LTI systems: call without
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
        """Compute MIMO common denominators; return them and adjusted numerators.

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
            Multi-dimensional array of numerator coefficients with shape
            (n, n, kd) array, where n = max(sys.noutputs, sys.ninputs), kd
            = max(denorder) + 1.  `num[i,j]` gives the numerator coefficient
            array for the ith output and jth input; padded for use in
            td04ad ('C' option); matches the denorder order; highest
            coefficient starts on the left.  If `allow_nonproper` = True
            and the order of a numerator exceeds the order of the common
            denominator, `num` will be returned as None.
        den: array
            Multi-dimensional array of coefficients for common denominator
            polynomial with shape (sys.ninputs, kd) (one row per
            input). The array is prepared for use in slycot td04ad, the
            first element is the highest-order polynomial coefficient of
            `s`, matching the order in denorder. If denorder < number of
            columns in den, the den is padded with zeros.
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
                den[j, :maxindex+1] = poly(poles[j]).real
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
                    # numerator has a higher order than the denominator, the
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
                    num[i, j, maxindex+1-len(numpoly):maxindex+1] = \
                        numpoly.real
                    # print(num[i, j])

        if havenonproper:
            num = None

        return num, den, denorder

    def sample(self, Ts, method='zoh', alpha=None, prewarp_frequency=None,
               name=None, copy_names=True, **kwargs):
        """Convert a continuous-time system to discrete time.

        Creates a discrete-time system from a continuous-time system by
        sampling.  Multiple methods of conversion are supported.

        Parameters
        ----------
        Ts : float
            Sampling period.
        method : {'gbt', 'bilinear', 'euler', 'backward_diff', 'zoh', 'matched'}
            Method to use for sampling:

            * 'gbt': generalized bilinear transformation
            * 'backward_diff': Backwards difference ('gbt' with alpha=1.0)
            * 'bilinear' (or 'tustin'): Tustin's approximation ('gbt' with
              alpha=0.5)
            * 'euler': Euler (or forward difference) method ('gbt' with
              alpha=0)
            * 'matched': pole-zero match method
            * 'zoh': zero-order hold (default)
        alpha : float within [0, 1]
            The generalized bilinear transformation weighting parameter,
            which should only be specified with `method` = 'gbt', and is
            ignored otherwise. See `scipy.signal.cont2discrete`.
        prewarp_frequency : float within [0, infinity)
            The frequency [rad/s] at which to match with the input
            continuous- time system's magnitude and phase (the gain=1
            crossover frequency, for example). Should only be specified
            with `method` = 'bilinear' or 'gbt' with `alpha` = 0.5 and
            ignored otherwise.
        name : string, optional
            Set the name of the sampled system.  If not specified and if
            `copy_names` is False, a generic name 'sys[id]' is generated with
            a unique integer id.  If `copy_names` is True, the new system
            name is determined by adding the prefix and suffix strings in
            `config.defaults['iosys.sampled_system_name_prefix']` and
            `config.defaults['iosys.sampled_system_name_suffix']`, with the
            default being to add the suffix '$sampled'.

        copy_names : bool, Optional
            If True, copy the names of the input signals, output
            signals, and states to the sampled system.

        Returns
        -------
        sysd : `TransferFunction` system
            Discrete-time system, with sample period Ts.

        Other Parameters
        ----------------
        inputs : int, list of str or None, optional
            Description of the system inputs.  If not specified, the
            original system inputs are used.  See `InputOutputSystem` for
            more information.
        outputs : int, list of str or None, optional
            Description of the system outputs.  Same format as `inputs`.

        Notes
        -----
        Available only for SISO systems.  Uses `scipy.signal.cont2discrete`.

        Examples
        --------
        >>> sys = ct.tf(1, [1, 1])
        >>> sysd = sys.sample(0.5, method='bilinear')

        """
        if not self.isctime():
            raise ValueError("System must be continuous-time system")
        if not self.issiso():
            raise ControlMIMONotImplemented("Not implemented for MIMO systems")
        if method == "matched":
            if prewarp_frequency is not None:
                warn('prewarp_frequency ignored: incompatible conversion')
            return _c2d_matched(self, Ts, name=name, **kwargs)
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
        """Return the zero-frequency ("DC") gain.

        For a continuous-time transfer function G(s), the DC gain is G(0)
        For a discrete-time transfer function G(z), the DC gain is G(1)

        Parameters
        ----------
        warn_infinite : bool, optional
            By default, don't issue a warning message if the zero-frequency
            gain is infinite.  Setting `warn_infinite` to generate the
            warning message.

        Returns
        -------
        gain : (noutputs, ninputs) ndarray or scalar
            Array or scalar value for SISO systems, depending on
            `config.defaults['control.squeeze_frequency_response']`.  The
            value of the array elements or the scalar is either the
            zero-frequency (or DC) gain, or `inf`, if the frequency
            response is singular.

            For real valued systems, the empty imaginary part of the
            complex zero-frequency response is discarded and a real array or
            scalar is returned.

        Examples
        --------
        >>> G = ct.tf([1], [1, 4])
        >>> G.dcgain()
        np.float64(0.25)

        """
        return self._dcgain(warn_infinite)

    # Determine if a system is static (memoryless)
    def _isstatic(self):
        return self._static             # Check done at initialization

    # Attributes for differentiation and delay
    #
    # These attributes are created here with sphinx docstrings so that the
    # autodoc generated documentation has a description.  The actual values
    # of the class attributes are set at the bottom of the file to avoid
    # problems with recursive calls.

    #: Differentiation operator (continuous time).
    #:
    #: The `s` constant can be used to create continuous-time transfer
    #: functions using algebraic expressions.
    #:
    #: Examples
    #: --------
    #: >>> s = TransferFunction.s                               # doctest: +SKIP
    #: >>> G  = (s + 1)/(s**2 + 2*s + 1)                        # doctest: +SKIP
    #:
    #: :meta hide-value:
    s = None

    #: Delay operator (discrete time).
    #:
    #: The `z` constant can be used to create discrete-time transfer
    #: functions using algebraic expressions.
    #:
    #: Examples
    #: --------
    #: >>> z = TransferFunction.z                               # doctest: +SKIP
    #: >>> G  = 2 * z / (4 * z**3 + 3*z - 1)                    # doctest: +SKIP
    #:
    #: :meta hide-value:
    z = None


# c2d function contributed by Benjamin White, Oct 2012
def _c2d_matched(sysC, Ts, **kwargs):
    if not sysC.issiso():
        raise ControlMIMONotImplemented("Not implemented for MIMO systems")

    # Pole-zero match method of continuous to discrete time conversion
    szeros, spoles, _ = tf2zpk(sysC.num[0][0], sysC.den[0][0])
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
    gain = sysC.dcgain() / zgain.real
    sysDnum, sysDden = zpk2tf(zzeros, zpoles, gain)
    return TransferFunction(sysDnum, sysDden, Ts, **kwargs)


# Utility function to convert a transfer function polynomial to a string
# Borrowed from poly1d library
def _tf_polynomial_to_string(coeffs, var='s'):
    """Convert a transfer function polynomial to a string."""
    thestr = "0"

    # Apply NumPy formatting
    with np.printoptions(threshold=sys.maxsize):
        coeffs = eval(repr(coeffs))

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
    """Convert a factorized polynomial to a string."""
    # Apply NumPy formatting
    with np.printoptions(threshold=sys.maxsize):
        roots = eval(repr(roots))

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
    """Superscript all digits in a polynomial string and convert
    float coefficients in scientific notation to prettier LaTeX
    representation.

    """
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

    If `sys` is already a transfer function, then it is returned.  If `sys`
    is a state space object, then it is converted to a transfer function
    and returned.  If `sys` is a scalar, then the number of inputs and
    outputs can be specified manually, as in::

    >>> from control.xferfcn import _convert_to_transfer_function
    >>> sys = _convert_to_transfer_function(3.) # Assumes inputs = outputs = 1
    >>> sys = _convert_to_transfer_function(1., inputs=3, outputs=2)

    In the latter example, the matrix transfer function for `sys` is::

      [[1., 1., 1.]
       [1., 1., 1.]].

    If `sys` is an array_like type, then it is converted to a constant-gain
    transfer function.

    Note: no renaming of inputs and outputs is performed; this should be done
    by the calling function.

    Arrays can also be passed as an argument.  For example::

      sys = _convert_to_transfer_function([[1., 0.], [2., 3.]])

    will give a system with numerator matrix ``[[[1.0], [0.0]], [[2.0],
    [3.0]]]`` and denominator matrix ``[[[1.0], [1.0]], [[1.0], [1.0]]]``.

    """
    from .statesp import StateSpace

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
            # Preallocate numerator and denominator arrays
            num = [[[] for j in range(sys.ninputs)]
                   for i in range(sys.noutputs)]
            den = [[[] for j in range(sys.ninputs)]
                   for i in range(sys.noutputs)]

            try:
                # Use Slycot to make the transformation
                # Make sure to convert system matrices to NumPy arrays
                from slycot import tb04ad
                tfout = tb04ad(
                    sys.nstates, sys.ninputs, sys.noutputs, array(sys.A),
                    array(sys.B), array(sys.C), array(sys.D), tol1=0.0)

                for i in range(sys.noutputs):
                    for j in range(sys.ninputs):
                        num[i][j] = list(tfout[6][i, j, :])
                        # Each transfer function matrix row
                        # has a common denominator.
                        den[i][j] = list(tfout[5][i, :])

            except ImportError:
                # If slycot not available, do conversion using sp.signal.ss2tf
                for j in range(sys.ninputs):
                    num_j, den_j = sp.signal.ss2tf(
                        sys.A, sys.B, sys.C, sys.D, input=j)
                    for i in range(sys.noutputs):
                        num[i][j] = num_j[i]
                        den[i][j] = den_j

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

    # If this is array_like, try to create a constant feedthrough
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
        a new system, even if `sys` is already a `TransferFunction` object.

    ``tf(num, den)``

        Create a transfer function system from its numerator and denominator
        polynomial coefficients.

        If `num` and `den` are 1D array_like objects, the function creates a
        SISO system.

        To create a MIMO system, `num` and `den` need to be 2D arrays of
        of array_like objects (a 3 dimensional data structure in total;
        for details see note below).  If the denominator for all transfer
        function is the same, `den` can be specified as a 1D array.

    ``tf(num, den, dt)``

        Create a discrete-time transfer function system; dt can either be a
        positive number indicating the sampling time or True if no
        specific timebase is given.

    ``tf([[G11, ..., G1m], ..., [Gp1, ..., Gpm]][, dt])``

        Create a p x m MIMO system from SISO transfer functions Gij.  See
        `combine_tf` for more details.

    ``tf('s')`` or ``tf('z')``

        Create a transfer function representing the differential operator
        ('s') or delay operator ('z').

    Parameters
    ----------
    sys : `LTI` (`StateSpace` or `TransferFunction`)
        A linear system that will be converted to a transfer function.
    arr : 2D list of `TransferFunction`
        2D list of SISO transfer functions to create MIMO transfer function.
    num : array_like, or list of list of array_like
        Polynomial coefficients of the numerator.
    den : array_like, or list of list of array_like
        Polynomial coefficients of the denominator.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None indicates
        unspecified timebase (either continuous or discrete time).
    display_format : None, 'poly' or 'zpk'
        Set the display format used in printing the `TransferFunction` object.
        Default behavior is polynomial display and can be changed by
        changing `config.defaults['xferfcn.display_format']`.

    Returns
    -------
    sys : `TransferFunction`
        The new linear system.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    input_prefix, output_prefix : string, optional
        Set the prefix for input and output signals.  Defaults = 'u', 'y'.
    name : string, optional
        System name. If unspecified, a generic name 'sys[id]' is generated
        with a unique integer id.

    Raises
    ------
    ValueError
        If `num` and `den` have invalid or unequal dimensions.
    TypeError
        If `num` or `den` are of incorrect type.

    See Also
    --------
    TransferFunction, ss, ss2tf, tf2ss

    Notes
    -----
    MIMO transfer functions are created by passing a 2D array of coefficients:
    ``num[i][j]`` contains the polynomial coefficients of the numerator
    for the transfer function from the (j+1)st input to the (i+1)st output,
    and ``den[i][j]`` works the same way.

    The list ``[2, 3, 4]`` denotes the polynomial :math:`2 s^2 + 3 s + 4`.

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

    >>> # Convert a state space system to a transfer function:
    >>> sys_ss = ct.ss([[1, -2], [3, -4]], [[5], [7]], [[6, 8]], 9)
    >>> sys_tf = ct.tf(sys_ss)

    """
    if len(args) == 1 and isinstance(args[0], str):
        # Make sure there were no extraneous keywords
        if kwargs:
            raise TypeError("unrecognized keywords: ", str(kwargs))

        # Look for special cases defining differential/delay operator
        if args[0] == 's':
            return TransferFunction.s
        elif args[0] == 'z':
            return TransferFunction.z

    elif len(args) == 1 and isinstance(args[0], list):
        # Allow passing an array of SISO transfer functions
        from .bdalg import combine_tf
        return combine_tf(*args)

    elif len(args) == 1:
        from .statesp import StateSpace
        if isinstance(sys := args[0], StateSpace):
            return ss2tf(sys, **kwargs)
        elif isinstance(sys, TransferFunction):
            # Use copy constructor
            return TransferFunction(sys, **kwargs)
        elif isinstance(data := args[0], np.ndarray) and data.ndim == 2 or \
             isinstance(data, list) and isinstance(data[0], list):
            raise NotImplementedError(
                "arrays of transfer functions not (yet) supported")
        else:
            raise TypeError("tf(sys): sys must be a StateSpace or "
                            "TransferFunction object.   It is %s." % type(sys))

    elif len(args) == 3:
        if 'dt' in kwargs:
            warn("received multiple dt arguments, "
                 f"using positional arg {args[2]}")
        kwargs['dt'] = args[2]
        args = args[:2]

    elif len(args) != 2:
        raise ValueError("Needs 1, 2, or 3 arguments; received %i." % len(args))

    #
    # Process the numerator and denominator arguments
    #
    # If we got through to here, we have two arguments (num, den) and
    # the keywords (including dt).  The only thing left to do is look
    # for some special cases, like having a common denominator.
    #
    num, den = args

    num = _clean_part(num, "numerator")
    den = _clean_part(den, "denominator")

    if den.size == 1 and num.size > 1:
        # Broadcast denominator to shape of numerator
        den = np.broadcast_to(den, num.shape).copy()

    return TransferFunction(num, den, **kwargs)


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
        Array containing the location of poles.
    gain : float
        System gain.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete time).
    inputs, outputs, states : str, or list of str, optional
        List of strings that name the individual signals.  If this parameter
        is not given or given as None, the signal names will be of the
        form 's[i]' (where 's' is one of 'u', 'y', or 'x'). See
        `InputOutputSystem` for more information.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.
    display_format : None, 'poly' or 'zpk', optional
        Set the display format used in printing the `TransferFunction` object.
        Default behavior is polynomial display and can be changed by
        changing `config.defaults['xferfcn.display_format']`.

    Returns
    -------
    out : `TransferFunction`
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

        For details see: `tf`.

    Parameters
    ----------
    sys : `StateSpace`
        A linear system.
    A : array_like or string
        System matrix.
    B : array_like or string
        Control matrix.
    C : array_like or string
        Output matrix.
    D : array_like or string
        Feedthrough matrix.
    **kwargs : keyword arguments
        Additional arguments passed to `tf` (e.g., signal names).

    Returns
    -------
    out : `TransferFunction`
        New linear system in transfer function form.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    name : string, optional
        System name. If unspecified, a generic name 'sys[id]' is generated
        with a unique integer id.

    Raises
    ------
    ValueError
        If matrix sizes are not self-consistent, or if an invalid number of
        arguments is passed in.
    TypeError
        If `sys` is not a `StateSpace` object.

    See Also
    --------
    tf, ss, tf2ss

    Examples
    --------
    >>> A = [[-1, -2], [3, -4]]
    >>> B = [[5], [6]]
    >>> C = [[7, 8]]
    >>> D = [[9]]
    >>> sys1 = ct.ss2tf(A, B, C, D)

    >>> sys_ss = ct.ss(A, B, C, D)
    >>> sys_tf = ct.ss2tf(sys_ss)

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
    Return transfer function data objects for a system.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        LTI system whose data will be returned.

    Returns
    -------
    num, den : numerator and denominator arrays
        Transfer function coefficients (SISO only).

    """
    tf = _convert_to_transfer_function(sys)

    return tf.num, tf.den


def _clean_part(data, name="<unknown>"):
    """
    Return a valid, cleaned up numerator or denominator
    for the `TransferFunction` class.

    Parameters
    ----------
    data : numerator or denominator of a transfer function.

    Returns
    -------
    data: list of lists of ndarrays, with int converted to float

    """
    valid_types = (int, float, complex, np.number)
    unsupported_types = (complex, np.complexfloating)
    valid_collection = (list, tuple, ndarray)

    if isinstance(data, np.ndarray) and data.ndim == 2 and \
       data.dtype == object and isinstance(data[0, 0], np.ndarray):
        # Data is already in the right format
        return data
    elif isinstance(data, ndarray) and data.ndim == 3 and \
          isinstance(data[0, 0, 0], valid_types):
        out = np.empty(data.shape[0:2], dtype=np.ndarray)
        for i, j in product(range(out.shape[0]), range(out.shape[1])):
            out[i, j] = data[i, j, :]
    elif (isinstance(data, valid_types) or
            (isinstance(data, ndarray) and data.ndim == 0)):
        # Data is a scalar (including 0d ndarray)
        out = np.empty((1,1), dtype=np.ndarray)
        out[0, 0] = array([data])
    elif (isinstance(data, valid_collection) and
            all([isinstance(d, valid_types) for d in data])):
        out = np.empty((1,1), dtype=np.ndarray)
        out[0, 0] = array(data)
    elif isinstance(data, (list, tuple)) and \
         isinstance(data[0], (list, tuple)) and \
         (isinstance(data[0][0], valid_collection) and
          all([isinstance(d, valid_types) for d in data[0][0]]) or \
          isinstance(data[0][0], valid_types)):
        out = np.empty((len(data), len(data[0])), dtype=np.ndarray)
        for i in range(out.shape[0]):
            if len(data[i]) != out.shape[1]:
                raise ValueError(
                    "Row 0 of the %s matrix has %i elements, but row "
                    "%i has %i." % (name, out.shape[1], i, len(data[i])))
            for j in range(out.shape[1]):
                out[i, j] = np.atleast_1d(data[i][j])
    else:
        # If the user passed in anything else, then it's unclear what
        # the meaning is.
        raise TypeError(
            "The numerator and denominator inputs must be scalars or vectors "
            "(for\nSISO), or lists of lists of vectors (for SISO or MIMO).")

    # Check for coefficients that are ints and convert to floats
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(len(out[i, j])):
                if isinstance(out[i, j][k], (int, np.integer)):
                    out[i, j][k] = float(out[i, j][k])
                elif isinstance(out[i, j][k], unsupported_types):
                    raise TypeError(
                        f"unsupported data type: {type(out[i, j][k])}")
    return out


#
# Define constants to represent differentiation, unit delay.
#
# Set the docstring explicitly to avoid having Sphinx document this as
# a method instead of a property/attribute.

TransferFunction.s = TransferFunction([1, 0], [1], 0, name='s')
TransferFunction.s.__doc__ = "Differentiation operator (continuous time)."

TransferFunction.z = TransferFunction([1, 0], [1], True, name='z')
TransferFunction.z.__doc__ = "Delay operator (discrete time)."


def _float2str(value):
    _num_format = config.defaults.get('xferfcn.floating_point_format', ':.4g')
    return f"{value:{_num_format}}"


def _create_poly_array(shape, default=None):
    out = np.empty(shape, dtype=np.ndarray)
    if default is not None:
        default = np.array(default)
        for i, j in product(range(shape[0]), range(shape[1])):
            out[i, j] = default
    return out
