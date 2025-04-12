# statesp.py - state space class and related functions
#
# Initial author: Richard M. Murray
# Creation date: 24 May 2009
# Pre-2014 revisions: Kevin K. Chen, Dec 2010
# Use `git shortlog -n -s statesp.py` for full list of contributors

"""State space class and related functions.

This module contains the `StateSpace class`, which is used to
represent linear systems in state space.

"""

import math
import sys
from collections.abc import Iterable
from warnings import warn

import numpy as np
import scipy as sp
import scipy.linalg
from numpy import array  # noqa: F401
from numpy import any, asarray, concatenate, cos, delete, empty, exp, eye, \
    isinf, pad, sin, squeeze, zeros
from numpy.linalg import LinAlgError, eigvals, matrix_rank, solve
from numpy.random import rand, randn
from scipy.signal import StateSpace as signalStateSpace
from scipy.signal import cont2discrete

import control

from . import bdalg, config
from .exception import ControlDimension, ControlMIMONotImplemented, \
    ControlSlycot, slycot_check
from .frdata import FrequencyResponseData
from .iosys import InputOutputSystem, NamedSignal, _process_iosys_keywords, \
    _process_signal_list, _process_subsys_index, common_timebase, issiso
from .lti import LTI, _process_frequency_response
from .mateqn import _check_shape
from .nlsys import InterconnectedSystem, NonlinearIOSystem

try:
    from slycot import ab13dd
except ImportError:
    ab13dd = None

__all__ = ['StateSpace', 'LinearICSystem', 'ss2io', 'tf2io', 'tf2ss',
           'ssdata', 'linfnorm', 'ss', 'rss', 'drss', 'summing_junction']

# Define module default parameter values
_statesp_defaults = {
    'statesp.remove_useless_states': False,
    'statesp.latex_num_format': '.3g',
    'statesp.latex_repr_type': 'partitioned',
    'statesp.latex_maxsize': 10,
    }


class StateSpace(NonlinearIOSystem, LTI):
    r"""StateSpace(A, B, C, D[, dt])

    State space representation for LTI input/output systems.

    The StateSpace class is used to represent state-space realizations of
    linear time-invariant (LTI) systems:

    .. math::

          dx/dt &= A x + B u \\
              y &= C x + D u

    where :math:`u` is the input, :math:`y` is the output, and
    :math:`x` is the state.  State space systems are usually created
    with the `ss` factory function.

    Parameters
    ----------
    A, B, C, D : array_like
        System matrices of the appropriate dimensions.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete time).

    Attributes
    ----------
    ninputs, noutputs, nstates : int
        Number of input, output and state variables.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    input_labels, output_labels, state_labels : list of str
        Names for the input, output, and state variables.
    name : string, optional
        System name.

    See Also
    --------
    ss, InputOutputSystem, NonlinearIOSystem

    Notes
    -----
    The main data members in the `StateSpace` class are the A, B, C, and D
    matrices.  The class also keeps track of the number of states (i.e.,
    the size of A).

    A discrete-time system is created by specifying a nonzero 'timebase', dt
    when the system is constructed:

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

    A state space system is callable and returns the value of the transfer
    function evaluated at a point in the complex plane.  See
    `StateSpace.__call__` for a more detailed description.

    Subsystems corresponding to selected input/output pairs can be
    created by indexing the state space system::

        subsys = sys[output_spec, input_spec]

    The input and output specifications can be single integers, lists of
    integers, or slices.  In addition, the strings representing the names
    of the signals can be used and will be replaced with the equivalent
    signal offsets.  The subsystem is created by truncating the inputs and
    outputs, but leaving the full set of system states.

    StateSpace instances have support for IPython HTML/LaTeX output, intended
    for pretty-printing in Jupyter notebooks.  The HTML/LaTeX output can be
    configured using `config.defaults['statesp.latex_num_format']`
    and `config.defaults['statesp.latex_repr_type']`.  The
    HTML/LaTeX output is tailored for MathJax, as used in Jupyter, and
    may look odd when typeset by non-MathJax LaTeX systems.

    `config.defaults['statesp.latex_num_format']` is a format string
    fragment, specifically the part of the format string after '{:'
    used to convert floating-point numbers to strings.  By default it
    is '.3g'.

    `config.defaults['statesp.latex_repr_type']` must either be
    'partitioned' or 'separate'.  If 'partitioned', the A, B, C, D
    matrices are shown as a single, partitioned matrix; if
    'separate', the matrices are shown separately.

    """
    def __init__(self, *args, **kwargs):
        """StateSpace(A, B, C, D[, dt])

        Construct a state space object.

        The default constructor is StateSpace(A, B, C, D), where A, B, C, D
        are matrices or equivalent objects.  To create a discrete-time
        system, use StateSpace(A, B, C, D, dt) where `dt` is the sampling
        time (or True for unspecified sampling time).  To call the copy
        constructor, call ``StateSpace(sys)``, where `sys` is a `StateSpace`
        object.

        See `StateSpace` and `ss` for more information.

        """
        #
        # Process positional arguments
        #

        if len(args) == 4:
            # The user provided A, B, C, and D matrices.
            A, B, C, D = args

        elif len(args) == 5:
            # Discrete time system
            A, B, C, D, dt = args
            if 'dt' in kwargs:
                warn("received multiple dt arguments, "
                     "using positional arg dt = %s" % dt)
            kwargs['dt'] = dt
            args = args[:-1]

        elif len(args) == 1:
            # Use the copy constructor
            if not isinstance(args[0], StateSpace):
                raise TypeError(
                    "the one-argument constructor can only take in a "
                    "StateSpace object; received %s" % type(args[0]))
            A = args[0].A
            B = args[0].B
            C = args[0].C
            D = args[0].D
            if 'dt' not in kwargs:
                kwargs['dt'] = args[0].dt

        else:
            raise TypeError(
                "Expected 1, 4, or 5 arguments; received %i." % len(args))

        # Convert all matrices to standard form (sizes checked later)
        A = _ssmatrix(A, square=True, name="A")
        B = _ssmatrix(
            B, axis=0 if np.asarray(B).ndim == 1 and len(B) == A.shape[0]
            else 1, name="B")
        C = _ssmatrix(
            C, axis=1 if np.asarray(C).ndim == 1 and len(C) == A.shape[0]
            else 0, name="C")
        if np.isscalar(D) and D == 0 and B.shape[1] > 0 and C.shape[0] > 0:
            # If D is a scalar zero, broadcast it to the proper size
            D = np.zeros((C.shape[0], B.shape[1]))
        D = _ssmatrix(D, name="D")

        # If only direct term is present, adjust sizes of C and D if needed
        if D.size > 0 and B.size == 0:
            B = np.zeros((0, D.shape[1]))
        if D.size > 0 and C.size == 0:
            C = np.zeros((D.shape[0], 0))

        # Matrices defining the linear system
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # Determine if the system is static (memoryless)
        static = (A.size == 0)

        #
        # Process keyword arguments
        #

        remove_useless_states = kwargs.pop(
            'remove_useless_states',
            config.defaults['statesp.remove_useless_states'])

        # Process iosys keywords
        defaults = args[0] if len(args) == 1 else \
            {'inputs': B.shape[1], 'outputs': C.shape[0],
             'states': A.shape[0]}
        name, inputs, outputs, states, dt = _process_iosys_keywords(
            kwargs, defaults, static=static)

        # Create updfcn and outfcn
        updfcn = lambda t, x, u, params: \
            self.A @ np.atleast_1d(x) + self.B @ np.atleast_1d(u)
        outfcn = lambda t, x, u, params: \
            self.C @ np.atleast_1d(x) + self.D @ np.atleast_1d(u)

        # Initialize NonlinearIOSystem object
        super().__init__(
            updfcn, outfcn,
            name=name, inputs=inputs, outputs=outputs,
            states=states, dt=dt, **kwargs)

        # Reset shapes if the system is static
        if static:
            A.shape = (0, 0)
            B.shape = (0, self.ninputs)
            C.shape = (self.noutputs, 0)

        # Check to make sure everything is consistent
        _check_shape(A, self.nstates, self.nstates, name="A")
        _check_shape(B, self.nstates, self.ninputs, name="B")
        _check_shape(C, self.noutputs, self.nstates, name="C")
        _check_shape(D, self.noutputs, self.ninputs, name="D")

        #
        # Final processing
        #
        # Check for states that don't do anything, and remove them
        if remove_useless_states:
            self._remove_useless_states()

    #
    # Class attributes
    #
    # These attributes are defined as class attributes so that they are
    # documented properly.  They are "overwritten" in __init__.
    #

    #: Number of system inputs.
    #:
    #: :meta hide-value:
    ninputs = 0

    #: Number of system outputs.
    #:
    #: :meta hide-value:
    noutputs = 0

    #: Number of system states.
    #:
    #: :meta hide-value:
    nstates = 0

    #: Dynamics matrix.
    #:
    #: :meta hide-value:
    A = []

    #: Input matrix.
    #:
    #: :meta hide-value:
    B = []

    #: Output matrix.
    #:
    #: :meta hide-value:
    C = []

    #: Direct term.
    #:
    #: :meta hide-value:
    D = []

    #
    # Getter and setter functions for legacy state attributes
    #
    # For this iteration, generate a deprecation warning whenever the
    # getter/setter is called.  For a future iteration, turn it into a
    # future warning, so that users will see it.
    #

    def _get_states(self):
        warn("The StateSpace `states` attribute will be deprecated in a "
             "future release.  Use `nstates` instead.",
             FutureWarning, stacklevel=2)
        return self.nstates

    def _set_states(self, value):
        warn("The StateSpace `states` attribute will be deprecated in a "
             "future release.  Use `nstates` instead.",
             FutureWarning, stacklevel=2)
        self.nstates = value

    #: Deprecated attribute; use `nstates` instead.
    #:
    #: The `state` attribute was used to store the number of states for : a
    #: state space system.  It is no longer used.  If you need to access the
    #: number of states, use `nstates`.
    states = property(_get_states, _set_states)

    def _remove_useless_states(self):
        """Check for states that don't do anything, and remove them.

        Scan the A, B, and C matrices for rows or columns of zeros.  If the
        zeros are such that a particular state has no effect on the input-
        output dynamics, then remove that state from the A, B, and C matrices.

        """

        # Search for useless states and get indices of these states.
        ax1_A = np.where(~self.A.any(axis=1))[0]
        ax1_B = np.where(~self.B.any(axis=1))[0]
        ax0_A = np.where(~self.A.any(axis=0))[-1]
        ax0_C = np.where(~self.C.any(axis=0))[-1]
        useless_1 = np.intersect1d(ax1_A, ax1_B, assume_unique=True)
        useless_2 = np.intersect1d(ax0_A, ax0_C, assume_unique=True)
        useless = np.union1d(useless_1, useless_2)

        # Remove the useless states.
        self.A = delete(self.A, useless, 0)
        self.A = delete(self.A, useless, 1)
        self.B = delete(self.B, useless, 0)
        self.C = delete(self.C, useless, 1)

        # Remove any state names that we don't need
        self.set_states(
            [self.state_labels[i] for i in range(self.nstates)
             if i not in useless])

    def __str__(self):
        """Return string representation of the state space system."""
        string = f"{InputOutputSystem.__str__(self)}\n\n"
        string += "\n\n".join([
            "{} = {}".format(Mvar,
                               "\n    ".join(str(M).splitlines()))
            for Mvar, M in zip(["A", "B", "C", "D"],
                               [self.A, self.B, self.C, self.D])])
        return string

    def _repr_eval_(self):
        # Loadable format
        out = "StateSpace(\n{A},\n{B},\n{C},\n{D}".format(
            A=self.A.__repr__(), B=self.B.__repr__(),
            C=self.C.__repr__(), D=self.D.__repr__())

        out += super()._dt_repr(separator=",\n", space="")
        if len(labels := super()._label_repr()) > 0:
            out += ",\n" + labels

        out += ")"
        return out

    def _repr_html_(self):
        """HTML representation of state-space model.

        Output is controlled by config options statesp.latex_repr_type,
        statesp.latex_num_format, and statesp.latex_maxsize.

        The output is primarily intended for Jupyter notebooks, which
        use MathJax to render the LaTeX, and the results may look odd
        when processed by a 'conventional' LaTeX system.

        Returns
        -------
        s : str
            HTML/LaTeX representation of model, or None if either matrix
            dimension is greater than statesp.latex_maxsize.

        """
        syssize = self.nstates + max(self.noutputs, self.ninputs)
        if syssize > config.defaults['statesp.latex_maxsize']:
            return None
        elif config.defaults['statesp.latex_repr_type'] == 'partitioned':
            return super()._repr_info_(html=True) + \
                "\n" + self._latex_partitioned()
        elif config.defaults['statesp.latex_repr_type'] == 'separate':
            return super()._repr_info_(html=True) + \
                "\n" + self._latex_separate()
        else:
            raise ValueError(
                "Unknown statesp.latex_repr_type '{cfg}'".format(
                    cfg=config.defaults['statesp.latex_repr_type']))

    def _latex_partitioned_stateless(self):
        """`Partitioned` matrix LaTeX representation for stateless systems

        Model is presented as a matrix, D.  No partition lines are shown.

        Returns
        -------
        s : str
            LaTeX representation of model.

        """
        # Apply NumPy formatting
        with np.printoptions(threshold=sys.maxsize):
            D = eval(repr(self.D))

        lines = [
            r'$$',
            (r'\left['
             + r'\begin{array}'
             + r'{' + 'rll' * self.ninputs + '}')
            ]

        for Di in asarray(D):
            lines.append('&'.join(_f2s(Dij) for Dij in Di)
                         + '\\\\')

        lines.extend([
            r'\end{array}'
            r'\right]',
            r'$$'])

        return '\n'.join(lines)

    def _latex_partitioned(self):
        """Partitioned matrix LaTeX representation of state-space model

        Model is presented as a matrix partitioned into A, B, C, and D
        parts.

        Returns
        -------
        s : str
            LaTeX representation of model.

        """
        if self.nstates == 0:
            return self._latex_partitioned_stateless()

        # Apply NumPy formatting
        with np.printoptions(threshold=sys.maxsize):
            A, B, C, D = (
                eval(repr(getattr(self, M))) for M in ['A', 'B', 'C', 'D'])

        lines = [
            r'$$',
            (r'\left['
             + r'\begin{array}'
             + r'{' + 'rll' * self.nstates + '|' + 'rll' * self.ninputs + '}')
            ]

        for Ai, Bi in zip(asarray(A), asarray(B)):
            lines.append('&'.join([_f2s(Aij) for Aij in Ai]
                                  + [_f2s(Bij) for Bij in Bi])
                         + '\\\\')
        lines.append(r'\hline')
        for Ci, Di in zip(asarray(C), asarray(D)):
            lines.append('&'.join([_f2s(Cij) for Cij in Ci]
                                  + [_f2s(Dij) for Dij in Di])
                         + '\\\\')

        lines.extend([
            r'\end{array}'
            + r'\right]',
            r'$$'])

        return '\n'.join(lines)

    def _latex_separate(self):
        """Separate matrices LaTeX representation of state-space model

        Model is presented as separate, named, A, B, C, and D matrices.

        Returns
        -------
        s : str
            LaTeX representation of model.

        """
        lines = [
            r'$$',
            r'\begin{array}{ll}',
            ]

        def fmt_matrix(matrix, name):
            matlines = [name
                        + r' = \left[\begin{array}{'
                        + 'rll' * matrix.shape[1]
                        + '}']
            for row in asarray(matrix):
                matlines.append('&'.join(_f2s(entry) for entry in row)
                                + '\\\\')
            matlines.extend([
                r'\end{array}'
                r'\right]'])
            return matlines

        if self.nstates > 0:
            lines.extend(fmt_matrix(self.A, 'A'))
            lines.append('&')
            lines.extend(fmt_matrix(self.B, 'B'))
            lines.append('\\\\')

            lines.extend(fmt_matrix(self.C, 'C'))
            lines.append('&')
        lines.extend(fmt_matrix(self.D, 'D'))

        lines.extend([
            r'\end{array}',
            r'$$'])

        return '\n'.join(lines)

    # Negation of a system
    def __neg__(self):
        """Negate a state space system."""
        return StateSpace(self.A, self.B, -self.C, -self.D, self.dt)

    # Addition of two state space systems (parallel interconnection)
    def __add__(self, other):
        """Add two LTI systems (parallel connection)."""
        from .xferfcn import TransferFunction

        # Convert transfer functions to state space
        if isinstance(other, TransferFunction):
            # Convert the other argument to state space
            other = _convert_to_statespace(other)

        # Check for a couple of special cases
        if isinstance(other, (int, float, complex, np.number)):
            # Just adding a scalar; put it in the D matrix
            A, B, C = self.A, self.B, self.C
            D = self.D + other
            dt = self.dt

        elif isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            # Special case for SISO
            if self.issiso():
                self = np.ones_like(other) * self
            if self.ninputs != other.shape[0]:
                raise ValueError("array has incompatible shape")
            A, B, C = self.A, self.B, self.C
            D = self.D + other
            dt = self.dt

        elif not isinstance(other, StateSpace):
            return NotImplemented       # let other.__rmul__ handle it

        else:
            # Promote SISO object to compatible dimension
            if self.issiso() and not other.issiso():
                self = np.ones((other.noutputs, other.ninputs)) * self
            elif not self.issiso() and other.issiso():
                other = np.ones((self.noutputs, self.ninputs)) * other

            # Check to make sure the dimensions are OK
            if ((self.ninputs != other.ninputs) or
                    (self.noutputs != other.noutputs)):
                raise ValueError(
                    "can't add systems with incompatible inputs and outputs")

            dt = common_timebase(self.dt, other.dt)

            # Concatenate the various arrays
            A = concatenate((
                concatenate((self.A, zeros((self.A.shape[0],
                                            other.A.shape[-1]))), axis=1),
                concatenate((zeros((other.A.shape[0], self.A.shape[-1])),
                             other.A), axis=1)), axis=0)
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
        from .xferfcn import TransferFunction

        # Convert transfer functions to state space
        if isinstance(other, TransferFunction):
            # Convert the other argument to state space
            other = _convert_to_statespace(other)

        # Check for a couple of special cases
        if isinstance(other, (int, float, complex, np.number)):
            # Just multiplying by a scalar; change the output
            A, C = self.A, self.C
            B = self.B * other
            D = self.D * other
            dt = self.dt

        elif isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            # Special case for SISO
            if self.issiso():
                self = bdalg.append(*([self] * other.shape[0]))
            # Dimension check after broadcasting
            if self.ninputs != other.shape[0]:
                raise ValueError("array has incompatible shape")
            A, C = self.A, self.C
            B = self.B @ other
            D = self.D @ other
            dt = self.dt

        elif not isinstance(other, StateSpace):
            return NotImplemented       # let other.__rmul__ handle it

        else:
            # Promote SISO object to compatible dimension
            if self.issiso() and not other.issiso():
                self = bdalg.append(*([self] * other.noutputs))
            elif not self.issiso() and other.issiso():
                other = bdalg.append(*([other] * self.ninputs))

            # Check to make sure the dimensions are OK
            if self.ninputs != other.noutputs:
                raise ValueError(
                    "can't multiply systems with incompatible"
                    " inputs and outputs")
            dt = common_timebase(self.dt, other.dt)

            # Concatenate the various arrays
            A = concatenate(
                (concatenate((other.A,
                              zeros((other.A.shape[0], self.A.shape[1]))),
                             axis=1),
                 concatenate((self.B @ other.C, self.A), axis=1)),
                axis=0)
            B = concatenate((other.B, self.B @ other.D), axis=0)
            C = concatenate((self.D @ other.C, self.C), axis=1)
            D = self.D @ other.D

        return StateSpace(A, B, C, D, dt)

    # Right multiplication of two state space systems (series interconnection)
    # Just need to convert LH argument to a state space object
    def __rmul__(self, other):
        """Right multiply two LTI objects (serial connection)."""
        from .xferfcn import TransferFunction

        # Convert transfer functions to state space
        if isinstance(other, TransferFunction):
            # Convert the other argument to state space
            other = _convert_to_statespace(other)

        # Check for a couple of special cases
        if isinstance(other, (int, float, complex, np.number)):
            # Just multiplying by a scalar; change the input
            B = other * self.B
            D = other * self.D
            return StateSpace(self.A, B, self.C, D, self.dt)

        elif isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            # Special case for SISO transfer function
            if self.issiso():
                self = bdalg.append(*([self] * other.shape[1]))
            # Dimension check after broadcasting
            if self.noutputs != other.shape[1]:
                raise ValueError("array has incompatible shape")
            C = other @ self.C
            D = other @ self.D
            return StateSpace(self.A, self.B, C, D, self.dt)

        if not isinstance(other, StateSpace):
            return NotImplemented

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self] * other.ninputs))
        elif not self.issiso() and other.issiso():
            other = bdalg.append(*([other] * self.noutputs))

        return other * self

    # TODO: general __truediv__ requires descriptor system support
    def __truediv__(self, other):
        """Division of state space systems by TFs, FRDs, scalars, and arrays"""
        # Let ``other.__rtruediv__`` handle it
        try:
            return self * (1 / other)
        except ValueError:
            return NotImplemented

    def __rtruediv__(self, other):
        """Division by state space system"""
        return other * self**-1

    def __pow__(self, other):
        """Power of a state space system"""
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if self.ninputs != self.noutputs:
            # System must have same number of inputs and outputs
            return NotImplemented
        if other < -1:
            return (self**-1)**(-other)
        elif other == -1:
            try:
                Di = scipy.linalg.inv(self.D)
            except scipy.linalg.LinAlgError:
                # D matrix must be nonsingular
                return NotImplemented
            Ai = self.A - self.B @ Di @ self.C
            Bi = self.B @ Di
            Ci = -Di @ self.C
            return StateSpace(Ai, Bi, Ci, Di, self.dt)
        elif other == 0:
            return StateSpace([], [], [], np.eye(self.ninputs), self.dt)
        elif other == 1:
            return self
        elif other > 1:
            return self * (self**(other - 1))

    def __call__(self, x, squeeze=None, warn_infinite=True):
        """Evaluate system transfer function at point in complex plane.

        Returns the value of the system's transfer function at a point `x`
        in the complex plane, where `x` is `s` for continuous-time systems
        and `z` for discrete-time systems.

        See `LTI.__call__` for details.

        Examples
        --------
        >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])
        >>> fresp = G(1j)  # evaluate at s = 1j

        """
        # Use Slycot if available
        out = self.horner(x, warn_infinite=warn_infinite)
        return _process_frequency_response(self, x, out, squeeze=squeeze)

    def slycot_laub(self, x):
        """Laub's method to evaluate response at complex frequency.

        Evaluate transfer function at complex frequency using Laub's
        method from Slycot.  Expects inputs and outputs to be
        formatted correctly. Use ``sys(x)`` for a more user-friendly
        interface.

        Parameters
        ----------
        x : complex array_like or complex
            Complex frequency.

        Returns
        -------
        output : (number_outputs, number_inputs, len(x)) complex ndarray
            Frequency response.

        """
        from slycot import tb05ad

        # Make sure the argument is a 1D array of complex numbers
        x_arr = np.atleast_1d(x).astype(complex, copy=False)

        # Make sure that we are operating on a simple list
        if len(x_arr.shape) > 1:
            raise ValueError("input list must be 1D")

        # preallocate
        n = self.nstates
        m = self.ninputs
        p = self.noutputs
        out = np.empty((p, m, len(x_arr)), dtype=complex)
        # The first call both evaluates C(sI-A)^-1 B and also returns
        # Hessenberg transformed matrices at, bt, ct.
        result = tb05ad(n, m, p, x_arr[0], self.A, self.B, self.C, job='NG')
        # When job='NG', result = (at, bt, ct, g_i, hinvb, info)
        at = result[0]
        bt = result[1]
        ct = result[2]

        # TB05AD frequency evaluation does not include direct feedthrough.
        out[:, :, 0] = result[3] + self.D

        # Now, iterate through the remaining frequencies using the
        # transformed state matrices, at, bt, ct.

        # Start at the second frequency, already have the first.
        for kk, x_kk in enumerate(x_arr[1:]):
            result = tb05ad(n, m, p, x_kk, at, bt, ct, job='NH')
            # When job='NH', result = (g_i, hinvb, info)

            # kk+1 because enumerate starts at kk = 0.
            # but zero-th spot is already filled.
            out[:, :, kk+1] = result[0] + self.D
        return out

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

        Notes
        -----
        Attempts to use Laub's method from Slycot library, with a fall-back
        to Python code.

        """
        # Make sure the argument is a 1D array of complex numbers
        x_arr = np.atleast_1d(x).astype(complex, copy=False)

        # return fast on systems with 0 or 1 state
        if self.nstates == 0:
            return self.D[:, :, np.newaxis] \
                * np.ones_like(x_arr, dtype=complex)
        elif self.nstates == 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                out = self.C[:, :, np.newaxis] \
                    / (x_arr - self.A[0, 0]) \
                    * self.B[:, :, np.newaxis] \
                    + self.D[:, :, np.newaxis]
            out[np.isnan(out)] = complex(np.inf, np.nan)
            return out

        try:
            out = self.slycot_laub(x_arr)
        except (ImportError, Exception):
            # Fall back because either Slycot unavailable or cannot handle
            # certain cases.

            # Make sure that we are operating on a simple list
            if len(x_arr.shape) > 1:
                raise ValueError("input list must be 1D")

            # Preallocate
            out = empty((self.noutputs, self.ninputs, len(x_arr)),
                        dtype=complex)

            # TODO: can this be vectorized?
            for idx, x_idx in enumerate(x_arr):
                try:
                    xr = solve(x_idx * eye(self.nstates) - self.A, self.B)
                    out[:, :, idx] = self.C @ xr + self.D
                except LinAlgError:
                    # Issue a warning message, for consistency with xferfcn
                    if warn_infinite:
                        warn("singular matrix in frequency response",
                             RuntimeWarning)

                    # Evaluating at a pole.  Return value depends if there
                    # is a zero at the same point or not.
                    if x_idx in self.zeros():
                        out[:, :, idx] = complex(np.nan, np.nan)
                    else:
                        out[:, :, idx] = complex(np.inf, np.nan)

        return out

    def freqresp(self, omega):
        """(deprecated) Evaluate transfer function at complex frequencies.

        .. deprecated::0.9.0
            Method has been given the more Pythonic name
            `StateSpace.frequency_response`. Or use
            `freqresp` in the MATLAB compatibility module.
        """
        warn("StateSpace.freqresp(omega) will be removed in a "
             "future release of python-control; use "
             "sys.frequency_response(omega), or freqresp(sys, omega) in the "
             "MATLAB compatibility module instead", FutureWarning)
        return self.frequency_response(omega)

    # Compute poles and zeros
    def poles(self):
        """Compute the poles of a state space system."""

        return eigvals(self.A).astype(complex) if self.nstates \
            else np.array([])

    def zeros(self):
        """Compute the zeros of a state space system."""

        if not self.nstates:
            return np.array([])

        # Use AB08ND from Slycot if it's available, otherwise use
        # scipy.lingalg.eigvals().
        try:
            from slycot import ab08nd

            out = ab08nd(self.A.shape[0], self.B.shape[1], self.C.shape[0],
                         self.A, self.B, self.C, self.D)
            nu = out[0]
            if nu == 0:
                return np.array([])
            else:
                # Use SciPy generalized eigenvalue function
                return sp.linalg.eigvals(out[8][0:nu, 0:nu],
                                         out[9][0:nu, 0:nu]).astype(complex)

        except ImportError:  # Slycot unavailable. Fall back to SciPy.
            if self.C.shape[0] != self.D.shape[1]:
                raise NotImplementedError(
                    "StateSpace.zero only supports systems with the same "
                    "number of inputs as outputs.")

            # This implements the QZ algorithm for finding transmission zeros
            # from
            # https://dspace.mit.edu/bitstream/handle/1721.1/841/P-0802-06587335.pdf.
            # The QZ algorithm solves the generalized eigenvalue problem: given
            # `L = [A, B; C, D]` and `M = [I_nxn 0]`, find all finite lambda
            # for which there exist nontrivial solutions of the equation
            # `Lz - lamba Mz`.
            #
            # The generalized eigenvalue problem is only solvable if its
            # arguments are square matrices.
            L = concatenate((concatenate((self.A, self.B), axis=1),
                             concatenate((self.C, self.D), axis=1)), axis=0)
            M = pad(eye(self.A.shape[0]), ((0, self.C.shape[0]),
                                           (0, self.B.shape[1])), "constant")
            return np.array([x for x in sp.linalg.eigvals(L, M,
                                                          overwrite_a=True)
                             if not isinf(x)], dtype=complex)

    # Feedback around a state space system
    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two LTI objects.

        Parameters
        ----------
        other : `InputOutputSystem`
            System in the feedback path.

        sign : float, optional
            Gain to use in feedback path.  Defaults to -1.

        """
        # Convert the system to state space, if possible
        try:
            other = _convert_to_statespace(other)
        except:
            pass

        if not isinstance(other, StateSpace):
            return NonlinearIOSystem.feedback(self, other, sign)

        # Check to make sure the dimensions are OK
        if self.ninputs != other.noutputs or self.noutputs != other.ninputs:
            raise ValueError("State space systems don't have compatible "
                             "inputs/outputs for feedback.")
        dt = common_timebase(self.dt, other.dt)

        A1 = self.A
        B1 = self.B
        C1 = self.C
        D1 = self.D
        A2 = other.A
        B2 = other.B
        C2 = other.C
        D2 = other.D

        F = eye(self.ninputs) - sign * D2 @ D1
        if matrix_rank(F) != self.ninputs:
            raise ValueError(
                "I - sign * D2 * D1 is singular to working precision.")

        # Precompute F\D2 and F\C2 (E = inv(F))
        # We can solve two linear systems in one pass, since the
        # coefficients matrix F is the same. Thus, we perform the LU
        # decomposition (cubic runtime complexity) of F only once!
        # The remaining back substitutions are only quadratic in runtime.
        E_D2_C2 = solve(F, concatenate((D2, C2), axis=1))
        E_D2 = E_D2_C2[:, :other.ninputs]
        E_C2 = E_D2_C2[:, other.ninputs:]

        T1 = eye(self.noutputs) + sign * D1 @ E_D2
        T2 = eye(self.ninputs) + sign * E_D2 @ D1

        A = concatenate(
            (concatenate(
                (A1 + sign * B1 @ E_D2 @ C1,
                 sign * B1 @ E_C2), axis=1),
             concatenate(
                 (B2 @ T1 @ C1,
                  A2 + sign * B2 @ D1 @ E_C2), axis=1)),
            axis=0)
        B = concatenate((B1 @ T2, B2 @ D1 @ T2), axis=0)
        C = concatenate((T1 @ C1, sign * D1 @ E_C2), axis=1)
        D = D1 @ T2

        return StateSpace(A, B, C, D, dt)

    def lft(self, other, nu=-1, ny=-1):
        """Return the linear fractional transformation.

        A definition of the LFT operator can be found in Appendix A.7,
        page 512 in [1]_.  An alternative definition can be found here:
        https://www.mathworks.com/help/control/ref/lft.html

        Parameters
        ----------
        other : `StateSpace`
            The lower LTI system.
        ny : int, optional
            Dimension of (plant) measurement output.
        nu : int, optional
            Dimension of (plant) control input.

        Returns
        -------
        `StateSpace`

        References
        ----------
        .. [1] S. Skogestad, Multivariable Feedback Control.  Second
           edition, 2005.

        """
        other = _convert_to_statespace(other)
        # maximal values for nu, ny
        if ny == -1:
            ny = min(other.ninputs, self.noutputs)
        if nu == -1:
            nu = min(other.noutputs, self.ninputs)
        # dimension check
        # TODO

        dt = common_timebase(self.dt, other.dt)

        # submatrices
        A = self.A
        B1 = self.B[:, :self.ninputs - nu]
        B2 = self.B[:, self.ninputs - nu:]
        C1 = self.C[:self.noutputs - ny, :]
        C2 = self.C[self.noutputs - ny:, :]
        D11 = self.D[:self.noutputs - ny, :self.ninputs - nu]
        D12 = self.D[:self.noutputs - ny, self.ninputs - nu:]
        D21 = self.D[self.noutputs - ny:, :self.ninputs - nu]
        D22 = self.D[self.noutputs - ny:, self.ninputs - nu:]

        # submatrices
        Abar = other.A
        Bbar1 = other.B[:, :ny]
        Bbar2 = other.B[:, ny:]
        Cbar1 = other.C[:nu, :]
        Cbar2 = other.C[nu:, :]
        Dbar11 = other.D[:nu, :ny]
        Dbar12 = other.D[:nu, ny:]
        Dbar21 = other.D[nu:, :ny]
        Dbar22 = other.D[nu:, ny:]

        # well-posed check
        F = np.block([[np.eye(ny), -D22], [-Dbar11, np.eye(nu)]])
        if matrix_rank(F) != ny + nu:
            raise ValueError("LFT not well-posed to working precision.")

        # solve for the resulting ss by solving for [y, u] using [x,
        # xbar] and [w1, w2].
        TH = np.linalg.solve(F, np.block(
            [[C2, np.zeros((ny, other.nstates)),
              D21, np.zeros((ny, other.ninputs - ny))],
             [np.zeros((nu, self.nstates)), Cbar1,
              np.zeros((nu, self.ninputs - nu)), Dbar12]]
        ))
        T11 = TH[:ny, :self.nstates]
        T12 = TH[:ny, self.nstates: self.nstates + other.nstates]
        T21 = TH[ny:, :self.nstates]
        T22 = TH[ny:, self.nstates: self.nstates + other.nstates]
        H11 = TH[:ny, self.nstates + other.nstates:self.nstates +
                 other.nstates + self.ninputs - nu]
        H12 = TH[:ny, self.nstates + other.nstates + self.ninputs - nu:]
        H21 = TH[ny:, self.nstates + other.nstates:self.nstates +
                 other.nstates + self.ninputs - nu]
        H22 = TH[ny:, self.nstates + other.nstates + self.ninputs - nu:]

        Ares = np.block([
            [A + B2 @ T21, B2 @ T22],
            [Bbar1 @ T11, Abar + Bbar1 @ T12]
        ])

        Bres = np.block([
            [B1 + B2 @ H21, B2 @ H22],
            [Bbar1 @ H11, Bbar2 + Bbar1 @ H12]
        ])

        Cres = np.block([
            [C1 + D12 @ T21, D12 @ T22],
            [Dbar21 @ T11, Cbar2 + Dbar21 @ T12]
        ])

        Dres = np.block([
            [D11 + D12 @ H21, D12 @ H22],
            [Dbar21 @ H11, Dbar22 + Dbar21 @ H12]
        ])
        return StateSpace(Ares, Bres, Cres, Dres, dt)

    def minreal(self, tol=0.0):
        """Remove unobservable and uncontrollable states.

        Calculate a minimal realization for a state space system,
        removing all unobservable and/or uncontrollable states.

        Parameters
        ----------
        tol : float
            Tolerance for determining whether states are unobservable
            or uncontrollable.

        """
        if self.nstates:
            try:
                from slycot import tb01pd
                B = empty((self.nstates, max(self.ninputs, self.noutputs)))
                B[:, :self.ninputs] = self.B
                C = empty((max(self.noutputs, self.ninputs), self.nstates))
                C[:self.noutputs, :] = self.C
                A, B, C, nr = tb01pd(self.nstates, self.ninputs, self.noutputs,
                                     self.A, B, C, tol=tol)
                return StateSpace(A[:nr, :nr], B[:nr, :self.ninputs],
                                  C[:self.noutputs, :nr], self.D, self.dt)
            except ImportError:
                raise TypeError("minreal requires slycot tb01pd")
        else:
            return StateSpace(self)

    def returnScipySignalLTI(self, strict=True):
        """Return a list of a list of `scipy.signal.lti` objects.

        For instance,

        >>> out = ssobject.returnScipySignalLTI()               # doctest: +SKIP
        >>> out[3][5]                                           # doctest: +SKIP

        is a `scipy.signal.lti` object corresponding to the transfer
        function from the 6th input to the 4th output.

        Parameters
        ----------
        strict : bool, optional
            True (default):
                The timebase `ssobject.dt` cannot be None; it must
                be continuous (0) or discrete (True or > 0).
            False:
              If `ssobject.dt` is None, continuous-time
              `scipy.signal.lti` objects are returned.

        Returns
        -------
        out : list of list of `scipy.signal.StateSpace`
            Continuous time (inheriting from `scipy.signal.lti`)
            or discrete time (inheriting from `scipy.signal.dlti`)
            SISO objects.

        """
        if strict and self.dt is None:
            raise ValueError("with strict=True, dt cannot be None")

        if self.dt:
            kwdt = {'dt': self.dt}
        else:
            # SciPy convention for continuous-time LTI systems: call without
            # dt keyword argument
            kwdt = {}

        # Preallocate the output.
        out = [[[] for _ in range(self.ninputs)] for _ in range(self.noutputs)]

        for i in range(self.noutputs):
            for j in range(self.ninputs):
                out[i][j] = signalStateSpace(asarray(self.A),
                                             asarray(self.B[:, j:j + 1]),
                                             asarray(self.C[i:i + 1, :]),
                                             asarray(self.D[i:i + 1, j:j + 1]),
                                             **kwdt)

        return out

    def append(self, other):
        """Append a second model to the present model.

        The second model is converted to state-space if necessary, inputs and
        outputs are appended and their order is preserved.

        Parameters
        ----------
        other : `StateSpace` or `TransferFunction`
            System to be appended.

        Returns
        -------
        sys : `StateSpace`
            System model with `other` appended to `self`.

        """
        if not isinstance(other, StateSpace):
            other = _convert_to_statespace(other)

        self.dt = common_timebase(self.dt, other.dt)

        n = self.nstates + other.nstates
        m = self.ninputs + other.ninputs
        p = self.noutputs + other.noutputs
        A = zeros((n, n))
        B = zeros((n, m))
        C = zeros((p, n))
        D = zeros((p, m))
        A[:self.nstates, :self.nstates] = self.A
        A[self.nstates:, self.nstates:] = other.A
        B[:self.nstates, :self.ninputs] = self.B
        B[self.nstates:, self.ninputs:] = other.B
        C[:self.noutputs, :self.nstates] = self.C
        C[self.noutputs:, self.nstates:] = other.C
        D[:self.noutputs, :self.ninputs] = self.D
        D[self.noutputs:, self.ninputs:] = other.D
        return StateSpace(A, B, C, D, self.dt)

    def __getitem__(self, key):
        """Array style access"""
        if not isinstance(key, Iterable) or len(key) != 2:
            raise IOError("must provide indices of length 2 for state space")

        # Convert signal names to integer offsets
        iomap = NamedSignal(self.D, self.output_labels, self.input_labels)
        indices = iomap._parse_key(key, level=1)  # ignore index checks
        outdx, output_labels = _process_subsys_index(
            indices[0], self.output_labels)
        inpdx, input_labels = _process_subsys_index(
            indices[1], self.input_labels)

        sysname = config.defaults['iosys.indexed_system_name_prefix'] + \
            self.name + config.defaults['iosys.indexed_system_name_suffix']
        return StateSpace(
            self.A, self.B[:, inpdx], self.C[outdx, :],
            self.D[outdx, :][:, inpdx], self.dt,
            name=sysname, inputs=input_labels, outputs=output_labels)

    def sample(self, Ts, method='zoh', alpha=None, prewarp_frequency=None,
               name=None, copy_names=True, **kwargs):
        """Convert a continuous-time system to discrete time.

        Creates a discrete-time system from a continuous-time system by
        sampling.  Multiple methods of conversion are supported.

        Parameters
        ----------
        Ts : float
            Sampling period.
        method : {'gbt', 'bilinear', 'euler', 'backward_diff', 'zoh'}
            Method to use for sampling:

            * 'gbt': generalized bilinear transformation
            * 'backward_diff': Backwards difference ('gbt' with alpha=1.0)
            * 'bilinear' (or 'tustin'): Tustin's approximation ('gbt' with
              alpha=0.5)
            * 'euler': Euler (or forward difference) method ('gbt' with
              alpha=0)
            * 'zoh': zero-order hold (default)
        alpha : float within [0, 1]
            The generalized bilinear transformation weighting parameter,
            which should only be specified with method='gbt', and is
            ignored otherwise.
        prewarp_frequency : float within [0, infinity)
            The frequency [rad/s] at which to match with the input
            continuous-time system's magnitude and phase (the gain = 1
            crossover frequency, for example). Should only be specified
            with `method` = 'bilinear' or 'gbt' with `alpha` = 0.5 and
            ignored otherwise.
        name : string, optional
            Set the name of the sampled system.  If not specified and if
            `copy_names` is False, a generic name 'sys[id]' is
            generated with a unique integer id.  If `copy_names` is
            True, the new system name is determined by adding the
            prefix and suffix strings in
            `config.defaults['iosys.sampled_system_name_prefix']` and
            `config.defaults['iosys.sampled_system_name_suffix']`, with
            the default being to add the suffix '$sampled'.
        copy_names : bool, Optional
            If True, copy the names of the input signals, output
            signals, and states to the sampled system.

        Returns
        -------
        sysd : `StateSpace`
            Discrete-time system, with sampling rate `Ts`.

        Other Parameters
        ----------------
        inputs : int, list of str or None, optional
            Description of the system inputs.  If not specified, the
            original system inputs are used.  See `InputOutputSystem` for
            more information.
        outputs : int, list of str or None, optional
            Description of the system outputs.  Same format as `inputs`.
        states : int, list of str, or None, optional
            Description of the system states.  Same format as `inputs`.

        Notes
        -----
        Uses `scipy.signal.cont2discrete`.

        Examples
        --------
        >>> G = ct.ss(0, 1, 1, 0)
        >>> sysd = G.sample(0.5, method='bilinear')

        """
        if not self.isctime():
            raise ValueError("System must be continuous-time system")
        if prewarp_frequency is not None:
            if method in ('bilinear', 'tustin') or \
                    (method == 'gbt' and alpha == 0.5):
                Twarp = 2*np.tan(prewarp_frequency*Ts/2)/prewarp_frequency
            else:
                warn('prewarp_frequency ignored: incompatible conversion')
                Twarp = Ts
        else:
            Twarp = Ts
        sys = (self.A, self.B, self.C, self.D)
        Ad, Bd, C, D, _ = cont2discrete(sys, Twarp, method, alpha)
        sysd = StateSpace(Ad, Bd, C, D, Ts)
        # copy over the system name, inputs, outputs, and states
        if copy_names:
            sysd._copy_names(self, prefix_suffix_name='sampled')
            if name is not None:
                sysd.name = name
        # pass desired signal names if names were provided
        return StateSpace(sysd, **kwargs)

    def dcgain(self, warn_infinite=False):
        """Return the zero-frequency ("DC") gain.

        The zero-frequency gain of a continuous-time state-space
        system is given by:

        .. math: G(0) = - C A^{-1} B + D

        and of a discrete-time state-space system by:

        .. math: G(1) = C (I - A)^{-1} B + D

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

        """
        return self._dcgain(warn_infinite)

    # TODO: decide if we need this function (already in NonlinearIOSystem
    def dynamics(self, t, x, u=None, params=None):
        """Compute the dynamics of the system.

        Given input `u` and state `x`, returns the dynamics of the state-space
        system. If the system is continuous, returns the time derivative dx/dt

            dx/dt = A x + B u

        where A and B are the state-space matrices of the system. If the
        system is discrete time, returns the next value of `x`:

            x[t+dt] = A x[t] + B u[t]

        The inputs `x` and `u` must be of the correct length for the system.

        The first argument `t` is ignored because `StateSpace` systems
        are time-invariant. It is included so that the dynamics can be passed
        to numerical integrators, such as `scipy.integrate.solve_ivp`
        and for consistency with `InputOutputSystem` models.

        Parameters
        ----------
        t : float (ignored)
            Time.
        x : array_like
            Current state.
        u : array_like (optional)
            Input, zero if omitted.

        Returns
        -------
        dx/dt or x[t+dt] : ndarray

        """
        if params is not None:
            warn("params keyword ignored for StateSpace object")

        x = np.reshape(x, (-1, 1))  # force to a column in case matrix
        if np.size(x) != self.nstates:
            raise ValueError("len(x) must be equal to number of states")
        if u is None:
            return (self.A @ x).reshape((-1,))  # return as row vector
        else:  # received t, x, and u, ignore t
            u = np.reshape(u, (-1, 1))  # force to column in case matrix
            if np.size(u) != self.ninputs:
                raise ValueError("len(u) must be equal to number of inputs")
            return (self.A @ x).reshape((-1,)) \
                + (self.B @ u).reshape((-1,))  # return as row vector

    # TODO: decide if we need this function (already in NonlinearIOSystem
    def output(self, t, x, u=None, params=None):
        """Compute the output of the system.

        Given input `u` and state `x`, returns the output `y` of the
        state-space system:

            y = C x + D u

        where A and B are the state-space matrices of the system.

        The first argument `t` is ignored because `StateSpace` systems
        are time-invariant. It is included so that the dynamics can be passed
        to most numerical integrators, such as SciPy's `integrate.solve_ivp`
        and for consistency with `InputOutputSystem` models.

        The inputs `x` and `u` must be of the correct length for the system.

        Parameters
        ----------
        t : float (ignored)
            Time.
        x : array_like
            Current state.
        u : array_like (optional)
            Input (zero if omitted).

        Returns
        -------
        y : ndarray

        """
        if params is not None:
            warn("params keyword ignored for StateSpace object")

        x = np.reshape(x, (-1, 1))  # force to a column in case matrix
        if np.size(x) != self.nstates:
            raise ValueError("len(x) must be equal to number of states")

        if u is None:
            return (self.C @ x).reshape((-1,))  # return as row vector
        else:  # received t, x, and u, ignore t
            u = np.reshape(u, (-1, 1))  # force to a column in case matrix
            if np.size(u) != self.ninputs:
                raise ValueError("len(u) must be equal to number of inputs")
            return (self.C @ x).reshape((-1,)) \
                + (self.D @ u).reshape((-1,))  # return as row vector

    # convenience alias, import needs submodule to avoid circular imports
    initial_response = control.timeresp.initial_response


class LinearICSystem(InterconnectedSystem, StateSpace):
    """Interconnection of a set of linear input/output systems.

    This class is used to implement a system that is an interconnection of
    linear input/output systems.  It has all of the structure of an
    `InterconnectedSystem`, but also maintains the required
    elements of the `StateSpace` class structure, allowing it to be
    passed to functions that expect a `StateSpace` system.

    This class is generated using `interconnect` and
    not called directly.

    """

    def __init__(self, io_sys, ss_sys=None, connection_type=None):
        #
        # Because this is a "hybrid" object, the initialization proceeds in
        # stages.  We first create an empty InputOutputSystem of the
        # appropriate size, then copy over the elements of the
        # InterconnectedSystem class.  From there we compute the
        # linearization of the system (if needed) and then populate the
        # StateSpace parameters.
        #
        # Create the (essentially empty) I/O system object
        InputOutputSystem.__init__(
            self, name=io_sys.name, inputs=io_sys.ninputs,
            outputs=io_sys.noutputs, states=io_sys.nstates, dt=io_sys.dt)

        # Copy over the attributes from the interconnected system
        self.syslist = io_sys.syslist
        self.syslist_index = io_sys.syslist_index
        self.state_offset = io_sys.state_offset
        self.input_offset = io_sys.input_offset
        self.output_offset = io_sys.output_offset
        self.connect_map = io_sys.connect_map
        self.input_map = io_sys.input_map
        self.output_map = io_sys.output_map
        self.params = io_sys.params
        self.connection_type = connection_type

        # If we didn't' get a state space system, linearize the full system
        if ss_sys is None:
            ss_sys = self.linearize(0, 0)

        # Initialize the state space object
        StateSpace.__init__(
            self, ss_sys, name=io_sys.name, inputs=io_sys.input_labels,
            outputs=io_sys.output_labels, states=io_sys.state_labels,
            params=io_sys.params, remove_useless_states=False)

    # Use StateSpace.__call__ to evaluate at a given complex value
    def __call__(self, *args, **kwargs):
        return StateSpace.__call__(self, *args, **kwargs)

    def __str__(self):
        string = InterconnectedSystem.__str__(self) + "\n\n"
        string += "\n\n".join([
            "{} = {}".format(Mvar,
                               "\n    ".join(str(M).splitlines()))
            for Mvar, M in zip(["A", "B", "C", "D"],
                               [self.A, self.B, self.C, self.D])])
        return string

    # Use InputOutputSystem repr for 'eval' since we can't recreate structure
    # (without this, StateSpace._repr_eval_ gets used...)
    def _repr_eval_(self):
        return InputOutputSystem._repr_eval_(self)

    def _repr_html_(self):
        syssize = self.nstates + max(self.noutputs, self.ninputs)
        if syssize > config.defaults['statesp.latex_maxsize']:
            return None
        elif config.defaults['statesp.latex_repr_type'] == 'partitioned':
            return InterconnectedSystem._repr_info_(self, html=True) + \
                "\n" + StateSpace._latex_partitioned(self)
        elif config.defaults['statesp.latex_repr_type'] == 'separate':
            return InterconnectedSystem._repr_info_(self, html=True) + \
                "\n" + StateSpace._latex_separate(self)
        else:
            raise ValueError(
                "Unknown statesp.latex_repr_type '{cfg}'".format(
                    cfg=config.defaults['statesp.latex_repr_type']))

    # The following text needs to be replicated from StateSpace in order for
    # this entry to show up properly in sphinx documentation (not sure why,
    # but it was the only way to get it to work).
    #
    #: Deprecated attribute; use `nstates` instead.
    #:
    #: The `state` attribute was used to store the number of states for : a
    #: state space system.  It is no longer used.  If you need to access the
    #: number of states, use `nstates`.
    states = property(StateSpace._get_states, StateSpace._set_states)


# Define a state space object that is an I/O system
def ss(*args, **kwargs):
    r"""ss(A, B, C, D[, dt])

    Create a state space system.

    The function accepts either 1, 4 or 5 positional parameters:

    ``ss(sys)``

        Convert a linear system into space system form. Always creates a
        new system, even if `sys` is already a state space system.

    ``ss(A, B, C, D)``

        Create a state space system from the matrices of its state and
        output equations:

        .. math::

            dx/dt &= A x + B u \\
                y &= C x + D  u

    ``ss(A, B, C, D, dt)``

        Create a discrete-time state space system from the matrices of
        its state and output equations:

        .. math::

            x[k+1] &= A x[k] + B u[k] \\
              y[k] &= C x[k] + D u[k]

        The matrices can be given as 2D array_like data types.  For SISO
        systems, `B` and `C` can be given as 1D arrays and D can be given
        as a scalar.


    ``ss(*args, inputs=['u1', ..., 'up'], outputs=['y1', ..., 'yq'], states=['x1', ..., 'xn'])``
        Create a system with named input, output, and state signals.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        A linear system.
    A, B, C, D : array_like or string
        System, control, output, and feed forward matrices.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete time).
    remove_useless_states : bool, optional
        If True, remove states that have no effect on the input/output
        dynamics.  If not specified, the value is read from
        `config.defaults['statesp.remove_useless_states']` (default = False).
    method : str, optional
        Set the method used for converting a transfer function to a state
        space system.  Current methods are 'slycot' and 'scipy'.  If set to
        None (default), try 'slycot' first and then 'scipy' (SISO only).

    Returns
    -------
    out : `StateSpace`
        Linear input/output system.

    Other Parameters
    ----------------
    inputs, outputs, states : str, or list of str, optional
        List of strings that name the individual signals.  If this parameter
        is not given or given as None, the signal names will be of the
        form 's[i]' (where 's' is one of 'u', 'y', or 'x'). See
        `InputOutputSystem` for more information.
    input_prefix, output_prefix, state_prefix : string, optional
        Set the prefix for input, output, and state signals.  Defaults =
        'u', 'y', 'x'.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Raises
    ------
    ValueError
        If matrix sizes are not self-consistent.

    See Also
    --------
    StateSpace, nlsys, tf, ss2tf, tf2ss, zpk

    Notes
    -----
    If a transfer function is passed as the sole positional argument, the
    system will be converted to state space form in the same way as calling
    `tf2ss`.  The `method` keyword can be used to select the
    method for conversion.

    Examples
    --------
    Create a linear I/O system object from matrices:

    >>> G = ct.ss([[-1, -2], [3, -4]], [[5], [7]], [[6, 8]], [[9]])

    Convert a transfer function to a state space system:

    >>> sys_tf = ct.tf([2.], [1., 3])
    >>> sys2 = ct.ss(sys_tf)

    """
    # See if this is a nonlinear I/O system (legacy usage)
    if len(args) > 0 and (hasattr(args[0], '__call__') or args[0] is None) \
       and not isinstance(args[0], (InputOutputSystem, LTI)):
        # Function as first (or second) argument => assume nonlinear IO system
        warn("using ss() to create nonlinear I/O systems is deprecated; "
             "use nlsys()", FutureWarning)
        return NonlinearIOSystem(*args, **kwargs)

    elif len(args) == 4 or len(args) == 5:
        # Create a state space function from A, B, C, D[, dt]
        sys = StateSpace(*args, **kwargs)

    elif len(args) == 1:
        sys = args[0]
        if isinstance(sys, LTI):
            # Check for system with no states and specified state names
            if sys.nstates is None and 'states' in kwargs:
                warn("state labels specified for "
                     "non-unique state space realization")

            # Allow method to be specified (e.g., tf2ss)
            method = kwargs.pop('method', None)

            # Create a state space system from an LTI system
            sys = StateSpace(
                _convert_to_statespace(
                    sys, method=method,
                    use_prefix_suffix=not sys._generic_name_check()),
                **kwargs)

        else:
            raise TypeError("ss(sys): sys must be a StateSpace or "
                            "TransferFunction object.  It is %s." % type(sys))
    else:
        raise TypeError(
            "Needs 1, 4, or 5 arguments; received %i." % len(args))

    return sys


# Convert a state space system into an input/output system (wrapper)
def ss2io(*args, **kwargs):
    """ss2io(sys[, ...])

    Create an I/O system from a state space linear system.

    .. deprecated:: 0.10.0
        This function will be removed in a future version of python-control.
        The `ss` function can be used directly to produce an I/O system.

    Create an `StateSpace` system with the given signal
    and system names.  See `ss` for more details.
    """
    warn("ss2io() is deprecated; use ss()", FutureWarning)
    return StateSpace(*args, **kwargs)


# Convert a transfer function into an input/output system (wrapper)
def tf2io(*args, **kwargs):
    """tf2io(sys[, ...])

    Convert a transfer function into an I/O system.

    .. deprecated:: 0.10.0
        This function will be removed in a future version of python-control.
        The `tf2ss` function can be used to produce a state space I/O system.

    The function accepts either 1 or 2 parameters:

    ``tf2io(sys)``

        Convert a linear system into space space form. Always creates
        a new system, even if `sys` is already a `StateSpace` object.

    ``tf2io(num, den)``

        Create a linear I/O system from its numerator and denominator
        polynomial coefficients.

        For details see: `tf`.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        A linear system.
    num : array_like, or list of list of array_like
        Polynomial coefficients of the numerator.
    den : array_like, or list of list of array_like
        Polynomial coefficients of the denominator.

    Returns
    -------
    out : `StateSpace`
        New I/O system (in state space form).

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
        If `num` and `den` have invalid or unequal dimensions, or if an
        invalid number of arguments is passed in.
    TypeError
        If `num` or `den` are of incorrect type, or if `sys` is not a
        `TransferFunction` object.

    See Also
    --------
    ss2io, tf2ss

    Examples
    --------
    >>> num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    >>> den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
    >>> sys1 = ct.tf2ss(num, den)

    >>> sys_tf = ct.tf(num, den)
    >>> G = ct.tf2ss(sys_tf)
    >>> G.ninputs, G.noutputs, G.nstates
    (2, 2, 8)

    """
    warn("tf2io() is deprecated; use tf2ss() or tf()", FutureWarning)
    return tf2ss(*args, **kwargs)


def tf2ss(*args, **kwargs):
    """tf2ss(sys)

    Transform a transfer function to a state space system.

    The function accepts either 1 or 2 parameters:

    ``tf2ss(sys)``

        Convert a transfer function into space space form.  Equivalent to
        `ss(sys)`.

    ``tf2ss(num, den)``

        Create a state space system from its numerator and denominator
        polynomial coefficients.

        For details see: `tf`.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        A linear system.
    num : array_like, or list of list of array_like
        Polynomial coefficients of the numerator.
    den : array_like, or list of list of array_like
        Polynomial coefficients of the denominator.

    Returns
    -------
    out : `StateSpace`
        New linear system in state space form.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    name : string, optional
        System name. If unspecified, a generic name 'sys[id]' is generated
        with a unique integer id.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'.  If set to None (default), try 'slycot'
        first and then 'scipy' (SISO only).

    Raises
    ------
    ValueError
        If `num` and `den` have invalid or unequal dimensions, or if an
        invalid number of arguments is passed in.
    TypeError
        If `num` or `den` are of incorrect type, or if `sys` is not a
        `TransferFunction` object.

    See Also
    --------
    ss, tf, ss2tf

    Notes
    -----
    The `slycot` routine used to convert a transfer function into state space
    form appears to have a bug and in some (rare) instances may not return
    a system with the same poles as the input transfer function.  For SISO
    systems, setting `method` = 'scipy' can be used as an alternative.

    Examples
    --------
    >>> num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    >>> den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
    >>> sys1 = ct.tf2ss(num, den)

    >>> sys_tf = ct.tf(num, den)
    >>> sys2 = ct.tf2ss(sys_tf)

    """

    from .xferfcn import TransferFunction
    if len(args) == 2 or len(args) == 3:
        # Assume we were given the num, den
        return StateSpace(
            _convert_to_statespace(TransferFunction(*args)), **kwargs)

    elif len(args) == 1:
        return ss(*args, **kwargs)

    else:
        raise ValueError("Needs 1 or 2 arguments; received %i." % len(args))


def ssdata(sys):
    """
    Return state space data objects for a system.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
        LTI system whose data will be returned.

    Returns
    -------
    A, B, C, D : ndarray
        State space data for the system.

    """
    ss = _convert_to_statespace(sys)
    return ss.A, ss.B, ss.C, ss.D


# TODO: combine with sysnorm?
def linfnorm(sys, tol=1e-10):
    """L-infinity norm of a linear system.

    Parameters
    ----------
    sys : `StateSpace` or `TransferFunction`
      System to evaluate L-infinity norm of.
    tol : real scalar
      Tolerance on norm estimate.

    Returns
    -------
    gpeak : non-negative scalar
      L-infinity norm.
    fpeak : non-negative scalar
      Frequency, in rad/s, at which gpeak occurs.

    See Also
    --------
    slycot.ab13dd

    Notes
    -----
    For stable systems, the L-infinity and H-infinity norms are equal;
    for unstable systems, the H-infinity norm is infinite, while the
    L-infinity norm is finite if the system has no poles on the
    imaginary axis.

    """
    if ab13dd is None:
        raise ControlSlycot("Can't find slycot module ab13dd")

    a, b, c, d = ssdata(_convert_to_statespace(sys))
    e = np.eye(a.shape[0])

    n = a.shape[0]
    m = b.shape[1]
    p = c.shape[0]

    if n == 0:
        # ab13dd doesn't accept empty A, B, C, D;
        # static gain case is easy enough to compute
        gpeak = scipy.linalg.svdvals(d)[0]
        # max SVD is constant with freq; arbitrarily choose 0 as peak
        fpeak = 0
        return gpeak, fpeak

    dico = 'C' if sys.isctime() else 'D'
    jobe = 'I'
    equil = 'S'
    jobd = 'Z' if all(0 == d.flat) else 'D'

    gpeak, fpeak = ab13dd(dico, jobe, equil, jobd, n, m, p, a, e, b, c, d, tol)

    if dico=='D':
        fpeak /= sys.dt

    return gpeak, fpeak


def rss(states=1, outputs=1, inputs=1, strictly_proper=False, **kwargs):
    """Create a stable random state space object.

    Parameters
    ----------
    states, outputs, inputs : int, list of str, or None
        Description of the system states, outputs, and inputs. This can be
        given as an integer count or as a list of strings that name the
        individual signals.  If an integer count is specified, the names of
        the signal will be of the form 's[i]' (where 's' is one of 'x',
        'y', or 'u').
    strictly_proper : bool, optional
        If set to True, returns a proper system (no direct term).
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete time).
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.

    Returns
    -------
    sys : `StateSpace`
        The randomly created linear system.

    Raises
    ------
    ValueError
        If any input is not a positive integer.

    Notes
    -----
    If the number of states, inputs, or outputs is not specified, then the
    missing numbers are assumed to be 1.  If `dt` is not specified or is
    given as 0 or None, the poles of the returned system will always have a
    negative real part.  If `dt` is True or a positive float, the poles of
    the returned system will have magnitude less than 1.

    """
    # Process keyword arguments
    kwargs.update({'states': states, 'outputs': outputs, 'inputs': inputs})
    name, inputs, outputs, states, dt = _process_iosys_keywords(kwargs)

    # Figure out the size of the system
    nstates, _ = _process_signal_list(states)
    ninputs, _ = _process_signal_list(inputs)
    noutputs, _ = _process_signal_list(outputs)

    sys = _rss_generate(
        nstates, ninputs, noutputs, 'c' if not dt else 'd', name=name,
        strictly_proper=strictly_proper)

    return StateSpace(
        sys, name=name, states=states, inputs=inputs, outputs=outputs, dt=dt,
        **kwargs)


def drss(*args, **kwargs):
    """
    drss([states, outputs, inputs, strictly_proper])

    Create a stable, discrete-time, random state space system.

    Create a stable *discrete-time* random state space object.  This
    function calls `rss` using either the `dt` keyword provided by
    the user or `dt` = True if not specified.

    Examples
    --------
    >>> G = ct.drss(states=4, outputs=2, inputs=1)
    >>> G.ninputs, G.noutputs, G.nstates
    (1, 2, 4)
    >>> G.isdtime()
    True


    """
    # Make sure the timebase makes sense
    if 'dt' in kwargs:
        dt = kwargs['dt']

        if dt == 0:
            raise ValueError("drss called with continuous timebase")
        elif dt is None:
            warn("drss called with unspecified timebase; "
                 "system may be interpreted as continuous time")
            kwargs['dt'] = True     # force rss to generate discrete-time sys
    else:
        dt = True
        kwargs['dt'] = True

    # Create the system
    sys = rss(*args, **kwargs)

    # Reset the timebase (in case it was specified as None)
    sys.dt = dt

    return sys


# Summing junction
def summing_junction(
        inputs=None, output=None, dimension=None, prefix='u', **kwargs):
    """Create a summing junction as an input/output system.

    This function creates a static input/output system that outputs the sum
    of the inputs, potentially with a change in sign for each individual
    input.  The input/output system that is created by this function can be
    used as a component in the `interconnect` function.

    Parameters
    ----------
    inputs : int, string or list of strings
        Description of the inputs to the summing junction.  This can be
        given as an integer count, a string, or a list of strings. If an
        integer count is specified, the names of the input signals will be
        of the form 'u[i]'.
    output : string, optional
        Name of the system output.  If not specified, the output will be 'y'.
    dimension : int, optional
        The dimension of the summing junction.  If the dimension is set to a
        positive integer, a multi-input, multi-output summing junction will be
        created.  The input and output signal names will be of the form
        '<signal>[i]' where 'signal' is the input/output signal name specified
        by the `inputs` and `output` keywords.  Default value is None.
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name 'sys[id]' is generated with a unique integer id.
    prefix : string, optional
        If `inputs` is an integer, create the names of the states using the
        given prefix (default = 'u').  The names of the input will be of the
        form 'prefix[i]'.

    Returns
    -------
    sys : `StateSpace`
        Linear input/output system object with no states and only a direct
        term that implements the summing junction.

    Examples
    --------
    >>> P = ct.tf(1, [1, 0], inputs='u', outputs='y')
    >>> C = ct.tf(10, [1, 1], inputs='e', outputs='u')
    >>> sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    >>> T = ct.interconnect([P, C, sumblk], inputs='r', outputs='y')
    >>> T.ninputs, T.noutputs, T.nstates
    (1, 1, 2)

    """
    # Utility function to parse input and output signal lists
    def _parse_list(signals, signame='input', prefix='u'):
        # Parse signals, including gains
        if isinstance(signals, int):
            nsignals = signals
            names = ["%s[%d]" % (prefix, i) for i in range(nsignals)]
            gains = np.ones((nsignals,))
        elif isinstance(signals, str):
            nsignals = 1
            gains = [-1 if signals[0] == '-' else 1]
            names = [signals[1:] if signals[0] == '-' else signals]
        elif isinstance(signals, list) and \
             all([isinstance(x, str) for x in signals]):
            nsignals = len(signals)
            gains = np.ones((nsignals,))
            names = []
            for i in range(nsignals):
                if signals[i][0] == '-':
                    gains[i] = -1
                    names.append(signals[i][1:])
                else:
                    names.append(signals[i])
        else:
            raise ValueError(
                "could not parse %s description '%s'"
                % (signame, str(signals)))

        # Return the parsed list
        return nsignals, names, gains

    # Parse system and signal names (with some minor pre-processing)
    if input is not None:
        kwargs['inputs'] = inputs       # positional/keyword -> keyword
    if output is not None:
        kwargs['output'] = output       # positional/keyword -> keyword
    name, inputs, output, states, dt = _process_iosys_keywords(
        kwargs, {'inputs': None, 'outputs': 'y'}, end=True)
    if inputs is None:
        raise TypeError("input specification is required")

    # Read the input list
    ninputs, input_names, input_gains = _parse_list(
        inputs, signame="input", prefix=prefix)
    noutputs, output_names, output_gains = _parse_list(
        output, signame="output", prefix='y')
    if noutputs > 1:
        raise NotImplementedError("vector outputs not yet supported")

    # If the dimension keyword is present, vectorize inputs and outputs
    if isinstance(dimension, int) and dimension >= 1:
        # Create a new list of input/output names and update parameters
        input_names = ["%s[%d]" % (name, dim)
                       for name in input_names
                       for dim in range(dimension)]
        ninputs = ninputs * dimension

        output_names = ["%s[%d]" % (name, dim)
                        for name in output_names
                        for dim in range(dimension)]
        noutputs = noutputs * dimension
    elif dimension is not None:
        raise ValueError(
            "unrecognized dimension value '%s'" % str(dimension))
    else:
        dimension = 1

    # Create the direct term
    D = np.kron(input_gains * output_gains[0], np.eye(dimension))

    # Create a linear system of the appropriate size
    ss_sys = StateSpace(
        np.zeros((0, 0)), np.ones((0, ninputs)), np.ones((noutputs, 0)), D)

    # Create a StateSpace
    return StateSpace(
        ss_sys, inputs=input_names, outputs=output_names, name=name)

#
# Utility functions
#

def _ssmatrix(data, axis=1, square=None, rows=None, cols=None, name=None):
    """Convert argument to a (possibly empty) 2D state space matrix.

    This function can be used to process the matrices that define a
    state-space system. The axis keyword argument makes it convenient
    to specify that if the input is a vector, it is a row (axis=1) or
    column (axis=0) vector.

    Parameters
    ----------
    data : array, list, or string
        Input data defining the contents of the 2D array.
    axis : 0 or 1
        If input data is 1D, which axis to use for return object.  The
        default is 1, corresponding to a row matrix.
    square : bool, optional
        If set to True, check that the input matrix is square.
    rows : int, optional
        If set, check that the input matrix has the given number of rows.
    cols : int, optional
        If set, check that the input matrix has the given number of columns.
    name : str, optional
        Name of the state-space matrix being checked (for error messages).

    Returns
    -------
    arr : 2D array, with shape (0, 0) if a is empty

    """
    # Process the name of the object, if available
    name = "" if name is None else " " + name

    # Convert the data into an array (always making a copy)
    arr = np.array(data, dtype=float)
    ndim = arr.ndim
    shape = arr.shape

    # Change the shape of the array into a 2D array
    if (ndim > 2):
        raise ValueError(f"state-space matrix{name} must be 2-dimensional")

    elif (ndim == 2 and shape == (1, 0)) or \
         (ndim == 1 and shape == (0, )):
        # Passed an empty matrix or empty vector; change shape to (0, 0)
        shape = (0, 0)

    elif ndim == 1:
        # Passed a row or column vector
        shape = (1, shape[0]) if axis == 1 else (shape[0], 1)

    elif ndim == 0:
        # Passed a constant; turn into a matrix
        shape = (1, 1)

    # Check to make sure any conditions are satisfied
    if square and shape[0] != shape[1]:
        raise ControlDimension(
            f"state-space matrix{name} must be a square matrix")

    if rows is not None and shape[0] != rows:
        raise ControlDimension(
            f"state-space matrix{name} has the wrong number of rows; "
            f"expected {rows} instead of {shape[0]}")

    if cols is not None and shape[1] != cols:
        raise ControlDimension(
            f"state-space matrix{name} has the wrong number of columns; "
            f"expected {cols} instead of {shape[1]}")

    #  Create the actual object used to store the result
    return arr.reshape(shape)


def _f2s(f):
    """Format floating point number f for StateSpace._repr_latex_.

    Numbers are converted to strings with statesp.latex_num_format.

    Inserts column separators, etc., as needed.
    """
    fmt = "{:" + config.defaults['statesp.latex_num_format'] + "}"
    sraw = fmt.format(f)
    # significant-exponent
    se = sraw.lower().split('e')
    # whole-fraction
    wf = se[0].split('.')
    s = wf[0]
    if wf[1:]:
        s += r'.&\hspace{{-1em}}{frac}'.format(frac=wf[1])
    else:
        s += r'\phantom{.}&\hspace{-1em}'

    if se[1:]:
        s += r'&\hspace{{-1em}}\cdot10^{{{:d}}}'.format(int(se[1]))
    else:
        s += r'&\hspace{-1em}\phantom{\cdot}'

    return s


def _convert_to_statespace(sys, use_prefix_suffix=False, method=None):
    """Convert a system to state space form (if needed).

    If `sys` is already a state space object, then it is returned.  If
    `sys` is a transfer function object, then it is converted to a state
    space and returned.

    Note: no renaming of inputs and outputs is performed; this should be done
    by the calling function.

    """
    import itertools

    from .xferfcn import TransferFunction

    if isinstance(sys, StateSpace):
        return sys

    elif isinstance(sys, TransferFunction):
        # Make sure the transfer function is proper
        if any([[len(num) for num in col] for col in sys.num] >
               [[len(num) for num in col] for col in sys.den]):
            raise ValueError("transfer function is non-proper; can't "
                             "convert to StateSpace system")

        if method is None and slycot_check() or method == 'slycot':
            if not slycot_check():
                raise ValueError("method='slycot' requires slycot")

            from slycot import td04ad

            # Change the numerator and denominator arrays so that the transfer
            # function matrix has a common denominator.
            # matrices are also sized/padded to fit td04ad
            num, den, denorder = sys.minreal()._common_den()
            num, den, denorder = sys._common_den()

            # transfer function to state space conversion now should work!
            ssout = td04ad('C', sys.ninputs, sys.noutputs,
                           denorder, den, num, tol=0)

            states = ssout[0]
            newsys = StateSpace(
                ssout[1][:states, :states], ssout[2][:states, :sys.ninputs],
                ssout[3][:sys.noutputs, :states], ssout[4], sys.dt)

        elif method in [None, 'scipy']:
            # SciPy tf->ss can't handle MIMO, but SISO is OK
            maxn = max(max(len(n) for n in nrow)
                       for nrow in sys.num)
            maxd = max(max(len(d) for d in drow)
                       for drow in sys.den)
            if 1 == maxn and 1 == maxd:
                D = empty((sys.noutputs, sys.ninputs), dtype=float)
                for i, j in itertools.product(range(sys.noutputs),
                                              range(sys.ninputs)):
                    D[i, j] = sys.num_array[i, j][0] / sys.den_array[i, j][0]
                newsys = StateSpace([], [], [], D, sys.dt)
            else:
                if not issiso(sys):
                    raise ControlMIMONotImplemented(
                        "MIMO system conversion not supported without Slycot")

                A, B, C, D = \
                    sp.signal.tf2ss(squeeze(sys.num), squeeze(sys.den))
                newsys = StateSpace(A, B, C, D, sys.dt)
        else:
            raise ValueError(f"unknown {method=}")

        # Copy over the signal (and system) names
        newsys._copy_names(
            sys,
            prefix_suffix_name='converted' if use_prefix_suffix else None)
        return newsys

    elif isinstance(sys, FrequencyResponseData):
        raise TypeError("Can't convert FRD to StateSpace system.")

    # If this is a matrix, try to create a constant feedthrough
    try:
        D = _ssmatrix(np.atleast_2d(sys), name="D")
        return StateSpace([], [], [], D, dt=None)

    except Exception:
        raise TypeError("Can't convert given type to StateSpace system.")


def _rss_generate(
        states, inputs, outputs, cdtype, strictly_proper=False, name=None):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

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
    if cdtype not in ['c', 'd']:
        raise ValueError("cdtype must be `c` or `d`")

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
            if cdtype == 'c':
                poles[i] = -exp(randn()) + 0.j
            else:
                poles[i] = 2. * rand() - 1.
            i += 1
        else:
            # Complex conjugate pair of oscillating poles.
            if cdtype == 'c':
                poles[i] = complex(-exp(randn()), 3. * exp(randn()))
            else:
                mag = rand()
                phase = 2. * math.pi * rand()
                poles[i] = complex(mag * cos(phase), mag * sin(phase))
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
            A = solve(T, A) @ T  # A = T \ A @ T
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
        if any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if any(Cmask):  # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else zeros(D.shape)

    if cdtype == 'c':
        ss_args = (A, B, C, D)
    else:
        ss_args = (A, B, C, D, True)
    return StateSpace(*ss_args, name=name)
