# frdata.py - frequency response data representation and functions
#
# Initial author: M.M. (Rene) van Paassen (using xferfcn.py as basis)
# Creation date: 02 Oct 2012

"""Frequency response data representation and functions.

This module contains the `FrequencyResponseData` (FRD) class and also
functions that operate on FRD data.

"""

from collections.abc import Iterable
from copy import copy
from warnings import warn

import numpy as np
from numpy import absolute, array, empty, eye, imag, linalg, ones, real, sort
from scipy.interpolate import splev, splprep

from . import bdalg, config
from .exception import pandas_check
from .iosys import InputOutputSystem, NamedSignal, _extended_system_name, \
    _process_iosys_keywords, _process_subsys_index, common_timebase
from .lti import LTI, _process_frequency_response

__all__ = ['FrequencyResponseData', 'FRD', 'frd']


class FrequencyResponseData(LTI):
    """FrequencyResponseData(frdata, omega[, smooth])

    Input/output model defined by frequency response data (FRD).

    The FrequencyResponseData (FRD) class is used to represent systems in
    frequency response data form.  It can be created manually using the
    class constructor, using the `frd` factory function, or
    via the `frequency_response` function.

    Parameters
    ----------
    frdata : 1D or 3D complex array_like
        The frequency response at each frequency point.  If 1D, the system is
        assumed to be SISO.  If 3D, the system is MIMO, with the first
        dimension corresponding to the output index of the FRD, the second
        dimension corresponding to the input index, and the 3rd dimension
        corresponding to the frequency points in `omega`.  When accessed as an
        attribute, `frdata` is always stored as a 3D array.
    omega : iterable of real frequencies
        List of monotonically increasing frequency points for the response.
    smooth : bool, optional
        If True, create an interpolation function that allows the frequency
        response to be computed at any frequency within the range of
        frequencies give in `omega`.  If False (default), frequency response
        can only be obtained at the frequencies specified in `omega`.
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous time, True
        indicates discrete time with unspecified sampling time, positive
        number is discrete time with specified sampling time, None
        indicates unspecified timebase (either continuous or discrete time).
    squeeze : bool
        By default, if a system is single-input, single-output (SISO) then
        the outputs (and inputs) are returned as a 1D array (indexed by
        frequency) and if a system is multi-input or multi-output, then the
        outputs are returned as a 2D array (indexed by output and
        frequency) or a 3D array (indexed by output, trace, and frequency).
        If `squeeze` = True, access to the output response will remove
        single-dimensional entries from the shape of the inputs and outputs
        even if the system is not SISO. If `squeeze` = False, the output is
        returned as a 3D array (indexed by the output, input, and
        frequency) even if the system is SISO. The default value can be set
        using `config.defaults['control.squeeze_frequency_response']`.
    sysname : str or None
        Name of the system that generated the data.

    Attributes
    ----------
    complex : array
        Complex frequency response, indexed by output index, input index, and
        frequency point, with squeeze processing.
    magnitude : array
        Magnitude of the frequency response, indexed by output index, input
        index, and frequency point, with squeeze processing.
    phase : array
        Phase of the frequency response, indexed by output index, input index,
        and frequency point, with squeeze processing.
    frequency : 1D array
        Array of frequency points for which data are available.
    ninputs, noutputs : int
        Number of input and output signals.
    shape : tuple
        2-tuple of I/O system dimension, (noutputs, ninputs).
    input_labels, output_labels : array of str
        Names for the input and output signals.
    name : str
        System name.  For data generated using
        `frequency_response`, stores the name of the
        system that created the data.

    Other Parameters
    ----------------
    plot_type : str, optional
        Set the type of plot to generate with `~FrequencyResponseData.plot`
        ('bode', 'nichols').
    title : str, optional
        Set the title to use when plotting.
    plot_magnitude, plot_phase : bool, optional
        If set to False, don't plot the magnitude or phase, respectively.
    return_magphase : bool, optional
        If True, then a frequency response data object will enumerate
        as a tuple of the form ``(mag, phase, omega)`` where where `mag`
        is the magnitude (absolute value, not dB or log10) of the system
        frequency response, `phase` is the wrapped phase in radians of the
        system frequency response, and `omega` is the (sorted) frequencies
        at which the response was evaluated.

    See Also
    --------
    frd, frequency_response, InputOutputSystem, TransferFunction

    Notes
    -----
    The main data members are `omega` and `frdata`, where `omega` is a 1D
    array of frequency points and and `frdata` is a 3D array of frequency
    responses, with the first dimension corresponding to the output index of
    the FRD, the second dimension corresponding to the input index, and the
    3rd dimension corresponding to the frequency points in omega.  For example,

    >>> frdata[2,5,:] = numpy.array([1., 0.8-0.2j, 0.2-0.8j])   # doctest: +SKIP

    means that the frequency response from the 6th input to the 3rd output at
    the frequencies defined in omega is set to the array above, i.e. the rows
    represent the outputs and the columns represent the inputs.

    A frequency response data object is callable and returns the value of the
    transfer function evaluated at a point in the complex plane (must be on
    the imaginary axis).  See `FrequencyResponseData.__call__`
    for a more detailed description.

    Subsystem response corresponding to selected input/output pairs can be
    created by indexing the frequency response data object::

        subsys = sys[output_spec, input_spec]

    The input and output specifications can be single integers, lists of
    integers, or slices.  In addition, the strings representing the names
    of the signals can be used and will be replaced with the equivalent
    signal offsets.

    """
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

    #: Squeeze processing parameter.
    #:
    #: By default, if a system is single-input, single-output (SISO) then
    #: the outputs (and inputs) are returned as a 1D array (indexed by
    #: frequency) and if a system is multi-input or multi-output, then the
    #: outputs are returned as a 2D array (indexed by output and frequency)
    #: or a 3D array (indexed by output, trace, and frequency).  If
    #: `squeeze` = True, access to the output response will remove
    #: single-dimensional entries from the shape of the inputs and outputs
    #: even if the system is not SISO. If `squeeze` = False, the output is
    #: returned as a 3D array (indexed by the output, input, and frequency)
    #: even if the system is SISO. The default value can be set using
    #: config.defaults['control.squeeze_frequency_response'].
    #:
    #: :meta hide-value:
    squeeze = None

    _epsw = 1e-8                #: Bound for exact frequency match

    def __init__(self, *args, **kwargs):
        """FrequencyResponseData(response, omega[, dt])

        Construct a frequency response data (FRD) object.

        The default constructor is `FrequencyResponseData(response, omega)`,
        where `omega` is an iterable of frequency points and `response` is
        the matching frequency data.  If `response` is a single list, 1D
        array, or tuple, a SISO system description is assumed. `response`
        can also be a 2D array, in which case a MIMO response is created.
        To call the copy constructor, call `FrequencyResponseData(sys)`,
        where `sys` is a FRD object.  The timebase for the frequency
        response can be provided using an optional third argument or the
        `dt` keyword.

        To construct frequency response data for an existing LTI object,
        other than an FRD, call `FrequencyResponseData(sys, omega)`.  This
        functionality can also be obtained using `frequency_response`
        (which has additional options available).

        See `FrequencyResponseData` and `frd` for more
        information.

        """
        smooth = kwargs.pop('smooth', False)

        #
        # Process positional arguments
        #
        if len(args) == 3:
            # Discrete time transfer function
            dt = args[-1]
            if 'dt' in kwargs:
                warn("received multiple dt arguments, "
                     "using positional arg dt = %s" % dt)
            kwargs['dt'] = dt
            args = args[:-1]

        if len(args) == 2:
            if not isinstance(args[0], FRD) and isinstance(args[0], LTI):
                # not an FRD, but still an LTI system, second argument
                # should be the frequency range
                otherlti = args[0]
                self.omega = sort(np.asarray(args[1], dtype=float))

                # calculate frequency response at specified points
                if otherlti.isctime():
                    s = 1j * self.omega
                    self.frdata = otherlti(s, squeeze=False)
                else:
                    z = np.exp(1j * self.omega * otherlti.dt)
                    self.frdata = otherlti(z, squeeze=False)
                arg_dt = otherlti.dt

                # Copy over signal and system names, if not specified
                kwargs['inputs'] = kwargs.get('inputs', otherlti.input_labels)
                kwargs['outputs'] = kwargs.get(
                    'outputs', otherlti.output_labels)
                if not otherlti._generic_name_check():
                    kwargs['name'] = kwargs.get('name', _extended_system_name(
                        otherlti.name, prefix_suffix_name='sampled'))

            else:
                # The user provided a response and a freq vector
                self.frdata = array(args[0], dtype=complex, ndmin=1)
                if self.frdata.ndim == 1:
                    self.frdata = self.frdata.reshape(1, 1, -1)
                self.omega = array(args[1], dtype=float, ndmin=1)
                if self.frdata.ndim != 3 or self.omega.ndim != 1 or \
                        self.frdata.shape[-1] != self.omega.shape[-1]:
                    raise TypeError(
                        "The frequency data constructor needs a 1-d or 3-d"
                        " response data array and a matching frequency vector"
                        " size")
                arg_dt = None

        elif len(args) == 1:
            # Use the copy constructor.
            if not isinstance(args[0], FRD):
                raise TypeError(
                    "The one-argument constructor can only take in"
                    " an FRD object.  Received %s." % type(args[0]))
            self.omega = args[0].omega
            self.frdata = args[0].frdata
            arg_dt = args[0].dt

            # Copy over signal and system names, if not specified
            kwargs['inputs'] = kwargs.get('inputs', args[0].input_labels)
            kwargs['outputs'] = kwargs.get('outputs', args[0].output_labels)

        else:
            raise ValueError(
                "Needs 1 or 2 arguments; received %i." % len(args))

        #
        # Process keyword arguments
        #

        # If data was generated by a system, keep track of that (used when
        # plotting data).  Otherwise, use the system name, if given.
        self.sysname = kwargs.pop('sysname', kwargs.get('name', None))

        # Keep track of default properties for plotting
        self.plot_phase = kwargs.pop('plot_phase', None)
        self.title = kwargs.pop('title', None)
        self.plot_type = kwargs.pop('plot_type', 'bode')

        # Keep track of return type
        self.return_magphase=kwargs.pop('return_magphase', False)
        if self.return_magphase not in (True, False):
            raise ValueError("unknown return_magphase value")
        self._return_singvals=kwargs.pop('_return_singvals', False)

        # Determine whether to squeeze the output
        self.squeeze=kwargs.pop('squeeze', None)
        if self.squeeze not in (None, True, False):
            raise ValueError("unknown squeeze value")

        defaults = {
            'inputs': self.frdata.shape[1] if not getattr(
                self, 'input_index', None) else self.input_labels,
            'outputs': self.frdata.shape[0] if not getattr(
                self, 'output_index', None) else self.output_labels,
            'name': getattr(self, 'name', None)}
        if arg_dt is not None:
            if isinstance(args[0], LTI):
                arg_dt = common_timebase(args[0].dt, arg_dt)
            kwargs['dt'] = arg_dt

        # Process signal names
        name, inputs, outputs, states, dt = _process_iosys_keywords(
                kwargs, defaults)
        InputOutputSystem.__init__(
            self, name=name, inputs=inputs, outputs=outputs, dt=dt, **kwargs)

        # create interpolation functions
        if smooth:
            # Set the order of the fit
            if self.omega.size < 2:
                raise ValueError("can't smooth with only 1 frequency")
            degree = 3 if self.omega.size > 3 else self.omega.size - 1

            self._ifunc = empty((self.frdata.shape[0], self.frdata.shape[1]),
                               dtype=tuple)
            for i in range(self.frdata.shape[0]):
                for j in range(self.frdata.shape[1]):
                    self._ifunc[i, j], u = splprep(
                        u=self.omega, x=[real(self.frdata[i, j, :]),
                                         imag(self.frdata[i, j, :])],
                        w=1.0/(absolute(self.frdata[i, j, :]) + 0.001),
                        s=0.0, k=degree)
        else:
            self._ifunc = None

    #
    # Frequency response properties
    #
    # Different properties of the frequency response that can be used for
    # analysis and characterization.
    #

    @property
    def magnitude(self):
        """Magnitude of the frequency response.

        Magnitude of the frequency response, indexed by either the output
        and frequency (if only a single input is given) or the output,
        input, and frequency (for multi-input systems).  See
        `FrequencyResponseData.squeeze` for a description of how this
        can be modified using the `squeeze` keyword.

        Input and output signal names can be used to index the data in
        place of integer offsets.

        :type: 1D, 2D, or 3D array

        """
        frdata = _process_frequency_response(
            self, self.omega, self.frdata, squeeze=self.squeeze)
        return NamedSignal(
            np.abs(frdata), self.output_labels, self.input_labels)

    @property
    def phase(self):
        """Phase of the frequency response.

        Phase of the frequency response in radians/sec, indexed by either
        the output and frequency (if only a single input is given) or the
        output, input, and frequency (for multi-input systems).  See
        `FrequencyResponseData.squeeze` for a description of how this
        can be modified using the `squeeze` keyword.

        Input and output signal names can be used to index the data in
        place of integer offsets.

        :type: 1D, 2D, or 3D array

        """
        frdata = _process_frequency_response(
            self, self.omega, self.frdata, squeeze=self.squeeze)
        return NamedSignal(
            np.angle(frdata), self.output_labels, self.input_labels)

    @property
    def frequency(self):
        """Frequencies at which the response is evaluated.

        :type: 1D array

        """
        return self.omega

    @property
    def complex(self):
        """Complex value of the frequency response.

        Value of the frequency response as a complex number, indexed by
        either the output and frequency (if only a single input is given)
        or the output, input, and frequency (for multi-input systems).  See
        `FrequencyResponseData.squeeze` for a description of how this
        can be modified using the `squeeze` keyword.

        Input and output signal names can be used to index the data in
        place of integer offsets.

        :type: 1D, 2D, or 3D array

        """
        frdata = _process_frequency_response(
            self, self.omega, self.frdata, squeeze=self.squeeze)
        return NamedSignal(
            frdata, self.output_labels, self.input_labels)

    @property
    def response(self):
        warn("response property is deprecated; use complex", FutureWarning)
        return self.complex

    @property
    def fresp(self):
        warn("fresp attribute is deprecated; use frdata", FutureWarning)
        return self.frdata

    def __str__(self):

        """String representation of the transfer function."""

        mimo = self.ninputs > 1 or self.noutputs > 1
        outstr = [f"{InputOutputSystem.__str__(self)}"]
        nl = "\n  " if mimo else "\n"
        sp = "  " if mimo else ""

        for i in range(self.ninputs):
            for j in range(self.noutputs):
                if mimo:
                    outstr.append(
                        "\nInput %i to output %i:" % (i + 1, j + 1))
                outstr.append(nl + 'Freq [rad/s]  Response')
                outstr.append(sp + '------------  ---------------------')
                outstr.extend(
                    [sp + '%12.3f  %10.4g%+10.4gj' % (w, re, im)
                     for w, re, im in zip(self.omega,
                                          real(self.frdata[j, i, :]),
                                          imag(self.frdata[j, i, :]))])

        return '\n'.join(outstr)

    def _repr_eval_(self):
        # Loadable format
        out = "FrequencyResponseData(\n{d},\n{w}{smooth}".format(
            d=repr(self.frdata), w=repr(self.omega),
            smooth=(self._ifunc and ", smooth=True") or "")

        out += self._dt_repr()
        if len(labels := self._label_repr()) > 0:
            out += ",\n" + labels

        out += ")"
        return out

    def __neg__(self):
        """Negate a transfer function."""

        return FRD(-self.frdata, self.omega)

    def __add__(self, other):
        """Add two LTI objects (parallel connection)."""

        if isinstance(other, FRD):
            # verify that the frequencies match
            if len(other.omega) != len(self.omega) or \
               (other.omega != self.omega).any():
                warn("Frequency points do not match; expect "
                     "truncation and interpolation.")

        # Convert the second argument to a frequency response function.
        # or re-base the frd to the current omega (if needed)
        if isinstance(other, (int, float, complex, np.number)):
            other = _convert_to_frd(
                other, omega=self.omega,
                inputs=self.ninputs, outputs=self.noutputs)
        else:
            other = _convert_to_frd(other, omega=self.omega)

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = np.ones((other.noutputs, other.ninputs)) * self
        elif not self.issiso() and other.issiso():
            other = np.ones((self.noutputs, self.ninputs)) * other

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.ninputs:
            raise ValueError(
                "The first summand has %i input(s), but the " \
                "second has %i." % (self.ninputs, other.ninputs))
        if self.noutputs != other.noutputs:
            raise ValueError(
                "The first summand has %i output(s), but the " \
                "second has %i." % (self.noutputs, other.noutputs))

        return FRD(self.frdata + other.frdata, other.omega)

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
        if isinstance(other, (int, float, complex, np.number)):
            return FRD(self.frdata * other, self.omega,
                       smooth=(self._ifunc is not None))
        else:
            other = _convert_to_frd(other, omega=self.omega)

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self] * other.noutputs))
        elif not self.issiso() and other.issiso():
            other = bdalg.append(*([other] * self.ninputs))

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.noutputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (self.ninputs, other.noutputs))

        inputs = other.ninputs
        outputs = self.noutputs
        frdata = empty((outputs, inputs, len(self.omega)),
                      dtype=self.frdata.dtype)
        for i in range(len(self.omega)):
            frdata[:, :, i] = self.frdata[:, :, i] @ other.frdata[:, :, i]
        return FRD(frdata, self.omega,
                   smooth=(self._ifunc is not None) and
                          (other._ifunc is not None))

    def __rmul__(self, other):
        """Right Multiply two LTI objects (serial connection)."""

        # Convert the second argument to an frd function.
        if isinstance(other, (int, float, complex, np.number)):
            return FRD(self.frdata * other, self.omega,
                       smooth=(self._ifunc is not None))
        else:
            other = _convert_to_frd(other, omega=self.omega)

        # Promote SISO object to compatible dimension
        if self.issiso() and not other.issiso():
            self = bdalg.append(*([self] * other.ninputs))
        elif not self.issiso() and other.issiso():
            other = bdalg.append(*([other] * self.noutputs))

        # Check that the input-output sizes are consistent.
        if self.noutputs != other.ninputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (other.ninputs, self.noutputs))

        inputs = self.ninputs
        outputs = other.noutputs

        frdata = empty((outputs, inputs, len(self.omega)),
                      dtype=self.frdata.dtype)
        for i in range(len(self.omega)):
            frdata[:, :, i] = other.frdata[:, :, i] @ self.frdata[:, :, i]
        return FRD(frdata, self.omega,
                   smooth=(self._ifunc is not None) and
                          (other._ifunc is not None))

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""

        if isinstance(other, (int, float, complex, np.number)):
            return FRD(self.frdata * (1/other), self.omega,
                       smooth=(self._ifunc is not None))
        else:
            other = _convert_to_frd(other, omega=self.omega)

        if (other.ninputs > 1 or other.noutputs > 1):
            # FRD.__truediv__ is currently only implemented for SISO systems
            return NotImplemented

        return FRD(self.frdata/other.frdata, self.omega,
                   smooth=(self._ifunc is not None) and
                          (other._ifunc is not None))

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rtruediv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, complex, np.number)):
            return FRD(other / self.frdata, self.omega,
                       smooth=(self._ifunc is not None))
        else:
            other = _convert_to_frd(other, omega=self.omega)

        if (self.ninputs > 1 or self.noutputs > 1):
            # FRD.__rtruediv__ is currently only implemented for SISO systems
            return NotImplemented

        return other / self

    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return FRD(ones(self.frdata.shape), self.omega,
                       smooth=(self._ifunc is not None))  # unity
        if other > 0:
            return self * (self**(other-1))
        if other < 0:
            return (FRD(ones(self.frdata.shape), self.omega) / self) * \
                (self**(other+1))

    # Define the `eval` function to evaluate an FRD at a given (real)
    # frequency.  Note that we choose to use `eval` instead of `evalfr` to
    # avoid confusion with `evalfr`, which takes a complex number as its
    # argument.  Similarly, we don't use `__call__` to avoid confusion between
    # G(s) for a transfer function and G(omega) for an FRD object.
    # update Sawyer B. Fuller 2020.08.14: __call__ added to provide a uniform
    # interface to systems in general and the lti.frequency_response method
    def eval(self, omega, squeeze=None):
        """Evaluate a transfer function at a frequency point.

        Note that a "normal" FRD only returns values for which there is an
        entry in the `omega` vector. An interpolating FRD can return
        intermediate values.

        Parameters
        ----------
        omega : float or 1D array_like
            Frequency(s) for evaluation, in radians per second.
        squeeze : bool, optional
            If `squeeze` = True, remove single-dimensional entries from the
            shape of the output even if the system is not SISO. If
            `squeeze` = False, keep all indices (output, input and, if
            `omega` is array_like, frequency) even if the system is
            SISO. The default value can be set using
            `config.defaults['control.squeeze_frequency_response']`.

        Returns
        -------
        frdata : complex ndarray
            The frequency response of the system.  If the system is SISO
            and `squeeze` is not True, the shape of the array matches the
            shape of `omega`.  If the system is not SISO or `squeeze` is
            False, the first two dimensions of the array are indices for
            the output and input and the remaining dimensions match `omega`.
            If `squeeze` is True then single-dimensional axes are removed.

        """
        omega_array = np.array(omega, ndmin=1)  # array of frequencies

        # Make sure that we are operating on a simple list
        if len(omega_array.shape) > 1:
            raise ValueError("input list must be 1D")

        # Make sure that frequencies are all real-valued
        if any(omega_array.imag > 0):
            raise ValueError("eval can only accept real-valued frequencies")

        if self._ifunc is None:
            elements = np.isin(self.omega, omega)  # binary array
            if sum(elements) < len(omega_array):
                raise ValueError(
                    "not all frequencies are in frequency list of FRD "
                    "system. Try an interpolating FRD for additional points.")
            else:
                out = self.frdata[:, :, elements]
        else:
            out = empty((self.noutputs, self.ninputs, len(omega_array)),
                        dtype=complex)
            for i in range(self.noutputs):
                for j in range(self.ninputs):
                    for k, w in enumerate(omega_array):
                        frraw = splev(w, self._ifunc[i, j], der=0)
                        out[i, j, k] = frraw[0] + 1.0j * frraw[1]

        return _process_frequency_response(self, omega, out, squeeze=squeeze)

    def __call__(self, x=None, squeeze=None, return_magphase=None):
        """Evaluate system transfer function at point in complex plane.

        Returns the value of the system's transfer function at a point `x`
        in the complex plane, where `x` is `s` for continuous-time systems
        and `z` for discrete-time systems.  For a frequency response data
        object, the argument should be an imaginary number (since only the
        frequency response is defined) and only the imaginary component of
        `x` will be used.

        By default, a (complex) scalar will be returned for SISO systems
        and a p x m array will be return for MIMO systems with m inputs and
        p outputs.  This can be changed using the `squeeze` keyword.

        To evaluate at a frequency `omega` in radians per second, enter ``x
        = omega * 1j`` for continuous-time systems, ``x = exp(1j * omega *
        dt)`` for discrete-time systems, or use the
        `~LTI.frequency_response` method.

        If `x` is not given, this function creates a copy of a frequency
        response data object with a different set of output settings.

        Parameters
        ----------
        x : complex scalar or 1D array_like
            Imaginary value(s) at which frequency response will be evaluated.
            The real component of `x` is ignored.  If not specified, return
            a copy of the frequency response data object with updated
            settings for output processing (`squeeze`, `return_magphase`).
        squeeze : bool, optional
            Squeeze output, as described below.  Default value can be set
            using `config.defaults['control.squeeze_frequency_response']`.
        return_magphase : bool, optional
            (`x` = None only) If True, then a frequency response data object
            will enumerate as a tuple of the form ``(mag, phase, omega)``
            where where `mag` is the magnitude (absolute value, not dB or
            log10) of the system frequency response, `phase` is the wrapped
            phase in radians of the system frequency response, and `omega` is
            the (sorted) frequencies at which the response was evaluated.

        Returns
        -------
        frdata : complex ndarray
            The value of the system transfer function at `x`.  If the system
            is SISO and `squeeze` is not True, the shape of the array matches
            the shape of `x`.  If the system is not SISO or `squeeze` is
            False, the first two dimensions of the array are indices for the
            output and input and the remaining dimensions match `x`.  If
            `squeeze` is True then single-dimensional axes are removed.

        Raises
        ------
        ValueError
            If `s` is not purely imaginary, because `FrequencyResponseData`
            systems are only defined at imaginary values (corresponding to
            real frequencies).

        """
        if x is None:
            # Create a copy of the response with new keywords
            response = copy(self)

            # Update any keywords that we were passed
            response.squeeze = self.squeeze if squeeze is None else squeeze
            response.return_magphase = self.return_magphase \
                if return_magphase is None else return_magphase

            return response

        if return_magphase is not None:
            raise ValueError("return_magphase not allowed when x != None")

        # Make sure that we are operating on a simple list
        if len(np.atleast_1d(x).shape) > 1:
            raise ValueError("input list must be 1D")

        if any(abs(np.atleast_1d(x).real) > 0):
            raise ValueError("__call__: FRD systems can only accept "
                             "purely imaginary frequencies")

        # need to preserve array or scalar status
        if hasattr(x, '__len__'):
            return self.eval(np.asarray(x).imag, squeeze=squeeze)
        else:
            return self.eval(complex(x).imag, squeeze=squeeze)

    # Implement iter to allow assigning to a tuple
    def __iter__(self):
        frdata = _process_frequency_response(
            self, self.omega, self.frdata, squeeze=self.squeeze)
        if self._return_singvals:
            # Legacy processing for singular values
            return iter((self.frdata[:, 0, :], self.omega))
        elif not self.return_magphase:
            return iter((self.omega, frdata))
        return iter((np.abs(frdata), np.angle(frdata), self.omega))

    def __getitem__(self, key):
        if not isinstance(key, Iterable) or len(key) != 2:
            # Implement (thin) getitem to allow access via legacy indexing
            return list(self.__iter__())[key]

        # Convert signal names to integer offsets (via NamedSignal object)
        iomap = NamedSignal(
            self.frdata[:, :, 0], self.output_labels, self.input_labels)
        indices = iomap._parse_key(key, level=1)  # ignore index checks
        outdx, outputs = _process_subsys_index(indices[0], self.output_labels)
        inpdx, inputs = _process_subsys_index(indices[1], self.input_labels)

        # Create the system name
        sysname = config.defaults['iosys.indexed_system_name_prefix'] + \
            self.name + config.defaults['iosys.indexed_system_name_suffix']

        return FrequencyResponseData(
            self.frdata[outdx, :][:, inpdx], self.omega, self.dt,
            inputs=inputs, outputs=outputs, name=sysname)

    # Implement (thin) len to emulate legacy testing interface
    def __len__(self):
        return 3 if self.return_magphase else 2

    def freqresp(self, omega):
        """(deprecated) Evaluate transfer function at complex frequencies.

        .. deprecated::0.9.0
            Method has been given the more Pythonic name
            `FrequencyResponseData.frequency_response`. Or use
            `freqresp` in the MATLAB compatibility module.

        """
        warn("FrequencyResponseData.freqresp(omega) will be removed in a "
             "future release of python-control; use "
             "FrequencyResponseData.frequency_response(omega), or "
             "freqresp(sys, omega) in the MATLAB compatibility module "
             "instead", FutureWarning)
        return self.frequency_response(omega)

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two FRD objects.

        Parameters
        ----------
        other : `LTI`
            System in the feedback path.

        sign : float, optional
            Gain to use in feedback path.  Defaults to -1.

        """
        other = _convert_to_frd(other, omega=self.omega)

        if (self.noutputs != other.ninputs or self.ninputs != other.noutputs):
            raise ValueError(
                "FRD.feedback, inputs/outputs mismatch")

        # TODO: handle omega re-mapping

        # reorder array axes in order to leverage numpy broadcasting
        myfrdata = np.moveaxis(self.frdata, 2, 0)
        otherfrdata = np.moveaxis(other.frdata, 2, 0)
        I_AB = eye(self.ninputs)[np.newaxis, :, :] + otherfrdata @ myfrdata
        resfrdata = (myfrdata @ linalg.inv(I_AB))
        frdata = np.moveaxis(resfrdata, 0, 2)

        return FRD(frdata, other.omega, smooth=(self._ifunc is not None))

    def append(self, other):
        """Append a second model to the present model.

        The second model is converted to FRD if necessary, inputs and
        outputs are appended and their order is preserved.

        Parameters
        ----------
        other : `LTI`
            System to be appended.

        Returns
        -------
        sys : `FrequencyResponseData`
            System model with `other` appended to `self`.

        """
        other = _convert_to_frd(other, omega=self.omega, inputs=other.ninputs,
                                outputs=other.noutputs)

        # TODO: handle omega re-mapping

        new_frdata = np.zeros(
            (self.noutputs + other.noutputs, self.ninputs + other.ninputs,
             self.omega.shape[-1]), dtype=complex)
        new_frdata[:self.noutputs, :self.ninputs, :] = np.reshape(
            self.frdata, (self.noutputs, self.ninputs, -1))
        new_frdata[self.noutputs:, self.ninputs:, :] = np.reshape(
            other.frdata, (other.noutputs, other.ninputs, -1))

        return FRD(new_frdata, self.omega, smooth=(self._ifunc is not None))

    # Plotting interface
    def plot(self, plot_type=None, *args, **kwargs):
        """Plot the frequency response using Bode or singular values plot.

        Plot the frequency response using either a standard Bode plot
        (plot_type='bode', default) or a singular values plot
        (plot_type='svplot').  See `bode_plot` and `singular_values_plot`
        for more detailed descriptions.

        """
        from .freqplot import bode_plot, singular_values_plot
        from .nichols import nichols_plot

        if plot_type is None:
            plot_type = self.plot_type

        if plot_type == 'bode':
            return bode_plot(self, *args, **kwargs)
        elif plot_type == 'nichols':
            return nichols_plot(self, *args, **kwargs)
        elif plot_type == 'svplot':
            return singular_values_plot(self, *args, **kwargs)
        else:
            raise ValueError(f"unknown plot type '{plot_type}'")

    # Convert to pandas
    def to_pandas(self):
        """Convert response data to pandas data frame.

        Creates a pandas data frame for the value of the frequency
        response at each `omega`.  The frequency response values are
        labeled in the form "H_{<out>, <in>}" where "<out>" and "<in>"
        are replaced with the output and input labels for the system.

        """
        if not pandas_check():
            ImportError('pandas not installed')
        import pandas

        # Create a dict for setting up the data frame
        data = {'omega': self.omega}
        data.update(
            {'H_{%s, %s}' % (out, inp): self.frdata[i, j] \
             for i, out in enumerate(self.output_labels) \
             for j, inp in enumerate(self.input_labels)})

        return pandas.DataFrame(data)


#
# Allow FRD as an alias for the FrequencyResponseData class
#
# Note: This class was initially given the name "FRD", but this caused
# problems with documentation on MacOS platforms, since files were generated
# for control.frd and control.FRD, which are not differentiated on most MacOS
# file systems, which are case insensitive.  Renaming the FRD class to be
# FrequencyResponseData and then assigning FRD to point to the same object
# fixes this problem.
#
FRD = FrequencyResponseData


def _convert_to_frd(sys, omega, inputs=1, outputs=1):
    """Convert a system to frequency response data form (if needed).

    If `sys` is already a frequency response data object, and its frequency
    range matches or overlaps the range given in `omega` then it is
    returned.  If `sys` is another LTI object or a transfer function, then
    it is converted to a frequency response data system at the specified
    values in `omega`. If `sys` is a scalar, then the number of inputs and
    outputs can be specified manually, as in:

    >>> import numpy as np
    >>> from control.frdata import _convert_to_frd

    >>> omega = np.logspace(-1, 1)
    >>> frd = _convert_to_frd(3., omega) # Assumes inputs = outputs = 1
    >>> frd.ninputs, frd.noutputs
    (1, 1)

    >>> frd = _convert_to_frd(1., omega, inputs=3, outputs=2)
    >>> frd.ninputs, frd.noutputs
    (3, 2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].

    """

    if isinstance(sys, FRD):
        omega.sort()
        if len(omega) == len(sys.omega) and \
           (abs(omega - sys.omega) < FRD._epsw).all():
            # frequencies match, and system was already frd; simply use
            return sys

        raise NotImplementedError(
            "Frequency ranges of FRD do not match, conversion not implemented")

    elif isinstance(sys, LTI):
        omega = np.sort(omega)
        if sys.isctime():
            frdata = sys(1j * omega)
        else:
            frdata = sys(np.exp(1j * omega * sys.dt))
        if len(frdata.shape) == 1:
            frdata = frdata[np.newaxis, np.newaxis, :]
        return FRD(frdata, omega, smooth=True)

    elif isinstance(sys, (int, float, complex, np.number)):
        frdata = ones((outputs, inputs, len(omega)), dtype=float)*sys
        return FRD(frdata, omega, smooth=True)

    # try converting constant matrices
    try:
        sys = array(sys)
        outputs, inputs = sys.shape
        frdata = empty((outputs, inputs, len(omega)), dtype=float)
        for i in range(outputs):
            for j in range(inputs):
                frdata[i, j, :] = sys[i, j]
        return FRD(frdata, omega, smooth=True)
    except Exception:
        pass

    raise TypeError("Can't convert given type '%s' to FRD system." %
                    sys.__class__)


def frd(*args, **kwargs):
    """frd(frdata, omega[, dt])

    Construct a frequency response data (FRD) model.

    A frequency response data model stores the (measured) frequency response
    of a system.  This factory function can be called in different ways:

    ``frd(frdata, omega)``

        Create an frd model with the given response data, in the form of
        complex response vector, at matching frequencies `omega` [in rad/s].

    ``frd(sys, omega)``

        Convert an LTI system into an frd model with data at frequencies
        `omega`.

    Parameters
    ----------
    frdata : array_like or LTI system
        Complex vector with the system response or an LTI system that can
        be used to compute the frequency response at a list of frequencies.
    sys : `StateSpace` or `TransferFunction`
        A linear system that will be evaluated for frequency response data.
    omega : array_like
        Vector of frequencies at which the response is evaluated.
    dt : float, True, or None
        System timebase.
    smooth : bool, optional
        If True, create an interpolation function that allows the
        frequency response to be computed at any frequency within the range
        of frequencies give in `omega`.  If False (default),
        frequency response can only be obtained at the frequencies
        specified in `omega`.

    Returns
    -------
    sys : `FrequencyResponseData`
        New frequency response data system.

    Other Parameters
    ----------------
    inputs, outputs : str, or list of str, optional
        List of strings that name the individual signals of the transformed
        system.  If not given, the inputs and outputs are the same as the
        original system.
    input_prefix, output_prefix : string, optional
        Set the prefix for input and output signals.  Defaults = 'u', 'y'.
    name : string, optional
        Set the name of the system. If unspecified and the system is
        sampled from an existing system, the new system name is determined
        by adding the prefix and suffix strings in
        `config.defaults['iosys.sampled_system_name_prefix']` and
        `config.defaults['iosys.sampled_system_name_suffix']`, with the
        default being to add the suffix '$sampled'.  Otherwise, a generic
        name 'sys[id]' is generated with a unique integer id

    See Also
    --------
    FrequencyResponseData, frequency_response, ss, tf

    Examples
    --------
    >>> # Create from measurements
    >>> response = [1.0, 1.0, 0.5]
    >>> omega = [1, 10, 100]
    >>> F = ct.frd(response, omega)

    >>> G = ct.tf([1], [1, 1])
    >>> omega = [1, 10, 100]
    >>> F = ct.frd(G, omega)

    """
    return FrequencyResponseData(*args, **kwargs)
