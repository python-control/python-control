# Copyright (c) 2010 by California Institute of Technology
# Copyright (c) 2012 by Delft University of Technology
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
# 3. Neither the names of the California Institute of Technology nor
#    the Delft University of Technology nor
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
# Author: M.M. (Rene) van Paassen (using xferfcn.py as basis)
# Date: 02 Oct 12


"""
Frequency response data representation and functions.

This module contains the FRD class and also functions that operate on
FRD data.
"""

# External function declarations
from warnings import warn
import numpy as np
from numpy import angle, array, empty, ones, \
    real, imag, absolute, eye, linalg, where, sort
from scipy.interpolate import splprep, splev
from .lti import LTI, _process_frequency_response
from . import config

__all__ = ['FrequencyResponseData', 'FRD', 'frd']


class FrequencyResponseData(LTI):
    """FrequencyResponseData(d, w[, smooth])

    A class for models defined by frequency response data (FRD).

    The FrequencyResponseData (FRD) class is used to represent systems in
    frequency response data form.

    Parameters
    ----------
    d : 1D or 3D complex array_like
        The frequency response at each frequency point.  If 1D, the system is
        assumed to be SISO.  If 3D, the system is MIMO, with the first
        dimension corresponding to the output index of the FRD, the second
        dimension corresponding to the input index, and the 3rd dimension
        corresponding to the frequency points in omega
    w : iterable of real frequencies
        List of frequency points for which data are available.
    smooth : bool, optional
        If ``True``, create an interpolation function that allows the
        frequency response to be computed at any frequency within the range of
        frequencies give in ``w``.  If ``False`` (default), frequency response
        can only be obtained at the frequencies specified in ``w``.

    Attributes
    ----------
    ninputs, noutputs : int
        Number of input and output variables.
    omega : 1D array
        Frequency points of the response.
    fresp : 3D array
        Frequency response, indexed by output index, input index, and
        frequency point.

    Notes
    -----
    The main data members are 'omega' and 'fresp', where 'omega' is a 1D array
    of frequency points and and 'fresp' is a 3D array of frequency responses,
    with the first dimension corresponding to the output index of the FRD, the
    second dimension corresponding to the input index, and the 3rd dimension
    corresponding to the frequency points in omega.  For example,

    >>> frdata[2,5,:] = numpy.array([1., 0.8-0.2j, 0.2-0.8j])

    means that the frequency response from the 6th input to the 3rd output at
    the frequencies defined in omega is set to the array above, i.e. the rows
    represent the outputs and the columns represent the inputs.

    A frequency response data object is callable and returns the value of the
    transfer function evaluated at a point in the complex plane (must be on
    the imaginary access).  See :meth:`~control.FrequencyResponseData.__call__`
    for a more detailed description.

    """

    # Allow NDarray * StateSpace to give StateSpace._rmul_() priority
    # https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    __array_priority__ = 11     # override ndarray and matrix types

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

    _epsw = 1e-8                #: Bound for exact frequency match

    def __init__(self, *args, **kwargs):
        """Construct an FRD object.

        The default constructor is FRD(d, w), where w is an iterable of
        frequency points, and d is the matching frequency data.

        If d is a single list, 1d array, or tuple, a SISO system description
        is assumed. d can also be

        To call the copy constructor, call FRD(sys), where sys is a
        FRD object.

        To construct frequency response data for an existing LTI
        object, other than an FRD, call FRD(sys, omega)

        """
        # TODO: discrete-time FRD systems?
        smooth = kwargs.get('smooth', False)

        if len(args) == 2:
            if not isinstance(args[0], FRD) and isinstance(args[0], LTI):
                # not an FRD, but still a system, second argument should be
                # the frequency range
                otherlti = args[0]
                self.omega = sort(np.asarray(args[1], dtype=float))
                # calculate frequency response at my points
                if otherlti.isctime():
                    s = 1j * self.omega
                    self.fresp = otherlti(s, squeeze=False)
                else:
                    z = np.exp(1j * self.omega * otherlti.dt)
                    self.fresp = otherlti(z, squeeze=False)

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
            raise ValueError(
                "Needs 1 or 2 arguments; received %i." % len(args))

        # create interpolation functions
        if smooth:
            self.ifunc = empty((self.fresp.shape[0], self.fresp.shape[1]),
                               dtype=tuple)
            for i in range(self.fresp.shape[0]):
                for j in range(self.fresp.shape[1]):
                    self.ifunc[i, j], u = splprep(
                        u=self.omega, x=[real(self.fresp[i, j, :]),
                                         imag(self.fresp[i, j, :])],
                        w=1.0/(absolute(self.fresp[i, j, :]) + 0.001), s=0.0)
        else:
            self.ifunc = None
        LTI.__init__(self, self.fresp.shape[1], self.fresp.shape[0])

    def __str__(self):
        """String representation of the transfer function."""

        mimo = self.ninputs > 1 or self.noutputs > 1
        outstr = ['Frequency response data']

        for i in range(self.ninputs):
            for j in range(self.noutputs):
                if mimo:
                    outstr.append("Input %i to output %i:" % (i + 1, j + 1))
                outstr.append('Freq [rad/s]  Response')
                outstr.append('------------  ---------------------')
                outstr.extend(
                    ['%12.3f  %10.4g%+10.4gj' % (w, re, im)
                     for w, re, im in zip(self.omega,
                                          real(self.fresp[j, i, :]),
                                          imag(self.fresp[j, i, :]))])

        return '\n'.join(outstr)

    def __repr__(self):
        """Loadable string representation,

        limited for number of data points.
        """
        return "FrequencyResponseData({d}, {w}{smooth})".format(
            d=repr(self.fresp), w=repr(self.omega),
            smooth=(self.ifunc and ", smooth=True") or "")

    def __neg__(self):
        """Negate a transfer function."""

        return FRD(-self.fresp, self.omega)

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
        other = _convert_to_FRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.ninputs:
            raise ValueError("The first summand has %i input(s), but the \
second has %i." % (self.ninputs, other.ninputs))
        if self.noutputs != other.noutputs:
            raise ValueError("The first summand has %i output(s), but the \
second has %i." % (self.noutputs, other.noutputs))

        return FRD(self.fresp + other.fresp, other.omega)

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
            return FRD(self.fresp * other, self.omega,
                       smooth=(self.ifunc is not None))
        else:
            other = _convert_to_FRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.ninputs != other.noutputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (self.ninputs, other.noutputs))

        inputs = other.ninputs
        outputs = self.noutputs
        fresp = empty((outputs, inputs, len(self.omega)),
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):
            fresp[:, :, i] = self.fresp[:, :, i] @ other.fresp[:, :, i]
        return FRD(fresp, self.omega,
                   smooth=(self.ifunc is not None) and
                          (other.ifunc is not None))

    def __rmul__(self, other):
        """Right Multiply two LTI objects (serial connection)."""

        # Convert the second argument to an frd function.
        if isinstance(other, (int, float, complex, np.number)):
            return FRD(self.fresp * other, self.omega,
                       smooth=(self.ifunc is not None))
        else:
            other = _convert_to_FRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.noutputs != other.ninputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (other.ninputs, self.noutputs))

        inputs = self.ninputs
        outputs = other.noutputs

        fresp = empty((outputs, inputs, len(self.omega)),
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):
            fresp[:, :, i] = other.fresp[:, :, i] @ self.fresp[:, :, i]
        return FRD(fresp, self.omega,
                   smooth=(self.ifunc is not None) and
                          (other.ifunc is not None))

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __truediv__(self, other):
        """Divide two LTI objects."""

        if isinstance(other, (int, float, complex, np.number)):
            return FRD(self.fresp * (1/other), self.omega,
                       smooth=(self.ifunc is not None))
        else:
            other = _convert_to_FRD(other, omega=self.omega)

        if (self.ninputs > 1 or self.noutputs > 1 or
            other.ninputs > 1 or other.noutputs > 1):
            raise NotImplementedError(
                "FRD.__truediv__ is currently only implemented for SISO "
                "systems.")

        return FRD(self.fresp/other.fresp, self.omega,
                   smooth=(self.ifunc is not None) and
                          (other.ifunc is not None))

    # TODO: Remove when transition to python3 complete
    def __div__(self, other):
        return self.__truediv__(other)

    # TODO: Division of MIMO transfer function objects is not written yet.
    def __rtruediv__(self, other):
        """Right divide two LTI objects."""
        if isinstance(other, (int, float, complex, np.number)):
            return FRD(other / self.fresp, self.omega,
                       smooth=(self.ifunc is not None))
        else:
            other = _convert_to_FRD(other, omega=self.omega)

        if (self.ninputs > 1 or self.noutputs > 1 or
            other.ninputs > 1 or other.noutputs > 1):
            raise NotImplementedError(
                "FRD.__rtruediv__ is currently only implemented for "
                "SISO systems.")

        return other / self

    # TODO: Remove when transition to python3 complete
    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __pow__(self, other):
        if not type(other) == int:
            raise ValueError("Exponent must be an integer")
        if other == 0:
            return FRD(ones(self.fresp.shape), self.omega,
                       smooth=(self.ifunc is not None))  # unity
        if other > 0:
            return self * (self**(other-1))
        if other < 0:
            return (FRD(ones(self.fresp.shape), self.omega) / self) * \
                (self**(other+1))

    # Define the `eval` function to evaluate an FRD at a given (real)
    # frequency.  Note that we choose to use `eval` instead of `evalfr` to
    # avoid confusion with :func:`evalfr`, which takes a complex number as its
    # argument.  Similarly, we don't use `__call__` to avoid confusion between
    # G(s) for a transfer function and G(omega) for an FRD object.
    # update Sawyer B. Fuller 2020.08.14: __call__ added to provide a uniform
    # interface to systems in general and the lti.frequency_response method
    def eval(self, omega, squeeze=None):
        """Evaluate a transfer function at angular frequency omega.

        Note that a "normal" FRD only returns values for which there is an
        entry in the omega vector. An interpolating FRD can return
        intermediate values.

        Parameters
        ----------
        omega : float or 1D array_like
            Frequencies in radians per second
        squeeze : bool, optional
            If squeeze=True, remove single-dimensional entries from the shape
            of the output even if the system is not SISO. If squeeze=False,
            keep all indices (output, input and, if omega is array_like,
            frequency) even if the system is SISO. The default value can be
            set using config.defaults['control.squeeze_frequency_response'].

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
        omega_array = np.array(omega, ndmin=1)  # array-like version of omega

        # Make sure that we are operating on a simple list
        if len(omega_array.shape) > 1:
            raise ValueError("input list must be 1D")

        # Make sure that frequencies are all real-valued
        if any(omega_array.imag > 0):
            raise ValueError("FRD.eval can only accept real-valued omega")

        if self.ifunc is None:
            elements = np.isin(self.omega, omega)  # binary array
            if sum(elements) < len(omega_array):
                raise ValueError(
                    "not all frequencies omega are in frequency list of FRD "
                    "system. Try an interpolating FRD for additional points.")
            else:
                out = self.fresp[:, :, elements]
        else:
            out = empty((self.noutputs, self.ninputs, len(omega_array)),
                        dtype=complex)
            for i in range(self.noutputs):
                for j in range(self.ninputs):
                    for k, w in enumerate(omega_array):
                        frraw = splev(w, self.ifunc[i, j], der=0)
                        out[i, j, k] = frraw[0] + 1.0j * frraw[1]

        return _process_frequency_response(self, omega, out, squeeze=squeeze)

    def __call__(self, s, squeeze=None):
        """Evaluate system's transfer function at complex frequencies.

        Returns the complex frequency response `sys(s)` of system `sys` with
        `m = sys.ninputs` number of inputs and `p = sys.noutputs` number of
        outputs.

        To evaluate at a frequency omega in radians per second, enter
        ``s = omega * 1j`` or use ``sys.eval(omega)``

        For a frequency response data object, the argument must be an
        imaginary number (since only the frequency response is defined).

        Parameters
        ----------
        s : complex scalar or 1D array_like
            Complex frequencies
        squeeze : bool, optional (default=True)
            If squeeze=True, remove single-dimensional entries from the shape
            of the output even if the system is not SISO. If squeeze=False,
            keep all indices (output, input and, if omega is array_like,
            frequency) even if the system is SISO. The default value can be
            set using config.defaults['control.squeeze_frequency_response'].

        Returns
        -------
        fresp : complex ndarray
            The frequency response of the system.  If the system is SISO and
            squeeze is not True, the shape of the array matches the shape of
            omega.  If the system is not SISO or squeeze is False, the first
            two dimensions of the array are indices for the output and input
            and the remaining dimensions match omega.  If ``squeeze`` is True
            then single-dimensional axes are removed.

        Raises
        ------
        ValueError
            If `s` is not purely imaginary, because
            :class:`FrequencyDomainData` systems are only defined at imaginary
            frequency values.

        """
        # Make sure that we are operating on a simple list
        if len(np.atleast_1d(s).shape) > 1:
            raise ValueError("input list must be 1D")

        if any(abs(np.atleast_1d(s).real) > 0):
            raise ValueError("__call__: FRD systems can only accept "
                             "purely imaginary frequencies")

        # need to preserve array or scalar status
        if hasattr(s, '__len__'):
            return self.eval(np.asarray(s).imag, squeeze=squeeze)
        else:
            return self.eval(complex(s).imag, squeeze=squeeze)

    def freqresp(self, omega):
        """(deprecated) Evaluate transfer function at complex frequencies.

        .. deprecated::0.9.0
            Method has been given the more pythonic name
            :meth:`FrequencyResponseData.frequency_response`. Or use
            :func:`freqresp` in the MATLAB compatibility module.
        """
        warn("FrequencyResponseData.freqresp(omega) will be removed in a "
             "future release of python-control; use "
             "FrequencyResponseData.frequency_response(omega), or "
             "freqresp(sys, omega) in the MATLAB compatibility module "
             "instead", DeprecationWarning)
        return self.frequency_response(omega)

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two FRD objects."""

        other = _convert_to_FRD(other, omega=self.omega)

        if (self.noutputs != other.ninputs or self.ninputs != other.noutputs):
            raise ValueError(
                "FRD.feedback, inputs/outputs mismatch")

        # TODO: handle omega re-mapping

        # reorder array axes in order to leverage numpy broadcasting
        myfresp = np.moveaxis(self.fresp, 2, 0)
        otherfresp = np.moveaxis(other.fresp, 2, 0)
        I_AB = eye(self.ninputs)[np.newaxis, :, :] + otherfresp @ myfresp
        resfresp = (myfresp @ linalg.inv(I_AB))
        fresp = np.moveaxis(resfresp, 0, 2)

        return FRD(fresp, other.omega, smooth=(self.ifunc is not None))

#
# Allow FRD as an alias for the FrequencyResponseData class
#
# Note: This class was initially given the name "FRD", but this caused
# problems with documentation on MacOS platforms, since files were generated
# for control.frd and control.FRD, which are not differentiated on most MacOS
# filesystems, which are case insensitive.  Renaming the FRD class to be
# FrequenceResponseData and then assigning FRD to point to the same object
# fixes this problem.
#


FRD = FrequencyResponseData


def _convert_to_FRD(sys, omega, inputs=1, outputs=1):
    """Convert a system to frequency response data form (if needed).

    If sys is already an frd, and its frequency range matches or
    overlaps the range given in omega then it is returned.  If sys is
    another LTI object or a transfer function, then it is converted to
    a frequency response data at the specified omega. If sys is a
    scalar, then the number of inputs and outputs can be specified
    manually, as in:

    >>> frd = _convert_to_FRD(3., omega) # Assumes inputs = outputs = 1
    >>> frd = _convert_to_FRD(1., omegs, inputs=3, outputs=2)

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
            fresp = sys(1j * omega)
        else:
            fresp = sys(np.exp(1j * omega * sys.dt))
        if len(fresp.shape) == 1:
            fresp = fresp[np.newaxis, np.newaxis, :]
        return FRD(fresp, omega, smooth=True)

    elif isinstance(sys, (int, float, complex, np.number)):
        fresp = ones((outputs, inputs, len(omega)), dtype=float)*sys
        return FRD(fresp, omega, smooth=True)

    # try converting constant matrices
    try:
        sys = array(sys)
        outputs, inputs = sys.shape
        fresp = empty((outputs, inputs, len(omega)), dtype=float)
        for i in range(outputs):
            for j in range(inputs):
                fresp[i, j, :] = sys[i, j]
        return FRD(fresp, omega, smooth=True)
    except Exception:
        pass

    raise TypeError('''Can't convert given type "%s" to FRD system.''' %
                    sys.__class__)


def frd(*args):
    """frd(d, w)

    Construct a frequency response data model

    frd models store the (measured) frequency response of a system.

    This function can be called in different ways:

    ``frd(response, freqs)``
        Create an frd model with the given response data, in the form of
        complex response vector, at matching frequency freqs [in rad/s]

    ``frd(sys, freqs)``
        Convert an LTI system into an frd model with data at frequencies
        freqs.

    Parameters
    ----------
    response: array_like, or list
        complex vector with the system response
    freq: array_lik or lis
        vector with frequencies
    sys: LTI (StateSpace or TransferFunction)
        A linear system

    Returns
    -------
    sys: FRD
        New frequency response system

    See Also
    --------
    FRD, ss, tf
    """
    return FRD(*args)
