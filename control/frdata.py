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

from __future__ import division

"""
Frequency response data representation and functions.

This module contains the FRD class and also functions that operate on
FRD data.
"""

# External function declarations
from warnings import warn
import numpy as np
from numpy import angle, array, empty, ones, isin, \
    real, imag, absolute, eye, linalg, where, dot, sort
from scipy.interpolate import splprep, splev
from .lti import LTI

__all__ = ['FrequencyResponseData', 'FRD', 'frd']


class FrequencyResponseData(LTI):
    """FrequencyResponseData(d, w)

    A class for models defined by frequency response data (FRD)

    The FrequencyResponseData (FRD) class is used to represent systems in
    frequency response data form.

    The main data members are 'omega' and 'fresp', where `omega` is a 1D array
    with the frequency points of the response, and `fresp` is a 3D array, with
    the first dimension corresponding to the output index of the FRD, the
    second dimension corresponding to the input index, and the 3rd dimension
    corresponding to the frequency points in omega.  For example,

    >>> frdata[2,5,:] = numpy.array([1., 0.8-0.2j, 0.2-0.8j])

    means that the frequency response from the 6th input to the 3rd
    output at the frequencies defined in omega is set to the array
    above, i.e. the rows represent the outputs and the columns
    represent the inputs.

    """

    # Allow NDarray * StateSpace to give StateSpace._rmul_() priority
    # https://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    __array_priority__ = 11     # override ndarray and matrix types

    epsw = 1e-8

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
        smooth = kwargs.get('smooth', False)

        if len(args) == 2:
            if not isinstance(args[0], FRD) and isinstance(args[0], LTI):
                # not an FRD, but still a system, second argument should be
                # the frequency range
                otherlti = args[0]
                self.omega = sort(np.asarray(args[1], dtype=float))
                numfreq = len(self.omega)
                # calculate frequency response at my points
                self.fresp = otherlti(1j * self.omega, squeeze=False)
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
            raise ValueError("Needs 1 or 2 arguments; received %i." % len(args))

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

        mimo = self.inputs > 1 or self.outputs > 1
        outstr = ['Frequency response data']

        #mt, pt, wt = self.frequency_response(self.omega)
        for i in range(self.inputs):
            for j in range(self.outputs):
                if mimo:
                    outstr.append("Input %i to output %i:" % (i + 1, j + 1))
                outstr.append('Freq [rad/s]  Response')
                outstr.append('------------  ---------------------')
                outstr.extend(
                    ['%12.3f  %10.4g%+10.4gj' % (w, m, p)
                     for m, p, w in zip(real(self.fresp[j, i, :]),
                                        imag(self.fresp[j, i, :]), self.omega)])

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
            other = _convertToFRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.inputs != other.outputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (self.inputs, other.outputs))

        inputs = other.inputs
        outputs = self.outputs
        fresp = empty((outputs, inputs, len(self.omega)),
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):
            fresp[:, :, i] = dot(self.fresp[:, :, i], other.fresp[:, :, i])
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
            other = _convertToFRD(other, omega=self.omega)

        # Check that the input-output sizes are consistent.
        if self.outputs != other.inputs:
            raise ValueError(
                "H = G1*G2: input-output size mismatch: "
                "G1 has %i input(s), G2 has %i output(s)." %
                (other.inputs, self.outputs))

        inputs = self.inputs
        outputs = other.outputs

        fresp = empty((outputs, inputs, len(self.omega)),
                      dtype=self.fresp.dtype)
        for i in range(len(self.omega)):
            fresp[:, :, i] = dot(other.fresp[:, :, i], self.fresp[:, :, i])
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
            other = _convertToFRD(other, omega=self.omega)

        if (self.inputs > 1 or self.outputs > 1 or
            other.inputs > 1 or other.outputs > 1):
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
            other = _convertToFRD(other, omega=self.omega)

        if (self.inputs > 1 or self.outputs > 1 or
            other.inputs > 1 or other.outputs > 1):
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
    def eval(self, omega, squeeze=True):
        """Evaluate a transfer function at a single angular frequency.

        self.evalfr(omega) returns the value of the frequency response
        at frequency omega.

        Note that a "normal" FRD only returns values for which there is an
        entry in the omega vector. An interpolating FRD can return
        intermediate values.     
        
        squeeze: bool, optional (default=True)
            If True and sys is single input, single output (SISO), return a 
            1D array or scalar depending on omega's length.

        """
        omega_array = np.array(omega, ndmin=1) # array-like version of omega 
        if any(omega_array.imag > 0): 
            raise ValueError("FRD.eval can only accept real-valued omega")

        if self.ifunc is None:
            elements = isin(self.omega, omega)
            if sum(elements) < len(omega_array):
                raise ValueError(
                    "not all frequencies omega are in frequency list of FRD "
                    "system. Try an interpolating FRD for additional points.")
            else:
                out = self.fresp[:, :, elements]
        else:
            out = empty((self.outputs, self.inputs, len(omega_array)))
            for i in range(self.outputs):
                for j in range(self.inputs):
                    for k, w in enumerate(omega_array):
                        frraw = splev(w, self.ifunc[i, j], der=0)
                        out[i, j, k] = frraw[0] + 1.0j * frraw[1]
        if not hasattr(omega, '__len__'):
            # omega is a scalar, squeeze down array along last dim
            out = np.squeeze(out, axis=2)
        if squeeze and self.issiso():
            out = out[0][0]
        return out

    def __call__(self, s, squeeze=True):
        """Evaluate the system's transfer function at complex frequencies s.

        For a SISO system, returns the complex value of the
        transfer function.  For a MIMO transfer fuction, returns a
        matrix of values.
        
        Raises an error if s is not purely imaginary. 
        
        squeeze: bool, optional (default=True)
            If True and sys is single input, single output (SISO), return a 
            1D array or scalar depending on omega's length.

        """
        if any(abs(np.array(s, ndmin=1).real) > 0):
            raise ValueError("__call__: FRD systems can only accept"
                            "purely imaginary frequencies")
        # need to preserve array or scalar status
        if hasattr(s, '__len__'):
            return self.eval(np.asarray(s).imag, squeeze=squeeze)
        else:
            return self.eval(complex(s).imag, squeeze=squeeze)
        
    def freqresp(self, omega):
        warn("FrequencyResponseData.freqresp(omega) will be deprecated in a "
             "future release of python-control; use "
             "FrequencyResponseData.frequency_response(sys, omega), or "
             "freqresp(sys, omega) in the MATLAB compatibility module "
             "instead", PendingDeprecationWarning)        
        return self.frequency_response(omega)

    def feedback(self, other=1, sign=-1):
        """Feedback interconnection between two FRD objects."""

        other = _convertToFRD(other, omega=self.omega)

        if (self.outputs != other.inputs or self.inputs != other.outputs):
            raise ValueError(
                "FRD.feedback, inputs/outputs mismatch")
        fresp = empty((self.outputs, self.inputs, len(other.omega)),
                      dtype=complex)
        # TODO: vectorize this
        # TODO: handle omega re-mapping
        # TODO: is there a reason to use linalg.solve instead of linalg.inv?
        # https://github.com/python-control/python-control/pull/314#discussion_r294075154
        for k, w in enumerate(other.omega):
            fresp[:, :, k] = np.dot(
                self.fresp[:, :, k],
                linalg.solve(
                    eye(self.inputs)
                    + np.dot(other.fresp[:, :, k], self.fresp[:, :, k]),
                    eye(self.inputs))
            )

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


def _convertToFRD(sys, omega, inputs=1, outputs=1):
    """Convert a system to frequency response data form (if needed).

    If sys is already an frd, and its frequency range matches or
    overlaps the range given in omega then it is returned.  If sys is
    another LTI object or a transfer function, then it is converted to
    a frequency response data at the specified omega. If sys is a
    scalar, then the number of inputs and outputs can be specified
    manually, as in:

    >>> frd = _convertToFRD(3., omega) # Assumes inputs = outputs = 1
    >>> frd = _convertToFRD(1., omegs, inputs=3, outputs=2)

    In the latter example, sys's matrix transfer function is [[1., 1., 1.]
                                                              [1., 1., 1.]].

    """

    if isinstance(sys, FRD):
        omega.sort()
        if len(omega) == len(sys.omega) and \
           (abs(omega - sys.omega) < FRD.epsw).all():
            # frequencies match, and system was already frd; simply use
            return sys

        raise NotImplementedError(
            "Frequency ranges of FRD do not match, conversion not implemented")

    elif isinstance(sys, LTI):
        omega = np.sort(omega)
        fresp = sys(1j * omega)
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
