# poly.m - simple set of polynomial basis functions
# TODO: rename this as taylor.m
# RMM, 10 Nov 2012
#
# This class implements a set of simple basis functions consisting of powers
# of t: 1, t, t^2, ...
#
# Copyright (c) 2012 by California Institute of Technology
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

import numpy as np
from scipy.special import factorial
from .basis import BasisFamily

class PolyFamily(BasisFamily):
    r"""Polynomial basis functions.

    This class represents the family of polynomials of the form

    .. math::
         \phi_i(t) = t^i

    """
    def __init__(self, N):
        """Create a polynomial basis of order N."""
        super(PolyFamily, self).__init__(N)

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t):
        """Evaluate the kth derivative of the ith basis function at time t."""
        if (i < k): return 0;           # higher derivative than power
        return factorial(i)/factorial(i-k) * np.power(t, i-k)
