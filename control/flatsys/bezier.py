# bezier.m - 1D Bezier curve basis functions
# RMM, 24 Feb 2021
#
# This class implements a set of basis functions based on Bezier curves:
#
#   \phi_i(t) = \sum_{i=0}^n {n \choose i} (T - t)^{n-i} t^i
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
from scipy.special import binom, factorial
from .basis import BasisFamily

class BezierFamily(BasisFamily):
    r"""Bezier curve basis functions.

    This class represents the family of polynomials of the form

    .. math::
         \phi_i(t) = \sum_{i=0}^n {n \choose i}
             \left( \frac{t}{T_\text{f}} - t \right)^{n-i}
             \left( \frac{t}{T_f} \right)^i

    """
    def __init__(self, N, T=1):
        """Create a polynomial basis of order N."""
        self.N = N                      # save number of basis functions
        self.T = T                      # save end of time interval

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t):
        """Evaluate the kth derivative of the ith basis function at time t."""
        if i >= self.N:
            raise ValueError("Basis function index too high")
        elif k >= self.N:
            # Higher order derivatives are zero
            return np.zeros(t.shape)

        # Compute the variables used in Bezier curve formulas
        n = self.N - 1
        u = t/self.T

        if k == 0:
            # No derivative => avoid expansion for speed
            return binom(n, i) * u**i * (1-u)**(n-i)

        # Return the kth derivative of the ith Bezier basis function
        return binom(n, i) * sum([
            (-1)**(j-i) *
            binom(n-i, j-i) * factorial(j)/factorial(j-k) * np.power(u, j-k)
            for j in range(max(i, k), n+1)
        ])
