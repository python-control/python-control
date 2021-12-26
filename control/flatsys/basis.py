# basis.py - BasisFamily class
# RMM, 10 Nov 2012
#
# The BasisFamily class is used to specify a set of basis functions for
# implementing differential flatness computations.
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


# Basis family class (for use as a base class)
class BasisFamily:
    """Base class for implementing basis functions for flat systems.

    A BasisFamily object is used to construct trajectories for a flat system.
    The class must implement a single function that computes the jth
    derivative of the ith basis function at a time t:

      :math:`z_i^{(q)}(t)` = basis.eval_deriv(self, i, j, t)

    Parameters
    ----------
    N : int
        Order of the basis set.

    """
    def __init__(self, N):
        """Create a basis family of order N."""
        self.N = N                    # save number of basis functions

    def __call__(self, i, t):
        """Evaluate the ith basis function at a point in time"""
        return self.eval_deriv(i, 0, t)

    def eval_deriv(self, i, j, t):
        raise NotImplementedError("Internal error; improper basis functions")
