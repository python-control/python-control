# poly.m - simple set of polynomial basis functions
# RMM, 10 Nov 2012
#
# TODO: rename this as taylor.m?

"""Simple set of polynomial basis functions.

This class implements a set of simple basis functions consisting of
powers of t: 1, t, t^2, ...

"""

import numpy as np
from scipy.special import factorial

from .basis import BasisFamily


class PolyFamily(BasisFamily):
    r"""Polynomial basis functions.

    This class represents the family of polynomials of the form

    .. math::
         \phi_i(t) = \left( \frac{t}{T} \right)^i

    Parameters
    ----------
    N : int
        Degree of the polynomial.

    T : float
        Final time (used for rescaling).  Default value is 1.

    """
    def __init__(self, N, T=1):
        """Create a polynomial basis of order N."""
        super(PolyFamily, self).__init__(N)
        self.T = float(T)       # save end of time interval

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t, var=None):
        """Evaluate kth derivative of ith basis function at time t.

        See `BasisFamily.eval_deriv` for more information.

        """
        if (i < k): return 0 * t        # higher derivative than power
        return factorial(i)/factorial(i-k) * \
            np.power(t/self.T, i-k) / np.power(self.T, k)
