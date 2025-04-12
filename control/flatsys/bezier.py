# bezier.m - 1D Bezier curve basis functions
# RMM, 24 Feb 2021

r"""1D Bezier curve basis functions.

This module defines the `BezierFamily` class, which implements a set of
basis functions based on Bezier curves:

.. math:: \phi_i(t) = \sum_{i=0}^n {n \choose i} (T - t)^{n-i} t^i

"""

import numpy as np
from scipy.special import binom, factorial

from .basis import BasisFamily


class BezierFamily(BasisFamily):
    r"""Bezier curve basis functions.

    This class represents the family of polynomials of the form

    .. math::
         \phi_i(t) = \sum_{i=0}^N {N \choose i}
             \left( \frac{t}{T} - t \right)^{N-i}
             \left( \frac{t}{T} \right)^i

    Parameters
    ----------
    N : int
        Degree of the Bezier curve.

    T : float
        Final time (used for rescaling).  Default value is 1.

    """
    def __init__(self, N, T=1):
        """Create a polynomial basis of order N."""
        super(BezierFamily, self).__init__(N)
        self.T = float(T)       # save end of time interval

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t, var=None):
        """Evaluate kth derivative of ith basis function at time t.

        See `BasisFamily.eval_deriv` for more information.

        """
        if i >= self.N:
            raise ValueError("Basis function index too high")
        elif k >= self.N:
            # Higher order derivatives are zero
            return 0 * t

        # Compute the variables used in Bezier curve formulas
        n = self.N - 1
        u = t/self.T

        if k == 0:
            # No derivative => avoid expansion for speed
            return binom(n, i) * u**i * (1-u)**(n-i)

        # Return the kth derivative of the ith Bezier basis function
        return binom(n, i) * sum([
            (-1)**(j-i) *
            binom(n-i, j-i) * factorial(j)/factorial(j-k) * \
            np.power(u, j-k) / np.power(self.T, k)
            for j in range(max(i, k), n+1)
        ])
