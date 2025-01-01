# basis.py - BasisFamily class
# RMM, 10 Nov 2012

"""The :mod:`control.flatsys.basis` module defines the
`flatsys.BasisFamily` class that used to specify a set of basis
functions for implementing differential flatness computations.

"""

import numpy as np


# Basis family class (for use as a base class)
class BasisFamily:
    """Base class for basis functions for flat systems.

    A BasisFamily object is used to construct trajectories for a flat system.
    The class must implement a single function that computes the jth
    derivative of the ith basis function at a time t:

      :math:`z_i^{(q)}(t)` = basis.eval_deriv(self, i, j, t)

    A basis set can either consist of a single variable that is used for
    each flat output (nvars = None) or a different variable for different
    flat outputs (nvars > 0).

    Attributes
    ----------
    N : int
        Order of the basis set.

    """
    def __init__(self, N):
        """Create a basis family of order N."""
        self.N = N                    # save number of basis functions
        self.nvars = None             # default number of variables
        self.coef_offset = [0]        # coefficient offset for each variable
        self.coef_length = [N]        # coefficient length for each variable

    def __repr__(self):
        return f'<{self.__class__.__name__}: nvars={self.nvars}, ' + \
            f'N={self.N}>'

    def __call__(self, i, t, var=None):
        """Evaluate the ith basis function at a point in time."""
        return self.eval_deriv(i, 0, t, var=var)

    def var_ncoefs(self, var):
        """Get the number of coefficients for a variable."""
        return self.N if self.nvars is None else self.coef_length[var]

    def eval(self, coeffs, tlist, var=None):
        """Compute function values given the coefficients and time points."""
        if self.nvars is None and var != None:
            raise SystemError("multi-variable call to a scalar basis")

        elif self.nvars is None:
            # Single variable basis
            return [
                sum([coeffs[i] * self(i, t) for i in range(self.N)])
                for t in tlist]

        elif var is None:
            # Multi-variable basis with single list of coefficients
            values = np.empty((self.nvars, tlist.size))
            offset = 0
            for j in range(self.nvars):
                coef_len = self.var_ncoefs(j)
                values[j] = np.array([
                    sum([coeffs[offset + i] * self(i, t, var=j)
                         for i in range(coef_len)])
                    for t in tlist])
                offset += coef_len
            return values

        else:
            return np.array([
                sum([coeffs[i] * self(i, t, var=var)
                     for i in range(self.var_ncoefs(var))])
                for t in tlist])

    def eval_deriv(self, i, j, t, var=None):
        """Evaluate the kth derivative of the ith basis function at time t."""
        raise NotImplementedError("Internal error; improper basis functions")
