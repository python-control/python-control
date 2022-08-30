# bspline.py - B-spline basis functions
# RMM, 2 Aug 2022
#
# This class implements a set of B-spline basis functions that implement a
# piecewise polynomial at a set of breakpoints t0, ..., tn with given orders
# and smoothness.
#

import numpy as np
from .basis import BasisFamily
from scipy.interpolate import BSpline, splev

class BSplineFamily(BasisFamily):
    """B-spline basis functions.

    This class represents a B-spline basis for piecewise polynomials defined
    across a set of breakpoints with given degree and smoothness.  On each
    interval between two breakpoints, we have a polynomial of a given degree
    and the spline is continuous up to a given smoothness at interior
    breakpoints.

    Parameters
    ----------
    breakpoints : 1D array or 2D array of float
        The breakpoints for the spline(s).

    degree : int or list of ints
        For each spline variable, the degree of the polynomial between
        breakpoints.  If a single number is given and more than one spline
        variable is specified, the same degree is used for each spline
        variable.

    smoothness : int or list of ints
        For each spline variable, the smoothness at breakpoints (number of
        derivatives that should match).

    vars : None or int, optional
        The number of spline variables.  If specified as None (default),
        then the spline basis describes a single variable, with no indexing.
        If the number of spine variables is > 0, then the spline basis is
        index using the `var` keyword.

    """
    def __init__(self, breakpoints, degree, smoothness=None, vars=None):
        """Create a B-spline basis for piecewise smooth polynomials."""
        # Process the breakpoints for the spline */
        breakpoints = np.array(breakpoints, dtype=float)
        if breakpoints.ndim == 2:
            raise NotImplementedError(
                "breakpoints for each spline variable not yet supported")
        elif breakpoints.ndim != 1:
            raise ValueError("breakpoints must be convertable to a 1D array")
        elif breakpoints.size < 2:
            raise ValueError("break point vector must have at least 2 values")
        elif np.any(np.diff(breakpoints) <= 0):
            raise ValueError("break points must be strictly increasing values")

        # Decide on the number of spline variables
        if vars is None:
            nvars = 1
            self.nvars = None           # track as single variable
        elif not isinstance(vars, int):
            raise TypeError("vars must be an integer")
        else:
            nvars = vars
            self.nvars = nvars

        #
        # Process B-spline parameters (degree, smoothness)
        #
        # B-splines are defined on a set of intervals separated by
        # breakpoints.  On each interval we have a polynomial of a certain
        # degree and the spline is continuous up to a given smoothness at
        # breakpoints.  The code in this section allows some flexibility in
        # the way that all of this information is supplied, including using
        # scalar values for parameters (which are then broadcast to each
        # output) and inferring values and dimensions from other
        # information, when possible.
        #

        # Utility function for broadcasting spline params (degree, smoothness)
        def process_spline_parameters(
            values, length, allowed_types, minimum=0,
            default=None, name='unknown'):

            # Preprocessing
            if values is None and default is None:
                return None
            elif values is None:
                values = default
            elif isinstance(values, np.ndarray):
                # Convert ndarray to list
                values = values.tolist()

            # Figure out what type of object we were passed
            if isinstance(values, allowed_types):
                # Single number of an allowed type => broadcast to list
                values = [values for i in range(length)]
            elif all([isinstance(v, allowed_types) for v in values]):
                # List of values => make sure it is the right size
                if len(values) != length:
                    raise ValueError(f"length of '{name}' does not match"
                                     f" number of variables")
            else:
                raise ValueError(f"could not parse '{name}' keyword")

            # Check to make sure the values are OK
            if values is not None and any([val < minimum for val in values]):
                raise ValueError(
                    f"invalid value for '{name}'; must be at least {minimum}")

            return values

        # Degree of polynomial
        degree = process_spline_parameters(
            degree, nvars, (int), name='degree', minimum=1)

        # Smoothness at breakpoints; set default to degree - 1 (max possible)
        smoothness = process_spline_parameters(
            smoothness, nvars, (int), name='smoothness', minimum=0,
            default=[d - 1 for d in degree])

        # Make sure degree is sufficent for the level of smoothness
        if any([degree[i] - smoothness[i] < 1 for i in range(nvars)]):
            raise ValueError("degree must be greater than smoothness")

        # Store the parameters for the spline (self.nvars already stored)
        self.breakpoints = breakpoints
        self.degree = degree
        self.smoothness = smoothness

        #
        # Compute parameters for a SciPy BSpline object
        #
        # To create a B-spline, we need to compute the knotpoints, keeping
        # track of the use of repeated knotpoints at the initial knot and
        # final knot as well as repeated knots at intermediate points
        # depending on the desired smoothness.
        #

        # Store the coefficients for each output (useful later)
        self.coef_offset, self.coef_length, offset = [], [], 0
        for i in range(nvars):
            # Compute number of coefficients for the piecewise polynomial
            ncoefs = (self.degree[i] + 1) * (len(self.breakpoints) - 1) - \
                (self.smoothness[i] + 1) * (len(self.breakpoints) - 2)

            self.coef_offset.append(offset)
            self.coef_length.append(ncoefs)
            offset += ncoefs
        self.N = offset         # save the total number of coefficients

        # Create knotpoints for each spline variable
        # TODO: extend to multi-dimensional breakpoints
        self.knotpoints = []
        for i in range(nvars):
            # Allocate space for the knotpoints
            self.knotpoints.append(np.empty(
                (self.degree[i] + 1) + (len(self.breakpoints) - 2) * \
                (self.degree[i] - self.smoothness[i]) + (self.degree[i] + 1)))

            # Initial knotpoints (multiplicity = order)
            self.knotpoints[i][0:self.degree[i] + 1] = self.breakpoints[0]
            offset = self.degree[i] + 1

            # Interior knotpoints (multiplicity = degree - smoothness)
            nknots = self.degree[i] - self.smoothness[i]
            assert nknots > 0           # just in case
            for j in range(1, self.breakpoints.size - 1):
                self.knotpoints[i][offset:offset+nknots] = self.breakpoints[j]
                offset += nknots

            # Final knotpoint (multiplicity = order)
            self.knotpoints[i][offset:offset + self.degree[i] + 1] = \
                self.breakpoints[-1]

    def __repr__(self):
        return f'<{self.__class__.__name__}: nvars={self.nvars}, ' + \
            f'degree={self.degree}, smoothness={self.smoothness}>'

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t, var=None):
        """Evaluate the kth derivative of the ith basis function at time t."""
        if self.nvars is None or (self.nvars == 1 and var is None):
            # Use same variable for all requests
            var = 0
        elif self.nvars > 1 and var is None:
            raise SystemError(
                "scalar variable call to multi-variable splines")

        # Create a coefficient vector for this spline
        coefs = np.zeros(self.coef_length[var]); coefs[i] = 1

        # Evaluate the derivative of the spline at the desired point in time
        return BSpline(self.knotpoints[var], coefs,
                       self.degree[var]).derivative(k)(t)
