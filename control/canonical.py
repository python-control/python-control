# canonical.py - functions for converting systems to canonical forms
# RMM, 10 Nov 2012

from .exception import ControlNotImplemented
from .lti import issiso
from .statesp import StateSpace
from .statefbk import ctrb, obsv

from numpy import zeros, zeros_like, shape, poly, iscomplex, vstack, hstack, dot, \
    transpose, empty
from numpy.linalg import solve, matrix_rank, eig

__all__ = ['canonical_form', 'reachable_form', 'observable_form', 'modal_form',
           'similarity_transform']

def canonical_form(xsys, form='reachable'):
    """Convert a system into canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state 'x'
    form : String
        Canonical form for transformation.  Chosen from:
          * 'reachable' - reachable canonical form
          * 'observable' - observable canonical form
          * 'modal' - modal canonical form

    Returns
    -------
    zsys : StateSpace object
        System in desired canonical form, with state 'z'
    T : matrix
        Coordinate transformation matrix, z = T * x
    """

    # Call the appropriate tranformation function
    if form == 'reachable':
        return reachable_form(xsys)
    elif form == 'observable':
        return observable_form(xsys)
    elif form == 'modal':
        return modal_form(xsys)
    else:
        raise ControlNotImplemented(
            "Canonical form '%s' not yet implemented" % form)


# Reachable canonical form
def reachable_form(xsys):
    """Convert a system into reachable canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state `x`

    Returns
    -------
    zsys : StateSpace object
        System in reachable canonical form, with state `z`
    T : matrix
        Coordinate transformation: z = T * x
    """
    # Check to make sure we have a SISO system
    if not issiso(xsys):
        raise ControlNotImplemented(
            "Canonical forms for MIMO systems not yet supported")

    # Create a new system, starting with a copy of the old one
    zsys = StateSpace(xsys)

    # Generate the system matrices for the desired canonical form
    zsys.B = zeros_like(xsys.B)
    zsys.B[0, 0] = 1.0
    zsys.A = zeros_like(xsys.A)
    Apoly = poly(xsys.A)                # characteristic polynomial
    for i in range(0, xsys.states):
        zsys.A[0, i] = -Apoly[i+1] / Apoly[0]
        if (i+1 < xsys.states):
            zsys.A[i+1, i] = 1.0

    # Compute the reachability matrices for each set of states
    Wrx = ctrb(xsys.A, xsys.B)
    Wrz = ctrb(zsys.A, zsys.B)

    if matrix_rank(Wrx) != xsys.states:
        raise ValueError("System not controllable to working precision.")

    # Transformation from one form to another
    Tzx = solve(Wrx.T, Wrz.T).T  # matrix right division, Tzx = Wrz * inv(Wrx)

    # Check to make sure inversion was OK.  Note that since we are inverting
    # Wrx and we already checked its rank, this exception should never occur
    if matrix_rank(Tzx) != xsys.states:         # pragma: no cover
        raise ValueError("Transformation matrix singular to working precision.")

    # Finally, compute the output matrix
    zsys.C = solve(Tzx.T, xsys.C.T).T  # matrix right division, zsys.C = xsys.C * inv(Tzx)

    return zsys, Tzx


def observable_form(xsys):
    """Convert a system into observable canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state `x`

    Returns
    -------
    zsys : StateSpace object
        System in observable canonical form, with state `z`
    T : matrix
        Coordinate transformation: z = T * x
    """
    # Check to make sure we have a SISO system
    if not issiso(xsys):
        raise ControlNotImplemented(
            "Canonical forms for MIMO systems not yet supported")

    # Create a new system, starting with a copy of the old one
    zsys = StateSpace(xsys)

    # Generate the system matrices for the desired canonical form
    zsys.C = zeros_like(xsys.C)
    zsys.C[0, 0] = 1
    zsys.A = zeros_like(xsys.A)
    Apoly = poly(xsys.A)                # characteristic polynomial
    for i in range(0, xsys.states):
        zsys.A[i, 0] = -Apoly[i+1] / Apoly[0]
        if (i+1 < xsys.states):
            zsys.A[i, i+1] = 1

    # Compute the observability matrices for each set of states
    Wrx = obsv(xsys.A, xsys.C)
    Wrz = obsv(zsys.A, zsys.C)

    # Transformation from one form to another
    Tzx = solve(Wrz, Wrx)  # matrix left division, Tzx = inv(Wrz) * Wrx

    if matrix_rank(Tzx) != xsys.states:
        raise ValueError("Transformation matrix singular to working precision.")

    # Finally, compute the output matrix
    zsys.B = Tzx.dot(xsys.B)

    return zsys, Tzx

def modal_form(xsys):
    """Convert a system into modal canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state `x`

    Returns
    -------
    zsys : StateSpace object
        System in modal canonical form, with state `z`
    T : matrix
        Coordinate transformation: z = T * x
    """
    # Check to make sure we have a SISO system
    if not issiso(xsys):
        raise ControlNotImplemented(
            "Canonical forms for MIMO systems not yet supported")

    # Create a new system, starting with a copy of the old one
    zsys = StateSpace(xsys)

    # Calculate eigenvalues and matrix of eigenvectors Tzx,
    eigval, eigvec = eig(xsys.A)

    # Eigenvalues and corresponding eigenvectors are not sorted,
    # thus modal transformation is ambiguous
    # Sort eigenvalues and vectors from largest to smallest eigenvalue
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    # If all eigenvalues are real, the matrix of eigenvectors is Tzx directly
    if not iscomplex(eigval).any():
        Tzx = eigvec
    else:
        # A is an arbitrary semisimple matrix

        # Keep track of complex conjugates (need only one)
        lst_conjugates = []
        Tzx = empty((0, xsys.A.shape[0])) # empty zero-height row matrix
        for val, vec in zip(eigval, eigvec.T):
            if iscomplex(val):
                if val not in lst_conjugates:
                    lst_conjugates.append(val.conjugate())
                    Tzx = vstack((Tzx, vec.real, vec.imag))
                else:
                    # if conjugate has already been seen, skip this eigenvalue
                    lst_conjugates.remove(val)
            else:
                Tzx = vstack((Tzx, vec.real))
        Tzx = Tzx.T

    # Generate the system matrices for the desired canonical form
    zsys.A = solve(Tzx, xsys.A).dot(Tzx)
    zsys.B = solve(Tzx, xsys.B)
    zsys.C = xsys.C.dot(Tzx)

    return zsys, Tzx


def similarity_transform(xsys, T, timescale=1):
    """Perform a similarity transformation, with option time rescaling.

    Transform a linear state space system to a new state space representation
    z = T x, where T is an invertible matrix.

    Parameters
    ----------
    T : 2D invertible array
        The matrix `T` defines the new set of coordinates z = T x.
    timescale : float
        If present, also rescale the time unit to tau = timescale * t

    Returns
    -------
    zsys : StateSpace object
        System in transformed coordinates, with state 'z'

    """
    # Create a new system, starting with a copy of the old one
    zsys = StateSpace(xsys)

    # Define a function to compute the right inverse (solve x M = y)
    def rsolve(M, y):
        return transpose(solve(transpose(M), transpose(y)))

    # Update the system matrices
    zsys.A = rsolve(T, dot(T, zsys.A)) / timescale
    zsys.B = dot(T, zsys.B) / timescale
    zsys.C = rsolve(T, zsys.C)

    return zsys
