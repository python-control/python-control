# canonical.py - functions for converting systems to canonical forms
# RMM, 10 Nov 2012

from .exception import ControlNotImplemented, ControlSlycot
from .namedio import issiso
from .statesp import StateSpace, _convert_to_statespace
from .statefbk import ctrb, obsv

import numpy as np

from numpy import zeros, zeros_like, shape, poly, iscomplex, vstack, hstack, \
    transpose, empty, finfo, float64
from numpy.linalg import solve, matrix_rank, eig

from scipy.linalg import schur

__all__ = ['canonical_form', 'reachable_form', 'observable_form', 'modal_form',
           'similarity_transform', 'bdschur']


def canonical_form(xsys, form='reachable'):
    """Convert a system into canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state 'x'
    form : str
        Canonical form for transformation.  Chosen from:
          * 'reachable' - reachable canonical form
          * 'observable' - observable canonical form
          * 'modal' - modal canonical form

    Returns
    -------
    zsys : StateSpace object
        System in desired canonical form, with state 'z'
    T : (M, M) real ndarray
        Coordinate transformation matrix, z = T * x

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> Gc, T = ct.canonical_form(Gs)  # default reachable
    >>> Gc.B
    array([[1.],
           [0.]])

    >>> Gc, T = ct.canonical_form(Gs, 'observable')
    >>> Gc.C
    array([[1., 0.]])

    >>> Gc, T = ct.canonical_form(Gs, 'modal')
    >>> Gc.A                                                    # doctest: +SKIP
    array([[-2.,  0.],
           [ 0., -1.]])

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
    T : (M, M) real ndarray
        Coordinate transformation: z = T * x

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> Gc, T = ct.reachable_form(Gs)  # default reachable
    >>> Gc.B
    array([[1.],
           [0.]])

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
    for i in range(0, xsys.nstates):
        zsys.A[0, i] = -Apoly[i+1] / Apoly[0]
        if (i+1 < xsys.nstates):
            zsys.A[i+1, i] = 1.0

    # Compute the reachability matrices for each set of states
    Wrx = ctrb(xsys.A, xsys.B)
    Wrz = ctrb(zsys.A, zsys.B)

    if matrix_rank(Wrx) != xsys.nstates:
        raise ValueError("System not controllable to working precision.")

    # Transformation from one form to another
    Tzx = solve(Wrx.T, Wrz.T).T  # matrix right division, Tzx = Wrz * inv(Wrx)

    # Check to make sure inversion was OK.  Note that since we are inverting
    # Wrx and we already checked its rank, this exception should never occur
    if matrix_rank(Tzx) != xsys.nstates:         # pragma: no cover
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
    T : (M, M) real ndarray
        Coordinate transformation: z = T * x

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> Gc, T = ct.observable_form(Gs)
    >>> Gc.C
    array([[1., 0.]])

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
    for i in range(0, xsys.nstates):
        zsys.A[i, 0] = -Apoly[i+1] / Apoly[0]
        if (i+1 < xsys.nstates):
            zsys.A[i, i+1] = 1

    # Compute the observability matrices for each set of states
    Wrx = obsv(xsys.A, xsys.C)
    Wrz = obsv(zsys.A, zsys.C)

    # Transformation from one form to another
    Tzx = solve(Wrz, Wrx)  # matrix left division, Tzx = inv(Wrz) * Wrx

    if matrix_rank(Tzx) != xsys.nstates:
        raise ValueError("Transformation matrix singular to working precision.")

    # Finally, compute the output matrix
    zsys.B = Tzx @ xsys.B

    return zsys, Tzx


def similarity_transform(xsys, T, timescale=1, inverse=False):
    """Perform a similarity transformation, with option time rescaling.

    Transform a linear state space system to a new state space representation
    z = T x, or x = T z, where T is an invertible matrix.

    Parameters
    ----------
    xsys : StateSpace object
           System to transform
    T : (M, M) array_like
        The matrix `T` defines the new set of coordinates z = T x.
    timescale : float, optional
        If present, also rescale the time unit to tau = timescale * t
    inverse: boolean, optional
        If True (default), transform so z = T x.  If False, transform
        so x = T z.

    Returns
    -------
    zsys : StateSpace object
        System in transformed coordinates, with state 'z'


    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> Gs.A
    array([[-3., -2.],
           [ 1.,  0.]])

    >>> T = np.array([[0, 1], [1, 0]])
    >>> Gt = ct.similarity_transform(Gs, T)
    >>> Gt.A
    array([[ 0.,  1.],
           [-2., -3.]])

    """
    # Create a new system, starting with a copy of the old one
    zsys = StateSpace(xsys)

    T = np.atleast_2d(T)

    # Define a function to compute the right inverse (solve x M = y)
    def rsolve(M, y):
        return transpose(solve(transpose(M), transpose(y)))

    # Update the system matrices
    if not inverse:
        zsys.A = rsolve(T, T @ zsys.A) / timescale
        zsys.B = T @ zsys.B / timescale
        zsys.C = rsolve(T, zsys.C)
    else:
        zsys.A = solve(T, zsys.A) @ T / timescale
        zsys.B = solve(T, zsys.B) / timescale
        zsys.C = zsys.C @ T

    return zsys


_IM_ZERO_TOL = np.finfo(np.float64).eps ** 0.5
_PMAX_SEARCH_TOL = 1.001


def _bdschur_defective(blksizes, eigvals):
    """Check  for defective modal decomposition

    Parameters
    ----------
    blksizes: (N,) int ndarray
       size of Schur blocks
    eigvals: (M,) real or complex ndarray
       Eigenvalues

    Returns
    -------
    True iff Schur blocks are defective.

    blksizes, eigvals are the 3rd and 4th results returned by mb03rd.
    """
    if any(blksizes > 2):
        return True

    if all(blksizes == 1):
        return False

    # check eigenvalues associated with blocks of size 2
    init_idxs = np.cumsum(np.hstack([0, blksizes[:-1]]))
    blk_idx2 = blksizes == 2

    im = eigvals[init_idxs[blk_idx2]].imag
    re = eigvals[init_idxs[blk_idx2]].real

    if any(abs(im) < _IM_ZERO_TOL * abs(re)):
        return True

    return False


def _bdschur_condmax_search(aschur, tschur, condmax):
    """Block-diagonal Schur decomposition search up to condmax

    Iterates mb03rd with different pmax values until:
      - result is non-defective;
      - or condition number of similarity transform is unchanging
        despite large pmax;
      - or condition number of similarity transform is close to condmax.

    Parameters
    ----------
    aschur: (N, N) real ndarray
      Real Schur-form matrix
    tschur: (N, N) real ndarray
      Orthogonal transformation giving aschur from some initial matrix a
    condmax: float
      Maximum condition number of final transformation.  Must be >= 1.

    Returns
    -------
    amodal: (N, N) real ndarray
       block diagonal Schur form
    tmodal: (N, N) real ndarray
       similarity transformation give amodal from aschur
    blksizes: (M,) int ndarray
       Array of Schur block sizes
    eigvals: (N,) real or complex ndarray
       Eigenvalues of amodal (and a, etc.)

    Notes
    -----
    Outputs as for slycot.mb03rd

    aschur, tschur are as returned by scipy.linalg.schur.
    """
    try:
        from slycot import mb03rd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'mb03rd'")

    # see notes on RuntimeError below
    pmaxlower = None

    # get lower bound; try condmax ** 0.5 first
    pmaxlower = condmax ** 0.5
    amodal, tmodal, blksizes, eigvals = mb03rd(
        aschur.shape[0], aschur, tschur, pmax=pmaxlower)
    if np.linalg.cond(tmodal) <= condmax:
        reslower = amodal, tmodal, blksizes, eigvals
    else:
        pmaxlower = 1.0
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, pmax=pmaxlower)
        cond = np.linalg.cond(tmodal)
        if cond > condmax:
            msg = f"minimum {cond=} > {condmax=}; try increasing condmax"
            raise RuntimeError(msg)

    pmax = pmaxlower

    # phase 1: search for upper bound on pmax
    for i in range(50):
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, pmax=pmax)
        cond = np.linalg.cond(tmodal)
        if cond < condmax:
            pmaxlower = pmax
            reslower = amodal, tmodal, blksizes, eigvals
        else:
            # upper bound found; go to phase 2
            pmaxupper = pmax
            break

        if _bdschur_defective(blksizes, eigvals):
            pmax *= 2
        else:
            return amodal, tmodal, blksizes, eigvals
    else:
        # no upper bound found; return current result
        return reslower

    # phase 2: bisection search
    for i in range(50):
        pmax = (pmaxlower * pmaxupper) ** 0.5
        amodal, tmodal, blksizes, eigvals = mb03rd(
            aschur.shape[0], aschur, tschur, pmax=pmax)
        cond = np.linalg.cond(tmodal)

        if cond < condmax:
            if not _bdschur_defective(blksizes, eigvals):
                return amodal, tmodal, blksizes, eigvals
            pmaxlower = pmax
            reslower = amodal, tmodal, blksizes, eigvals
        else:
            pmaxupper = pmax

        if pmaxupper / pmaxlower < _PMAX_SEARCH_TOL:
            # hit search limit
            return reslower
    else:
        raise ValueError('bisection failed to converge; pmaxlower={}, pmaxupper={}'.format(pmaxlower, pmaxupper))


def bdschur(a, condmax=None, sort=None):
    """Block-diagonal Schur decomposition

    Parameters
    ----------
        a : (M, M) array_like
            Real matrix to decompose
        condmax : None or float, optional
            If None (default), use 1/sqrt(eps), which is approximately 1e8
        sort : {None, 'continuous', 'discrete'}
            Block sorting; see below.

    Returns
    -------
        amodal : (M, M) real ndarray
            Block-diagonal Schur decomposition of `a`
        tmodal : (M, M) real ndarray
            Similarity transform relating `a` and `amodal`
        blksizes : (N,) int ndarray
            Array of Schur block sizes

    Notes
    -----
    If `sort` is None, the blocks are not sorted.

    If `sort` is 'continuous', the blocks are sorted according to
    associated eigenvalues.  The ordering is first by real part of
    eigenvalue, in descending order, then by absolute value of
    imaginary part of eigenvalue, also in decreasing order.

    If `sort` is 'discrete', the blocks are sorted as for
    'continuous', but applied to log of eigenvalues
    (i.e., continuous-equivalent eigenvalues).

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> amodal, tmodal, blksizes = ct.bdschur(Gs.A)
    >>> amodal                                                   #doctest: +SKIP
    array([[-2.,  0.],
           [ 0., -1.]])

    """
    if condmax is None:
        condmax = np.finfo(np.float64).eps ** -0.5

    if not (np.isscalar(condmax) and condmax >= 1.0):
        raise ValueError('condmax="{}" must be a scalar >= 1.0'.format(condmax))

    a = np.atleast_2d(a)
    if a.shape[0] == 0 or a.shape[1] == 0:
        return a.copy(), np.eye(a.shape[1], a.shape[0]), np.array([])

    aschur, tschur = schur(a)
    amodal, tmodal, blksizes, eigvals = _bdschur_condmax_search(
        aschur, tschur, condmax)

    if sort in ('continuous', 'discrete'):
        idxs = np.cumsum(np.hstack([0, blksizes[:-1]]))
        ev_per_blk = [complex(eigvals[i].real, abs(eigvals[i].imag))
                      for i in idxs]

        if sort == 'discrete':
            ev_per_blk = np.log(ev_per_blk)

        # put most unstable first
        sortidx = np.argsort(ev_per_blk)[::-1]

        # block indices
        blkidxs = [np.arange(i0, i0+ilen)
                   for i0, ilen in zip(idxs, blksizes)]

        # reordered
        permidx = np.hstack([blkidxs[i] for i in sortidx])
        rperm = np.eye(amodal.shape[0])[permidx]

        tmodal = tmodal @ rperm.T
        amodal = rperm @ amodal @ rperm.T
        blksizes = blksizes[sortidx]

    elif sort is None:
        pass

    else:
        raise ValueError('unknown sort value "{}"'.format(sort))

    return amodal, tmodal, blksizes


def modal_form(xsys, condmax=None, sort=False):
    """Convert a system into modal canonical form

    Parameters
    ----------
    xsys : StateSpace object
        System to be transformed, with state `x`
    condmax : None or float, optional
        An upper bound on individual transformations.  If None, use
        `bdschur` default.
    sort : bool, optional
        If False (default), Schur blocks will not be sorted.  See `bdschur`
        for sort order.

    Returns
    -------
    zsys : StateSpace object
        System in modal canonical form, with state `z`
    T : (M, M) ndarray
        Coordinate transformation: z = T * x

    Examples
    --------
    >>> Gs = ct.tf2ss([1], [1, 3, 2])
    >>> Gc, T = ct.modal_form(Gs)  # default reachable
    >>> Gc.A                                                    # doctest: +SKIP
    array([[-2.,  0.],
           [ 0., -1.]])

    """

    if sort:
        discrete = xsys.dt is not None and xsys.dt > 0
        bd_sort = 'discrete' if discrete else 'continuous'
    else:
        bd_sort = None

    xsys = _convert_to_statespace(xsys)
    amodal, tmodal, _ = bdschur(xsys.A, condmax=condmax, sort=bd_sort)

    return similarity_transform(xsys, tmodal, inverse=True), tmodal
