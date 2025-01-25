# sysnorm.py - functions for computing system norms
#
# Initial author: Henrik Sandberg
# Creation date: 21 Dec 2023

"""Functions for computing system norms."""

import warnings

import numpy as np
import numpy.linalg as la

import control as ct

__all__ = ['system_norm', 'norm']

#------------------------------------------------------------------------------

def _h2norm_slycot(sys, print_warning=True):
    """H2 norm of a linear system. For internal use. Requires Slycot.

    See Also
    --------
    slycot.ab13bd

    """
    # See: https://github.com/python-control/Slycot/issues/199
    try:
        from slycot import ab13bd
    except ImportError:
        ct.ControlSlycot("Can't find slycot module ab13bd")

    try:
        from slycot.exceptions import SlycotArithmeticError
    except ImportError:
        raise ct.ControlSlycot(
            "Can't find slycot class SlycotArithmeticError")

    A, B, C, D = ct.ssdata(ct.ss(sys))

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    dico = 'C' if sys.isctime() else 'D'  # Continuous or discrete time
    jobn = 'H'  # H2 (and not L2 norm)

    if n == 0:
    # ab13bd does not accept empty A, B, C
        if dico == 'C':
            if any(D.flat != 0):
                if print_warning:
                    warnings.warn(
                        "System has a direct feedthrough term!", UserWarning)
                return float("inf")
            else:
                return 0.0
        elif dico == 'D':
            return np.sqrt(D@D.T)

    try:
        norm = ab13bd(dico, jobn, n, m, p, A, B, C, D)
    except SlycotArithmeticError as e:
        if e.info == 3:
            if print_warning:
                warnings.warn(
                    "System has pole(s) on the stability boundary!",
                    UserWarning)
            return float("inf")
        elif e.info == 5:
            if print_warning:
                warnings.warn(
                    "System has a direct feedthrough term!", UserWarning)
            return float("inf")
        elif e.info == 6:
            if print_warning:
                warnings.warn("System is unstable!", UserWarning)
            return float("inf")
        else:
            raise e
    return norm

#------------------------------------------------------------------------------

def system_norm(system, p=2, tol=1e-6, print_warning=True, method=None):
    """Computes the input/output norm of system.

    Parameters
    ----------
    system : LTI (`StateSpace` or `TransferFunction`)
        System in continuous or discrete time for which the norm should
        be computed.
    p : int or str
        Type of norm to be computed. `p` = 2 gives the H2 norm, and
        `p` = 'inf' gives the L-infinity norm.
    tol : float
        Relative tolerance for accuracy of L-infinity norm
        computation. Ignored unless `p` = 'inf'.
    print_warning : bool
        Print warning message in case norm value may be uncertain.
    method : str, optional
        Set the method used for computing the result.  Current methods are
        'slycot' and 'scipy'. If set to None (default), try 'slycot' first
        and then 'scipy'.

    Returns
    -------
    norm_value : float
        Norm value of system.

    Notes
    -----
    Does not yet compute the L-infinity norm for discrete-time systems
    with pole(s) at the origin unless Slycot is used.

    Examples
    --------
    >>> Gc = ct.tf([1], [1, 2, 1])
    >>> round(ct.norm(Gc, 2), 3)
    0.5
    >>> round(ct.norm(Gc, 'inf', tol=1e-5, method='scipy'), 3)
    np.float64(1.0)

    """
    if not isinstance(system, (ct.StateSpace, ct.TransferFunction)):
        raise TypeError(
            "Parameter `system`: must be a `StateSpace` or `TransferFunction`")

    G = ct.ss(system)
    A = G.A
    B = G.B
    C = G.C
    D = G.D

    # Decide what method to use
    method = ct.mateqn._slycot_or_scipy(method)

    # -------------------
    # H2 norm computation
    # -------------------
    if p == 2:
        # --------------------
        # Continuous time case
        # --------------------
        if G.isctime():

            # Check for cases with infinite norm
            poles_real_part = G.poles().real
            if any(np.isclose(poles_real_part, 0.0)):  # Poles on imaginary axis
                if print_warning:
                    warnings.warn(
                        "Poles close to, or on, the imaginary axis. "
                        "Norm value may be uncertain.", UserWarning)
                return float('inf')
            elif any(poles_real_part > 0.0):  # System unstable
                if print_warning:
                    warnings.warn("System is unstable!", UserWarning)
                return float('inf')
            elif any(D.flat != 0):  # System has direct feedthrough
                if print_warning:
                    warnings.warn(
                        "System has a direct feedthrough term!", UserWarning)
                return float('inf')

            else:
                # Use slycot, if available, to compute (finite) norm
                if method == 'slycot':
                    return _h2norm_slycot(G, print_warning)

                # Else use scipy
                else:
                    # Solve for controllability Gramian
                    P = ct.lyap(A, B@B.T, method=method)

                    # System is stable to reach this point, and P should be
                    # positive semi-definite.  Test next is a precaution in
                    # case the Lyapunov equation is ill conditioned.
                    if any(la.eigvals(P).real < 0.0):
                        if print_warning:
                            warnings.warn(
                                "There appears to be poles close to the "
                                "imaginary axis. Norm value may be uncertain.",
                                UserWarning)
                        return float('inf')
                    else:
                        # Argument in sqrt should be non-negative
                        norm_value = np.sqrt(np.trace(C@P@C.T))
                        if np.isnan(norm_value):
                            raise ct.ControlArgument(
                                "Norm computation resulted in NaN.")
                        else:
                            return norm_value

        # ------------------
        # Discrete time case
        # ------------------
        elif G.isdtime():

            # Check for cases with infinite norm
            poles_abs = abs(G.poles())
            if any(np.isclose(poles_abs, 1.0)):  # Poles on imaginary axis
                if print_warning:
                    warnings.warn(
                        "Poles close to, or on, the complex unit circle. "
                        "Norm value may be uncertain.", UserWarning)
                return float('inf')
            elif any(poles_abs > 1.0):  # System unstable
                if print_warning:
                    warnings.warn("System is unstable!", UserWarning)
                return float('inf')
            else:
                # Use slycot, if available, to compute (finite) norm
                if method == 'slycot':
                    return _h2norm_slycot(G, print_warning)

                # Else use scipy
                else:
                    P = ct.dlyap(A, B@B.T, method=method)

                # System is stable to reach this point, and P should be
                # positive semi-definite.  Test next is a precaution in
                # case the Lyapunov equation is ill conditioned.
                if any(la.eigvals(P).real < 0.0):
                    if print_warning:
                        warnings.warn(
                            "There appears to be poles close to the complex "
                            "unit circle. Norm value may be uncertain.",
                            UserWarning)
                    return float('inf')
                else:
                    # Argument in sqrt should be non-negative
                    norm_value = np.sqrt(np.trace(C@P@C.T + D@D.T))
                    if np.isnan(norm_value):
                        raise ct.ControlArgument(
                            "Norm computation resulted in NaN.")
                    else:
                        return norm_value

    # ---------------------------
    # L-infinity norm computation
    # ---------------------------
    elif p == "inf":

        # Check for cases with infinite norm
        poles = G.poles()
        if G.isdtime():  # Discrete time
            if any(np.isclose(abs(poles), 1.0)):  # Poles on unit circle
                if print_warning:
                    warnings.warn(
                        "Poles close to, or on, the complex unit circle. "
                        "Norm value may be uncertain.", UserWarning)
                return float('inf')
        else:  # Continuous time
            if any(np.isclose(poles.real, 0.0)):  # Poles on imaginary axis
                if print_warning:
                    warnings.warn(
                        "Poles close to, or on, the imaginary axis. "
                        "Norm value may be uncertain.", UserWarning)
                return float('inf')

        # Use slycot, if available, to compute (finite) norm
        if method == 'slycot':
            return ct.linfnorm(G, tol)[0]

        # Else use scipy
        else:

            # ------------------
            # Discrete time case
            # ------------------
            # Use inverse bilinear transformation of discrete-time system
            # to s-plane if no poles on |z|=1 or z=0.  Allows us to use
            # test for continuous-time systems next.
            if G.isdtime():
                Ad = A
                Bd = B
                Cd = C
                Dd = D
                if any(np.isclose(la.eigvals(Ad), 0.0)):
                    raise ct.ControlArgument(
                        "L-infinity norm computation for discrete-time "
                        "system with pole(s) in z=0 currently not supported "
                        "unless Slycot installed.")

                # Inverse bilinear transformation
                In = np.eye(len(Ad))
                Adinv = la.inv(Ad+In)
                A = 2*(Ad-In)@Adinv
                B = 2*Adinv@Bd
                C = 2*Cd@Adinv
                D = Dd - Cd@Adinv@Bd

            # --------------------
            # Continuous time case
            # --------------------
            def _Hamilton_matrix(gamma):
                """Constructs Hamiltonian matrix. For internal use."""
                R = Ip*gamma**2 - D.T@D
                invR = la.inv(R)
                return np.block([
                    [A+B@invR@D.T@C, B@invR@B.T],
                    [-C.T@(Ip+D@invR@D.T)@C, -(A+B@invR@D.T@C).T]])

            gaml = la.norm(D,ord=2)    # Lower bound
            gamu = max(1.0, 2.0*gaml)  # Candidate upper bound
            Ip = np.eye(len(D))

            while any(np.isclose(
                    la.eigvals(_Hamilton_matrix(gamu)).real, 0.0)):
                # Find actual upper bound
                gamu *= 2.0

            while (gamu-gaml)/gamu > tol:
                gam = (gamu+gaml)/2.0
                if any(np.isclose(la.eigvals(_Hamilton_matrix(gam)).real, 0.0)):
                    gaml = gam
                else:
                    gamu = gam
            return gam

    # ----------------------
    # Other norm computation
    # ----------------------
    else:
        raise ct.ControlArgument(
            f"Norm computation for p={p} currently not supported.")


norm = system_norm
