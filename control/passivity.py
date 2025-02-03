# passivity.py - functions for passive control
#
# Initial author: Mark Yeatman
# Creation date: July 17, 2022

"""Functions for passive control."""

import numpy as np

from control import statesp
from control.exception import ControlArgument, ControlDimension

try:
    import cvxopt as cvx
except ImportError:
    cvx = None


__all__ = ["get_output_fb_index", "get_input_ff_index",  "ispassive",
           "solve_passivity_LMI"]


def solve_passivity_LMI(sys, rho=None, nu=None):
    """Compute passivity indices and/or solves feasibility via a LMI.

    Constructs a linear matrix inequality (LMI) such that if a solution
    exists and the last element of the solution is positive, the system
    `sys` is passive. Inputs of None for `rho` or `nu` indicate that the
    function should solve for that index (they are mutually exclusive, they
    can't both be None, otherwise you're trying to solve a nonconvex
    bilinear matrix inequality.) The last element of the output `solution`
    is either the output or input passivity index, for `rho` = None and
    `nu` = None, respectively.

    Parameters
    ----------
    sys : LTI
        System to be checked.
    rho : float or None
        Output feedback passivity index.
    nu : float or None
        Input feedforward passivity index.

    Returns
    -------
    solution : ndarray
        The LMI solution.

    References
    ----------
    .. [1] McCourt, Michael J., and Panos J. Antsaklis, "Demonstrating
        passivity and dissipativity using computational methods."

    .. [2] Nicholas Kottenstette and Panos J. Antsaklis,
        "Relationships Between Positive Real, Passive Dissipative, &
        Positive Systems", equation 36.

    """
    if cvx is None:
        raise ModuleNotFoundError("cvxopt required for passivity module")

    if sys.ninputs != sys.noutputs:
        raise ControlDimension(
            "The number of system inputs must be the same as the number of "
            "system outputs.")

    if rho is None and nu is None:
        raise ControlArgument("rho or nu must be given a numerical value.")

    sys = statesp._convert_to_statespace(sys)

    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    # account for strictly proper systems
    [_, m] = D.shape
    [n, _] = A.shape

    def make_LMI_matrix(P, rho, nu, one):
        q = sys.noutputs
        Q = -rho*np.eye(q, q)
        S = 1.0/2.0*(one+rho*nu)*np.eye(q)
        R = -nu*np.eye(m)
        if sys.isctime():
            off_diag = P@B - (C.T@S + C.T@Q@D)
            return np.vstack((
                np.hstack((A.T @ P + P@A - C.T@Q@C,  off_diag)),
                np.hstack((off_diag.T, -(D.T@Q@D + D.T@S + S.T@D + R)))
            ))
        else:
            off_diag = A.T@P@B - (C.T@S + C.T@Q@D)
            return np.vstack((
                np.hstack((A.T @ P  @ A - P - C.T@Q@C,  off_diag)),
                np.hstack((off_diag.T, B.T@P@B-(D.T@Q@D + D.T@S + S.T@D + R)))
            ))

    def make_P_basis_matrices(n, rho, nu):
        """Make list of matrix constraints for passivity LMI.

        Utility function to make basis matrices for a LMI from a
        symmetric matrix P of size n by n representing a parameterized
        symbolic matrix.

        """
        matrix_list = []
        for i in range(0, n):
            for j in range(0, n):
                if j <= i:
                    P = np.zeros((n, n))
                    P[i, j] = 1
                    P[j, i] = 1
                    matrix_list.append(make_LMI_matrix(P, 0, 0, 0).flatten())
        zeros = 0.0*np.eye(n)
        if rho is None:
            matrix_list.append(make_LMI_matrix(zeros, 1, 0, 0).flatten())
        elif nu is None:
            matrix_list.append(make_LMI_matrix(zeros, 0, 1, 0).flatten())
        return matrix_list


    def P_pos_def_constraint(n):
        """Make a list of matrix constraints for P >= 0.

        Utility function to make basis matrices for a LMI that ensures
        parameterized symbolic matrix of size n by n is positive definite
        """
        matrix_list = []
        for i in range(0, n):
            for j in range(0, n):
                if j <= i:
                    P = np.zeros((n, n))
                    P[i, j] = -1
                    P[j, i] = -1
                    matrix_list.append(P.flatten())
        if rho is None or nu is None:
            matrix_list.append(np.zeros((n, n)).flatten())
        return matrix_list

    n = sys.nstates

    # coefficients for passivity indices and feasibility matrix
    sys_matrix_list = make_P_basis_matrices(n, rho, nu)

    # get constants for numerical values of rho and nu
    sys_constants = list()
    if rho is not None and nu is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), rho, nu, 1.0)
    elif rho is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), rho, 0.0, 1.0)
    elif nu is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), 0.0, nu, 1.0)

    sys_coefficents = np.vstack(sys_matrix_list).T

    # LMI to ensure P is positive definite
    P_matrix_list = P_pos_def_constraint(n)
    P_coefficents = np.vstack(P_matrix_list).T
    P_constants = np.zeros((n, n))

    # cost function
    number_of_opt_vars = int(
        (n**2-n)/2 + n)
    c = cvx.matrix(0.0, (number_of_opt_vars, 1))

    #we're maximizing a passivity index, include it in the cost function
    if rho is None or nu is None:
        c = cvx.matrix(np.append(np.array(c), -1.0))

    Gs = [cvx.matrix(sys_coefficents)] + [cvx.matrix(P_coefficents)]
    hs = [cvx.matrix(sys_constants)] + [cvx.matrix(P_constants)]

    # crunch feasibility solution
    cvx.solvers.options['show_progress'] = False
    try:
        sol = cvx.solvers.sdp(c, Gs=Gs, hs=hs)
        return sol["x"]

    except ZeroDivisionError as e:
        raise ValueError(
            "The system is probably ill conditioned. Consider perturbing "
            "the system matrices by a small amount."
        ) from e


def get_output_fb_index(sys):
    """Return the output feedback passivity (OFP) index for the system.

    The OFP is the largest gain that can be placed in positive feedback
    with a system such that the new interconnected system is passive.

    Parameters
    ----------
    sys : LTI
        System to be checked.

    Returns
    -------
    float
        The OFP index.

    """
    sol = solve_passivity_LMI(sys, nu=0.0)
    if sol is None:
        raise RuntimeError("LMI passivity problem is infeasible")
    else:
        return sol[-1]


def get_input_ff_index(sys):
    """Input feedforward passivity (IFP) index for a system.

    The input feedforward passivity (IFP) is the largest gain that can be
    placed in negative parallel interconnection with a system such that the
    new interconnected system is passive.

    Parameters
    ----------
    sys : LTI
        System to be checked.

    Returns
    -------
    float
        The IFP index.

    """
    sol = solve_passivity_LMI(sys, rho=0.0)
    if sol is None:
        raise RuntimeError("LMI passivity problem is infeasible")
    else:
        return sol[-1]


def get_relative_index(sys):
    """Return the relative passivity index for the system.

    (not implemented yet)
    """
    raise NotImplementedError("Relative passivity index not implemented")


def get_combined_io_index(sys):
    """Return the combined I/O passivity index for the system.

    (not implemented yet)
    """
    raise NotImplementedError("Combined I/O passivity index not implemented")


def get_directional_index(sys):
    """Return the directional passivity index for the system.

    (not implemented yet)
    """
    raise NotImplementedError("Directional passivity index not implemented")


def ispassive(sys, ofp_index=0, ifp_index=0):
    r"""Indicate if a linear time invariant (LTI) system is passive.

    Checks if system is passive with the given output feedback (OFP)
    and input feedforward (IFP) passivity indices.

    Parameters
    ----------
    sys : LTI
        System to be checked.
    ofp_index : float
        Output feedback passivity index.
    ifp_index : float
        Input feedforward passivity index.

    Returns
    -------
    bool
        The system is passive.

    Notes
    -----
    Querying if the system is passive in the sense of

    .. math:: V(x) >= 0 \land \dot{V}(x) <= y^T u

    is equivalent to the default case of `ofp_index` = 0 and `ifp_index` =
    0.  Note that computing the `ofp_index` and `ifp_index` for a system,
    then using both values simultaneously as inputs to this function is not
    guaranteed to have an output of True (the system might not be passive
    with both indices at the same time).

    For more details, see [1]_.

    References
    ----------

    .. [1] McCourt, Michael J., and Panos J. Antsaklis "Demonstrating
       passivity and dissipativity using computational methods."
       Technical Report of the ISIS Group at the University of Notre
       Dame.  ISIS-2013-008, Aug. 2013.

    """
    return solve_passivity_LMI(sys, rho=ofp_index, nu=ifp_index) is not None
