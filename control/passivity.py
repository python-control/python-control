'''
Author: Mark Yeatman  
Date: July 17, 2022
'''

import numpy as np
from control import statesp as ss
from control.exception import ControlArgument, ControlDimension

try:
    import cvxopt as cvx
except ImportError as e:
    cvx = None

eps = np.nextafter(0, 1)


def __make_P_basis_matrices__(n, rho, nu, make_LMI_matrix_func):
    '''
    Utility function to make basis matrices for a LMI from a 
    functional make_LMI_matrix_func and a symmetric matrix P of size n by n
    representing a parametrized symbolic matrix
    '''
    matrix_list = []
    for i in range(0, n):
        for j in range(0, n):
            if j <= i:
                P = np.zeros((n, n))
                P[i, j] = 1.0
                P[j, i] = 1.0
                matrix_list.append(make_LMI_matrix_func(P, 0, 0, 0).flatten())
    P = eps*np.eye(n)
    matrix_list.append(make_LMI_matrix_func(P, rho, nu, 0).flatten())
    return matrix_list


def __P_pos_def_constraint__(n):
    '''
    Utility function to make basis matrices for a LMI that ensures parametrized symbolic matrix 
    of size n by n is positive definite.
    '''
    matrix_list = []
    for i in range(0, n):
        for j in range(0, n):
            if j <= i:
                P = np.zeros((n, n))
                P[i, j] = -1.0
                P[j, i] = -1.0
                matrix_list.append(P.flatten())
    matrix_list.append(np.zeros((n, n)).flatten())
    return matrix_list


def __solve_passivity_LMI__(sys, rho=None, nu=None):
    '''
    Constructs a linear matrix inequality such that if a solution exists 
    and the last element of the solution is positive, the system is passive.

    The sources for the algorithm are: 

    McCourt, Michael J., and Panos J. Antsaklis
        "Demonstrating passivity and dissipativity using computational methods." 

    Nicholas Kottenstette and Panos J. Antsaklis
        "Relationships Between Positive Real, Passive Dissipative, & Positive Systems" 
        equation 36.
    '''
    if cvx is None:
        raise ModuleNotFoundError("cvxopt required for passivity module")

    if sys.ninputs != sys.noutputs:
        raise ControlDimension(
            "The number of system inputs must be the same as the number of system outputs.")

    if rho is None and nu is None:
        raise ControlArgument("rho or nu must be given a float value.")

    sys = ss._convert_to_statespace(sys)

    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    # account for strictly proper systems
    [n, m] = D.shape
    D = D + eps * np.eye(n, m)

    [n, _] = A.shape
    A = A - eps*np.eye(n)

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

    n = sys.nstates

    # LMI for passivity indices from A,B,C,D
    sys_matrix_list = list()
    sys_constants = -np.vstack((
        np.hstack((np.zeros_like(A),  np.zeros_like(C.T))),
        np.hstack((np.zeros_like(C), np.zeros_like(D))))
    )

    if rho is not None:
        sys_matrix_list = __make_P_basis_matrices__(
            n, eps, 1.0, make_LMI_matrix)
        sys_constants = -make_LMI_matrix(np.zeros_like(A), rho, eps, 1.0)
    else:
        sys_matrix_list = __make_P_basis_matrices__(
            n, 1.0, eps, make_LMI_matrix)
        sys_constants = -make_LMI_matrix(np.zeros_like(A), eps, nu, 1.0)

    sys_coefficents = np.vstack(sys_matrix_list).T

    # LMI to ensure P is positive definite
    P_matrix_list = __P_pos_def_constraint__(n)
    P_coefficents = np.vstack(P_matrix_list).T
    P_constants = np.zeros((n, n))

    # cost function
    number_of_opt_vars = int(
        (n**2-n)/2 + n)
    c = cvx.matrix(0.0, (number_of_opt_vars, 1))
    c = cvx.matrix(np.append(np.array(c), -1.0))

    Gs = [cvx.matrix(sys_coefficents)] + [cvx.matrix(P_coefficents)]
    hs = [cvx.matrix(sys_constants)] + [cvx.matrix(P_constants)]

    # crunch feasibility solution
    cvx.solvers.options['show_progress'] = False
    sol = cvx.solvers.sdp(c, Gs=Gs, hs=hs)
    return sol["x"]


def get_output_fb_index(sys):
    '''
    Returns the output feedback passivity index for the input system. This is largest gain that can be placed in
    positive feedback with a system such that the new interconnected system is passive.
    Parameters
    ----------
    sys: An LTI system
        System to be checked.

    Returns
    -------
    float: 
        The OFP index 
    '''
    sol = __solve_passivity_LMI__(sys, nu=eps)
    if sol is not None:
        return sol[-1]
    else:
        return -np.inf


def get_input_ff_index(sys, index_type=None):
    '''
    Returns the input feedforward passivity (IFP) index for the input system. This is the largest gain that can be 
    placed in negative parallel interconnection with a system such that the new interconnected system is passive.
    Parameters
    ----------
    sys: An LTI system
        System to be checked.

    Returns
    -------
    float: 
        The IFP index 
    '''

    sol = __solve_passivity_LMI__(sys, rho=eps)
    if sol is not None:
        return sol[-1]
    else:
        return -np.inf


def ispassive(sys, rho=None, nu=None):
    '''
    Indicates if a linear time invariant (LTI) system is passive

    Parameters
    ----------
    sys: An LTI system
        System to be checked.

    Returns
    -------
    bool, float, or None: 
        The input system is passive, or the passivity index "opposite" the input. 
        If the problem is unfeasiable when requesting the passivity index, returns None.
    '''
    output_fb_index = get_output_fb_index(sys)
    input_ff_index = get_input_ff_index(sys)
    return output_fb_index >= 0 or input_ff_index >= 0
