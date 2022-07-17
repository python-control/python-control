'''
Author: Mark Yeatman  
Date: May 15, 2022
'''

from msilib.schema import Error
import numpy as np
from control import statesp as ss
from control.modelsimp import minreal

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


def __ispassive__(sys, rho=None, nu=None):
    if cvx is None:
        raise ModuleNotFoundError("cvxopt required for passivity module")

    if sys.isdtime():
        raise Exception(
            "Passivity for discrete time systems not supported yet.")

    if sys.ninputs != sys.noutputs:
        raise Exception(
            "The number of system inputs must be the same as the number of system outputs.")

    if rho is None and nu is None:
        raise Exception("rho or nu must be given a float value.")

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
        off_diag = P@B - 1.0/2.0*(one+rho*nu)*C.T + rho*C.T*D
        return np.vstack((
            np.hstack((A.T @ P + P@A + rho*C.T@C,  off_diag)),
            np.hstack((off_diag.T, rho*D.T@D -
                       (one+rho*nu)*(D+D.T)+nu*np.eye(m)))
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


def ispassive(sys, rho=None, nu=None):
    '''
    Indicates if a linear time invariant (LTI) system is passive

    Constructs a linear matrix inequality and a feasibility optimization
    such that if a solution exists, the system is passive.

    The sources for the algorithm are: 

    McCourt, Michael J., and Panos J. Antsaklis
        "Demonstrating passivity and dissipativity using computational methods." 

    Nicholas Kottenstette and Panos J. Antsaklis
        "Relationships Between Positive Real, Passive Dissipative, & Positive Systems" 
        equation 36.

    Parameters
    ----------
    sys: An LTI system
        System to be checked.
    nu: float
        Concrete value for input passivity index. 
    rho: float
        Concrete value for output passivity index. 

    Returns
    -------
    bool or float: 
        The input system passive, or the passivity index "opposite" the input. 
    '''
    if rho is not None and nu is not None:
        return __ispassive__(sys, rho, nu) is not None
    elif rho is None and nu is not None:
        rho = __ispassive__(sys, nu=nu)[-1]
        print(rho)
        return rho
    elif nu is None and rho is not None:
        nu = __ispassive__(sys, rho=rho)[-1]
        print(nu)
        return nu
    else:
        rho = __ispassive__(sys, nu=eps)[-1]
        nu = __ispassive__(sys, rho=eps)[-1]
        print((rho, nu))
        return rho > 0 or nu > 0


def getPassiveIndex(sys, index_type=None):
    '''
    Returns the passivity index associated with the input string. 
    Parameters
    ----------
    sys: An LTI system
        System to be checked.
    index_type: String
        Must be 'input' or 'output'. Indicates which passivity index will be returned. 

    Returns
    -------
    float: 
        The passivity index 
    '''
    if index_type is None:
        raise Exception("Must provide index_type of 'input' or 'output'.")
    if index_type == "input":
        nu = ispassive(sys, rho=eps)
        return nu
    if index_type == "output":
        rho = ispassive(sys, nu=eps)
        return rho
