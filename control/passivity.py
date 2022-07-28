'''
Author: Mark Yeatman  
Date: July 17, 2022
'''

import numpy as np
from control import statesp
from control.exception import ControlArgument, ControlDimension

try:
    import cvxopt as cvx
except ImportError as e:
    cvx = None

eps = np.nextafter(0, 1)

__all__ = ["get_output_fb_index", "get_input_ff_index", 
    "ispassive", "solve_passivity_LMI"]


def solve_passivity_LMI(sys, rho=None, nu=None):
    '''Computes passivity indices and/or solves feasiblity via a linear matrix inequality (LMI).

    Constructs an LMI such that if a solution exists and the last element of the 
    solution is positive, the system is passive. Inputs of None for rho or nu indicates that 
    the function should solve for that index (they are mutually exclusive, they can't both be None, 
    otherwise you're trying to solve a nonconvex bilinear matrix inequality.) The last element is either the 
    output or input passivity index, for rho=None and nu=None respectively.

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
    rho: Float or None
        Output feedback passivity index
    nu: Float or None
        Input feedforward passivity index
        
    Returns
    -------
    numpy array: 
        The LMI solution
    '''
    if cvx is None:
        raise ModuleNotFoundError("cvxopt required for passivity module")

    if sys.ninputs != sys.noutputs:
        raise ControlDimension(
            "The number of system inputs must be the same as the number of system outputs.")

    if rho is None and nu is None:
        raise ControlArgument("rho or nu must be given a numerical value.")

    sys = statesp._convert_to_statespace(sys)

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

    def make_P_basis_matrices(n, rho, nu):
        '''Makes list of matrix constraints for passivity LMI.

        Utility function to make basis matrices for a LMI from a 
        symmetric matrix P of size n by n representing a parametrized symbolic matrix
        '''
        matrix_list = []
        for i in range(0, n):
            for j in range(0, n):
                if j <= i:
                    P = np.zeros((n, n))
                    P[i, j] = 1
                    P[j, i] = 1
                    matrix_list.append(make_LMI_matrix(P, 0, 0, 0).flatten())
        zeros = eps*np.eye(n)
        if rho is None:
            matrix_list.append(make_LMI_matrix(zeros, 1, 0, 0).flatten())
        elif nu is None:
            matrix_list.append(make_LMI_matrix(zeros, 0, 1, 0).flatten())
        return matrix_list


    def P_pos_def_constraint(n):
        '''Makes a list of matrix constraints for P >= 0.

        Utility function to make basis matrices for a LMI that ensures parametrized symbolic matrix 
        of size n by n is positive definite.
        '''
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

    # coefficents for passivity indices and feasibility matrix 
    sys_matrix_list = make_P_basis_matrices(n, rho, nu)

    # get constants for numerical values of rho and nu
    sys_constants = list()
    if rho is not None and nu is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), rho, nu, 1.0)
    elif rho is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), rho, eps, 1.0)
    elif nu is not None:
        sys_constants = -make_LMI_matrix(np.zeros_like(A), eps, nu, 1.0)
    
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
    sol = cvx.solvers.sdp(c, Gs=Gs, hs=hs)
    return sol["x"]


def get_output_fb_index(sys):
    '''Returns the output feedback passivity (OFP) index for the input system. 
    
    The OFP is the largest gain that can be placed in positive feedback 
    with a system such that the new interconnected system is passive.

    Parameters
    ----------
    sys: An LTI system
        System to be checked.

    Returns
    -------
    float: 
        The OFP index 
    '''
    sol = solve_passivity_LMI(sys, nu=eps)
    if sol is None:
        return -np.inf
    else:
        return sol[-1]


def get_input_ff_index(sys):
    '''Returns the input feedforward passivity (IFP) index for the input system. 
    
    The IFP is the largest gain that can be placed in negative parallel interconnection 
    with a system such that the new interconnected system is passive.

    Parameters
    ----------
    sys: An LTI system
        System to be checked.

    Returns
    -------
    float: 
        The IFP index 
    '''

    sol = solve_passivity_LMI(sys, rho=eps)
    if sol is None:
        return -np.inf
    else:
        return sol[-1]


def ispassive(sys, ofp_index = 0, ifp_index = 0):
    '''Indicates if a linear time invariant (LTI) system is passive.

    Checks if system is passive with the given output feedback (OFP) and input feedforward (IFP)
    passivity indices. Querying if the system is passive in the sense of V(x)>=0 and \\dot{V}(x) <= y.T*u, 
    is equiavlent to the default case of ofp_index = 0, ifp_index = 0.

    Note that computing the ofp_index and ifp_index for a system, then using both values simultaneously 
    to as inputs to this function is not guaranteed to have an output of 'True' 
    (the system might not be passive with both indices at the same time). For more details, see:
        McCourt, Michael J., and Panos J. Antsaklis
            "Demonstrating passivity and dissipativity using computational methods." 

    Parameters
    ----------
    sys: An LTI system
        System to be checked.
    ofp_index: float
        Output feedback passivity index.
    ifp_index: float
        Input feedforward passivity index.

    Returns
    -------
    bool: 
        The system is passive.
    '''
    return solve_passivity_LMI(sys, rho = ofp_index, nu = ifp_index) is not None
