'''
Author: Mark Yeatman  
Date: May 15, 2022
'''

import numpy as np
from control import statesp as ss

try:
    import cvxopt as cvx
except ImportError as e:
    cvx = None


def ispassive(sys):
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

    Returns
    -------
    bool: 
        The input system passive.
    '''
    if cvx is None:
        raise ModuleNotFoundError("cvxopt required for passivity module")

    sys = ss._convert_to_statespace(sys)

    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    # account for strictly proper systems
    [n, m] = D.shape
    D = D + np.nextafter(0, 1)*np.eye(n, m)

    [n, _] = A.shape
    A = A - np.nextafter(0, 1)*np.eye(n)

    def make_LMI_matrix(P):
        if sys.isctime():
            return np.vstack((
                np.hstack((A.T @ P + P@A, P@B)),
                np.hstack((B.T@P, np.zeros_like(D))))
            )
        else:
            return np.vstack((
                np.hstack((2*A.T @ P @A - P , 2*A.T @ P@B)),
                np.hstack(((2*A.T @ P@B).T, 2*B.T@P@B)))
            )
        

    matrix_list = []
    state_space_size = sys.nstates
    for i in range(0, state_space_size):
        for j in range(0, state_space_size):
            if j <= i:
                P = np.zeros_like(A)
                P[i, j] = 1.0
                P[j, i] = 1.0
                matrix_list.append(make_LMI_matrix(P).flatten())

    coefficents = np.vstack(matrix_list).T

    constants = -np.vstack((
        np.hstack((np.zeros_like(A),  - C.T)),
        np.hstack((- C, -D - D.T)))
    )

    number_of_opt_vars = int(
        (state_space_size**2-state_space_size)/2 + state_space_size)
    c = cvx.matrix(0.0, (number_of_opt_vars, 1))

    # crunch feasibility solution
    cvx.solvers.options['show_progress'] = False
    sol = cvx.solvers.sdp(c,
                          Gs=[cvx.matrix(coefficents)],
                          hs=[cvx.matrix(constants)])

    return (sol["x"] is not None)
