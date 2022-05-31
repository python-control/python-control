'''
Author: Mark Yeatman  
Date: May 15, 2022
'''

from . import statesp as ss
from sympy import symbols, Matrix, symarray
from lmi_sdp import LMI_NSD, to_cvxopt
from cvxopt import solvers

import numpy as np


def is_passive(sys):
    '''
    Indicates if a linear time invarient system is passive

    Constructs a linear matrix inequality and a feasibility optimization
    such that is a solution exists, the system is passive.

    The source for the algorithm is: 
    McCourt, Michael J., and Panos J. Antsaklis. "Demonstrating passivity and dissipativity using computational methods." ISIS 8 (2013).
    '''

    A = sys.A
    B = sys.B
    C = sys.C
    D = sys.D

    P = Matrix(symarray('p', A.shape))

    # enforce symmetry in P
    size = A.shape[0]
    for i in range(0, size):
        for j in range(0, size):
            P[i, j] = P[j, i]

    # construct matrix for storage function x'*V*x
    V = Matrix.vstack(
        Matrix.hstack(A.T * P + P*A, P*B - C.T),
        Matrix.hstack(B.T*P - C, Matrix(-D - D.T))
    )

    # construct LMI, convert to form for feasibility solver
    LMI_passivty = LMI_NSD(V, 0*V)
    min_obj = 0 * symbols("x")
    variables = V.free_symbols
    solvers.options['show_progress'] = False
    c, Gs, hs = to_cvxopt(min_obj, LMI_passivty, variables)

    # crunch feasibility solution
    sol = solvers.sdp(c, Gs=Gs, hs=hs)

    return (sol["x"] is not None)
