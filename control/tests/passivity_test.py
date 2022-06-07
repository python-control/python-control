'''
Author: Mark Yeatman  
Date: May 30, 2022
'''

import numpy
from control import ss, passivity
from control.tests.conftest import cvxoptonly

@cvxoptonly
def test_is_passive():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    sys = ss(A, B, C, D)

    assert(passivity.is_passive(sys))

    D = -D
    sys = ss(A, B, C, D)

    assert(not passivity.is_passive(sys))

