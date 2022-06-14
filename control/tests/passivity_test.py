'''
Author: Mark Yeatman  
Date: May 30, 2022
'''

import numpy
from control import ss, passivity
from control.tests.conftest import cvxoptonly


@cvxoptonly
def test_ispassive():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    sys = ss(A, B, C, D)

    # happy path is passive
    assert(passivity.ispassive(sys))

    # happy path not passive
    D = -D
    sys = ss(A, B, C, D)

    assert(not passivity.ispassive(sys))


@cvxoptonly
def test_ispassive_edge_cases():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])

    D *= 0

    # strictly proper
    sys = ss(A, B, C, D)
    assert(passivity.ispassive(sys))

    # ill conditioned
    A = A*1e12
    sys = ss(A, B, C, D)
    assert(passivity.ispassive(sys))

    # different combinations of zero A,B,C,D are 0
    B *= 0
    C *= 0
    assert(passivity.ispassive(sys))

    A *= 0
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    assert(passivity.ispassive(sys))

    B *= 0
    C *= 0
    assert(passivity.ispassive(sys))

    A *= 0
    assert(passivity.ispassive(sys))
