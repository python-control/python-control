'''
Author: Mark Yeatman  
Date: May 30, 2022
'''
import pytest
import numpy
from control import ss, passivity, tf
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


A_d = numpy.array([[-2, 0], [0, 0]])
A = numpy.array([[-3, 0], [0, -2]])
B = numpy.array([[0], [1]])
C = numpy.array([[-1, 2]])
D = numpy.array([[1.5]])


@cvxoptonly
@pytest.mark.parametrize(
    "test_input,expected",
    [((A, B, C, D*0.0), True),
     ((A_d, B, C, D), True),
     ((A*1e12, B, C, D*0), True),
     ((A, B*0, C*0, D), True),
     ((A*0, B, C, D), True),
     ((A*0, B*0, C*0, D*0), True)])
def test_ispassive_edge_cases(test_input, expected):

    # strictly proper
    A = test_input[0]
    B = test_input[1]
    C = test_input[2]
    D = test_input[3]
    sys = ss(A, B, C, D)
    assert(passivity.ispassive(sys) == expected)


def test_transfer_function():
    sys = tf([1], [1, -2])
    assert(passivity.ispassive(sys))

    sys = tf([1], [1, 2])
    assert(not passivity.ispassive(sys))


def test_oo_style():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    sys = ss(A, B, C, D)
    assert(sys.ispassive())

    sys = tf([1], [1, -2])
    assert(sys.ispassive())
