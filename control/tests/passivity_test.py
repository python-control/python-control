'''
Author: Mark Yeatman  
Date: May 30, 2022
'''
import pytest
import numpy
from control import ss, passivity, tf, sample_system, parallel, feedback
from control.tests.conftest import cvxoptonly
from control.exception import ControlDimension

pytestmark = cvxoptonly


def test_ispassive_ctime():
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


def test_ispassive_dtime():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    sys = ss(A, B, C, D)
    sys = sample_system(sys, 1, method='bilinear')
    assert(passivity.ispassive(sys))


def test_passivity_indices_ctime():
    sys = tf([1, 1, 5, 0.1], [1, 2, 3, 4])

    nu = passivity.getPassiveIndex(sys, 'input')
    rho = passivity.getPassiveIndex(sys, 'output')

    assert(isinstance(nu, float))

    sys_ff_nu = parallel(-nu, sys)
    sys_fb_rho = feedback(rho, sys, sign=1)

    assert(sys_ff_nu.ispassive())
    assert(sys_fb_rho.ispassive())

    sys_ff_nu = parallel(-nu-1e-6, sys)
    sys_fb_rho = feedback(rho+1e-6, sys, sign=1)

    assert(not sys_ff_nu.ispassive())
    assert(not sys_fb_rho.ispassive())


def test_passivity_indices_dtime():
    sys = tf([1, 1, 5, 0.1], [1, 2, 3, 4])
    sys = sample_system(sys, Ts=0.01, alpha=1, method="bilinear")
    nu = passivity.getPassiveIndex(sys, 'input')
    rho = passivity.getPassiveIndex(sys, 'output')

    assert(isinstance(nu, float))

    sys_ff_nu = parallel(-nu, sys)
    sys_fb_rho = feedback(rho, sys, sign=1)

    assert(sys_ff_nu.ispassive())
    assert(sys_fb_rho.ispassive())

    sys_ff_nu = parallel(-nu-1e-6, sys)
    sys_fb_rho = feedback(rho+1e-6, sys, sign=1)

    assert(not sys_ff_nu.ispassive())
    assert(not sys_fb_rho.ispassive())


def test_system_dimension():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2], [0, 1]])
    D = numpy.array([[1.5], [1]])
    sys = ss(A, B, C, D)

    with pytest.raises(ControlDimension):
        passivity.ispassive(sys)


A_d = numpy.array([[-2, 0], [0, 0]])
A = numpy.array([[-3, 0], [0, -2]])
B = numpy.array([[0], [1]])
C = numpy.array([[-1, 2]])
D = numpy.array([[1.5]])


@pytest.mark.parametrize(
    "test_input,expected",
    [((A, B, C, D*0.0), True),
     ((A_d, B, C, D), True),
     ((A*1e12, B, C, D*0), True),
     ((A, B*0, C*0, D), True),
     ((A*0, B, C, D), True)])
def test_ispassive_edge_cases(test_input, expected):
    A = test_input[0]
    B = test_input[1]
    C = test_input[2]
    D = test_input[3]
    sys = ss(A, B, C, D)
    assert(passivity.ispassive(sys) == expected)


def test_ispassive_all_zeros():
    A = numpy.array([[0]])
    B = numpy.array([[0]])
    C = numpy.array([[0]])
    D = numpy.array([[0]])
    sys = ss(A, B, C, D)

    with pytest.raises(ValueError):
        passivity.ispassive(sys)


def test_transfer_function():
    sys = tf([1], [1, 2])
    assert(passivity.ispassive(sys))

    sys = tf([1], [1, -2])
    assert(not passivity.ispassive(sys))


def test_oo_style():
    A = numpy.array([[0, 1], [-2, -2]])
    B = numpy.array([[0], [1]])
    C = numpy.array([[-1, 2]])
    D = numpy.array([[1.5]])
    sys = ss(A, B, C, D)
    assert(sys.ispassive())

    sys = tf([1], [1, 2])
    assert(sys.ispassive())
