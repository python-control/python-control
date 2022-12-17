'''
Author: Mark Yeatman
Date: May 30, 2022
'''
import pytest
import numpy
from control import ss, passivity, tf, sample_system, parallel, feedback
from control.tests.conftest import cvxoptonly
from control.exception import ControlArgument, ControlDimension

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

    iff_index = passivity.get_input_ff_index(sys)
    ofb_index = passivity.get_output_fb_index(sys)

    assert(isinstance(ofb_index, float))

    sys_ff = parallel(sys, -iff_index)
    sys_fb = feedback(sys, ofb_index, sign=1)

    assert(sys_ff.ispassive())
    assert(sys_fb.ispassive())

    sys_ff = parallel(sys, -iff_index-1e-6)
    sys_fb = feedback(sys, ofb_index+1e-6,  sign=1)

    assert(not sys_ff.ispassive())
    assert(not sys_fb.ispassive())


def test_passivity_indices_dtime():
    sys = tf([1, 1, 5, 0.1], [1, 2, 3, 4])
    sys = sample_system(sys, Ts=0.1)
    iff_index = passivity.get_input_ff_index(sys)
    ofb_index = passivity.get_output_fb_index(sys)

    assert(isinstance(iff_index, float))

    sys_ff = parallel(sys, -iff_index)
    sys_fb = feedback(sys, ofb_index,  sign=1)

    assert(sys_ff.ispassive())
    assert(sys_fb.ispassive())

    sys_ff = parallel(sys, -iff_index-1e-2)
    sys_fb = feedback(sys, ofb_index+1e-2, sign=1)

    assert(not sys_ff.ispassive())
    assert(not sys_fb.ispassive())


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
    "system_matrices, expected",
    [((A, B, C, D*0), True),
     ((A_d, B, C, D), True),
     ((A, B*0, C*0, D), True),
     ((A*0, B, C, D), True),
     ((A*0, B*0, C*0, D*0), True)])
def test_ispassive_edge_cases(system_matrices, expected):
    sys = ss(*system_matrices)
    assert passivity.ispassive(sys) == expected


def test_rho_and_nu_are_none():
    A = numpy.array([[0]])
    B = numpy.array([[0]])
    C = numpy.array([[0]])
    D = numpy.array([[0]])
    sys = ss(A, B, C, D)

    with pytest.raises(ControlArgument):
        passivity.solve_passivity_LMI(sys)


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
