"""interconnect_test.py - test input/output interconnect function

RMM, 22 Jan 2021

This set of unit tests covers the various operatons of the interconnect()
function, as well as some of the support functions associated with
interconnect().

Note: additional tests are available in iosys_test.py, which focuses on the
raw InterconnectedSystem constructor.  This set of unit tests focuses on
functionality implemented in the interconnect() function itself.

"""

import pytest

import numpy as np
import scipy as sp

import control as ct

@pytest.mark.parametrize("inputs, output, dimension, D", [
    [1,            1,       None, [[1]] ],
    ['u',          'y',     None, [[1]] ],
    [['u'],        ['y'],   None, [[1]] ],
    [2,            1,       None, [[1, 1]] ],
    [['r', '-y'],  ['e'],   None, [[1, -1]] ],
    [5,            1,       None, np.ones((1, 5)) ],
    ['u',          'y',     1,    [[1]] ],
    ['u',          'y',     2,    [[1, 0], [0, 1]] ],
    [['r', '-y'],  ['e'],   2,    [[1, 0, -1, 0], [0, 1, 0, -1]] ],
])
def test_summing_junction(inputs, output, dimension, D):
    ninputs = 1 if isinstance(inputs, str) else \
        inputs if isinstance(inputs, int) else len(inputs)
    sum = ct.summing_junction(
        inputs=inputs, output=output, dimension=dimension)
    dim = 1 if dimension is None else dimension
    np.testing.assert_allclose(sum.A, np.ndarray((0, 0)))
    np.testing.assert_allclose(sum.B, np.ndarray((0, ninputs*dim)))
    np.testing.assert_allclose(sum.C, np.ndarray((dim, 0)))
    np.testing.assert_allclose(sum.D, D)


def test_summation_exceptions():
    # Bad input description
    with pytest.raises(ValueError, match="could not parse input"):
        sumblk = ct.summing_junction(np.pi, 'y')

    # Bad output description
    with pytest.raises(ValueError, match="could not parse output"):
        sumblk = ct.summing_junction('u', np.pi)

    # Bad input dimension
    with pytest.raises(ValueError, match="unrecognized dimension"):
        sumblk = ct.summing_junction('u', 'y', dimension=False)


def test_interconnect_implicit():
    """Test the use of implicit connections in interconnect()"""
    import random

    # System definition
    P = ct.ss2io(
        ct.rss(2, 1, 1, strictly_proper=True),
        inputs='u', outputs='y', name='P')
    kp = ct.tf(random.uniform(1, 10), [1])
    ki = ct.tf(random.uniform(1, 10), [1, 0])
    C = ct.tf2io(kp + ki, inputs='e', outputs='u', name='C')

    # Block diagram computation
    Tss = ct.feedback(P * C, 1)

    # Construct the interconnection explicitly
    Tio_exp = ct.interconnect(
        (C, P),
        connections = [['P.u', 'C.u'], ['C.e', '-P.y']],
        inplist='C.e', outlist='P.y')

    # Compare to bdalg computation
    np.testing.assert_almost_equal(Tio_exp.A, Tss.A)
    np.testing.assert_almost_equal(Tio_exp.B, Tss.B)
    np.testing.assert_almost_equal(Tio_exp.C, Tss.C)
    np.testing.assert_almost_equal(Tio_exp.D, Tss.D)

    # Construct the interconnection via a summing junction
    sumblk = ct.summing_junction(inputs=['r', '-y'], output='e', name="sum")
    Tio_sum = ct.interconnect(
        (C, P, sumblk), inplist=['r'], outlist=['y'])

    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # Setting connections to False should lead to an empty connection map
    empty = ct.interconnect(
        (C, P, sumblk), connections=False, inplist=['r'], outlist=['y'])
    np.testing.assert_allclose(empty.connect_map, np.zeros((4, 3)))

    # Implicit summation across repeated signals
    kp_io = ct.tf2io(kp, inputs='e', outputs='u', name='kp')
    ki_io = ct.tf2io(ki, inputs='e', outputs='u', name='ki')
    Tio_sum = ct.interconnect(
        (kp_io, ki_io, P, sumblk), inplist=['r'], outlist=['y'])
    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # TODO: interconnect a MIMO system using implicit connections
    # P = control.ss2io(
    #     control.rss(2, 2, 2, strictly_proper=True),
    #     input_prefix='u', output_prefix='y', name='P')
    # C = control.ss2io(
    #     control.rss(2, 2, 2),
    #     input_prefix='e', output_prefix='u', name='C')
    # sumblk = control.summing_junction(
    #     inputs=['r', '-y'], output='e', dimension=2)
    # S = control.interconnect([P, C, sumblk], inplist='r', outlist='y')

    # Make sure that repeated inplist/outlist names work
    pi_io = ct.interconnect(
        (kp_io, ki_io), inplist=['e'], outlist=['u'])
    pi_ss = ct.tf2ss(kp + ki)
    np.testing.assert_almost_equal(pi_io.A, pi_ss.A)
    np.testing.assert_almost_equal(pi_io.B, pi_ss.B)
    np.testing.assert_almost_equal(pi_io.C, pi_ss.C)
    np.testing.assert_almost_equal(pi_io.D, pi_ss.D)

    # Default input and output lists, along with singular versions
    Tio_sum = ct.interconnect(
        (kp_io, ki_io, P, sumblk), input='r', output='y')
    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # Signal not found
    with pytest.raises(ValueError, match="could not find"):
        Tio_sum = ct.interconnect(
            (C, P, sumblk), inplist=['x'], outlist=['y'])

    with pytest.raises(ValueError, match="could not find"):
        Tio_sum = ct.interconnect(
            (C, P, sumblk), inplist=['r'], outlist=['x'])

def test_interconnect_docstring():
    """Test the examples from the interconnect() docstring"""

    # MIMO interconnection (note: use [C, P] instead of [P, C] for state order)
    P = ct.LinearIOSystem(
           ct.rss(2, 2, 2, strictly_proper=True), name='P')
    C = ct.LinearIOSystem(ct.rss(2, 2, 2), name='C')
    T = ct.interconnect(
        [C, P],
        connections = [
          ['P.u[0]', 'C.y[0]'], ['P.u[1]', 'C.y[1]'],
          ['C.u[0]', '-P.y[0]'], ['C.u[1]', '-P.y[1]']],
        inplist = ['C.u[0]', 'C.u[1]'],
        outlist = ['P.y[0]', 'P.y[1]'],
    )
    T_ss = ct.feedback(P * C, ct.ss([], [], [], np.eye(2)))
    np.testing.assert_almost_equal(T.A, T_ss.A)
    np.testing.assert_almost_equal(T.B, T_ss.B)
    np.testing.assert_almost_equal(T.C, T_ss.C)
    np.testing.assert_almost_equal(T.D, T_ss.D)

    # Implicit interconnection (note: use [C, P, sumblk] for proper state order)
    P = ct.tf2io(ct.tf(1, [1, 0]), inputs='u', outputs='y')
    C = ct.tf2io(ct.tf(10, [1, 1]), inputs='e', outputs='u')
    sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    T = ct.interconnect([C, P, sumblk], inplist='r', outlist='y')
    T_ss = ct.feedback(P * C, 1)
    np.testing.assert_almost_equal(T.A, T_ss.A)
    np.testing.assert_almost_equal(T.B, T_ss.B)
    np.testing.assert_almost_equal(T.C, T_ss.C)
    np.testing.assert_almost_equal(T.D, T_ss.D)


def test_interconnect_exceptions():
    # First make sure the docstring example works
    P = ct.tf2io(ct.tf(1, [1, 0]), input='u', output='y')
    C = ct.tf2io(ct.tf(10, [1, 1]), input='e', output='u')
    sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    T = ct.interconnect((P, C, sumblk), input='r', output='y')
    assert (T.ninputs, T.noutputs, T.nstates) == (1, 1, 2)

    # Unrecognized arguments
    # LinearIOSystem
    with pytest.raises(TypeError, match="unknown parameter"):
        P = ct.LinearIOSystem(ct.rss(2, 1, 1), output_name='y')

    # Interconnect
    with pytest.raises(TypeError, match="unknown parameter"):
        T = ct.interconnect((P, C, sumblk), input_name='r', output='y')

    # Interconnected system
    with pytest.raises(TypeError, match="unknown parameter"):
        T = ct.InterconnectedSystem((P, C, sumblk), input_name='r', output='y')

    # NonlinearIOSytem
    with pytest.raises(TypeError, match="unknown parameter"):
        nlios =  ct.NonlinearIOSystem(
            None, lambda t, x, u, params: u*u, input_count=1, output_count=1)

    # Summing junction
    with pytest.raises(TypeError, match="input specification is required"):
        sumblk = ct.summing_junction()

    with pytest.raises(TypeError, match="unknown parameter"):
        sumblk = ct.summing_junction(input_count=2, output_count=2)
