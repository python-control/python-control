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
def test_summation_block(inputs, output, dimension, D):
    ninputs = 1 if isinstance(inputs, str) else \
        inputs if isinstance(inputs, int) else len(inputs)
    sum = ct.summation_block(
        inputs=inputs, output=output, dimension=dimension)
    dim = 1 if dimension is None else dimension
    np.testing.assert_array_equal(sum.A, np.ndarray((0, 0)))
    np.testing.assert_array_equal(sum.B, np.ndarray((0, ninputs*dim)))
    np.testing.assert_array_equal(sum.C, np.ndarray((dim, 0)))
    np.testing.assert_array_equal(sum.D, D)


def test_summation_exceptions():
    # Bad input description
    with pytest.raises(ValueError, match="could not parse input"):
        sumblk = ct.summation_block(None, 'y')

    # Bad output description
    with pytest.raises(ValueError, match="could not parse output"):
        sumblk = ct.summation_block('u', None)

    # Bad input dimension
    with pytest.raises(ValueError, match="unrecognized dimension"):
        sumblk = ct.summation_block('u', 'y', dimension=False)


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

    # Construct the interconnection via a summation block
    sumblk = ct.summation_block(inputs=['r', '-y'], output='e', name="sum")
    Tio_sum = ct.interconnect(
        (C, P, sumblk), inplist=['r'], outlist=['y'])

    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # Setting connections to False should lead to an empty connection map
    empty = ct.interconnect(
        (C, P, sumblk), connections=False, inplist=['r'], outlist=['y'])
    np.testing.assert_array_equal(empty.connect_map, np.zeros((4, 3)))

    # Implicit summation across repeated signals
    kp_io = ct.tf2io(kp, inputs='e', outputs='u', name='kp')
    ki_io = ct.tf2io(ki, inputs='e', outputs='u', name='ki')
    Tio_sum = ct.interconnect(
        (kp_io, ki_io, P, sumblk), inplist=['r'], outlist=['y'])
    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # Make sure that repeated inplist/outlist names generate an error
    # Input not unique
    Cbad = ct.tf2io(ct.tf(10, [1, 1]), inputs='r', outputs='x', name='C')
    with pytest.raises(ValueError, match="not unique"):
        Tio_sum = ct.interconnect(
            (Cbad, P, sumblk), inplist=['r'], outlist=['y'])

    # Output not unique
    Cbad = ct.tf2io(ct.tf(10, [1, 1]), inputs='e', outputs='y', name='C')
    with pytest.raises(ValueError, match="not unique"):
        Tio_sum = ct.interconnect(
            (Cbad, P, sumblk), inplist=['r'], outlist=['y'])

    # Signal not found
    with pytest.raises(ValueError, match="could not find"):
        Tio_sum = ct.interconnect(
            (C, P, sumblk), inplist=['x'], outlist=['y'])

    with pytest.raises(ValueError, match="could not find"):
        Tio_sum = ct.interconnect(
            (C, P, sumblk), inplist=['r'], outlist=['x'])
