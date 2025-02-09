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
import math

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
        ct.summing_junction(np.pi, 'y')

    # Bad output description
    with pytest.raises(ValueError, match="could not parse output"):
        ct.summing_junction('u', np.pi)

    # Bad input dimension
    with pytest.raises(ValueError, match="unrecognized dimension"):
        ct.summing_junction('u', 'y', dimension=False)


@pytest.mark.parametrize("dim", [1, 3])
def test_interconnect_implicit(dim):
    """Test the use of implicit connections in interconnect()"""
    import random

    if dim != 1 and not ct.slycot_check():
        pytest.xfail("slycot not installed")

    # System definition
    P = ct.rss(2, dim, dim, strictly_proper=True, name='P')

    # Controller defintion: PI in each input/output pair
    kp = ct.tf(np.ones((dim, dim, 1)), np.ones((dim, dim, 1))) \
        * random.uniform(1, 10)
    ki = random.uniform(1, 10)
    num, den = np.zeros((dim, dim, 1)), np.ones((dim, dim, 2))
    for i, j in zip(range(dim), range(dim)):
        num[i, j] = ki
        den[i, j] = np.array([1, 0])
    ki = ct.tf(num, den)
    C = ct.tf(kp + ki, name='C',
              inputs=[f'e[{i}]' for i in range(dim)],
              outputs=[f'u[{i}]' for i in range(dim)])

    # same but static C2
    C2 = ct.tf(kp * random.uniform(1, 10), name='C2',
               inputs=[f'e[{i}]' for i in range(dim)],
               outputs=[f'u[{i}]' for i in range(dim)])

    # Block diagram computation
    Tss = ct.feedback(P * C, np.eye(dim))
    Tss2 = ct.feedback(P * C2, np.eye(dim))

    # Construct the interconnection explicitly
    Tio_exp = ct.interconnect(
        (C, P),
        connections=[['P.u', 'C.u'], ['C.e', '-P.y']],
        inplist='C.e', outlist='P.y')

    # Compare to bdalg computation
    np.testing.assert_almost_equal(Tio_exp.A, Tss.A)
    np.testing.assert_almost_equal(Tio_exp.B, Tss.B)
    np.testing.assert_almost_equal(Tio_exp.C, Tss.C)
    np.testing.assert_almost_equal(Tio_exp.D, Tss.D)

    # Construct the interconnection via a summing junction
    sumblk = ct.summing_junction(
        inputs=['r', '-y'], output='e', dimension=dim, name="sum")
    Tio_sum = ct.interconnect(
        [C, P, sumblk], inplist=['r'], outlist=['y'], debug=True)

    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # test whether signal names work for static system C2
    Tio_sum2 = ct.interconnect(
        [C2, P, sumblk], inplist='r', outlist='y')

    np.testing.assert_almost_equal(Tio_sum2.A, Tss2.A)
    np.testing.assert_almost_equal(Tio_sum2.B, Tss2.B)
    np.testing.assert_almost_equal(Tio_sum2.C, Tss2.C)
    np.testing.assert_almost_equal(Tio_sum2.D, Tss2.D)

    # Setting connections to False should lead to an empty connection map
    empty = ct.interconnect(
        [C, P, sumblk], connections=False, inplist=['r'], outlist=['y'])
    np.testing.assert_allclose(empty.connect_map, np.zeros((4*dim, 3*dim)))

    # Implicit summation across repeated signals (using updated labels)
    kp_io = ct.tf(
        kp, inputs=dim, input_prefix='e',
        outputs=dim, output_prefix='u', name='kp')
    ki_io = ct.tf(
        ki, inputs=dim, input_prefix='e',
        outputs=dim, output_prefix='u', name='ki')
    Tio_sum = ct.interconnect(
        [kp_io, ki_io, P, sumblk], inplist=['r'], outlist=['y'])
    np.testing.assert_almost_equal(Tio_sum.A, Tss.A)
    np.testing.assert_almost_equal(Tio_sum.B, Tss.B)
    np.testing.assert_almost_equal(Tio_sum.C, Tss.C)
    np.testing.assert_almost_equal(Tio_sum.D, Tss.D)

    # Make sure that repeated inplist/outlist names work
    pi_io = ct.interconnect(
        [kp_io, ki_io], inplist=['e'], outlist=['u'])
    pi_ss = ct.tf2ss(kp + ki)
    np.testing.assert_almost_equal(pi_io.A, pi_ss.A)
    np.testing.assert_almost_equal(pi_io.B, pi_ss.B)
    np.testing.assert_almost_equal(pi_io.C, pi_ss.C)
    np.testing.assert_almost_equal(pi_io.D, pi_ss.D)

    # Default input and output lists, along with singular versions
    Tio_sum = ct.interconnect(
        [kp_io, ki_io, P, sumblk], input='r', output='y', debug=True)
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
    P = ct.StateSpace(
           ct.rss(2, 2, 2, strictly_proper=True), name='P')
    C = ct.StateSpace(ct.rss(2, 2, 2), name='C')
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
    P = ct.tf(1, [1, 0], inputs='u', outputs='y')
    C = ct.tf(10, [1, 1], inputs='e', outputs='u')
    sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    T = ct.interconnect([C, P, sumblk], inplist='r', outlist='y')
    T_ss = ct.ss(ct.feedback(P * C, 1))

    # Test in a manner that recognizes that recognizes non-unique realization
    np.testing.assert_almost_equal(
        np.sort(np.linalg.eig(T.A)[0]), np.sort(np.linalg.eig(T_ss.A)[0]))
    np.testing.assert_almost_equal(T.C @ T.B, T_ss.C @ T_ss.B)
    np.testing.assert_almost_equal(T.C @ T. A @ T.B, T_ss.C @ T_ss.A @ T_ss.B)
    np.testing.assert_almost_equal(T.D, T_ss.D)

@pytest.mark.parametrize("show_names", (True, False))
def test_connection_table(capsys, show_names):
    P = ct.ss(1,1,1,0, inputs='u', outputs='y', name='P')
    C = ct.tf(10, [.1, 1], inputs='e', outputs='u', name='C')
    L = ct.interconnect([C, P], inputs='e', outputs='y')
    L.connection_table(show_names=show_names)
    captured_from_method = capsys.readouterr().out

    ct.connection_table(L, show_names=show_names)
    captured_from_function = capsys.readouterr().out

    # break the following strings separately because the printout order varies
    # because signal names are stored as a set
    mystrings = \
            ["signal    | source                        | destination",
             "------------------------------------------------------------------"]
    if show_names:
        mystrings += \
            ["e         | input                         | C",
             "u         | C                             | P",
             "y         | P                             | output"]
    else:
        mystrings += \
            ["e         | input                         | system 0",
             "u         | system 0                      | system 1",
             "y         | system 1                      | output"]

    for str_ in mystrings:
        assert str_ in captured_from_method
        assert str_ in captured_from_function

    # check auto-sum
    P1 = ct.ss(1,1,1,0, inputs='u', outputs='y', name='P1')
    P2 = ct.tf(10, [.1, 1], inputs='e', outputs='y', name='P2')
    P3 = ct.tf(10, [.1, 1], inputs='x', outputs='y', name='P3')
    P = ct.interconnect([P1, P2, P3], inputs=['e', 'u', 'x'], outputs='y')
    P.connection_table(show_names=show_names)
    captured_from_method = capsys.readouterr().out

    ct.connection_table(P, show_names=show_names)
    captured_from_function = capsys.readouterr().out

    mystrings = \
            ["signal    | source                        | destination",
     "-------------------------------------------------------------------"]
    if show_names:
        mystrings += \
            ["u         | input                         | P1",
             "e         | input                         | P2",
             "x         | input                         | P3",
             "y         | P1, P2, P3                    | output"]
    else:
        mystrings += \
            ["u         | input                         | system 0",
             "e         | input                         | system 1",
             "x         | input                         | system 2",
             "y         | system 0, system 1, system 2  | output"]

    for str_ in mystrings:
        assert str_ in captured_from_method
        assert str_ in captured_from_function

    # check auto-split
    P1 = ct.ss(1,1,1,0, inputs='u', outputs='x', name='P1')
    P2 = ct.tf(10, [.1, 1], inputs='u', outputs='y', name='P2')
    P3 = ct.tf(10, [.1, 1], inputs='u', outputs='z', name='P3')
    P = ct.interconnect([P1, P2, P3], inputs=['u'], outputs=['x','y','z'])
    P.connection_table(show_names=show_names)
    captured_from_method = capsys.readouterr().out

    ct.connection_table(P, show_names=show_names)
    captured_from_function = capsys.readouterr().out

    mystrings = \
            ["signal    | source                        | destination",
             "-------------------------------------------------------------------"]
    if show_names:
        mystrings += \
            ["u         | input                         | P1, P2, P3",
             "x         | P1                            | output  ",
             "y         | P2                            | output",
             "z         | P3                            | output"]
    else:
        mystrings += \
            ["u         | input                         | system 0, system 1, system 2",
             "x         | system 0                      | output  ",
             "y         | system 1                      | output",
             "z         | system 2                      | output"]

    for str_ in mystrings:
        assert str_ in captured_from_method
        assert str_ in captured_from_function

    # check change column width
    P.connection_table(show_names=show_names, column_width=20)
    captured_from_method = capsys.readouterr().out

    ct.connection_table(P, show_names=show_names, column_width=20)
    captured_from_function = capsys.readouterr().out

    mystrings = \
            ["signal    | source            | destination",
             "------------------------------------------------"]
    if show_names:
        mystrings += \
            ["u         | input             | P1, P2, P3",
             "x         | P1                | output  ",
             "y         | P2                | output",
             "z         | P3                | output"]
    else:
        mystrings += \
            ["u         | input             | system 0, syste.. ",
             "x         | system 0          | output  ",
             "y         | system 1          | output",
             "z         | system 2          | output"]

    for str_ in mystrings:
        assert str_ in captured_from_method
        assert str_ in captured_from_function


def test_interconnect_exceptions():
    # First make sure the docstring example works
    P = ct.tf(1, [1, 0], input='u', output='y')
    C = ct.tf(10, [1, 1], input='e', output='u')
    sumblk = ct.summing_junction(inputs=['r', '-y'], output='e')
    T = ct.interconnect((P, C, sumblk), input='r', output='y')
    assert (T.ninputs, T.noutputs, T.nstates) == (1, 1, 2)

    # Unrecognized arguments
    # StateSpace
    with pytest.raises(TypeError, match="unrecognized keyword"):
        P = ct.StateSpace(ct.rss(2, 1, 1), output_name='y')

    # Interconnect
    with pytest.raises(TypeError, match="unrecognized keyword"):
        T = ct.interconnect((P, C, sumblk), input_name='r', output='y')

    # Interconnected system
    with pytest.raises(TypeError, match="unrecognized keyword"):
        T = ct.InterconnectedSystem((P, C, sumblk), input_name='r', output='y')

    # NonlinearIOSytem
    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.NonlinearIOSystem(
            None, lambda t, x, u, params: u*u, input_count=1, output_count=1)

    # Summing junction
    with pytest.raises(TypeError, match="input specification is required"):
        sumblk = ct.summing_junction()

    with pytest.raises(TypeError, match="unrecognized keyword"):
        sumblk = ct.summing_junction(input_count=2, output_count=2)


def test_string_inputoutput():
    # regression test for gh-692
    P1 = ct.rss(2, 1, 1)
    P1_iosys = ct.StateSpace(P1, inputs='u1', outputs='y1')
    P2 = ct.rss(2, 1, 1)
    P2_iosys = ct.StateSpace(P2, inputs='y1', outputs='y2')

    P_s1 = ct.interconnect(
        [P1_iosys, P2_iosys], inputs='u1', outputs=['y2'], debug=True)
    assert P_s1.input_index == {'u1' : 0}
    assert P_s1.output_index == {'y2' : 0}

    P_s2 = ct.interconnect([P1_iosys, P2_iosys], input='u1', outputs=['y2'])
    assert P_s2.input_index == {'u1' : 0}
    assert P_s2.output_index == {'y2' : 0}

    P_s1 = ct.interconnect([P1_iosys, P2_iosys], inputs=['u1'], outputs='y2')
    assert P_s1.input_index == {'u1' : 0}
    assert P_s1.output_index == {'y2' : 0}

    P_s2 = ct.interconnect([P1_iosys, P2_iosys], inputs=['u1'], output='y2')
    assert P_s2.input_index == {'u1' : 0}
    assert P_s2.output_index == {'y2' : 0}


def test_linear_interconnect():
    tf_ctrl = ct.tf(1, (10.1, 1), inputs='e', outputs='u', name='ctrl')
    tf_plant = ct.tf(1, (10.1, 1), inputs='u', outputs='y', name='plant')
    ss_ctrl = ct.ss(1, 2, 1, 0, inputs='e', outputs='u', name='ctrl')
    ss_plant = ct.ss(1, 2, 1, 0, inputs='u', outputs='y', name='plant')
    nl_ctrl = ct.NonlinearIOSystem(
        lambda t, x, u, params: x*x, lambda t, x, u, params: u*x,
        states=1, inputs='e', outputs='u', name='ctrl')
    nl_plant = ct.NonlinearIOSystem(
        lambda t, x, u, params: x*x, lambda t, x, u, params: u*x,
        states=1, inputs='u', outputs='y', name='plant')
    sumblk = ct.summing_junction(inputs=['r', '-y'], outputs=['e'], name='sum')

    # Interconnections of linear I/O systems should be linear I/O system
    assert isinstance(
        ct.interconnect([tf_ctrl, tf_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert isinstance(
        ct.interconnect([ss_ctrl, ss_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert isinstance(
        ct.interconnect([tf_ctrl, ss_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert isinstance(
        ct.interconnect([ss_ctrl, tf_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)

    # Interconnections with nonliner I/O systems should not be linear
    assert not isinstance(
        ct.interconnect([nl_ctrl, ss_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert not isinstance(
        ct.interconnect([nl_ctrl, tf_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert not isinstance(
        ct.interconnect([ss_ctrl, nl_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)
    assert not isinstance(
        ct.interconnect([tf_ctrl, nl_plant, sumblk], inputs='r', outputs='y'),
        ct.StateSpace)

    # Implicit converstion of transfer function should retain name
    clsys = ct.interconnect(
        [tf_ctrl, ss_plant, sumblk],
        connections=[
            ['plant.u', 'ctrl.u'],
            ['ctrl.e', 'sum.e'],
            ['sum.y', 'plant.y']
        ],
        inplist=['sum.r'], inputs='r',
        outlist=['plant.y'], outputs='y')
    assert clsys.syslist[0].name == 'ctrl'

@pytest.mark.parametrize(
    "connections, inplist, outlist, inputs, outputs", [
        pytest.param(
            [['sys2', 'sys1']], 'sys1', 'sys2', None, None,
            id="sysname only, no i/o args"),
        pytest.param(
            [['sys2', 'sys1']], 'sys1', 'sys2', 3, 3,
            id="i/o signal counts"),
        pytest.param(
            [[('sys2', [0, 1, 2]), ('sys1', [0, 1, 2])]],
            [('sys1', [0, 1, 2])], [('sys2', [0, 1, 2])],
            3, 3,
            id="signal lists, i/o counts"),
        pytest.param(
            [['sys2.u[0:3]', 'sys1.y[:]']],
            'sys1.u[:]', ['sys2.y[0:3]'], None, None,
            id="signal slices"),
        pytest.param(
            ['sys2.u', 'sys1.y'], 'sys1.u', 'sys2.y', None, None,
            id="signal basenames"),
        pytest.param(
            [[('sys2', [0, 1, 2]), ('sys1', [0, 1, 2])]],
            [('sys1', [0, 1, 2])], [('sys2', [0, 1, 2])],
            None, None,
            id="signal lists, no i/o counts"),
        pytest.param(
            [[(1, ['u[0]', 'u[1]', 'u[2]']), (0, ['y[0]', 'y[1]', 'y[2]'])]],
            [('sys1', [0, 1, 2])], [('sys2', [0, 1, 2])],
            3, ['y1', 'y2', 'y3'],
            id="mixed specs"),
        pytest.param(
            [[f'sys2.u[{i}]', f'sys1.y[{i}]'] for i in range(3)],
            [f'sys1.u[{i}]' for i in range(3)],
            [f'sys2.y[{i}]' for i in range(3)],
            [f'u[{i}]' for i in range(3)], [f'y[{i}]' for i in range(3)],
            id="full enumeration"),
])
def test_interconnect_series(connections, inplist, outlist, inputs, outputs):
    # Create an interconnected system for testing
    sys1 = ct.rss(4, 3, 3, name='sys1')
    sys2 = ct.rss(4, 3, 3, name='sys2')
    series = sys2 * sys1

    # Simple series interconnection
    icsys = ct.interconnect(
        [sys1, sys2], connections=connections,
        inplist=inplist, outlist=outlist, inputs=inputs, outputs=outputs
    )
    np.testing.assert_allclose(icsys.A, series.A)
    np.testing.assert_allclose(icsys.B, series.B)
    np.testing.assert_allclose(icsys.C, series.C)
    np.testing.assert_allclose(icsys.D, series.D)


@pytest.mark.parametrize(
    "connections, inplist, outlist", [
    pytest.param(
        [['P', 'C'], ['C', '-P']], 'C', 'P',
        id="sysname only, no i/o args"),
    pytest.param(
        [['P.u', 'C.y'], ['C.u', '-P.y']], 'C.u', 'P.y',
        id="sysname only, no i/o args"),
    pytest.param(
        [['P.u[:]', 'C.y[0:2]'],
         [('C', 'u'), ('P', ['y[0]', 'y[1]'], -1)]],
        ['C.u[0]', 'C.u[1]'], ('P', [0, 1]),
        id="mixed cases"),
])
def test_interconnect_feedback(connections, inplist, outlist):
    # Create an interconnected system for testing
    P = ct.rss(4, 2, 2, name='P', strictly_proper=True)
    C = ct.rss(4, 2, 2, name='C')
    feedback = ct.feedback(P * C, np.eye(2))

    # Simple feedback interconnection
    icsys = ct.interconnect(
        [C, P], connections=connections,
        inplist=inplist, outlist=outlist
    )
    np.testing.assert_allclose(icsys.A, feedback.A)
    np.testing.assert_allclose(icsys.B, feedback.B)
    np.testing.assert_allclose(icsys.C, feedback.C)
    np.testing.assert_allclose(icsys.D, feedback.D)


@pytest.mark.parametrize(
    "pinputs, poutputs, connections, inplist, outlist", [
    pytest.param(
        ['w[0]', 'w[1]', 'u[0]', 'u[1]'],               # pinputs
        ['z[0]', 'z[1]', 'y[0]', 'y[1]'],               # poutputs
        [[('P', [2, 3]), ('C', [0, 1])], [('C', [0, 1]), ('P', [2, 3], -1)]],
        [('C', [0, 1]), ('P', [0, 1])],                 # inplist
        [('P', [0, 1, 2, 3]), ('C', [0, 1])],           # outlist
        id="signal indices"),
    pytest.param(
        ['w[0]', 'w[1]', 'u[0]', 'u[1]'],               # pinputs
        ['z[0]', 'z[1]', 'y[0]', 'y[1]'],               # poutputs
        [[('P', [2, 3]), ('C', [0, 1])], [('C', [0, 1]), ('P', [2, 3], -1)]],
        ['C', ('P', [0, 1])], ['P', 'C'],               # inplist, outlist
        id="signal indices, when needed"),
    pytest.param(
        4, 4,                                           # default I/O names
        [['P.u[2:4]', 'C.y[:]'], ['C.u', '-P.y[2:]']],
        ['C', 'P.u[:2]'], ['P.y[:]', 'P.u[2:]'],        # inplist, outlist
        id="signal slices"),
    pytest.param(
        ['w[0]', 'w[1]', 'u[0]', 'u[1]'],               # pinputs
        ['z[0]', 'z[1]', 'y[0]', 'y[1]'],               # poutputs
        [['P.u', 'C.y'], ['C.u', '-P.y']],              # connections
        ['C.u', 'P.w'], ['P.z', 'P.y', 'C.y'],          # inplist, outlist
        id="basename, control output"),
    pytest.param(
        ['w[0]', 'w[1]', 'u[0]', 'u[1]'],               # pinputs
        ['z[0]', 'z[1]', 'y[0]', 'y[1]'],               # poutputs
        [['P.u', 'C.y'], ['C.u', '-P.y']],              # connections
        ['C.u', 'P.w'], ['P.z', 'P.y', 'P.u'],          # inplist, outlist
        id="basename, process input"),
])
def test_interconnect_partial_feedback(
        pinputs, poutputs, connections, inplist, outlist):
    P = ct.rss(
        states=6, name='P', strictly_proper=True,
        inputs=pinputs, outputs=poutputs)
    C = ct.rss(4, 2, 2, name='C')

    # Low level feedback connection (feedback around "lower" process I/O)
    partial = ct.interconnect(
        [C, P],
        connections=[
            [(1, 2), (0, 0)], [(1, 3), (0, 1)],
            [(0, 0), (1, 2, -1)], [(0, 1), (1, 3, -1)]],
        inplist=[(0, 0), (0, 1), (1, 0), (1, 1)],       # C.u, P.w
        outlist=[(1, 0), (1, 1), (1, 2), (1, 3),
                 (0, 0), (0, 1)],                       # P.z, P.y, C.y
    )

    # High level feedback conections
    icsys = ct.interconnect(
        [C, P], connections=connections,
        inplist=inplist, outlist=outlist
    )
    np.testing.assert_allclose(icsys.A, partial.A)
    np.testing.assert_allclose(icsys.B, partial.B)
    np.testing.assert_allclose(icsys.C, partial.C)
    np.testing.assert_allclose(icsys.D, partial.D)


def test_interconnect_doctest():
    P = ct.rss(
        states=6, name='P', strictly_proper=True,
        inputs=['u[0]', 'u[1]', 'v[0]', 'v[1]'],
        outputs=['y[0]', 'y[1]', 'z[0]', 'z[1]'])
    C = ct.rss(4, 2, 2, name='C', input_prefix='e', output_prefix='u')
    sumblk = ct.summing_junction(
        inputs=['r', '-y'], outputs='e', dimension=2, name='sum')

    clsys1 = ct.interconnect(
        [C, P, sumblk],
        connections=[
            ['P.u[0]', 'C.u[0]'], ['P.u[1]', 'C.u[1]'],
            ['C.e[0]', 'sum.e[0]'], ['C.e[1]', 'sum.e[1]'],
            ['sum.y[0]', 'P.y[0]'], ['sum.y[1]', 'P.y[1]'],
        ],
        inplist=['sum.r[0]', 'sum.r[1]', 'P.v[0]', 'P.v[1]'],
        outlist=['P.y[0]', 'P.y[1]', 'P.z[0]', 'P.z[1]', 'C.u[0]', 'C.u[1]']
    )

    clsys2 = ct.interconnect(
        [C, P, sumblk],
        connections=[
            ['P.u[0:2]', 'C.u[0:2]'],
            ['C.e[0:2]', 'sum.e[0:2]'],
            ['sum.y[0:2]', 'P.y[0:2]']
        ],
        inplist=['sum.r[0:2]', 'P.v[0:2]'],
        outlist=['P.y[0:2]', 'P.z[0:2]', 'C.u[0:2]']
    )
    np.testing.assert_equal(clsys2.A, clsys1.A)
    np.testing.assert_equal(clsys2.B, clsys1.B)
    np.testing.assert_equal(clsys2.C, clsys1.C)
    np.testing.assert_equal(clsys2.D, clsys1.D)

    clsys3 = ct.interconnect(
        [C, P, sumblk],
        connections=[['P.u', 'C.u'], ['C.e', 'sum.e'], ['sum.y', 'P.y']],
        inplist=['sum.r', 'P.v'], outlist=['P.y', 'P.z', 'C.u']
    )
    np.testing.assert_equal(clsys3.A, clsys1.A)
    np.testing.assert_equal(clsys3.B, clsys1.B)
    np.testing.assert_equal(clsys3.C, clsys1.C)
    np.testing.assert_equal(clsys3.D, clsys1.D)

    clsys4 = ct.interconnect(
        [C, P, sumblk],
        connections=[['P.u', 'C'], ['C', 'sum'], ['sum.y', 'P.y']],
        inplist=['sum.r', 'P.v'], outlist=['P', 'C.u']
    )
    np.testing.assert_equal(clsys4.A, clsys1.A)
    np.testing.assert_equal(clsys4.B, clsys1.B)
    np.testing.assert_equal(clsys4.C, clsys1.C)
    np.testing.assert_equal(clsys4.D, clsys1.D)

    clsys5 = ct.interconnect(
        [C, P, sumblk],
        inplist=['sum.r', 'P.v'], outlist=['P', 'C.u']
    )
    np.testing.assert_equal(clsys5.A, clsys1.A)
    np.testing.assert_equal(clsys5.B, clsys1.B)
    np.testing.assert_equal(clsys5.C, clsys1.C)
    np.testing.assert_equal(clsys5.D, clsys1.D)


def test_interconnect_rewrite():
    sys = ct.rss(
        states=2, name='sys', strictly_proper=True,
        inputs=['u[0]', 'u[1]', 'v[0]', 'v[1]', 'w[0]', 'w[1]'],
        outputs=['y[0]', 'y[1]', 'z[0]', 'z[1]', 'z[2]'])

    # Create an input/output system w/out inplist, outlist
    icsys = ct.interconnect(
        [sys], connections=[['sys.v', 'sys.y']],
        inputs=['u', 'w'],
        outputs=['y', 'z'])

    assert icsys.input_labels == ['u[0]', 'u[1]', 'w[0]', 'w[1]']


def test_interconnect_params():
    # Create a nominally unstable system
    sys1 = ct.nlsys(
        lambda t, x, u, params: params['a'] * x[0] + u[0],
        states=1, inputs='u', outputs='y', params={'a': 2, 'c':2})

    # Simple system for serial interconnection
    sys2 = ct.nlsys(
        None, lambda t, x, u, params: u[0],
        inputs='r', outputs='u', params={'a': 4, 'b': 3})

    # Make sure default parameters get set as expected
    sys = ct.interconnect([sys1, sys2], inputs='r', outputs='y')
    assert sys.params == {'a': 4, 'c': 2, 'b': 3}
    assert sys.dynamics(0, [1], [0]).item() == 4

    # Make sure we can override the parameters
    sys = ct.interconnect(
        [sys1, sys2], inputs='r', outputs='y', params={'b': 1})
    assert sys.params == {'b': 1}
    assert sys.dynamics(0, [1], [0]).item() == 2
    assert sys.dynamics(0, [1], [0], params={'a': 5}).item() == 5

    # Create final series interconnection, with proper parameter values
    sys = ct.interconnect(
        [sys1, sys2], inputs='r', outputs='y', params={'a': 1})
    assert sys.params == {'a': 1}

    # Make sure we can call the update function
    sys.updfcn(0, [0], [0], {})

    # Make sure the serial interconnection is unstable to start
    assert sys.linearize([0], [0]).poles()[0].real == 1

    # Change the parameter and make sure it takes
    assert sys.linearize([0], [0], params={'a': -1}).poles()[0].real == -1

    # Now try running a simulation
    timepts = np.linspace(0, 10)
    resp = ct.input_output_response(sys, timepts, 0, params={'a': -1})
    assert resp.states[0, -1].item() < 2 * math.exp(-10)


# Bug identified in issue #1015
def test_parallel_interconnect():
    sys1 = ct.rss(2, 1, 1, name='S1')
    sys2 = ct.rss(2, 1, 1, name='S2')

    sys_bd = sys1 + sys2
    sys_ic = ct.interconnect(
        [sys1, sys2],
        inplist=[['S1.u[0]', 'S2.u[0]']],
        outlist=[['S1.y[0]', 'S2.y[0]']])
    np.testing.assert_allclose(sys_bd.A, sys_ic.A)
    np.testing.assert_allclose(sys_bd.B, sys_ic.B)
    np.testing.assert_allclose(sys_bd.C, sys_ic.C)
    np.testing.assert_allclose(sys_bd.D, sys_ic.D)
