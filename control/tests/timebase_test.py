import pytest
import inspect
import numpy as np
import control as ct

# Utility function to convert state space system to nlsys
def ss2io(sys):
    return ct.nlsys(
        sys.updfcn, sys.outfcn, states=sys.nstates,
        inputs=sys.ninputs, outputs=sys.noutputs, dt=sys.dt)

@pytest.mark.parametrize(
    "dt1, dt2, dt3", [
        (0, 0, 0),
        (0, 0.1, ValueError),
        (0, None, 0),
        (0, 'float', 0),
        (0, 'array', 0),
        (None, 'array', None),
        (None, 'array', None),
        (0, True, ValueError),
        (0.1, 0, ValueError),
        (0.1, 0.1, 0.1),
        (0.1, None, 0.1),
        (0.1, True, 0.1),
        (0.1, 'array', 0.1),
        (0.1, 'float', 0.1),
        (None, 0, 0),
        ('float', 0, 0),
        ('array', 0, 0),
        ('float', None, None),
        ('array', None, None),
        (None, 0.1, 0.1),
        ('array', 0.1, 0.1),
        ('float', 0.1, 0.1),
        (None, None, None),
        (None, True, True),
        (True, 0, ValueError),
        (True, 0.1, 0.1),
        (True, None, True),
        (True, True, True),
        (0.2, None, 0.2),
        (0.2, 0.1, ValueError),
     ])
@pytest.mark.parametrize("op", [ct.series, ct.parallel, ct.feedback])
@pytest.mark.parametrize("type", [ct.StateSpace, ct.ss, ct.tf, ss2io])
def test_composition(dt1, dt2, dt3, op, type):
    A, B, C, D = [[1, 1], [0, 1]], [[0], [1]], [[1, 0]], 0
    Karray = np.array([[1]])
    kfloat = 1

    # Define the system
    if isinstance(dt1, (int, float)) or dt1 is None:
        sys1 = ct.StateSpace(A, B, C, D, dt1)
        sys1 = type(sys1)
    elif dt1 == 'array':
        sys1 = Karray
    elif dt1 == 'float':
        sys1 = kfloat

    if isinstance(dt2, (int, float)) or dt2 is None:
        sys2 = ct.StateSpace(A, B, C, D, dt2)
        sys2 = type(sys2)
    elif dt2 == 'array':
        sys2 = Karray
    elif dt2 == 'float':
        sys2 = kfloat

    if inspect.isclass(dt3) and issubclass(dt3, Exception):
        with pytest.raises(dt3, match="incompatible timebases"):
            sys3 = op(sys1, sys2)
    else:
        sys3 = op(sys1, sys2)
        assert sys3.dt == dt3


@pytest.mark.parametrize("dt", [None, 0, 0.1])
def test_composition_override(dt):
    # Define the system
    A, B, C, D = [[1, 1], [0, 1]], [[0], [1]], [[1, 0]], 0
    sys1 = ct.ss(A, B, C, D, None, inputs='u1', outputs='y1')
    sys2 = ct.ss(A, B, C, D, None, inputs='y1', outputs='y2')

    # Show that we can override the type
    sys3 = ct.interconnect([sys1, sys2], inputs='u1', outputs='y2', dt=dt)
    assert sys3.dt == dt

    # Overriding the type with an inconsistent type generates an error
    sys1 = ct.StateSpace(A, B, C, D, 0.1, inputs='u1', outputs='y1')
    if dt != 0.1 and dt is not None:
        with pytest.raises(ValueError, match="incompatible timebases"):
            sys3 = ct.interconnect(
                [sys1, sys2], inputs='u1', outputs='y2', dt=dt)

    sys1 = ct.StateSpace(A, B, C, D, 0, inputs='u1', outputs='y1')
    if dt != 0 and dt is not None:
        with pytest.raises(ValueError, match="incompatible timebases"):
            sys3 = ct.interconnect(
                [sys1, sys2], inputs='u1', outputs='y2', dt=dt)


# Make sure all system creation functions treat timebases uniformly
@pytest.mark.parametrize(
    "fcn, args", [
        (ct.ss, [-1, 1, 1, 1]),
        (ct.tf, [[1, 2], [3, 4, 5]]),
        (ct.zpk, [[-1], [-2, -3], 1]),
        (ct.frd, [[1, 1, 1], [1, 2, 3]]),
        (ct.nlsys, [lambda t, x, u, params: -x, None]),
    ])
@pytest.mark.parametrize(
    "kwargs, expected", [
        ({}, 0),
        ({'dt': 0}, 0),
        ({'dt': 0.1}, 0.1),
        ({'dt': True}, True),
        ({'dt': None}, None),
    ])
def test_default(fcn, args, kwargs, expected):
    sys = fcn(*args, **kwargs)
    assert sys.dt == expected

    # Some commands allow dt via extra argument
    if fcn in [ct.ss, ct.tf, ct.zpk, ct.frd] and kwargs.get('dt'):
        sys = fcn(*args, kwargs['dt'])
        assert sys.dt == expected

        # Make sure an error is generated if dt is redundant
        with pytest.warns(UserWarning, match="received multiple dt"):
            sys = fcn(*args, kwargs['dt'], **kwargs)
