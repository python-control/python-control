"""namedio_test.py - test named input/output object operations

RMM, 13 Mar 2022

This test suite checks to make sure that named input/output class
operations are working.  It doesn't do exhaustive testing of
operations on input/output objects.  Separate unit tests should be
created for that purpose.
"""

import re
from copy import copy
import warnings

import numpy as np
import control as ct
import pytest


def test_named_ss():
    # Create a system to play with
    sys = ct.rss(2, 2, 2)
    assert sys.input_labels == ['u[0]', 'u[1]']
    assert sys.output_labels == ['y[0]', 'y[1]']
    assert sys.state_labels == ['x[0]', 'x[1]']

    # Get the state matrices for later use
    A, B, C, D = sys.A, sys.B, sys.C, sys.D

    # Set up a named state space systems with default names
    ct.namedio.NamedIOSystem._idCounter = 0
    sys = ct.ss(A, B, C, D)
    assert sys.name == 'sys[0]'
    assert sys.input_labels == ['u[0]', 'u[1]']
    assert sys.output_labels == ['y[0]', 'y[1]']
    assert sys.state_labels == ['x[0]', 'x[1]']
    assert repr(sys) == \
        "<LinearIOSystem:sys[0]:['u[0]', 'u[1]']->['y[0]', 'y[1]']>"

    # Pass the names as arguments
    sys = ct.ss(
        A, B, C, D, name='system',
        inputs=['u1', 'u2'], outputs=['y1', 'y2'], states=['x1', 'x2'])
    assert sys.name == 'system'
    assert ct.namedio.NamedIOSystem._idCounter == 1
    assert sys.input_labels == ['u1', 'u2']
    assert sys.output_labels == ['y1', 'y2']
    assert sys.state_labels == ['x1', 'x2']
    assert repr(sys) == \
        "<LinearIOSystem:system:['u1', 'u2']->['y1', 'y2']>"

    # Do the same with rss
    sys = ct.rss(['x1', 'x2', 'x3'], ['y1', 'y2'], 'u1', name='random')
    assert sys.name == 'random'
    assert ct.namedio.NamedIOSystem._idCounter == 1
    assert sys.input_labels == ['u1']
    assert sys.output_labels == ['y1', 'y2']
    assert sys.state_labels == ['x1', 'x2', 'x3']
    assert repr(sys) == \
        "<LinearIOSystem:random:['u1']->['y1', 'y2']>"


# List of classes that are expected
fun_instance = {
    ct.rss: (ct.InputOutputSystem, ct.LinearIOSystem, ct.StateSpace),
    ct.drss: (ct.InputOutputSystem, ct.LinearIOSystem, ct.StateSpace),
    ct.FRD: (ct.lti.LTI),
    ct.NonlinearIOSystem: (ct.InputOutputSystem),
    ct.ss: (ct.InputOutputSystem, ct.LinearIOSystem, ct.StateSpace),
    ct.StateSpace: (ct.StateSpace),
    ct.tf: (ct.TransferFunction),
    ct.TransferFunction: (ct.TransferFunction),
}

# List of classes that are not expected
fun_notinstance = {
    ct.FRD: (ct.InputOutputSystem, ct.LinearIOSystem, ct.StateSpace),
    ct.StateSpace: (ct.InputOutputSystem, ct.TransferFunction),
    ct.TransferFunction: (ct.InputOutputSystem, ct.StateSpace),
}


@pytest.mark.parametrize("fun, args, kwargs", [
    [ct.rss, (4, 1, 1), {}],
    [ct.rss, (3, 2, 1), {}],
    [ct.drss, (4, 1, 1), {}],
    [ct.drss, (3, 2, 1), {}],
    [ct.FRD, ([1, 2, 3,], [1, 2, 3]), {}],
    [ct.NonlinearIOSystem,
     (lambda t, x, u, params: -x, None),
     {'inputs': 2, 'outputs':2, 'states':2}],
    [ct.ss, ([[1, 2], [3, 4]], [[0], [1]], [[1, 0]], 0), {}],
    [ct.ss, ([], [], [], 3), {}], # static system
    [ct.StateSpace, ([[1, 2], [3, 4]], [[0], [1]], [[1, 0]], 0), {}],
    [ct.tf, ([1, 2], [3, 4, 5]), {}],
    [ct.tf, (2, 3), {}], # static system
    [ct.TransferFunction, ([1, 2], [3, 4, 5]), {}],
])
def test_io_naming(fun, args, kwargs):
    # Reset the ID counter to get uniform generic names
    ct.namedio.NamedIOSystem._idCounter = 0

    # Create the system w/out any names
    sys_g = fun(*args, **kwargs)

    # Make sure the class are what we expect
    if fun in fun_instance:
        assert isinstance(sys_g, fun_instance[fun])

    if fun in fun_notinstance:
        assert not isinstance(sys_g, fun_notinstance[fun])

    # Make sure the names make sense
    assert sys_g.name == 'sys[0]'
    assert sys_g.input_labels == [f'u[{i}]' for i in range(sys_g.ninputs)]
    assert sys_g.output_labels == [f'y[{i}]' for i in range(sys_g.noutputs)]
    if sys_g.nstates is not None:
        assert sys_g.state_labels == [f'x[{i}]' for i in range(sys_g.nstates)]

    #
    # Reset the names to something else and make sure they stick
    #
    sys_r = copy(sys_g)

    input_labels = [f'u{i}' for i in range(sys_g.ninputs)]
    sys_r.set_inputs(input_labels)
    assert sys_r.input_labels == input_labels

    output_labels = [f'y{i}' for i in range(sys_g.noutputs)]
    sys_r.set_outputs(output_labels)
    assert sys_r.output_labels == output_labels

    if sys_g.nstates is not None:
        state_labels = [f'x{i}' for i in range(sys_g.nstates)]
        sys_r.set_states(state_labels)
        assert sys_r.state_labels == state_labels

    sys_r.name = 'sys'          # make sure name is non-generic

    #
    # Set names using keywords and make sure they stick
    #

    # How the keywords are used depends on the type of system
    if fun in (ct.rss, ct.drss):
        # Pass the labels instead of the numbers
        sys_k = fun(state_labels, output_labels, input_labels, name='mysys')

    elif sys_g.nstates is None:
        # Don't pass state labels if TransferFunction
        sys_k = fun(
            *args, inputs=input_labels, outputs=output_labels, name='mysys')

    else:
        sys_k = fun(
            *args, inputs=input_labels, outputs=output_labels,
            states=state_labels, name='mysys')

    assert sys_k.name == 'mysys'
    assert sys_k.input_labels == input_labels
    assert sys_k.output_labels == output_labels
    if sys_g.nstates is not None:
        assert sys_k.state_labels == state_labels

    #
    # Convert the system to state space and make sure labels transfer
    #
    if ct.slycot_check() and not isinstance(
            sys_r, (ct.FrequencyResponseData, ct.NonlinearIOSystem)):
        sys_ss = ct.ss(sys_r)
        assert sys_ss != sys_r
        assert sys_ss.input_labels == input_labels
        assert sys_ss.output_labels == output_labels
        if not isinstance(sys_r, ct.StateSpace):
            # System should get unique name
            assert sys_ss.name != sys_r.name

        # Reassign system and signal names
        sys_ss = ct.ss(
            sys_g, inputs=input_labels, outputs=output_labels, name='new')
        assert sys_ss.name == 'new'
        assert sys_ss.input_labels == input_labels
        assert sys_ss.output_labels == output_labels

    #
    # Convert the system to a transfer function and make sure labels transfer
    #
    if not isinstance(
            sys_r, (ct.FrequencyResponseData, ct.NonlinearIOSystem)) and \
       ct.slycot_check():
        sys_tf = ct.tf(sys_r)
        assert sys_tf != sys_r
        assert sys_tf.input_labels == input_labels
        assert sys_tf.output_labels == output_labels

        # Reassign system and signal names
        sys_tf = ct.tf(
            sys_g, inputs=input_labels, outputs=output_labels, name='new')
        assert sys_tf.name == 'new'
        assert sys_tf.input_labels == input_labels
        assert sys_tf.output_labels == output_labels

    #
    # Convert the system to a LinearIOSystem and make sure labels transfer
    #
    if not isinstance(
            sys_r, (ct.FrequencyResponseData, ct.NonlinearIOSystem)) and \
                    ct.slycot_check():
        sys_lio = ct.LinearIOSystem(sys_r)
        assert sys_lio != sys_r
        assert sys_lio.input_labels == input_labels
        assert sys_lio.output_labels == output_labels

        # Reassign system and signal names
        sys_lio = ct.LinearIOSystem(
            sys_g, inputs=input_labels, outputs=output_labels, name='new')
        assert sys_lio.name == 'new'
        assert sys_lio.input_labels == input_labels
        assert sys_lio.output_labels == output_labels


# Internal testing of StateSpace initialization
def test_init_namedif():
    # Set up the initial system
    sys = ct.rss(2, 1, 1)

    # Rename the system, inputs, and outouts
    sys_new = sys.copy()
    ct.StateSpace.__init__(
        sys_new, sys, inputs='u', outputs='y', name='new')
    assert sys_new.name == 'new'
    assert sys_new.input_labels == ['u']
    assert sys_new.output_labels == ['y']

    # Call constructor without re-initialization
    sys_keep = sys.copy()
    ct.StateSpace.__init__(sys_keep, sys, init_namedio=False)
    assert sys_keep.name == sys_keep.name
    assert sys_keep.input_labels == sys_keep.input_labels
    assert sys_keep.output_labels == sys_keep.output_labels

    # Make sure that passing an unrecognized keyword generates an error
    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.StateSpace.__init__(
            sys_keep, sys, inputs='u', outputs='y', init_namedio=False)

# Test state space conversion
def test_convert_to_statespace():
    # Set up the initial systems
    sys = ct.tf(ct.rss(2, 1, 1), inputs='u', outputs='y', name='sys')
    sys_static = ct.tf(1, 2, inputs='u', outputs='y', name='sys_static')

    # check that name, inputs, and outputs passed through
    sys_new = ct.ss(sys)
    assert sys_new.name == 'sys$converted'
    assert sys_new.input_labels == ['u']
    assert sys_new.output_labels == ['y']
    sys_new = ct.ss(sys_static)
    assert sys_new.name == 'sys_static$converted'
    assert sys_new.input_labels == ['u']
    assert sys_new.output_labels == ['y']

    # Make sure we can rename system name, inputs, outputs
    sys_new = ct.ss(sys, inputs='u', outputs='y', name='new')
    assert sys_new.name == 'new'
    assert sys_new.input_labels == ['u']
    assert sys_new.output_labels == ['y']
    sys_new = ct.ss(sys_static, inputs='u', outputs='y', name='new')
    assert sys_new.name == 'new'
    assert sys_new.input_labels == ['u']
    assert sys_new.output_labels == ['y']

    # Try specifying the state names (via low level test)
    with pytest.warns(UserWarning, match="non-unique state space realization"):
        sys_new = ct.ss(sys, inputs='u', outputs='y', states=['x1', 'x2'])
        assert sys_new.input_labels == ['u']
        assert sys_new.output_labels == ['y']
        assert sys_new.state_labels == ['x1', 'x2']


# Duplicate name warnings
def test_duplicate_sysname():
    # Start with an unnamed system
    sys = ct.rss(4, 1, 1)

    # No warnings should be generated if we reuse an an unnamed system
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # strip out matrix warnings
        warnings.filterwarnings("ignore", "the matrix subclass",
                                category=PendingDeprecationWarning)
        res = sys * sys

    # Generate a warning if the system is named
    sys = ct.rss(4, 1, 1, name='sys')
    with pytest.warns(UserWarning, match="duplicate object found"):
        res = sys * sys
