"""namedio_test.py - test named input/output object operations

RMM, 13 Mar 2022

This test suite checks to make sure that named input/output class
operations are working.  It doesn't do exhaustive testing of
operations on input/output objects.  Separate unit tests should be
created for that purpose.
"""

import re

import numpy as np
import control as ct
import pytest

def test_named_ss():
    # Create a system to play with
    sys = ct.rss(2, 2, 2)
    assert sys.input_list == ['u[0]', 'u[1]']
    assert sys.output_list == ['y[0]', 'y[1]']
    assert sys.state_list == ['x[0]', 'x[1]']

    # Get the state matrices for later use
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    
    # Set up a named state space systems with default names
    ct.namedio._NamedIOObject._idCounter = 0
    sys = ct.ss(A, B, C, D)
    assert sys.name == 'sys[0]'
    assert sys.input_list == ['u[0]', 'u[1]']
    assert sys.output_list == ['y[0]', 'y[1]']
    assert sys.state_list == ['x[0]', 'x[1]']

    # Pass the names as arguments
    sys = ct.ss(
        A, B, C, D, name='system',
        inputs=['u1', 'u2'], outputs=['y1', 'y2'], states=['x1', 'x2'])
    assert sys.name == 'system'
    assert ct.namedio._NamedIOObject._idCounter == 1
    assert sys.input_list == ['u1', 'u2']
    assert sys.output_list == ['y1', 'y2']
    assert sys.state_list == ['x1', 'x2']

    # Do the same with rss
    sys = ct.rss(['x1', 'x2', 'x3'], ['y1', 'y2'], 'u1', name='random')
    assert sys.name == 'random'
    assert ct.namedio._NamedIOObject._idCounter == 1
    assert sys.input_list == ['u1']
    assert sys.output_list == ['y1', 'y2']
    assert sys.state_list == ['x1', 'x2', 'x3']
