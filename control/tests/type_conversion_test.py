# type_conversion_test.py - test type conversions
# RMM, 3 Jan 2021
#
# This set of tests looks at how various classes are converted when using
# algebraic operations.  See GitHub issue #459 for some discussion on what the
# desired combinations should be.

import control as ct
import numpy as np
import operator
import pytest

@pytest.fixture()
def sys_dict():
    sdict = {}
    sdict['ss'] = ct.ss([[-1]], [[1]], [[1]], [[0]])
    sdict['tf'] = ct.tf([1],[0.5, 1])
    sdict['frd'] = ct.frd([10+0j, 9 + 1j, 8 + 2j], [1,2,3])
    sdict['lio'] = ct.LinearIOSystem(ct.ss([[-1]], [[5]], [[5]], [[0]]))
    sdict['ios'] = ct.NonlinearIOSystem(
        sdict['lio']._rhs, sdict['lio']._out, 1, 1, 1)
    sdict['arr'] = np.array([[2.0]])
    sdict['flt'] = 3.
    return sdict

type_dict = {
    'ss': ct.StateSpace, 'tf': ct.TransferFunction,
    'frd': ct.FrequencyResponseData, 'lio': ct.LinearICSystem,
    'ios': ct.InterconnectedSystem, 'arr': np.ndarray, 'flt': float}

#
# Table of expected conversions
#
# This table describes all of the conversions that are supposed to
# happen for various system combinations. This is written out this way
# to make it easy to read, but this is converted below into a list of
# specific tests that can be iterated over.
#
# Items marked as 'E' should generate an exception.
#
# Items starting with 'x' currently generate an expected exception but
# should eventually generate a useful result (when everything is
# implemented properly).
#
# Note 1: some of the entries below are currently converted to to lower level
# types than needed.  In particular, LinearIOSystems should combine with
# StateSpace and TransferFunctions in a way that preserves I/O system
# structure when possible.
#
# Note 2: eventually the operator entry for this table can be pulled out and
# tested as a separate parameterized variable (since all operators should
# return consistent values).

rtype_list =           ['ss',  'tf', 'frd', 'lio', 'ios', 'arr', 'flt']
conversion_table = [
    # op        left     ss     tf    frd    lio    ios    arr    flt
    ('add',     'ss',  ['ss',  'ss',  'xrd', 'ss',  'xos', 'ss',  'ss' ]),
    ('add',     'tf',  ['tf',  'tf',  'xrd', 'tf',  'xos', 'tf',  'tf' ]),
    ('add',     'frd', ['xrd', 'xrd', 'frd', 'xrd', 'E',   'xrd', 'xrd']),
    ('add',     'lio', ['xio', 'xio', 'xrd', 'lio', 'ios', 'xio', 'xio']),
    ('add',     'ios', ['xos', 'xos', 'E',   'ios', 'ios', 'xos', 'xos']),
    ('add',     'arr', ['ss',  'tf',  'xrd', 'xio', 'xos', 'arr', 'arr']),
    ('add',     'flt', ['ss',  'tf',  'xrd', 'xio', 'xos', 'arr', 'flt']),
    
    # op        left     ss     tf    frd    lio    ios    arr    flt
    ('sub',     'ss',  ['ss',  'ss',  'xrd', 'ss',  'xos', 'ss',  'ss' ]),
    ('sub',     'tf',  ['tf',  'tf',  'xrd', 'tf',  'xos', 'tf',  'tf' ]),
    ('sub',     'frd', ['xrd', 'xrd', 'frd', 'xrd', 'E',   'xrd', 'xrd']),
    ('sub',     'lio', ['xio', 'xio', 'xrd', 'lio', 'ios', 'xio', 'xio']),
    ('sub',     'ios', ['xos', 'xio', 'E',   'ios', 'xos'  'xos', 'xos']),
    ('sub',     'arr', ['ss',  'tf',  'xrd', 'xio', 'xos', 'arr', 'arr']),
    ('sub',     'flt', ['ss',  'tf',  'xrd', 'xio', 'xos', 'arr', 'flt']),
    
    # op        left     ss     tf    frd    lio    ios    arr    flt
    ('mul',     'ss',  ['ss',  'ss',  'xrd', 'xio', 'xos', 'ss',  'ss' ]),
    ('mul',     'tf',  ['tf',  'tf',  'xrd', 'tf',  'xos', 'tf',  'tf' ]),
    ('mul',     'frd', ['xrd', 'xrd', 'frd', 'xrd', 'E',   'xrd', 'frd']),
    ('mul',     'lio', ['xio', 'xio', 'xrd', 'lio', 'ios', 'xio', 'xio']),
    ('mul',     'ios', ['xos', 'xos', 'E',   'ios', 'ios', 'xos', 'xos']),
    ('mul',     'arr', ['ss',  'tf',  'xrd', 'xio', 'xos', 'arr', 'arr']),
    ('mul',     'flt', ['ss',  'tf',  'frd', 'xio', 'xos', 'arr', 'flt']),
    
    # op        left     ss     tf    frd    lio    ios    arr    flt
    ('truediv', 'ss',  ['xs',  'tf',  'xrd', 'xio', 'xos', 'xs',  'xs' ]),
    ('truediv', 'tf',  ['tf',  'tf',  'xrd', 'tf',  'xos', 'tf',  'tf' ]),
    ('truediv', 'frd', ['xrd', 'xrd', 'frd', 'xrd', 'E',   'xrd', 'frd']),
    ('truediv', 'lio', ['xio', 'tf',  'xrd', 'xio', 'xio', 'xio', 'xio']),
    ('truediv', 'ios', ['xos', 'xos', 'E',   'xos', 'xos'  'xos', 'xos']),
    ('truediv', 'arr', ['xs',  'tf',  'xrd', 'xio', 'xos', 'arr', 'arr']),
    ('truediv', 'flt', ['xs',  'tf',  'frd', 'xio', 'xos', 'arr', 'flt'])]

# Now create list of the tests we actually want to run
test_matrix = []
for i, (opname, ltype, expected_list) in enumerate(conversion_table):
    for rtype, expected in zip(rtype_list, expected_list):
        # Add this to the list of tests to run
        test_matrix.append([opname, ltype, rtype, expected])
    
@pytest.mark.parametrize("opname, ltype, rtype, expected", test_matrix)
def test_xferfcn_ndarray_precedence(opname, ltype, rtype, expected, sys_dict):
    op = getattr(operator, opname)
    leftsys = sys_dict[ltype]
    rightsys = sys_dict[rtype]

    # Get rid of warnings for InputOutputSystem objects by making a copy
    if isinstance(leftsys, ct.InputOutputSystem) and leftsys == rightsys:
        rightsys = leftsys.copy()
            
    # Make sure we get the right result
    if expected == 'E' or expected[0] == 'x':
        # Exception expected
        with pytest.raises(TypeError):
            op(leftsys, rightsys)
    else:
        # Operation should work and return the given type
        result = op(leftsys, rightsys)
                
        # Print out what we are testing in case something goes wrong
        assert isinstance(result, type_dict[expected])
