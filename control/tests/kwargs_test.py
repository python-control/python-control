# kwargs_test.py - test for uncrecognized keywords
# RMM, 20 Mar 2022
#
# Allowing unrecognized keywords to be passed to a function without
# generating and error message can generate annoying bugs, since you
# sometimes think you are telling the function to do something and actually
# you have a misspelling or other error and your input is being ignored.
#
# This unit test looks through all functions in the package for any that
# allow kwargs as part of the function signature and makes sure that there
# is a unit test that checks for unrecognized keywords.

import inspect
import pytest
import warnings

import control
import control.flatsys

# List of all of the test modules where kwarg unit tests are defined
import control.tests.flatsys_test as flatsys_test
import control.tests.frd_test as frd_test
import control.tests.interconnect_test as interconnect_test
import control.tests.statefbk_test as statefbk_test
import control.tests.trdata_test as trdata_test


@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys.")
])
def test_kwarg_search(module, prefix):
    # Look through every object in the package
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue
        
        # Look for functions with keyword arguments
        if inspect.isfunction(obj):
            # Get the signature for the function
            sig = inspect.signature(obj)

            # See if there is a variable keyword argument
            for argname, par in sig.parameters.items():
                if par.kind == inspect.Parameter.VAR_KEYWORD:
                    # Make sure there is a unit test defined
                    assert prefix + name in kwarg_unittest

                    # Make sure there is a unit test
                    if not hasattr(kwarg_unittest[prefix + name], '__call__'):
                        warnings.warn("No unit test defined for '%s'"
                                      % prefix + name)

        # Look for classes and then check member functions
        if inspect.isclass(obj):
            test_kwarg_search(obj, prefix + obj.__name__ + '.')


# Create a SISO system for use in parameterized tests
sys = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)


# Parameterized tests for looking for unrecognized keyword errors
@pytest.mark.parametrize("function, args, kwargs", [
    [control.dlqr, (sys, [[1, 0], [0, 1]], [[1]]), {}],
    [control.drss, (2, 1, 1), {}],
    [control.input_output_response, (sys, [0, 1, 2], [1, 1, 1]), {}],
    [control.lqr, (sys, [[1, 0], [0, 1]], [[1]]), {}],
    [control.pzmap, (sys,), {}],
    [control.rlocus, (control.tf([1], [1, 1]), ), {}],
    [control.root_locus, (control.tf([1], [1, 1]), ), {}],
    [control.rss, (2, 1, 1), {}],
    [control.ss, (0, 0, 0, 0), {'dt': 1}],
    [control.ss2io, (sys,), {}],
    [control.summing_junction, (2,), {}],
    [control.tf, ([1], [1, 1]), {}],
    [control.tf2io, (control.tf([1], [1, 1]),), {}],
    [control.InputOutputSystem, (1, 1, 1), {}],
    [control.StateSpace, ([[-1, 0], [0, -1]], [[1], [1]], [[1, 1]], 0), {}],
    [control.TransferFunction, ([1], [1, 1]), {}],
])
def test_unrecognized_kwargs(function, args, kwargs):
    # Call the function normally and make sure it works
    function(*args, **kwargs)

    # Now add an unrecognized keyword and make sure there is an error
    with pytest.raises(TypeError, match="unrecognized keyword"):
        function(*args, **kwargs, unknown=None)


# Parameterized tests for looking for keyword errors handled by matplotlib
@pytest.mark.parametrize("function, args, kwargs", [
    [control.bode, (sys, ), {}],
    [control.bode_plot, (sys, ), {}],
    [control.gangof4, (sys, sys), {}],
    [control.gangof4_plot, (sys, sys), {}],
    [control.nyquist, (sys, ), {}],
    [control.nyquist_plot, (sys, ), {}],
])
def test_matplotlib_kwargs(function, args, kwargs):
    # Call the function normally and make sure it works
    function(*args, **kwargs)

    # Now add an unrecognized keyword and make sure there is an error
    with pytest.raises(AttributeError, match="has no property"):
        function(*args, **kwargs, unknown=None)
    

#
# List of all unit tests that check for unrecognized keywords
#
# Every function that accepts variable keyword arguments (**kwargs) should
# have an entry in this table, to make sure that nothing is missing.  This
# will also force people who add new functions to put in an appropriate unit
# test.
#

kwarg_unittest = {
    'bode': test_matplotlib_kwargs,
    'bode_plot': test_matplotlib_kwargs,
    'describing_function_plot': None,
    'dlqr': statefbk_test.TestStatefbk.test_lqr_errors,
    'drss': test_unrecognized_kwargs,
    'find_eqpt': None,
    'gangof4': test_matplotlib_kwargs,
    'gangof4_plot': test_matplotlib_kwargs,
    'input_output_response': test_unrecognized_kwargs,
    'interconnect': interconnect_test.test_interconnect_exceptions,
    'linearize': None,
    'lqr': statefbk_test.TestStatefbk.test_lqr_errors,
    'nyquist': test_matplotlib_kwargs,
    'nyquist_plot': test_matplotlib_kwargs,
    'pzmap': None,
    'rlocus': test_unrecognized_kwargs,
    'root_locus': test_unrecognized_kwargs,
    'rss': test_unrecognized_kwargs,
    'set_defaults': None,
    'singular_values_plot': None,
    'ss': test_unrecognized_kwargs,
    'ss2io': test_unrecognized_kwargs,
    'ss2tf': test_unrecognized_kwargs,
    'summing_junction': interconnect_test.test_interconnect_exceptions,
    'tf': test_unrecognized_kwargs,
    'tf2io' : test_unrecognized_kwargs,
    'flatsys.point_to_point':
        flatsys_test.TestFlatSys.test_point_to_point_errors,
    'FrequencyResponseData.__init__':
        frd_test.TestFRD.test_unrecognized_keyword,
    'InputOutputSystem.__init__': None,
    'InputOutputSystem.linearize': None,
    'InterconnectedSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'InterconnectedSystem.linearize': None,
    'LinearICSystem.linearize': None,
    'LinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'LinearIOSystem.linearize': None,
    'NonlinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'NonlinearIOSystem.linearize': None,
    'StateSpace.__init__': None,
    'TimeResponseData.__call__': trdata_test.test_response_copy,
    'TransferFunction.__init__': None,
    'flatsys.FlatSystem.linearize': None,
    'flatsys.LinearFlatSystem.linearize': None,
}
