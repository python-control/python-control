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
import matplotlib.pyplot as plt

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

        # Only look for functions with keyword arguments
        if not inspect.isfunction(obj):
            continue

        # Get the signature for the function
        sig = inspect.signature(obj)

        # Skip anything that is inherited
        if inspect.isclass(module) and obj.__name__ not in module.__dict__:
            continue

        # See if there is a variable keyword argument
        for argname, par in sig.parameters.items():
            if not par.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Make sure there is a unit test defined
            assert prefix + name in kwarg_unittest

            # Make sure there is a unit test
            if not hasattr(kwarg_unittest[prefix + name], '__call__'):
                warnings.warn("No unit test defined for '%s'" % prefix + name)
                source = None
            else:
                source = inspect.getsource(kwarg_unittest[prefix + name])

            # Make sure the unit test looks for unrecognized keyword
            if source and source.find('unrecognized keyword') < 0:
                warnings.warn(
                    f"'unrecognized keyword' not found in unit test "
                    f"for {name}")

        # Look for classes and then check member functions
        if inspect.isclass(obj):
            test_kwarg_search(obj, prefix + obj.__name__ + '.')


@pytest.mark.usefixtures('editsdefaults')
def test_unrecognized_kwargs():
    # Create a SISO system for use in parameterized tests
    sys = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)

    table = [
        [control.dlqe, (sys, [[1]], [[1]]), {}],
        [control.dlqr, (sys, [[1, 0], [0, 1]], [[1]]), {}],
        [control.drss, (2, 1, 1), {}],
        [control.input_output_response, (sys, [0, 1, 2], [1, 1, 1]), {}],
        [control.lqe, (sys, [[1]], [[1]]), {}],
        [control.lqr, (sys, [[1, 0], [0, 1]], [[1]]), {}],
        [control.linearize, (sys, 0, 0), {}],
        [control.pzmap, (sys,), {}],
        [control.rlocus, (control.tf([1], [1, 1]), ), {}],
        [control.root_locus, (control.tf([1], [1, 1]), ), {}],
        [control.rss, (2, 1, 1), {}],
        [control.set_defaults, ('control',), {'default_dt': True}],
        [control.ss, (0, 0, 0, 0), {'dt': 1}],
        [control.ss2io, (sys,), {}],
        [control.ss2tf, (sys,), {}],
        [control.summing_junction, (2,), {}],
        [control.tf, ([1], [1, 1]), {}],
        [control.tf2io, (control.tf([1], [1, 1]),), {}],
        [control.tf2ss, (control.tf([1], [1, 1]),), {}],
        [control.InputOutputSystem, (),
         {'inputs': 1, 'outputs': 1, 'states': 1}],
        [control.InputOutputSystem.linearize, (sys, 0, 0), {}],
        [control.StateSpace, ([[-1, 0], [0, -1]], [[1], [1]], [[1, 1]], 0), {}],
        [control.TransferFunction, ([1], [1, 1]), {}],
    ]

    for function, args, kwargs in table:
        # Call the function normally and make sure it works
        function(*args, **kwargs)

        # Now add an unrecognized keyword and make sure there is an error
        with pytest.raises(TypeError, match="unrecognized keyword"):
            function(*args, **kwargs, unknown=None)

        # If we opened any figures, close them to avoid matplotlib warnings
        if plt.gca():
            plt.close('all')


def test_matplotlib_kwargs():
    # Create a SISO system for use in parameterized tests
    sys = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)
    ctl = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)

    table = [
        [control.bode, (sys, ), {}],
        [control.bode_plot, (sys, ), {}],
        [control.describing_function_plot,
         (sys, control.descfcn.saturation_nonlinearity(1), [1, 2, 3, 4]), {}],
        [control.gangof4, (sys, ctl), {}],
        [control.gangof4_plot, (sys, ctl), {}],
        [control.nyquist, (sys, ), {}],
        [control.nyquist_plot, (sys, ), {}],
        [control.singular_values_plot, (sys, ), {}],
    ]

    for function, args, kwargs in table:
        # Call the function normally and make sure it works
        function(*args, **kwargs)

        # Now add an unrecognized keyword and make sure there is an error
        with pytest.raises(AttributeError, match="has no property"):
            function(*args, **kwargs, unknown=None)

        # If we opened any figures, close them to avoid matplotlib warnings
        if plt.gca():
            plt.close('all')


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
    'describing_function_plot': test_matplotlib_kwargs,
    'dlqe': test_unrecognized_kwargs,
    'dlqr': test_unrecognized_kwargs,
    'drss': test_unrecognized_kwargs,
    'gangof4': test_matplotlib_kwargs,
    'gangof4_plot': test_matplotlib_kwargs,
    'input_output_response': test_unrecognized_kwargs,
    'interconnect': interconnect_test.test_interconnect_exceptions,
    'linearize': test_unrecognized_kwargs,
    'lqe': test_unrecognized_kwargs,
    'lqr': test_unrecognized_kwargs,
    'nyquist': test_matplotlib_kwargs,
    'nyquist_plot': test_matplotlib_kwargs,
    'pzmap': test_unrecognized_kwargs,
    'rlocus': test_unrecognized_kwargs,
    'root_locus': test_unrecognized_kwargs,
    'rss': test_unrecognized_kwargs,
    'set_defaults': test_unrecognized_kwargs,
    'singular_values_plot': test_matplotlib_kwargs,
    'ss': test_unrecognized_kwargs,
    'ss2io': test_unrecognized_kwargs,
    'ss2tf': test_unrecognized_kwargs,
    'summing_junction': interconnect_test.test_interconnect_exceptions,
    'tf': test_unrecognized_kwargs,
    'tf2io' : test_unrecognized_kwargs,
    'tf2ss' : test_unrecognized_kwargs,
    'flatsys.point_to_point':
        flatsys_test.TestFlatSys.test_point_to_point_errors,
    'FrequencyResponseData.__init__':
        frd_test.TestFRD.test_unrecognized_keyword,
    'InputOutputSystem.__init__': test_unrecognized_kwargs,
    'InputOutputSystem.linearize': test_unrecognized_kwargs,
    'InterconnectedSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'LinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'NonlinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'StateSpace.__init__': test_unrecognized_kwargs,
    'TimeResponseData.__call__': trdata_test.test_response_copy,
    'TransferFunction.__init__': test_unrecognized_kwargs,
}
