# kwargs_test.py - test for uncrecognized keywords
# RMM, 20 Mar 2022
#
# Allowing unrecognized keywords to be passed to a function without
# generating an error message can generate annoying bugs, since you
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
import control.tests.optimal_test as optimal_test
import control.tests.statefbk_test as statefbk_test
import control.tests.stochsys_test as stochsys_test
import control.tests.trdata_test as trdata_test

@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys."), (control.optimal, "optimal.")
])
def test_kwarg_search(module, prefix):
    # Look through every object in the package
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue

        # Look for classes and then check member functions
        if inspect.isclass(obj):
            test_kwarg_search(obj, prefix + obj.__name__ + '.')

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


@pytest.mark.parametrize(
    "function, nsssys, ntfsys, moreargs, kwargs",
    [(control.dlqe, 1, 0, ([[1]], [[1]]), {}),
     (control.dlqr, 1, 0, ([[1, 0], [0, 1]], [[1]]), {}),
     (control.drss, 0, 0, (2, 1, 1), {}),
     (control.input_output_response, 1, 0, ([0, 1, 2], [1, 1, 1]), {}),
     (control.lqe, 1, 0, ([[1]], [[1]]), {}),
     (control.lqr, 1, 0, ([[1, 0], [0, 1]], [[1]]), {}),
     (control.linearize, 1, 0, (0, 0), {}),
     (control.pzmap, 1, 0, (), {}),
     (control.rlocus, 0, 1, (), {}),
     (control.root_locus, 0, 1, (), {}),
     (control.rss, 0, 0, (2, 1, 1), {}),
     (control.set_defaults, 0, 0, ('control',), {'default_dt': True}),
     (control.ss, 0, 0, (0, 0, 0, 0), {'dt': 1}),
     (control.ss2io, 1, 0,  (), {}),
     (control.ss2tf, 1, 0, (), {}),
     (control.summing_junction, 0, 0, (2,), {}),
     (control.tf, 0, 0, ([1], [1, 1]), {}),
     (control.tf2io, 0, 1, (), {}),
     (control.tf2ss, 0, 1, (), {}),
     (control.zpk, 0, 0, ([1], [2, 3], 4), {}),
     (control.InputOutputSystem, 0, 0, (),
      {'inputs': 1, 'outputs': 1, 'states': 1}),
     (control.InputOutputSystem.linearize, 1, 0, (0, 0), {}),
     (control.LinearIOSystem.sample, 1, 0, (0.1,), {}),
     (control.StateSpace, 0, 0,
      ([[-1, 0], [0, -1]], [[1], [1]], [[1, 1]], 0), {}),
     (control.TransferFunction, 0, 0, ([1], [1, 1]), {})]
)
def test_unrecognized_kwargs(function, nsssys, ntfsys, moreargs, kwargs,
                             mplcleanup, editsdefaults):
    # Create SISO systems for use in parameterized tests
    sssys = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)
    tfsys = control.tf([1], [1, 1])

    args = (sssys, )*nsssys + (tfsys, )*ntfsys + moreargs

    # Call the function normally and make sure it works
    function(*args, **kwargs)

    # Now add an unrecognized keyword and make sure there is an error
    with pytest.raises(TypeError, match="unrecognized keyword"):
        function(*args, **kwargs, unknown=None)


@pytest.mark.parametrize(
    "function, nsysargs, moreargs, kwargs",
    [(control.bode, 1, (), {}),
     (control.bode_plot, 1, (), {}),
     (control.describing_function_plot, 1,
      (control.descfcn.saturation_nonlinearity(1), [1, 2, 3, 4]), {}),
     (control.gangof4, 2, (), {}),
     (control.gangof4_plot, 2, (), {}),
     (control.nyquist, 1, (), {}),
     (control.nyquist_plot, 1, (), {}),
     (control.singular_values_plot, 1, (), {})]
)
def test_matplotlib_kwargs(function, nsysargs, moreargs, kwargs, mplcleanup):
    # Create a SISO system for use in parameterized tests
    sys = control.ss([[-1, 1], [0, -1]], [[0], [1]], [[1, 0]], 0, dt=None)

    # Call the function normally and make sure it works
    args = (sys, )*nsysargs + moreargs
    function(*args, **kwargs)

    # Now add an unrecognized keyword and make sure there is an error
    with pytest.raises(AttributeError,
                       match="(has no property|unexpected keyword)"):
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
    'create_estimator_iosystem': stochsys_test.test_estimator_errors,
    'create_statefbk_iosystem': statefbk_test.TestStatefbk.test_statefbk_errors,
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
    'sample_system' : test_unrecognized_kwargs,
    'c2d' : test_unrecognized_kwargs,
    'zpk': test_unrecognized_kwargs,
    'flatsys.point_to_point':
        flatsys_test.TestFlatSys.test_point_to_point_errors,
    'flatsys.solve_flat_ocp':
        flatsys_test.TestFlatSys.test_solve_flat_ocp_errors,
    'optimal.create_mpc_iosystem': optimal_test.test_mpc_iosystem_rename,
    'optimal.solve_ocp': optimal_test.test_ocp_argument_errors,
    'optimal.solve_oep': optimal_test.test_oep_argument_errors,
    'FrequencyResponseData.__init__':
        frd_test.TestFRD.test_unrecognized_keyword,
    'InputOutputSystem.__init__': test_unrecognized_kwargs,
    'InputOutputSystem.linearize': test_unrecognized_kwargs,
    'InterconnectedSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'LinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'LinearIOSystem.sample': test_unrecognized_kwargs,
    'NonlinearIOSystem.__init__':
        interconnect_test.test_interconnect_exceptions,
    'StateSpace.__init__': test_unrecognized_kwargs,
    'StateSpace.sample': test_unrecognized_kwargs,
    'TimeResponseData.__call__': trdata_test.test_response_copy,
    'TransferFunction.__init__': test_unrecognized_kwargs,
    'TransferFunction.sample': test_unrecognized_kwargs,
    'optimal.OptimalControlProblem.__init__':
        optimal_test.test_ocp_argument_errors,
    'optimal.OptimalControlProblem.compute_trajectory':
        optimal_test.test_ocp_argument_errors,
    'optimal.OptimalControlProblem.create_mpc_iosystem':
        optimal_test.test_ocp_argument_errors,
    'optimal.OptimalEstimationProblem.__init__':
        optimal_test.test_oep_argument_errors,
    'optimal.OptimalEstimationProblem.create_mhe_iosystem':
        optimal_test.test_oep_argument_errors,
}

#
# Look for keywords with mutable defaults
#
# This test goes through every function and looks for signatures that have a
# default value for a keyword that is mutable.  An error is generated unless
# the function is listed in the `mutable_ok` set (which should only be used
# for cases were the code has been explicitly checked to make sure that the
# value of the mutable is not modified in the code).
#
mutable_ok = {                                          # initial and date
    control.flatsys.SystemTrajectory.__init__,          # RMM, 18 Nov 2022
    control.freqplot._add_arrows_to_line2D,             # RMM, 18 Nov 2022
    control.namedio._process_dt_keyword,                # RMM, 13 Nov 2022
    control.namedio._process_namedio_keywords,          # RMM, 18 Nov 2022
}

@pytest.mark.parametrize("module", [control, control.flatsys])
def test_mutable_defaults(module, recurse=True):
    # Look through every object in the package
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue

        # Look for classes and then check member functions
        if inspect.isclass(obj):
            test_mutable_defaults(obj, True)

        # Look for modules and check for internal functions (w/ no recursion)
        if inspect.ismodule(obj) and recurse:
            test_mutable_defaults(obj, False)

        # Only look at functions and skip any that are marked as OK
        if not inspect.isfunction(obj) or obj in mutable_ok:
            continue

        # Get the signature for the function
        sig = inspect.signature(obj)

        # Skip anything that is inherited
        if inspect.isclass(module) and obj.__name__ not in module.__dict__:
            continue

        # See if there is a variable keyword argument
        for argname, par in sig.parameters.items():
            if par.default is inspect._empty or \
               not par.kind == inspect.Parameter.KEYWORD_ONLY and \
               not par.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                continue

            # Check to see if the default value is mutable
            if par.default is not None and not \
               isinstance(par.default, (bool, int, float, tuple, str)):
                pytest.fail(
                    f"function '{obj.__name__}' in module '{module.__name__}'"
                    f" has mutable default for keyword '{par.name}'")

