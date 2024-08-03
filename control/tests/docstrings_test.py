# docstrings_test.py - test for undocumented arguments
# RMM, 28 Jul 2024
#
# This unit test looks through all functions in the package and attempts to
# identify arguments that are not documented.  It will check anything that
# is an explicitly listed argument, as well as attempt to find keyword
# arguments that are extracted using kwargs.pop(), config._get_param(), or
# config.use_legacy_defaults.

import inspect
import warnings

import pytest
import re

import control
import control.flatsys

# List of functions that we can skip testing (special cases)
function_skiplist = [
    control.ControlPlot.reshape,                # needed for legacy interface
    control.phase_plot,                         # legacy function
]

# List of keywords that we can skip testing (special cases)
keyword_skiplist = {
    control.input_output_response: ['method'],
    control.nyquist_plot: ['color'],            # checked separately
    control.optimal.solve_ocp: ['method'],      # deprecated
    control.sisotool: ['kvect'],                # deprecated
}

# Decide on the level of verbosity (use -rP when running pytest)
verbose = 1

@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys."),
    (control.optimal, "optimal."), (control.phaseplot, "phaseplot.")
])
def test_docstrings(module, prefix):
    # Look through every object in the package
    if verbose > 1:
        print(f"Checking module {module}")
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and (
                not inspect.getmodule(obj).__name__.startswith('control')
                or prefix != "" and inspect.getmodule(obj) != module):
            # Skip anything that isn't part of the control package
            continue

        if inspect.isclass(obj):
            if verbose > 1:
                print(f"  Checking class {name}")
            # Check member functions within the class
            test_docstrings(obj, prefix + name + '.')

        if inspect.isfunction(obj):
            # Skip anything that is inherited, hidden, or deprecated
            if inspect.isclass(module) and name not in module.__dict__ \
               or name.startswith('_') or obj in function_skiplist:
                continue

            # Get the docstring (skip w/ warning if there isn't one)
            if verbose > 1:
                print(f"  Checking function {name}")
            if obj.__doc__ is None:
                warnings.warn(
                    f"{module.__name__}.{name} is missing docstring")
                continue
            else:
                docstring = inspect.getdoc(obj)
                source = inspect.getsource(obj)

            # Skip deprecated functions
            if f"{name} is deprecated" in docstring or \
               "function is deprecated" in docstring or \
               ".. deprecated::" in docstring:
                if verbose > 1:
                    print("    [deprecated]")
                continue

            elif f"{name} is deprecated" in source:
                if verbose:
                    print(f"    {name} is deprecated, but not documented")
                warnings.warn(f"{name} deprecated, but not documented")
                continue

            # Get the signature for the function
            sig = inspect.signature(obj)

            # Go through each parameter and make sure it is in the docstring
            for argname, par in sig.parameters.items():

                # Look for arguments that we can skip
                if argname == 'self' or argname[0] == '_' or \
                   obj in keyword_skiplist and argname in keyword_skiplist[obj]:
                    continue

                # Check for positional arguments
                if par.kind == inspect.Parameter.VAR_POSITIONAL:
                    # Too complicated to check
                    if f"*{argname}" not in docstring and verbose:
                        print(f"      {name} has positional arguments; "
                              "check manually")
                    continue

                # Check for keyword arguments (then look at code for parsing)
                elif par.kind == inspect.Parameter.VAR_KEYWORD:
                    # See if we documented the keyward argumnt directly
                    if f"**{argname}" in docstring:
                        continue

                    # Look for direct kwargs argument access
                    kwargnames = set()
                    for _, kwargname in re.findall(
                            argname + r"(\[|\.pop\(|\.get\()'([\w]+)'",
                            source):
                        if verbose > 2:
                            print("    Found direct keyword argument",
                                  kwargname)
                        kwargnames.add(kwargname)

                    # Look for kwargs access via _process_legacy_keyword
                    for kwargname in re.findall(
                            r"_process_legacy_keyword\([\s]*" + argname +
                            r",[\s]*'[\w]+',[\s]*'([\w]+)'", source):
                        if verbose > 2:
                            print("    Found legacy keyword argument",
                                  {kwargname})
                        kwargnames.add(kwargname)

                    for kwargname in kwargnames:
                        if obj in keyword_skiplist and \
                           kwargname in keyword_skiplist[obj]:
                            continue
                        if verbose > 3:
                            print(f"    Checking keyword argument {kwargname}")
                        assert _check_docstring(
                            name, kwargname, inspect.getdoc(obj),
                            prefix=prefix)

                # Make sure this argument is documented properly in docstring
                else:
                    if verbose > 3:
                        print(f"    Checking argument {argname}")
                    assert _check_docstring(
                        name, argname, docstring, prefix=prefix)


# Utility function to check for an argument in a docstring
def _check_docstring(funcname, argname, docstring, prefix=""):
    funcname = prefix + funcname
    if re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))*:",
            docstring):
        # Found the string, but not in numpydoc form
        if verbose:
            print(f"      {funcname}: {argname} docstring missing space")
        warnings.warn(f"{funcname} '{argname}' docstring missing space")
        return True

    elif not re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))* :",
            docstring):
        # return False
        #
        # Just issue a warning for now
        if verbose:
            print(f"      {funcname}: {argname} not documented")
        warnings.warn(f"{funcname} '{argname}' not documented")
        return True

    return True
