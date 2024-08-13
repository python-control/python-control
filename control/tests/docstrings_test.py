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
    control.drss,                               # documention in rss
]

# Checksums to use for checking whether a docstring has changed
function_docstring_hash = {
    control.append:                     '48548c4c4e0083312b3ea9e56174b0b5',
    control.describing_function_plot:   '95f894706b1d3eeb3b854934596af09f',
    control.dlqe:                       '9db995ed95c2214ce97074b0616a3191',
    control.dlqr:                       '896cfa651dbbd80e417635904d13c9d6',
    control.lqe:                        '567bf657538935173f2e50700ba87168',
    control.lqr:                        'a3e0a85f781fc9c0f69a4b7da4f0bd22',
    control.frd:                        '099464bf2d14f25a8769ef951adf658b',
    control.margin:                     'f02b3034f5f1d44ce26f916cc3e51600',
    control.parallel:                   '025c5195a34c57392223374b6244a8c4',
    control.series:                     '9aede1459667738f05cf4fc46603a4f6',
    control.ss:                         '1b9cfad5dbdf2f474cfdeadf5cb1ad80',
    control.ss2tf:                      '48ff25d22d28e7b396e686dd5eb58831',
    control.tf:                         '53a13f4a7f75a31c81800e10c88730ef',
    control.tf2ss:                      '086a3692659b7321c2af126f79f4bc11',
    control.markov:                     '753309de348132ef238e78ac756412c1',
    control.gangof4:                    '0e52eb6cf7ce024f9a41f3ae3ebf04f7',
}

# List of keywords that we can skip testing (special cases)
keyword_skiplist = {
    control.input_output_response: ['method'],
    control.nyquist_plot: ['color'],                        # separate check
    control.optimal.solve_ocp: ['method', 'return_x'],      # deprecated
    control.sisotool: ['kvect'],                            # deprecated
    control.nyquist_response: ['return_contour'],           # deprecated
    control.create_estimator_iosystem: ['state_labels'],    # deprecated
    control.bode_plot: ['sharex', 'sharey', 'margin_info'], # deprecated
    control.eigensys_realization: ['arg'],                  # quasi-positional
}

# Decide on the level of verbosity (use -rP when running pytest)
verbose = 1

@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys."),
    (control.optimal, "optimal."), (control.phaseplot, "phaseplot.")
])
def test_parameter_docs(module, prefix):
    checked = set()             # Keep track of functions we have checked

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
            test_parameter_docs(obj, prefix + name + '.')

        if inspect.isfunction(obj):
            # Skip anything that is inherited, hidden, deprecated, or checked
            if inspect.isclass(module) and name not in module.__dict__ \
               or name.startswith('_') or obj in function_skiplist or \
               obj in checked:
                continue
            else:
                checked.add(obj)

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
            if ".. deprecated::" in docstring:
                if verbose > 1:
                    print("    [deprecated]")
                continue
            elif re.search(name + r"(\(\))? is deprecated", docstring) or \
                 "function is deprecated" in docstring:
                if verbose > 1:
                    print("    [deprecated, but not numpydoc compliant]")
                elif verbose:
                    print(f"    {name} deprecation is not numpydoc compliant")
                warnings.warn(f"{name} deprecated, but not numpydoc compliant")
                continue

            elif re.search(name + r"(\(\))? is deprecated", source):
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
                    if obj in function_docstring_hash:
                        import hashlib
                        hash = hashlib.md5(
                            (docstring + source).encode('utf-8')).hexdigest()
                        if function_docstring_hash[obj] != hash:
                            pytest.fail(
                                f"source/docstring for {name}() modified; "
                                f"recheck docstring and update hash to "
                                f"{hash=}")
                        continue

                    # Too complicated to check
                    if f"*{argname}" not in docstring:
                        if verbose:
                            print(f"      {name} has positional arguments; "
                                  "check manually")
                        warnings.warn(
                            f"{name} {argname} has positional arguments; "
                            "docstring not checked")
                    continue

                # Check for keyword arguments (then look at code for parsing)
                elif par.kind == inspect.Parameter.VAR_KEYWORD:
                    # See if we documented the keyward argumnt directly
                    # if f"**{argname} :" in docstring:
                    #     continue

                    # Look for direct kwargs argument access
                    kwargnames = set()
                    for _, kwargname in re.findall(
                            argname + r"(\[|\.pop\(|\.get\()'([\w]+)'",
                            source):
                        if verbose > 2:
                            print("    Found direct keyword argument",
                                  kwargname)
                        kwargnames.add(kwargname)

                    # Look for kwargs accessed via _get_param
                    for kwargname in re.findall(
                            r"_get_param\(\s*'\w*',\s*'([\w]+)',\s*" + argname,
                            source):
                        if verbose > 2:
                            print("    Found config keyword argument",
                                  {kwargname})
                        kwargnames.add(kwargname)

                    # Look for kwargs accessed via _process_legacy_keyword
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
                        _check_parameter_docs(
                            name, kwargname, inspect.getdoc(obj),
                            prefix=prefix)

                # Make sure this argument is documented properly in docstring
                else:
                    if verbose > 3:
                        print(f"    Checking argument {argname}")
                    _check_parameter_docs(
                        name, argname, docstring, prefix=prefix)


@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys."),
    (control.optimal, "optimal."), (control.phaseplot, "phaseplot.")
])
def test_deprecated_functions(module, prefix):
    checked = set()             # Keep track of functions we have checked

    # Look through every object in the package
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and (
                not inspect.getmodule(obj).__name__.startswith('control')
                or prefix != "" and inspect.getmodule(obj) != module):
            # Skip anything that isn't part of the control package
            continue

        if inspect.isclass(obj):
            # Check member functions within the class
            test_deprecated_functions(obj, prefix + name + '.')

        if inspect.isfunction(obj):
            # Skip anything that is inherited, hidden, or checked
            if inspect.isclass(module) and name not in module.__dict__ \
               or name[0] == '_' or obj in checked:
                continue
            else:
                checked.add(obj)

            # Get the docstring (skip w/ warning if there isn't one)
            if obj.__doc__ is None:
                warnings.warn(
                    f"{module.__name__}.{name} is missing docstring")
                continue
            else:
                docstring = inspect.getdoc(obj)
                source = inspect.getsource(obj)

            # Look for functions marked as deprecated in doc string
            if ".. deprecated::" in docstring:
                # Make sure a FutureWarning is issued
                if not re.search("FutureWarning", source):
                    pytest.fail(
                        f"{name} deprecated but does not issue FutureWarning")
            else:
                if re.search(name + r"(\(\))? is deprecated", docstring) or \
                   re.search(name + r"(\(\))? is deprecated", source):
                    pytest.fail(
                        f"{name} deprecated but w/ non-standard docs/warnings")
                assert name != 'ss2io'


# Utility function to check for an argument in a docstring
def _check_parameter_docs(funcname, argname, docstring, prefix=""):
    funcname = prefix + funcname

    # Find the "Parameters" section of docstring, where we start searching
    if not (match := re.search(r"\nParameters\n----", docstring)):
        pytest.fail(f"{funcname} docstring missing Parameters section")
    else:
        start = match.start()

    # Find the "Returns" section of the docstring (to be skipped, if present)
    match_returns = re.search(r"\nReturns\n----", docstring)

    # Find the "Other Parameters" section of the docstring, if present
    match_other = re.search(r"\nOther Parameters\n----", docstring)

    # Remove the returns section from docstring, in case output arguments
    # match input argument names (it happens...)
    if match_other and match_returns:
        docstring = docstring[start:match_returns.start()] + \
            docstring[match_other.start():]
    else:
        docstring = docstring[start:]

    # Look for the parameter name in the docstring
    if match := re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))*:",
            docstring):
        # Found the string, but not in numpydoc form
        if verbose:
            print(f"      {funcname}: {argname} docstring missing space")
        warnings.warn(f"{funcname} '{argname}' docstring missing space")

    elif not (match := re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))* :",
            docstring)):
        if verbose:
            print(f"      {funcname}: {argname} not documented")
        pytest.fail(f"{funcname} '{argname}' not documented")

    # Make sure there isn't another instance
    second_match = re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))*[ ]*:",
            docstring[match.end():])
    if second_match:
        pytest.fail(f"{funcname} '{argname}' documented twice")
