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
    control.append:                     'be014503250ef73253a5372a0d082566',
    control.describing_function_plot:   '726a10eef8f2b50ef46a203653f398c7',
    control.dlqe:                       '9f637afdf36c7e17b7524f9a736772b6',
    control.dlqr:                       'a9265c5ed66388661729edb2379fd9a1',
    control.lqe:                        'd265b0daf0369569e4b755fa35db7a54',
    control.lqr:                        '0b76455c2b873abcbcd959e069d9d241',
    control.frd:                        '7ac3076368f11e407653cd1046bbc98d',
    control.margin:                     '8ee27989f1ca521ce9affe5900605b75',
    control.parallel:                   'daa3b8708200a364d9b5536b6cbb5c91',
    control.series:                     '7241169911b641c43f9456bd12168271',
    control.ss:                         'aa77e816305850502c21bc40ce796f40',
    control.ss2tf:                      '8d663d474ade2950dd22ec86fe3a53b7',
    control.tf:                         '4e8d21e71312d83ba2e15b9c095fd962',
    control.tf2ss:                      '0e5da4f3ed4aaf000f3b454c466f9013',
    control.markov:                     '47f3b856ec47df84b2af2c165abaabfc',
    control.gangof4:                    '5e4b4cf815ef76d6c73939070bcd1489',
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
                            docstring.encode('utf-8')).hexdigest()
                        assert function_docstring_hash[obj] == hash
                        continue

                    # Too complicated to check
                    if f"*{argname}" not in docstring and verbose:
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
