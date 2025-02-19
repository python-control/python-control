# docstrings_test.py - test for undocumented arguments
# RMM, 28 Jul 2024
#
# This unit test looks through all functions in the package and attempts to
# identify arguments that are not documented.  It will check anything that
# is an explicitly listed argument, as well as attempt to find keyword
# arguments that are extracted using kwargs.pop(), config._get_param(), or
# config.use_legacy_defaults.
#
# This module can also be run in standalone mode:
#
#     python docstrings_test.py [verbose]
#
# where 'verbose' is an integer indicating what level of verbosity is
# desired (0 = only warnings/errors, 10 = everything).

import inspect
import re

import sys
import warnings

import numpydoc.docscrape as npd
import pytest

import control
import control.flatsys
import control.matlab

# List of functions that we can skip testing (special cases)
function_skiplist = [
    control.ControlPlot.reshape,                # needed for legacy interface
    control.phase_plot,                         # legacy function
    control.drss,                               # documention in rss
    control.LinearICSystem,                     # intermediate I/O class
    control.LTI,                                # intermediate I/O class
    control.NamedSignal,                        # internal I/O class
    control.TimeResponseList,                   # internal response class
    control.FrequencyResponseList,              # internal response class
    control.NyquistResponseList,                # internal response class
    control.PoleZeroList,                       # internal response class
    control.FrequencyResponseData,              # check separately (iosys)
    control.InterconnectedSystem,               # check separately (iosys)
    control.flatsys.FlatSystem,                 # check separately (iosys)
]

# List of keywords that we can skip testing (special cases)
keyword_skiplist = {
    control.input_output_response: ['method', 't_eval'],    # solve_ivp_kwargs
    control.nyquist_plot: ['color'],                        # separate check
    control.optimal.solve_optimal_trajectory:
      ['method', 'return_x'],                               # deprecated
    control.sisotool: ['kvect'],                            # deprecated
    control.nyquist_response: ['return_contour'],           # deprecated
    control.create_estimator_iosystem: ['state_labels'],    # deprecated
    control.bode_plot: ['sharex', 'sharey', 'margin_info'], # deprecated
    control.eigensys_realization: ['arg'],                  # quasi-positional
    control.find_operating_point: ['method'],               # internal use
    control.zpk: ['args'],                                  # 'dt' (manual)
    control.StateSpace.dynamics: ['params'],                # not allowed
    control.StateSpace.output: ['params'],                  # not allowed
    control.flatsys.point_to_point: [
        'method', 'options',                                # minimize_kwargs
    ],
    control.flatsys.solve_flat_optimal: [
        'method', 'options',                                # minimize_kwargs
    ],
    control.optimal.OptimalControlProblem: [
        'method', 'options'     # solve_ivp_kwargs, minimize_kwargs
    ],
    control.optimal.OptimalControlResult: [
        'return_x', 'return_states', 'transpose'],          # legacy
    control.optimal.OptimalControlProblem.compute_trajectory: [
        'return_x',                                         # legacy
    ],
    control.optimal.OptimalEstimationProblem: [
        'method', 'options'     # solve_ivp_kwargs, minimize_kwargs
    ],
    control.optimal.OptimalEstimationResult: [
        'return_x', 'return_states', 'transpose'],          # legacy
    control.optimal.OptimalEstimationProblem.create_mhe_iosystem: [
        'inputs', 'outputs', 'states',                      # doc'd elsewhere
    ],
}

# Set global variables
verbose = 0             # Level of verbosity (use -rP when running pytest)
standalone = False      # Controls how failures are treated
max_summary_len = 64    # Maximum length of a summary line

module_list = [
    (control, ""), (control.flatsys, "flatsys."),
    (control.optimal, "optimal."), (control.phaseplot, "phaseplot."),
    (control.matlab, "matlab.")]

@pytest.mark.parametrize("module, prefix", module_list)
def test_parameter_docs(module, prefix):
    checked = set()             # Keep track of functions we have checked

    # Look through every object in the package
    _info(f"Checking module {module}", 0)
    for name, obj in inspect.getmembers(module):
        if getattr(obj, '__module__', None):
            objname = ".".join([obj.__module__.removeprefix("control."), name])
        else:
            objname = name
        _info(f"Checking object {objname}", 4)

        # Parse the docstring using numpydoc
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # debug via sphinx, not here
            doc = None if obj is None else npd.FunctionDoc(obj)

        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            _info(f"member '{objname}' is outside `control` module", 5)
            continue

        # Skip non-top-level functions without documentation
        if prefix != "" and inspect.getmodule(obj) != module and doc is None:
            _info(f"skipping {objname} [no docstring]", 1)
            continue

        # If this is a class, recurse through methods
        # TODO: check top level documenation here (__init__, attributes?)
        if inspect.isclass(obj):
            _info(f"Checking class {objname}", 1)

            # Check member functions within the class
            test_parameter_docs(obj, prefix + name + '.')

            # Drop through and continue checks as a function

        # Skip anything that is inherited, hidden, or already checked
        if not (inspect.isfunction(obj) or inspect.isclass(obj) and
                not issubclass(obj, Exception)) or \
           inspect.isclass(module) and name not in module.__dict__ \
           or name.startswith('_') or obj in function_skiplist \
           or obj in checked:
            _info(f"skipping {objname} [inherited, hidden, or checked]", 4)
            continue

        # Don't fail on non-top-level functions without parameter lists
        _info(f"Checking function {objname} against numpydoc", 2)
        _check_numpydoc_style(obj, doc)

        # Add this to the list of functions we have checked
        checked.add(obj)

        # Get the docstring (skip w/ warning if there isn't one)
        _info(f"Checking function {objname} against python-control", 2)
        if obj.__doc__ is None:
            _warn(f"{objname} is missing docstring", 2)
            continue
        elif doc is None:
            _fail(f"{objname} docstring not parseable", 2)
            continue
        else:
            docstring = inspect.getdoc(obj)

        if inspect.isclass(obj):
            # Just check __init__()
            source = inspect.getsource(obj.__init__)
        else:
            source = inspect.getsource(obj)

        # Skip deprecated functions (and check for proper annotation)
        doc_extended = "\n".join(doc["Extended Summary"])
        if ".. deprecated::" in doc_extended:
            _info("  [deprecated]", 2)
            continue
        elif re.search(name + r"(\(\))? is deprecated", doc_extended) or \
             "function is deprecated" in doc_extended:
            _info("  [deprecated, but not numpydoc compliant]", 2)
            _warn(f"{objname} deprecated, but not numpydoc compliant", 0)
            continue
        elif re.search(name + r"(\(\))? is deprecated", source):
            _warn(f"{objname} is deprecated, but not documented", 1)
            continue

        # Get the signature for the function
        sig = inspect.signature(obj)

	# If first argument is *args, try to use docstring instead
        sig = _replace_var_positional_with_docstring(sig, doc)

        # Skip functions whose documentation is found elsewhere
        if doc["Parameters"] == [] and re.search(
                r"See[\s]+`[\w.]+`[\s]+(for|and)", doc_extended):
               _info("skipping {objname}; references another function", 4)
               continue

        # Go through each parameter and make sure it is in the docstring
        for argname, par in sig.parameters.items():
            # Look for arguments that we can skip
            if argname == 'self' or argname[0] == '_' or \
               obj in keyword_skiplist and argname in keyword_skiplist[obj]:
                continue

            # Check for positional arguments (*arg)
            if par.kind == inspect.Parameter.VAR_POSITIONAL:
                if f"*{argname}" not in docstring:
                    _fail(
                        f"{objname} has undocumented, unbound positional "
                        f"argument '{argname}'; "
                        "use docstring signature instead")
                    continue

            # Check for keyword arguments (then look at code for parsing)
            elif par.kind == inspect.Parameter.VAR_KEYWORD:
                # See if we documented the keyward argument directly
                # if f"**{argname} :" in docstring:
                #     continue

                # Look for direct kwargs argument access
                kwargnames = set()
                for _, kwargname in re.findall(
                        argname + r"(\[|\.pop\(|\.get\()'([\w]+)'", source):
                    _info(f"Found direct keyword argument {kwargname}", 2)
                    if not kwargname.startswith('_'):
                        kwargnames.add(kwargname)

                # Look for kwargs accessed via _get_param
                for kwargname in re.findall(
                        r"_get_param\(\s*'\w*',\s*'([\w]+)',\s*" + argname,
                        source):
                    _info(f"Found config keyword argument {kwargname}", 2)
                    kwargnames.add(kwargname)

                # Look for kwargs accessed via _process_legacy_keyword
                for kwargname in re.findall(
                        r"_process_legacy_keyword\([\s]*" + argname +
                        r",[\s]*'[\w]+',[\s]*'([\w]+)'", source):
                    _info(f"Found legacy keyword argument {kwargname}", 2)
                    kwargnames.add(kwargname)

                for kwargname in kwargnames:
                    if obj in keyword_skiplist and \
                       kwargname in keyword_skiplist[obj]:
                        continue
                    _info(f"Checking keyword argument {kwargname}", 3)
                    _check_parameter_docs(
                        name, kwargname, inspect.getdoc(obj),
                        prefix=prefix)

            # Make sure this argument is documented properly in docstring
            else:
                _info(f"Checking argument {argname}", 3)
                _check_parameter_docs(
                        objname, argname, docstring, prefix=prefix)

        # Look at the return values
        for val in doc["Returns"]:
            if val.name == '' and \
               (match := re.search(r"([\w]+):", val.type)) is not None:
                retname = match.group(1)
                _warn(
                    f"{obj} return value '{retname}' "
                    "docstring missing space")

        # Look at the exceptions
        for exc in doc["Raises"]:
            _check_numpydoc_param(
                obj.__name__, exc, noname_ok=True, section="Raises")


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

        # Parse the docstring using numpydoc
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # debug via sphinx, not here
            doc = None if obj is None else npd.FunctionDoc(obj)

        if inspect.isfunction(obj):
            # Skip anything that is inherited, hidden, or checked
            if inspect.isclass(module) and name not in module.__dict__ \
               or name[0] == '_' or obj in checked:
                continue
            else:
                checked.add(obj)

            # Get the docstring (skip w/ warning if there isn't one)
            if obj.__doc__ is None:
                _warn(f"{obj} is missing docstring")
                continue
            else:
                docstring = inspect.getdoc(obj)
                source = inspect.getsource(obj)

            # Look for functions marked as deprecated in doc string
            doc_extended = "\n".join(doc["Extended Summary"])
            if ".. deprecated::" in doc_extended:
                # Make sure a FutureWarning is issued
                if not re.search("FutureWarning", source):
                    _fail(f"{obj} deprecated but does not issue "
                          "FutureWarning")
            else:
                if re.search(name + r"(\(\))? is deprecated", docstring) or \
                   re.search(name + r"(\(\))? is deprecated", source):
                    _fail(
                        f"{obj} deprecated but with non-standard "
                        "docs/warnings")

#
# Tests for I/O system classes
#
# The tests below try to make sure that we document I/O system classes
# and the factory functions that create them in a uniform way.
#

ct = control
fs = control.flatsys

# Dictionary of factory functions associated with primary classes
iosys_class_factory_function = {
    fs.FlatSystem: fs.flatsys,
    ct.FrequencyResponseData: ct.frd,
    ct.InterconnectedSystem: ct.interconnect,
    ct.LinearICSystem: ct.interconnect,
    ct.NonlinearIOSystem: ct.nlsys,
    ct.StateSpace: ct.ss,
    ct.TransferFunction: ct.tf,
}

#
# List of arguments described in class docstrings
#
# These are the minimal arguments needed to initialize the class.  Optional
# arguments should be documented in the factory functions and do not need
# to be duplicated in the class documentation (=> don't list here).
#
iosys_class_args = {
    fs.FlatSystem: ['forward', 'reverse'],
    ct.FrequencyResponseData: ['frdata', 'omega', 'dt'],
    ct.NonlinearIOSystem: [
        'updfcn', 'outfcn', 'inputs', 'outputs', 'states', 'params', 'dt'],
    ct.StateSpace: ['A', 'B', 'C', 'D', 'dt'],
    ct.TransferFunction: ['num', 'den', 'dt'],
    ct.InterconnectedSystem: [
        'syslist', 'connections', 'inplist', 'outlist', 'params']
}

#
# List of attributes described in class docstrings
#
# This is the list of attributes for the class that are not already listed
# as parameters used to initialize the class.  These should all be defined
# in the class docstring.
#
# Attributes that are part of all I/O system classes should be listed in
# `std_iosys_class_attributes`.  Attributes that are not commonly needed are
# defined as part of a parent class can just be documented there, and
# should be listed in `iosys_parent_attributes` (these will be searched
# using the MRO).

std_iosys_class_attributes = [
    'ninputs', 'noutputs', 'input_labels', 'output_labels', 'name', 'shape']

# List of attributes defined for specific I/O systems
iosys_class_attributes = {
    fs.FlatSystem: [],
    ct.FrequencyResponseData: [],
    ct.NonlinearIOSystem: ['nstates', 'state_labels'],
    ct.StateSpace: ['nstates', 'state_labels'],
    ct.TransferFunction: [],
    ct.InterconnectedSystem: [
        'connect_map', 'input_map', 'output_map',
        'input_offset', 'output_offset', 'state_offset', 'syslist_index',
        'nstates', 'state_labels' ]
}

# List of attributes defined in a parent class (no need to warn)
iosys_parent_attributes = [
    'input_index', 'output_index', 'state_index',       # rarely used
    'states', 'nstates', 'state_labels',                # not need in TF, FRD
    'params', 'outfcn', 'updfcn',                       # NL I/O, SS overlap
    'repr_format'                                       # rarely used
]

#
# List of arguments described (only) in factory function docstrings
#
# These lists consist of the arguments that should be documented in the
# factory functions and should not be duplicated in the class
# documentation, even though in some cases they are actually processed in
# the class __init__ function.
#
std_factory_args = [
    'inputs', 'outputs', 'name', 'input_prefix', 'output_prefix']

factory_args = {
    fs.flatsys: ['states', 'state_prefix'],
    ct.frd: ['sys'],
    ct.nlsys: ['state_prefix'],
    ct.ss: ['sys', 'states', 'state_prefix'],
    ct.tf: ['sys'],
    ct.interconnect: ['dt']
}


@pytest.mark.parametrize(
    "cls, fcn, args",
    [(cls, iosys_class_factory_function[cls], iosys_class_args[cls])
     for cls in iosys_class_args.keys()])
def test_iosys_primary_classes(cls, fcn, args):
    docstring = inspect.getdoc(cls)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # debug via sphinx, not here
        doc = npd.FunctionDoc(cls)
    _check_numpydoc_style(cls, doc)

    # Make sure the typical arguments are there
    for argname in args + std_iosys_class_attributes + \
            iosys_class_attributes[cls]:
        _check_parameter_docs(cls.__name__, argname, docstring)

    # Make sure we reference the factory function
    if re.search(
            f"`(~[\\w.]*)*{fcn.__name__}`"
            r"[\s]+factory[\s]+function", "\n".join(doc["Extended Summary"]),
            re.DOTALL) is None:
        _fail(
            f"{cls.__name__} summary does not reference factory function "
            f"{fcn.__name__}")

    if doc["See Also"] == []:
        _fail(
            f'{cls.__name__} does not have "See Also" section; '
            f"must include and reference {fcn.__name__}")
    else:
        found_factory_function = False
        for name, _ in doc["See Also"][0][0]:
            if name == f"{fcn.__name__}":
                found_factory_function = True
                break;
        if not found_factory_function:
            _fail(
                f'{cls.__name__} "See Also" section does not reference '
                f"factory function {fcn.__name__}")

    # Make sure we don't reference parameters from the factory function
    for argname in factory_args[fcn]:
        if re.search(f"[\\s]+{argname}(, .*)*[\\s]*:", docstring) is not None:
            _fail(
                f"{cls.__name__} references factory function parameter "
                f"'{argname}'")


@pytest.mark.parametrize("cls", iosys_class_args.keys())
def test_iosys_attribute_lists(cls, ignore_future_warning):
    fcn = iosys_class_factory_function[cls]

    # Create a system that we can scan for attributes
    sys = ct.rss(2, 1, 1)
    ignore_args = []
    match fcn:
        case ct.tf:
            sys = ct.tf(sys)
            ignore_args = ['state_labels']
        case ct.frd:
            sys = ct.frd(sys, [0.1, 1, 10])
            ignore_args = ['state_labels']
            ignore_args += ['fresp', 'response']        # deprecated
        case ct.interconnect:
            sys = ct.nlsys(sys, name='sys')
            sys = ct.interconnect([sys], inplist='sys.u', outlist='sys.y')
        case ct.nlsys:
            sys = ct.nlsys(sys)
        case fs.flatsys:
            sys = fs.flatsys(sys)
            sys = fs.flatsys(sys.forward, sys.reverse)

    docstring = inspect.getdoc(cls)
    for name, value in inspect.getmembers(sys):
        if name.startswith('_') or name in ignore_args or \
           inspect.ismethod(value):
            # Skip hidden and ignored attributes; methods checked elsewhere
            continue

        # Try to find documentation in primary class
        if _check_parameter_docs(
                cls.__name__, name, docstring, fail_if_missing=False):
            continue

        # Couldn't find in main documentation; look in parent classes
        for parent in cls.__mro__:
            if parent == object:
                _fail(
                    f"{cls.__name__} attribute '{name}' not documented")
                break

            if _check_parameter_docs(
                    parent.__name__, name, inspect.getdoc(parent),
                    fail_if_missing=False):
                if name not in iosys_parent_attributes + factory_args[fcn]:
                    _warn(
                        f"{cls.__name__} attribute '{name}' only documented "
                        f"in parent class {parent.__name__}")
                break


@pytest.mark.parametrize("cls", [ct.InputOutputSystem, ct.LTI])
def test_iosys_container_classes(cls):
    # Create a system that we can scan for attributes
    sys = cls(states=2, outputs=1, inputs=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # debug via sphinx, not here
        doc = npd.FunctionDoc(cls)
    _check_numpydoc_style(cls, doc)

    for name, obj in inspect.getmembers(sys):
        if name.startswith('_') or inspect.ismethod(obj):
            # Skip hidden variables; class methods are checked elsewhere
            continue

        # Look through all classes in hierarchy
        _info(f"{name=}", 1)
        for parent in cls.__mro__:
            if parent == object:
                _fail(
                    f"{cls.__name__} attribute '{name}' not documented")
                break

            _info(f"  {parent=}", 2)
            if _check_parameter_docs(
                    parent.__name__, name, inspect.getdoc(parent),
                    fail_if_missing=False):
                break


@pytest.mark.parametrize("cls", [ct.LTI, ct.LinearICSystem])
def test_iosys_intermediate_classes(cls):
    docstring = inspect.getdoc(cls)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # debug via sphinx, not here
        doc = npd.FunctionDoc(cls)
    _check_numpydoc_style(cls, doc)

    # Make sure there is not a parameters section
    # TODO: replace with numpdoc check
    if re.search(r"\nParameters\n----", docstring) is not None:
        _fail(f"intermediate {cls} docstring contains Parameters section")
        return


@pytest.mark.parametrize("fcn", factory_args.keys())
def test_iosys_factory_functions(fcn):
    docstring = inspect.getdoc(fcn)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # debug via sphinx, not here
        doc = npd.FunctionDoc(fcn)
    _check_numpydoc_style(fcn, doc)

    cls = list(iosys_class_factory_function.keys())[
        list(iosys_class_factory_function.values()).index(fcn)]

    # Make sure we reference parameters in class and factory function docstring
    for argname in iosys_class_args[cls] + std_factory_args + factory_args[fcn]:
        _check_parameter_docs(fcn.__name__, argname, docstring)

    # Make sure we don't reference any class attributes
    for argname in std_iosys_class_attributes + iosys_class_attributes[cls]:
        if argname in std_factory_args:
            continue
        if re.search(f"[\\s]+{argname}(, .*)*[\\s]*:", docstring) is not None:
            _fail(
                f"{fcn.__name__} references class attribute '{argname}'")


# Utility function to check for an argument in a docstring
def _check_parameter_docs(
        funcname, argname, docstring, prefix="", fail_if_missing=True):
    funcname = prefix + funcname

    # Find the "Parameters" section of docstring, where we start searching
    # TODO: rewrite to use numpydoc
    if not (match := re.search(r"\nParameters\n----", docstring)):
        if fail_if_missing:
            _fail(f"{funcname} docstring missing Parameters section")
            return False        # for standalone mode
        else:
            return False
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
    elif match_returns:
        docstring = docstring[start:match_returns.start()]
    else:
        docstring = docstring[start:]

    # Look for the parameter name in the docstring
    argname_ = argname + r"( \(or .*\))*"
    if match := re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname_ + r"(, (\w+|\.{3}))*:",
            docstring):
        # Found the string, but not in numpydoc form
        _warn(f"{funcname}: {argname} docstring missing space")

    elif not (match := re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname_ + r"(, (\w+|\.{3}))* :",
            docstring)):
        if fail_if_missing:
            _fail(f"{funcname} '{argname}' not documented")
            return False        # for standalone mode
        else:
            _info(f"{funcname} '{argname}' not documented (OK)", 6)
            return False

    # Make sure there isn't another instance
    second_match = re.search(
            "\n" + r"((\w+|\.{3}), )*" + argname + r"(, (\w+|\.{3}))*[ ]*:",
            docstring[match.end():])
    if second_match:
        _fail(f"{funcname} '{argname}' documented twice")
        return False            # for standalone mode

    return True


# Utility function to check numpydoc style consistency
def _check_numpydoc_style(obj, doc):
    name = ".".join([obj.__module__.removeprefix("control."), obj.__name__])

    # Standard checks for all objects
    summary = "\n".join(doc["Summary"])
    if len(doc["Summary"]) > 1:
        _warn(f"{name} summary is more than one line")
    if summary and summary[-1] != '.' and re.match(":$", summary) is None:
        _warn(f"{name} summary doesn't end in period")
    if summary[0:1].islower():
        _warn(f"{name} summary starts with lower case letter")
    if len(summary) > max_summary_len:
        _warn(f"{name} summary is longer than {max_summary_len} characters")

    # Look for Python objects that are not marked properly
    python_objects = ['True', 'False', 'None']
    for pyobj in python_objects:
        for section in ["Extended Summary", "Notes"]:
            text = "\n".join(doc[section])
            if re.search(f"`{pyobj}`", text) is not None:
                _warn(f"{pyobj} appears in {section} for {name} with backticks")

    control_classes = [
        'InputOutputSystem', 'NonlinearIOSystem', 'StateSpace',
        'TransferFunction', 'FrequencyResponseData', 'LinearICSystem',
        'Flatsystem', 'InterconnectedSystem', 'TimeResponseData',
        'NyquistResponseData', 'PoleZeroData', 'RootLocusData',
        'ControlPlot', 'OperatingPoint', 'flatsys.Flatsystem']
    for pyobj in control_classes:
        if obj.__name__ == pyobj:
            continue
        for section in ["Extended Summary", "Notes"]:
            text = "\n".join(doc[section])
            if re.search(f"[^`]{pyobj}[^`.]", text) is not None:
                _warn(f"{pyobj} in {section} for {name} w/o backticks")

        for section in [
                "Parameters", "Returns", "Additional Parameters", "Yields"]:
            if section not in doc:
                continue
            for arg in doc[section]:
                text = arg.type + "\n".join(arg.desc)
                if re.search(f"(^|[^`]){pyobj}([^`.]|$)", text) is not None:
                    _warn(f"{pyobj} in {section} for {name} w/o backticks")

    if inspect.isclass(obj):
        # Specialized checks for classes
        if doc["Returns"] != []:
            _fail(f'Class {name} should not have "Returns" section')

    elif inspect.isfunction(obj):
        # Specialized checks for functions
        if doc["Returns"] == [] and obj.__doc__ and 'return' in obj.__doc__:
            _fail(f'Class {name} does not have a "Returns" section')

    else:
        raise TypeError("unknown object type for {obj}")

    for param in doc["Parameters"] + doc["Other Parameters"]:
        _check_numpydoc_param(name, param, section="Parameters")
    for param in doc["Attributes"]:
        _check_numpydoc_param(name, param, section="Attributes")
    for param in doc["Returns"]:
        _check_numpydoc_param(
            name, param, empty_ok=True, noname_ok=True, section="Returns")
    for param in doc["Yields"]:
        _check_numpydoc_param(
            name, param, empty_ok=True, noname_ok=True, section="Yields")


# Utility function for checking NumPyDoc parametres
def _check_numpydoc_param(
        name, param, empty_ok=False, noname_ok=False, section="??"):
    param_desc = "\n".join(param.desc)
    param_name = f"{name} " + \
        (f" '{param.name}'" if param.name != '' else f" '{param.type}'")

    # Check for empty section
    if param.name == "" and param.type == '':
        _fail(f"Empty {section} section in {name}")

    # Make sure we have a name and description
    if param.name == "" and not noname_ok:
        _fail(f"{param_name} has improperly formatted parameter")
        return
    elif param_desc == "":
        if not empty_ok:
            _warn(f"{param_name} isn't documented")
        return

    # Description should end in a period (colon also allowed)
    if re.search(r"\.$|\.[\s]|:$", param_desc, re.MULTILINE) is None:
        _warn(f"{param_name} description doesn't contain period")
    if param_desc[0:1].islower():
        _warn(f"{param_name} description starts with lower case letter")

    # Look for Python objects that are not marked properly
    python_objects = ['True', 'False', 'None']
    for pyobj in python_objects:
        if re.search(f"`{pyobj}`", param_desc) is not None:
            _warn(f"{pyobj} appears in {param_name} description with backticks")


# Utility function to replace positional signature with docstring signature
def _replace_var_positional_with_docstring(sig, doc):
    # If no documentation is available, there is nothing we can do...
    if doc is None:
        return sig

    # Check to see if the first argument is positional
    parameter_items = iter(sig.parameters.items())
    try:
        argname, par = next(parameter_items)
        if par.kind != inspect.Parameter.VAR_POSITIONAL or \
           (signature := doc["Signature"]) == '':
            return sig
    except StopIteration:
        return sig

    # Try parsing the docstring signature
    arg_list = []
    while (1):
        if (match_fcn := re.match(
                r"^([\s]*\|[\s]*)*[\w]+\(", signature)) is None:
            break
        arg_idx = match_fcn.span(0)[1]
        while (1):
            match_arg = re.match(
                r"[\s]*([\w]+)(,|,\[|\[,|\)|\]\))(,[\s]*|[\s]*[.]{3},[\s]*)*",
                signature[arg_idx:])
            if match_arg is None:
                break
            else:
                arg_idx += match_arg.span(0)[1]
                arg_list.append(match_arg.group(1))
        signature = signature[arg_idx:]
    if arg_list == []:
        return sig

    # Create the new parameter list
    parameter_list = [
        inspect.Parameter(arg, inspect.Parameter.POSITIONAL_ONLY)
        for arg in arg_list]

    # Add any remaining parameters that were in the original signature
    for argname, par in parameter_items:
        if argname not in arg_list:
            parameter_list.append(par)

    # Return the new signature
    return sig.replace(parameters=parameter_list)


# Utility function to warn with verbose output
def _info(str, level):
    if verbose > level:
        print("  " * level + str)

def _warn(str, level=-1):
    print("WARN: " + "  " * level + str)
    if not standalone:
        warnings.warn(str, stacklevel=2)

def _fail(str, level=-1):
    if verbose > level:
        print("FAIL: " + "  " * level + str)
    if not standalone:
        pytest.fail(str)

#
# Test function for the unit test
#
class simple_class:
    def simple_function(arg1, arg2, opt1=None, **kwargs):
        """Simple function for testing."""
        kwargs['test'] = None

Failed = pytest.fail.Exception

doc_header = simple_class.simple_function.__doc__ + "\n"
doc_parameters = "\nParameters\n----------\n"
doc_arg1 = "arg1 : int\n    Argument 1.\n"
doc_arg2 = "arg2 : int\n    Argument 2.\n"
doc_arg2_nospace = "arg2: int\n    Argument 2.\n"
doc_arg3 = "arg3 : int\n    Non-existent argument 1.\n"
doc_opt1 = "opt1 : int\n    Keyword argument 1.\n"
doc_test = "test : int\n    Internal keyword argument 1.\n"
doc_returns = "\nReturns\n-------\n"
doc_ret = "out : int\n"
doc_ret_nospace = "out: int\n"

@pytest.mark.parametrize("docstring, exception, match", [
    (None, UserWarning, "missing docstring"),
    (doc_header + doc_parameters + doc_arg1 + doc_arg2 + doc_opt1 +
     doc_test + doc_returns + doc_ret, None, ""),
    (doc_header + doc_parameters + doc_arg1 + doc_arg2 + doc_opt1 + doc_test,
     None, ""),                 # no return section (OK)
    (doc_header + doc_parameters + doc_arg1 + doc_arg2_nospace + doc_opt1 +
     doc_test + doc_returns + doc_ret, UserWarning, "missing space"),
    (doc_header + doc_parameters + doc_arg1 + doc_opt1 +
     doc_test + doc_returns + doc_ret, Failed, "'arg2' not documented"),
    (doc_header + doc_parameters + doc_arg1 + doc_arg2 + doc_arg2 + doc_opt1 +
     doc_test + doc_returns + doc_ret, Failed, "'arg2' documented twice"),
    (doc_header + doc_parameters + doc_arg1 + doc_arg2 + doc_opt1 +
     doc_returns + doc_ret, Failed, "'test' not documented"),
    (doc_header + doc_parameters + doc_arg1 + doc_arg2_nospace + doc_opt1 +
     doc_test + doc_returns + doc_ret_nospace, UserWarning, "missing space"),
    (doc_header + doc_returns + doc_ret_nospace,
     Failed, "missing Parameters section"),
    (doc_header + "\nSee `other_function` for details", None, ""),
    (doc_header + "\n.. deprecated::", None, ""),
    (doc_header + "\n\n simple_function() is deprecated",
     UserWarning, "deprecated, but not numpydoc compliant"),
])
def test_check_parameter_docs(docstring, exception, match):
    simple_class.simple_function.__doc__ = docstring
    if exception is None:
        # Pass prefix to allow empty parameters to work
        assert test_parameter_docs(simple_class, "test") is None
    elif exception in [UserWarning]:
        with pytest.warns(exception, match=match):
            test_parameter_docs(simple_class, "") is None
    elif exception in [Failed]:
        with pytest.raises(exception, match=match):
            test_parameter_docs(simple_class, "") is None


if __name__ == "__main__":
    verbose = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    standalone = True

    for module, prefix in module_list:
        _info(f"--- test_parameter_docs(): {module.__name__} ----", 0)
        test_parameter_docs(module, prefix)

    for module, prefix in module_list:
        _info(f"--- test_deprecated_functions(): {module.__name__} ----", 0)
        test_deprecated_functions

    for cls, fcn, args in [
            (cls, iosys_class_factory_function[cls], iosys_class_args[cls])
            for cls in iosys_class_args.keys()]:
        _info(f"--- test_iosys_primary_classes(): {cls.__name__} ----", 0)
        test_iosys_primary_classes(cls, fcn, args)

    for cls in iosys_class_args.keys():
        _info(f"--- test_iosys_attribute_lists(): {cls.__name__} ----", 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            test_iosys_attribute_lists(cls, None)

    for cls in [ct.InputOutputSystem, ct.LTI]:
        _info(f"--- test_iosys_container_classes(): {cls.__name__} ----", 0)
        test_iosys_container_classes(cls)

    for cls in [ct.LTI, ct.LinearICSystem]:
        _info(f"--- test_iosys_intermediate_classes(): {cls.__name__} ----", 0)
        test_iosys_intermediate_classes(cls)

    for fcn in factory_args.keys():
        _info(f"--- test_iosys_factory_functions(): {fcn.__name__} ----", 0)
        test_iosys_factory_functions(fcn)
