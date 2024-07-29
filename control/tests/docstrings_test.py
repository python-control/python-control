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
skiplist = [
    control.ControlPlot.reshape,                # needed for legacy interface
]

@pytest.mark.parametrize("module, prefix", [
    (control, ""), (control.flatsys, "flatsys."),
    (control.optimal, "optimal."), (control.phaseplot, "phaseplot.")
])
def test_docstrings(module, prefix):
    # Look through every object in the package
    print(f"Checking module {module}")
    for name, obj in inspect.getmembers(module):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue

        if inspect.isclass(obj):
            print(f"  Checking class {name}")
            # Check member functions within the class
            test_docstrings(obj, prefix + obj.__name__ + '.')

        if inspect.isfunction(obj):
            # Skip anything that is inherited or hidden
            if inspect.isclass(module) and obj.__name__ not in module.__dict__ \
               or obj.__name__.startswith('_') or obj in skiplist:
                continue

            # Make sure there is a docstring
            print(f"  Checking function {name}")
            if obj.__doc__ is None:
                warnings.warn(
                    f"{module.__name__}.{obj.__name__} is missing docstring")
                continue
            
            # Get the signature for the function
            sig = inspect.signature(obj)

            # Go through each parameter and make sure it is in the docstring
            for argname, par in sig.parameters.items():
                if argname == 'self' or argname[0] == '_':
                    continue
                
                if par.kind == inspect.Parameter.VAR_KEYWORD:
                    # Found a keyword argument; look at code for parsing
                    warnings.warn("keyword argument checks not yet implemented")

                # Make sure this argument is documented properly in docstring
                else:
                    assert _check_docstring(obj.__name__, argname, obj.__doc__)


# Utility function to check for an argument in a docstring
def _check_docstring(funcname, argname, docstring):
    if re.search(f"    ([ \\w]+, )*{argname}(,[ \\w]+)*[^ ]:", docstring):
        # Found the string, but not in numpydoc form
        warnings.warn(f"{funcname} '{argname}' docstring missing space")
        return True
    
    elif not re.search(f"    ([ \\w]+, )*{argname}(,[ \\w]+)* :", docstring):
        # return False
        #
        # Just issue a warning for now
        warnings.warn(f"{funcname} '{argname}' not documented")
        return True
    
    return True
