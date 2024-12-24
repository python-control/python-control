# test_sphinxdocs.py - pytest checks for user guide
# RMM, 23 Dec 2024
#
# This set of tests is used to make sure that all primary functions are
# referenced in the documentation.

import inspect
import os
import sys
import warnings
from importlib import resources

import pytest
import numpydoc.docscrape as npd

import control
import control.flatsys

# Location of the documentation and files to check
sphinx_dir = str(resources.files('control')) + '/../doc/generated/'

# Functions that should not be referenced
legacy_functions = [
    'balred',                   # balanced_reduction
    'bode',                     # bode_plot
    'c2d',                      # sample_system
    'era',                      # eigensys_realization
    'evalfr',                   # use __call__()
    'find_eqpt',                # find_operating_point
    'gangof4',                  # gangof4_plot
    'hsvd',                     # hankel_singular_values
    'minreal',                  # minimal_realization
    'modred',                   # model_reduction
    'nichols',                  # nichols_plot
    'nyquist',                  # nyquist_plot
    'pzmap',                    # pole_zero_plot
    'rlocus',                   # root_locus_plot
    'rlocus',                   # root_locus_plot
    'root_locus',               # root_locus_plot
]

# Functons that we can skip
object_skiplist = [
    control.NamedSignal,        # users don't need to know about this
    control.common_timebase,    # mainly internal use
    control.cvxopt_check,       # mainly internal use
    control.pandas_check,       # mainly internal use
    control.slycot_check,       # mainly internal use
]

# Global list of objects we have checked
checked = set()

# Decide on the level of verbosity (use -rP when running pytest)
verbose = 0
standalone = False

control_module_list = [
    control, control.flatsys, control.optimal, control.phaseplot
]
@pytest.mark.parametrize("module", control_module_list)
def test_sphinx_functions(module):

    # Look through every object in the package
    _info(f"Checking module {module}", 1)

    for name, obj in inspect.getmembers(module):
        objname = ".".join([module.__name__, name])

        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue

        elif inspect.isclass(obj) and issubclass(obj, Exception):
            continue

        elif inspect.isclass(obj) or inspect.isfunction(obj):
            # Skip anything that is inherited, hidden, deprecated, or checked
            if inspect.isclass(module) and name not in module.__dict__ \
               or name.startswith('_') or obj in checked:
                continue
            else:
                checked.add(obj)

            # Get the relevant information about this object
            exists = os.path.exists(sphinx_dir + objname + ".rst")
            deprecated = _check_deprecated(obj)
            skip = obj in object_skiplist
            referenced = f" {objname} referenced in sphinx docs"
            legacy = name in legacy_functions

            _info(f"  Checking {objname}", 2)
            match exists, skip, deprecated, legacy:
                case True, True, _, _:
                    _info(f"skipped object" + referenced, -1)
                case True, _, True, _:
                    _warn(f"deprecated object" + referenced)
                case True, _, _, True:
                    _warn(f"legacy object" + referenced)
                case False, False, False, False:
                    _fail(f"{objname} not referenced in sphinx docs")


def _check_deprecated(obj):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # debug via sphinx, not here
        doc = npd.FunctionDoc(obj)

    doc_extended = "" if doc is None else "\n".join(doc["Extended Summary"])
    return ".. deprecated::" in doc_extended


# Utility function to warn with verbose output
def _info(str, level):
    if verbose > level:
        print(("INFO: " if level < 0 else "  " * level) + str)

def _warn(str, level=-1):
    if verbose > level:
        print("WARN: " + "  " * level + str)
    if not standalone:
        warnings.warn(str, stacklevel=2)

def _fail(str, level=-1):
    if verbose > level:
        print("FAIL: " + "  " * level + str)
    if not standalone:
        pytest.fail(str)


if __name__ == "__main__":
    verbose = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    standalone = True

    for module in control_module_list:
        test_sphinx_functions(module)
