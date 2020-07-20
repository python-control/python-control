# config.py - package defaults
# RMM, 4 Nov 2012
#
# This file contains default values and utility functions for setting
# variables that control the behavior of the control package.
# Eventually it will be possible to read and write configuration
# files.  For now, you can just choose between MATLAB and FBS default
# values + tweak a few other things.

import warnings

__all__ = ['defaults', 'set_defaults', 'reset_defaults',
           'use_matlab_defaults', 'use_fbs_defaults',
           'use_legacy_defaults', 'use_numpy_matrix']

# Package level default values
_control_defaults = {
    'control.default_dt':0
}
defaults = dict(_control_defaults)


def set_defaults(module, **keywords):
    """Set default values of parameters for a module.

    The set_defaults() function can be used to modify multiple parameter
    values for a module at the same time, using keyword arguments:

        control.set_defaults('module', param1=val, param2=val)

    """
    if not isinstance(module, str):
        raise ValueError("module must be a string")
    for key, val in keywords.items():
        defaults[module + '.' + key] = val


def reset_defaults():
    """Reset configuration values to their default (initial) values."""
    # System level defaults
    defaults.update(_control_defaults)

    from .freqplot import _bode_defaults, _freqplot_defaults
    defaults.update(_bode_defaults)
    defaults.update(_freqplot_defaults)

    from .nichols import _nichols_defaults
    defaults.update(_nichols_defaults)

    from .pzmap import _pzmap_defaults
    defaults.update(_pzmap_defaults)

    from .rlocus import _rlocus_defaults
    defaults.update(_rlocus_defaults)

    from .xferfcn import _xferfcn_defaults
    defaults.update(_xferfcn_defaults)

    from .statesp import _statesp_defaults
    defaults.update(_statesp_defaults)

    from .iosys import _iosys_defaults
    defaults.update(_iosys_defaults)


def _get_param(module, param, argval=None, defval=None, pop=False):
    """Return the default value for a configuration option.

    The _get_param() function is a utility function used to get the value of a
    parameter for a module based on the default parameter settings and any
    arguments passed to the function.  The precedence order for parameters is
    the value passed to the function (as a keyword), the value from the
    config.defaults dictionary, and the default value `defval`.

    Parameters
    ----------
    module : str
        Name of the module whose parameters are being requested.
    param : str
        Name of the parameter value to be determeind.
    argval : object or dict
        Value of the parameter as passed to the function.  This can either be
        an object or a dictionary (i.e. the keyword list from the function
        call).  Defaults to None.
    defval : object
        Default value of the parameter to use, if it is not located in the
        `config.defaults` dictionary.  If a dictionary is provided, then
        `module.param` is used to determine the default value.  Defaults to
        None.
    pop : bool
        If True and if argval is a dict, then pop the remove the parameter
        entry from the argval dict after retreiving it.  This allows the use
        of a keyword argument list to be passed through to other functions
        internal to the function being called.

    """

    # Make sure that we were passed sensible arguments
    if not isinstance(module, str) or not isinstance(param, str):
        raise ValueError("module and param must be strings")

    # Construction the name of the key, for later use
    key = module + '.' + param

    # If we were passed a dict for the argval, get the param value from there
    if isinstance(argval, dict):
        argval = argval.pop(param, None) if pop else argval.get(param, None)

    # If we were passed a dict for the defval, get the param value from there
    if isinstance(defval, dict):
        defval = defval.get(key, None)

    # Return the parameter value to use (argval > defaults > defval)
    return argval if argval is not None else defaults.get(key, defval)


# Set defaults to match MATLAB
def use_matlab_defaults():
    """Use MATLAB compatible configuration settings.

    The following conventions are used:
        * Bode plots plot gain in dB, phase in degrees, frequency in
          rad/sec, with grids
        * State space class and functions use Numpy matrix objects

    """
    set_defaults('bode', dB=True, deg=True, Hz=False, grid=True)
    set_defaults('statesp', use_numpy_matrix=True)


# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    """Use `Feedback Systems <http://fbsbook.org>`_ (FBS) compatible settings.

    The following conventions are used:
        * Bode plots plot gain in powers of ten, phase in degrees,
          frequency in rad/sec, no grid

    """
    set_defaults('bode', dB=False, deg=True, Hz=False, grid=False)


# Decide whether to use numpy.matrix for state space operations
def use_numpy_matrix(flag=True, warn=True):
    """Turn on/off use of Numpy `matrix` class for state space operations.

    Parameters
    ----------
    flag : bool
        If flag is `True` (default), use the Numpy (soon to be deprecated)
        `matrix` class to represent matrices in the `~control.StateSpace`
        class and functions.  If flat is `False`, then matrices are
        represented by a 2D `ndarray` object.

    warn : bool
        If flag is `True` (default), issue a warning when turning on the use
        of the Numpy `matrix` class.  Set `warn` to false to omit display of
        the warning message.

    """
    if flag and warn:
        warnings.warn("Return type numpy.matrix is soon to be deprecated.",
                      stacklevel=2)
    set_defaults('statesp', use_numpy_matrix=flag)

def use_legacy_defaults(version):
    """ Sets the defaults to whatever they were in a given release.

    Parameters
    ----------
    version : string
        version number of the defaults desired. Currently only supports `0.8.3`. 
    """
    if version == '0.8.3': 
        use_numpy_matrix(True) # alternatively: set_defaults('statesp', use_numpy_matrix=True)
        set_defaults('control', default_dt=None)
    else:
        raise ValueError('''version number not recognized. Possible values are: ['0.8.3']''') 