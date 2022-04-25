# config.py - package defaults
# RMM, 4 Nov 2012
#
# This file contains default values and utility functions for setting
# variables that control the behavior of the control package.
# Eventually it will be possible to read and write configuration
# files.  For now, you can just choose between MATLAB and FBS default
# values + tweak a few other things.


import collections
import warnings

__all__ = ['defaults', 'set_defaults', 'reset_defaults',
           'use_matlab_defaults', 'use_fbs_defaults',
           'use_legacy_defaults', 'use_numpy_matrix']

# Package level default values
_control_defaults = {
    'control.default_dt': 0,
    'control.squeeze_frequency_response': None,
    'control.squeeze_time_response': None,
    'forced_response.return_x': False,
}


class DefaultDict(collections.UserDict):
    """Map names for settings from older version to their renamed ones.

    If a user wants to write to an old setting, issue a warning and write to
    the renamed setting instead. Accessing the old setting returns the value
    from the new name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(self._check_deprecation(key), value)

    def __missing__(self, key):
        # An old key should never have been set. If it is being accessed
        # through __getitem__, return the value from the new name.
        repl = self._check_deprecation(key)
        if self.__contains__(repl):
            return self[repl]
        else:
            raise KeyError(key)

    def _check_deprecation(self, key):
        if self.__contains__(f"deprecated.{key}"):
            repl = self[f"deprecated.{key}"]
            warnings.warn(f"config.defaults['{key}'] has been renamed to "
                          f"config.defaults['{repl}'].",
                          FutureWarning, stacklevel=3)
            return repl
        else:
            return key


defaults = DefaultDict(_control_defaults)


def set_defaults(module, **keywords):
    """Set default values of parameters for a module.

    The set_defaults() function can be used to modify multiple parameter
    values for a module at the same time, using keyword arguments:

        control.set_defaults('module', param1=val, param2=val)

    """
    if not isinstance(module, str):
        raise ValueError("module must be a string")
    for key, val in keywords.items():
        keyname = module + '.' + key
        if keyname not in defaults and f"deprecated.{keyname}" not in defaults:
            raise TypeError(f"unrecognized keyword: {key}")
        defaults[module + '.' + key] = val


def reset_defaults():
    """Reset configuration values to their default (initial) values."""
    # System level defaults
    defaults.update(_control_defaults)

    from .freqplot import _freqplot_defaults, _nyquist_defaults
    defaults.update(_freqplot_defaults)
    defaults.update(_nyquist_defaults)

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

    from .optimal import _optimal_defaults
    defaults.update(_optimal_defaults)


def _get_param(module, param, argval=None, defval=None, pop=False, last=False):
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
    pop : bool, optional
        If True and if argval is a dict, then pop the remove the parameter
        entry from the argval dict after retreiving it.  This allows the use
        of a keyword argument list to be passed through to other functions
        internal to the function being called.
    last : bool, optional
        If True, check to make sure dictionary is empy after processing.

    """

    # Make sure that we were passed sensible arguments
    if not isinstance(module, str) or not isinstance(param, str):
        raise ValueError("module and param must be strings")

    # Construction the name of the key, for later use
    key = module + '.' + param

    # If we were passed a dict for the argval, get the param value from there
    if isinstance(argval, dict):
        val = argval.pop(param, None) if pop else argval.get(param, None)
        if last and argval:
            raise TypeError("unrecognized keywords: " + str(argval))
        argval = val

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
    set_defaults('freqplot', dB=True, deg=True, Hz=False, grid=True)
    set_defaults('statesp', use_numpy_matrix=True)


# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    """Use `Feedback Systems <http://fbsbook.org>`_ (FBS) compatible settings.

    The following conventions are used:
        * Bode plots plot gain in powers of ten, phase in degrees,
          frequency in rad/sec, no grid
        * Nyquist plots use dashed lines for mirror image of Nyquist curve

    """
    set_defaults('freqplot', dB=False, deg=True, Hz=False, grid=False)
    set_defaults('nyquist', mirror_style='--')


# Decide whether to use numpy.matrix for state space operations
def use_numpy_matrix(flag=True, warn=True):
    """Turn on/off use of Numpy `matrix` class for state space operations.

    Parameters
    ----------
    flag : bool
        If flag is `True` (default), use the deprecated Numpy
        `matrix` class to represent matrices in the `~control.StateSpace`
        class and functions.  If flat is `False`, then matrices are
        represented by a 2D `ndarray` object.

    warn : bool
        If flag is `True` (default), issue a warning when turning on the use
        of the Numpy `matrix` class.  Set `warn` to false to omit display of
        the warning message.

    Notes
    -----
    Prior to release 0.9.x, the default type for 2D arrays is the Numpy
    `matrix` class.  Starting in release 0.9.0, the default type for state
    space operations is a 2D array.
    """
    if flag and warn:
        warnings.warn("Return type numpy.matrix is deprecated.",
                      stacklevel=2, category=DeprecationWarning)
    set_defaults('statesp', use_numpy_matrix=flag)


def use_legacy_defaults(version):
    """ Sets the defaults to whatever they were in a given release.

    Parameters
    ----------
    version : string
        Version number of the defaults desired. Ranges from '0.1' to '0.8.4'.
    """
    import re
    (major, minor, patch) = (None, None, None)  # default values

    # Early release tag format: REL-0.N
    match = re.match("REL-0.([12])", version)
    if match: (major, minor, patch) = (0, int(match.group(1)), 0)

    # Early release tag format: control-0.Np
    match = re.match("control-0.([3-6])([a-d])", version)
    if match: (major, minor, patch) = \
       (0, int(match.group(1)), ord(match.group(2)) - ord('a') + 1)

    # Early release tag format: v0.Np
    match = re.match("[vV]?0.([3-6])([a-d])", version)
    if match: (major, minor, patch) = \
       (0, int(match.group(1)), ord(match.group(2)) - ord('a') + 1)

    # Abbreviated version format: vM.N or M.N
    match = re.match("([vV]?[0-9]).([0-9])", version)
    if match: (major, minor, patch) = \
       (int(match.group(1)), int(match.group(2)), 0)

    # Standard version format: vM.N.P or M.N.P
    match = re.match("[vV]?([0-9]).([0-9]).([0-9])", version)
    if match: (major, minor, patch) = \
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # Make sure we found match
    if major is None or minor is None:
        raise ValueError("Version number not recognized. Try M.N.P format.")

    #
    # Go backwards through releases and reset defaults
    #
    reset_defaults()            # start from a clean slate

    # Version 0.9.2:
    if major == 0 and minor < 9 or (minor == 9 and patch < 2):
        from math import inf

        # Reset Nyquist defaults
        set_defaults('nyquist', indent_radius=0.1, max_curve_magnitude=inf,
                     max_curve_offset=0, primary_style=['-', '-'],
                     mirror_style=['--', '--'], start_marker_size=0)

    # Version 0.9.0:
    if major == 0 and minor < 9:
        # switched to 'array' as default for state space objects
        set_defaults('statesp', use_numpy_matrix=True)

        # switched to 0 (=continuous) as default timestep
        set_defaults('control', default_dt=None)

        # changed iosys naming conventions
        set_defaults('iosys', state_name_delim='.',
                     duplicate_system_name_prefix='copy of ',
                     duplicate_system_name_suffix='',
                     linearized_system_name_prefix='',
                     linearized_system_name_suffix='_linearized')

        # turned off _remove_useless_states
        set_defaults('statesp', remove_useless_states=True)

        # forced_response no longer returns x by default
        set_defaults('forced_response', return_x=True)

        # time responses are only squeezed if SISO
        set_defaults('control', squeeze_time_response=True)

        # switched mirror_style of nyquist from '-' to '--'
        set_defaults('nyquist', mirror_style='-')

    return (major, minor, patch)
