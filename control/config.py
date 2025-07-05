# config.py - package defaults
# RMM, 4 Nov 2012
#
# TODO: add ability to read/write configuration files (a la matplotlib)

"""Functions to access default parameter values.

This module contains default values and utility functions for setting
parameters that control the behavior of the control package.

"""

import collections
import warnings

from .exception import ControlArgument

__all__ = ['defaults', 'set_defaults', 'reset_defaults',
           'use_matlab_defaults', 'use_fbs_defaults',
           'use_legacy_defaults']

# Package level default values
_control_defaults = {
    'control.default_dt': 0,
    'control.squeeze_frequency_response': None,
    'control.squeeze_time_response': None,
    'forced_response.return_x': False,
}


class DefaultDict(collections.UserDict):
    """Default parameters dictionary, with legacy warnings.

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

    # New get function for Python 3.12+ to replicate old behavior
    def get(self, key, defval=None):
        # If the key exists, return it
        if self.__contains__(key):
            return self[key]

        # If not, see if it is deprecated
        repl = self._check_deprecation(key)
        if self.__contains__(repl):
            return self.get(repl, defval)

        # Otherwise, call the usual dict.get() method
        return super().get(key, defval)

    def _check_deprecation(self, key):
        if self.__contains__(f"deprecated.{key}"):
            repl = self[f"deprecated.{key}"]
            warnings.warn(f"config.defaults['{key}'] has been renamed to "
                          f"config.defaults['{repl}'].",
                          FutureWarning, stacklevel=3)
            return repl
        else:
            return key

    #
    # Context manager functionality
    #

    def __call__(self, mapping):
        self.saved_mapping = dict()
        self.temp_mapping = mapping.copy()
        return self

    def __enter__(self):
        for key, val in self.temp_mapping.items():
            if not key in self:
                raise ValueError(f"unknown parameter '{key}'")
            self.saved_mapping[key] = self[key]
            self[key] = val
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, val in self.saved_mapping.items():
            self[key] = val
        del self.saved_mapping, self.temp_mapping
        return None

defaults = DefaultDict(_control_defaults)


def set_defaults(module, **keywords):
    """Set default values of parameters for a module.

    The set_defaults() function can be used to modify multiple parameter
    values for a module at the same time, using keyword arguments.

    Parameters
    ----------
    module : str
        Name of the module for which the defaults are being given.
    **keywords : keyword arguments
        Parameter value assignments.

    Examples
    --------
    >>> ct.defaults['freqplot.number_of_samples']
    1000
    >>> ct.set_defaults('freqplot', number_of_samples=100)
    >>> ct.defaults['freqplot.number_of_samples']
    100
    >>> # do some customized freqplotting

    """
    if not isinstance(module, str):
        raise ValueError("module must be a string")
    for key, val in keywords.items():
        keyname = module + '.' + key
        if keyname not in defaults and f"deprecated.{keyname}" not in defaults:
            raise TypeError(f"unrecognized keyword: {key}")
        defaults[module + '.' + key] = val


# TODO: allow individual modules and individual parameters to be reset
def reset_defaults():
    """Reset configuration values to their default (initial) values.

    Examples
    --------
    >>> ct.defaults['freqplot.number_of_samples']
    1000
    >>> ct.set_defaults('freqplot', number_of_samples=100)
    >>> ct.defaults['freqplot.number_of_samples']
    100

    >>> # do some customized freqplotting
    >>> ct.reset_defaults()
    >>> ct.defaults['freqplot.number_of_samples']
    1000

    """
    # System level defaults
    defaults.update(_control_defaults)

    from .ctrlplot import _ctrlplot_defaults, reset_rcParams
    reset_rcParams()
    defaults.update(_ctrlplot_defaults)

    from .freqplot import _freqplot_defaults, _nyquist_defaults
    defaults.update(_freqplot_defaults)
    defaults.update(_nyquist_defaults)

    from .nichols import _nichols_defaults
    defaults.update(_nichols_defaults)

    from .pzmap import _pzmap_defaults
    defaults.update(_pzmap_defaults)

    from .rlocus import _rlocus_defaults
    defaults.update(_rlocus_defaults)

    from .sisotool import _sisotool_defaults
    defaults.update(_sisotool_defaults)

    from .iosys import _iosys_defaults
    defaults.update(_iosys_defaults)

    from .xferfcn import _xferfcn_defaults
    defaults.update(_xferfcn_defaults)

    from .statesp import _statesp_defaults
    defaults.update(_statesp_defaults)

    from .optimal import _optimal_defaults
    defaults.update(_optimal_defaults)

    from .timeplot import _timeplot_defaults
    defaults.update(_timeplot_defaults)

    from .phaseplot import _phaseplot_defaults
    defaults.update(_phaseplot_defaults)


def _get_param(module, param, argval=None, defval=None, pop=False, last=False):
    """Return the default value for a configuration option.

    The _get_param() function is a utility function used to get the value of a
    parameter for a module based on the default parameter settings and any
    arguments passed to the function.  The precedence order for parameters is
    the value passed to the function (as a keyword), the value from the
    `config.defaults` dictionary, and the default value `defval`.

    Parameters
    ----------
    module : str
        Name of the module whose parameters are being requested.
    param : str
        Name of the parameter value to be determined.
    argval : object or dict
        Value of the parameter as passed to the function.  This can either be
        an object or a dictionary (i.e. the keyword list from the function
        call).  Defaults to None.
    defval : object
        Default value of the parameter to use, if it is not located in the
        `config.defaults` dictionary.  If a dictionary is provided, then
        'module.param' is used to determine the default value.  Defaults to
        None.
    pop : bool, optional
        If True and if argval is a dict, then pop the remove the parameter
        entry from the argval dict after retrieving it.  This allows the use
        of a keyword argument list to be passed through to other functions
        internal to the function being called.
    last : bool, optional
        If True, check to make sure dictionary is empty after processing.

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
        * Frequency plots use the label "Magnitude" for the system gain.

    Examples
    --------
    >>> ct.use_matlab_defaults()
    >>> # do some matlab style plotting

    """
    set_defaults('freqplot', dB=True, deg=True, Hz=False, grid=True)
    set_defaults('freqplot', magnitude_label="Magnitude")


# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    """Use Feedback Systems (FBS) compatible settings.

    The following conventions from `Feedback Systems <https://fbsbook.org>`_
    are used:

        * Bode plots plot gain in powers of ten, phase in degrees,
          frequency in rad/sec, no grid
        * Frequency plots use the label "Gain" for the system gain.
        * Nyquist plots use dashed lines for mirror image of Nyquist curve

    Examples
    --------
    >>> ct.use_fbs_defaults()
    >>> # do some FBS style plotting

    """
    set_defaults('freqplot', dB=False, deg=True, Hz=False, grid=False)
    set_defaults('freqplot', magnitude_label="Gain")
    set_defaults('nyquist', mirror_style='--')


def use_legacy_defaults(version):
    """ Sets the defaults to whatever they were in a given release.

    Parameters
    ----------
    version : string
        Version number of `python-control` to use for setting defaults.

    Examples
    --------
    >>> ct.use_legacy_defaults("0.9.0")
    (0, 9, 0)
    >>> # do some legacy style plotting

    """
    import re
    (major, minor, patch) = (None, None, None)  # default values

    # Early release tag format: REL-0.N
    match = re.match(r"^REL-0.([12])$", version)
    if match: (major, minor, patch) = (0, int(match.group(1)), 0)

    # Early release tag format: control-0.Np
    match = re.match(r"^control-0.([3-6])([a-d])$", version)
    if match: (major, minor, patch) = \
       (0, int(match.group(1)), ord(match.group(2)) - ord('a') + 1)

    # Early release tag format: v0.Np
    match = re.match(r"^[vV]?0\.([3-6])([a-d])$", version)
    if match: (major, minor, patch) = \
       (0, int(match.group(1)), ord(match.group(2)) - ord('a') + 1)

    # Abbreviated version format: vM.N or M.N
    match = re.match(r"^[vV]?([0-9]*)\.([0-9]*)$", version)
    if match: (major, minor, patch) = \
       (int(match.group(1)), int(match.group(2)), 0)

    # Standard version format: vM.N.P or M.N.P
    match = re.match(r"^[vV]?([0-9]*)\.([0-9]*)\.([0-9]*)$", version)
    if match: (major, minor, patch) = \
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # Make sure we found match
    if major is None or minor is None:
        raise ValueError("Version number not recognized. Try M.N.P format.")

    #
    # Go backwards through releases and reset defaults
    #
    reset_defaults()            # start from a clean slate

    # Version 0.10.2:
    if major == 0 and minor < 10 or (minor == 10 and patch < 2):
        from math import inf

        # Reset Nyquist defaults
        set_defaults('nyquist', arrows=2, max_curve_magnitude=20,
                     blend_fraction=0, indent_points=50)

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
        warnings.warn("NumPy matrix class no longer supported")

        # switched to 0 (=continuous) as default timebase
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


def _process_legacy_keyword(kwargs, oldkey, newkey, newval, warn_oldkey=True):
    """Utility function for processing legacy keywords.

    .. deprecated:: 0.10.2
        Replace with `_process_param` or `_process_kwargs`.

    Use this function to handle a legacy keyword that has been renamed.
    This function pops the old keyword off of the kwargs dictionary and
    issues a warning.  If both the old and new keyword are present, a
    `ControlArgument` exception is raised.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments (from function call).
    oldkey : str
        Old (legacy) parameter name.
    newkey : str
        Current name of the parameter.
    newval : object
        Value of the current parameter (from the function signature).
    warn_oldkey : bool
        If set to False, suppress generation of a warning about using a
        legacy keyword.  This is useful if you have two versions of a
        keyword and you want to allow either to be used (see the `cost` and
        `trajectory_cost` keywords in `flatsys.point_to_point` for an
        example of this).

    Returns
    -------
    val : object
        Value of the (new) keyword.

    """
    # TODO: turn on this warning when ready to deprecate
    # warnings.warn(
    #     "replace `_process_legacy_keyword` with `_process_param` "
    #     "or `_process_kwargs`", PendingDeprecationWarning)
    if oldkey in kwargs:
        if warn_oldkey:
            warnings.warn(
                f"keyword '{oldkey}' is deprecated; use '{newkey}'",
                FutureWarning, stacklevel=3)
        if newval is not None:
            raise ControlArgument(
                f"duplicate keywords '{oldkey}' and '{newkey}'")
        else:
            return kwargs.pop(oldkey)
    else:
        return newval


def _process_param(name, defval, kwargs, alias_mapping, sigval=None):
    """Process named parameter, checking aliases and legacy usage.

    Helper function to process function arguments by mapping aliases to
    either their default keywords or to a named argument.  The alias
    mapping is a dictionary that returns a tuple consisting of valid
    aliases and legacy aliases::

       alias_mapping = {
            'argument_name_1': (['alias', ...], ['legacy', ...]),
            ...}

    If `param` is a named keyword in the function signature with default
    value `defval`, a typical calling sequence at the start of a function
    is::

        param = _process_param('param', defval, kwargs, function_aliases)

    If `param` is a variable keyword argument (in `kwargs`), `defval` can
    be passed as either None or the default value to use if `param` is not
    present in `kwargs`.

    Parameters
    ----------
    name : str
        Name of the parameter to be checked.
    defval : object or dict
        Default value for the parameter.
    kwargs : dict
        Dictionary of variable keyword arguments.
    alias_mapping : dict
        Dictionary providing aliases and legacy names.
    sigval : object, optional
        Default value specified in the function signature (default = None).
        If specified, an error will be generated if `defval` is different
        than `sigval` and an alias or legacy keyword is given.

    Returns
    -------
    newval : object
        New value of the named parameter.

    Raises
    ------
    TypeError
        If multiple keyword aliases are used for the same parameter.

    Warns
    -----
    PendingDeprecationWarning
        If legacy name is used to set the value for the variable.

    """
    # Check to see if the parameter is in the keyword list
    if name in kwargs:
        if defval != sigval:
            raise TypeError(f"multiple values for parameter {name}")
        newval = kwargs.pop(name)
    else:
        newval = defval

    # Get the list of aliases and legacy names
    aliases, legacy = alias_mapping[name]

    for kw in legacy:
        if kw in kwargs:
            warnings.warn(
                f"alias `{kw}` is legacy name; use `{name}` instead",
                PendingDeprecationWarning)
            kwval = kwargs.pop(kw)
            if newval != defval and kwval != newval:
                raise TypeError(
                    f"multiple values for parameter `{name}` (via {kw})")
            newval = kwval

    for kw in aliases:
        if kw in kwargs:
            kwval = kwargs.pop(kw)
            if newval != defval and kwval != newval:
                raise TypeError(
                    f"multiple values for parameter `{name}` (via {kw})")
            newval = kwval

    return newval


def _process_kwargs(kwargs, alias_mapping):
    """Process aliases and legacy keywords.

    Helper function to process function arguments by mapping aliases to
    their default keywords.  The alias mapping is a dictionary that returns
    a tuple consisting of valid aliases and legacy aliases::

       alias_mapping = {
            'argument_name_1': (['alias', ...], ['legacy', ...]),
            ...}

    If an alias is present in the dictionary of keywords, it will be used
    to set the value of the argument.  If a legacy keyword is used, a
    warning is issued.

    Parameters
    ----------
    kwargs : dict
        Dictionary of variable keyword arguments.
    alias_mapping : dict
        Dictionary providing aliases and legacy names.

    Raises
    ------
    TypeError
        If multiple keyword aliased are used for the same parameter.

    Warns
    -----
    PendingDeprecationWarning
        If legacy name is used to set the value for the variable.

    """
    for name in alias_mapping or []:
        aliases, legacy = alias_mapping[name]

        for kw in legacy:
            if kw in kwargs:
                warnings.warn(
                    f"alias `{kw}` is legacy name; use `{name}` instead",
                    PendingDeprecationWarning)
                if name in kwargs:
                    raise TypeError(
                        f"multiple values for parameter `{name}` (via {kw})")
                kwargs[name] = kwargs.pop(kw)

        for kw in aliases:
            if kw in kwargs:
                if name in kwargs:
                    raise TypeError(
                        f"multiple values for parameter `{name}` (via {kw})")
                kwargs[name] = kwargs.pop(kw)
