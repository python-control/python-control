# config.py - package defaults
# RMM, 4 Nov 2012
#
# This file contains default values and utility functions for setting
# variables that control the behavior of the control package.
# Eventually it will be possible to read and write configuration
# files.  For now, you can just choose between MATLAB and FBS default
# values + tweak a few other things.

import warnings

__all__ = ['reset_defaults', 'use_matlab_defaults', 'use_fbs_defaults',
           'use_numpy_matrix']

# Dictionary of default values
defaults = {
    'bode.dB':False, 'bode.deg':True, 'bode.Hz':False, 'bode.grid':True,
    'freqplot.feature_periphery_decades':1, 'freqplot.number_of_samples':None,
    'statesp.use_numpy_matrix':True,
}


def reset_defaults():
    """Reset configuration values to their default values."""
    defaults['bode.dB'] = False
    defaults['bode.deg'] =  True
    defaults['bode.Hz'] = False
    defaults['bode.grid'] = True
    defaults['freqplot.number_of_samples'] = None
    defaults['freqplot.feature_periphery_decades'] = 1.0
    defaults['statesp.use_numpy_matrix'] = True


# Set defaults to match MATLAB
def use_matlab_defaults():
    """Use MATLAB compatible configuration settings.

    The following conventions are used:
        * Bode plots plot gain in dB, phase in degrees, frequency in
          Hertz, with grids

    """
    # Bode plot defaults
    from .freqplot import bode_plot
    defaults['bode.dB'] = True
    defaults['bode.deg'] = True
    defaults['bode.Hz'] = True
    defaults['bode.grid'] = True
    defaults['statesp.use_numpy_matrix'] = True


# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    """Use `Feedback Systems <http://fbsbook.org>`_ (FBS) compatible settings.

    The following conventions are used:

        * Bode plots plot gain in powers of ten, phase in degrees,
          frequency in Hertz, no grid

    """
    # Bode plot defaults
    from .freqplot import bode_plot
    defaults['bode.dB'] = False
    defaults['bode.deg'] = True
    defaults['bode.Hz'] = False
    defaults['bode.grid'] = False


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
    defaults['statesp.use_numpy_matrix'] = flag
