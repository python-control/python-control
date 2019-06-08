# config.py - package defaults
# RMM, 4 Nov 2012
#
# This file contains default values and utility functions for setting
# variables that control the behavior of the control package.
# Eventually it will be possible to read and write configuration
# files.  For now, you can just choose between MATLAB and FBS default
# values.

import warnings
import numpy as np
from .statesp import StateSpaceMatrix

# Bode plot defaults
bode_dB = False                 # Bode plot magnitude units
bode_deg = True                 # Bode Plot phase units
bode_Hz = False                 # Bode plot frequency units
bode_number_of_samples = None   # Bode plot number of samples
bode_feature_periphery_decade = 1.0  # Bode plot feature periphery in decades

# State space return type (change to StateSpaceMatrix at next major revision)
ss_return_type = np.matrix


def reset_defaults():
    """Reset configuration values to their default values."""
    global bode_dB; bode_dB = False
    global bode_deg; bode_deg = True
    global bode_Hz; bode_Hz = False
    global bode_number_of_samples; bode_number_of_samples = None
    global bode_feature_periphery_decade; bode_feature_periphery_decade = 1.0
    global ss_return_type; ss_return_type = np.matrix   # TODO: update


# Set defaults to match MATLAB
def use_matlab_defaults():
    """
    Use MATLAB compatible configuration settings

    The following conventions are used:
        * Bode plots plot gain in dB, phase in degrees, frequency in Hertz
    """
    # Bode plot defaults
    global bode_dB; bode_dB = True
    global bode_deg; bode_deg = True
    global bode_Hz; bode_Hz = True


# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    """
    Use `Feedback Systems <http://fbsbook.org>`_ (FBS) compatible settings

    The following conventions are used:
        * Bode plots plot gain in powers of ten, phase in degrees,
          frequency in Hertz
    """
    # Bode plot defaults
    global bode_dB; bode_dB = False
    global bode_deg; bode_deg = True
    global bode_Hz; bode_Hz = False

#
# State space function return type
#
# These functions are used to set the return type for state space functions
# that return a matrix.  In the original version of python-control, these
# functions returned a numpy.matrix object.  To handle the eventual
# deprecation of the numpy.matrix type, the StateSpaceMatrix object type,
# which implements matrix multiplications and related numpy.matrix operations
# was created.  To avoid breaking existing code, the return type from
# functions that used to return an np.matrix object now call the
# get_ss_return_type function to obtain the return type.  The return type can
# also be set using the `return_type` keyword, overriding the default state
# space matrix return type.
#


# Set state space return type
def set_ss_return_type(subtype, warn=True):
    global ss_return_type
    # If return_type is np.matrix, issue a pending deprecation warning
    if (subtype is np.matrix and warn):
        warnings.warn("Return type numpy.matrix is soon to be deprecated.",
                      stacklevel=2)
    ss_return_type = subtype


# Get the state space return type
def get_ss_return_type(subtype=None, warn=True):
    global ss_return_type
    return_type = ss_return_type if subtype is None else subtype
    # If return_type is np.matrix, issue a pending deprecation warning
    if (return_type is np.matrix and warn):
        warnings.warn("Returning numpy.matrix, soon to be deprecated; "
                      "make sure calling code can handle nparray.",
                      stacklevel=2)
    return return_type


# Function to turn on/off use of np.matrix type
def use_numpy_matrix(flag=True, warn=True):
    if flag and warn:
        warnings.warn("Return type numpy.matrix is soon to be deprecated.",
                      stacklevel=2)
        set_ss_return_type(np.matrix, warn=False)
    else:
        set_ss_return_type(StateSpaceMatrix)
