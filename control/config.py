# config.py - package defaults
# RMM, 4 Nov 2012
#
# This file contains default values and utility functions for setting
# variables that control the behavior of the control package.
# Eventually it will be possible to read and write configuration
# files.  For now, you can just choose between MATLAB and FBS default
# values.

# Bode plot defaults
bode_dB = False                 # Bode plot magnitude units
bode_deg = True                 # Bode Plot phase units
bode_Hz = False                 # Bode plot frequency units
bode_number_of_samples = None   # Bode plot number of samples
bode_feature_periphery_decade = 1.0  # Bode plot feature periphery in decades

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
    global bode_Hz; bode_Hz = True
