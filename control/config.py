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

# Set defaults to match MATLAB
def use_matlab_defaults():
    # Bode plot defaults
    global bode_dB; bode_dB = True
    global bode_deg; bode_deg = True
    global bode_Hz; bode_Hz = True

# Set defaults to match FBS (Astrom and Murray)
def use_fbs_defaults():
    # Bode plot defaults
    global bode_dB; bode_dB = False
    global bode_deg; bode_deg = True
    global bode_Hz; bode_Hz = True
