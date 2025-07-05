# Original author: Richard M. Murray
# Creation date: 29 May 09
# Pre-2014 revisions: Kevin K. Chen, Dec 2010

"""MATLAB compatibility subpackage.

This subpackage contains a number of functions that emulate some of
the functionality of MATLAB.  The intent of these functions is to
provide a simple interface to the Python Control Systems Library
(python-control) for people who are familiar with the MATLAB Control
Systems Toolbox (tm).

"""

# Silence unused imports (F401), * imports (F403), unknown symbols (F405)
# ruff: noqa: F401, F403, F405

# Import MATLAB-like functions that are defined in other packages
from scipy.signal import zpk2ss, ss2zpk, tf2zpk, zpk2tf
from numpy import linspace, logspace

# If configuration is not yet set, import and use MATLAB defaults
import sys
if not ('.config' in sys.modules):
    from .. import config
    config.use_matlab_defaults()

# Control system library
from ..statesp import *
from ..xferfcn import *
from ..lti import *
from ..iosys import *
from ..frdata import *
from ..dtime import *
from ..exception import ControlArgument

# Import MATLAB-like functions that can be used as-is
from ..ctrlutil import *
from ..freqplot import gangof4
from ..nichols import nichols
from ..bdalg import *
from ..pzmap import *
from ..statefbk import *
from ..delay import *
from ..modelsimp import *
from ..mateqn import *
from ..margins import margin
from ..rlocus import rlocus
from ..dtime import c2d
from ..sisotool import sisotool
from ..stochsys import lqe, dlqe
from ..nlsys import find_operating_point

# Functions that are renamed in MATLAB
pole, zero = poles, zeros
freqresp = frequency_response
trim = find_operating_point

# Import functions specific to Matlab compatibility package
from .timeresp import *
from .wrappers import *

# Set up defaults corresponding to MATLAB conventions
from ..config import *
use_matlab_defaults()
