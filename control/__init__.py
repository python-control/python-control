# __init__.py - initialization for control systems toolbox
#
# Initial author: Richard M. Murray
# Creation date: 24 May 2009
# Use `git shortlog -n -s` for full list of contributors

"""The Python Control Systems Library (python-control) provides common
functions for analyzing and designing feedback control systems.

The initial goal for the package is to implement all of the
functionality required to work through the examples in the textbook
`Feedback Systems <https://fbsbook.org>`_ by Astrom and Murray.  In
addition to standard techniques available for linear control systems,
support for nonlinear systems (including trajectory generation, gain
scheduling, phase plane diagrams, and describing functions) is
included.  A :ref:`matlab-module` is available that provides many of
the common functions corresponding to commands available in the MATLAB
Control Systems Toolbox.

Documentation is available in two forms: docstrings provided with the code,
and the python-control User Guide, available from the `python-control
homepage <https://www.python-control.org>`_.

The docstring examples assume the following import commands::

  >>> import numpy as np
  >>> import control as ct

Available subpackages
---------------------

The main control package includes the most common functions used in
analysis, design, and simulation of feedback control systems.  Several
additional subpackages and modules are available that provide more
specialized functionality:

* :mod:`~control.flatsys`: Differentially flat systems
* :mod:`~control.matlab`: MATLAB compatibility module
* :mod:`~control.optimal`: Optimization-based control
* :mod:`~control.phaseplot`: 2D phase plane diagrams

These subpackages and modules are described in more detail in the
subpackage and module docstrings and in the User Guide.

"""

# Import functions from within the control system library
# Note: the functions we use are specified as __all__ variables in the modules

# don't warn about `import *`
# ruff: noqa: F403
# don't warn about unknown names; they come via `import *`
# ruff: noqa: F405

# Input/output system modules
from .iosys import *
from .nlsys import *
from .lti import *
from .statesp import *
from .xferfcn import *
from .frdata import *

# Time responses and plotting
from .timeresp import *
from .timeplot import *

from .bdalg import *
from .ctrlplot import *
from .delay import *
from .descfcn import *
from .dtime import *
from .freqplot import *
from .margins import *
from .mateqn import *
from .modelsimp import *
from .nichols import *
from .phaseplot import *
from .pzmap import *
from .rlocus import *
from .statefbk import *
from .stochsys import *
from .ctrlutil import *
from .canonical import *
from .robust import *
from .config import *
from .sisotool import *
from .passivity import *
from .sysnorm import *

# Allow access to phase_plane functions as ct.phaseplot.fcn or ct.pp.fcn
from . import phaseplot as phaseplot
pp = phaseplot

# Exceptions
from .exception import *

# Version information
try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"

# Initialize default parameter values
reset_defaults()
