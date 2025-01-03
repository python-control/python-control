# template.py - template file for python-control module
# RMM, 3 Jan 2024

"""Template file for python-control module.

This file provides a template that can be used when creating a new
file/module in python-control.  The key elements of a module are included
in this template, following the suggestions in the Developer Guidelines.

The first line of a module file should be the name of the file and a short
description.  The next few lines can contain information about who created
the file (your name/initials and date).  For this file I used the short
version (initials, date), but a longer version would be to do something of
the form::

  # filename.py - short one line description
  #
  # Initial author: Full name
  # Creation date: date the file was created

After the header comments, the next item is the module docstring, which
should be a multi-line comment, like this one.  The first line of the
comment is a one line summary phrase, starting with a capital letter and
ending in a period (often the same as the line at the very top).  The rest
of the docstring is an extended summary (this one is a bit longer than
would be typical).

After the docstring, you should have the following elements (in Python):

  * Package imports, using the `isort -m2` format (library, standard, custom)
  * __all__ command, listing public objects in the file
  * Class definitions (if any)
  * Public function definitions
  * Internal function definitions (starting with '_')
  * Function aliases (short = long_name)

The rest of this file contains examples of these elements.

"""

import warnings                 # Python packages

import numpy as np              # Standard external packages

from . import config            # Other modules/packages in python-control
from .lti import LTI            # Public function or class from a module

__all__ = ['SampleClass', 'sample_function']


class SampleClass():
    """Sample class in the python-control package.

    This is an example of a class definition.  The docstring follows
    numpydoc format.  The first line should be a summary (which will show
    up in `autosummary` entries in the Sphinx documentation) and then an
    extended summary describing what the class does.  Then the usual
    sections, per numpydoc.

    Additional guidelines on what should be listed in the various sections
    can be found in the 'Class docstrings' section of the Developer
    Guidelines.

    Parameters
    ----------
    sys : InputOutputSystem
        Short description of the parameter.

    Attributes
    ----------
    data : array
         Short description of an attribute.

    """
    def __init__(self, sys):
        # No docstring required here
        self.sys = sys          # Parameter passed as argument
        self.data = sys.name    # Attribute created within class

    def sample_method(self, data):
        """Sample method within in a class.

        This is an example of a method within a class.  Document using
        numpydoc format.

        """
        return None


def sample_function(data, option=False, **kwargs):
    """Sample function in the template module.

    This is an example of a public function within the template module.
    This function will usually be placed in the `control` namespace by
    updating `__init.py` to import the function (often by importing the
    entire module).

    Docstring should be in standard numpy doc format.  The extended summary
    (this text) should describe the basic operation of the function, with
    technical details in the "Notes" section.

    Parameters
    ----------
    data : array
         Sample parameter for sample function, with short docstring.
    option : bool, optional
         Optional parameter, with default value `False`.

    Returns
    -------
    out : float
        Short description of the function output.

    Additional Parameters
    ---------------------
    inputs : int, str, or list of str
        Parameters that are less commonly used, in this case a keyword
        parameter.

    See Also
    --------
    function1, function2

    Notes
    -----
    This section can contain a more detailed description of how the system
    works.  OK to include some limited mathematics, either via inline math
    directions for a short formula (like this: ..math: a = b c) or via a
    displayed equation:

    ..math::

        a = \int_0^t f(t) dt

    The trick in the docstring is to write something that looks good in
    pure text format but is also processed by sphinx correctly.

    If you refer to parameters, such as the `data` argument to this
    function, but them in single backticks (which will render them in code
    style in Sphinx).  You should also do this for Python contains like
    `True, `False`, and `None`.

    """
    inputs = kwargs['inputs']
    if option is True:
        return data
    else:
        return None

#
# Internal functions
#
# Functions that are not intended for public use can go anyplace, but I
# usually put them at the bottom of the file (out of the way).  Their name
# should start with an underscore.  Docstrings are optional, but if you
# don't include a docstring, make sure to include comments describing how
# the function works.
#


# Sample internal function to process data
def _internal_function(data):
    return None


# Aliases (short versions of long function names)
sf = sample_function
