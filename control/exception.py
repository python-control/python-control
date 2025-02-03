# exception.py - exception definitions for the control package
#
# Initial author: Richard M. Murray
# Creation date: 31 May 2010

"""Exception definitions for the control package."""

class ControlSlycot(ImportError):
    """Slycot import failed."""
    pass

class ControlDimension(ValueError):
    """Raised when dimensions of system objects are not correct."""
    pass

class ControlArgument(TypeError):
    """Raised when arguments to a function are not correct."""
    pass

class ControlIndexError(IndexError):
    """Raised when arguments to an indexed object are not correct."""
    pass

class ControlMIMONotImplemented(NotImplementedError):
    """Function is not currently implemented for MIMO systems."""
    pass

class ControlNotImplemented(NotImplementedError):
    """Functionality is not yet implemented."""
    pass

# Utility function to see if Slycot is installed
slycot_installed = None
def slycot_check():
    """Return True if Slycot is installed, otherwise False."""
    global slycot_installed
    if slycot_installed is None:
        try:
            import slycot  # noqa: F401
            slycot_installed = True
        except:
            slycot_installed = False
    return slycot_installed


# Utility function to see if pandas is installed
pandas_installed = None
def pandas_check():
    """Return True if pandas is installed, otherwise False."""
    global pandas_installed
    if pandas_installed is None:
        try:
            import pandas  # noqa: F401
            pandas_installed = True
        except:
            pandas_installed = False
    return pandas_installed

# Utility function to see if cvxopt is installed
cvxopt_installed = None
def cvxopt_check():
    """Return True if cvxopt is installed, otherwise False."""
    global cvxopt_installed
    if cvxopt_installed is None:
        try:
            import cvxopt  # noqa: F401
            cvxopt_installed = True
        except:
            cvxopt_installed = False
    return cvxopt_installed
