"""lti.py

The lti module contains the Lti parent class to the child classes StateSpace and
TransferFunction.  It is designed for use in the python-control library.

Routines in this module:

Lti.__init__

"""

class Lti:

    """Lti is a parent class to linear time invariant control (LTI) objects.
    
    Lti is the parent to the StateSpace and TransferFunction child classes. It
    contains the number of inputs and outputs, but this can be expanded in the
    future.

    The StateSpace and TransferFunction child classes contain several common
    "virtual" functions.  These are:

    __init__
    copy
    __str__
    __neg__
    __add__
    __radd__
    __sub__
    __rsub__
    __mul__
    __rmul__
    __div__
    __rdiv__
    evalfr
    freqresp
    pole
    zero
    feedback
    returnScipySignalLti
    
    """
    
    def __init__(self, inputs=1, outputs=1):
        """Assign the LTI object's numbers of inputs and ouputs."""

        # Data members common to StateSpace and TransferFunction.
        self.inputs = inputs
        self.outputs = outputs

# Check to see if two timebases are equal
def timebaseEqual(dt1, dt2):
    # TODO: add docstring
    if (type(dt1) == bool or type(dt2) == bool):
        # Make sure both are unspecified discrete timebases
        return type(dt1) == type(dt2) and dt1 == dt2
    elif (type(dt1) == None or type(dt2) == None):
        # One or the other is unspecified => the other can be anything
        return True
    else:
        return dt1 == dt2
