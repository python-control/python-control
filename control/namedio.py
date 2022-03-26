# namedio.py - internal named I/O object class
# RMM, 13 Mar 2022
#
# This file implements the _NamedIOSystem class, which is used as a parent
# class for FrequencyResponseData, InputOutputSystem, LTI, TimeResponseData,
# and other similar classes to allow naming of signals.

import numpy as np


class _NamedIOSystem(object):
    _idCounter = 0

    def _name_or_default(self, name=None):
        if name is None:
            name = "sys[{}]".format(_NamedIOSystem._idCounter)
            _NamedIOSystem._idCounter += 1
        return name

    def __init__(
            self, name=None, inputs=None, outputs=None, states=None):

        # system name
        self.name = self._name_or_default(name)

        # Parse and store the number of inputs and outputs
        self.set_inputs(inputs)
        self.set_outputs(outputs)
        self.set_states(states)

    #
    # Class attributes
    #
    # These attributes are defined as class attributes so that they are
    # documented properly.  They are "overwritten" in __init__.
    #

    #: Number of system inputs.
    #:
    #: :meta hide-value:
    ninputs = None

    #: Number of system outputs.
    #:
    #: :meta hide-value:
    noutputs = None

    #: Number of system states.
    #:
    #: :meta hide-value:
    nstates = None

    def __repr__(self):
        return str(type(self)) + ": " + self.name if self.name is not None \
            else str(type(self))

    def __str__(self):
        """String representation of an input/output object"""
        str = "Object: " + (self.name if self.name else "(None)") + "\n"
        str += "Inputs (%s): " % self.ninputs
        for key in self.input_index:
            str += key + ", "
        str += "\nOutputs (%s): " % self.noutputs
        for key in self.output_index:
            str += key + ", "
        if self.nstates is not None:
            str += "\nStates (%s): " % self.nstates
            for key in self.state_index:
                str += key + ", "
        return str

    # Find a signal by name
    def _find_signal(self, name, sigdict):
        return sigdict.get(name, None)

    def set_inputs(self, inputs, prefix='u'):
        """Set the number/names of the system inputs.

        Parameters
        ----------
        inputs : int, list of str, or None
            Description of the system inputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `inputs` is an integer, create the names of the states using
            the given prefix (default = 'u').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.ninputs, self.input_index = \
            _process_signal_list(inputs, prefix=prefix)

    def find_input(self, name):
        """Find the index for an input given its name (`None` if not found)"""
        return self.input_index.get(name, None)

    # Property for getting and setting list of input signals
    input_labels = property(
        lambda self: list(self.input_index.keys()),     # getter
        set_inputs)                                     # setter

    def set_outputs(self, outputs, prefix='y'):
        """Set the number/names of the system outputs.

        Parameters
        ----------
        outputs : int, list of str, or None
            Description of the system outputs.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `outputs` is an integer, create the names of the states using
            the given prefix (default = 'y').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.noutputs, self.output_index = \
            _process_signal_list(outputs, prefix=prefix)

    def find_output(self, name):
        """Find the index for an output given its name (`None` if not found)"""
        return self.output_index.get(name, None)

    # Property for getting and setting list of output signals
    output_labels = property(
        lambda self: list(self.output_index.keys()),     # getter
        set_outputs)                                     # setter

    def set_states(self, states, prefix='x'):
        """Set the number/names of the system states.

        Parameters
        ----------
        states : int, list of str, or None
            Description of the system states.  This can be given as an integer
            count or as a list of strings that name the individual signals.
            If an integer count is specified, the names of the signal will be
            of the form `u[i]` (where the prefix `u` can be changed using the
            optional prefix parameter).
        prefix : string, optional
            If `states` is an integer, create the names of the states using
            the given prefix (default = 'x').  The names of the input will be
            of the form `prefix[i]`.

        """
        self.nstates, self.state_index = \
            _process_signal_list(states, prefix=prefix)

    def find_state(self, name):
        """Find the index for a state given its name (`None` if not found)"""
        return self.state_index.get(name, None)

    # Property for getting and setting list of state signals
    state_labels = property(
        lambda self: list(self.state_index.keys()),     # getter
        set_states)                                     # setter

    def issiso(self):
        """Check to see if a system is single input, single output"""
        return self.ninputs == 1 and self.noutputs == 1

    def _isstatic(self):
        """Check to see if a system is a static system (no states)"""
        return self.nstates == 0


# Utility function to parse a list of signals
def _process_signal_list(signals, prefix='s'):
    if signals is None:
        # No information provided; try and make it up later
        return None, {}

    elif isinstance(signals, (int, np.integer)):
        # Number of signals given; make up the names
        return signals, {'%s[%d]' % (prefix, i): i for i in range(signals)}

    elif isinstance(signals, str):
        # Single string given => single signal with given name
        return 1, {signals: 0}

    elif all(isinstance(s, str) for s in signals):
        # Use the list of strings as the signal names
        return len(signals), {signals[i]: i for i in range(len(signals))}

    else:
        raise TypeError("Can't parse signal list %s" % str(signals))
