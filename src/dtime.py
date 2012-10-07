"""dtime.py

Utility functions for disrete time systems.

Routines in this module:

isdtime()
isctime()
timebase()
"""

"""Copyright (c) 2012 by California Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the California Institute of Technology nor
   the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

Author: Richard M. Murray
Date: 6 October 2012

$Id: dtime.py 185 2012-08-30 05:44:32Z murrayrm $

"""

from statesp import StateSpace
from xferfcn import TransferFunction

# Return the timebase of a system
def timebase(sys):
    # TODO: add docstring
    # If we get passed a constant, timebase is None
    if isinstance(sys, (int, float, long, complex)):
        return None

    # Check for a transfer fucntion or state space object
    if isinstance(sys, (StateSpace, TransferFunction)):
        if sys.dt > 0:
            return 'dtime';
        elif sys.dt == 0:
            return 'ctime';
        elif sys.dt == None:
            return None

    # Got pased something we don't recognize or bad timebase
    return False;

# Check to see if a system is a discrete time system
def isdtime(sys, strict=False):
    # TODO: add docstring
    # Check to see if this is a constant
    if isinstance(sys, (int, float, long, complex)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer fucntion or state space object
    if isinstance(sys, (StateSpace, TransferFunction)):
        # Look for dt > 0 or dt == None (if not strict)
        return sys.dt > 0 or (not strict and sys.dt == None)

    # Got possed something we don't recognize
    return False

# Check to see if a system is a continuous time system
def isctime(sys, strict=False):
    # TODO: add docstring
    # Check to see if this is a constant
    if isinstance(sys, (int, float, long, complex)):
        # OK as long as strict checking is off
        return True if not strict else False

    # Check for a transfer fucntion or state space object
    if isinstance(sys, (StateSpace, TransferFunction)):
        # Look for dt == 0 or dt == None (if not strict)
        return sys.dt == 0 or (not strict and sys.dt == None)

    # Got possed something we don't recognize
    return False

        
        
