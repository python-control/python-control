# exception.py - exception definitions for the control package
#
# Author: Richard M. Murray
# Date: 31 May 2010
#
# This file contains definitions of standard exceptions for the control package
#
# Copyright (c) 2010 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$

class ControlSlycot(ImportError):
    """Exception for Slycot import.  Used when we can't import a function
    from the slycot package"""
    pass

class ControlDimension(ValueError):
    """Raised when dimensions of system objects are not correct"""
    pass

class ControlArgument(TypeError):
    """Raised when arguments to a function are not correct"""
    pass

class ControlMIMONotImplemented(NotImplementedError):
    """Function is not currently implemented for MIMO systems"""
    pass

class ControlNotImplemented(NotImplementedError):
    """Functionality is not yet implemented"""
    pass

# Utility function to see if slycot is installed
slycot_installed = None
def slycot_check():
    global slycot_installed
    if slycot_installed is None:
        try:
            import slycot
            slycot_installed = True
        except:
            slycot_installed = False
    return slycot_installed


# Utility function to see if pandas is installed
pandas_installed = None
def pandas_check():
    global pandas_installed
    if pandas_installed is None:
        try:
            import pandas
            pandas_installed = True
        except:
            pandas_installed = False
    return pandas_installed
