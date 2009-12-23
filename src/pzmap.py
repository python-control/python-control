# pzmap.py - computations involving poles and zeros
#
# Author: Richard M. Murray
# Date: 7 Sep 09
# 
# This file contains functions that compute poles, zeros and related
# quantities for a linear system.
#
# Copyright (c) 2009 by California Institute of Technology
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
# $Id:pzmap.py 819 2009-05-29 21:28:07Z murray $

import matplotlib.pyplot as plt
import scipy as sp
import xferfcn

# Compute poles and zeros for a system
def pzmap(sys, Plot=True):
    if (isinstance(sys, xferfcn.TransferFunction)):
        poles = sp.roots(sys.den);
        zeros = sp.roots(sys.num);
    else:
        raise TypeException

    if (Plot):
        # Plot the locations of the poles and zeros
        plt.plot(sp.real(poles), sp.imag(poles), 'x'); plt.hold(True);
        plt.plot(sp.real(zeros), sp.imag(zeros), 'o'); 

        # Add axes
        #! Not implemented

    # Return locations of poles and zeros as a tuple
    return poles, zeros
