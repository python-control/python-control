# stateSpace.py - state space class for control systems library
#
# Author: Richard M. Murray
# Date: 24 May 09
# 
# This file contains the MIMO class, which is used to represent
# linear systems in state space.  This is the primary representation
# for the control system library.
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
# $Id: statesp.py 816 2009-05-29 21:06:27Z murray $

import scipy as sp
import scipy.signal as signal

#
# MIMO class
#
# The MIMO class is used throughout the control systems library to
# represent systems in statespace form.  This class is derived from
# the ltisys class defined in the scipy.signal package, allowing many
# of the functions that already existing in that package to be used
# directly.
#
class MIMO(signal.lti):
    """The MIMO class is used to represent linear input/output systems.
    """
    # Style to use for printing
    def __str__(self):
        str =  "A = " + self.A.__str__() + "\n\n"
        str += "B = " + self.B.__str__() + "\n\n"
        str += "C = " + self.C.__str__() + "\n\n"
        str += "D = " + self.D.__str__() + "\n"
        return str

    # Method for generating the frequency response of the system
    def freqresp(self, omega=None):
        # Generate and save a transfer function matrix
        #! This is currently limited to SISO systems
        nout, nin = self.D.shape

        # Compute the denominator from the A matrix
        den = sp.poly1d(sp.poly(self.A))

        # Compute the numerator based on zeros
        #! This is currently limited to SISO systems
        num = sp.poly1d(\
            sp.poly(self.A - sp.dot(self.B, self.C)) + (self.D[0] - 1) * den)

        # Generate the frequency response at each frequency
        fresp = map(lambda w: num(w*1j) / den(w*1j), omega)
        mag = sp.sqrt(sp.multiply(fresp, sp.conjugate(fresp)))
        phase = sp.angle(fresp)

        return mag, phase, omega

    # Method for evaluating a system at one frequency
    def evalfr(self, freq):
        #! Not implemented
        return None
