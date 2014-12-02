#! TODO: add module docstring
# delay.py - functions involving time delays
#
# Author: Sawyer Fuller
# Date: 26 Aug 2010
# 
# This file contains functions for implementing time delays (currently
# only the pade() function).
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

# Python 3 compatability (needs to go here)
from __future__ import print_function

def pade(T, n=1):
    """ 
    Create a linear system that approximates a delay.
    
    Return the numerator and denominator coefficients of the Pade approximation.
    
    Parameters
    ----------
    T : number
        time delay
    n : integer
        order of approximation
        
    Returns
    -------       
    num, den : array
        Polynomial coefficients of the delay model, in descending powers of s.
    
    Notes
    -----
    Based on an algorithm in Golub and van Loan, "Matrix Computation" 3rd.
    Ed. pp. 572-574.
    """
    if T == 0:
        num = [1,]
        den = [1,]
    else:
        num = [0. for i in range(n+1)]
        num[-1] = 1.
        den = [0. for i in range(n+1)]
        den[-1] = 1.
        c = 1.
        for k in range(1, n+1):
            c = T * c * (n - k + 1)/(2 * n - k + 1)/k
            num[n - k] = c * (-1)**k
            den[n - k] = c 
        num = [coeff/den[0] for coeff in num]
        den = [coeff/den[0] for coeff in den]
    return num, den 
