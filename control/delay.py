# -*-coding: utf-8-*-
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


__all__ = ['pade']

def pade(T, n=1, numdeg=None):
    """
    Create a linear system that approximates a delay.

    Return the numerator and denominator coefficients of the Pade approximation.

    Parameters
    ----------
    T : number
        time delay
    n : positive integer
        degree of denominator of approximation
    numdeg: integer, or None (the default)
            If None, numerator degree equals denominator degree
            If >= 0, specifies degree of numerator
            If < 0, numerator degree is n+numdeg

    Returns
    -------
    num, den : array
        Polynomial coefficients of the delay model, in descending powers of s.

    Notes
    -----
    Based on:
      1. Algorithm 11.3.1 in Golub and van Loan, "Matrix Computation" 3rd.
         Ed. pp. 572-574
      2. M. Vajta, "Some remarks on Padé-approximations",
         3rd TEMPUS-INTCOM Symposium
    """
    if numdeg is None:
        numdeg = n
    elif numdeg < 0:
        numdeg += n

    if not T >= 0:
        raise ValueError("require T >= 0")
    if not n >= 0:
        raise ValueError("require n >= 0")
    if not (0 <= numdeg <= n):
        raise ValueError("require 0 <= numdeg <= n")

    if T == 0:
        num = [1,]
        den = [1,]
    else:
        num = [0. for i in range(numdeg+1)]
        num[-1] = 1.
        cn = 1.
        for k in range(1, numdeg+1):
            # derived from Gloub and van Loan eq. for Dpq(z) on p. 572
            # this accumulative style follows Alg 11.3.1
            cn *= -T * (numdeg - k + 1)/(numdeg + n - k + 1)/k
            num[numdeg-k] = cn

        den = [0. for i in range(n+1)]
        den[-1] = 1.
        cd = 1.
        for k in range(1, n+1):
            # see cn above
            cd *= T * (n - k + 1)/(numdeg + n - k + 1)/k
            den[n-k] = cd

        num = [coeff/den[0] for coeff in num]
        den = [coeff/den[0] for coeff in den]
    return num, den
