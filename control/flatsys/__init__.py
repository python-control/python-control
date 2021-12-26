# flatsys/__init__.py: flat systems package initialization file
#
# Copyright (c) 2019 by California Institute of Technology
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
# Author: Richard M. Murray
# Date: 1 Jul 2019

r"""The :mod:`control.flatsys` package contains a set of classes and functions
that can be used to compute trajectories for differentially flat systems.

A differentially flat system is defined by creating an object using the
:class:`~control.flatsys.FlatSystem` class, which has member functions for
mapping the system state and input into and out of flat coordinates.  The
:func:`~control.flatsys.point_to_point` function can be used to create a
trajectory between two endpoints, written in terms of a set of basis functions
defined using the :class:`~control.flatsys.BasisFamily` class.  The resulting
trajectory is return as a :class:`~control.flatsys.SystemTrajectory` object
and can be evaluated using the :func:`~control.flatsys.SystemTrajectory.eval`
member function.

"""

# Basis function families
from .basis import BasisFamily
from .poly import PolyFamily
from .bezier import BezierFamily

# Classes
from .systraj import SystemTrajectory
from .flatsys import FlatSystem
from .linflat import LinearFlatSystem

# Package functions
from .flatsys import point_to_point
