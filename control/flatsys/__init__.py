# flatsys/__init__.py: flat systems package initialization file
#
# Author: Richard M. Murray
# Date: 1 Jul 2019

r"""Flat systems subpackage.

This subpackage contains a set of classes and functions to compute
trajectories for differentially flat systems.

A differentially flat system is defined by creating an object using the
`FlatSystem` class, which has member functions for mapping the
system state and input into and out of flat coordinates.  The
`point_to_point` function can be used to create a trajectory
between two endpoints, written in terms of a set of basis functions defined
using the `BasisFamily` class.  The resulting trajectory is return
as a `SystemTrajectory` object and can be evaluated using the
`SystemTrajectory.eval` member function.  Alternatively, the
`solve_flat_optimal` function can be used to solve an optimal control
problem with trajectory and final costs or constraints.

The docstring examples assume that the following import commands::

  >>> import numpy as np
  >>> import control as ct
  >>> import control.flatsys as fs

"""

# Basis function families
from .basis import BasisFamily as BasisFamily
from .bezier import BezierFamily as BezierFamily
from .bspline import BSplineFamily as BSplineFamily
from .poly import PolyFamily as PolyFamily

# Classes
from .systraj import SystemTrajectory as SystemTrajectory
from .flatsys import FlatSystem as FlatSystem
from .flatsys import flatsys as flatsys
from .linflat import LinearFlatSystem as LinearFlatSystem

# Package functions
from .flatsys import point_to_point as point_to_point
from .flatsys import solve_flat_optimal as solve_flat_optimal
from .flatsys import solve_flat_ocp as solve_flat_ocp
