.. _class-ref:
.. currentmodule:: control

**********************
Control system classes
**********************

Input/output system classes
===========================

The classes listed below are used to represent models of input/output
systems (both linear time-invariant and nonlinear).  They are usually
created from factory functions such as :func:`tf` and :func:`ss`, so the
user should normally not need to instantiate these directly.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst
   :nosignatures:

   InputOutputSystem
   NonlinearIOSystem
   LTI
   StateSpace
   TransferFunction
   FrequencyResponseData
   InterconnectedSystem
   LinearICSystem

The following figure illustrates the relationship between the classes and
some of the functions that can be used to convert objects from one class to
another:

.. image:: figures/classes.pdf
  :width: 800


Response and plotting classes
=============================

These classes are used as the outputs of `_response`, `_map`, and
`_plot` functions:

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst
   :nosignatures:

   ControlPlot
   FrequencyResponseData
   TimeResponseData
   NyquistResponseData
   PoleZeroData

More informaton on the functions used to create these classes can be
found in the :ref:iosys-module chapter.


Nonlinear system classes
========================

These classes are used for various nonlinear input/output system
operations:

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst
   :nosignatures:

   DescribingFunctionNonlinearity
   DescribingFunctionResponse
   flatsys.BasisFamily
   flatsys.BezierFamily
   flatsys.BSplineFamily
   flatsys.FlatSystem
   flatsys.LinearFlatSystem
   flatsys.PolyFamily
   flatsys.SystemTrajectory
   OperatingPoint
   optimal.OptimalControlProblem
   optimal.OptimalControlResult
   optimal.OptimalEstimationProblem
   optimal.OptimalEstimationResult

More informaton on the functions used to create these classes can be
found in the :ref:nonlinear-systems chapter.
