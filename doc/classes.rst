.. _class-ref:
.. currentmodule:: control

**********************
Control system classes
**********************

The classes listed below are used to represent models of input/output
systems (both linear time-invariant and nonlinear).  They are usually
created from factory functions such as :func:`tf` and :func:`ss`, so the
user should normally not need to instantiate these directly.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   InputOutputSystem
   LTI
   StateSpace
   TransferFunction
   FrequencyResponseData
   NonlinearIOSystem
   InterconnectedSystem
   LinearICSystem

The following figure illustrates the relationship between the classes and
some of the functions that can be used to convert objects from one class to
another:

.. image:: figures/classes.pdf
  :width: 800

Additional classes
==================

.. todo:: Break these up into more useful sections

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst
   :nosignatures:

   ControlPlot
   DescribingFunctionNonlinearity
   DescribingFunctionResponse
   flatsys.BasisFamily
   flatsys.BezierFamily
   flatsys.BSplineFamily
   flatsys.FlatSystem
   flatsys.LinearFlatSystem
   flatsys.PolyFamily
   flatsys.SystemTrajectory
   FrequencyResponseList
   NyquistResponseData
   OperatingPoint
   optimal.OptimalControlProblem
   optimal.OptimalControlResult
   optimal.OptimalEstimationProblem
   optimal.OptimalEstimationResult
   PoleZeroData
   TimeResponseData
   TimeResponseList

The use of these classes is described in more detail in the
:ref:`flatsys-module` module and the :ref:`optimal-module` module
