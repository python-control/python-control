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

.. image:: classes.pdf
  :width: 800

Additional classes
==================
.. autosummary::
   :template: custom-class-template.rst
   :nosignatures:

   DescribingFunctionNonlinearity
   DescribingFunctionResponse
   flatsys.BasisFamily
   flatsys.FlatSystem
   flatsys.LinearFlatSystem
   flatsys.PolyFamily
   flatsys.SystemTrajectory
   optimal.OptimalControlProblem
   optimal.OptimalControlResult
   optimal.OptimalEstimationProblem
   optimal.OptimalEstimationResult

The use of these classes is described in more detail in the
:ref:`flatsys-module` module and the :ref:`optimal-module` module
