.. _class-ref:
.. currentmodule:: control

**********************
Control system classes
**********************

The classes listed below are used to represent models of linear time-invariant
(LTI) systems.  They are usually created from factory functions such as
:func:`tf` and :func:`ss`, so the user should normally not need to instantiate
these directly.
		   
.. autosummary::
   :toctree: generated/

   TransferFunction
   StateSpace
   FRD
   InputOutputSystem

Input/Output system subclasses
==============================

Input/output systems are accessed primarily via a set of subclasses
that allow for linear, nonlinear, and interconnected elements:

.. autosummary::
   :toctree: generated/

   LinearIOSystem
   NonlinearIOSystem
   InterconnectedSystem
