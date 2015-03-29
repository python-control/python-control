Creating System Models
**********************

Python-control provides a number of methods for creating LTI control
systems.

.. module:: control

==========================  ============================================
:func:`ss`                  create state-space (SS) models
:func:`tf`                  create transfer function (TF) models
==========================  ============================================

System creation
================
.. autoclass:: control.StateSpace
.. autofunction:: control.ss
.. autoclass:: control.TransferFunction
.. autofunction:: control.tf

Utility functions and converstions
==================================
.. autofunction:: control.drss
.. autofunction:: control.isctime
.. autofunction:: control.isdtime
.. autofunction:: control.issys
.. autofunction:: control.pade
.. autofunction:: control.sample_system
.. autofunction:: control.ss2tf
.. autofunction:: control.ssdata
.. autofunction:: control.tf2ss
.. autofunction:: control.tfdata
.. autofunction:: control.timebase
.. autofunction:: control.timebaseEqual
