============
Introduction
============

Welcome to the Python-Control project.

The python-control package is a set of python classes and functions
that implement common operations for the analysis and design of
feedback control systems.  The initial goal is to implement all of the
functionality required to work through the examples in the textbook
Feedback Systems by Åström and Murray. A MATLAB compatibility package
(control.matlab) is available that provides functions corresponding to
the commands available in the MATLAB Control Systems Toolbox.

In addition to the documentation here, there is a project wiki that
contains some additional information about how to use the package
(including some detailed worked examples):

  http://python-control.sourceforge.net

Some Differences from MATLAB
----------------------------
* You must include commas in vectors.  So [1 2 3] must be [1, 2, 3].
* Functions that return multiple arguments use tuples
* Can't use braces for collections; use tuples instead
* Transfer functions are only implemented for SISO systems (due to
  limitations in the underlying signals.lti class); use state space
  representations for MIMO systems.
