============
Introduction
============

Welcome to the Python Control Systems Toolbox (python-control) User's
Manual.  This manual contains information on using the python-control
package, including documentation for all functions in the package and
examples illustrating their use.

Overview of the Toolbox
-----------------------

The python-control package is a set of python classes and functions that
implement common operations for the analysis and design of feedback control
systems.  The initial goal is to implement all of the functionality required
to work through the examples in the textbook `Feedback Systems
<http://www.cds.caltech.edu/~murray/FBSwiki>`_ by Astrom and Murray. A
MATLAB compatibility package (control.matlab) is available that provides
many of the common functions corresponding to commands available in the
MATLAB Control Systems Toolbox.

In addition to the documentation here, there is a project wiki that
contains some additional information about how to use the package
(including some detailed worked examples):

  http://python-control.sourceforge.net

Some Differences from MATLAB
----------------------------
The python-control package makes use of NumPy and SciPy.  A list of
general differences between NumPy and MATLAB can be found here:

  http://www.scipy.org/NumPy_for_Matlab_Users

In terms of the python-control package more specifically, here are
some thing to keep in mind:

* You must include commas in vectors.  So [1 2 3] must be [1, 2, 3].
* Functions that return multiple arguments use tuples
* You cannot use braces for collections; use tuples instead
* Transfer functions are currently only implemented for SISO systems; use
  state space representations for MIMO systems.

Getting Started
---------------
1. Download latest release from http://sf.net/projects/python-control/files.

2. Untar the source code in a temporary directory and run 'python setup.py
   install' to build and install the code

3. To see if things are working correctly, run ipython -pylab and run
   the script 'examples/secord-matlab.py'.  This should generate a
   step response, Bode plot and Nyquist plot for a simple second order
   system.  (For more detailed tests, run nosetests in the main directory.)

4. To see the commands that are available, run the following commands in
   ipython::

       >>> import control
       >>> ?control

5. If you want to have a MATLAB-like environment for running the control
   toolbox, use::

       >>> from control.matlab import *
       >>> ?control.matlab
