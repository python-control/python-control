************
Introduction
************

Welcome to the Python Control Systems Toolbox (python-control) User's
Manual.  This manual contains information on using the python-control
package, including documentation for all functions in the package and
examples illustrating their use.

Overview of the Toolbox
=======================

The python-control package is a set of python classes and functions that
implement common operations for the analysis and design of feedback control
systems.  The initial goal is to implement all of the functionality required
to work through the examples in the textbook `Feedback Systems
<http://www.cds.caltech.edu/~murray/FBSwiki>`_ by Astrom and Murray. A
MATLAB compatibility package (control.matlab) is available that provides
many of the common functions corresponding to commands available in the
MATLAB Control Systems Toolbox.

Some Differences from MATLAB
============================
The python-control package makes use of NumPy and SciPy.  A list of general
differences between NumPy and MATLAB can be found `here
<http://www.scipy.org/NumPy_for_Matlab_Users>`_.

In terms of the python-control package more specifically, here are
some thing to keep in mind:

* You must include commas in vectors.  So [1 2 3] must be [1, 2, 3].
* Functions that return multiple arguments use tuples
* You cannot use braces for collections; use tuples instead

Installation
============

The `python-control` package may be installed using pip, conda or the
standard distutils/setuptools mechanisms.

To install using pip::

  pip install slycot   # optional
  pip install control

Many parts of `python-control` will work without `slycot`, but some
functionality is limited or absent, and installation of `slycot` is
recommended.  

*Note*: the `slycot` library only works on some platforms, mostly
linux-based.  Users should check to insure that slycot is installed
correctly by running the command::

  python -c "import slycot"

and verifying that no error message appears.  It may be necessary to install
`slycot` from source, which requires a working FORTRAN compiler and the
`lapack` library.  More information on the slycot package can be obtained
from the `slycot project page <https://github.com/python-control/Slycot>`_.

For users with the Anaconda distribution of Python, the following
commands can be used::

  conda install numpy scipy matplotlib    # if not yet installed
  conda install -c python-control -c cyclus slycot control

This installs `slycot` and `python-control` from the `python-control`
channel and uses the `cyclus` channel to obtain the required `lapack`
package. 

Alternatively, to use setuptools, first `download the source <https://github.com/python-control/python-control/releases>`_ and unpack
it.  To install in your home directory, use::

  python setup.py install --user

or to install for all users (on Linux or Mac OS)::

  python setup.py build
  sudo python setup.py install

The package requires `numpy` and `scipy`, and the plotting routines require
`matplotlib`.  In addition, some routines require the `slycot` module,
described above.

Getting Started
===============

There are two different ways to use the package.  For the default interface
described in :ref:`function-ref`, simply import the control package as follows::

    >>> import control

If you want to have a MATLAB-like environment, use the :ref:`matlab-module`::

    >>> from control.matlab import *
