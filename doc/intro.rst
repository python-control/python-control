************
Introduction
************

Welcome to the Python Control Systems Library (python-control) User
Guide.  This guide contains information on using the python-control
package, including documentation for all functions in the package and
examples illustrating their use.

Package overview
================

.. automodule:: control
   :no-members:
   :no-inherited-members:
   :no-special-members:

Installation
============

The `python-control` package can be installed using conda or pip.  The
package requires `NumPy`_ and `SciPy`_, and the plotting routines
require `Matplotlib <https://matplotlib.org>`_.  In addition, some
routines require the `Slycot
<https://github.com/python-control/Slycot>`_ library in order to
implement more advanced features (including some MIMO functionality).

For users with the Anaconda distribution of Python, the following
command can be used::

  conda install -c conda-forge control slycot

This installs `slycot` and `python-control` from conda-forge, including the
`openblas` package.  NumPy, SciPy, and Matplotlib will also be installed if
they are not already present.

.. note::
   Mixing packages from conda-forge and the default conda channel
   can sometimes cause problems with dependencies, so it is usually best to
   install NumPy, SciPy, and Matplotlib from conda-forge as well.

To install using pip::

  pip install slycot   # optional
  pip install control

.. note::
   If you install Slycot using pip you'll need a development
   environment (e.g., Python development files, C, and Fortran compilers).
   Pip installation can be particularly complicated for Windows.

Many parts of `python-control` will work without `slycot`, but some
functionality is limited or absent, and installation of `slycot` is
recommended. Users can check to insure that slycot is installed
correctly by running the command::

  python -c "import slycot"

and verifying that no error message appears. More information on the 
Slycot package can be obtained from the `Slycot project page
<https://github.com/python-control/Slycot>`_.

Alternatively, to install `python-control` from source, first
`download the source code
<https://github.com/python-control/python-control/releases>`_ and
unpack it.  To install in your Python environment, use::

  pip install .

The python-control package can also be used with `Google Colab
<colab.google.com>`_ by including the following lines to import the
control package::

  try:
      import control as ct
      print("python-control", ct.__version__)
  except ImportError:
      !pip install control
      import control as ct

Note that Google colab does not currently support Slycot, so some
functionality may not be available.

Some differences from MATLAB
============================

Users familiar with the MATLAB control systems toolbox will find much
of the functionality implemented in `python-control`, though using
Python constructs and coding conventions.  The python-control package
makes heavy use of `NumPy <http://www.numpy.org>`_ and `SciPy
<https://www.scipy.org>`_ and many differences are reflected in the
use of those .  A list of general differences between NumPy and MATLAB
can be found `here
<https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_.

In terms of the python-control package more specifically, here are
some things to keep in mind:

* Vectors and matrices used as arguments to functions can be written
  using lists, with commas required between elements and column
  vectors implemented as nested list .  So [1 2 3] must be written as
  [1, 2, 3] and matrices are written using 2D nested lists, e.g., [[1,
  2], [3, 4]].
* Functions that return multiple values use either objects (with
  elements for each return value) or tuples.  The number of elements
  in a tuple is fixed and so functions that return variable numbers of
  return values will have a parameter of the form `return_<val>`
  that is used to return additional data.
* You cannot use braces for collections; use tuples instead.
* Time series data have time as the final index (see
  :ref:`time-series-convention`).
