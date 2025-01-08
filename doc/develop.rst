.. currentmodule:: control

***************
Developer Notes
***************

This chapter contains notes for developers who wish to contribute to
the Python Control Systems Library (python-control).  It is mainly a
listing of the practices that have evolved over the course of
development since the package was created in 2009.


Package structure
=================

The python-control package is maintained on GitHub, with documentation
hosted by ReadTheDocs and a mailing list on SourceForge:

  * Project home page: http://python-control.org
  * Source code repository: https://github.com/python-control/python-control
  * Documentation: http://python-control.readthedocs.org/
  * Issue tracker: https://github.com/python-control/python-control/issues
  * Mailing list: http://sourceforge.net/p/python-control/mailman/

GitHub repository file and directory layout:
  - **python-control/** - main repository

    * LICENSE, Manifest, pyproject.toml, README.rst - package information

    * **control/** - primary package source code

      + __init__.py, _version.py, config.py - package definition
        and configuration

      + iosys.py, nlsys.py, lti.py, statesp.py, xferfcn.py,
        frdata.py - I/O system classes

      + bdalg.py, delay.py, canonical.py, margins.py,
        sysnorm.py, modelsimp.py, passivity.py, robust.py,
        statefbk.py, stochsys.py - analysis and synthesis routintes

      + ctrlplot.py, descfcn.py, freqplot.py, grid.py,
        nichols.py, pzmap.py, rlocus.py, sisotool.py,
        timeplot.py, timeresp.py - response and plotting rouintes

      + ctrlutil.py, dtime.py, exception.py, mateqn.py - utility funcitons

      + phaseplot.py - phase plot module

      + optimal.py - optimal control module

      + **flatsys/** - flat systems sub-package

        - __init__.py, basis.py, bezier.py, bspline.py,
          flatsys.py, linflat.py, poly.py, systraj.py -
          sub-package files

      + **matlab/** - MATLAB compatibility subpackage

        - __init.py, timeresp.py, wrappers.py - subpackage files

      + **tests/** - unit tests

    * **.github/** - GitHub workflows

    * **benchmarks/** - benchmarking files (not well-maintained)

    * **doc/** - user guide and reference manual

      + index.rst - main documentation index

      + conf.py, Makefile - sphinx configuration files

      + intro.rst, linear.rst, statesp.rst, xferfcn.rst, nonlinear.rst,
        flatsys.rst, iosys.rst, nlsys.rst, optimal.rst, phaseplot.rst,
        response.rst, descfcn.rst, stochastic.rst, examples.rst - User
        Guide

      + functions.rst, classes.rst, config.rst, matlab.rst, develop.rst -
        Reference Manual

      + **examples/**

        - \*.py, \*.rst - Python scripts (linked to ../examples/\*.py)

        - \*.ipynb - Jupyter notebooks (linked to ../examples.ipynb)

      + **figures/**

	- \*.pdf, \*.png - Figures for inclusion in documentation

    * **examples/**

      + \*.py - Python scripts

      + \*.ipynb - Jupyter notebooks


Naming conventions
==================

Generally speaking, standard Python and NumPy naming conventions are
used throughout the package.

* Python PEP 8 (code style): https://peps.python.org/pep-0008/


Filenames
---------

* Source files are lower case, usually less than 10 characters (and 8
  or less is better).

* Unit tests (in `control/tests/`) are of the form `module_test.py` or
  `module_function.py`.


Class names
-----------

* Most class names are in camel case, with long form descriptions of
  the object purpose/contents (`TimeResponseData`).

* Input/output class names are written out in long form as they aren't
  too long (`StateSpace`, `TransferFunction`), but for very long names
  'IO' can be used in place of 'InputOutput' (`NonlinearIOSystem`) and
  'IC' can be used in place of 'Interconnected' (`LinearICSystem`).

* Some older classes don't follow these guidelines (e.g., `LTI` instead
  of `LinearTimeInvariantSystem` or `LTISystem`).


Function names
--------------

* Function names are lower case with words separated by underscores.

* Function names usually describe what they do
  (`create_statefbk_iosys`, `find_operating_points`) or what they
  generate (`input_output_response`, `find_operating_point`).

* Some abbreviations and shortened versions are used when names get
  very long (e.g., `create_statefbk_iosys` instead of
  `create_state_feedback_input_output_system`.

* Factory functions for I/O systems use short names (partly from MATLAB
  conventions, partly because they are pretty frequently used):
  `frd`, `flatsys`, `nlsys`, `ss`, and `tf`.

* Short versions of common commands with longer names are created by
  creating an object with the shorter name as a copy of the main
  object: `bode = bode_plot`, `step = step_response`, etc.

* The MATLAB compatibility library (`control.matlab`) uses names that
  try to line up with MATLAB (e.g., `lsim` instead of `forced_response`).


Parameter names
---------------

Parameter names are not (yet) very uniform across the package.  A few
general patterns are emerging:

* Use longer description parameter names that describe the action or
  role (e.g., `trajectory_constraints` and `print_summary` in
  `optimal.solve_ocp` (which probably should be named
  `optimal.`find_optimal_trajectory`...).

System-creating commands:

* Commands that create an I/O system should allow the use of the
  following standard parameters:

  - `name`: system name

  - `inputs`, `outputs`, `states`: number or names of inputs, outputs, state

  - `input_prefix`, `output_prefix`, `state_prefix`: change the default
    prefixes used for naming signals.

  - `dt`: set the timebase.  This one takes a bit of care, since if it
    is not specified then it defaults to
    `config.defaults['control.default_dt']`.  This is different than
    setting `dt=None`, so you `dt` should always be part of **kwargs.

  These keywords can be parsed in a consistent way using the
  `iosys._process_iosys_keywords` function.

System arguments:

* `sys` when an argument is a single input/output system (e.g. `bandwidth`).

* `syslist` when an argument is a list of systems (e.g.,
  `interconnect`).  A single system should also be OK.

* `sysdata` when an argument can either be a system, a list of
  systems, or data describing a response (e.g, `nyquist_response`).

  .. todo:: For a future release (v 0.11.x?) we should make this more
            consistent across the package.

Signal arguments:

* Factory functions use `inputs`, `outputs`, and `states` to provide
  either the number of each signal or a list of labels for the
  signals.


Documentation guidelines
========================

The python-control package is documented using docstrings and Sphinx.
Reference documentation (class and function descriptions, with details
on parameters) should all go in docstrings.  User documentation in
more narrative form should be in the `.rst` files in `doc/`, where it
can be incorporated into the User Guide.  All significant
functionality should have a narrative description in the User Guide in
addition to docstrings.

Generally speaking, standard Python and NumPy documentation
conventions are used throughout the package:

* Python PEP 257 (docstrings): https://peps.python.org/pep-0257/
* Numpydoc Style guide: https://numpydoc.readthedocs.io/en/latest/format.html


General docstring info
----------------------

* Use single backticks around all Python objects.  Double backticks
  should never be used.  (The `doc/conf.py` file defines
  `default_role` to be `py:obj`, so everything in a single backtick
  will be rendered in code form and linked to the appropriate
  documentation if it exists.)

* All function names, parameter names, and Python objects (`True`,
  `False`, `None`) should be written as code (enclose in backticks).


Function docstrings
-------------------

Follow numpydoc format with the following additional details:

* All functions should have a short (< 64 character) summary line that
  starts with a capital letter and ends with a period.

* All parameter descriptions should start with a capital letter and end
  with a period.  An exception is parameters that have a list of
  possible values, in which case a phrase sending in `:` followed by a
  list (without punctuation) is OK.

* All parameters and keywords must be documented.  The
  `docstrings_test.py` unit test tries to flag as many of these as
  possible.

* Include an "Examples" section for all non-trivial functions, in a
  form that can be checked by running `make doctest` in the `doc`
  directory.  This is also part of the CI checks.


Class docstrings
----------------

Follow numpydoc format with the follow additional details:

* Parameters used in creating an object go in the class docstring and
  not in the `__init__` docstring (which is not included in the
  Sphinx-based documentation).  OK for the `__init__` function to have
  no docstring.

* Parameters that are also attributes only need to be documented once
  (in the "Parameters" or "Additional Parameters" section of the class
  docstring).

* Attributes that are created within a class and might be of interest
  to the user should be documented in the "Attributes" section of the
  class docstring.

* Classes should not include a "Returns" section (since they always
  return an instance of the class).

* Functions and attributes that are not intended to be accessed by
  users should start with an underscore.

I/O system classes:

* Subclasses of `InputOutputSystem` should always have a factory
  function that is used to create them.  The class documentation only
  needs to document the required parameters; the full list of
  parameters (and optional keywords) can and should be documented in
  the factory function docstring.


User Guide
----------

The purpose of the User Guide is provide a *narrative* description of
the key functions of the package.  It is not expected to cover every
command, but should allow someone who knows about control system
design to get up and running quickly.

The User Guide consists of chapters that are each their own separate
`.rst` file and each of them generates a separate page.  Chapters are
divided into sections whose names appear in the indexo on the left of
the web page when that chapter is being viewed.  In some cases a
section may be in its own file, including in the chapter page by using
the `include` directive (see `nlsys.py` for an example).

Sphinx files guidlines:

* Each file should declare the `currentmodule` at or near the top of
  the file.  Except for sub-packages (`control.flatsys`) and modules
  that need to be imported separately (`control.optimal`),
  `currentmodule` should be set to control.


Reference Manual
----------------

The reference manual should provide a fairly comprehensive description
of every class, function and configuration variable in the package.


Utility functions
=================

The following utility functions can be used to help with standard
processing and parsing operations:

.. autosummary::
   :toctree: generated/

   config._process_legacy_keyword
   exception.cvxopt_check
   exception.pandas_check
   exception.slycot_check
   iosys._process_iosys_keywords
   statesp._convert_to_statespace
   xferfcn._convert_to_transfer_function


Sample files
============


Code template
-------------

The following file is a template for a python-control module.  It can
be found in `python-control/doc/examples/template.py`.

.. literalinclude:: examples/template.py
   :language: python
   :linenos:


Documentation template
----------------------

The following file is a template for a documentation file.  It can be
found in `python-control/doc/examples/template.rst`.

.. literalinclude:: examples/template.rst
   :language: text
   :linenos:
   :lines: 3-
