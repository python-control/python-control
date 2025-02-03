.. currentmodule:: control

***************
Developer Notes
***************

This chapter contains notes for developers who wish to contribute to
the Python Control Systems Library (python-control).  It is mainly a
listing of the practices that have evolved over the course of
development since the package was created in 2009.


Package Structure
=================

The python-control package is maintained on GitHub, with documentation
hosted by ReadTheDocs and a mailing list on SourceForge:

  * Project home page: https://python-control.org
  * Source code repository: https://github.com/python-control/python-control
  * Documentation: https://python-control.readthedocs.io/
  * Issue tracker: https://github.com/python-control/python-control/issues
  * Mailing list: https://sourceforge.net/p/python-control/mailman/

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
        statefbk.py, stochsys.py - analysis and synthesis routines

      + ctrlplot.py, descfcn.py, freqplot.py, grid.py,
        nichols.py, pzmap.py, rlocus.py, sisotool.py,
        timeplot.py, timeresp.py - response and plotting routines

      + ctrlutil.py, dtime.py, exception.py, mateqn.py - utility functions

      + phaseplot.py - phase plot module

      + optimal.py - optimal control module

      + **flatsys/** - flat systems subpackage

        - __init__.py, basis.py, bezier.py, bspline.py, flatsys.py,
          linflat.py, poly.py, systraj.py - subpackage files

      + **matlab/** - MATLAB compatibility subpackage

        - __init__.py, timeresp.py, wrappers.py - subpackage files

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


Naming Conventions
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
  (`create_statefbk_iosystem`, `find_operating_points`) or what they
  generate (`input_output_response`, `find_operating_point`).

* Some abbreviations and shortened versions are used when names get
  very long (e.g., `create_statefbk_iosystem` instead of
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
  `optimal.solve_optimal_trajectory`.

System-creating commands:

* Commands that create an I/O system should allow the use of the
  following standard parameters:

  - `name`: system name

  - `inputs`, `outputs`, `states`: number or names of inputs, outputs, state

  - `input_prefix`, `output_prefix`, `state_prefix`: change the default
    prefixes used for naming signals.

  - `dt`: set the timebase.  This one takes a bit of care, since if it is
    not specified then it defaults to
    `config.defaults['control.default_dt']`.  This is different than
    setting `dt` = None, so `dt` should always be part of `**kwargs`.

  These keywords can be parsed in a consistent way using the
  `iosys._process_iosys_keywords` function.

System arguments:

* :code:`sys` when an argument is a single input/output system
  (e.g. `bandwidth`).

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

Order of arguments for functions taking inputs, outputs, state, time,
frequency, etc:

* The default order for providing arguments in state space models is
  ``(t, x, u, params)``.  This is the generic order that should be
  used in functions that take signals as parameters, but permuted so
  that required arguments go first, common arguments go next (as
  keywords, in the order listed above if they also work as positional
  arguments), and infrequent arguments go last (in order listed
  above).  For example::

    def model_update(t, x, u, params)
    resp = initial_response(sys, timepts, x0)             # x0 required
    resp = input_output_response(sys, timepts, u, x0=x0)  # u required
    resp = TimeResponseData(
        timepts, outputs, states=states, inputs=inputs)

  In the last command, note that states precedes inputs because not
  all TimeResponseData elements have inputs (e.g., `initial_response`).

* The default order for providing arguments in the frequency domain is
  system/response first, then frequency::

    resp = frequency_response(sys, omega)
    sys_frd = frd(sys_tf, omega)
    sys = frd(response, omega)

Time and frequency responses:

* Use `timepts` for lists of times and `omega` for lists of
  frequencies at which systems are evaluated.  For example::

    ioresp = ct.input_output_response(sys, timepts, U)
    cplt = ct.bode_plot(sys, omega)

* Use `inputs`, `outputs`, `states`, :code:`time` for time response
  data attributes.  These should be used as parameter names when
  creating `TimeResponseData` objects and also as attributes when
  retrieving response data (with dimensions dependent on `squeeze`
  processing).  These are stored internally in non-squeezed form using
  `u`, `y`, `x`, and `t`, but the internal data should generally not
  be accessed directly.  For example::

    plt.plot(ioresp.time, ioresp.outputs[0])
    tresp = ct.TimeResponseData(time, outputs, states, ...)  # (internal call)

  - Note that the use of `inputs`, `outputs`, and `states` for both
    factory function specifications as well as response function
    attributes is a bit confusing.

* Use `frdata`, `omega` for frequency response data attributes.  These
  should be used as parameter names when creating
  `FrequencyResponseData` objects and also as attributes when
  retrieving response data.  The `frdata` attribute is stored as a 3D
  array indexed by outputs, inputs, frequency.

* Use `complex`, `magnitude`, `phase` for frequency response
  data attributes with squeeze processing.   For example::

    ax = plt.subplots(2, 1)
    ax[0].loglog(fresp.omega, fresp.magnitude)
    ax[1].semilogx(fresp.omega, fresp.phase)

  - The frequency response is stored internally in non-squeezed form
    as `fresp`, but this is generally not accessed directly by users.

  - Note that when creating time response data the independent
    variable (time) is the first argument whereas for frequency
    response data the independent variable (omega) is the second
    argument.  This is because we also create frequency response data
    from a linear system using a call ``frd(sys, omega)``, and
    rename frequency response data using a call ``frd(sys,
    name='newname')``, so the system/data need to be the first
    argument.  For time response data we use the convention that we
    start with time and then list the arguments in the most frequently
    used order.

* Use `response` or `resp` for generic response objects (time,
  frequency, describing function, Nyquist, etc).

  - Note that when responses are evaluated as tuples, the ordering of
    the dependent and independent variables switches between time and
    frequency domain::

      t, y = ct.step_response(sys)
      mag, phase, omega = ct.frequency_response(sys)

    To avoid confusion, it is better to use response objects::

      tresp = ct.step_response(sys)
      t, y = tresp.time, tresp.outputs

      fresp = ct.frequency_response(sys)
      omega, response = fresp.omega, fresp.response
      mag, phase, omega = fresp.magnitude, fresp.phase, fresp.omega


Documentation Guidelines
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

The guiding principle used to guide how docstrings are written is
similar to NumPy (as articulated in the `numpydoc style guide
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_):

   A guiding principle is that human readers of the text are given
   precedence over contorting docstrings so our tools produce nice
   output. Rather than sacrificing the readability of the docstrings,
   we have written pre-processors to assist Sphinx in its task.

To that end, docstrings in `python-control` should use the following
guidelines:

* Use single backticks around all Python objects. The Sphinx
  configuration file (`doc/conf.py`) defines `default_role` to be
  `py:obj`, so everything in a single backtick will be rendered in
  code form and linked to the appropriate documentation if it exists.

  - Note: consistent with numpydoc recommendations, parameters names
    for functions should be in single backticks, even though they
    don't generate a link (but the font will still be OK).

  - The `doc/_static/custom.css` file defines the style for Python
    objects and is configured so that linked objects will appear in a
    bolder type, so that it is easier to see what things you can click
    on to get more information.

  - By default, the string \`sys\` in docstrings would normally
    generate a link to the :mod:`sys` Python module.  To avoid this,
    `conf.py` includes code that converts \`sys\` in docstrings to
    \:code\:\`sys`, which renders as :code:`sys` (code style, with no
    link).  In ``.rst`` files this construction should be done
    manually, since ``.rst`` files are not pre-processed as a
    docstring.

* Use double backticks for inline code, such as a Python code fragments.

  - In principle single backticks might actually work OK given the way
    that the `py:obj` processing works in Sphinx, but the inclusion of
    code is somewhat rare and the extra two backticks seem like a
    small sacrifice (and far from a "contortion").

* Avoid the use of backticks and \:math\: for simple formulas where
  the additional annotation or formatting does not add anything.  For
  example "-c <= x <= c" (without the double quotes) in
  `relay_hysteresis_nonlinearity`.

  - Some of these formulas might be interpreted as Python code
    fragments, but they only need to be in double quotes if that makes
    the documentation easier to understand.

  - Examples:

      * \`dt\` > 0 not \`\`dt > 0\`\` (`dt` is a parameter)
      * \`squeeze\` = True not \`\`squeeze = True\`\` nor squeeze = True.
      * -c <= x <= c not \`\`-c <= x <= c\`\` nor \:math\:\`-c \\leq x
        \\leq c`.
      * \:math\:\`|x| < \\epsilon\` (becomes :math:`|x| < \epsilon`)

* Built-in Python objects (True, False, None) should be written with no
  backticks and should be properly capitalized.

  - Another possibility here is to use a single backtick around
    built-in objects, and the `py:obj` processing will then generate a
    link back to the primary Python documentation.  That seems
    distracting for built-ins like `True`, `False` and `None` (written
    here in single backticks) and using double backticks looks fine in
    Sphinx (``True``, ``False``, ``None``), but seemed to cross the
    "contortions" threshold.

* Strings used as arguments to parameters should be in single
  (forward) ticks ('eval', 'rows', etc) and don't need to be rendered
  as code if just listed as part of a docstring.

  - The rationale here is similar to built-ins: adding 4 backticks
    just to get them in a code font seems unnecessary.

  - Note that if a string is included in Python assignment statement
    (e.g., ``method='slycot'``) it looks quite ugly in text form to
    have it enclosed in double backticks (\`\`method='slycot'\`\`), so
    OK to use method='slycot' (no backticks) or `method` = 'slycot'
    (backticks with extra spaces).

* References to the `defaults` dictionary should be of the form
  \`config.defaults['module.param']\` (like a parameter), which
  renders as `config.defaults['module.param']` in Sphinx.

  - It would be nice to have the term show up as a link to the
    documentation for that parameter (in the
    :ref:`package-configuration-parameters` section of the Reference
    Manual), but the special processing to do that hasn't been
    implemented.

  - Depending on placement, you can end up with lots of white space
    around defaults parameters (also true in the docstrings).

* Math formulas can be written as plain text unless the require
  special symbols (this is consistent with numpydoc) or include Python
  code.  Use the ``:math:`` directive to handle symbols.

Examples of different styles:

* Single backticks to a a function: `interconnect`

* Single backticks to a parameter (no link): `squeeze`

* Double backticks to a code fragment: ``subsys = sys[i][j]``.

* Built-in Python objects: True, False, None

* Defaults parameter: `config.defaults['control.squeeze_time_response']`

* Inline math: :math:`\eta = m \xi + \beta`


Function docstrings
-------------------

Follow numpydoc format with the following additional details:

* All functions should have a short (< 64 character) summary line that
  starts with a capital letter and ends with a period.

* All parameter descriptions should start with a capital letter and
  end with a period.  An exception is parameters that have a list of
  possible values, in which case a phrase sending in a colon (:)
  followed by a list (without punctuation) is OK.

* All parameters and keywords must be documented.  The
  `docstrings_test.py` unit test tries to flag as many of these as
  possible.

* Include an "Examples" section for all non-trivial functions, in a
  form that can be checked by running `make doctest` in the `doc`
  directory.  This is also part of the CI checks.

For functions that return a named tuple, bundle object, or class
instance, the return documentation should include the primary elements
of the return value::

  Returns
  -------
  resp : `TimeResponseData`
      Input/output response data object.  When accessed as a tuple, returns
      ``time, outputs`` (default) or ``time, outputs, states`` if
      `return_states` is True.  The `~TimeResponseData.plot` method can be
      used to create a plot of the time response(s) (see `time_response_plot`
      for more information).
  resp.time : array
      Time values of the output.
  resp.outputs : array
      Response of the system.  If the system is SISO and `squeeze` is not
      True, the array is 1D (indexed by time).  If the system is not SISO or
      `squeeze` is False, the array is 2D (indexed by output and time).
  resp.states : array
      Time evolution of the state vector, represented as a 2D array indexed by
      state and time.
  resp.inputs : array
      Input(s) to the system, indexed by input and time.


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

* Attributes that are created within a class and that might be of
  interest to the user should be documented in the "Attributes"
  section of the class docstring.

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
divided into sections whose names appear in the index on the left of
the web page when that chapter is being viewed.  In some cases a
section may be in its own file, included in the chapter page by using
the `include` directive (see `nlsys.py` for an example).

Sphinx files guidelines:

* Each file should declare the `currentmodule` at or near the top of
  the file.  Except for subpackages (`control.flatsys`) and modules
  that need to be imported separately (`control.optimal`),
  `currentmodule` should be set to control.

* When possible, sample code in the User Guide should use Sphinx
  doctest directives so that the code is executed by `make doctest`.
  Two styles are possible: doctest-style blocks (showing code with a
  prompt and the expected response) and code blocks (using the
  `testcode` directive).

* When referring to the python-control package, several different forms
  can be used:

  - Full name: "the Python Control Systems Library (python-control)"
    (used sparingly, mainly at the tops of chapters).

  - Adjective form: "the python-control package" or "a python-control
    module" (this is the most common form).

  - Noun form: "`python-control`" (only used occasionally).

* Unlike docstrings, the documentation in the User Guide should use
  backticks and \:math\: more liberally when it is appropriate to
  highlight/format code properly.  However, Python built-ins should
  still just be written as True, False, and None (no backticks), for
  formatting consistency.

  - The Sphinx documentation is not read in "raw" form, so OK to add
    the additional annotations.

  - The Python built-ins occur frequently and are capitalized, and so
    the additional formatting doesn't add much and would be
    inconsistent if you jump from the User Guide to the Reference
    Manual (e.g., to look at a function more closely via a link in the
    User Guide).


Reference Manual
----------------

The Reference Manual should provide a fairly comprehensive description
of every class, function, and configuration variable in the package.
All primary functions and classes bust be included here, since the
Reference Manual generates the stub files used by Sphinx.


Modules and subpackages
-----------------------

When documenting (independent) modules and subpackages (refereed to
here collectively as modules), use the following guidelines for
documentation:

* In module docstrings, refer to module functions and classes without
  including the module prefix.  This will let Sphinx set up the links
  to the functions in the proper way and has the advantage that it
  keeps the docstrings shorter.

* Objects in the parent (`control`) package should be referenced using
  the `~control` prefix, so that Sphinx generates the links properly
  (otherwise it only looks within the package).

* In the User Guide, set ``currentmodule`` to ``control`` and refer to
  the module objects using the prefix `~prefix` in the text portions
  of the document but `px` (shortened prefix) in the code sections.
  This will let users copy and past code from the examples and is
  consistent with the use of the `ct` short prefix.  Since this is in
  the User Guide, the additional characters are not as big an issue.

* If you include an `autosummary` of functions in the User Guide
  section, list the functions using the regular prefix (without ``~``)
  to remind everyone the function is in a module.

* When referring to a module function or class in a docstring or User
  Guide section that is not part of the module, use the fully
  qualified function or class (\'prefix.function\').

The main overarching principle should be to make sure that references
to objects that have more detailed information should show up as a
link, not as code.


Utility Functions
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


Sample Files
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
