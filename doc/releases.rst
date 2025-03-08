*************
Release Notes
*************

This chapter contains a listing of the major releases of the Python
Control Systems Library (python-control) along with a brief summary of
the significant changes in each release.

The information listed here is primarily intended for users.  More
detailed notes on each release, including links to individual pull
requests and issues, are available on the `python-control GitHub
release page
<https://github.com/python-control/python-control/releases>`_.


Version 0.10
============

Version 0.10 of the python-control package introduced the
``_response/_plot`` pattern, described in more detail in
:ref:`response-chapter`, in which input/output system responses
generate an object representing the response that can then be used for
plotting (via the ``.plot()`` method) or other uses.  Significant
changes were also made to input/output system functionality, including
the ability to index systems and signal using signal labels.

.. toctree::
   :maxdepth: 1

   releases/0.10.2-notes
   releases/0.10.1-notes
   releases/0.10.0-notes


Version 0.9
===========

Version 0.9 of the python-control package included significant
upgrades the the `interconnect` functionality to allow automatic
signal interconnetion and the introduction of an :ref:`optimal control
module <optimal-module>` for optimal trajectory generation.  In
addition, the default timebase for I/O systems was set to 0 in Version
0.9 (versus None in previous versions).

.. toctree::
   :maxdepth: 1

   releases/0.9.4-notes
   releases/0.9.3-notes
   releases/0.9.2-notes
   releases/0.9.1-notes
   releases/0.9.0-notes


Earlier Versions
================

Summary release notes are included for these collections of early
releases of the python-control package.

.. toctree::
   :maxdepth: 1

   releases/0.8.x-notes
   releases/0.3-7.x-notes
