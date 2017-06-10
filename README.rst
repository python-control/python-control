.. image:: https://travis-ci.org/python-control/python-control.svg?branch=master
   :target: https://travis-ci.org/python-control/python-control
.. image:: https://coveralls.io/repos/python-control/python-control/badge.png
   :target: https://coveralls.io/r/python-control/python-control

Python Control Systems Library
==============================

The Python Control Systems Library is a Python module that implements basic
operations for analysis and design of feedback control systems.

Features
--------

- Linear input/output systems in state-space and frequency domain
- Block diagram algebra: serial, parallel, and feedback interconnections
- Time response: initial, step, impulse
- Frequency response: Bode and Nyquist plots
- Control analysis: stability, reachability, observability, stability margins
- Control design: eigenvalue placement, linear quadratic regulator
- Estimator design: linear quadratic estimator (Kalman filter)


Links
=====

- Project home page: http://python-control.sourceforge.net
- Source code repository: https://github.com/python-control/python-control
- Documentation: http://python-control.readthedocs.org/
- Issue tracker: https://github.com/python-control/python-control/issues
- Mailing list: http://sourceforge.net/p/python-control/mailman/


Dependencies
============

The package requires numpy, scipy, and matplotlib.  In addition, some routines
use a module called slycot, that is a Python wrapper around some FORTRAN
routines.  Many parts of python-control will work without slycot, but some
functionality is limited or absent, and installation of slycot is recommended
(see below).  Note that in order to install slycot, you will need a FORTRAN
compiler on your machine.  The Slycot wrapper can be found at:

https://github.com/python-control/Slycot

Installation
============

The package may be installed using pip or distutils.

Pip
---

To install using pip::

  pip install slycot   # optional
  pip install control

Distutils
---------

To install in your home directory, use::

  python setup.py install --user

To install for all users (on Linux or Mac OS)::

  python setup.py build
  sudo python setup.py install


Development
===========

Code
----

You can check out the latest version of the source code with the command::

  git clone https://github.com/python-control/python-control.git

Testing
-------

You can run a set of unit tests to make sure that everything is working
correctly.  After installation, run::

  python setup.py test

License
-------

This is free software released under the terms of `the BSD 3-Clause
License <http://opensource.org/licenses/BSD-3-Clause>`_.  There is no
warranty; not even for merchantability or fitness for a particular
purpose.  Consult LICENSE for copying conditions.

When code is modified or re-distributed, the LICENSE file should
accompany the code or any subset of it, however small.  As an
alternative, the LICENSE text can be copied within files, if so
desired.

Contributing
------------

Your contributions are welcome!  Simply fork the GitHub repository and send a
`pull request`_.

.. _pull request: https://github.com/python-control/python-control/pulls

