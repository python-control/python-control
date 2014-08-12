Python Control System Library
=============================

.. image:: https://travis-ci.org/python-control/python-control.svg?branch=master
    :target: https://travis-ci.org/python-control/python-control
.. image:: https://coveralls.io/repos/python-control/python-control/badge.png
        :target: https://coveralls.io/r/python-control/python-control

RMM, 23 May 09

This directory contains the source code for the Python Control Systems
Library (python-control).  This package provides a library of standard
control system algorithms in the python programming environment.

Installation
------------

Using pip
~~~~~~~~~~~

Pip is a python packaging system. It can be installed on debian based
linux distros with the command::

        sudo apt-get install pip

Pip can then be used to install python-control::

        sudo pip install control


From Source
~~~~~~~~~~~

Standard python package installation::

        python setup.py install

To see if things are working, you can run the script
examples/secord-matlab.py (using ipython -pylab).  It should generate a step
response, Bode plot and Nyquist plot for a simple second order linear
system.

Testing
-------

You can also run a set of unit tests to make sure that everything is working
correctly.  After installation, run::

        python runtests.py

Slycot
------

Routines from the Slycot wrapper are used for providing the
functionality of several routines for state-space, transfer functions
and robust control. Many parts of python-control will still work
without slycot, but some functionality is limited or absent, and
installation of Slycot is definitely recommended.  The Slycot wrapper
can be found at:

https://github.com/jgoppert/Slycot

and can be installed with::

        sudo pip install slycot
