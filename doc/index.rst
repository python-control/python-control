##############################
Python Control Systems Library
##############################

The Python Control Systems Library (`python-control`) is a Python package that
implements basic operations for analysis and design of feedback control systems.

.. rubric:: Features

- Linear input/output systems in state-space and frequency domain
- Block diagram algebra: serial, parallel, and feedback interconnections
- Time response: initial, step, impulse
- Frequency response: Bode and Nyquist plots
- Control analysis: stability, reachability, observability, stability margins
- Control design: eigenvalue placement, LQR, H2, Hinf
- Model reduction: balanced realizations, Hankel singular values
- Estimator design: linear quadratic estimator (Kalman filter)

.. rubric:: Documentation

.. toctree::
   :maxdepth: 2

   intro
   conventions
   control
   classes
   matlab
   flatsys
   iosys
   examples

* :ref:`genindex`

.. rubric:: Development

You can check out the latest version of the source code with the command::

  git clone https://github.com/python-control/python-control.git

You can run a set of unit tests to make sure that everything is working
correctly.  After installation, run::

  python setup.py test

Your contributions are welcome!  Simply fork the `GitHub repository <https://github.com/python-control/python-control>`_ and send a
`pull request`_.

.. _pull request: https://github.com/python-control/python-control/pulls

.. rubric:: Links

- Issue tracker: https://github.com/python-control/python-control/issues
- Mailing list: http://sourceforge.net/p/python-control/mailman/
