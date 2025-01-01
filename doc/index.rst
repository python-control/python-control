##############################
Python Control Systems Library
##############################

The Python Control Systems Library (`python-control`) is a Python package that
implements basic operations for analysis and design of feedback control systems.

.. rubric:: Features

- Linear input/output systems in state space and frequency domain
- Nonlinear input/output system modeling, simulation, and analysis
- Block diagram algebra: serial, parallel, and feedback interconnections
- Time response: initial, step, impulse, and forced response
- Frequency response: Bode, Nyquist, and Nichols plots
- Control analysis: stability, reachability, observability, stability
  margins, phase plane plots, root locus plots
- Control design: eigenvalue placement, LQR, H2, Hinf, and MPC/RHC
- Trajectory generation: optimal control and differential flatness
- Model reduction: balanced realizations and Hankel singular values
- Estimator design: linear quadratic estimator (Kalman filter), MLE, and MHE

.. rubric:: Links:

- GitHub repository: https://github.com/python-control/python-control
- Issue tracker: https://github.com/python-control/python-control/issues
- Mailing list: http://sourceforge.net/p/python-control/mailman/

.. rubric:: How to cite

An `article <https://ieeexplore.ieee.org/abstract/document/9683368>`_
about the library is available on IEEE Explore. If the Python Control
Systems Library helped you in your research, please cite::

  @inproceedings{python-control2021,
    title={The Python Control Systems Library (python-control)},
    author={Fuller, Sawyer and Greiner, Ben and Moore, Jason and
            Murray, Richard and van Paassen, Ren{\'e} and Yorke, Rory},
    booktitle={60th IEEE Conference on Decision and Control (CDC)},
    pages={4875--4881},
    year={2021},
    organization={IEEE}
  }

or the GitHub site: https://github.com/python-control/python-control.

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :numbered: 2

   intro
   Tutorial <examples/python-control_tutorial.ipynb>
   Linear systems <linear>
   I/O response and plotting <response>
   Nonlinear systems <nonlinear>
   Interconnected I/O systems <iosys>
   Stochastic systems <stochastic>
   examples
   genindex

.. toctree::
   :caption: Reference Manual
   :maxdepth: 1

   functions
   classes
   config
   matlab

***********
Development
***********

You can check out the latest version of the source code with the command::

  git clone https://github.com/python-control/python-control.git

You can run the unit tests with `pytest`_ to make sure that everything is
working correctly.  Inside the source directory, run::

  pytest -v

or to test the installed package::

  pytest --pyargs control -v

.. _pytest: https://docs.pytest.org/

Your contributions are welcome!  Simply fork the `GitHub repository <https://github.com/python-control/python-control>`_ and send a
`pull request`_.

.. _pull request: https://github.com/python-control/python-control/pulls

Please see the `Developer's Wiki`_ for detailed instructions.

.. _Developer's Wiki: https://github.com/python-control/python-control/wiki
