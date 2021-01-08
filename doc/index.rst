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

You can run the unit tests with `pytest`_ to make sure that everything is
working correctly.  Inside the source directory, run::

  pytest -v

or to test the installed package::

  pytest --pyargs control -v

.. _pytest: https://docs.pytest.org/

.. rubric:: Contributing

Your contributions are welcome! Simply fork the `GitHub repository <https://github.com/python-control/python-control>`_ and send a
`pull request`_.

.. _pull request: https://github.com/python-control/python-control/pulls

The following details suggested steps for making your own contributions to the project using GitHub

1. Fork on GitHub: login/create an account and click Fork button at the top right corner of https://github.com/python-control/python-control/.

2. Clone to computer (Replace [you] with your Github username)::

    git clone https://github.com/[you]/python-control.git
    cd python_control

3. Set up remote upstream::

    git remote add upstream https://github.com/python-control/python-control.git

4. Start working on a new issue or feature by first creating a new branch with a descriptive name::

    git checkout -b <my-new-branch-name>

5. Write great code. Suggestion: write the tests you would like your code to satisfy before writing the code itself. This is known as test-driven development.

6. Run tests and fix as necessary until everything passes::

    pytest -v

  (for documentation, run ``make html`` in ``doc`` directory)

7. Commit changes::

    git add <changed files>
    git commit -m "commit message"

8. Update & sync your local code to the upstream version on Github before submitting (especially if it has been awhile)::

    git checkout master; git fetch --all; git merge upstream/master; git push

  and then bring those changes into your branch::

    git checkout <my-new-branch-name>; git rebase master

9. Push your branch to GitHub::

    git push origin <my-new-branch-name>

10. Issue pull request to submit your code modifications to Github by going to your fork on Github, clicking Pull Request, and entering a description.
11. Repeat steps 5--9 until feature is complete


.. rubric:: Links

- Issue tracker: https://github.com/python-control/python-control/issues
- Mailing list: http://sourceforge.net/p/python-control/mailman/
