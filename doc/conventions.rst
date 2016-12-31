.. _conventions-ref:

.. currentmodule:: control

###################
Library conventions
###################

The python-control library uses a set of standard conventions for the way
that different types of standard information used by the library.

.. _time-series-convention:

Time series data
================

This is a convention for function arguments and return values that
represent time series: sequences of values that change over time. It
is used throughout the library, for example in the functions
:func:`forced_response`, :func:`step_response`, :func:`impulse_response`,
and :func:`initial_response`.

.. note::
    This convention is different from the convention used in the library
    :mod:`scipy.signal`. In Scipy's convention the meaning of rows and columns
    is interchanged.  Thus, all 2D values must be transposed when they are
    used with functions from :mod:`scipy.signal`.

Types:

    * **Arguments** can be **arrays**, **matrices**, or **nested lists**.
    * **Return values** are **arrays** (not matrices).

The time vector is either 1D, or 2D with shape (1, n)::

      T = [[t1,     t2,     t3,     ..., tn    ]]

Input, state, and output all follow the same convention. Columns are different
points in time, rows are different components. When there is only one row, a
1D object is accepted or returned, which adds convenience for SISO systems::

      U = [[u1(t1), u1(t2), u1(t3), ..., u1(tn)]
           [u2(t1), u2(t2), u2(t3), ..., u2(tn)]
           ...
           ...
           [ui(t1), ui(t2), ui(t3), ..., ui(tn)]]

      Same for X, Y

So, U[:,2] is the system's input at the third point in time; and U[1] or U[1,:]
is the sequence of values for the system's second input.

The initial conditions are either 1D, or 2D with shape (j, 1)::

     X0 = [[x1]
           [x2]
           ...
           ...
           [xj]]

As all simulation functions return *arrays*, plotting is convenient::

    t, y = step(sys)
    plot(t, y)

The output of a MIMO system can be plotted like this::

    t, y, x = lsim(sys, u, t)
    plot(t, y[0], label='y_0')
    plot(t, y[1], label='y_1')

The convention also works well with the state space form of linear systems. If
``D`` is the feedthrough *matrix* of a linear system, and ``U`` is its input
(*matrix* or *array*), then the feedthrough part of the system's response,
can be computed like this::

    ft = D * U

Package configuration
=====================

The python-control library can be customized to allow for different plotting
conventions.  The currently configurable options allow the units for Bode
plots to be set as dB for gain, degrees for phase and Hertz for frequency
(MATLAB conventions) or the gain can be given in magnitude units (powers of
10), corresponding to the conventions used in `Feedback Systems
<http://www.cds.caltech.edu/~murray/FBSwiki>`_.

Variables that can be configured, along with their default values:
  * bode_dB (False): Bode plot magnitude plotted in dB (otherwise powers of 10)
  * bode_deg (True): Bode plot phase plotted in degrees (otherwise radians)
  * bode_Hz (False): Bode plot frequency plotted in Hertz (otherwise rad/sec)
  * bode_number_of_samples (None): Number of frequency points in Bode plots
  * bode_feature_periphery_decade (1.0): How many decades to include in the
    frequency range on both sides of features (poles, zeros). 

Functions that can be used to set standard configurations:

.. autosummary::
    :toctree: generated/

    use_fbs_defaults
    use_matlab_defaults
