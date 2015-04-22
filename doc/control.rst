.. _function-ref:

==================
Function reference
==================

.. automodule:: control
    :no-members:
    :no-inherited-members:

.. currentmodule:: control

System creation
===============
.. autosummary::
    :toctree: generated/

    ss
    tf
    rss
    drss

Frequency domain plotting
=========================

.. autosummary::
    :toctree: generated/

    bode
    bode_plot
    nyquist
    nyquist_plot
    gangof4
    gangof4_plot
    nichols
    nichols_plot

Time domain simulation
======================

.. autosummary::
    :toctree: generated/

    forced_response
    impulse_response
    initial_response
    step_response
    phase_plot

.. _time-series-convention:

Convention for Time Series
--------------------------

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


Block diagram algebra
=====================
.. autosummary::
    :toctree: generated/

    series
    parallel
    feedback
    negate

Control system analysis
=======================
.. autosummary::
    :toctree: generated/

    dcgain
    evalfr
    freqresp
    margin
    stability_margins
    phase_crossover_frequencies
    pole
    zero
    pzmap
    root_locus

Matrix computations
===================
.. autosummary::
    :toctree: generated/

    care
    dare
    lyap
    dlyap
    ctrb
    obsv
    gram

Control system synthesis
========================
.. autosummary::
    :toctree: generated/

    acker
    lqr
    place

Model simplification tools
==========================
.. autosummary::
    :toctree: generated/

    minreal
    balred
    hsvd
    modred
    era
    markov

Utility functions and conversions
=================================
.. autosummary::
    :toctree: generated/

    unwrap
    db2mag
    mag2db
    isctime
    isdtime
    issys
    pade
    sample_system
    canonical_form
    reachable_form
    ss2tf
    ssdata
    tf2ss
    tfdata
    timebase
    timebaseEqual
