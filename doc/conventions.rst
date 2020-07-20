.. _conventions-ref:

.. currentmodule:: control

*******************
Library conventions
*******************

The python-control library uses a set of standard conventions for the way
that different types of standard information used by the library.

LTI system representation
=========================

Linear time invariant (LTI) systems are represented in python-control in
state space, transfer function, or frequency response data (FRD) form.  Most
functions in the toolbox will operate on any of these data types and
functions for converting between compatible types is provided.

State space systems
-------------------
The :class:`StateSpace` class is used to represent state-space realizations
of linear time-invariant (LTI) systems:

.. math::

  \frac{dx}{dt} &= A x + B u \\
  y &= C x + D u

where u is the input, y is the output, and x is the state.

To create a state space system, use the :class:`StateSpace` constructor:

  sys = StateSpace(A, B, C, D)

State space systems can be manipulated using standard arithmetic operations
as well as the :func:`feedback`, :func:`parallel`, and :func:`series`
function.  A full list of functions can be found in :ref:`function-ref`.

Transfer functions
------------------
The :class:`TransferFunction` class is used to represent input/output
transfer functions

.. math::

  G(s) = \frac{\text{num}(s)}{\text{den}(s)}
       = \frac{a_0 s^m + a_1 s^{m-1} + \cdots + a_m}
              {b_0 s^n + b_1 s^{n-1} + \cdots + b_n},

where n is generally greater than or equal to m (for a proper transfer
function).

To create a transfer function, use the :class:`TransferFunction`
constructor:

  sys = TransferFunction(num, den)

Transfer functions can be manipulated using standard arithmetic operations
as well as the :func:`feedback`, :func:`parallel`, and :func:`series`
function.  A full list of functions can be found in :ref:`function-ref`.

FRD (frequency response data) systems
-------------------------------------
The :class:`FrequencyResponseData` (FRD) class is used to represent systems in
frequency response data form.

The main data members are `omega` and `fresp`, where `omega` is a 1D array
with the frequency points of the response, and `fresp` is a 3D array, with
the first dimension corresponding to the output index of the FRD, the second
dimension corresponding to the input index, and the 3rd dimension
corresponding to the frequency points in omega.

FRD systems have a somewhat more limited set of functions that are
available, although all of the standard algebraic manipulations can be
performed.

Discrete time systems
---------------------
A discrete time system is created by specifying a nonzero 'timebase', dt.
The timebase argument can be given when a system is constructed:

* dt = 0: continuous time system (default)
* dt > 0: discrete time system with sampling period 'dt'
* dt = True: discrete time with unspecified sampling period
* dt = None: no timebase specified 

Only the :class:`StateSpace`, :class:`TransferFunction`, and
:class:`InputOutputSystem` classes allow explicit representation of
discrete time systems.

Systems must have compatible timebases in order to be combined. A discrete time 
system with unspecified sampling time (`dt = True`) can be combined with a system 
having a specified sampling time; the result will be a discrete time system with the sample time of the latter
system.  Similarly, a system with timebase `None` can be combined with a system having a specified
timebase; the result will have the timebase of the latter system. For continuous 
time systems, the :func:`sample_system` function or the :meth:`StateSpace.sample` and :meth:`TransferFunction.sample` methods
can be used to create a discrete time system from a continuous time system.
See :ref:`utility-and-conversions`. The default value of 'dt' can be changed by
changing the value of ``control.config.defaults['control.default_dt']``.

Conversion between representations
----------------------------------
LTI systems can be converted between representations either by calling the
constructor for the desired data type using the original system as the sole
argument or using the explicit conversion functions :func:`ss2tf` and
:func:`tf2ss`.

.. currentmodule:: control
.. _time-series-convention:

Time series data
================
A variety of functions in the library return time series data: sequences of
values that change over time.  A common set of conventions is used for
returning such data: columns represent different points in time, rows are
different components (e.g., inputs, outputs or states).  For return
arguments, an array of times is given as the first returned argument,
followed by one or more arrays of variable values.  This convention is used
throughout the library, for example in the functions
:func:`forced_response`, :func:`step_response`, :func:`impulse_response`,
and :func:`initial_response`.

.. note::
    The convention used by python-control is different from the convention
    used in the `scipy.signal
    <https://docs.scipy.org/doc/scipy/reference/signal.html>`_ library. In
    Scipy's convention the meaning of rows and columns is interchanged.
    Thus, all 2D values must be transposed when they are used with functions
    from `scipy.signal`_.

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

    t, y = step_response(sys)
    plot(t, y)

The output of a MIMO system can be plotted like this::

    t, y, x = forced_response(sys, u, t)
    plot(t, y[0], label='y_0')
    plot(t, y[1], label='y_1')

The convention also works well with the state space form of linear systems. If
``D`` is the feedthrough *matrix* of a linear system, and ``U`` is its input
(*matrix* or *array*), then the feedthrough part of the system's response,
can be computed like this::

    ft = D * U

Package configuration parameters
================================

The python-control library can be customized to allow for different default
values for selected parameters.  This includes the ability to set the style
for various types of plots and establishing the underlying representation for
state space matrices.

To set the default value of a configuration variable, set the appropriate
element of the `control.config.defaults` dictionary:

.. code-block:: python

    control.config.defaults['module.parameter'] = value

The `~control.config.set_defaults` function can also be used to set multiple
configuration parameters at the same time:

.. code-block:: python

    control.config.set_defaults('module', param1=val1, param2=val2, ...]

Finally, there are also functions available set collections of variables based
on standard configurations.

Selected variables that can be configured, along with their default values:

  * bode.dB (False): Bode plot magnitude plotted in dB (otherwise powers of 10)
    
  * bode.deg (True): Bode plot phase plotted in degrees (otherwise radians)
    
  * bode.Hz (False): Bode plot frequency plotted in Hertz (otherwise rad/sec)
    
  * bode.grid (True): Include grids for magnitude and phase plots
    
  * freqplot.number_of_samples (None): Number of frequency points in Bode plots
    
  * freqplot.feature_periphery_decade (1.0): How many decades to include in the
    frequency range on both sides of features (poles, zeros).
    
  * statesp.use_numpy_matrix (True): set the return type for state space matrices to
    `numpy.matrix` (verus numpy.ndarray)

  * statesp.default_dt and xferfcn.default_dt (None): set the default value of dt when 
  constructing new LTI systems

  * statesp.remove_useless_states (True): remove states that have no effect on the 
  input-output dynamics of the system 

Additional parameter variables are documented in individual functions

Functions that can be used to set standard configurations:

.. autosummary::
    :toctree: generated/

    reset_defaults
    use_fbs_defaults
    use_matlab_defaults
    use_numpy_matrix
    use_legacy_defaults
