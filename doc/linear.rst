.. module:: control

********************************************
Linear System Modeling, Analysis, and Design
********************************************

Linear time invariant (LTI) systems are represented in python-control in
state space, transfer function, or frequency response data (FRD) form.  Most
functions in the toolbox will operate on any of these data types, and
functions for converting between compatible types are provided.

Creating LTI systems
====================

State space systems
-------------------
The :class:`StateSpace` class is used to represent state-space realizations
of linear time-invariant (LTI) systems:

.. math::

  \frac{dx}{dt} &= A x + B u \\
  y &= C x + D u

where u is the input, y is the output, and x is the state.

To create a state space system, use the :func:`ss` function::

  sys = ct.ss(A, B, C, D)

State space systems can be manipulated using standard arithmetic operations
as well as the :func:`feedback`, :func:`parallel`, and :func:`series`
function.  A full list of functions can be found in :ref:`function-ref`.

Systems, inputs, outputs, and states can be given labels to allow more
customized access to system information::

  sys = ct.ss(
      A, B, C, D, name='sys',
      states=['x1', 'x2'], inputs=['u'], outputs=['y'])

State space can be manipulated using standard arithmetic operations as
well as the :func:`feedback`, :func:`parallel`, and :func:`series`
function.  A full list of functions can be found in
:ref:`function-ref`.

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

To create a transfer function, use the :func:`tf` function::

  sys = ct.tf(num, den)

The system name as well as input and output labels can be specified in
the same way as state space systems::

  sys = ct.tf(A, B, C, D, name='sys', inputs=['u'], outputs=['y'])

Transfer functions can be manipulated using standard arithmetic operations
as well as the :func:`feedback`, :func:`parallel`, and :func:`series`
function.  A full list of functions can be found in :ref:`function-ref`.

Frequency response data (FRD) systems
-------------------------------------
The :class:`FrequencyResponseData` (FRD) class is used to represent systems in
frequency response data form.

The main data members are `omega` and `fresp`, where `omega` is a 1D array
with the frequency points of the response, and `fresp` is a 3D array, with
the first dimension corresponding to the output index of the system, the
second dimension corresponding to the input index, and the 3rd dimension
corresponding to the frequency points in omega.

FRD systems can be created with the :func:`frd` factory function.
Frequency response data systems have a somewhat more limited set of
functions that are available, although all of the standard algebraic
manipulations can be performed.

The FRD class is also used as the return type for the
:func:`frequency_response` function (and the equivalent method for the
:class:`StateSpace` and :class:`TransferFunction` classes).  This
object can be assigned to a tuple using::

    mag, phase, omega = response

where `mag` is the magnitude (absolute value, not dB or log10) of the
system frequency response, `phase` is the wrapped phase in radians of
the system frequency response, and `omega` is the (sorted) frequencies
at which the response was evaluated.  If the system is SISO and the
`squeeze` argument to :func:`frequency_response` is not True,
`magnitude` and `phase` are 1D, indexed by frequency.  If the system
is not SISO or `squeeze` is False, the array is 3D, indexed by the
output, input, and frequency.  If `squeeze` is True then
single-dimensional axes are removed.  The processing of the `squeeze`
keyword can be changed by calling the response function with a new
argument::

    mag, phase, omega = response(squeeze=False)

Frequency response objects are also available as named properties of the
``response`` object: ``response.magnitude``, ``response.phase``, and
``response.response`` (for the complex response).  For MIMO systems, these
elements of the frequency response can be accessed using the names of the
inputs and outputs::

  response.magnitude['y[0]', 'u[1]']

where the signal names are based on the system that generated the frequency
response.

Note: The ``fresp`` data member is stored as a NumPy array and cannot be
accessed with signal names.  Use ``response.response`` to access the
complex frequency response using signal names.

Multi-input, multi-output (MIMO) systems
----------------------------------------

.. todo:: Add information on building MIMO systems

Subsets of input/output pairs for LTI systems can be obtained by indexing
the system using either numerical indices (including slices) or signal
names::

    subsys = sys[[0, 2], 0:2]
    subsys = sys[['y[0]', 'y[2]'], ['u[0]', 'u[1]']]

Signal names for an indexed subsystem are preserved from the original
system and the subsystem name is set according to the values of
``config.defaults['iosys.indexed_system_name_prefix']`` and
``config.defaults['iosys.indexed_system_name_suffix']``.  The default
subsystem name is the original system name with '$indexed' appended.

.. include:: statesp.rst

.. include:: xferfcn.rst

Discrete time systems
=====================

.. todo:: add anchor for time base (?)

A discrete time system is created by specifying a nonzero 'timebase', dt.
The timebase argument can be given when a system is constructed:

* `dt = 0`: continuous time system (default)
* `dt > 0`: discrete time system with sampling period 'dt'
* `dt = True`: discrete time with unspecified sampling period
* `dt = None`: no timebase specified

Systems must have compatible timebases in order to be combined. A discrete
time system with unspecified sampling time (`dt = True`) can be combined with
a system having a specified sampling time; the result will be a discrete time
system with the sample time of the latter system.  Similarly, a system with
timebase `None` can be combined with a system having a specified timebase; the
result will have the timebase of the latter system. For continuous time
systems, the :func:`sample_system` function or the :meth:`StateSpace.sample`
and :meth:`TransferFunction.sample` methods can be used to create a discrete
time system from a continuous time system.  See
:ref:`utility-and-conversions`. The default value of `dt` can be changed by
changing the value of `config.defaults['control.default_dt']`.


Model conversion and reduction
===============================

Conversion between representations
----------------------------------
LTI systems can be converted between representations either by calling the
constructor for the desired data type using the original system as the sole
argument or using the explicit conversion functions :func:`ss2tf` and
:func:`tf2ss`.

Model reduction
---------------

.. automodule:: modelsimp
