.. _iosys-module:

********************
Input/output systems
********************

.. automodule:: control.iosys
   :no-members:
   :no-inherited-members:

Module usage
============

An input/output system is defined as a dynamical system that has a system
state as well as inputs and outputs (either inputs or states can be empty).
The dynamics of the system can be in continuous or discrete time.  To simulate
an input/output system, use the :func:`~control.input_output_response`
function::

  t, y = input_output_response(io_sys, T, U, X0, params)

An input/output system can be linearized around an equilibrium point to obtain
a :class:`~control.StateSpace` linear system.  Use the
:func:`~control.find_eqpt` function to obtain an equilibrium point and the
:func:`~control.linearize` function to linearize about that equilibrium point::

  xeq, ueq = find_eqpt(io_sys, X0, U0)
  ss_sys = linearize(io_sys, xeq, ueq)

Input/output systems can be created from state space LTI systems by using the
:class:`~control.LinearIOSystem` class`::

  io_sys = LinearIOSystem(ss_sys)

Nonlinear input/output systems can be created using the
:class:`~control.NonlinearIOSystem` class, which requires the definition of an
update function (for the right hand side of the differential or different
equation) and and output function (computes the outputs from the state)::

  io_sys = NonlinearIOSystem(updfcn, outfcn, inputs=M, outputs=P, states=N)

More complex input/output systems can be constructed by using the
:class:`~control.InterconnectedSystem` class, which allows a collection of
input/output subsystems to be combined with internal connections between the
subsystems and a set of overall system inputs and outputs that link to the
subsystems::

    steering = ct.InterconnectedSystem(
        (plant, controller), name='system',
        connections=(('controller.e', '-plant.y')),
        inplist=('controller.e'), inputs='r',
        outlist=('plant.y'), outputs='y')

Interconnected systems can also be created using block diagram manipulations
such as the :func:`~control.series`, :func:`~control.parallel`, and
:func:`~control.feedback` functions.  The :class:`~control.InputOutputSystem`
class also supports various algebraic operations such as `*` (series
interconnection) and `+` (parallel interconnection).

Example
=======

To illustrate the use of the input/output systems module, we create a
model for a predator/prey system, following the notation and parameter
values in FBS2e.

We begin by defining the dynamics of the system

.. code-block:: python

  import control
  import numpy as np
  import matplotlib.pyplot as plt

  def predprey_rhs(t, x, u, params):
      # Parameter setup
      a = params.get('a', 3.2)
      b = params.get('b', 0.6)
      c = params.get('c', 50.)
      d = params.get('d', 0.56)
      k = params.get('k', 125)
      r = params.get('r', 1.6)
      
      # Map the states into local variable names
      H = x[0]
      L = x[1]

      # Compute the control action (only allow addition of food)
      u_0 = u if u > 0 else 0
  
      # Compute the discrete updates
      dH = (r + u_0) * H * (1 - H/k) - (a * H * L)/(c + H)
      dL = b * (a * H *  L)/(c + H) - d * L
  
      return [dH, dL]

We now create an input/output system using these dynamics:

.. code-block:: python

  io_predprey = control.NonlinearIOSystem(
      predprey_rhs, None, inputs=('u'), outputs=('H', 'L'),
      states=('H', 'L'), name='predprey')

Note that since we have not specified an output function, the entire state
will be used as the output of the system.

The `io_predprey` system can now be simulated to obtain the open loop dynamics
of the system:

.. code-block:: python

  X0 = [25, 20]                 # Initial H, L
  T = np.linspace(0, 70, 500)   # Simulation 70 years of time
  
  # Simulate the system
  t, y = control.input_output_response(io_predprey, T, 0, X0)
  
  # Plot the response
  plt.figure(1)
  plt.plot(t, y[0])
  plt.plot(t, y[1])
  plt.legend(['Hare', 'Lynx'])
  plt.show(block=False)

We can also create a feedback controller to stabilize a desired population of
the system.  We begin by finding the (unstable) equilibrium point for the
system and computing the linearization about that point.

.. code-block:: python

  eqpt = control.find_eqpt(io_predprey, X0, 0)
  xeq = eqpt[0]                         # choose the nonzero equilibrium point
  lin_predprey = control.linearize(io_predprey, xeq, 0)

We next compute a controller that stabilizes the equilibrium point using
eigenvalue placement and computing the feedforward gain using the number of
lynxes as the desired output (following FBS2e, Example 7.5):

.. code-block:: python

  K = control.place(lin_predprey.A, lin_predprey.B, [-0.1, -0.2])
  A, B = lin_predprey.A, lin_predprey.B
  C = np.array([[0, 1]])                # regulated output = number of lynxes
  kf = -1/(C @ np.linalg.inv(A - B @ K) @ B)

To construct the control law, we build a simple input/output system that
applies a corrective input based on deviations from the equilibrium point.
This system has no dynamics, since it is a static (affine) map, and can
constructed using the `~control.ios.NonlinearIOSystem` class:

.. code-block:: python

  io_controller = control.NonlinearIOSystem(
    None,
    lambda t, x, u, params: -K @ (u[1:] - xeq) + kf * (u[0] - xeq[1]),
    inputs=('Ld', 'u1', 'u2'), outputs=1, name='control')

The input to the controller is `u`, consisting of the vector of hare and lynx
populations followed by the desired lynx population.

To connect the controller to the predatory-prey model, we create an
`InterconnectedSystem`:

.. code-block:: python

  io_closed = control.InterconnectedSystem(
    (io_predprey, io_controller),	# systems
    connections=(
      ('predprey.u', 'control.y[0]'),
      ('control.u1',  'predprey.H'),
      ('control.u2',  'predprey.L')
    ),
    inplist=('control.Ld'),
    outlist=('predprey.H', 'predprey.L', 'control.y[0]')
  )
       
Finally, we simulate the closed loop system:

.. code-block:: python

  # Simulate the system
  t, y = control.input_output_response(io_closed, T, 30, [15, 20])
  
  # Plot the response
  plt.figure(2)
  plt.subplot(2, 1, 1)
  plt.plot(t, y[0])
  plt.plot(t, y[1])
  plt.legend(['Hare', 'Lynx'])
  plt.subplot(2, 1, 2)
  plt.plot(t, y[2])
  plt.legend(['input'])
  plt.show(block=False)

Module classes and functions
============================

Input/output system classes
---------------------------
.. autosummary::
   
   InputOutputSystem
   InterconnectedSystem
   LinearIOSystem
   NonlinearIOSystem

Input/output system functions
-----------------------------
.. autosummary::

   find_eqpt
   linearize
   input_output_response
   ss2io
   tf2io

