.. _obc-module:

**************************
Optimization-based control
**************************

.. automodule:: control.obc
   :no-members:
   :no-inherited-members:

Optimal control problem setup
=============================

Consider now the *optimal control problem*:

.. math::

  \min_{u(\cdot)} 
  \int_0^T L(x,u)\, dt + V \bigl( x(T) \bigr)

subject to the constraint

.. math::

  \dot x = f(x, u), \qquad x\in\mathbb{R}^n,\, u\in\mathbb{R}^m.

Abstractly, this is a constrained optimization problem where we seek a
*feasible trajectory* :math:`(x(t), u(t))` that minimizes the cost function

.. math::

  J(x, u) = \int_0^T L(x,u)\, dt + V \bigl( x(T) \bigr).

More formally, this problem is equivalent to the "standard" problem of
minimizing a cost function :math:`J(x, u)` where :math:`(x, u) \in L_2[0,T]`
(the set of square integrable functions) and :math:`h(z) = \dot x(t) -
f(x(t), u(t)) = 0` models the dynamics.  The term :math:`L(x, u)` is
referred to as the integral (or trajectory) cost and :math:`V(x(T))` is the
final (or terminal) cost.

It is often convenient to ask that the final value of the trajectory,
denoted :math:`x_\text{f}`, be specified.  We can do this by requiring that
:math:`x(T) = x_\text{f}` or by using a more general form of constraint:

.. math::
   
  \psi_i(x(T)) = 0, \qquad i = 1, \dots, q.

The fully constrained case is obtained by setting :math:`q = n` and defining
:math:`\psi_i(x(T)) = x_i(T) - x_{i,\text{f}}`.  For a control problem with
a full set of terminal constraints, :math:`V(x(T))` can be omitted (since
its value is fixed).

Finally, we may wish to consider optimizations in which either the state or
the inputs are constrained by a set of nonlinear functions of the form

.. math::
   
  \text{lb}_i \leq g_i(x, u) \leq \text{ub}_i, \qquad i = 1, \dots, k.

where :math:`\text{lb}_i` and :math:`\text{ub}_i` represent lower and upper
bounds on the constraint function :math:`g_i`.  Note that these constraints
can be on the input, the state, or combinations of input and state,
depending on the form of :math:`g_i`.  Furthermore, these constraints are
intended to hold at all instants in time along the trajectory.

A common use of optimization-based control techniques is the implementation
of model predictive control (also called receding horizon control).  In
model predict control, a finite horizon optimal control problem is solved,
generating open-loop state and control trajectories.  The resulting control
trajectory is applied to the system for a fraction of the horizon
length. This process is then repeated, resulting in a sampled data feedback
law.  This approach is illustrated in the following figure:

.. image:: mpc-overview.png

Every :math:`\Delta T` seconds, an optimal control problem is solved over a
:math:`T` second horizon, starting from the current state.  The first
:math:`\Delta T` seconds of the optimal control :math:`u_T^{\*}(\cdot;
x(t))` is then applied to the system. If we let :math:`x_T^{\*}(\cdot;
x(t))` represent the optimal trajectory starting from :math:`x(t)`$ then the
system state evolves from :math:`x(t)` at current time :math:`t` to
:math:`x_T^{*}(\delta T, x(t))` at the next sample time :math:`t + \Delta
T`, assuming no model uncertainty.

In reality, the system will not follow the predicted path exactly, so that
the red (computed) and blue (actual) trajectories will diverge.  We thus
recompute the optimal path from the new state at time :math:`t + \Delta T`,
extending our horizon by an additional :math:`\Delta T` units of time.  This
approach can be shown to generate stabilizing control laws under suitable
conditions (see, for example, the FBS2e supplement on `Optimization-Based
Control <https://fbswiki.org/wiki/index.php/OBC>`_.
  
Module usage
============

The `obc` module provides a means of computing optimal trajectories for
nonlinear systems and implementing optimization-based controllers, including
model predictive control.  It follows the basic problem setup described
above, but carries out all computations in *discrete time* (so that
integrals become sums) and over a *finite horizon*.

To describe an optimal control problem we need an input/output system, a
time horizon, a cost function, and (optionally) a set of constraints on the
state and/or input, either along the trajectory and at the terminal time.
The `obc` module operates by converting the optimal control problem into a
standard optimization problem that can be solved by
:func:`scipy.optimize.minimize`.  The optimal control problem can be solved
by using the `~control.obc.compute_optimal_input` function:

  import control.obc as obc
  inputs = obc.compute_optimal_inputs(sys, horizon, X0, cost, constraints)

The `sys` parameter should be a :class:`~control.InputOutputSystem` and the
`horizon` parameter should represent a time vector that gives the list of
times at which the `cost` and `constraints` should be evaluated. By default,
`constraints` are taken to be trajectory constraints holding at all points
on the trajectory.  The `terminal_constraint` parameter can be used to
specify a constraint that only holds at the final point of the trajectory
and the `terminal_cost` paramter can be used to specify a terminal cost
function.


Example
=======

Module classes and functions
============================
.. autosummary::
   :toctree: generated/

   ~control.obc.OptimalControlProblem
   ~control.obc.compute_optimal_input
   ~control.obc.create_mpc_iosystem
   ~control.obc.input_poly_constraint
   ~control.obc.input_range_constraint
   ~control.obc.output_poly_constraint
   ~control.obc.output_range_constraint
   ~control.obc.state_poly_constraint
   ~control.obc.state_range_constraint
