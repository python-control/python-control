.. _flatsys-module:

***************************
Differentially flat systems
***************************

.. automodule:: control.flatsys
   :no-members:
   :no-inherited-members:
   :no-special-members:

Overview of differential flatness
=================================

A nonlinear differential equation of the form 

.. math::
    \dot x = f(x, u), \qquad x \in R^n, u \in R^m

is *differentially flat* if there exists a function :math:`\alpha` such that

.. math::
    z = \alpha(x, u, \dot u\, \dots, u^{(p)})

and we can write the solutions of the nonlinear system as functions of
:math:`z` and a finite number of derivatives

.. math::
    x &= \beta(z, \dot z, \dots, z^{(q)}) \\
    u &= \gamma(z, \dot z, \dots, z^{(q)}).
    :label: flat2state

For a differentially flat system, all of the feasible trajectories for
the system can be written as functions of a flat output :math:`z(\cdot)` and
its derivatives.  The number of flat outputs is always equal to the
number of system inputs.

Differentially flat systems are useful in situations where explicit
trajectory generation is required. Since the behavior of a flat system
is determined by the flat outputs, we can plan trajectories in output
space, and then map these to appropriate inputs.  Suppose we wish to
generate a feasible trajectory for the the nonlinear system

.. math::
    \dot x = f(x, u), \qquad x(0) = x_0,\, x(T) = x_f.

If the system is differentially flat then

.. math::
    x(0) &= \beta\bigl(z(0), \dot z(0), \dots, z^{(q)}(0) \bigr) = x_0, \\
    x(T) &= \gamma\bigl(z(T), \dot z(T), \dots, z^{(q)}(T) \bigr) = x_f,

and we see that the initial and final condition in the full state
space depends on just the output :math:`z` and its derivatives at the
initial and final times.  Thus any trajectory for :math:`z` that satisfies
these boundary conditions will be a feasible trajectory for the
system, using equation :eq:`flat2state` to determine the
full state space and input trajectories.

In particular, given initial and final conditions on :math:`z` and its
derivatives that satisfy the initial and final conditions any curve
:math:`z(\cdot)` satisfying those conditions will correspond to a feasible
trajectory of the system.  We can parameterize the flat output trajectory
using a set of smooth basis functions :math:`\psi_i(t)`:

.. math::
  z(t) = \sum_{i=1}^N \alpha_i \psi_i(t), \qquad \alpha_i \in R

We seek a set of coefficients :math:`\alpha_i`, :math:`i = 1, \dots, N` such
that :math:`z(t)` satisfies the boundary conditions for :math:`x(0)` and
:math:`x(T)`.  The derivatives of the flat output can be computed in terms of
the derivatives of the basis functions:

.. math::
  \dot z(t) &= \sum_{i=1}^N \alpha_i \dot \psi_i(t) \\
  &\,\vdots \\
  \dot z^{(q)}(t) &= \sum_{i=1}^N \alpha_i \psi^{(q)}_i(t).

We can thus write the conditions on the flat outputs and their
derivatives as

.. math::
  \begin{bmatrix}
    \psi_1(0) & \psi_2(0) & \dots & \psi_N(0) \\
    \dot \psi_1(0) & \dot \psi_2(0) & \dots & \dot \psi_N(0) \\
    \vdots & \vdots & & \vdots \\
    \psi^{(q)}_1(0) & \psi^{(q)}_2(0) & \dots & \psi^{(q)}_N(0) \\[1ex]
    \psi_1(T) & \psi_2(T) & \dots & \psi_N(T) \\
    \dot \psi_1(T) & \dot \psi_2(T) & \dots & \dot \psi_N(T) \\
    \vdots & \vdots & & \vdots \\
    \psi^{(q)}_1(T) & \psi^{(q)}_2(T) & \dots & \psi^{(q)}_N(T) \\
  \end{bmatrix}
  \begin{bmatrix} \alpha_1 \\ \vdots \\ \alpha_N \end{bmatrix} =
  \begin{bmatrix}
    z(0) \\ \dot z(0) \\ \vdots \\ z^{(q)}(0) \\[1ex]
    z(T) \\ \dot z(T) \\ \vdots \\ z^{(q)}(T) \\
  \end{bmatrix}

This equation is a *linear* equation of the form 

.. math::
   M \alpha = \begin{bmatrix} \bar z(0) \\ \bar z(T) \end{bmatrix}

where :math:`\bar z` is called the *flat flag* for the system.
Assuming that :math:`M` has a sufficient number of columns and that it is full
column rank, we can solve for a (possibly non-unique) :math:`\alpha` that
solves the trajectory generation problem.

Module usage
============

To create a trajectory for a differentially flat system, a
:class:`~control.flatsys.FlatSystem` object must be created.  This is
done by specifying the `forward` and `reverse` mappings between the
system state/input and the differentially flat outputs and their
derivatives ("flat flag").

The :func:`~control.flatsys.FlatSystem.forward` method computes the
flat flag given a state and input:

    zflag = sys.forward(x, u)

The :func:`~control.flatsys.FlatSystem.reverse` method computes the state
and input given the flat flag:

    x, u = sys.reverse(zflag)

The flag :math:`\bar z` is implemented as a list of flat outputs :math:`z_i`
and their derivatives up to order :math:`q_i`:

    zflag[i][j] = :math:`z_i^{(j)}`

The number of flat outputs must match the number of system inputs.

For a linear system, a flat system representation can be generated using the
:class:`~control.flatsys.LinearFlatSystem` class::

    sys = control.flatsys.LinearFlatSystem(linsys)

For more general systems, the `FlatSystem` object must be created manually::

    sys = control.flatsys.FlatSystem(nstate, ninputs, forward, reverse)

In addition to the flat system description, a set of basis functions
:math:`\phi_i(t)` must be chosen.  The `FlatBasis` class is used to represent
the basis functions.  A polynomial basis function of the form 1, :math:`t`,
:math:`t^2`, ... can be computed using the `PolyBasis` class, which is
initialized by passing the desired order of the polynomial basis set::

    polybasis = control.flatsys.PolyBasis(N)

Once the system and basis function have been defined, the
:func:`~control.flatsys.point_to_point` function can be used to compute a
trajectory between initial and final states and inputs::

    traj = control.flatsys.point_to_point(
        sys, Tf, x0, u0, xf, uf, basis=polybasis)

The returned object has class :class:`~control.flatsys.SystemTrajectory` and
can be used to compute the state and input trajectory between the initial and
final condition::

    xd, ud = traj.eval(T)

where `T` is a list of times on which the trajectory should be evaluated
(e.g., `T = numpy.linspace(0, Tf, M)`.

The :func:`~control.flatsys.point_to_point` function also allows the
specification of a cost function and/or constraints, in the same
format as :func:`~control.optimal.solve_ocp`.

Example
=======

To illustrate how we can use a two degree-of-freedom design to improve the
performance of the system, consider the problem of steering a car to change
lanes on a road. We use the non-normalized form of the dynamics, which are
derived *Feedback Systems* by Astrom and Murray, Example 3.11.

.. code-block:: python

    import control.flatsys as fs

    # Function to take states, inputs and return the flat flag
    def vehicle_flat_forward(x, u, params={}):
        # Get the parameter values
        b = params.get('wheelbase', 3.)

        # Create a list of arrays to store the flat output and its derivatives
        zflag = [np.zeros(3), np.zeros(3)]

        # Flat output is the x, y position of the rear wheels
        zflag[0][0] = x[0]
        zflag[1][0] = x[1]

        # First derivatives of the flat output
        zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt
        zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

        # First derivative of the angle
        thdot = (u[0]/b) * np.tan(u[1])

        # Second derivatives of the flat output (setting vdot = 0)
        zflag[0][2] = -u[0] * thdot * np.sin(x[2])
        zflag[1][2] =  u[0] * thdot * np.cos(x[2])

        return zflag

    # Function to take the flat flag and return states, inputs
    def vehicle_flat_reverse(zflag, params={}):
        # Get the parameter values
        b = params.get('wheelbase', 3.)

        # Create a vector to store the state and inputs
        x = np.zeros(3)
        u = np.zeros(2)

        # Given the flat variables, solve for the state
        x[0] = zflag[0][0]  # x position
        x[1] = zflag[1][0]  # y position
        x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot

        # And next solve for the inputs
        u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
        u[1] = np.arctan2(
            (zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])), u[0]/b)

        return x, u

    vehicle_flat = fs.FlatSystem(
        3, 2, forward=vehicle_flat_forward, reverse=vehicle_flat_reverse)

To find a trajectory from an initial state :math:`x_0` to a final state
:math:`x_\text{f}` in time :math:`T_\text{f}` we solve a point-to-point
trajectory generation problem. We also set the initial and final inputs, which
sets the vehicle velocity :math:`v` and steering wheel angle :math:`\delta` at
the endpoints.

.. code-block:: python

    # Define the endpoints of the trajectory
    x0 = [0., -2., 0.]; u0 = [10., 0.]
    xf = [100., 2., 0.]; uf = [10., 0.]
    Tf = 10

    # Define a set of basis functions to use for the trajectories
    poly = fs.PolyFamily(6)

    # Find a trajectory between the initial condition and the final condition
    traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)

    # Create the trajectory
    t = np.linspace(0, Tf, 100)
    x, u = traj.eval(t)

Module classes and functions
============================

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   ~control.flatsys.BasisFamily
   ~control.flatsys.BezierFamily
   ~control.flatsys.FlatSystem
   ~control.flatsys.LinearFlatSystem
   ~control.flatsys.PolyFamily
   ~control.flatsys.SystemTrajectory

.. autosummary::
   :toctree: generated/

   ~control.flatsys.point_to_point
