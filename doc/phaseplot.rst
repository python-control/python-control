.. currentmodule:: control

.. _phase-plane-plots:

Phase Plane Plots
=================

Insight into nonlinear systems can often be obtained by looking at phase
plane diagrams.  The :func:`phase_plane_plot` function allows the
creation of a 2-dimensional phase plane diagram for a system.  This
functionality is supported by a set of mapping functions that are part of
the `phaseplot` module.

The default method for generating a phase plane plot is to provide a
2D dynamical system along with a range of coordinates and time limit:

.. testsetup:: phaseplot

    import matplotlib.pyplot as plt
    plt.close('all')

.. testcode:: phaseplot

    def sys_update(t, x, u, params):
        return np.array([[0, 1], [-1, -1]]) @ x
    sys = ct.nlsys(
        sys_update, states=['position', 'velocity'],
	inputs=0, name='damped oscillator')
    axis_limits = [-1, 1, -1, 1]
    T = 8
    ct.phase_plane_plot(sys, axis_limits, T)

.. testcode:: phaseplot
    :hide:

    import matplotlib.pyplot as plt
    plt.savefig('figures/phaseplot-dampedosc-default.png')

.. image:: figures/phaseplot-dampedosc-default.png
   :align: center

By default, the plot includes streamlines generated from starting
points on limits of the plot, with arrows showing the flow of the
system, as well as any equilibrium points for the system.  A variety
of options are available to modify the information that is plotted,
including plotting a grid of vectors instead of streamlines and
turning on and off various features of the plot.

To illustrate some of these possibilities, consider a phase plane plot for
an inverted pendulum system, which is created using a mesh grid:

.. testcode:: phaseplot
    :hide:

    plt.figure()

.. testcode:: phaseplot

    def invpend_update(t, x, u, params):
        m, l, b, g = params['m'], params['l'], params['b'], params['g']
        return [x[1], -b/m * x[1] + (g * l / m) * np.sin(x[0]) + u[0]/m]
    invpend = ct.nlsys(invpend_update, states=2, inputs=1, name='invpend')

    ct.phase_plane_plot(
        invpend, [-2 * np.pi, 2 * np.pi, -2, 2], 5,
        gridtype='meshgrid', gridspec=[5, 8], arrows=3,
        plot_equilpoints={'gridspec': [12, 9]},
        params={'m': 1, 'l': 1, 'b': 0.2, 'g': 1})
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\dot\theta$ [rad/sec]")

.. testcode:: phaseplot
    :hide:

    plt.savefig('figures/phaseplot-invpend-meshgrid.png')

.. image:: figures/phaseplot-invpend-meshgrid.png
   :align: center

This figure shows several features of more complex phase plane plots:
multiple equilibrium points are shown, with saddle points showing
separatrices, and streamlines generated along a 5x8 mesh of initial
conditions.  At each mesh point, a streamline is created that goes 5 time
units forward and backward in time.  A separate grid specification is used
to find equilibrium points and separatrices (since the course grid spacing
of 5x8 does not find all possible equilibrium points).  Together, the
multiple features in the phase plane plot give a good global picture of the
topological structure of solutions of the dynamical system.

Phase plots can be built up by hand using a variety of helper functions that
are part of the :mod:`phaseplot` (pp) module:

.. testcode:: phaseplot
    :hide:

    plt.figure()

.. testcode:: phaseplot

    import control.phaseplot as pp

    def oscillator_update(t, x, u, params):
        return [x[1] + x[0] * (1 - x[0]**2 - x[1]**2),
                -x[0] + x[1] * (1 - x[0]**2 - x[1]**2)]
    oscillator = ct.nlsys(
        oscillator_update, states=2, inputs=0, name='nonlinear oscillator')

    ct.phase_plane_plot(oscillator, [-1.5, 1.5, -1.5, 1.5], 0.9)
    pp.streamlines(
        oscillator, np.array([[0, 0]]), 1.5,
        gridtype='circlegrid', gridspec=[0.5, 6], dir='both')
    pp.streamlines(
        oscillator, np.array([[1, 0]]), 2 * np.pi, arrows=6, color='b')
    plt.gca().set_aspect('equal')

.. testcode:: phaseplot
    :hide:

    plt.savefig('figures/phaseplot-oscillator-helpers.png')

.. image:: figures/phaseplot-oscillator-helpers.png
   :align: center

The following helper functions are available:

.. autosummary::

   phaseplot.equilpoints
   phaseplot.separatrices
   phaseplot.streamlines
   phaseplot.vectorfield

The :func:`phase_plane_plot` function calls these helper functions
based on the options it is passed.

Note that unlike other plotting functions, phase plane plots do not
involve computing a response and then plotting the result via a
``plot()`` method.  Instead, the plot is generated directly be a call
to the :func:`phase_plane_plot` function (or one of the
:mod:`~control.phaseplot` helper functions).
