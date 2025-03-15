"""phaseplot_test.py - test phase plot functions

RMM, 17 24 2011 (based on TestMatlab from v0.4c)

This test suite calls various phaseplot functions.  Since the plots
themselves can't be verified, this is mainly here to make sure all
of the function arguments are handled correctly.  If you run an
individual test by itself and then type show(), it should pop open
the figures so that you can check them visually.
"""

import warnings
from math import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import control as ct
import control.phaseplot as pp
from control import phase_plot


# Legacy tests
@pytest.mark.usefixtures("mplcleanup", "ignore_future_warning")
class TestPhasePlot:
    def testInvPendSims(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10),
                  X0 = ([1,1], [-1,1]))

    def testInvPendNoSims(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10));

    def testInvPendTimePoints(self):
        phase_plot(self.invpend_ode, (-6,6,10), (-6,6,10),
                  X0 = ([1,1], [-1,1]), T=np.linspace(0,5,100))

    def testInvPendLogtime(self):
        phase_plot(self.invpend_ode, X0 =
                  [ [-2*pi, 1.6], [-2*pi, 0.5], [-1.8, 2.1],
                    [-1, 2.1], [4.2, 2.1], [5, 2.1],
                    [2*pi, -1.6], [2*pi, -0.5], [1.8, -2.1],
                    [1, -2.1], [-4.2, -2.1], [-5, -2.1] ],
                  T = np.linspace(0, 40, 200),
                  logtime=(3, 0.7),
                  verbose=False)

    def testInvPendAuto(self):
        phase_plot(self.invpend_ode, lingrid = 0, X0=
                  [[-2.3056, 2.1], [2.3056, -2.1]], T=6, verbose=False)

    def testInvPendFBS(self):
        # Outer trajectories
        phase_plot(
            self.invpend_ode, timepts=[1, 4, 10],
            X0=[[-2*pi, 1.6], [-2*pi, 0.5], [-1.8, 2.1], [-1, 2.1],
                [4.2, 2.1], [5, 2.1], [2*pi, -1.6], [2*pi, -0.5],
                [1.8, -2.1], [1, -2.1], [-4.2, -2.1], [-5, -2.1]],
            T = np.linspace(0, 40, 800),
            params=(1, 1, 0.2, 1))

        # Separatrices

    def testOscillatorParams(self):
        # default values
        m = 1
        b = 1
        k = 1
        phase_plot(self.oscillator_ode, timepts = [0.3, 1, 2, 3], X0 =
                  [[-1,1], [-0.3,1], [0,1], [0.25,1], [0.5,1], [0.7,1],
                   [1,1], [1.3,1], [1,-1], [0.3,-1], [0,-1], [-0.25,-1],
                   [-0.5,-1], [-0.7,-1], [-1,-1], [-1.3,-1]],
                  T = np.linspace(0, 10, 100), parms = (m, b, k))

    def testNoArrows(self):
        # Test case from aramakrl that was generating a type error
        # System does not have arrows
        # cf. issue #96,
        # https://github.com/python-control/python-control/issues/96
        def d1(x1x2,t):
            x1,x2 = x1x2
            return np.array([x2, x2 - 2*x1])

        x1x2_0 = np.array([[-1.,1.], [-1.,-1.], [1.,1.], [1.,-1.],
                           [-1.,0.],[1.,0.],[0.,-1.],[0.,1.],[0.,0.]])

        plt.figure(1)
        phase_plot(d1,X0=x1x2_0,T=100)

    # Sample dynamical systems - inverted pendulum
    def invpend_ode(self, x, t, m=1., l=1., b=0.2, g=1):
        import numpy as np
        return (x[1], -b/m*x[1] + (g*l/m) * np.sin(x[0]))

    # Sample dynamical systems - oscillator
    def oscillator_ode(self, x, t, m=1., b=1, k=1, extra=None):
        return (x[1], -k/m*x[0] - b/m*x[1])


@pytest.mark.parametrize(
    "func, args, kwargs", [
        [ct.phaseplot.vectorfield, [], {}],
        [ct.phaseplot.vectorfield, [],
         {'color': 'k', 'gridspec': [4, 3], 'params': {}}],
        [ct.phaseplot.streamlines, [1], {'params': {}, 'arrows': 5}],
        [ct.phaseplot.streamlines, [],
         {'dir': 'forward', 'gridtype': 'meshgrid', 'color': 'k'}],
        [ct.phaseplot.streamlines, [1],
         {'dir': 'reverse', 'gridtype': 'boxgrid', 'color': None}],
        [ct.phaseplot.streamlines, [1],
         {'dir': 'both', 'gridtype': 'circlegrid', 'gridspec': [0.5, 5]}],
        [ct.phaseplot.equilpoints, [], {}],
        [ct.phaseplot.equilpoints, [], {'color': 'r', 'gridspec': [5, 5]}],
        [ct.phaseplot.separatrices, [], {}],
        [ct.phaseplot.separatrices, [], {'color': 'k', 'arrows': 4}],
        [ct.phaseplot.separatrices, [5], {'params': {}, 'gridspec': [5, 5]}],
        [ct.phaseplot.separatrices, [5], {'color': ('r', 'g')}],
    ])
@pytest.mark.usefixtures('mplcleanup')
def test_helper_functions(func, args, kwargs):
    # Test with system
    sys = ct.nlsys(
        lambda t, x, u, params: [x[0] - 3*x[1], -3*x[0] + x[1]],
        states=2, inputs=0)
    _out = func(sys, [-1, 1, -1, 1], *args, **kwargs)

    # Test with function
    rhsfcn = lambda t, x: sys.dynamics(t, x, 0, {})
    _out = func(rhsfcn, [-1, 1, -1, 1], *args, **kwargs)


@pytest.mark.usefixtures('mplcleanup')
def test_system_types():
    # Sample dynamical systems - inverted pendulum
    def invpend_ode(t, x, m=0, l=0, b=0, g=0):
        return (x[1], -b/m*x[1] + (g*l/m) * np.sin(x[0]))

    # Use callable form, with parameters (if not correct, will get /0 error)
    ct.phase_plane_plot(
        invpend_ode, [-5, 5, -2, 2], params={'args': (1, 1, 0.2, 1)},
        plot_streamlines=True)

    # Linear I/O system
    ct.phase_plane_plot(
        ct.ss([[0, 1], [-1, -1]], [[0], [1]], [[1, 0]], 0),
        plot_streamlines=True)


@pytest.mark.usefixtures('mplcleanup')
def test_phaseplane_errors():
    with pytest.raises(ValueError, match="invalid grid specification"):
        ct.phase_plane_plot(ct.rss(2, 1, 1), gridspec='bad',
                            plot_streamlines=True)

    with pytest.raises(ValueError, match="unknown grid type"):
        ct.phase_plane_plot(ct.rss(2, 1, 1), gridtype='bad',
                            plot_streamlines=True)

    with pytest.raises(ValueError, match="system must be planar"):
        ct.phase_plane_plot(ct.rss(3, 1, 1),
                            plot_streamlines=True)

    with pytest.raises(ValueError, match="params must be dict with key"):
        def invpend_ode(t, x, m=0, l=0, b=0, g=0):
            return (x[1], -b/m*x[1] + (g*l/m) * np.sin(x[0]))
        ct.phase_plane_plot(
            invpend_ode, [-5, 5, 2, 2], params={'stuff': (1, 1, 0.2, 1)},
                            plot_streamlines=True)

    with pytest.raises(ValueError, match="gridtype must be 'meshgrid' when using streamplot"):
        ct.phase_plane_plot(ct.rss(2, 1, 1), plot_streamlines=False,
                            plot_streamplot=True, gridtype='boxgrid')

    # Warning messages for invalid solutions: nonlinear spring mass system
    sys = ct.nlsys(
        lambda t, x, u, params: np.array(
            [x[1], -0.25 * (x[0] - 0.01 * x[0]**3) - 0.1 * x[1]]),
        states=2, inputs=0)
    with pytest.warns(
            UserWarning, match=r"initial_state=\[.*\], solve_ivp failed"):
        ct.phase_plane_plot(
            sys, [-12, 12, -10, 10], 15, gridspec=[2, 9],
            plot_separatrices=False, plot_streamlines=True)

    # Turn warnings off
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ct.phase_plane_plot(
            sys, [-12, 12, -10, 10], 15, gridspec=[2, 9],
            plot_streamlines=True, plot_separatrices=False,
            suppress_warnings=True)

@pytest.mark.usefixtures('mplcleanup')
def test_phase_plot_zorder():
    # some of these tests are a bit akward since the streamlines and separatrices
    # are stored in the same list, so we separate them by color
    key_color = "tab:blue" # must not be 'k', 'r', 'b' since they are used by separatrices

    def get_zorders(cplt):
        max_zorder = lambda items: max([line.get_zorder() for line in items])
        assert isinstance(cplt.lines[0], list)
        streamline_lines = [line for line in cplt.lines[0] if line.get_color() == key_color]
        separatrice_lines = [line for line in cplt.lines[0] if line.get_color() != key_color]
        streamlines = max_zorder(streamline_lines) if streamline_lines else None
        separatrices = max_zorder(separatrice_lines) if separatrice_lines else None
        assert cplt.lines[1] == None or isinstance(cplt.lines[1], mpl.quiver.Quiver)
        quiver = cplt.lines[1].get_zorder() if cplt.lines[1] else None
        assert cplt.lines[2] == None or isinstance(cplt.lines[2], list)
        equilpoints = max_zorder(cplt.lines[2]) if cplt.lines[2] else None
        assert cplt.lines[3] == None or isinstance(cplt.lines[3], mpl.streamplot.StreamplotSet)
        streamplot = max(cplt.lines[3].lines.get_zorder(), cplt.lines[3].arrows.get_zorder()) if cplt.lines[3] else None
        return streamlines, quiver, streamplot, separatrices, equilpoints

    def assert_orders(streamlines, quiver, streamplot, separatrices, equilpoints):
        print(streamlines, quiver, streamplot, separatrices, equilpoints)
        if streamlines is not None:
            assert streamlines < separatrices < equilpoints
        if quiver is not None:
            assert quiver < separatrices < equilpoints
        if streamplot is not None:
            assert streamplot < separatrices < equilpoints

    def sys(t, x):
        return np.array([4*x[1], -np.sin(4*x[0])])

    # ensure correct zordering for all three flow types
    res_streamlines = ct.phase_plane_plot(sys, plot_streamlines=dict(color=key_color))
    assert_orders(*get_zorders(res_streamlines))
    res_vectorfield = ct.phase_plane_plot(sys, plot_vectorfield=True)
    assert_orders(*get_zorders(res_vectorfield))
    res_streamplot = ct.phase_plane_plot(sys, plot_streamplot=True)
    assert_orders(*get_zorders(res_streamplot))

    # ensure that zorder can still be overwritten
    res_reversed = ct.phase_plane_plot(sys, plot_streamlines=dict(color=key_color, zorder=50), plot_vectorfield=dict(zorder=40),
                                       plot_streamplot=dict(zorder=30), plot_separatrices=dict(zorder=20), plot_equilpoints=dict(zorder=10))
    streamlines, quiver, streamplot, separatrices, equilpoints = get_zorders(res_reversed)
    assert streamlines > quiver > streamplot > separatrices > equilpoints


@pytest.mark.usefixtures('mplcleanup')
def test_stream_plot_magnitude():
    def sys(t, x):
        return np.array([4*x[1], -np.sin(4*x[0])])

    # plt context with linewidth
    with plt.rc_context({'lines.linewidth': 4}):
        res = ct.phase_plane_plot(sys, plot_streamplot=dict(vary_linewidth=True))
    linewidths = res.lines[3].lines.get_linewidths()
    # linewidths are scaled to be between 0.25 and 2 times default linewidth
    # but the extremes may not exist if there is no line at that point
    assert min(linewidths) < 2 and max(linewidths) > 7

    # make sure changing the colormap works
    res = ct.phase_plane_plot(sys, plot_streamplot=dict(vary_color=True, cmap='viridis'))
    assert res.lines[3].lines.get_cmap().name == 'viridis'
    res = ct.phase_plane_plot(sys, plot_streamplot=dict(vary_color=True, cmap='turbo'))
    assert res.lines[3].lines.get_cmap().name == 'turbo'

    # make sure changing the norm at least doesn't throw an error
    ct.phase_plane_plot(sys, plot_streamplot=dict(vary_color=True, norm=mpl.colors.LogNorm()))


@pytest.mark.usefixtures('mplcleanup')
def test_basic_phase_plots(savefigs=False):
    sys = ct.nlsys(
        lambda t, x, u, params: np.array([[0, 1], [-1, -1]]) @ x,
        states=['position', 'velocity'], inputs=0, name='damped oscillator')

    plt.figure()
    axis_limits = [-1, 1, -1, 1]
    T = 8
    ct.phase_plane_plot(sys, axis_limits, T, plot_streamlines=True)
    if savefigs:
        plt.savefig('phaseplot-dampedosc-default.png')

    def invpend_update(t, x, u, params):
        m, l, b, g = params['m'], params['l'], params['b'], params['g']
        return [x[1], -b/m * x[1] + (g * l / m) * np.sin(x[0]) + u[0]/m]
    invpend = ct.nlsys(invpend_update, states=2, inputs=1, name='invpend')

    plt.figure()
    ct.phase_plane_plot(
        invpend, [-2*pi, 2*pi, -2, 2], 5,
        gridtype='meshgrid', gridspec=[5, 8], arrows=3,
        plot_separatrices={'gridspec': [12, 9]}, plot_streamlines=True,
        params={'m': 1, 'l': 1, 'b': 0.2, 'g': 1})
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\dot\theta$ [rad/sec]")

    if savefigs:
        plt.savefig('phaseplot-invpend-meshgrid.png')

    def oscillator_update(t, x, u, params):
        return [x[1] + x[0] * (1 - x[0]**2 - x[1]**2),
                -x[0] + x[1] * (1 - x[0]**2 - x[1]**2)]
    oscillator = ct.nlsys(
        oscillator_update, states=2, inputs=0, name='nonlinear oscillator')

    plt.figure()
    ct.phase_plane_plot(oscillator, [-1.5, 1.5, -1.5, 1.5], 0.9,
                        plot_streamlines=True)
    pp.streamlines(
    oscillator, np.array([[0, 0]]), 1.5,
    gridtype='circlegrid', gridspec=[0.5, 6], dir='both')
    pp.streamlines(oscillator, np.array([[1, 0]]), 2*pi, arrows=6, color='b')
    plt.gca().set_aspect('equal')

    if savefigs:
        plt.savefig('phaseplot-oscillator-helpers.png')

    plt.figure()
    ct.phase_plane_plot(
        invpend, [-2*pi, 2*pi, -2, 2],
        plot_streamplot=dict(vary_color=True, vary_density=True),
        gridspec=[60, 20], params={'m': 1, 'l': 1, 'b': 0.2, 'g': 1}
    )
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel(r"$\dot\theta$ [rad/sec]")

    if savefigs:
        plt.savefig('phaseplot-invpend-streamplot.png')


if __name__ == "__main__":
    #
    # Interactive mode: generate plots for manual viewing
    #
    # Running this script in python (or better ipython) will show a
    # collection of figures that should all look OK on the screeen.
    #

    # In interactive mode, turn on ipython interactive graphics
    plt.ion()

    # Start by clearing existing figures
    plt.close('all')

    test_basic_phase_plots(savefigs=True)
