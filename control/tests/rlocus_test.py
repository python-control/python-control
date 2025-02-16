"""rlocus_test.py - unit test for root locus diagrams

RMM, 1 Jul 2011
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

import control as ct
from control.rlocus import root_locus
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.bdalg import feedback


@pytest.mark.usefixtures("mplcleanup")
class TestRootLocus:
    """These are tests for the feedback function in rlocus.py."""

    @pytest.fixture(params=[pytest.param((sysclass, sargs + (dt, )),
                                         id=f"{systypename}-{dtstring}")
                            for sysclass, systypename, sargs in [
                                    (TransferFunction, 'TF', ([1, 2],
                                                              [1, 2, 3])),
                                    (StateSpace, 'SS', ([[1., 4.], [3., 2.]],
                                                        [[1.], [-4.]],
                                                        [[1., 0.]],
                                                        [[0.]])),
                                    ]
                            for dt, dtstring in [(0, 'ctime'),
                                                 (True, 'dtime')]
                            ])
    def sys(self, request):
        """Return some simple LTI systems for testing"""
        # avoid construction during collection time: prevent unfiltered
        # deprecation warning
        sysfn, args = request.param
        return sysfn(*args)

    def check_cl_poles(self, sys, pole_list, k_list):
        for k, poles in zip(k_list, pole_list):
            poles_expected = np.sort(feedback(sys, k).poles())
            poles = np.sort(poles)
            np.testing.assert_array_almost_equal(poles, poles_expected)

    @pytest.mark.filterwarnings("ignore:.*return value.*:FutureWarning")
    def testRootLocus(self, sys):
        """Basic root locus (no plot)"""
        klist = [-1, 0, 1]

        roots, k_out = root_locus(sys, klist, plot=False)
        np.testing.assert_equal(len(roots), len(klist))
        np.testing.assert_allclose(klist, k_out)
        self.check_cl_poles(sys, roots, klist)

        # now check with plotting
        roots, k_out = root_locus(sys, klist, plot=True)
        np.testing.assert_equal(len(roots), len(klist))
        np.testing.assert_allclose(klist, k_out)
        self.check_cl_poles(sys, roots, klist)

    @pytest.mark.filterwarnings("ignore:.*return value.*:FutureWarning")
    def test_without_gains(self, sys):
        roots, kvect = root_locus(sys, plot=False)
        self.check_cl_poles(sys, roots, kvect)

    @pytest.mark.parametrize("grid", [None, True, False, 'empty'])
    @pytest.mark.parametrize("method", ['plot', 'map', 'response', 'pzmap'])
    def test_root_locus_plot_grid(self, sys, grid, method):
        import mpl_toolkits.axisartist as AA

        # Generate the root locus plot
        plt.clf()
        if method == 'plot':
            ct.root_locus_plot(sys, grid=grid)
        elif method == 'map':
            ct.root_locus_map(sys).plot(grid=grid)
        elif method == 'response':
            response = ct.root_locus_map(sys)
            ct.root_locus_plot(response, grid=grid)
        elif method == 'pzmap':
            response = ct.root_locus_map(sys)
            ct.pole_zero_plot(response, grid=grid)

        # Count the number of dotted/dashed lines in the plot
        ax = plt.gca()
        n_gridlines = sum([int(
            line.get_linestyle() in [':', 'dotted', '--', 'dashed'] or
            line.get_linewidth() < 1
        ) for line in ax.lines])

        # Make sure they line up with what we expect
        if grid == 'empty':
            assert n_gridlines == 0
            assert not isinstance(ax, AA.Axes)
        elif grid is False:
            assert n_gridlines == 2 if sys.isctime() else 3
            assert not isinstance(ax, AA.Axes)
        elif sys.isdtime(strict=True):
            assert n_gridlines > 2
            assert not isinstance(ax, AA.Axes)
        else:
            # Continuous time, with grid => check that AxisArtist was used
            assert isinstance(ax, AA.Axes)
            for spine in ['wnxneg', 'wnxpos', 'wnyneg', 'wnypos']:
                assert spine in ax.axis

        # TODO: check validity of grid

    @pytest.mark.filterwarnings("ignore:.*return value.*:FutureWarning")
    def test_root_locus_neg_false_gain_nonproper(self):
        """ Non proper TranferFunction with negative gain: Not implemented"""
        with pytest.raises(ValueError, match="with equal order"):
            root_locus(TransferFunction([-1, 2], [1, 2]), plot=True)

    # TODO: cover and validate negative false_gain branch in _default_gains()

    @pytest.mark.skip("Zooming functionality no longer implemented")
    @pytest.mark.skipif(plt.get_current_fig_manager().toolbar is None,
                        reason="Requires the zoom toolbar")
    def test_root_locus_zoom(self):
        """Check the zooming functionality of the Root locus plot"""
        system = TransferFunction([1000], [1, 25, 100, 0])
        plt.figure()
        root_locus(system, plot=True)
        fig = plt.gcf()
        ax_rlocus = fig.axes[0]

        event = type('test', (object,), {'xdata': 14.7607954359,
                                         'ydata': -35.6171379864,
                                         'inaxes': ax_rlocus.axes})()
        ax_rlocus.set_xlim((-10.813628105112421, 14.760795435937652))
        ax_rlocus.set_ylim((-35.61713798641108, 33.879716621220311))
        plt.get_current_fig_manager().toolbar.mode = 'zoom rect'
        _RLClickDispatcher(event, system, fig, ax_rlocus, '-') # noqa: F821

        zoom_x = ax_rlocus.lines[-2].get_data()[0][0:5]
        zoom_y = ax_rlocus.lines[-2].get_data()[1][0:5]
        zoom_y = [abs(y) for y in zoom_y]

        zoom_x_valid = [
            -5., - 4.61281263, - 4.16689986, - 4.04122642, - 3.90736502]
        zoom_y_valid = [0., 0., 0., 0., 0.]

        assert_array_almost_equal(zoom_x, zoom_x_valid)
        assert_array_almost_equal(zoom_y, zoom_y_valid)

    @pytest.mark.filterwarnings("ignore:.*return value.*:FutureWarning")
    @pytest.mark.timeout(2)
    def test_rlocus_default_wn(self):
        """Check that default wn calculation works properly"""
        #
        # System that triggers use of y-axis as basis for wn (for coverage)
        #
        # This system generates a root locus plot that used to cause the
        # creation (and subsequent deletion) of a large number of natural
        # frequency contours within the `_default_wn` function in `rlocus.py`.
        # This unit test makes sure that is fixed by generating a test case
        # that will take a long time to do the calculation (minutes).
        #
        import scipy as sp

        # Define a system that exhibits this behavior
        sys = ct.tf(*sp.signal.zpk2tf(
            [-1e-2, 1-1e7j, 1+1e7j], [0, -1e7j, 1e7j], 1))

        ct.root_locus(sys, plot=True)


@pytest.mark.parametrize(
    "sys, grid, xlim, ylim, interactive", [
        (ct.tf([1], [1, 2, 1]), None, None, None, False),
    ])
@pytest.mark.usefixtures("mplcleanup")
def test_root_locus_plots(sys, grid, xlim, ylim, interactive):
    ct.root_locus_map(sys).plot(
        grid=grid, xlim=xlim, ylim=ylim, interactive=interactive)
    # TODO: add tests to make sure everything "looks" OK


# Test deprecated keywords
@pytest.mark.parametrize("keyword", ["kvect", "k"])
@pytest.mark.usefixtures("mplcleanup")
def test_root_locus_legacy(keyword):
    sys = ct.rss(2, 1, 1)
    with pytest.warns(FutureWarning, match=f"'{keyword}' is deprecated"):
        ct.root_locus_plot(sys, **{keyword: [0, 1, 2]})


# Generate plots used in documentation
@pytest.mark.usefixtures("mplcleanup")
def test_root_locus_documentation(savefigs=False):
    plt.figure()
    sys = ct.tf([1, 2], [1, 2, 3], name='SISO transfer function')
    response = ct.pole_zero_map(sys)
    ct.pole_zero_plot(response)
    if savefigs:
        plt.savefig('pzmap-siso_ctime-default.png')

    plt.figure()
    ct.root_locus_map(sys).plot()
    if savefigs:
        plt.savefig('rlocus-siso_ctime-default.png')

    # TODO: generate event in order to generate real title
    plt.figure()
    cplt = ct.root_locus_map(sys).plot(initial_gain=3.506)
    ax = cplt.axes[0, 0]
    freqplot_rcParams = ct.config._get_param('ctrlplot', 'rcParams')
    with plt.rc_context(freqplot_rcParams):
        ax.set_title(
            "Clicked at: -2.729+1.511j  gain = 3.506  damping = 0.8748")
    if savefigs:
        plt.savefig('rlocus-siso_ctime-clicked.png')

    plt.figure()
    sysd = sys.sample(0.1)
    ct.root_locus_plot(sysd)
    if savefigs:
        plt.savefig('rlocus-siso_dtime-default.png')

    plt.figure()
    sys1 = ct.tf([1, 2], [1, 2, 3], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    ct.root_locus_plot([sys1, sys2], grid=False)
    if savefigs:
        plt.savefig('rlocus-siso_multiple-nogrid.png')


# https://github.com/python-control/python-control/issues/1063
def test_rlocus_singleton():
    # Generate a root locus map for a singleton
    L = ct.tf([1, 1], [1, 2, 3])
    rldata = ct.root_locus_map(L, 1)
    np.testing.assert_equal(rldata.gains, np.array([1]))
    assert rldata.loci.shape == (1, 2)

    # Generate the root locus plot (no loci)
    cplt = rldata.plot()
    assert len(cplt.lines[0, 0]) == 1      # poles (one set of markers)
    assert len(cplt.lines[0, 1]) == 1      # zeros
    assert len(cplt.lines[0, 2]) == 2      # loci (two 0-length lines)


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

    # Define systems to be tested
    sys_secord = ct.tf([1], [1, 1, 1], name="2P")
    sys_seczero = ct.tf([1, 0, -1], [1, 1, 1], name="2P, 2Z")
    sys_fbs_a = ct.tf([1, 1], [1, 0, 0], name="FBS 12_19a")
    sys_fbs_b = ct.tf(
        ct.tf([1, 1], [1, 2, 0]) * ct.tf([1], [1, 2 ,4]), name="FBS 12_19b")
    sys_fbs_c = ct.tf([1, 1], [1, 0, 1, 0], name="FBS 12_19c")
    sys_fbs_d = ct.tf([1, 2, 2], [1, 0, 1, 0], name="FBS 12_19d")
    sys_poles = sys_fbs_d.poles()
    sys_zeros = sys_fbs_d.zeros()
    sys_discrete = ct.zpk(
        sys_zeros / 3, sys_poles / 3, 1, dt=True, name="discrete")

    # Run through a large number of test cases
    test_cases = [
        # sys          grid   xlim      ylim      inter
        (sys_secord,   None,  None,     None,     None),
        (sys_seczero,  None,  None,     None,     None),
        (sys_fbs_a,    None,  None,     None,     None),
        (sys_fbs_b,    None,  None,     None,     False),
        (sys_fbs_c,    None,  None,     None,     None),
        (sys_fbs_c,    None,  None,     [-2, 2],  None),
        (sys_fbs_c,    True,  [-3, 3],  None,     None),
        (sys_fbs_d,    None,  None,     None,     None),
        (ct.zpk(sys_zeros * 10, sys_poles * 10, 1, name="12_19d * 10"),
                       None,  None,     None,     None),
        (ct.zpk(sys_zeros / 10, sys_poles / 10, 1, name="12_19d / 10"),
                       True,  None,     None,     None),
        (sys_discrete, None,  None,     None,     None),
        (sys_discrete, True,  None,     None,     None),
        (sys_fbs_d,    True,  None,     None,     True),
    ]

    for sys, grid, xlim, ylim, interactive in test_cases:
        plt.figure()
        test_root_locus_plots(
            sys, grid=grid, xlim=xlim, ylim=ylim, interactive=interactive)
        ct.suptitle(
            f"sys={sys.name}, {grid=}, {xlim=}, {ylim=}, {interactive=}",
            frame='figure')

    # Run tests that generate plots for the documentation
    test_root_locus_documentation(savefigs=True)
