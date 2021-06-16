"""rlocus_test.py - unit test for root locus diagrams

RMM, 1 Jul 2011
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

import control as ct
from control.rlocus import root_locus, _RLClickDispatcher
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
            poles_expected = np.sort(feedback(sys, k).pole())
            poles = np.sort(poles)
            np.testing.assert_array_almost_equal(poles, poles_expected)

    def testRootLocus(self, sys):
        """Basic root locus (no plot)"""
        klist = [-1, 0, 1]

        roots, k_out = root_locus(sys, klist, plot=False)
        np.testing.assert_equal(len(roots), len(klist))
        np.testing.assert_allclose(klist, k_out)
        self.check_cl_poles(sys, roots, klist)

    def test_without_gains(self, sys):
        roots, kvect = root_locus(sys, plot=False)
        self.check_cl_poles(sys, roots, kvect)

    @pytest.mark.parametrize('grid', [None, True, False])
    def test_root_locus_plot_grid(self, sys, grid):
        rlist, klist = root_locus(sys, grid=grid)
        ax = plt.gca()
        n_gridlines = sum([int(line.get_linestyle() in [':', 'dotted',
                                                        '--', 'dashed'])
                           for line in ax.lines])
        if grid is False:
            assert n_gridlines == 2
        else:
            assert n_gridlines > 2
        # TODO check validity of grid

    def test_root_locus_warnings(self):
        sys = TransferFunction([1000], [1, 25, 100, 0])
        with pytest.warns(FutureWarning, match="Plot.*deprecated"):
            rlist, klist = root_locus(sys, Plot=True)
        with pytest.warns(FutureWarning, match="PrintGain.*deprecated"):
            rlist, klist = root_locus(sys, PrintGain=True)

    def test_root_locus_neg_false_gain_nonproper(self):
        """ Non proper TranferFunction with negative gain: Not implemented"""
        with pytest.raises(ValueError, match="with equal order"):
            root_locus(TransferFunction([-1, 2], [1, 2]))

    # TODO: cover and validate negative false_gain branch in _default_gains()

    def test_root_locus_zoom(self):
        """Check the zooming functionality of the Root locus plot"""
        system = TransferFunction([1000], [1, 25, 100, 0])
        plt.figure()
        root_locus(system)
        fig = plt.gcf()
        ax_rlocus = fig.axes[0]

        event = type('test', (object,), {'xdata': 14.7607954359,
                                         'ydata': -35.6171379864,
                                         'inaxes': ax_rlocus.axes})()
        ax_rlocus.set_xlim((-10.813628105112421, 14.760795435937652))
        ax_rlocus.set_ylim((-35.61713798641108, 33.879716621220311))
        plt.get_current_fig_manager().toolbar.mode = 'zoom rect'
        _RLClickDispatcher(event, system, fig, ax_rlocus, '-')

        zoom_x = ax_rlocus.lines[-2].get_data()[0][0:5]
        zoom_y = ax_rlocus.lines[-2].get_data()[1][0:5]
        zoom_y = [abs(y) for y in zoom_y]

        zoom_x_valid = [
            -5., - 4.61281263, - 4.16689986, - 4.04122642, - 3.90736502]
        zoom_y_valid = [0., 0., 0., 0., 0.]

        assert_array_almost_equal(zoom_x, zoom_x_valid)
        assert_array_almost_equal(zoom_y, zoom_y_valid)

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
        import signal

        # Define a system that exhibits this behavior
        sys = ct.tf(*sp.signal.zpk2tf(
            [-1e-2, 1-1e7j, 1+1e7j], [0, -1e7j, 1e7j], 1))

        ct.root_locus(sys)
