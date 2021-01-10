"""sisotool_test.py"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from control.sisotool import sisotool
from control.rlocus import _RLClickDispatcher
from control.xferfcn import TransferFunction

from control.tests.conftest import X11only

@pytest.mark.usefixtures("mplcleanup")
class TestSisotool:
    """These are tests for the sisotool in sisotool.py."""

    @pytest.fixture
    def sys(self):
        """Return a generic SISO transfer function"""
        return TransferFunction([1000], [1, 25, 100, 0])

    @X11only
    def test_sisotool(self, sys):
        sisotool(sys, Hz=False)
        fig = plt.gcf()
        ax_mag, ax_rlocus, ax_phase, ax_step = fig.axes[:4]

        # Check the initial root locus plot points
        initial_point_0 = (np.array([-22.53155977]), np.array([0.]))
        initial_point_1 = (np.array([-1.23422011]), np.array([-6.54667031]))
        initial_point_2 = (np.array([-1.23422011]), np.array([06.54667031]))
        assert_array_almost_equal(ax_rlocus.lines[0].get_data(),
                                  initial_point_0, 4)
        assert_array_almost_equal(ax_rlocus.lines[1].get_data(),
                                  initial_point_1, 4)
        assert_array_almost_equal(ax_rlocus.lines[2].get_data(),
                                  initial_point_2, 4)

        # Check the step response before moving the point
        # new array needed because change in compute step response default time
        step_response_original = np.array(
            [0.    , 0.0069, 0.0448, 0.124 , 0.2427, 0.3933, 0.5653, 0.7473,
             0.928 , 1.0969])
        #old: np.array([0., 0.0217, 0.1281, 0.3237, 0.5797, 0.8566, 1.116,
            # 1.3261, 1.4659, 1.526])
        assert_array_almost_equal(
            ax_step.lines[0].get_data()[1][:10], step_response_original, 4)

        bode_plot_params = {
            'omega': None,
            'dB': False,
            'Hz': False,
            'deg': True,
            'omega_limits': None,
            'omega_num': None,
            'sisotool': True,
            'fig': fig,
            'margins': True
        }

        # Move the rootlocus to another point
        event = type('test', (object,), {'xdata': 2.31206868287,
                                         'ydata': 15.5983051046,
                                         'inaxes': ax_rlocus.axes})()
        _RLClickDispatcher(event=event, sys=sys, fig=fig,
                           ax_rlocus=ax_rlocus, sisotool=True, plotstr='-',
                           bode_plot_params=bode_plot_params, tvect=None)

        # Check the moved root locus plot points
        moved_point_0 = (np.array([-29.91742755]), np.array([0.]))
        moved_point_1 = (np.array([2.45871378]), np.array([-15.52647768]))
        moved_point_2 = (np.array([2.45871378]), np.array([15.52647768]))
        assert_array_almost_equal(ax_rlocus.lines[-3].get_data(),
                                  moved_point_0, 4)
        assert_array_almost_equal(ax_rlocus.lines[-2].get_data(),
                                  moved_point_1, 4)
        assert_array_almost_equal(ax_rlocus.lines[-1].get_data(),
                                  moved_point_2, 4)

        # Check if the bode_mag line has moved
        bode_mag_moved = np.array(
            [111.83321224, 92.29238035, 76.02822315, 62.46884113, 51.14108703,
             41.6554004, 33.69409534, 27.00237344, 21.38086717, 16.67791585])
        assert_array_almost_equal(ax_mag.lines[0].get_data()[1][10:20],
                                  bode_mag_moved, 4)

        # Check if the step response has changed
        # new array needed because change in compute step response default time
        step_response_moved = np.array(
            [0., 0.0072, 0.0516, 0.1554, 0.3281, 0.5681, 0.8646, 1.1987,
             1.5452, 1.875])
        # old: array([0., 0.0239, 0.161 , 0.4547, 0.8903, 1.407,
        #             1.9121, 2.2989, 2.4686, 2.353])
        assert_array_almost_equal(
            ax_step.lines[0].get_data()[1][:10], step_response_moved, 4)
