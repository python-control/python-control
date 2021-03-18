"""sisotool_test.py"""

from control.exception import ControlMIMONotImplemented
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from control.sisotool import sisotool
from control.rlocus import _RLClickDispatcher
from control.xferfcn import TransferFunction
from control.statesp import StateSpace


@pytest.mark.usefixtures("mplcleanup")
class TestSisotool:
    """These are tests for the sisotool in sisotool.py."""

    @pytest.fixture
    def sys(self):
        """Return a generic SISO transfer function"""
        return TransferFunction([1000], [1, 25, 100, 0])

    @pytest.fixture
    def sysdt(self):
        """Return a generic SISO transfer function"""
        return TransferFunction([1000], [1, 25, 100, 0], True)

    @pytest.fixture
    def sys222(self):
        """2-states square system (2 inputs x 2 outputs)"""
        A222 = [[4., 1.],
                [2., -3]]
        B222 = [[5., 2.],
                [-3., -3.]]
        C222 = [[2., -4],
                [0., 1.]]
        D222 = [[3., 2.],
                [1., -1.]]
        return StateSpace(A222, B222, C222, D222)

    @pytest.fixture
    def sys221(self):
        """2-states, 2 inputs x 1 output"""
        A222 = [[4., 1.],
                [2., -3]]
        B222 = [[5., 2.],
                [-3., -3.]]
        C221 = [[0., 1.]]
        D221 = [[1., -1.]]
        return StateSpace(A222, B222, C221, D221)

    def test_sisotool(self, sys):
        sisotool(sys, Hz=False)
        fig = plt.gcf()
        ax_mag, ax_rlocus, ax_phase, ax_step = fig.axes[:4]

        # Check the initial root locus plot points
        initial_point_0 = (np.array([-22.53155977]), np.array([0.]))
        initial_point_1 = (np.array([-1.23422011]), np.array([-6.54667031]))
        initial_point_2 = (np.array([-1.23422011]), np.array([6.54667031]))
        assert_array_almost_equal(ax_rlocus.lines[0].get_data(),
                                  initial_point_0, 4)
        assert_array_almost_equal(ax_rlocus.lines[1].get_data(),
                                  initial_point_1, 4)
        assert_array_almost_equal(ax_rlocus.lines[2].get_data(),
                                  initial_point_2, 4)

        # Check the step response before moving the point
        step_response_original = np.array(
            [0.    , 0.0216, 0.1271, 0.3215, 0.5762, 0.8522, 1.1114, 1.3221,
             1.4633, 1.5254])
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
            [674.0242, 667.8354, 661.7033, 655.6275, 649.6074, 643.6426,
              637.7324, 631.8765, 626.0742, 620.3252])
        assert_array_almost_equal(ax_mag.lines[0].get_data()[1][10:20],
                                  bode_mag_moved, 4)

        # Check if the step response has changed
        step_response_moved = np.array(
            [0.    , 0.0237, 0.1596, 0.4511, 0.884 , 1.3985, 1.9031, 2.2922,
             2.4676, 2.3606])
        assert_array_almost_equal(
            ax_step.lines[0].get_data()[1][:10], step_response_moved, 4)

    def test_sisotool_tvect(self, sys):
        # test supply tvect
        tvect = np.linspace(0, 1, 10)
        sisotool(sys, tvect=tvect)
        fig = plt.gcf()
        ax_rlocus, ax_step = fig.axes[1], fig.axes[3]

        # Move the rootlocus to another point and confirm same tvect
        event = type('test', (object,), {'xdata': 2.31206868287,
                                         'ydata': 15.5983051046,
                                         'inaxes': ax_rlocus.axes})()
        _RLClickDispatcher(event=event, sys=sys, fig=fig,
                           ax_rlocus=ax_rlocus, sisotool=True, plotstr='-',
                           bode_plot_params=dict(), tvect=tvect)
        assert_array_almost_equal(tvect, ax_step.lines[0].get_data()[0])

    def test_sisotool_tvect_dt(self, sysdt):
        # test supply tvect
        tvect = np.linspace(0, 1, 10)
        sisotool(sysdt, tvect=tvect)
        fig = plt.gcf()
        ax_rlocus, ax_step = fig.axes[1], fig.axes[3]

        # Move the rootlocus to another point and confirm same tvect
        event = type('test', (object,), {'xdata': 2.31206868287,
                                         'ydata': 15.5983051046,
                                         'inaxes': ax_rlocus.axes})()
        _RLClickDispatcher(event=event, sys=sysdt, fig=fig,
                           ax_rlocus=ax_rlocus, sisotool=True, plotstr='-',
                           bode_plot_params=dict(), tvect=tvect)
        assert_array_almost_equal(tvect, ax_step.lines[0].get_data()[0])

    def test_sisotool_mimo(self,  sys222, sys221):
        # a 2x2 should not raise an error:
        sisotool(sys222)

        # but 2 input, 1 output should
        with pytest.raises(ControlMIMONotImplemented):
            sisotool(sys221)
