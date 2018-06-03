import unittest
import numpy as np
from control.sisotool import sisotool
from control.tests.margin_test import assert_array_almost_equal
from control.rlocus import _RLClickDispatcher
from control.xferfcn import TransferFunction
import matplotlib.pyplot as plt

class TestSisotool(unittest.TestCase):
    """These are tests for the sisotool in sisotool.py."""

    def setUp(self):
        # One random SISO system.
        self.system = TransferFunction([1000],[1,25,100,0])

    def test_sisotool(self):
        sisotool(self.system,Hz=False)
        fig = plt.gcf()
        ax_mag,ax_rlocus,ax_phase,ax_step = fig.axes[0],fig.axes[1],fig.axes[2],fig.axes[3]

        # Check the initial root locus plot points
        initial_point_0 = (np.array([-22.53155977]),np.array([0.]))
        initial_point_1 = (np.array([-1.23422011]), np.array([-6.54667031]))
        initial_point_2 = (np.array([-1.23422011]), np.array([06.54667031]))
        assert_array_almost_equal(ax_rlocus.lines[0].get_data(),initial_point_0)
        assert_array_almost_equal(ax_rlocus.lines[1].get_data(),initial_point_1)
        assert_array_almost_equal(ax_rlocus.lines[2].get_data(),initial_point_2)

        # Check the step response before moving the point
        step_response_original = np.array([ 0.,          0.02233651,  0.13118374,  0.33078542,  0.5907113,   0.87041549, 1.13038536,  1.33851053,  1.47374666,  1.52757114])
        assert_array_almost_equal(ax_step.lines[0].get_data()[1][:10],step_response_original)

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
        event = type('test', (object,), {'xdata': 2.31206868287,'ydata':15.5983051046, 'inaxes':ax_rlocus.axes})()
        _RLClickDispatcher(event=event, sys=self.system, fig=fig,ax_rlocus=ax_rlocus,sisotool=True, plotstr='-' ,bode_plot_params=bode_plot_params, tvect=None)

        # Check the moved root locus plot points
        moved_point_0 = (np.array([-29.91742755]), np.array([0.]))
        moved_point_1 = (np.array([2.45871378]), np.array([-15.52647768]))
        moved_point_2 = (np.array([2.45871378]), np.array([15.52647768]))
        assert_array_almost_equal(ax_rlocus.lines[-3].get_data(),moved_point_0)
        assert_array_almost_equal(ax_rlocus.lines[-2].get_data(),moved_point_1)
        assert_array_almost_equal(ax_rlocus.lines[-1].get_data(),moved_point_2)

        # Check if the bode_mag line has moved
        bode_mag_moved = np.array([ 111.83321224,   92.29238035,   76.02822315,   62.46884113,   51.14108703, 41.6554004,    33.69409534,   27.00237344,   21.38086717,   16.67791585])
        assert_array_almost_equal(ax_mag.lines[0].get_data()[1][10:20],bode_mag_moved)

        # Check if the step response has changed
        step_response_moved = np.array([[ 0.,          0.02458187,  0.16529784 , 0.46602716 , 0.91012035 , 1.43364313, 1.93996334 , 2.3190105  , 2.47041552 , 2.32724853] ])
        assert_array_almost_equal(ax_step.lines[0].get_data()[1][:10],step_response_moved)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestSisotool)

if __name__ == "__main__":
    unittest.main()
