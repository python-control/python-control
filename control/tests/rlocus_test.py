#!/usr/bin/env python
#
# rlocus_test.py - unit test for root locus diagrams
# RMM, 1 Jul 2011

import unittest
import numpy as np
from control.rlocus import root_locus, _RLClickDispatcher
from control.xferfcn import TransferFunction
from control.statesp import StateSpace
from control.bdalg import feedback
import matplotlib.pyplot as plt
from control.tests.margin_test import assert_array_almost_equal


class TestRootLocus(unittest.TestCase):
    """These are tests for the feedback function in rlocus.py."""

    def setUp(self):
        """This contains some random LTI systems and scalars for testing."""

        # Two random SISO systems.
        sys1 = TransferFunction([1, 2], [1, 2, 3])
        sys2 = StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
            [[1., 0.]], [[0.]])
        self.systems = (sys1, sys2)

    def check_cl_poles(self, sys, pole_list, k_list):
        for k, poles in zip(k_list, pole_list):
            poles_expected = np.sort(feedback(sys, k).pole())
            poles = np.sort(poles)
            np.testing.assert_array_almost_equal(poles, poles_expected)

    def testRootLocus(self):
        """Basic root locus plot"""
        klist = [-1, 0, 1]
        for sys in self.systems:
            roots, k_out = root_locus(sys, klist, Plot=False)
            np.testing.assert_equal(len(roots), len(klist))
            np.testing.assert_array_equal(klist, k_out)
            self.check_cl_poles(sys, roots, klist)

    def test_without_gains(self):
        for sys in self.systems:
            roots, kvect = root_locus(sys, Plot=False)
            self.check_cl_poles(sys, roots, kvect)

    def test_root_locus_zoom(self):
        """Check the zooming functionality of the Root locus plot"""
        system = TransferFunction([1000], [1, 25, 100, 0])
        root_locus(system)
        fig = plt.gcf()
        ax_rlocus = fig.axes[0]

        event = type('test', (object,), {'xdata': 14.7607954359, 'ydata': -35.6171379864, 'inaxes': ax_rlocus.axes})()
        ax_rlocus.set_xlim((-10.813628105112421, 14.760795435937652))
        ax_rlocus.set_ylim((-35.61713798641108, 33.879716621220311))
        plt.get_current_fig_manager().toolbar.mode = 'zoom rect'
        _RLClickDispatcher(event, system, fig, ax_rlocus, '-')

        zoom_x = ax_rlocus.lines[-2].get_data()[0][0:5]
        zoom_y = ax_rlocus.lines[-2].get_data()[1][0:5]
        zoom_y = [abs(y) for y in zoom_y]

        zoom_x_valid = [-5. ,- 4.61281263, - 4.16689986, - 4.04122642, - 3.90736502]
        zoom_y_valid = [0. ,0., 0., 0., 0.]

        assert_array_almost_equal(zoom_x,zoom_x_valid)
        assert_array_almost_equal(zoom_y,zoom_y_valid)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestRootLocus)

if __name__ == "__main__":
    unittest.main()
