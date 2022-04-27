"""nichols_test.py - test Nichols plot

RMM, 31 Mar 2011
"""

import matplotlib.pyplot as plt

import pytest

from control import StateSpace, nichols_plot, nichols, nichols_grid, pade, tf


@pytest.fixture()
def tsys():
    """Set up a system to test operations on."""
    A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
    B = [[1.], [-3.], [-2.]]
    C = [[4., 2., -3.]]
    D = [[0.]]
    return StateSpace(A, B, C, D)


def test_nichols(tsys, mplcleanup):
    """Generate a Nichols plot."""
    nichols_plot(tsys)


def test_nichols_alias(tsys, mplcleanup):
    """Test the control.nichols alias and the grid=False parameter"""
    nichols(tsys, grid=False)


@pytest.mark.usefixtures("mplcleanup")
class TestNicholsGrid:
    def test_ax(self):
        # check grid is plotted into gca, or specified axis
        fig, axs = plt.subplots(2,2)
        plt.sca(axs[0,1])

        cl_mag_lines = nichols_grid()[1]
        assert cl_mag_lines[0].axes is axs[0, 1]

        cl_mag_lines = nichols_grid(ax=axs[1,1])[1]
        assert cl_mag_lines[0].axes is axs[1, 1]
        # nichols_grid didn't change what the "current axes" are
        assert plt.gca() is axs[0, 1]


    def test_cl_phase_label_control(self):
        # test label_cl_phases argument
        cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels \
            = nichols_grid()
        assert len(cl_phase_labels) > 0

        cl_mag_lines, cl_phase_lines, cl_mag_labels, cl_phase_labels \
            = nichols_grid(label_cl_phases=False)
        assert len(cl_phase_labels) == 0


    def test_labels_clipped(self):
        # regression test: check that contour labels are clipped
        mcontours, ncontours, mlabels, nlabels = nichols_grid()
        assert all(ml.get_clip_on() for ml in mlabels)
        assert all(nl.get_clip_on() for nl in nlabels)


    def test_minimal_phase(self):
        # regression test: phase extent is minimal
        g = tf([1],[1,1]) * tf([1],[1/1, 2*0.1/1, 1])
        nichols(g)
        ax = plt.gca()
        assert ax.get_xlim()[1] <= 0


    def test_fixed_view(self):
        # respect xlim, ylim set by user
        g = (tf([1],[1/1, 2*0.01/1, 1])
             * tf([1],[1/100**2, 2*0.001/100, 1])
             * tf(*pade(0.01, 5)))

        # normally a broad axis
        nichols(g)

        assert(plt.xlim()[0] == -1440)
        assert(plt.ylim()[0] <= -240)

        nichols(g, grid=False)

        # zoom in
        plt.axis([-360,0,-40,50])

        # nichols_grid doesn't expand limits
        nichols_grid()
        assert(plt.xlim()[0] == -360)
        assert(plt.ylim()[1] >= -40)
