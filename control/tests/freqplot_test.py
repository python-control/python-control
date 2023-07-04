# freqplot_test.py - test out frequency response plots
# RMM, 23 Jun 2023

import pytest
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from control.tests.conftest import slycotonly
pytestmark = pytest.mark.usefixtures("mplcleanup")

def test_basic_freq_plots(savefigs=False):
    # Basic SISO Bode plot
    plt.figure()
    # ct.frequency_response(sys_siso).plot()
    sys1 = ct.tf([1], [1, 2, 1], name='System 1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='System 2')
    response = ct.frequency_response([sys1, sys2])
    ct.bode_plot(response)
    if savefigs:
        plt.savefig('freqplot-siso_bode-default.png')

    # Basic MIMO Bode plot
    plt.figure()
    sys_mimo = ct.tf2ss(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
    ct.frequency_response(sys_mimo).plot()
    if savefigs:
        plt.savefig('freqplot-mimo_bode-default.png')

    # Magnitude only plot
    plt.figure()
    ct.frequency_response(sys_mimo).plot(plot_phase=False)
    if savefigs:
        plt.savefig('freqplot-mimo_bode-magonly.png')

    # Phase only plot
    plt.figure()
    ct.frequency_response(sys_mimo).plot(plot_magnitude=False)


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

    # Define and run a selected set of interesting tests
    # TODO: TBD (see timeplot_test.py for format)

    test_basic_freq_plots(savefigs=True)

    #
    # Run a few more special cases to show off capabilities (and save some
    # of them for use in the documentation).
    #

    pass
