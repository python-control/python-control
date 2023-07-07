# freqplot_test.py - test out frequency response plots
# RMM, 23 Jun 2023

import pytest
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from control.tests.conftest import slycotonly
pytestmark = pytest.mark.usefixtures("mplcleanup")

#
# Define a system for testing out different sharing options
#

omega = np.logspace(-2, 2, 5)
fresp1 = np.array([10 + 0j, 5 - 5j, 1 - 1j, 0.5 - 1j, -.1j])
fresp2 = np.array([1j, 0.5 - 0.5j, -0.5, 0.1 - 0.1j, -.05j]) * 0.1
fresp3 = np.array([10 + 0j, -20j, -10, 2j, 1])
fresp4 = np.array([10 + 0j, 5 - 5j, 1 - 1j, 0.5 - 1j, -.1j]) * 0.01

fresp = np.empty((2, 2, omega.size), dtype=complex)
fresp[0, 0] = fresp1
fresp[0, 1] = fresp2
fresp[1, 0] = fresp3
fresp[1, 1] = fresp4
manual_response = ct.FrequencyResponseData(
    fresp, omega, sysname="Manual Response")

@pytest.mark.parametrize(
    "sys", [
        ct.tf([1], [1, 2, 1], name='System 1'),         # SISO
        manual_response,                                   # simple MIMO
    ])
# @pytest.mark.parametrize("pltmag", [True, False])
# @pytest.mark.parametrize("pltphs", [True, False])
# @pytest.mark.parametrize("shrmag", ['row', 'all', False, None])
# @pytest.mark.parametrize("shrphs", ['row', 'all', False, None])
# @pytest.mark.parametrize("shrfrq", ['col', 'all', False, None])
# @pytest.mark.parametrize("secsys", [False, True])
@pytest.mark.parametrize(       # combinatorial-style test (faster)
    "pltmag, pltphs, shrmag, shrphs, shrfrq, secsys",
    [(True,  True,   None,   None,   None,   False),
     (True,  False,  None,   None,   None,   False),
     (False, True,   None,   None,   None,   False),
     (True,  True,   None,   None,   None,   True),
     (True,  True,   'row',  'row',  'col',  False),
     (True,  True,   'row',  'row',  'all',  True),
     (True,  True,   'all',  'row',  None,  False),
     (True,  True,   'row',  'all',  None,  True),
     (True,  True,   'none', 'none', None,  True),
     (True,  False,  'all',  'row',  None,  False),
     (True,  True,   True,   'row',  None,  True),
     (True,  True,   None,   'row',  True,  False),
     (True,  True,   'row',  None,   None,  True),
     ])
def test_response_plots(
        sys, pltmag, pltphs, shrmag, shrphs, shrfrq, secsys, clear=True):

    # Save up the keyword arguments
    kwargs = dict(
        plot_magnitude=pltmag, plot_phase=pltphs,
        share_magnitude=shrmag, share_phase=shrphs, share_frequency=shrfrq,
        # overlay_outputs=ovlout, overlay_inputs=ovlinp
    )

    # Create the response
    if isinstance(sys, ct.FrequencyResponseData):
        response = sys
    else:
        response = ct.frequency_response(sys)

    # Look for cases where there are no data to plot
    if not pltmag and not pltphs:
        return None

    # Plot the frequency response
    plt.figure()
    out = response.plot(**kwargs)

    # Make sure all of the outputs are of the right type
    nlines_plotted = 0
    for ax_lines in np.nditer(out, flags=["refs_ok"]):
        for line in ax_lines.item():
            assert isinstance(line, mpl.lines.Line2D)
            nlines_plotted += 1

    # Make sure number of plots is correct
    nlines_expected = response.ninputs * response.noutputs * \
        (2 if pltmag and pltphs else 1)
    assert nlines_plotted == nlines_expected

    # Save the old axes to compare later
    old_axes = plt.gcf().get_axes()

    # Add additional data (and provide info in the title)
    if secsys:
        newsys = ct.rss(
            4, sys.noutputs, sys.ninputs, strictly_proper=True)
        ct.frequency_response(newsys).plot(**kwargs)

        # Make sure we have the same axes
        new_axes = plt.gcf().get_axes()
        assert new_axes == old_axes

        # Make sure every axes has multiple lines
        for ax in new_axes:
            assert len(ax.get_lines()) > 1

    # Update the title so we can see what is going on
    fig = out[0, 0][0].axes.figure
    fig.suptitle(
        fig._suptitle._text +
        f" [{sys.noutputs}x{sys.ninputs}, pm={pltmag}, pp={pltphs},"
        f" sm={shrmag}, sp={shrphs}, sf={shrfrq}]",     # TODO: ", "
        # f"oo={ovlout}, oi={ovlinp}, ss={secsys}]",    # TODO: add back
        fontsize='small')

    # Get rid of the figure to free up memory
    if clear:
        plt.close('.Figure')


# Use the manaul response to verify that different settings are working
def test_manual_response_limits():
    # Default response: limits should be the same across rows
    out = manual_response.plot()
    axs = ct.get_plot_axes(out)
    for i in range(manual_response.noutputs):
        for j in range(1, manual_response.ninputs):
            # Everything in the same row should have the same limits
            assert axs[i*2, 0].get_ylim() == axs[i*2, j].get_ylim()
            assert axs[i*2 + 1, 0].get_ylim() == axs[i*2 + 1, j].get_ylim()
    # Different rows have different limits
    assert axs[0, 0].get_ylim() != axs[2, 0].get_ylim()
    assert axs[1, 0].get_ylim() != axs[3, 0].get_ylim()

    # TODO: finish writing tests

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
    sys_mimo = ct.tf(
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


def test_gangof4_plots(savefigs=False):
    proc = ct.tf([1], [1, 1, 1], name="process")
    ctrl = ct.tf([100], [1, 5], name="control")

    plt.figure()
    ct.gangof4_plot(proc, ctrl)

    if savefigs:
        plt.savefig('freqplot-gangof4.png')


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

    # Define a set of systems to test
    sys_siso = ct.tf([1], [1, 2, 1], name="SISO")
    sys_mimo = ct.tf(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
    sys_test = manual_response

    # Run through a large number of test cases
    test_cases = [
        # sys      pltmag pltphs shrmag  shrphs  shrfrq  secsys
        (sys_siso, True,  True,  None,   None,   None,   False),
        (sys_siso, True,  True,  None,   None,   None,   True),
        (sys_mimo, True,  True,  'row',  'row',  'col',  False),
        (sys_mimo, True,  True,  'row',  'row',  'col',  True),
        (sys_test, True,  True,  'row',  'row',  'col',  False),
        (sys_test, True,  True,  'row',  'row',  'col',  True),
        (sys_test, True,  True,  'none', 'none', 'col',  True),
        (sys_test, True,  True,  'all',  'row',  'col',  False),
        (sys_test, True,  True,  'row',  'all',  'col',  True),
        (sys_test, True,  True,  None,   'row',  'col',  False),
        (sys_test, True,  True,  'row',  None,   'col',  True),
    ]
    for args in test_cases:
        test_response_plots(*args, clear=False)

    # Define and run a selected set of interesting tests
    # TODO: TBD (see timeplot_test.py for format)

    test_basic_freq_plots(savefigs=True)
    test_gangof4_plots(savefigs=True)

    #
    # Run a few more special cases to show off capabilities (and save some
    # of them for use in the documentation).
    #

    pass
