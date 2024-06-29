# freqplot_test.py - test out frequency response plots
# RMM, 23 Jun 2023

import pytest
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from control.tests.conftest import slycotonly, editsdefaults
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
        manual_response,                                # simple MIMO
    ])
# @pytest.mark.parametrize("pltmag", [True, False])
# @pytest.mark.parametrize("pltphs", [True, False])
# @pytest.mark.parametrize("shrmag", ['row', 'all', False, None])
# @pytest.mark.parametrize("shrphs", ['row', 'all', False, None])
# @pytest.mark.parametrize("shrfrq", ['col', 'all', False, None])
# @pytest.mark.parametrize("secsys", [False, True])
@pytest.mark.parametrize(       # combinatorial-style test (faster)
    "pltmag, pltphs, shrmag, shrphs, shrfrq, ovlout, ovlinp, secsys",
    [(True,  True,   None,   None,   None,   False,  False,  False),
     (True,  False,  None,   None,   None,   True,   False,  False),
     (False, True,   None,   None,   None,   False,  True,   False),
     (True,  True,   None,   None,   None,   False,  False,  True),
     (True,  True,   'row',  'row',  'col',  False,  False,  False),
     (True,  True,   'row',  'row',  'all',  False,  False,  True),
     (True,  True,   'all',  'row',  None,   False,  False,  False),
     (True,  True,   'row',  'all',  None,   False,  False,  True),
     (True,  True,   'none', 'none', None,   False,  False,  True),
     (True,  False,  'all',  'row',  None,   False,  False,  False),
     (True,  True,   True,   'row',  None,   False,  False,  True),
     (True,  True,   None,   'row',  True,   False,  False,  False),
     (True,  True,   'row',  None,   None,   False,  False,  True),
     ])
@pytest.mark.usefixtures("editsdefaults")
def test_response_plots(
        sys, pltmag, pltphs, shrmag, shrphs, shrfrq, secsys,
        ovlout, ovlinp, clear=True):

    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # Save up the keyword arguments
    kwargs = dict(
        plot_magnitude=pltmag, plot_phase=pltphs,
        share_magnitude=shrmag, share_phase=shrphs, share_frequency=shrfrq,
        overlay_outputs=ovlout, overlay_inputs=ovlinp,
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

    # Check the shape
    if ovlout and ovlinp:
        assert out.shape == (pltmag + pltphs, 1)
    elif ovlout:
        assert out.shape == (pltmag + pltphs, sys.ninputs)
    elif ovlinp:
        assert out.shape == (sys.noutputs * (pltmag + pltphs), 1)
    else:
        assert out.shape == (sys.noutputs * (pltmag + pltphs), sys.ninputs)

    # Make sure all of the outputs are of the right type
    nlines_plotted = 0
    for ax_lines in np.nditer(out, flags=["refs_ok"]):
        for line in ax_lines.item() or []:
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
    ct.suptitle(
        fig._suptitle._text +
        f" [{sys.noutputs}x{sys.ninputs}, pm={pltmag}, pp={pltphs},"
        f" sm={shrmag}, sp={shrphs}, sf={shrfrq}]",     # TODO: ", "
        # f"oo={ovlout}, oi={ovlinp}, ss={secsys}]",    # TODO: add back
        frame='figure', fontsize='small')

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


@pytest.mark.parametrize(
    "plt_fcn", [ct.bode_plot, ct.nichols_plot, ct.singular_values_plot])
@pytest.mark.usefixtures("editsdefaults")
def test_line_styles(plt_fcn):
    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # Define a couple of systems for testing
    sys1 = ct.tf([1], [1, 2, 1], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    sys3 = ct.tf([0.2, 0.1], [1, 0.1, 0.3, 0.1, 0.1], name='sys3')

    # Create a plot for the first system, with custom styles
    lines_default = plt_fcn(sys1)

    # Now create a plot using *fmt customization
    lines_fmt = plt_fcn(sys2, None, 'r--')
    assert lines_fmt.reshape(-1)[0][0].get_color() == 'r'
    assert lines_fmt.reshape(-1)[0][0].get_linestyle() == '--'

    # Add a third plot using keyword customization
    lines_kwargs = plt_fcn(sys3, color='g', linestyle=':')
    assert lines_kwargs.reshape(-1)[0][0].get_color() == 'g'
    assert lines_kwargs.reshape(-1)[0][0].get_linestyle() == ':'


def test_basic_freq_plots(savefigs=False):
    # Basic SISO Bode plot
    plt.figure()
    # ct.frequency_response(sys_siso).plot()
    sys1 = ct.tf([1], [1, 2, 1], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    response = ct.frequency_response([sys1, sys2])
    ct.bode_plot(response, initial_phase=0)
    if savefigs:
        plt.savefig('freqplot-siso_bode-default.png')

    plt.figure()
    omega = np.logspace(-2, 2, 500)
    ct.frequency_response([sys1, sys2], omega).plot(initial_phase=0)
    if savefigs:
        plt.savefig('freqplot-siso_bode-omega.png')

    # Basic MIMO Bode plot
    plt.figure()
    sys_mimo = ct.tf(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="sys_mimo")
    ct.frequency_response(sys_mimo).plot()
    if savefigs:
        plt.savefig('freqplot-mimo_bode-default.png')

    # Magnitude only plot, with overlayed inputs and outputs
    plt.figure()
    ct.frequency_response(sys_mimo).plot(
        plot_phase=False, overlay_inputs=True, overlay_outputs=True)
    if savefigs:
        plt.savefig('freqplot-mimo_bode-magonly.png')

    # Phase only plot
    plt.figure()
    ct.frequency_response(sys_mimo).plot(plot_magnitude=False)

    # Singular values plot
    plt.figure()
    ct.singular_values_response(sys_mimo).plot()
    if savefigs:
        plt.savefig('freqplot-mimo_svplot-default.png')

    # Nichols chart
    plt.figure()
    ct.nichols_plot(response)
    if savefigs:
        plt.savefig('freqplot-siso_nichols-default.png')

    # Nyquist plot - default settings
    plt.figure()
    sys = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys')
    ct.nyquist(sys)
    if savefigs:
        plt.savefig('freqplot-nyquist-default.png')

    # Nyquist plot - custom settings
    plt.figure()
    sys = ct.tf([1, 0.2], [1, 0, 1]) * ct.tf([1], [1, 0])
    nyqresp = ct.nyquist_response(sys)
    nyqresp.plot(
        max_curve_magnitude=6, max_curve_offset=1,
        arrows=[0, 0.15, 0.3, 0.6, 0.7, 0.925], label='sys')
    print("Encirclements =", nyqresp.count)
    if savefigs:
        plt.savefig('freqplot-nyquist-custom.png')


def test_gangof4_plots(savefigs=False):
    proc = ct.tf([1], [1, 1, 1], name="process")
    ctrl = ct.tf([100], [1, 5], name="control")

    plt.figure()
    ct.gangof4_plot(proc, ctrl)

    if savefigs:
        plt.savefig('freqplot-gangof4.png')


@pytest.mark.parametrize("response_cmd, return_type", [
    (ct.frequency_response, ct.FrequencyResponseData),
    (ct.nyquist_response, ct.freqplot.NyquistResponseData),
    (ct.singular_values_response, ct.FrequencyResponseData),
])
@pytest.mark.usefixtures("editsdefaults")
def test_first_arg_listable(response_cmd, return_type):
    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    sys = ct.rss(2, 1, 1)

    # If we pass a single system, should get back a single system
    result = response_cmd(sys)
    assert isinstance(result, return_type)

    # Save the results from a single plot
    lines_single = result.plot()

    # If we pass a list of systems, we should get back a list
    result = response_cmd([sys, sys, sys])
    assert isinstance(result, list)
    assert len(result) == 3
    assert all([isinstance(item, return_type) for item in result])

    # Make sure that plot works
    lines_list = result.plot()
    if response_cmd == ct.frequency_response:
        assert lines_list.shape == lines_single.shape
        assert len(lines_list.reshape(-1)[0]) == \
            3 * len(lines_single.reshape(-1)[0])
    else:
        assert lines_list.shape[0] == 3 * lines_single.shape[0]

    # If we pass a singleton list, we should get back a list
    result = response_cmd([sys])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], return_type)


@pytest.mark.usefixtures("editsdefaults")
def test_bode_share_options():
    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # Default sharing should share along rows and cols for mag and phase
    lines = ct.bode_plot(manual_response)
    axs = ct.get_plot_axes(lines)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # Share y limits along rows
            assert axs[i, j].get_ylim() == axs[i, 0].get_ylim()

            # Share x limits along columns
            assert axs[i, j].get_xlim() == axs[-1, j].get_xlim()

    # Sharing along y axis for mag but not phase
    plt.figure()
    lines = ct.bode_plot(manual_response, share_phase='none')
    axs = ct.get_plot_axes(lines)
    for i in range(int(axs.shape[0] / 2)):
        for j in range(axs.shape[1]):
            if i != 0:
                # Different rows are different
                assert axs[i*2 + 1, 0].get_ylim() != axs[1, 0].get_ylim()
            elif j != 0:
                # Different columns are different
                assert axs[i*2 + 1, j].get_ylim() != axs[i*2 + 1, 0].get_ylim()

    # Turn off sharing for magnitude and phase
    plt.figure()
    lines = ct.bode_plot(manual_response, sharey='none')
    axs = ct.get_plot_axes(lines)
    for i in range(int(axs.shape[0] / 2)):
        for j in range(axs.shape[1]):
            if i != 0:
                # Different rows are different
                assert axs[i*2, 0].get_ylim() != axs[0, 0].get_ylim()
                assert axs[i*2 + 1, 0].get_ylim() != axs[1, 0].get_ylim()
            elif j != 0:
                # Different columns are different
                assert axs[i*2, j].get_ylim() != axs[i*2, 0].get_ylim()
                assert axs[i*2 + 1, j].get_ylim() != axs[i*2 + 1, 0].get_ylim()

    # Turn off sharing in x axes
    plt.figure()
    lines = ct.bode_plot(manual_response, sharex='none')
    # TODO: figure out what to check


@pytest.mark.parametrize("plot_type", ['bode', 'svplot', 'nichols'])
def test_freqplot_plot_type(plot_type):
    if plot_type == 'svplot':
        response = ct.singular_values_response(ct.rss(2, 1, 1))
    else:
        response = ct.frequency_response(ct.rss(2, 1, 1))
    lines = response.plot(plot_type=plot_type)
    if plot_type == 'bode':
        assert lines.shape == (2, 1)
    else:
        assert lines.shape == (1, )

@pytest.mark.parametrize("plt_fcn", [ct.bode_plot, ct.singular_values_plot])
@pytest.mark.usefixtures("editsdefaults")
def test_freqplot_omega_limits(plt_fcn):
    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # Utility function to check visible limits
    def _get_visible_limits(ax):
        xticks = np.array(ax.get_xticks())
        limits = ax.get_xlim()
        return np.array([min(xticks[xticks >= limits[0]]),
                         max(xticks[xticks <= limits[1]])])

    # Generate a test response with a fixed set of limits
    response = ct.singular_values_response(
        ct.tf([1], [1, 2, 1]), np.logspace(-1, 1))

    # Generate a plot without overridding the limits
    lines = plt_fcn(response)
    ax = ct.get_plot_axes(lines)
    np.testing.assert_allclose(
        _get_visible_limits(ax.reshape(-1)[0]), np.array([0.1, 10]))

    # Now reset the limits
    lines = plt_fcn(response, omega_limits=(1, 100))
    ax = ct.get_plot_axes(lines)
    np.testing.assert_allclose(
        _get_visible_limits(ax.reshape(-1)[0]), np.array([1, 100]))


def test_gangof4_trace_labels():
    P1 = ct.rss(2, 1, 1, name='P1')
    P2 = ct.rss(3, 1, 1, name='P2')
    C = ct.rss(1, 1, 1, name='C')

    # Make sure default labels are as expected
    out = ct.gangof4_response(P1, C).plot()
    out = ct.gangof4_response(P2, C).plot()
    axs = ct.get_plot_axes(out)
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'None'
    assert legend[1].get_text() == 'None'
    plt.close()

    # Override labels
    out = ct.gangof4_response(P1, C).plot(label='xxx, line1, yyy')
    out = ct.gangof4_response(P2, C).plot(label='xxx, line2, yyy')
    axs = ct.get_plot_axes(out)
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'xxx, line1, yyy'
    assert legend[1].get_text() == 'xxx, line2, yyy'
    plt.close()


@pytest.mark.parametrize(
    "plt_fcn", [ct.bode_plot, ct.singular_values_plot, ct.nyquist_plot])
@pytest.mark.usefixtures("editsdefaults")
def test_freqplot_trace_labels(plt_fcn):
    sys1 = ct.rss(2, 1, 1, name='sys1')
    sys2 = ct.rss(3, 1, 1, name='sys2')

    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # Make sure default labels are as expected
    out = plt_fcn([sys1, sys2])
    axs = ct.get_plot_axes(out)
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'sys1'
    assert legend[1].get_text() == 'sys2'
    plt.close()

    # Override labels all at once
    out = plt_fcn([sys1, sys2], label=['line1', 'line2'])
    axs = ct.get_plot_axes(out)
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()

    # Override labels one at a time
    out = plt_fcn(sys1, label='line1')
    out = plt_fcn(sys2, label='line2')
    axs = ct.get_plot_axes(out)
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()

    if plt_fcn == ct.bode_plot:
        # Multi-dimensional data
        sys1 = ct.rss(2, 2, 2, name='sys1')
        sys2 = ct.rss(3, 2, 2, name='sys2')

        # Check out some errors first
        with pytest.raises(ValueError, match="number of labels must match"):
            ct.bode_plot([sys1, sys2], label=['line1'])

        with pytest.xfail(reason="need better broadcast checking on labels"):
            with pytest.raises(
                    ValueError, match="labels must be given for each"):
                ct.bode_plot(sys1, overlay_inputs=True, label=['line1'])

        # Now do things that should work
        out = ct.bode_plot(
            [sys1, sys2],
            label=[
                [['line1', 'line1'], ['line1', 'line1']],
                [['line2', 'line2'], ['line2', 'line2']],
            ])
        axs = ct.get_plot_axes(out)
        legend = axs[0, -1].get_legend().get_texts()
        assert legend[0].get_text() == 'line1'
        assert legend[1].get_text() == 'line2'
        plt.close()


@pytest.mark.parametrize(
    "plt_fcn", [
        ct.bode_plot, ct.singular_values_plot, ct.nyquist_plot,
        ct.nichols_plot])
@pytest.mark.parametrize(
    "ninputs, noutputs", [(1, 1), (1, 2), (2, 1), (2, 3)])
@pytest.mark.usefixtures("editsdefaults")
def test_freqplot_ax_keyword(plt_fcn, ninputs, noutputs):
    if plt_fcn in [ct.nyquist_plot, ct.nichols_plot] and \
       (ninputs != 1 or noutputs != 1):
        pytest.skip("MIMO not implemented for Nyquist/Nichols")

    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', suptitle_frame='figure')

    # System to use
    sys = ct.rss(4, ninputs, noutputs)

    # Create an initial figure
    out1 = plt_fcn(sys)

    # Draw again on the same figure, using array
    axs = ct.get_plot_axes(out1)
    out2 = plt_fcn(sys, ax=axs)
    np.testing.assert_equal(ct.get_plot_axes(out1), ct.get_plot_axes(out2))

    # Pass things in as a list instead
    axs_list = axs.tolist()
    out3 = plt_fcn(sys, ax=axs)
    np.testing.assert_equal(ct.get_plot_axes(out1), ct.get_plot_axes(out3))

    # Flatten the list
    axs_list = axs.squeeze().tolist()
    out3 = plt_fcn(sys, ax=axs_list)
    np.testing.assert_equal(ct.get_plot_axes(out1), ct.get_plot_axes(out3))


def test_mixed_systypes():
    s = ct.tf('s')
    sys_tf = ct.tf(
        (0.02 * s**3 - 0.1 * s) / (s**4 + s**3 + s**2 + 0.25 * s + 0.04),
        name='tf')
    sys_ss = ct.ss(sys_tf * 2, name='ss')
    sys_frd1 = ct.frd(sys_tf / 2, np.logspace(-1, 1, 15), name='frd1')
    sys_frd2 = ct.frd(sys_tf / 4, np.logspace(-3, 2, 20), name='frd2')

    # Simple case: compute responses separately and plot
    resp_tf = ct.frequency_response(sys_tf)
    resp_ss = ct.frequency_response(sys_ss)
    plt.figure()
    ct.bode_plot([resp_tf, resp_ss, sys_frd1, sys_frd2], plot_phase=False)
    ct.suptitle("bode_plot([resp_tf, resp_ss, sys_frd1, sys_frd2])")

    # Same thing, but using frequency response
    plt.figure()
    resp = ct.frequency_response([sys_tf, sys_ss, sys_frd1, sys_frd2])
    resp.plot(plot_phase=False)
    ct.suptitle("frequency_response([sys_tf, sys_ss, sys_frd1, sys_frd2])")

    # Same thing, but using bode_plot
    plt.figure()
    resp = ct.bode_plot([sys_tf, sys_ss, sys_frd1, sys_frd2], plot_phase=False)
    ct.suptitle("bode_plot([sys_tf, sys_ss, sys_frd1, sys_frd2])")


def test_suptitle():
    sys = ct.rss(2, 2, 2)

    # Default location: center of axes
    out = ct.bode_plot(sys)
    assert plt.gcf()._suptitle._x != 0.5

    # Try changing the the title
    ct.suptitle("New title")
    assert plt.gcf()._suptitle._text == "New title"

    # Change the location of the title
    ct.suptitle("New title", frame='figure')
    assert plt.gcf()._suptitle._x == 0.5

    # Change the location of the title back
    ct.suptitle("New title", frame='axes')
    assert plt.gcf()._suptitle._x != 0.5

    # Bad frame
    with pytest.raises(ValueError, match="unknown"):
        ct.suptitle("New title", frame='nowhere')

    # Bad keyword
    with pytest.raises(AttributeError, match="unexpected keyword|no property"):
        ct.suptitle("New title", unknown=None)


@pytest.mark.parametrize("plt_fcn", [ct.bode_plot, ct.singular_values_plot])
def test_freqplot_errors(plt_fcn):
    if plt_fcn == ct.bode_plot:
        # Turning off both magnitude and phase
        with pytest.raises(ValueError, match="no data to plot"):
            ct.bode_plot(
                manual_response, plot_magnitude=False, plot_phase=False)

    # Specifying frequency parameters with response data
    response = ct.singular_values_response(ct.rss(2, 1, 1))
    with pytest.warns(UserWarning, match="`omega_num` ignored "):
        plt_fcn(response, omega_num=100)
    with pytest.warns(UserWarning, match="`omega` ignored "):
        plt_fcn(response, omega=np.logspace(-2, 2))

    # Bad frequency limits
    with pytest.raises(ValueError, match="invalid limits"):
        plt_fcn(response, omega_limits=[1e2, 1e-2])


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
        test_response_plots(*args, ovlinp=False, ovlout=False, clear=False)

    # Reset suptitle_frame to the default value
    ct.reset_defaults()

    # Define and run a selected set of interesting tests
    # TODO: TBD (see timeplot_test.py for format)

    test_basic_freq_plots(savefigs=True)
    test_gangof4_plots(savefigs=True)

    #
    # Run a few more special cases to show off capabilities (and save some
    # of them for use in the documentation).
    #
    test_mixed_systypes()
