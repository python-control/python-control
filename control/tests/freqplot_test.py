# freqplot_test.py - test out frequency response plots
# RMM, 23 Jun 2023

import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import control as ct

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
    ct.set_defaults('freqplot', title_frame='figure')

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
    cplt = response.plot(**kwargs)

    # Check the shape
    if ovlout and ovlinp:
        assert cplt.lines.shape == (pltmag + pltphs, 1)
    elif ovlout:
        assert cplt.lines.shape == (pltmag + pltphs, sys.ninputs)
    elif ovlinp:
        assert cplt.lines.shape == (sys.noutputs * (pltmag + pltphs), 1)
    else:
        assert cplt.lines.shape == \
            (sys.noutputs * (pltmag + pltphs), sys.ninputs)

    # Make sure all of the outputs are of the right type
    nlines_plotted = 0
    for ax_lines in np.nditer(cplt.lines, flags=["refs_ok"]):
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
    cplt.set_plot_title(
        cplt.figure._suptitle._text +
        f" [{sys.noutputs}x{sys.ninputs}, pm={pltmag}, pp={pltphs},"
        f" sm={shrmag}, sp={shrphs}, sf={shrfrq}]",     # TODO: ", "
        # f"oo={ovlout}, oi={ovlinp}, ss={secsys}]",    # TODO: add back
        frame='figure')

    # Get rid of the figure to free up memory
    if clear:
        plt.close('.Figure')


# Use the manaul response to verify that different settings are working
def test_manual_response_limits():
    # Default response: limits should be the same across rows
    cplt = manual_response.plot()
    axs = cplt.axes
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
    ct.set_defaults('freqplot', title_frame='figure')

    # Define a couple of systems for testing
    sys1 = ct.tf([1], [1, 2, 1], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    sys3 = ct.tf([0.2, 0.1], [1, 0.1, 0.3, 0.1, 0.1], name='sys3')

    # Create a plot for the first system, with custom styles
    plt_fcn(sys1)

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
    ct.set_defaults('freqplot', title_frame='figure')

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
    ct.set_defaults('freqplot', title_frame='figure')

    # Default sharing should share along rows and cols for mag and phase
    cplt = ct.bode_plot(manual_response)
    axs = cplt.axes
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # Share y limits along rows
            assert axs[i, j].get_ylim() == axs[i, 0].get_ylim()

            # Share x limits along columns
            assert axs[i, j].get_xlim() == axs[-1, j].get_xlim()

    # Sharing along y axis for mag but not phase
    plt.figure()
    cplt = ct.bode_plot(manual_response, share_phase='none')
    axs = cplt.axes
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
    cplt = ct.bode_plot(manual_response, sharey='none')
    axs = cplt.axes
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
    cplt = ct.bode_plot(manual_response, sharex='none')
    # TODO: figure out what to check


@pytest.mark.parametrize("plot_type", ['bode', 'svplot', 'nichols'])
def test_freqplot_plot_type(plot_type):
    if plot_type == 'svplot':
        response = ct.singular_values_response(ct.rss(2, 1, 1))
    else:
        response = ct.frequency_response(ct.rss(2, 1, 1))
    cplt = response.plot(plot_type=plot_type)
    if plot_type == 'bode':
        assert cplt.lines.shape == (2, 1)
    else:
        assert cplt.lines.shape == (1, )

@pytest.mark.parametrize("plt_fcn", [ct.bode_plot, ct.singular_values_plot])
@pytest.mark.usefixtures("editsdefaults")
def test_freqplot_omega_limits(plt_fcn):
    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', title_frame='figure')

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
    cplt = plt_fcn(response)
    ax = cplt.axes
    np.testing.assert_allclose(
        _get_visible_limits(ax.reshape(-1)[0]), np.array([0.1, 10]))

    # Now reset the limits
    cplt = plt_fcn(response, omega_limits=(1, 100))
    ax = cplt.axes
    np.testing.assert_allclose(
        _get_visible_limits(ax.reshape(-1)[0]), np.array([1, 100]))


def test_gangof4_trace_labels():
    P1 = ct.rss(2, 1, 1, name='P1')
    P2 = ct.rss(3, 1, 1, name='P2')
    C1 = ct.rss(1, 1, 1, name='C1')
    C2 = ct.rss(1, 1, 1, name='C2')

    # Make sure default labels are as expected
    cplt = ct.gangof4_response(P1, C1).plot()
    cplt = ct.gangof4_response(P2, C2).plot()
    axs = cplt.axes
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'P=P1, C=C1'
    assert legend[1].get_text() == 'P=P2, C=C2'
    plt.close()

    # Suffix truncation
    cplt = ct.gangof4_response(P1, C1).plot()
    cplt = ct.gangof4_response(P2, C1).plot()
    axs = cplt.axes
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'P=P1'
    assert legend[1].get_text() == 'P=P2'
    plt.close()

    # Prefix turncation
    cplt = ct.gangof4_response(P1, C1).plot()
    cplt = ct.gangof4_response(P1, C2).plot()
    axs = cplt.axes
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'C=C1'
    assert legend[1].get_text() == 'C=C2'
    plt.close()

    # Override labels
    cplt = ct.gangof4_response(P1, C1).plot(label='xxx, line1, yyy')
    cplt = ct.gangof4_response(P2, C2).plot(label='xxx, line2, yyy')
    axs = cplt.axes
    legend = axs[0, 1].get_legend().get_texts()
    assert legend[0].get_text() == 'xxx, line1, yyy'
    assert legend[1].get_text() == 'xxx, line2, yyy'
    plt.close()


@pytest.mark.parametrize(
    "plt_fcn", [ct.bode_plot, ct.singular_values_plot, ct.nyquist_plot])
@pytest.mark.usefixtures("editsdefaults")
def test_freqplot_line_labels(plt_fcn):
    sys1 = ct.rss(2, 1, 1, name='sys1')
    sys2 = ct.rss(3, 1, 1, name='sys2')

    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', title_frame='figure')

    # Make sure default labels are as expected
    cplt = plt_fcn([sys1, sys2])
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'sys1'
    assert legend[1].get_text() == 'sys2'
    plt.close()

    # Override labels all at once
    cplt = plt_fcn([sys1, sys2], label=['line1', 'line2'])
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()

    # Override labels one at a time
    cplt = plt_fcn(sys1, label='line1')
    cplt = plt_fcn(sys2, label='line2')
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, 0].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()


@pytest.mark.skip(reason="line label override not yet implemented")
@pytest.mark.parametrize("kwargs, labels", [
    ({}, ['sys1', 'sys2']),
    ({'overlay_outputs': True}, [
        'x sys1 out1 y', 'x sys1 out2 y', 'x sys2 out1 y', 'x sys2 out2 y']),
])
def test_line_labels_bode(kwargs, labels):
    # Multi-dimensional data
    sys1 = ct.rss(2, 2, 2)
    sys2 = ct.rss(3, 2, 2)

    # Check out some errors first
    with pytest.raises(ValueError, match="number of labels must match"):
        ct.bode_plot([sys1, sys2], label=['line1'])

    cplt = ct.bode_plot([sys1, sys2], label=labels, **kwargs)
    axs = cplt.axes
    legend_texts = axs[0, -1].get_legend().get_texts()
    for i, legend in enumerate(legend_texts):
        assert legend.get_text() == labels[i]
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
    ct.set_defaults('freqplot', title_frame='figure')

    # System to use
    sys = ct.rss(4, ninputs, noutputs)

    # Create an initial figure
    cplt1 = plt_fcn(sys)

    # Draw again on the same figure, using array
    axs = cplt1.axes
    cplt2 = plt_fcn(sys, ax=axs)
    np.testing.assert_equal(cplt1.axes, cplt2.axes)

    # Pass things in as a list instead
    axs_list = axs.tolist()
    cplt3 = plt_fcn(sys, ax=axs)
    np.testing.assert_equal(cplt1.axes, cplt3.axes)

    # Flatten the list
    axs_list = axs.squeeze().tolist()
    cplt4 = plt_fcn(sys, ax=axs_list)
    np.testing.assert_equal(cplt1.axes, cplt4.axes)


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
    cplt = ct.bode_plot(
        [resp_tf, resp_ss, sys_frd1, sys_frd2], plot_phase=False)
    cplt.set_plot_title("bode_plot([resp_tf, resp_ss, sys_frd1, sys_frd2])")

    # Same thing, but using frequency response
    plt.figure()
    resp = ct.frequency_response([sys_tf, sys_ss, sys_frd1, sys_frd2])
    cplt = resp.plot(plot_phase=False)
    cplt.set_plot_title(
        "frequency_response([sys_tf, sys_ss, sys_frd1, sys_frd2])")

    # Same thing, but using bode_plot
    plt.figure()
    cplt = ct.bode_plot([sys_tf, sys_ss, sys_frd1, sys_frd2], plot_phase=False)
    cplt.set_plot_title("bode_plot([sys_tf, sys_ss, sys_frd1, sys_frd2])")


def test_suptitle():
    sys = ct.rss(2, 2, 2, strictly_proper=True)

    # Default location: center of axes
    cplt = ct.bode_plot(sys)
    assert plt.gcf()._suptitle._x != 0.5

    # Try changing the the title
    cplt.set_plot_title("New title")
    assert plt.gcf()._suptitle._text == "New title"

    # Change the location of the title
    cplt.set_plot_title("New title", frame='figure')
    assert plt.gcf()._suptitle._x == 0.5

    # Change the location of the title back
    cplt.set_plot_title("New title", frame='axes')
    assert plt.gcf()._suptitle._x != 0.5

    # Bad frame
    with pytest.raises(ValueError, match="unknown"):
        cplt.set_plot_title("New title", frame='nowhere')

    # Bad keyword
    with pytest.raises(
            TypeError, match="unexpected keyword|no property"):
        cplt.set_plot_title("New title", unknown=None)

    # Make sure title is still there if we display margins underneath
    sys = ct.rss(2, 1, 1, name='sys')
    cplt = ct.bode_plot(sys, display_margins=True)
    assert re.match(r"^Bode plot for sys$", cplt.figure._suptitle._text)
    assert re.match(r"^sys: Gm = .*, Pm = .*$", cplt.axes[0, 0].get_title())


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


def test_freqresplist_unknown_kw():
    sys1 = ct.rss(2, 1, 1)
    sys2 = ct.rss(2, 1, 1)
    resp = ct.frequency_response([sys1, sys2])
    assert isinstance(resp, ct.FrequencyResponseList)

    with pytest.raises(AttributeError, match="unexpected keyword"):
        resp.plot(unknown=True)

@pytest.mark.parametrize("nsys, display_margins, gridkw, match", [
    (1, True, {}, None),
    (1, False, {}, None),
    (1, False, {}, None),
    (1, True, {'grid': True}, None),
    (1, 'overlay', {}, None),
    (1, 'overlay', {'grid': True}, None),
    (1, 'overlay', {'grid': False}, None),
    (2, True, {}, None),
    (2, 'overlay', {}, "not supported for multi-trace plots"),
    (2, True, {'grid': 'overlay'}, None),
    (3, True, {'grid': True}, None),
])
def test_display_margins(nsys, display_margins, gridkw, match):
    sys1 = ct.tf([10], [1, 1, 1, 1], name='sys1')
    sys2 = ct.tf([20], [2, 2, 2, 1], name='sys2')
    sys3 = ct.tf([30], [2, 3, 3, 1], name='sys3')

    sysdata = [sys1, sys2, sys3][0:nsys]

    plt.figure()
    if match is None:
        cplt = ct.bode_plot(sysdata, display_margins=display_margins, **gridkw)
    else:
        with pytest.raises(NotImplementedError, match=match):
            ct.bode_plot(sysdata, display_margins=display_margins, **gridkw)
        return

    cplt.set_plot_title(
        cplt.figure._suptitle._text + f" [d_m={display_margins}, {gridkw=}")

    # Make sure the grid is there if it should be
    if gridkw.get('grid') or not display_margins:
        assert all(
            [line.get_visible() for line in cplt.axes[0, 0].get_xgridlines()])
    else:
        assert not any(
            [line.get_visible() for line in cplt.axes[0, 0].get_xgridlines()])

    # Make sure margins are displayed
    if display_margins == True:
        ax_title = cplt.axes[0, 0].get_title()
        assert len(ax_title.split('\n')) == nsys
    elif display_margins == 'overlay':
        assert cplt.axes[0, 0].get_title() == ''


def test_singular_values_plot_colors():
    # Define some systems for testing
    sys1 = ct.rss(4, 2, 2, strictly_proper=True)
    sys2 = ct.rss(4, 2, 2, strictly_proper=True)

    # Get the default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot the systems individually and make sure line colors are OK
    cplt = ct.singular_values_plot(sys1)
    assert cplt.lines.size == 1
    assert len(cplt.lines[0]) == 2
    assert cplt.lines[0][0].get_color() == color_cycle[0]
    assert cplt.lines[0][1].get_color() == color_cycle[0]

    cplt = ct.singular_values_plot(sys2)
    assert cplt.lines.size == 1
    assert len(cplt.lines[0]) == 2
    assert cplt.lines[0][0].get_color() == color_cycle[1]
    assert cplt.lines[0][1].get_color() == color_cycle[1]
    plt.close('all')

    # Plot the systems as a list and make sure colors are OK
    cplt = ct.singular_values_plot([sys1, sys2])
    assert cplt.lines.size == 2
    assert len(cplt.lines[0]) == 2
    assert len(cplt.lines[1]) == 2
    assert cplt.lines[0][0].get_color() == color_cycle[0]
    assert cplt.lines[0][1].get_color() == color_cycle[0]
    assert cplt.lines[1][0].get_color() == color_cycle[1]
    assert cplt.lines[1][1].get_color() == color_cycle[1]


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

    # Reset title_frame to the default value
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
    test_display_margins(2, True, {})
    test_display_margins(2, 'overlay', {})
    test_display_margins(2, True, {'grid': True})
