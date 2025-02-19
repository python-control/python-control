# timeplot_test.py - test out time response plots
# RMM, 23 Jun 2023

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import control as ct
from control.tests.conftest import slycotonly

# Detailed test of (almost) all functionality
#
# The commented out rows lead to very long testing times => these should be
# used only for developmental testing and not day-to-day testing.
@pytest.mark.parametrize(
    "sys", [
        # ct.rss(1, 1, 1, strictly_proper=True, name="rss"),
        ct.nlsys(
            lambda t, x, u, params: -x + u, None,
            inputs=1, outputs=1, states=1, name="nlsys"),
        # ct.rss(2, 1, 2, strictly_proper=True, name="rss"),
        ct.rss(2, 2, 1, strictly_proper=True, name="rss"),
        # ct.drss(2, 2, 2, name="drss"),
        # ct.rss(2, 2, 3, strictly_proper=True, name="rss"),
    ])
# @pytest.mark.parametrize("transpose", [False, True])
# @pytest.mark.parametrize("plot_inputs", [False, None, True, 'overlay'])
# @pytest.mark.parametrize("plot_outputs", [True, False])
# @pytest.mark.parametrize("overlay_signals", [False, True])
# @pytest.mark.parametrize("overlay_traces", [False, True])
# @pytest.mark.parametrize("second_system", [False, True])
# @pytest.mark.parametrize("fcn", [
#     ct.step_response, ct.impulse_response, ct.initial_response,
#     ct.forced_response])
@pytest.mark.parametrize(       # combinatorial-style test (faster)
    "fcn,                  pltinp,    pltout, cmbsig, cmbtrc, trpose, secsys",
    [(ct.step_response,    False,     True,   False,  False,  False,  False),
     (ct.step_response,    None,      True,   False,  False,  False,  False),
     (ct.step_response,    True,      True,   False,  False,  False,  False),
     (ct.step_response,    'overlay', True,   False,  False,  False,  False),
     (ct.step_response,    'overlay', True,   True,   False,  False,  False),
     (ct.step_response,    'overlay', True,   False,  True,   False,  False),
     (ct.step_response,    'overlay', True,   False,  False,  True,   False),
     (ct.step_response,    'overlay', True,   False,  False,  False,  True),
     (ct.step_response,    False,     False,  False,  False,  False,  False),
     (ct.step_response,    None,      False,  False,  False,  False,  False),
     (ct.step_response,    'overlay', False,  False,  False,  False,  False),
     (ct.step_response,    True,      True,   False,  True,   False,  False),
     (ct.step_response,    True,      True,   False,  False,  False,  True),
     (ct.step_response,    True,      True,   False,  True,   False,  True),
     (ct.step_response,    True,      True,   True,   False,   True,   True),
     (ct.step_response,    True,      True,   False,  True,    True,   True),
     (ct.impulse_response, False,     True,   True,   False,  False,  False),
     (ct.initial_response, None,      True,   False,  False,  False,  False),
     (ct.initial_response, False,     True,   False,  False,  False,  False),
     (ct.initial_response, True,      True,   False,  False,  False,  False),
     (ct.forced_response,  True,      True,   False,  False,  False,  False),
     (ct.forced_response,  None,      True,   False,  False,  False,  False),
     (ct.forced_response,  False,     True,   False,  False,  False,  False),
     (ct.forced_response,  True,      True,   True,   False,  False,  False),
     (ct.forced_response,  True,      True,   True,   True,   False,  False),
     (ct.forced_response,  True,      True,   True,   True,   True,   False),
     (ct.forced_response,  True,      True,   True,   True,   True,   True),
     (ct.forced_response,  'overlay', True,   True,   True,   False,  True),
     (ct.input_output_response,
                           True,      True,   False,  False,  False,  False),
     ])

@pytest.mark.usefixtures('mplcleanup')
def test_response_plots(
        fcn, sys, pltinp, pltout, cmbsig, cmbtrc,
        trpose, secsys, clear=True):
    # Figure out the time range to use and check some special cases
    if not isinstance(sys, ct.lti.LTI):
        if fcn == ct.impulse_response:
            pytest.skip("impulse response not implemented for nlsys")

        # Nonlinear systems require explicit time limits
        T = 10
        timepts = np.linspace(0, T)

    elif isinstance(sys, ct.TransferFunction) and fcn == ct.initial_response:
        pytest.skip("initial response not tested for tf")

    else:
        # Linear systems figure things out on their own
        T = None
        timepts = np.linspace(0, 10)    # for input_output_response

    # Save up the keyword arguments
    kwargs = dict(
        plot_inputs=pltinp, plot_outputs=pltout, transpose=trpose,
        overlay_signals=cmbsig, overlay_traces=cmbtrc)

    # Create the response
    if fcn is ct.input_output_response and \
       not isinstance(sys, ct.NonlinearIOSystem):
        # Skip transfer functions and other non-state space systems
        return None
    if fcn in [ct.input_output_response, ct.forced_response]:
        U = np.zeros((sys.ninputs, timepts.size))
        for i in range(sys.ninputs):
            U[i] = np.cos(timepts * i + i)
        args = [timepts, U]

    elif fcn == ct.initial_response:
        args = [T, np.ones(sys.nstates)]   # T, X0

    elif not isinstance(sys, ct.lti.LTI):
        args = [T]              # nonlinear systems require final time

    else:                       # step, initial, impulse responses
        args = []

    # Create a new figure (in case previous one is of the same size) and plot
    if not clear:
        plt.figure()
    response = fcn(sys, *args)

    # Look for cases where there are no data to plot
    if not pltout and (
            pltinp is False or response.ninputs == 0 or
            pltinp is None and response.plot_inputs is False):
        with pytest.raises(ValueError, match=".* no data to plot"):
            cplt = response.plot(**kwargs)
        return None
    elif not pltout and pltinp == 'overlay':
        with pytest.raises(ValueError, match="can't overlay inputs"):
            cplt = response.plot(**kwargs)
        return None
    elif pltinp in [True, 'overlay'] and response.ninputs == 0:
        with pytest.raises(ValueError, match=".* but no inputs"):
            cplt = response.plot(**kwargs)
        return None

    cplt = response.plot(**kwargs)

    # Make sure all of the outputs are of the right type
    nlines_plotted = 0
    for ax_lines in np.nditer(cplt.lines, flags=["refs_ok"]):
        for line in ax_lines.item():
            assert isinstance(line, mpl.lines.Line2D)
            nlines_plotted += 1

    # Make sure number of plots is correct
    if pltinp is None:
        if fcn in [ct.forced_response, ct.input_output_response]:
            pltinp = True
        else:
            pltinp = False
    ntraces = max(1, response.ntraces)
    nlines_expected = (response.ninputs if pltinp else 0) * ntraces + \
        (response.noutputs if pltout else 0) * ntraces
    assert nlines_plotted == nlines_expected

    # Save the old axes to compare later
    old_axes = plt.gcf().get_axes()

    # Add additional data (and provide info in the title)
    if secsys:
        newsys = ct.rss(
            sys.nstates, sys.noutputs, sys.ninputs, strictly_proper=True)
        if fcn not in [ct.initial_response, ct.forced_response,
                       ct.input_output_response] and \
           isinstance(sys, ct.lti.LTI):
            # Reuse the previously computed time to make plots look nicer
            fcn(newsys, *args, T=response.time[-1]).plot(**kwargs)
        else:
            # Compute and plot new response (time is one of the arguments)
            fcn(newsys, *args).plot(**kwargs)

        # Make sure we have the same axes
        new_axes = plt.gcf().get_axes()
        assert new_axes == old_axes

        # Make sure every axes has more than one line
        for ax in new_axes:
            assert len(ax.get_lines()) > 1

    # Update the title so we can see what is going on
    fig = cplt.figure
    fig.suptitle(
        fig._suptitle._text +
        f" [{sys.noutputs}x{sys.ninputs}, cs={cmbsig}, "
        f"ct={cmbtrc}, pi={pltinp}, tr={trpose}]",
        fontsize='small')

    # Get rid of the figure to free up memory
    if clear:
        plt.clf()


@pytest.mark.usefixtures('mplcleanup')
def test_axes_setup():
    sys_2x3 = ct.rss(4, 2, 3)
    sys_2x3b = ct.rss(4, 2, 3)
    sys_3x2 = ct.rss(4, 3, 2)
    sys_3x1 = ct.rss(4, 3, 1)

    # Two plots of the same size leaves axes unchanged
    cplt1 = ct.step_response(sys_2x3).plot()
    cplt2 = ct.step_response(sys_2x3b).plot()
    np.testing.assert_equal(cplt1.axes, cplt2.axes)
    plt.close()

    # Two plots of same net size leaves axes unchanged (unfortunately)
    cplt1 = ct.step_response(sys_2x3).plot()
    cplt2 = ct.step_response(sys_3x2).plot()
    np.testing.assert_equal(
        cplt1.axes.reshape(-1), cplt2.axes.reshape(-1))
    plt.close()

    # Plots of different shapes generate new plots
    cplt1 = ct.step_response(sys_2x3).plot()
    cplt2 = ct.step_response(sys_3x1).plot()
    ax1_list = cplt1.axes.reshape(-1).tolist()
    ax2_list = cplt2.axes.reshape(-1).tolist()
    for ax in ax1_list:
        assert ax not in ax2_list
    plt.close()

    # Passing a list of axes preserves those axes
    cplt1 = ct.step_response(sys_2x3).plot()
    cplt2 = ct.step_response(sys_3x1).plot()
    cplt3 = ct.step_response(sys_2x3b).plot(ax=cplt1.axes)
    np.testing.assert_equal(cplt1.axes, cplt3.axes)
    plt.close()

    # Sending an axes array of the wrong size raises exception
    with pytest.raises(ValueError, match="not the right shape"):
        cplt = ct.step_response(sys_2x3).plot()
        ct.step_response(sys_3x1).plot(ax=cplt.axes)
    sys_2x3 = ct.rss(4, 2, 3)
    sys_2x3b = ct.rss(4, 2, 3)
    sys_3x2 = ct.rss(4, 3, 2)
    sys_3x1 = ct.rss(4, 3, 1)


@slycotonly
@pytest.mark.usefixtures('mplcleanup')
def test_legend_map():
    sys_mimo = ct.tf2ss(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
    response = ct.step_response(sys_mimo)
    response.plot(
        legend_map=np.array([['center', 'upper right'],
                             [None, 'center right']]),
        plot_inputs=True, overlay_signals=True, transpose=True,
        title='MIMO step response with custom legend placement')


@pytest.mark.usefixtures('mplcleanup')
def test_combine_time_responses():
    sys_mimo = ct.rss(4, 2, 2)
    timepts = np.linspace(0, 10, 100)

    # Combine two responses with ntrace = 0
    U = np.vstack([np.sin(timepts), np.cos(2*timepts)])
    resp1 = ct.input_output_response(sys_mimo, timepts, U)

    U = np.vstack([np.cos(2*timepts), np.sin(timepts)])
    resp2 = ct.input_output_response(sys_mimo, timepts, U)

    combresp1 = ct.combine_time_responses([resp1, resp2])
    assert combresp1.ntraces == 2
    np.testing.assert_equal(combresp1.y[:, 0, :], resp1.y)
    np.testing.assert_equal(combresp1.y[:, 1, :], resp2.y)

    # Combine two responses with ntrace != 0
    resp3 = ct.step_response(sys_mimo, timepts)
    resp4 = ct.step_response(sys_mimo, timepts)
    combresp2 = ct.combine_time_responses([resp3, resp4])
    assert combresp2.ntraces == resp3.ntraces + resp4.ntraces
    np.testing.assert_equal(combresp2.y[:, 0:2, :], resp3.y)
    np.testing.assert_equal(combresp2.y[:, 2:4, :], resp4.y)

    # Mixture
    combresp3 = ct.combine_time_responses([resp1, resp2, resp3])
    assert combresp3.ntraces == resp3.ntraces + resp4.ntraces
    np.testing.assert_equal(combresp3.y[:, 0, :], resp1.y)
    np.testing.assert_equal(combresp3.y[:, 1, :], resp2.y)
    np.testing.assert_equal(combresp3.y[:, 2:4, :], resp3.y)
    assert combresp3.trace_types == [None, None] + resp3.trace_types
    assert combresp3.trace_labels == \
        [resp1.title, resp2.title] + resp3.trace_labels

    # Rename the traces
    labels = ["T1", "T2", "T3", "T4"]
    combresp4 = ct.combine_time_responses(
        [resp1, resp2, resp3], trace_labels=labels)
    assert combresp4.trace_labels == labels
    assert combresp4.trace_types == [None, None, 'step', 'step']

    # Automatically generated trace label names and types
    resp5 = ct.step_response(sys_mimo, timepts)
    resp5.title = "test"
    resp5.trace_labels = None
    resp5.trace_types = None
    combresp5 = ct.combine_time_responses([resp1, resp5])
    assert combresp5.trace_labels == [resp1.title] + \
        ["test, trace 0", "test, trace 1"]
    assert combresp5.trace_types == [None, None, None]

    # ntraces = 0 with trace_types != None
    # https://github.com/python-control/python-control/issues/1025
    resp6 = ct.forced_response(sys_mimo, timepts, U)
    combresp6 = ct.combine_time_responses([resp1, resp6])
    assert combresp6.trace_types == [None, 'forced']

    with pytest.raises(ValueError, match="must have the same number"):
        resp = ct.step_response(ct.rss(4, 2, 3), timepts)
        ct.combine_time_responses([resp1, resp])

    with pytest.raises(ValueError, match="trace labels does not match"):
        ct.combine_time_responses(
            [resp1, resp2], trace_labels=["T1", "T2", "T3"])

    with pytest.raises(ValueError, match="must have the same time"):
        resp = ct.step_response(ct.rss(4, 2, 3), timepts/2)
        ct.combine_time_responses([resp1, resp])


@pytest.mark.parametrize("resp_fcn", [
    ct.step_response, ct.initial_response, ct.impulse_response,
    ct.forced_response, ct.input_output_response])
@pytest.mark.usefixtures('mplcleanup')
def test_list_responses(resp_fcn):
    sys1 = ct.rss(2, 2, 2, strictly_proper=True)
    sys2 = ct.rss(2, 2, 2, strictly_proper=True)

    # Figure out the expected shape of the system
    match resp_fcn:
        case ct.step_response | ct.impulse_response:
            shape = (2, 2)
            kwargs = {}
        case ct.initial_response:
            shape = (2, 1)
            kwargs = {}
        case ct.forced_response | ct.input_output_response:
            shape = (4, 1)      # outputs and inputs both plotted
            T = np.linspace(0, 10)
            U = [np.sin(T), np.cos(T)]
            kwargs = {'T': T, 'U': U}

    resp1 = resp_fcn(sys1, **kwargs)
    resp2 = resp_fcn(sys2, **kwargs)

    # Sequential plotting results in colors rotating
    plt.figure()
    cplt1 = resp1.plot()
    cplt2 = resp2.plot()
    assert cplt1.shape == shape         # legacy access (OK here)
    assert cplt2.shape == shape         # legacy access (OK here)
    for row in range(2):        # just look at the outputs
        for col in range(shape[1]):
            assert cplt1.lines[row, col][0].get_color() == 'tab:blue'
            assert cplt2.lines[row, col][0].get_color() == 'tab:orange'

    plt.figure()
    resp_combined = resp_fcn([sys1, sys2], **kwargs)
    assert isinstance(resp_combined, ct.timeresp.TimeResponseList)
    assert resp_combined[0].time[-1] == max(resp1.time[-1], resp2.time[-1])
    assert resp_combined[1].time[-1] == max(resp1.time[-1], resp2.time[-1])
    cplt = resp_combined.plot()
    assert cplt.lines.shape == shape
    for row in range(2):        # just look at the outputs
        for col in range(shape[1]):
            assert cplt.lines[row, col][0].get_color() == 'tab:blue'
            assert cplt.lines[row, col][1].get_color() == 'tab:orange'


@slycotonly
@pytest.mark.usefixtures('mplcleanup')
def test_linestyles():
    # Check to make sure we can change line styles
    sys_mimo = ct.tf2ss(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
    cplt = ct.step_response(sys_mimo).plot('k--', plot_inputs=True)
    for ax in np.nditer(cplt.lines, flags=["refs_ok"]):
        for line in ax.item():
            assert line.get_color() == 'k'
            assert line.get_linestyle() == '--'

    cplt = ct.step_response(sys_mimo).plot(
        plot_inputs='overlay', overlay_signals=True, overlay_traces=True,
        output_props=[{'color': c} for c in ['blue', 'orange']],
        input_props=[{'color': c} for c in ['red', 'green']],
        trace_props=[{'linestyle': s} for s in ['-', '--']])

    assert cplt.lines.shape == (1, 1)
    lines = cplt.lines[0, 0]
    assert lines[0].get_color() == 'blue' and lines[0].get_linestyle() == '-'
    assert lines[1].get_color() == 'orange' and lines[1].get_linestyle() == '-'
    assert lines[2].get_color() == 'red' and lines[2].get_linestyle() == '-'
    assert lines[3].get_color() == 'green' and lines[3].get_linestyle() == '-'
    assert lines[4].get_color() == 'blue' and lines[4].get_linestyle() == '--'
    assert lines[5].get_color() == 'orange' and lines[5].get_linestyle() == '--'
    assert lines[6].get_color() == 'red' and lines[6].get_linestyle() == '--'
    assert lines[7].get_color() == 'green' and lines[7].get_linestyle() == '--'


@pytest.mark.parametrize("resp_fcn", [
    ct.step_response, ct.initial_response, ct.impulse_response,
    ct.forced_response, ct.input_output_response])
@pytest.mark.usefixtures('editsdefaults', 'mplcleanup')
def test_timeplot_trace_labels(resp_fcn):
    plt.close('all')
    sys1 = ct.rss(2, 2, 2, strictly_proper=True, name='sys1')
    sys2 = ct.rss(2, 2, 2, strictly_proper=True, name='sys2')

    # Figure out the expected shape of the system
    match resp_fcn:
        case ct.step_response | ct.impulse_response:
            kwargs = {}
        case ct.initial_response:
            kwargs = {}
        case ct.forced_response | ct.input_output_response:
            T = np.linspace(0, 10)
            U = [np.sin(T), np.cos(T)]
            kwargs = {'T': T, 'U': U}

    # Use figure frame for suptitle to speed things up
    ct.set_defaults('freqplot', title_frame='figure')

    # Make sure default labels are as expected
    cplt = resp_fcn([sys1, sys2], **kwargs).plot()
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, -1].get_legend().get_texts()
    assert legend[0].get_text() == 'sys1'
    assert legend[1].get_text() == 'sys2'
    plt.close()

    # Override labels all at once
    cplt = resp_fcn([sys1, sys2], **kwargs).plot(label=['line1', 'line2'])
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, -1].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()

    # Override labels one at a time
    cplt = resp_fcn(sys1, **kwargs).plot(label='line1')
    cplt = resp_fcn(sys2, **kwargs).plot(label='line2')
    axs = cplt.axes
    if axs.ndim == 1:
        legend = axs[0].get_legend().get_texts()
    else:
        legend = axs[0, -1].get_legend().get_texts()
    assert legend[0].get_text() == 'line1'
    assert legend[1].get_text() == 'line2'
    plt.close()


@pytest.mark.usefixtures('mplcleanup')
def test_full_label_override():
    sys1 = ct.rss(2, 2, 2, strictly_proper=True, name='sys1')
    sys2 = ct.rss(2, 2, 2, strictly_proper=True, name='sys2')

    labels_2d = np.array([
        ["outsys1u1y1", "outsys1u1y2", "outsys1u2y1", "outsys1u2y2",
         "outsys2u1y1", "outsys2u1y2", "outsys2u2y1", "outsys2u2y2"],
        ["inpsys1u1y1", "inpsys1u1y2", "inpsys1u2y1", "inpsys1u2y2",
        "inpsys2u1y1", "inpsys2u1y2", "inpsys2u2y1", "inpsys2u2y2"]])


    labels_4d = np.empty((2, 2, 2, 2), dtype=object)
    for i, sys in enumerate(['sys1', 'sys2']):
        for j, trace in enumerate(['u1', 'u2']):
            for k, out in enumerate(['y1', 'y2']):
                labels_4d[i, j, k, 0] = "out" + sys + trace + out
                labels_4d[i, j, k, 1] = "inp" + sys + trace + out

    # Test 4D labels
    cplt = ct.step_response([sys1, sys2]).plot(
        overlay_signals=True, overlay_traces=True, plot_inputs=True,
        label=labels_4d)
    axs = cplt.axes
    assert axs.shape == (2, 1)
    legend_text = axs[0, 0].get_legend().get_texts()
    for i, label in enumerate(labels_2d[0]):
        assert legend_text[i].get_text() == label
    legend_text = axs[1, 0].get_legend().get_texts()
    for i, label in enumerate(labels_2d[1]):
        assert legend_text[i].get_text() == label

    # Test 2D labels
    cplt = ct.step_response([sys1, sys2]).plot(
        overlay_signals=True, overlay_traces=True, plot_inputs=True,
        label=labels_2d)
    axs = cplt.axes
    assert axs.shape == (2, 1)
    legend_text = axs[0, 0].get_legend().get_texts()
    for i, label in enumerate(labels_2d[0]):
        assert legend_text[i].get_text() == label
    legend_text = axs[1, 0].get_legend().get_texts()
    for i, label in enumerate(labels_2d[1]):
        assert legend_text[i].get_text() == label


@pytest.mark.usefixtures('mplcleanup')
def test_relabel():
    sys1 = ct.rss(2, inputs='u', outputs='y')
    sys2 = ct.rss(1, 1, 1)      # uses default i/o labels

    # Generate a plot with specific labels
    ct.step_response(sys1).plot()

    # Generate a new plot, which overwrites labels
    cplt = ct.step_response(sys2).plot()
    ax = cplt.axes
    assert ax[0, 0].get_ylabel() == 'y[0]'

    # Regenerate the first plot
    plt.figure()
    ct.step_response(sys1).plot()

    # Generate a new plt, without relabeling
    with pytest.warns(FutureWarning, match="deprecated"):
        cplt = ct.step_response(sys2).plot(relabel=False)
        assert cplt.axes[0, 0].get_ylabel() == 'y'


def test_errors():
    sys = ct.rss(2, 1, 1)
    stepresp = ct.step_response(sys)
    with pytest.raises(AttributeError,
                       match="(has no property|unexpected keyword)"):
        stepresp.plot(unknown=None)

    with pytest.raises(AttributeError,
                       match="(has no property|unexpected keyword)"):
        ct.time_response_plot(stepresp, unknown=None)

    with pytest.raises(ValueError, match="unrecognized value"):
        stepresp.plot(plot_inputs='unknown')

    for kw in ['input_props', 'output_props', 'trace_props']:
        propkw = {kw: {'color': 'green'}}
        with pytest.warns(UserWarning, match="ignored since fmt string"):
            cplt = stepresp.plot('k-', **propkw)
            assert cplt.lines[0, 0][0].get_color() == 'k'

    # Make sure TimeResponseLists also work
    stepresp = ct.step_response([sys, sys])
    with pytest.raises(AttributeError,
                       match="(has no property|unexpected keyword)"):
        stepresp.plot(unknown=None)


def test_legend_customization():
    sys = ct.rss(4, 2, 1, name='sys')
    timepts = np.linspace(0, 10)
    U = np.sin(timepts)
    resp = ct.input_output_response(sys, timepts, U)

    # Generic input/output plot
    cplt = resp.plot(overlay_signals=True)
    axs = cplt.axes
    assert axs[0, 0].get_legend()._loc == 7                 # center right
    assert len(axs[0, 0].get_legend().get_texts()) == 2
    assert axs[1, 0].get_legend() == None
    plt.close()

    # Hide legend
    cplt = resp.plot(overlay_signals=True, show_legend=False)
    axs = cplt.axes
    assert axs[0, 0].get_legend() == None
    assert axs[1, 0].get_legend() == None
    plt.close()

    # Put legend in both axes
    cplt = resp.plot(
        overlay_signals=True, legend_map=[['center left'], ['center right']])
    axs = cplt.axes
    assert axs[0, 0].get_legend()._loc == 6                 # center left
    assert len(axs[0, 0].get_legend().get_texts()) == 2
    assert axs[1, 0].get_legend()._loc == 7                 # center right
    assert len(axs[1, 0].get_legend().get_texts()) == 1
    plt.close()


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
    sys_siso = ct.tf2ss([1], [1, 2, 1], name="SISO")
    sys_mimo = ct.tf2ss(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")

    # Define and run a selected set of interesting tests
    # def test_response_plots(
    #      fcn, sys, plot_inputs, plot_outputs, overlay_signals,
    #      overlay_traces, transpose, second_system, clear=True):
    N, T, F = None, True, False
    test_cases = [
        # response fcn       system    in         out cs ct tr ss
        (ct.step_response,   sys_siso, N,         T,  F, F, F, F), # 1
        (ct.step_response,   sys_siso, T,         F,  F, F, F, F), # 2
        (ct.step_response,   sys_siso, T,         T,  F, F, F, T), # 3
        (ct.step_response,   sys_siso, 'overlay', T,  F, F, F, T), # 4
        (ct.step_response,   sys_mimo, F,         T,  F, F, F, F), # 5
        (ct.step_response,   sys_mimo, T,         T,  F, F, F, F), # 6
        (ct.step_response,   sys_mimo, 'overlay', T,  F, F, F, F), # 7
        (ct.step_response,   sys_mimo, T,         T,  T, F, F, F), # 8
        (ct.step_response,   sys_mimo, T,         T,  T, T, F, F), # 9
        (ct.step_response,   sys_mimo, T,         T,  F, F, T, F), # 10
        (ct.step_response,   sys_mimo, T,         T,  T, F, T, F), # 11
        (ct.step_response,   sys_mimo, 'overlay', T,  T, F, T, F), # 12
        (ct.forced_response, sys_mimo, N,         T,  T, F, T, F), # 13
        (ct.forced_response, sys_mimo, 'overlay', T,  F, F, F, F), # 14
    ]
    for args in test_cases:
        test_response_plots(*args, clear=F)

    #
    # Run a few more special cases to show off capabilities (and save some
    # of them for use in the documentation).
    #

    test_legend_map()           # show ability to set legend location

    # Basic step response
    plt.figure()
    ct.step_response(sys_mimo).plot()
    plt.savefig('timeplot-mimo_step-default.png')

    # Step response with plot_inputs, overlay_signals
    plt.figure()
    ct.step_response(sys_mimo).plot(
        plot_inputs=True, overlay_signals=True,
        title="Step response for 2x2 MIMO system " +
        "[plot_inputs, overlay_signals]")
    plt.savefig('timeplot-mimo_step-pi_cs.png')

    # Input/output response with overlaid inputs, legend_map
    plt.figure()
    timepts = np.linspace(0, 10, 100)
    U = np.vstack([np.sin(timepts), np.cos(2*timepts)])
    ct.input_output_response(sys_mimo, timepts, U).plot(
        plot_inputs='overlay',
        legend_map=np.array([['lower right'], ['lower right']]),
        title="I/O response for 2x2 MIMO system " +
        "[plot_inputs='overlay', legend_map]")
    plt.savefig('timeplot-mimo_ioresp-ov_lm.png')

    # Multi-trace plot, transpose
    plt.figure()
    U = np.vstack([np.sin(timepts), np.cos(2*timepts)])
    resp1 = ct.input_output_response(sys_mimo, timepts, U)

    U = np.vstack([np.cos(2*timepts), np.sin(timepts)])
    resp2 = ct.input_output_response(sys_mimo, timepts, U)

    ct.combine_time_responses(
        [resp1, resp2], trace_labels=["Scenario #1", "Scenario #2"]).plot(
            transpose=True,
            title="I/O responses for 2x2 MIMO system, multiple traces "
            "[transpose]")
    plt.savefig('timeplot-mimo_ioresp-mt_tr.png')

    plt.figure()
    cplt = ct.step_response(sys_mimo).plot(
        plot_inputs='overlay', overlay_signals=True, overlay_traces=True,
        output_props=[{'color': c} for c in ['blue', 'orange']],
        input_props=[{'color': c} for c in ['red', 'green']],
        trace_props=[{'linestyle': s} for s in ['-', '--']])
    plt.savefig('timeplot-mimo_step-linestyle.png')

    sys1 = ct.rss(4, 2, 2)
    sys2 = ct.rss(4, 2, 2)
    resp_list = ct.step_response([sys1, sys2])

    fig = plt.figure()
    cplt = ct.combine_time_responses(
        [ct.step_response(sys1, resp_list[0].time),
         ct.step_response(sys2, resp_list[1].time)]
    ).plot(overlay_traces=True)
    cplt.set_plot_title("[Combine] " + fig._suptitle._text)

    fig = plt.figure()
    ct.step_response(sys1).plot()
    cplt = ct.step_response(sys2).plot()
    cplt.set_plot_title("[Sequential] " + fig._suptitle._text)

    fig = plt.figure()
    ct.step_response(sys1).plot(color='b')
    cplt = ct.step_response(sys2).plot(color='r')
    cplt.set_plot_title("[Seq w/color] " + fig._suptitle._text)

    fig = plt.figure()
    cplt = ct.step_response([sys1, sys2]).plot()
    cplt.set_plot_title("[List] " + fig._suptitle._text)
