# timeplot_test.py - test out time response plots
# RMM, 23 Jun 2023

import pytest
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .conftest import slycotonly

# Detailed test of (almost) all functionality
# (uncomment rows for developmental testing, but otherwise takes too long)
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
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("plot_inputs", [None, True, False, 'overlay'])
@pytest.mark.parametrize("plot_outputs", [True, False])
@pytest.mark.parametrize("combine_signals", [True, False])
@pytest.mark.parametrize("combine_traces", [True, False])
@pytest.mark.parametrize("second_system", [False, True])
@pytest.mark.parametrize("fcn", [
    ct.step_response, ct.impulse_response, ct.initial_response,
    ct.forced_response, ct.input_output_response])
def test_response_plots(
        fcn, sys, plot_inputs, plot_outputs, combine_signals, combine_traces,
        transpose, second_system, clear=True):
    # Figure out the time range to use and check some special cases
    if not isinstance(sys, ct.lti.LTI):
        if fcn == ct.impulse_response:
            pytest.skip("impulse response not implemented for nlsys")

        # Nonlinear systems require explicit time limits
        T = 10
        timepts = np.linspace(0, T)
    else:
        # Linear systems figure things out on their own
        T = None
        timepts = np.linspace(0, 10)    # for input_output_response

    # Save up the keyword arguments
    kwargs = dict(
        plot_inputs=plot_inputs, plot_outputs=plot_outputs, transpose=transpose,
        combine_signals=combine_signals, combine_traces=combine_traces)

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
    if not plot_outputs and (
            plot_inputs is False or response.ninputs == 0 or
            plot_inputs is None and response.plot_inputs is False):
        with pytest.raises(ValueError, match=".* no data to plot"):
            out = response.plot(**kwargs)
        return None
    elif not plot_outputs and plot_inputs == 'overlay':
        with pytest.raises(ValueError, match="can't overlay inputs"):
            out = response.plot(**kwargs)
        return None
    elif plot_inputs in [True, 'overlay'] and response.ninputs == 0:
        with pytest.raises(ValueError, match=".* but no inputs"):
            out = response.plot(**kwargs)
        return None

    out = response.plot(**kwargs)

    # TODO: add some basic checks here

    # Add additional data (and provide infon in the title)
    if second_system:
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

    # TODO: add some basic checks here

    # Update the title so we can see what is going on
    fig = out[0, 0][0].axes.figure
    fig.suptitle(
        fig._suptitle._text +
        f" [{sys.noutputs}x{sys.ninputs}, cs={combine_signals}, "
        f"ct={combine_traces}, pi={plot_inputs}, tr={transpose}]",
        fontsize='small')

    # Get rid of the figure to free up memory
    if clear:
        plt.clf()


@slycotonly
def test_legend_map():
    sys_mimo = ct.tf2ss(
        [[[1], [0.1]], [[0.2], [1]]],
        [[[1, 0.6, 1], [1, 1, 1]], [[1, 0.4, 1], [1, 2, 1]]], name="MIMO")
    response = ct.step_response(sys_mimo)
    response.plot(
        legend_map=np.array([['center', 'upper right'],
                             [None, 'center right']]),
        plot_inputs=True, combine_signals=True, transpose=True,
        title='MIMO step response with custom legend placement')


def test_errors():
    sys = ct.rss(2, 1, 1)
    stepresp = ct.step_response(sys)
    with pytest.raises(TypeError, match="unrecognized keyword"):
        stepresp.plot(unknown=None)

    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.ioresp_plot(stepresp, unknown=None)

    with pytest.raises(ValueError, match="unrecognized value"):
        stepresp.plot(plot_inputs='unknown')


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
    #      fcn, sys, plot_inputs, plot_outputs, combine_signals,
    #      combine_traces, transpose, second_system, clear=True):
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

    # Step response with plot_inputs, combine_signals
    plt.figure()
    ct.step_response(sys_mimo).plot(
        plot_inputs=True, combine_signals=True,
        title="Step response for 2x2 MIMO system " +
        "[plot_inputs, combine_signals]")
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

    ct.combine_traces(
        [resp1, resp2], trace_labels=["Scenario #1", "Scenario #2"]).plot(
            transpose=True,
            title="I/O responses for 2x2 MIMO system, multiple traces "
            "[transpose]")
    plt.savefig('timeplot-mimo_ioresp-mt_tr.png')
