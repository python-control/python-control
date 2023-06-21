# timeplot_test.py - test out time response plots
# RMM, 23 Jun 2023

import pytest
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt

# Step responses
@pytest.mark.parametrize("nin, nout", [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("plot_inputs", [None, True, False])
def test_simple_response(nout, nin, transpose, plot_inputs):
    sys = ct.rss(4, nout, nin)
    stepresp = ct.step_response(sys)
    stepresp.plot(plot_inputs=plot_inputs, transpose=transpose)

    # Add additional data (and provide infon in the title)
    ct.step_response(ct.rss(4, nout, nin), stepresp.time[-1]).plot(
        plot_inputs=plot_inputs, transpose=transpose,
        title=stepresp.title + f" [{plot_inputs=}, {transpose=}]")


@pytest.mark.parametrize("transpose", [True, False])
def test_combine_signals(transpose):
    sys = ct.rss(4, 2, 3)
    stepresp = ct.step_response(sys)
    stepresp.plot(
        combine_signals=True, transpose=transpose,
        title=f"Step response: combine_signals = True; transpose={transpose}")


@pytest.mark.parametrize("transpose", [True, False])
def test_combine_traces(transpose):
    sys = ct.rss(4, 2, 3)
    stepresp = ct.step_response(sys)
    stepresp.plot(
        combine_traces=True, transpose=transpose,
        title=f"Step response: combine_traces = True; transpose={transpose}")


@pytest.mark.parametrize("transpose", [True, False])
def test_combine_signals_traces(transpose):
    sys = ct.rss(4, 5, 3)
    stepresp = ct.step_response(sys)
    stepresp.plot(
        combine_signals=True, combine_traces=True, transpose=transpose,
        title=f"Step response: combine_signals/traces = True;" +
        f"transpose={transpose}")


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

    print ("Simple step responses")
    for size in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]:
        for transpose in [False, True]:
            for plot_inputs in [None, True, False]:
                plt.figure()
                test_simple_response(
                    *size, transpose=transpose, plot_inputs=plot_inputs)

    print ("Combine signals")
    for transpose in [False, True]:
        plt.figure()
        test_combine_signals(transpose)

    print ("Combine traces")
    for transpose in [False, True]:
        plt.figure()
        test_combine_traces(transpose)

    print ("Combine signals and traces")
    for transpose in [False, True]:
        plt.figure()
        test_combine_signals_traces(transpose)

