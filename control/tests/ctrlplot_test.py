# ctrlplot_test.py - test out control plotting utilities
# RMM, 27 Jun 2024

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import control as ct

# List of all plotting functions
resp_plot_fcns = [
    # response function                 plotting function
    (ct.frequency_response,             ct.bode_plot),
    (ct.frequency_response,             ct.nichols_plot),
    (ct.singular_values_response,       ct.singular_values_plot),
    (ct.gangof4_response,               ct.gangof4_plot),
    (ct.describing_function_response,   ct.describing_function_plot),
    (None,                              ct.phase_plane_plot),
    (ct.pole_zero_map,                  ct.pole_zero_plot),
    (ct.nyquist_response,               ct.nyquist_plot),
    (ct.root_locus_map,                 ct.root_locus_plot),
    (ct.initial_response,               ct.time_response_plot),
    (ct.step_response,                  ct.time_response_plot),
    (ct.impulse_response,               ct.time_response_plot),
    (ct.forced_response,                ct.time_response_plot),
    (ct.input_output_response,          ct.time_response_plot),
]

nolabel_plot_fcns = [ct.describing_function_plot, ct.phase_plane_plot]
deprecated_fcns = [ct.phase_plot]

# Make sure we didn't miss any plotting functions
def test_find_respplot_functions():
    # Get the list of plotting functions
    plot_fcns = {respplot[1] for respplot in resp_plot_fcns}

    # Look through every object in the package
    found = 0
    for name, obj in inspect.getmembers(ct):
        # Skip anything that is outside of this module
        if inspect.getmodule(obj) is not None and \
           not inspect.getmodule(obj).__name__.startswith('control'):
            # Skip anything that isn't part of the control package
            continue

        # Only look for non-deprecated functions ending in 'plot'
        if not inspect.isfunction(obj) or name[-4:] != 'plot' or \
           obj in deprecated_fcns:
            continue

        # Make sure that we have this on our list of functions
        assert obj in plot_fcns
        found += 1

    assert found == len(plot_fcns)


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_ax_processing(resp_fcn, plot_fcn):
    # Create some systems to use
    sys1 = ct.rss(2, 1, 1, strictly_proper=True)
    sys2 = ct.rss(4, 1, 1, strictly_proper=True)

    # Set up arguments
    kwargs = meth_kwargs = plot_fcn_kwargs = {}
    get_line_color = lambda cplt: cplt.lines.reshape(-1)[0][0].get_color()
    match resp_fcn, plot_fcn:
        case ct.describing_function_response, _:
            F = ct.descfcn.saturation_nonlinearity(1)
            amp = np.linspace(1, 4, 10)
            args = (sys1, F, amp)

        case ct.gangof4_response, _:
            args = (sys1, sys2)

        case ct.frequency_response, ct.nichols_plot:
            args = (sys1, )
            meth_kwargs = {'plot_type': 'nichols'}

        case ct.root_locus_map, ct.root_locus_plot:
            args = (sys1, )
            meth_kwargs = plot_fcn_kwargs = {'interactive': False}

        case (ct.forced_response | ct.input_output_response, _):
            timepts = np.linspace(1, 10)
            U = np.sin(timepts)
            args = (sys1, timepts, U)

        case None, ct.phase_plane_plot:
            args = (sys1, )
            get_line_color = None
            warnings.warn("ct.phase_plane_plot returns nonstandard lines")

        case _, _:
            args = (sys1, )

    # Call the plot through the response function
    if resp_fcn is not None:
        resp = resp_fcn(*args, **kwargs)
        cplt1 = resp.plot(**kwargs, **meth_kwargs)
    else:
        # No response function available; just plot the data
        cplt1 = plot_fcn(*args, **kwargs, **meth_kwargs)
    assert isinstance(cplt1, ct.ControlPlot)

    # Call the plot directly, plotting on top of previous plot
    if plot_fcn == ct.time_response_plot:
        # Can't call the time_response_plot() with system => reuse data
        cplt2 = plot_fcn(resp, **kwargs, **plot_fcn_kwargs)
    else:
        cplt2 = plot_fcn(*args, **kwargs, **plot_fcn_kwargs)
    assert isinstance(cplt2, ct.ControlPlot)

    # Plot should have landed on top of previous plot, in different colors
    assert cplt2.figure == cplt1.figure
    assert np.all(cplt2.axes == cplt1.axes)
    assert len(cplt2.lines[0]) == len(cplt1.lines[0])
    if get_line_color is not None:
        assert get_line_color(cplt2) != get_line_color(cplt1)

    # Pass axes explicitly
    if resp_fcn is not None:
        cplt3 = resp.plot(**kwargs, **meth_kwargs, ax=cplt1.axes)
    else:
        cplt3 = plot_fcn(*args, **kwargs, **meth_kwargs, ax=cplt1.axes)
    assert cplt3.figure == cplt1.figure

    # Plot should have landed on top of previous plot, in different colors
    assert np.all(cplt3.axes == cplt1.axes)
    assert len(cplt3.lines[0]) == len(cplt1.lines[0])
    if get_line_color is not None:
        assert get_line_color(cplt3) != get_line_color(cplt1)
        assert get_line_color(cplt3) != get_line_color(cplt2)

    #
    # Plot on a user-contructed figure
    #

    # Store modified properties from previous figure
    cplt_titlesize = cplt3.figure._suptitle.get_fontsize()
    cplt_labelsize = \
        cplt3.axes.reshape(-1)[0].get_yticklabels()[0].get_fontsize()

    # Set up some axes with a known title
    fig, axs = plt.subplots(2, 3)
    title = "User-constructed figure"
    plt.suptitle(title)
    titlesize = fig._suptitle.get_fontsize()
    assert titlesize != cplt_titlesize
    labelsize = axs[0, 0].get_yticklabels()[0].get_fontsize()
    assert labelsize != cplt_labelsize

    # Figure out what to pass as the ax keyword
    match resp_fcn, plot_fcn:
        case _, ct.bode_plot:
            ax = [axs[0, 1], axs[1, 1]]

        case ct.gangof4_response, _:
            ax = [axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2]]

        case (ct.forced_response | ct.input_output_response, _):
            ax = [axs[0, 1], axs[1, 1]]

        case _, _:
            ax = [axs[0, 1]]

    # Call the plotting function, passing the axes
    if resp_fcn is not None:
        resp = resp_fcn(*args, **kwargs)
        cplt4 = resp.plot(**kwargs, **meth_kwargs, ax=ax)
    else:
        # No response function available; just plot the data
        cplt4 = plot_fcn(*args, **kwargs, **meth_kwargs, ax=ax)

    # Check to make sure original settings did not change
    assert fig._suptitle.get_text() == title
    assert fig._suptitle.get_fontsize() == titlesize
    assert ax[0].get_yticklabels()[0].get_fontsize() == labelsize


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_label_processing(resp_fcn, plot_fcn):
    # Utility function to make sure legends are OK
    def assert_legend(cplt, expected_texts):
        # Check to make sure the labels are OK in legend
        legend = None
        for ax in cplt.axes.flatten():
            legend = ax.get_legend()
            if legend is not None:
                break
        if expected_texts is None:
            assert legend is None
        else:
            assert legend is not None
            legend_texts = [entry.get_text() for entry in legend.get_texts()]
            assert legend_texts == expected_texts

    # Create some systems to use
    sys1 = ct.rss(2, 1, 1, strictly_proper=True, name="sys[1]")
    sys1c = ct.rss(4, 1, 1, strictly_proper=True, name="sys[1]_C")
    sys2 = ct.rss(4, 1, 1, strictly_proper=True, name="sys[2]")

    # Set up arguments
    kwargs = meth_kwargs = plot_fcn_kwargs = {}
    default_labels = ["sys[1]", "sys[2]"]
    expected_labels = ["sys1_", "sys2_"]
    match resp_fcn, plot_fcn:
        case ct.describing_function_response, _:
            F = ct.descfcn.saturation_nonlinearity(1)
            amp = np.linspace(1, 4, 10)
            args1 = (sys1, F, amp)
            args2 = (sys2, F, amp)

        case ct.gangof4_response, _:
            args1 = (sys1, sys1c)
            args2 = (sys2, sys1c)
            default_labels = ["P=sys[1]", "P=sys[2]"]

        case ct.frequency_response, ct.nichols_plot:
            args1 = (sys1, )
            args2 = (sys2, )
            meth_kwargs = {'plot_type': 'nichols'}

        case ct.root_locus_map, ct.root_locus_plot:
            args1 = (sys1, )
            args2 = (sys2, )
            meth_kwargs = plot_fcn_kwargs = {'interactive': False}

        case (ct.forced_response | ct.input_output_response, _):
            timepts = np.linspace(1, 10)
            U = np.sin(timepts)
            args1 = (resp_fcn(sys1, timepts, U), )
            args2 = (resp_fcn(sys2, timepts, U), )
            argsc = (resp_fcn([sys1, sys2], timepts, U), )

        case (ct.impulse_response | ct.initial_response | ct.step_response, _):
            args1 = (resp_fcn(sys1), )
            args2 = (resp_fcn(sys2), )
            argsc = (resp_fcn([sys1, sys2]), )

        case _, _:
            args1 = (sys1, )
            args2 = (sys2, )

    if plot_fcn in nolabel_plot_fcns:
        pytest.skip(f"labels not implemented for {plot_fcn}")

    # Generate the first plot, with default labels
    cplt1 = plot_fcn(*args1, **kwargs, **plot_fcn_kwargs)
    assert isinstance(cplt1, ct.ControlPlot)
    assert_legend(cplt1, None)

    # Generate second plot with default labels
    cplt2 = plot_fcn(*args2, **kwargs, **plot_fcn_kwargs)
    assert isinstance(cplt2, ct.ControlPlot)
    assert_legend(cplt2, default_labels)
    plt.close()

    # Generate both plots at the same time
    if len(args1) == 1 and plot_fcn != ct.time_response_plot:
        cplt = plot_fcn([*args1, *args2], **kwargs, **plot_fcn_kwargs)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, default_labels)
    elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
        # Use TimeResponseList.plot() to generate combined response
        cplt = argsc[0].plot(**kwargs, **plot_fcn_kwargs)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, default_labels)
    plt.close()

    # Generate plots sequentially, with updated labels
    cplt1 = plot_fcn(
        *args1, **kwargs, **plot_fcn_kwargs, label=expected_labels[0])
    assert isinstance(cplt1, ct.ControlPlot)
    assert_legend(cplt1, None)

    cplt2 = plot_fcn(
        *args2, **kwargs, **plot_fcn_kwargs, label=expected_labels[1])
    assert isinstance(cplt2, ct.ControlPlot)
    assert_legend(cplt2, expected_labels)
    plt.close()

    # Generate both plots at the same time, with updated labels
    if len(args1) == 1 and plot_fcn != ct.time_response_plot:
        cplt = plot_fcn(
            [*args1, *args2], **kwargs, **plot_fcn_kwargs,
            label=expected_labels)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, expected_labels)
    elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
        # Use TimeResponseList.plot() to generate combined response
        cplt = argsc[0].plot(
            **kwargs, **plot_fcn_kwargs, label=expected_labels)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, expected_labels)


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_title_processing(resp_fcn, plot_fcn):
    # Create some systems to use
    sys1 = ct.rss(2, 1, 1, strictly_proper=True, name="sys[1]")
    sys1c = ct.rss(4, 1, 1, strictly_proper=True, name="sys[1]_C")
    sys2 = ct.rss(2, 1, 1, strictly_proper=True, name="sys[2]")

    # Set up arguments
    kwargs = meth_kwargs = plot_fcn_kwargs = {}
    default_title = "sys[1], sys[2]"
    expected_title = "sys1_, sys2_"
    match resp_fcn, plot_fcn:
        case ct.describing_function_response, _:
            F = ct.descfcn.saturation_nonlinearity(1)
            amp = np.linspace(1, 4, 10)
            args1 = (sys1, F, amp)
            args2 = (sys2, F, amp)

        case ct.gangof4_response, _:
            args1 = (sys1, sys1c)
            args2 = (sys2, sys1c)
            default_title = "P=sys[1], C=sys[1]_C, P=sys[2], C=sys[1]_C"

        case ct.frequency_response, ct.nichols_plot:
            args1 = (sys1, )
            args2 = (sys2, )
            meth_kwargs = {'plot_type': 'nichols'}

        case ct.root_locus_map, ct.root_locus_plot:
            args1 = (sys1, )
            args2 = (sys2, )
            meth_kwargs = plot_fcn_kwargs = {'interactive': False}

        case (ct.forced_response | ct.input_output_response, _):
            timepts = np.linspace(1, 10)
            U = np.sin(timepts)
            args1 = (resp_fcn(sys1, timepts, U), )
            args2 = (resp_fcn(sys2, timepts, U), )
            argsc = (resp_fcn([sys1, sys2], timepts, U), )

        case (ct.impulse_response | ct.initial_response | ct.step_response, _):
            args1 = (resp_fcn(sys1), )
            args2 = (resp_fcn(sys2), )
            argsc = (resp_fcn([sys1, sys2]), )

        case _, _:
            args1 = (sys1, )
            args2 = (sys2, )

    # Store the expected title prefix
    match resp_fcn, plot_fcn:
        case _, ct.bode_plot:
            title_prefix = "Bode plot for "
        case _, ct.nichols_plot:
            title_prefix = "Nichols plot for "
        case _, ct.singular_values_plot:
            title_prefix = "Singular values for "
        case _, ct.gangof4_plot:
            title_prefix = "Gang of Four for "
        case _, ct.describing_function_plot:
            title_prefix = "Nyquist plot for "
        case _, ct.phase_plane_plot:
            title_prefix = "Phase portrait for "
        case _, ct.pole_zero_plot:
            title_prefix = "Pole/zero plot for "
        case _, ct.nyquist_plot:
            title_prefix = "Nyquist plot for "
        case _, ct.root_locus_plot:
            title_prefix = "Root locus plot for "
        case ct.initial_response, _:
            title_prefix = "Initial response for "
        case ct.step_response, _:
            title_prefix = "Step response for "
        case ct.impulse_response, _:
            title_prefix = "Impulse response for "
        case ct.forced_response, _:
            title_prefix = "Forced response for "
        case ct.input_output_response, _:
            title_prefix = "Input/output response for "
        case _:
            raise RuntimeError(f"didn't recognize {resp_fnc}, {plot_fnc}")

    # Generate the first plot, with default title
    cplt1 = plot_fcn(*args1, **kwargs, **plot_fcn_kwargs)
    assert cplt1.figure._suptitle._text.startswith(title_prefix)

    # Skip functions not intended for sequential calling
    if plot_fcn not in nolabel_plot_fcns:
        # Generate second plot with default title
        cplt2 = plot_fcn(*args2, **kwargs, **plot_fcn_kwargs)
        assert cplt1.figure._suptitle._text == title_prefix + default_title
        plt.close()

        # Generate both plots at the same time
        if len(args1) == 1 and plot_fcn != ct.time_response_plot:
            cplt = plot_fcn([*args1, *args2], **kwargs, **plot_fcn_kwargs)
            assert cplt.figure._suptitle._text == title_prefix + default_title
        elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
            # Use TimeResponseList.plot() to generate combined response
            cplt = argsc[0].plot(**kwargs, **plot_fcn_kwargs)
            assert cplt.figure._suptitle._text == title_prefix + default_title
        plt.close()

    # Generate plots sequentially, with updated titles
    cplt1 = plot_fcn(
        *args1, **kwargs, **plot_fcn_kwargs, title="My first title")
    cplt2 = plot_fcn(
        *args2, **kwargs, **plot_fcn_kwargs, title="My new title")
    assert cplt2.figure._suptitle._text == "My new title"
    plt.close()

    # Update using set_plot_title
    cplt2.set_plot_title("Another title")
    assert cplt2.figure._suptitle._text == "Another title"
    plt.close()

    # Generate the plots with no title
    cplt = plot_fcn(
        *args1, **kwargs, **plot_fcn_kwargs, title=False)
    assert cplt.figure._suptitle == None


@pytest.mark.usefixtures('mplcleanup')
def test_rcParams():
    sys = ct.rss(2, 2, 2)

    # Create new set of rcParams
    my_rcParams = {}
    for key in [
            'axes.labelsize', 'axes.titlesize', 'figure.titlesize',
            'legend.fontsize', 'xtick.labelsize', 'ytick.labelsize']:
        match plt.rcParams[key]:
            case 8 | 9 | 10:
                my_rcParams[key] = plt.rcParams[key] + 1
            case 'medium':
                my_rcParams[key] = 11.5
            case 'large':
                my_rcParams[key] = 9.5
            case _:
                raise ValueError(f"unknown rcParam type for {key}")

    # Generate a figure with the new rcParams
    out = ct.step_response(sys).plot(rcParams=my_rcParams)
    ax, fig = out.axes[0, 0], out.figure

    # Check to make sure new settings were used
    assert ax.xaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    assert ax.yaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    assert ax.title.get_fontsize() == my_rcParams['axes.titlesize']
    assert ax.get_xticklabels()[0].get_fontsize() == \
        my_rcParams['xtick.labelsize']
    assert ax.get_yticklabels()[0].get_fontsize() == \
        my_rcParams['ytick.labelsize']
    assert fig._suptitle.get_fontsize() == my_rcParams['figure.titlesize']


def test_deprecation_warning():
    sys = ct.rss(2, 2, 2)
    lines = ct.step_response(sys).plot(overlay_traces=True)
    with pytest.warns(FutureWarning, match="deprecated"):
        assert len(lines[0, 0]) == 2
