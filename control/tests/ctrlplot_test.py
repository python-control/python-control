# ctrlplot_test.py - test out control plotting utilities
# RMM, 27 Jun 2024

import inspect
import itertools
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
legacy_plot_fcns = [ct.gangof4_plot]
multiaxes_plot_fcns = [ct.bode_plot, ct.gangof4_plot, ct.time_response_plot]
deprecated_fcns = [ct.phase_plot]


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


def setup_plot_arguments(resp_fcn, plot_fcn, compute_time_response=True):
    # Create some systems to use
    sys1 = ct.rss(2, 1, 1, strictly_proper=True, name="sys[1]")
    sys1c = ct.rss(2, 1, 1, strictly_proper=True, name="sys[1]_C")
    sys2 = ct.rss(2, 1, 1, strictly_proper=True, name="sys[2]")

    # Set up arguments
    kwargs = resp_kwargs = plot_kwargs = meth_kwargs = {}
    argsc = None
    match resp_fcn, plot_fcn:
        case ct.describing_function_response, _:
            sys1 = ct.tf([1], [1, 2, 2, 1], name="sys[1]")
            sys2 = ct.tf([1.1], [1, 2, 2, 1], name="sys[2]")
            F = ct.descfcn.saturation_nonlinearity(1)
            amp = np.linspace(1, 4, 10)
            args1 = (sys1, F, amp)
            args2 = (sys2, F, amp)
            resp_kwargs = plot_kwargs = {'refine': False}

        case ct.gangof4_response, _:
            args1 = (sys1, sys1c)
            args2 = (sys2, sys1c)

        case ct.frequency_response, ct.nichols_plot:
            args1 = (sys1, None)        # to allow *fmt in linestyle test
            args2 = (sys2, )
            meth_kwargs = {'plot_type': 'nichols'}

        case ct.frequency_response, ct.bode_plot:
            args1 = (sys1, None)        # to allow *fmt in linestyle test
            args2 = (sys2, )

        case ct.singular_values_response, ct.singular_values_plot:
            args1 = (sys1, None)        # to allow *fmt in linestyle test
            args2 = (sys2, )

        case ct.root_locus_map, ct.root_locus_plot:
            args1 = (sys1, )
            args2 = (sys2, )
            plot_kwargs = {'interactive': False}

        case (ct.forced_response | ct.input_output_response, _):
            timepts = np.linspace(1, 10)
            U = np.sin(timepts)
            if compute_time_response:
                args1 = (resp_fcn(sys1, timepts, U), )
                args2 = (resp_fcn(sys2, timepts, U), )
                argsc = (resp_fcn([sys1, sys2], timepts, U), )
            else:
                args1 = (sys1, timepts, U)
                args2 = (sys2, timepts, U)
                argsc = None

        case (ct.impulse_response | ct.initial_response | ct.step_response, _):
            if compute_time_response:
                args1 = (resp_fcn(sys1), )
                args2 = (resp_fcn(sys2), )
                argsc = (resp_fcn([sys1, sys2]), )
            else:
                args1 = (sys1, )
                args2 = (sys2, )
                argsc = ([sys1, sys2], )

        case (None, ct.phase_plane_plot):
            args1 = (sys1, )
            args2 = (sys2, )
            plot_kwargs = {'plot_streamlines': True}

        case _, _:
            args1 = (sys1, )
            args2 = (sys2, )

    return args1, args2, argsc, kwargs, meth_kwargs, plot_kwargs, resp_kwargs


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
    # Set up arguments
    args, _, _, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn, compute_time_response=False)
    get_line_color = lambda cplt: cplt.lines.reshape(-1)[0][0].get_color()
    match resp_fcn, plot_fcn:
        case None, ct.phase_plane_plot:
            get_line_color = None
            warnings.warn("ct.phase_plane_plot returns nonstandard lines")

    # Call the plot through the response function
    if resp_fcn is not None:
        resp = resp_fcn(*args, **kwargs, **resp_kwargs)
        cplt1 = resp.plot(**kwargs, **meth_kwargs)
    else:
        # No response function available; just plot the data
        cplt1 = plot_fcn(*args, **kwargs, **plot_kwargs)
    assert isinstance(cplt1, ct.ControlPlot)

    # Call the plot directly, plotting on top of previous plot
    if plot_fcn == ct.time_response_plot:
        # Can't call the time_response_plot() with system => reuse data
        cplt2 = plot_fcn(resp, **kwargs, **plot_kwargs)
    else:
        cplt2 = plot_fcn(*args, **kwargs, **plot_kwargs)
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
        cplt3 = plot_fcn(*args, **kwargs, **plot_kwargs, ax=cplt1.axes)
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
        resp = resp_fcn(*args, **kwargs, **resp_kwargs)
        resp.plot(**kwargs, **meth_kwargs, ax=ax)
    else:
        # No response function available; just plot the data
        plot_fcn(*args, **kwargs, **plot_kwargs, ax=ax)

    # Make sure the plot ended up in the right place
    assert len(axs[0, 0].get_lines()) == 0      # upper left
    assert len(axs[0, 1].get_lines()) != 0      # top middle
    assert len(axs[1, 0].get_lines()) == 0      # lower left
    if resp_fcn != ct.gangof4_response:
        assert len(axs[1, 2].get_lines()) == 0  # lower right (normally empty)
    else:
        assert len(axs[1, 2].get_lines()) != 0  # gangof4 uses this axes

    # Check to make sure original settings did not change
    assert fig._suptitle.get_text() == title
    assert fig._suptitle.get_fontsize() == titlesize
    assert ax[0].get_yticklabels()[0].get_fontsize() == labelsize

    # Make sure that docstring documents ax keyword
    if plot_fcn not in legacy_plot_fcns:
        if plot_fcn in multiaxes_plot_fcns:
            assert "ax : array of `matplotlib.axes.Axes`, optional" \
                in plot_fcn.__doc__
        else:
            assert "ax : `matplotlib.axes.Axes`, optional" in plot_fcn.__doc__


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_label_processing(resp_fcn, plot_fcn):
    # Set up arguments
    args1, args2, argsc, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn)
    default_labels = ["sys[1]", "sys[2]"]
    expected_labels = ["sys1_", "sys2_"]
    match resp_fcn, plot_fcn:
        case ct.gangof4_response, _:
            default_labels = ["P=sys[1]", "P=sys[2]"]

    if plot_fcn in nolabel_plot_fcns:
        pytest.skip(f"labels not implemented for {plot_fcn}")

    # Generate the first plot, with default labels
    cplt1 = plot_fcn(*args1, **kwargs, **plot_kwargs)
    assert isinstance(cplt1, ct.ControlPlot)
    assert_legend(cplt1, None)

    # Generate second plot with default labels
    cplt2 = plot_fcn(*args2, **kwargs, **plot_kwargs)
    assert isinstance(cplt2, ct.ControlPlot)
    assert_legend(cplt2, default_labels)
    plt.close()

    # Generate both plots at the same time
    if len(args1) == 1 and plot_fcn != ct.time_response_plot:
        cplt = plot_fcn([*args1, *args2], **kwargs, **plot_kwargs)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, default_labels)
    elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
        # Use TimeResponseList.plot() to generate combined response
        cplt = argsc[0].plot(**kwargs, **meth_kwargs)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, default_labels)
    plt.close()

    # Generate plots sequentially, with updated labels
    cplt1 = plot_fcn(
        *args1, **kwargs, **plot_kwargs, label=expected_labels[0])
    assert isinstance(cplt1, ct.ControlPlot)
    assert_legend(cplt1, None)

    cplt2 = plot_fcn(
        *args2, **kwargs, **plot_kwargs, label=expected_labels[1])
    assert isinstance(cplt2, ct.ControlPlot)
    assert_legend(cplt2, expected_labels)
    plt.close()

    # Generate both plots at the same time, with updated labels
    if len(args1) == 1 and plot_fcn != ct.time_response_plot:
        cplt = plot_fcn(
            [*args1, *args2], **kwargs, **plot_kwargs,
            label=expected_labels)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, expected_labels)
    elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
        # Use TimeResponseList.plot() to generate combined response
        cplt = argsc[0].plot(
            **kwargs, **meth_kwargs, label=expected_labels)
        assert isinstance(cplt, ct.ControlPlot)
        assert_legend(cplt, expected_labels)
    plt.close()

    # Make sure that docstring documents label
    if plot_fcn not in legacy_plot_fcns:
        assert "label : str or array_like of str, optional" in plot_fcn.__doc__


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_linestyle_processing(resp_fcn, plot_fcn):
    # Set up arguments
    args1, args2, _, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn)

    # Set line color
    cplt1 = plot_fcn(*args1, **kwargs, **plot_kwargs, color='r')
    assert cplt1.lines.reshape(-1)[0][0].get_color() == 'r'

    # Second plot, new line color
    cplt2 = plot_fcn(*args2, **kwargs, **plot_kwargs, color='g')
    assert cplt2.lines.reshape(-1)[0][0].get_color() == 'g'

    # Make sure that docstring documents line properties
    if plot_fcn not in legacy_plot_fcns:
        assert "line properties" in plot_fcn.__doc__ or \
            "color : matplotlib color spec, optional" in plot_fcn.__doc__

    # Set other characteristics if documentation says we can
    if "line properties" in plot_fcn.__doc__:
        cplt = plot_fcn(*args1, **kwargs, **plot_kwargs, linewidth=5)
        assert cplt.lines.reshape(-1)[0][0].get_linewidth() == 5

    # If fmt string is allowed, use it to set line color and style
    if "*fmt" in plot_fcn.__doc__:
        cplt = plot_fcn(*args1, 'r--', **kwargs, **plot_kwargs)
        assert cplt.lines.reshape(-1)[0][0].get_color() == 'r'
        assert cplt.lines.reshape(-1)[0][0].get_linestyle() == '--'


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_siso_plot_legend_processing(resp_fcn, plot_fcn):
    # Set up arguments
    args1, args2, argsc, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn)
    default_labels = ["sys[1]", "sys[2]"]
    match resp_fcn, plot_fcn:
        case ct.gangof4_response, _:
            # Multi-axes plot => test in next function
            return

    if plot_fcn in nolabel_plot_fcns:
        # Make sure that using legend keywords generates an error
        with pytest.raises(TypeError, match="unexpected|unrecognized"):
            cplt = plot_fcn(*args1, legend_loc=None)
        with pytest.raises(TypeError, match="unexpected|unrecognized"):
            cplt = plot_fcn(*args1, legend_map=None)
        with pytest.raises(TypeError, match="unexpected|unrecognized"):
            cplt = plot_fcn(*args1, show_legend=None)
        return

    # Single system, with forced legend
    cplt = plot_fcn(*args1, **kwargs, **plot_kwargs, show_legend=True)
    assert_legend(cplt, default_labels[:1])
    plt.close()

    # Single system, with forced location
    cplt = plot_fcn(*args1, **kwargs, **plot_kwargs, legend_loc=10)
    assert cplt.axes[0, 0].get_legend()._loc == 10
    plt.close()

    # Generate two plots, but turn off legends
    if len(args1) == 1 and plot_fcn != ct.time_response_plot:
        cplt = plot_fcn(
            [*args1, *args2], **kwargs, **plot_kwargs, show_legend=False)
        assert_legend(cplt, None)
    elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
        # Use TimeResponseList.plot() to generate combined response
        cplt = argsc[0].plot(**kwargs, **meth_kwargs, show_legend=False)
        assert_legend(cplt, None)
    plt.close()

    # Make sure that docstring documents legend_loc, show_legend
    assert "legend_loc : int or str, optional" in plot_fcn.__doc__
    assert "show_legend : bool, optional" in plot_fcn.__doc__

    # Make sure that single axes plots generate an error with legend_map
    if plot_fcn not in multiaxes_plot_fcns:
        with pytest.raises(TypeError, match="unexpected"):
            cplt = plot_fcn(*args1, legend_map=False)
    else:
        assert "legend_map : array of str" in plot_fcn.__doc__


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_mimo_plot_legend_processing(resp_fcn, plot_fcn):
    # Generate the response that we will use for plotting
    match resp_fcn, plot_fcn:
        case ct.frequency_response, ct.bode_plot:
            resp = ct.frequency_response([ct.rss(4, 2, 2), ct.rss(3, 2, 2)])
        case ct.step_response, ct.time_response_plot:
            resp = ct.step_response([ct.rss(4, 2, 2), ct.rss(3, 2, 2)])
        case ct.gangof4_response, ct.gangof4_plot:
            resp = ct.gangof4_response(ct.rss(4, 1, 1), ct.rss(3, 1, 1))
        case _, ct.time_response_plot:
            # Skip remaining time response plots to avoid duplicate tests
            return
        case _, _:
            # Skip everything else that doesn't support multi-axes plots
            assert plot_fcn not in multiaxes_plot_fcns
            return

    # Generate a standard plot with legend in the center
    cplt1 = resp.plot(legend_loc=10)
    assert cplt1.axes.ndim == 2
    for legend_idx, ax in enumerate(cplt1.axes.flatten()):
        if ax.get_legend() is not None:
            break;
    assert legend_idx != 0      # Make sure legend is not in first subplot
    assert ax.get_legend()._loc == 10
    plt.close()

    # Regenerate the plot with no legend
    cplt2 = resp.plot(show_legend=False)
    for ax in cplt2.axes.flatten():
        if ax.get_legend() is not None:
            break;
    assert ax.get_legend() is None
    plt.close()

    # Regenerate the plot with no legend in a different way
    cplt2 = resp.plot(legend_loc=False)
    for ax in cplt2.axes.flatten():
        if ax.get_legend() is not None:
            break;
    assert ax.get_legend() is None
    plt.close()

    # Regenerate the plot with no legend in a different way
    cplt2 = resp.plot(legend_map=False)
    for ax in cplt2.axes.flatten():
        if ax.get_legend() is not None:
            break;
    assert ax.get_legend() is None
    plt.close()

    # Put the legend in a different (first) subplot
    legend_map = np.full(cplt2.shape, None, dtype=object)
    legend_map[0, 0] = 5
    legend_map[-1, -1] = 6
    cplt3 = resp.plot(legend_map=legend_map)
    assert cplt3.axes[0, 0].get_legend()._loc == 5
    assert cplt3.axes[-1, -1].get_legend()._loc == 6
    plt.close()


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_plot_title_processing(resp_fcn, plot_fcn):
    # Set up arguments
    args1, args2, argsc, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn)
    default_title = "sys[1], sys[2]"
    match resp_fcn, plot_fcn:
        case ct.gangof4_response, _:
            default_title = "P=sys[1], C=sys[1]_C, P=sys[2], C=sys[1]_C"

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
            raise RuntimeError(f"didn't recognize {resp_fcn}, {plot_fcn}")

    # Generate the first plot, with default title
    cplt1 = plot_fcn(*args1, **kwargs, **plot_kwargs)
    assert cplt1.figure._suptitle._text.startswith(title_prefix)

    # Skip functions not intended for sequential calling
    if plot_fcn not in nolabel_plot_fcns:
        # Generate second plot with default title
        cplt2 = plot_fcn(*args2, **kwargs, **plot_kwargs)
        assert cplt1.figure._suptitle._text == title_prefix + default_title
        plt.close()

        # Generate both plots at the same time
        if len(args1) == 1 and plot_fcn != ct.time_response_plot:
            cplt = plot_fcn([*args1, *args2], **kwargs, **plot_kwargs)
            assert cplt.figure._suptitle._text == title_prefix + default_title
        elif len(args1) == 1 and plot_fcn == ct.time_response_plot:
            # Use TimeResponseList.plot() to generate combined response
            cplt = argsc[0].plot(**kwargs, **meth_kwargs)
            assert cplt.figure._suptitle._text == title_prefix + default_title
        plt.close()

    # Generate plots sequentially, with updated titles
    cplt1 = plot_fcn(
        *args1, **kwargs, **plot_kwargs, title="My first title")
    cplt2 = plot_fcn(
        *args2, **kwargs, **plot_kwargs, title="My new title")
    assert cplt2.figure._suptitle._text == "My new title"
    plt.close()

    # Update using set_plot_title
    cplt2.set_plot_title("Another title")
    assert cplt2.figure._suptitle._text == "Another title"
    plt.close()

    # Generate the plots with no title
    cplt = plot_fcn(
        *args1, **kwargs, **plot_kwargs, title=False)
    assert cplt.figure._suptitle == None
    plt.close()

    # Make sure that docstring documents title
    if plot_fcn not in legacy_plot_fcns:
        assert "title : str, optional" in plot_fcn.__doc__


@pytest.mark.parametrize("plot_fcn", multiaxes_plot_fcns)
@pytest.mark.usefixtures('mplcleanup')
def test_tickmark_label_processing(plot_fcn):
    # Generate the response that we will use for plotting
    match plot_fcn:
        case ct.bode_plot:
            resp = ct.frequency_response(ct.rss(4, 2, 2))
        case ct.time_response_plot:
            resp = ct.step_response(ct.rss(4, 2, 2))
        case ct.gangof4_plot:
            resp = ct.gangof4_response(ct.rss(4, 1, 1), ct.rss(3, 1, 1))
        case _:
            pytest.fail("unknown plot_fcn")

    # Turn off axis sharing => all axes have ticklabels
    cplt = resp.plot(sharex=False, sharey=False)
    for i, j in itertools.product(
            range(cplt.axes.shape[0]), range(cplt.axes.shape[1])):
        assert len(cplt.axes[i, j].get_xticklabels()) > 0
        assert len(cplt.axes[i, j].get_yticklabels()) > 0
    plt.clf()

    # Turn on axis sharing => only outer axes have ticklabels
    cplt = resp.plot(sharex=True, sharey=True)
    for i, j in itertools.product(
            range(cplt.axes.shape[0]), range(cplt.axes.shape[1])):
        if i < cplt.axes.shape[0] - 1:
            assert len(cplt.axes[i, j].get_xticklabels()) == 0
        else:
            assert len(cplt.axes[i, j].get_xticklabels()) > 0

        if j > 0:
            assert len(cplt.axes[i, j].get_yticklabels()) == 0
        else:
            assert len(cplt.axes[i, j].get_yticklabels()) > 0


@pytest.mark.parametrize("resp_fcn, plot_fcn", resp_plot_fcns)
@pytest.mark.usefixtures('mplcleanup', 'editsdefaults')
def test_rcParams(resp_fcn, plot_fcn):
    # Set up arguments
    args1, args2, argsc, kwargs, meth_kwargs, plot_kwargs, resp_kwargs = \
        setup_plot_arguments(resp_fcn, plot_fcn)
    # Create new set of rcParams
    my_rcParams = {}
    for key in ct.ctrlplot.rcParams:
        match plt.rcParams[key]:
            case 8 | 9 | 10:
                my_rcParams[key] = plt.rcParams[key] + 1
            case 'medium':
                my_rcParams[key] = 11.5
            case 'large':
                my_rcParams[key] = 9.5
            case _:
                raise ValueError(f"unknown rcParam type for {key}")
    checked_params = my_rcParams.copy()         # make sure we check everything

    # Generate a figure with the new rcParams
    if plot_fcn not in nolabel_plot_fcns:
        cplt = plot_fcn(
            *args1, **kwargs, **plot_kwargs, rcParams=my_rcParams,
            show_legend=True)
    else:
        cplt = plot_fcn(*args1, **kwargs, **plot_kwargs, rcParams=my_rcParams)

    # Check lower left figure (should always have ticks, labels)
    ax, fig = cplt.axes[-1, 0], cplt.figure

    # Check to make sure new settings were used
    assert ax.xaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    assert ax.yaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    checked_params.pop('axes.labelsize')

    assert ax.title.get_fontsize() == my_rcParams['axes.titlesize']
    checked_params.pop('axes.titlesize')

    assert ax.get_xticklabels()[0].get_fontsize() == \
        my_rcParams['xtick.labelsize']
    checked_params.pop('xtick.labelsize')

    assert ax.get_yticklabels()[0].get_fontsize() == \
        my_rcParams['ytick.labelsize']
    checked_params.pop('ytick.labelsize')

    assert fig._suptitle.get_fontsize() == my_rcParams['figure.titlesize']
    checked_params.pop('figure.titlesize')

    if plot_fcn not in nolabel_plot_fcns:
        for ax in cplt.axes.flatten():
            legend = ax.get_legend()
            if legend is not None:
                break
        assert legend is not None
        assert legend.get_texts()[0].get_fontsize() == \
            my_rcParams['legend.fontsize']
    checked_params.pop('legend.fontsize')

    # Make sure we checked everything
    assert not checked_params
    plt.close()

    # Change the default rcParams
    ct.ctrlplot.rcParams.update(my_rcParams)
    if plot_fcn not in nolabel_plot_fcns:
        cplt = plot_fcn(
            *args1, **kwargs, **plot_kwargs, show_legend=True)
    else:
        cplt = plot_fcn(*args1, **kwargs, **plot_kwargs)

    # Check everything
    ax, fig = cplt.axes[-1, 0], cplt.figure
    assert ax.xaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    assert ax.yaxis.get_label().get_fontsize() == my_rcParams['axes.labelsize']
    assert ax.title.get_fontsize() == my_rcParams['axes.titlesize']
    assert ax.get_xticklabels()[0].get_fontsize() == \
        my_rcParams['xtick.labelsize']
    assert ax.get_yticklabels()[0].get_fontsize() == \
        my_rcParams['ytick.labelsize']
    assert fig._suptitle.get_fontsize() == my_rcParams['figure.titlesize']
    if plot_fcn not in nolabel_plot_fcns:
        for ax in cplt.axes.flatten():
            legend = ax.get_legend()
            if legend is not None:
                break
        assert legend is not None
        assert legend.get_texts()[0].get_fontsize() == \
            my_rcParams['legend.fontsize']
    plt.close()

    # Make sure that resetting parameters works correctly
    ct.reset_defaults()
    for key in ct.ctrlplot.rcParams:
        assert ct.defaults['ctrlplot.rcParams'][key] != my_rcParams[key]
        assert ct.ctrlplot.rcParams[key] != my_rcParams[key]


def test_deprecation_warnings():
    sys = ct.rss(2, 2, 2)
    lines = ct.step_response(sys).plot(overlay_traces=True)
    with pytest.warns(FutureWarning, match="deprecated"):
        assert len(lines[0, 0]) == 2

    cplt = ct.step_response(sys).plot()
    with pytest.warns(FutureWarning, match="deprecated"):
        axs = ct.get_plot_axes(cplt)
        assert np.all(axs == cplt.axes)

    with pytest.warns(FutureWarning, match="deprecated"):
        axs = ct.get_plot_axes(cplt.lines)
        assert np.all(axs == cplt.axes)

    with pytest.warns(FutureWarning, match="deprecated"):
        ct.suptitle("updated title")
        assert cplt.figure._suptitle.get_text() == "updated title"


def test_ControlPlot_init():
    sys = ct.rss(2, 2, 2)
    cplt = ct.step_response(sys).plot()

    # Create a ControlPlot from data, without the axes or figure
    cplt_raw = ct.ControlPlot(cplt.lines)
    assert np.all(cplt_raw.lines == cplt.lines)
    assert np.all(cplt_raw.axes == cplt.axes)
    assert cplt_raw.figure == cplt.figure


def test_pole_zero_subplots(savefig=False):
    ax_array = ct.pole_zero_subplots(2, 1, grid=[True, False])
    sys1 = ct.tf([1, 2], [1, 2, 3], name='sys1')
    sys2 = ct.tf([1, 0.2], [1, 1, 3, 1, 1], name='sys2')
    ct.root_locus_plot([sys1, sys2], ax=ax_array[0, 0])
    cplt = ct.root_locus_plot([sys1, sys2], ax=ax_array[1, 0])
    with pytest.warns(UserWarning, match="Tight layout not applied"):
        cplt.set_plot_title("Root locus plots (w/ specified axes)")
    if savefig:
        plt.savefig("ctrlplot-pole_zero_subplots.png")

    # Single type of of grid for all axes
    ax_array = ct.pole_zero_subplots(2, 2, grid='empty')
    assert ax_array[0, 0].xaxis.get_label().get_text() == ''

    # Discrete system grid
    ax_array = ct.pole_zero_subplots(2, 2, grid=True, dt=1)
    assert ax_array[0, 0].xaxis.get_label().get_text() == 'Real'
    assert ax_array[0, 0].get_lines()[0].get_color() == 'grey'


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

    #
    # Combination plot
    #

    P = ct.tf([0.02], [1, 0.1, 0.01])   # servomechanism
    C1 = ct.tf([1, 1], [1, 0])          # unstable
    L1 = P * C1
    C2 = ct.tf([1, 0.05], [1, 0])       # stable
    L2 = P * C2

    plt.rcParams.update(ct.rcParams)
    fig = plt.figure(figsize=[7, 4])
    ax_mag = fig.add_subplot(2, 2, 1)
    ax_phase = fig.add_subplot(2, 2, 3)
    ax_nyquist = fig.add_subplot(1, 2, 2)

    ct.bode_plot(
        [L1, L2], ax=[ax_mag, ax_phase],
        label=["$L_1$ (unstable)", "$L_2$ (unstable)"],
        show_legend=False)
    ax_mag.set_title("Bode plot for $L_1$, $L_2$")
    ax_mag.tick_params(labelbottom=False)
    fig.align_labels()

    ct.nyquist_plot(L1, ax=ax_nyquist, label="$L_1$ (unstable)")
    ct.nyquist_plot(
        L2, ax=ax_nyquist, label="$L_2$ (stable)",
        max_curve_magnitude=22, legend_loc='upper right')
    ax_nyquist.set_title("Nyquist plot for $L_1$, $L_2$")

    fig.suptitle("Loop analysis for servomechanism control design")
    plt.tight_layout()
    plt.savefig('ctrlplot-servomech.png')

    plt.figure()
    test_pole_zero_subplots(savefig=True)
