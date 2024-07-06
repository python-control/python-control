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
def test_plot_functions(resp_fcn, plot_fcn):
    # Create some systems to use
    sys1 = ct.rss(2, 1, 1, strictly_proper=True)
    sys2 = ct.rss(4, 1, 1, strictly_proper=True)

    # Set up arguments
    kwargs = meth_kwargs = plot_fcn_kwargs = {}
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

        case _, _:
            args = (sys1, )

    # Call the plot through the response function
    if resp_fcn is not None:
        resp = resp_fcn(*args, **kwargs)
        cplt1 = resp.plot(**kwargs, **meth_kwargs)
        assert isinstance(cplt1, ct.ControlPlot)

    # Call the plot directly, plotting on top of previous plot
    if plot_fcn not in [None, ct.time_response_plot]:
        cplt2 = plot_fcn(*args, **kwargs, **plot_fcn_kwargs)
        assert isinstance(cplt2, ct.ControlPlot)

        # Plot should have landed on top of previous plot
        if resp_fcn is not None:
            assert cplt2.figure == cplt1.figure
            assert np.all(cplt2.axes == cplt1.axes)
            assert len(cplt2.lines[0]) == len(cplt1.lines[0])

    # Pass axes explicitly
    if resp_fcn is not None:
        cplt3 = resp.plot(**kwargs, **meth_kwargs, ax=cplt1.axes)
        assert cplt3.figure == cplt1.figure
        assert np.all(cplt3.axes == cplt1.axes)
        assert len(cplt3.lines[0]) == len(cplt1.lines[0])


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
