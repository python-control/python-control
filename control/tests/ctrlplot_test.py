# ctrlplot_test.py - test out control plotting utilities
# RMM, 27 Jun 2024

import matplotlib.pyplot as plt
import pytest

import control as ct


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
