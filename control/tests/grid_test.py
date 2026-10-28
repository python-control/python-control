import numpy as np
import matplotlib.pyplot as plt

from control.grid import sgrid


def test_sgrid_does_not_unwrap_longitude(mplcleanup):
    ax, fig = sgrid()
    finder = ax.get_grid_helper().grid_finder.extreme_finder

    assert finder.lon_cycle is None
    limits = finder(lambda x, y: (x, y), 90, 0, 270.001, 1)
    np.testing.assert_allclose(limits, (90, 270, 0, 1.05))
    plt.close(fig)
