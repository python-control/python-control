"""nichols_test.py - test Nichols plot

RMM, 31 Mar 2011
"""

import pytest

from control import StateSpace, nichols_plot, nichols


@pytest.fixture()
def tsys():
    """Set up a system to test operations on."""
    A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
    B = [[1.], [-3.], [-2.]]
    C = [[4., 2., -3.]]
    D = [[0.]]
    return StateSpace(A, B, C, D)


def test_nichols(tsys, mplcleanup):
    """Generate a Nichols plot."""
    nichols_plot(tsys)


def test_nichols_alias(tsys, mplcleanup):
    """Test the control.nichols alias and the grid=False parameter"""
    nichols(tsys, grid=False)
