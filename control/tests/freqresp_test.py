"""freqresp_test.py - test frequency response functions

RMM, 30 May 2016 (based on timeresp_test.py)

This is a rudimentary set of tests for frequency response functions,
including bode plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

import control as ctrl
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.matlab import ss, tf, bode, rss
from control.tests.conftest import slycotonly


pytestmark = pytest.mark.usefixtures("mplcleanup")


def test_siso():
    """Test SISO frequency response"""
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = 0
    sys = StateSpace(A, B, C, D)
    omega = np.linspace(10e-2, 10e2, 1000)

    # test frequency response
    sys.freqresp(omega)

    # test bode plot
    bode(sys)

    # Convert to transfer function and test bode
    systf = tf(sys)
    bode(systf)


@pytest.mark.filterwarnings("ignore:.*non-positive left xlim:UserWarning")
def test_superimpose():
    """Test superimpose multiple calls.

    Test to make sure that multiple calls to plots superimpose their
    data on the same axes unless told to do otherwise
    """
    # Generate two plots in a row; should be on the same axes
    plt.figure(1)
    plt.clf()
    ctrl.bode_plot(ctrl.tf([1], [1, 2, 1]))
    ctrl.bode_plot(ctrl.tf([5], [1, 1]))

    # Check that there are two axes and that each axes has two lines
    len(plt.gcf().axes) == 2
    for ax in plt.gcf().axes:
        # Make sure there are 2 lines in each subplot
        assert len(ax.get_lines()) == 2

    # Generate two plots as a list; should be on the same axes
    plt.figure(2)
    plt.clf()
    ctrl.bode_plot([ctrl.tf([1], [1, 2, 1]), ctrl.tf([5], [1, 1])])

    # Check that there are two axes and that each axes has two lines
    assert len(plt.gcf().axes) == 2
    for ax in plt.gcf().axes:
        # Make sure there are 2 lines in each subplot
        assert len(ax.get_lines()) == 2

    # Generate two separate plots; only the second should appear
    plt.figure(3)
    plt.clf()
    ctrl.bode_plot(ctrl.tf([1], [1, 2, 1]))
    plt.clf()
    ctrl.bode_plot(ctrl.tf([5], [1, 1]))

    # Check to make sure there are two axes and that each axes has one line
    assert len(plt.gcf().axes) == 2
    for ax in plt.gcf().axes:
        # Make sure there is only 1 line in the subplot
        assert len(ax.get_lines()) == 1

    # Now add a line to the magnitude plot and make sure if is there
    for ax in plt.gcf().axes:
        if ax.get_label() == 'control-bode-magnitude':
            break

    ax.semilogx([1e-2, 1e1], 20 * np.log10([1, 1]), 'k-')
    assert len(ax.get_lines()) == 2


def test_doubleint():
    """Test typcast bug with double int

    30 May 2016, RMM: added to replicate typecast bug in freqresp.py
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = 0
    sys = ss(A, B, C, D)
    bode(sys)


@slycotonly
def test_mimo():
    """Test MIMO frequency response calls"""
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0]])
    D = np.array([[0, 0]])
    omega = np.linspace(10e-2, 10e2, 1000)
    sysMIMO = ss(A, B, C, D)

    sysMIMO.freqresp(omega)
    tf(sysMIMO)


def test_bode_margin():
    """Test bode margins"""
    num = [1000]
    den = [1, 25, 100, 0]
    sys = ctrl.tf(num, den)
    plt.figure()
    ctrl.bode_plot(sys, margins=True, dB=False, deg=True, Hz=False)
    fig = plt.gcf()
    allaxes = fig.get_axes()

    mag_to_infinity = (np.array([6.07828691, 6.07828691]),
                       np.array([1., 1e-8]))
    assert_allclose(mag_to_infinity, allaxes[0].lines[2].get_data())

    gm_to_infinty = (np.array([10., 10.]),
                     np.array([4e-1, 1e-8]))
    assert_allclose(gm_to_infinty, allaxes[0].lines[3].get_data())

    one_to_gm = (np.array([10., 10.]),
                 np.array([1., 0.4]))
    assert_allclose(one_to_gm, allaxes[0].lines[4].get_data())

    pm_to_infinity = (np.array([6.07828691, 6.07828691]),
                      np.array([100000., -157.46405841]))
    assert_allclose(pm_to_infinity, allaxes[1].lines[2].get_data())

    pm_to_phase = (np.array([6.07828691, 6.07828691]),
                   np.array([-157.46405841, -180.]))
    assert_allclose(pm_to_phase, allaxes[1].lines[3].get_data())

    phase_to_infinity = (np.array([10., 10.]),
                         np.array([1e-8, -1.8e2]))
    assert_allclose(phase_to_infinity, allaxes[1].lines[4].get_data())


@pytest.fixture
def dsystem_dt(request):
    """Test systems for test_discrete"""
    # SISO state space systems with either fixed or unspecified sampling times
    sys = rss(3, 1, 1)

    # MIMO state space systems with either fixed or unspecified sampling times
    A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
    B = [[1., 4.], [-3., -3.], [-2., 1.]]
    C = [[4., 2., -3.], [1., 4., 3.]]
    D = [[-2., 4.], [0., 1.]]

    dt = request.param
    systems = {'sssiso': StateSpace(sys.A, sys.B, sys.C, sys.D, dt),
               'ssmimo': StateSpace(A, B, C, D, dt),
               'tf': TransferFunction([1, 1], [1, 2, 1], dt)}
    return systems


@pytest.fixture
def dsystem_type(request, dsystem_dt):
    """Return system by typekey"""
    systype = request.param
    return dsystem_dt[systype]


@pytest.mark.parametrize("dsystem_dt", [0.1, True], indirect=True)
@pytest.mark.parametrize("dsystem_type", ['sssiso', 'ssmimo', 'tf'],
                         indirect=True)
def test_discrete(dsystem_type):
    """Test discrete time frequency response"""
    dsys = dsystem_type
    # Set frequency range to just below Nyquist freq (for Bode)
    omega_ok = np.linspace(10e-4, 0.99, 100) * np.pi / dsys.dt

    # Test frequency response
    dsys.freqresp(omega_ok)

    # Check for warning if frequency is out of range
    with pytest.warns(UserWarning, match="above.*Nyquist"):
        # Look for a warning about sampling above Nyquist frequency
        omega_bad = np.linspace(10e-4, 1.1, 10) * np.pi / dsys.dt
        dsys.freqresp(omega_bad)

    # Test bode plots (currently only implemented for SISO)
    if (dsys.inputs == 1 and dsys.outputs == 1):
        # Generic call (frequency range calculated automatically)
        bode(dsys)

        # Convert to transfer function and test bode again
        systf = tf(dsys)
        bode(systf)

        # Make sure we can pass a frequency range
        bode(dsys, omega_ok)

    else:
        # Calling bode should generate a not implemented error
        with pytest.raises(NotImplementedError):
            bode((dsys,))


def test_options(editsdefaults):
    """Test ability to set parameter values"""
    # Generate a Bode plot of a transfer function
    sys = ctrl.tf([1000], [1, 25, 100, 0])
    fig1 = plt.figure()
    ctrl.bode_plot(sys, dB=False, deg=True, Hz=False)

    # Save the parameter values
    left1, right1 = fig1.axes[0].xaxis.get_data_interval()
    numpoints1 = len(fig1.axes[0].lines[0].get_data()[0])

    # Same transfer function, but add a decade on each end
    ctrl.config.set_defaults('freqplot', feature_periphery_decades=2)
    fig2 = plt.figure()
    ctrl.bode_plot(sys, dB=False, deg=True, Hz=False)
    left2, right2 = fig2.axes[0].xaxis.get_data_interval()

    # Make sure we got an extra decade on each end
    assert_allclose(left2, 0.1 * left1)
    assert_allclose(right2, 10 * right1)

    # Same transfer function, but add more points to the plot
    ctrl.config.set_defaults(
        'freqplot', feature_periphery_decades=2, number_of_samples=13)
    fig3 = plt.figure()
    ctrl.bode_plot(sys, dB=False, deg=True, Hz=False)
    numpoints3 = len(fig3.axes[0].lines[0].get_data()[0])

    # Make sure we got the right number of points
    assert numpoints1 != numpoints3
    assert numpoints3 == 13
