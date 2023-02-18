"""freqresp_test.py - test frequency response functions

RMM, 30 May 2016 (based on timeresp_test.py)

This is a rudimentary set of tests for frequency response functions,
including bode plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import math
import pytest

import control as ctrl
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.matlab import ss, tf, bode, rss
from control.freqplot import bode_plot, nyquist_plot, singular_values_plot
from control.tests.conftest import slycotonly

pytestmark = pytest.mark.usefixtures("mplcleanup")


@pytest.fixture
def ss_siso():
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = 0
    return StateSpace(A, B, C, D)


@pytest.fixture
def ss_mimo():
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    C = np.array([[1, 0]])
    D = np.array([[0, 0]])
    return StateSpace(A, B, C, D)


def test_freqresp_siso(ss_siso):
    """Test SISO frequency response"""
    omega = np.linspace(10e-2, 10e2, 1000)

    # test frequency response
    ctrl.freqresp(ss_siso, omega)


@slycotonly
def test_freqresp_mimo(ss_mimo):
    """Test MIMO frequency response calls"""
    omega = np.linspace(10e-2, 10e2, 1000)
    ctrl.freqresp(ss_mimo, omega)
    tf_mimo = tf(ss_mimo)
    ctrl.freqresp(tf_mimo, omega)


def test_bode_basic(ss_siso):
    """Test bode plot call (Very basic)"""
    # TODO: proper test
    tf_siso = tf(ss_siso)
    bode(ss_siso)
    bode(tf_siso)
    assert len(bode_plot(tf_siso, plot=False, omega_num=20)[0] == 20)
    omega = bode_plot(tf_siso, plot=False, omega_limits=(1, 100))[2]
    assert_allclose(omega[0], 1)
    assert_allclose(omega[-1], 100)
    assert len(bode_plot(tf_siso, plot=False, omega=np.logspace(-1,1,10))[0])\
         == 10


def test_nyquist_basic(ss_siso):
    """Test nyquist plot call (Very basic)"""
    # TODO: proper test
    tf_siso = tf(ss_siso)
    nyquist_plot(ss_siso)
    nyquist_plot(tf_siso)
    count, contour = nyquist_plot(
        tf_siso, plot=False, return_contour=True, omega_num=20)
    assert len(contour) == 20

    with pytest.warns(UserWarning, match="encirclements was a non-integer"):
        count, contour = nyquist_plot(
            tf_siso, plot=False, omega_limits=(1, 100), return_contour=True)
    assert_allclose(contour[0], 1j)
    assert_allclose(contour[-1], 100j)

    count, contour = nyquist_plot(
        tf_siso, plot=False, omega=np.logspace(-1, 1, 10), return_contour=True)
    assert len(contour) == 10


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

    30 May 2016, RMM: added to replicate typecast bug in frequency_response.py
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = 0
    sys = ss(A, B, C, D)
    bode(sys)


@pytest.mark.parametrize(
    "Hz, Wcp, Wcg",
    [pytest.param(False, 6.0782869, 10., id="omega"),
     pytest.param(True, 0.9673894, 1.591549, id="Hz")])
@pytest.mark.parametrize(
    "deg, p0, pm",
    [pytest.param(False, -np.pi, -2.748266, id="rad"),
     pytest.param(True, -180, -157.46405841, id="deg")])
@pytest.mark.parametrize(
    "dB, maginfty1, maginfty2, gminv",
    [pytest.param(False, 1, 1e-8, 0.4, id="mag"),
     pytest.param(True, 0, -1e+5, -7.9588, id="dB")])
def test_bode_margin(dB, maginfty1, maginfty2, gminv,
                     deg, p0, pm,
                     Hz, Wcp, Wcg):
    """Test bode margins"""
    num = [1000]
    den = [1, 25, 100, 0]
    sys = ctrl.tf(num, den)
    plt.figure()
    ctrl.bode_plot(sys, margins=True, dB=dB, deg=deg, Hz=Hz)
    fig = plt.gcf()
    allaxes = fig.get_axes()

    mag_to_infinity = (np.array([Wcp, Wcp]),
                       np.array([maginfty1, maginfty2]))
    assert_allclose(mag_to_infinity,
                    allaxes[0].lines[2].get_data(),
                    rtol=1e-5)

    gm_to_infinty = (np.array([Wcg, Wcg]),
                     np.array([gminv, maginfty2]))
    assert_allclose(gm_to_infinty,
                    allaxes[0].lines[3].get_data(),
                    rtol=1e-5)

    one_to_gm = (np.array([Wcg, Wcg]),
                 np.array([maginfty1, gminv]))
    assert_allclose(one_to_gm, allaxes[0].lines[4].get_data(),
                    rtol=1e-5)

    pm_to_infinity = (np.array([Wcp, Wcp]),
                      np.array([1e5, pm]))
    assert_allclose(pm_to_infinity,
                    allaxes[1].lines[2].get_data(),
                    rtol=1e-5)

    pm_to_phase = (np.array([Wcp, Wcp]),
                   np.array([pm, p0]))
    assert_allclose(pm_to_phase, allaxes[1].lines[3].get_data(),
                    rtol=1e-5)

    phase_to_infinity = (np.array([Wcg, Wcg]),
                         np.array([0, p0]))
    assert_allclose(phase_to_infinity, allaxes[1].lines[4].get_data(),
                    rtol=1e-5)


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
               'tf': TransferFunction([2, 1], [2, 1, 1], dt)}
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
    dsys.frequency_response(omega_ok)

    # Check for warning if frequency is out of range
    with pytest.warns(UserWarning, match="above.*Nyquist"):
        # Look for a warning about sampling above Nyquist frequency
        omega_bad = np.linspace(10e-4, 1.1, 10) * np.pi / dsys.dt
        dsys.frequency_response(omega_bad)

    # Test bode plots (currently only implemented for SISO)
    if (dsys.ninputs == 1 and dsys.noutputs == 1):
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

@pytest.mark.parametrize(
    "TF, initial_phase, default_phase, expected_phase",
    [pytest.param(ctrl.tf([1], [1, 0]),
                  None, -math.pi/2, -math.pi/2,         id="order1, default"),
     pytest.param(ctrl.tf([1], [1, 0]),
                  180, -math.pi/2, 3*math.pi/2,         id="order1, 180"),
     pytest.param(ctrl.tf([1], [1, 0, 0]),
                  None, -math.pi, -math.pi,             id="order2, default"),
     pytest.param(ctrl.tf([1], [1, 0, 0]),
                  180, -math.pi, math.pi,               id="order2, 180"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0]),
                  None, -3*math.pi/2, -3*math.pi/2,     id="order2, default"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0]),
                  180, -3*math.pi/2, math.pi/2,         id="order2, 180"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0]),
                  None, 0, 0,                           id="order4, default"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0]),
                  180, 0, 0,                            id="order4, 180"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0]),
                  -360, 0, -2*math.pi,                  id="order4, -360"),
     ])
def test_initial_phase(TF, initial_phase, default_phase, expected_phase):
    # Check initial phase of standard transfer functions
    mag, phase, omega = ctrl.bode(TF)
    assert(abs(phase[0] - default_phase) < 0.1)

    # Now reset the initial phase to +180 and see if things work
    mag, phase, omega = ctrl.bode(TF, initial_phase=initial_phase)
    assert(abs(phase[0] - expected_phase) < 0.1)

    # Make sure everything works in rad/sec as well
    if initial_phase:
        plt.xscale('linear')  # avoids xlim warning on next line
        plt.clf()  # clear previous figure (speeds things up)
        mag, phase, omega = ctrl.bode(
            TF, initial_phase=initial_phase/180. * math.pi, deg=False)
        assert(abs(phase[0] - expected_phase) < 0.1)


@pytest.mark.parametrize(
    "TF, wrap_phase, min_phase, max_phase",
    [pytest.param(ctrl.tf([1], [1, 0]),
                  None, -math.pi/2, 0,              id="order1, default"),
     pytest.param(ctrl.tf([1], [1, 0]),
                  True, -math.pi, math.pi,          id="order1, True"),
     pytest.param(ctrl.tf([1], [1, 0]),
                  -270, -3*math.pi/2, math.pi/2,    id="order1, -270"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0]),
                  None, -3*math.pi/2, 0,            id="order3, default"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0]),
                  True, -math.pi, math.pi,          id="order3, True"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0]),
                  -270, -3*math.pi/2, math.pi/2,    id="order3, -270"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0, 0]),
                  True, -3*math.pi/2, 0,            id="order5, default"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0, 0]),
                  True, -math.pi, math.pi,          id="order5, True"),
     pytest.param(ctrl.tf([1], [1, 0, 0, 0, 0, 0]),
                  -270, -3*math.pi/2, math.pi/2,    id="order5, -270"),
    ])
def test_phase_wrap(TF, wrap_phase, min_phase, max_phase):
    mag, phase, omega = ctrl.bode(TF, wrap_phase=wrap_phase)
    assert(min(phase) >= min_phase)
    assert(max(phase) <= max_phase)


def test_phase_wrap_multiple_systems():
    sys_unstable = ctrl.zpk([],[1,1], gain=1)

    mag, phase, omega = ctrl.bode(sys_unstable, plot=False)
    assert(np.min(phase) >= -2*np.pi)
    assert(np.max(phase) <= -1*np.pi)

    mag, phase, omega = ctrl.bode((sys_unstable, sys_unstable), plot=False)
    assert(np.min(phase) >= -2*np.pi)
    assert(np.max(phase) <= -1*np.pi)


def test_freqresp_warn_infinite():
    """Test evaluation warnings for transfer functions w/ pole at the origin"""
    sys_finite = ctrl.tf([1], [1, 0.01])
    sys_infinite = ctrl.tf([1], [1, 0.01, 0])

    # Transfer function with finite zero frequency gain
    np.testing.assert_almost_equal(sys_finite(0), 100)
    np.testing.assert_almost_equal(sys_finite(0, warn_infinite=False), 100)
    np.testing.assert_almost_equal(sys_finite(0, warn_infinite=True), 100)

    # Transfer function with infinite zero frequency gain
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        np.testing.assert_almost_equal(
            sys_infinite(0), complex(np.inf, np.nan))
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        np.testing.assert_almost_equal(
            sys_infinite(0, warn_infinite=True), complex(np.inf, np.nan))
    np.testing.assert_almost_equal(
        sys_infinite(0, warn_infinite=False), complex(np.inf, np.nan))

    # Switch to state space
    sys_finite = ctrl.tf2ss(sys_finite)
    sys_infinite = ctrl.tf2ss(sys_infinite)

    # State space system with finite zero frequency gain
    np.testing.assert_almost_equal(sys_finite(0), 100)
    np.testing.assert_almost_equal(sys_finite(0, warn_infinite=False), 100)
    np.testing.assert_almost_equal(sys_finite(0, warn_infinite=True), 100)

    # State space system with infinite zero frequency gain
    with pytest.warns(RuntimeWarning, match="singular matrix"):
        np.testing.assert_almost_equal(
            sys_infinite(0), complex(np.inf, np.nan))
    with pytest.warns(RuntimeWarning, match="singular matrix"):
        np.testing.assert_almost_equal(
            sys_infinite(0, warn_infinite=True), complex(np.inf, np.nan))
    np.testing.assert_almost_equal(sys_infinite(
        0, warn_infinite=False), complex(np.inf, np.nan))


def test_dcgain_consistency():
    """Test to make sure that DC gain is consistently evaluated"""
    # Set up transfer function with pole at the origin
    sys_tf = ctrl.tf([1], [1, 0])
    assert 0 in sys_tf.poles()

    # Set up state space system with pole at the origin
    sys_ss = ctrl.tf2ss(sys_tf)
    assert 0 in sys_ss.poles()

    # Finite (real) numerator over 0 denominator => inf + nanj
    np.testing.assert_equal(
        sys_tf(0, warn_infinite=False), complex(np.inf, np.nan))
    np.testing.assert_equal(
        sys_ss(0, warn_infinite=False), complex(np.inf, np.nan))
    np.testing.assert_equal(
        sys_tf(0j, warn_infinite=False), complex(np.inf, np.nan))
    np.testing.assert_equal(
        sys_ss(0j, warn_infinite=False), complex(np.inf, np.nan))
    np.testing.assert_equal(
        sys_tf.dcgain(), np.inf)
    np.testing.assert_equal(
        sys_ss.dcgain(), np.inf)

    # Set up transfer function with pole, zero at the origin
    sys_tf = ctrl.tf([1, 0], [1, 0])
    assert 0 in sys_tf.poles()
    assert 0 in sys_tf.zeros()

    # Pole and zero at the origin should give nan + nanj for the response
    np.testing.assert_equal(
        sys_tf(0, warn_infinite=False), complex(np.nan, np.nan))
    np.testing.assert_equal(
        sys_tf(0j, warn_infinite=False), complex(np.nan, np.nan))
    np.testing.assert_equal(
        sys_tf.dcgain(), np.nan)

    # Set up state space version
    sys_ss = ctrl.tf2ss(ctrl.tf([1, 0], [1, 1])) * \
        ctrl.tf2ss(ctrl.tf([1], [1, 0]))

    # Different systems give different representations => test accordingly
    if 0 in sys_ss.poles() and 0 in sys_ss.zeros():
        # Pole and zero at the origin => should get (nan + nanj)
        np.testing.assert_equal(
            sys_ss(0, warn_infinite=False), complex(np.nan, np.nan))
        np.testing.assert_equal(
            sys_ss(0j, warn_infinite=False), complex(np.nan, np.nan))
        np.testing.assert_equal(
            sys_ss.dcgain(), np.nan)
    elif 0 in sys_ss.poles():
        # Pole at the origin, but zero elsewhere => should get (inf + nanj)
        np.testing.assert_equal(
            sys_ss(0, warn_infinite=False), complex(np.inf, np.nan))
        np.testing.assert_equal(
            sys_ss(0j, warn_infinite=False), complex(np.inf, np.nan))
        np.testing.assert_equal(
            sys_ss.dcgain(), np.inf)
    else:
        # Near pole/zero cancellation => nothing sensible to check
        pass

    # Pole with non-zero, complex numerator => inf + infj
    s = ctrl.tf('s')
    sys_tf = (s + 1) / (s**2 + 1)
    assert 1j in sys_tf.poles()

    # Set up state space system with pole on imaginary axis
    sys_ss = ctrl.tf2ss(sys_tf)
    assert 1j in sys_tf.poles()

    # Make sure we get correct response if evaluated at the pole
    np.testing.assert_equal(
        sys_tf(1j, warn_infinite=False), complex(np.inf, np.inf))

    # For state space, numerical errors come into play
    resp_ss = sys_ss(1j, warn_infinite=False)
    if np.isfinite(resp_ss):
        assert abs(resp_ss) > 1e15
    else:
        if resp_ss != complex(np.inf, np.inf):
            pytest.xfail("statesp evaluation at poles not fully implemented")
        else:
            np.testing.assert_equal(resp_ss, complex(np.inf, np.inf))

    # DC gain is finite
    np.testing.assert_almost_equal(sys_tf.dcgain(), 1.)
    np.testing.assert_almost_equal(sys_ss.dcgain(), 1.)

    # Make sure that we get the *signed* DC gain
    sys_tf = -1 / (s + 1)
    np.testing.assert_almost_equal(sys_tf.dcgain(), -1)

    sys_ss = ctrl.tf2ss(sys_tf)
    np.testing.assert_almost_equal(sys_ss.dcgain(), -1)


# Testing of the singular_value_plot function
class TSys:
    """Struct of test system"""
    def __init__(self, sys=None, call_kwargs=None):
        self.sys = sys
        self.kwargs = call_kwargs if call_kwargs else {}

    def __repr__(self):
        """Show system when debugging"""
        return self.sys.__repr__()


@pytest.fixture
def ss_mimo_ct():
    A = np.diag([-1/75.0, -1/75.0])
    B = np.array([[87.8, -86.4],
                  [108.2, -109.6]])/75.0
    C = np.eye(2)
    D = np.zeros((2, 2))
    T = TSys(ss(A, B, C, D))
    T.omegas = [0.0, [0.0], np.array([0.0, 0.01])]
    T.sigmas = [np.array([[197.20868123], [1.39141948]]),
                np.array([[197.20868123], [1.39141948]]),
                np.array([[197.20868123, 157.76694498], [1.39141948, 1.11313558]])
    ]
    return T


@pytest.fixture
def ss_miso_ct():
    A = np.diag([-1 / 75.0])
    B = np.array([[87.8, -86.4]]) / 75.0
    C = np.array([[1]])
    D = np.zeros((1, 2))
    T = TSys(ss(A, B, C, D))
    T.omegas = [0.0, np.array([0.0, 0.01])]
    T.sigmas = [np.array([[123.1819792]]),
                np.array([[123.1819792, 98.54558336]])]
    return T


@pytest.fixture
def ss_simo_ct():
    A = np.diag([-1 / 75.0])
    B = np.array([[1.0]]) / 75.0
    C = np.array([[87.8], [108.2]])
    D = np.zeros((2, 1))
    T = TSys(ss(A, B, C, D))
    T.omegas = [0.0, np.array([0.0, 0.01])]
    T.sigmas = [np.array([[139.34159465]]),
                np.array([[139.34159465, 111.47327572]])]
    return T


@pytest.fixture
def ss_siso_ct():
    A = np.diag([-1 / 75.0])
    B = np.array([[1.0]]) / 75.0
    C = np.array([[87.8]])
    D = np.zeros((1, 1))
    T = TSys(ss(A, B, C, D))
    T.omegas = [0.0, np.array([0.0, 0.01])]
    T.sigmas = [np.array([[87.8]]),
                np.array([[87.8, 70.24]])]
    return T


@pytest.fixture
def ss_mimo_dt():
    A = np.array([[0.98675516, 0.],
                  [0., 0.98675516]])
    B = np.array([[1.16289679, -1.14435402],
                  [1.43309149, -1.45163427]])
    C = np.eye(2)
    D = np.zeros((2, 2))
    T = TSys(ss(A, B, C, D, dt=1.0))
    T.omegas = [0.0, np.array([0.0, 0.001, 0.01])]
    T.sigmas = [np.array([[197.20865428], [1.39141936]]),
                np.array([[197.20865428, 196.6563423, 157.76758858],
                         [1.39141936, 1.38752248, 1.11314018]])]
    return T


@pytest.fixture
def tsystem(request, ss_mimo_ct, ss_miso_ct, ss_simo_ct, ss_siso_ct, ss_mimo_dt):

    systems = {"ss_mimo_ct": ss_mimo_ct,
               "ss_miso_ct": ss_miso_ct,
               "ss_simo_ct": ss_simo_ct,
               "ss_siso_ct": ss_siso_ct,
               "ss_mimo_dt": ss_mimo_dt
               }
    return systems[request.param]


@pytest.mark.parametrize("tsystem",
                         ["ss_mimo_ct", "ss_miso_ct", "ss_simo_ct", "ss_siso_ct", "ss_mimo_dt"], indirect=["tsystem"])
def test_singular_values_plot(tsystem):
    sys = tsystem.sys
    for omega_ref, sigma_ref in zip(tsystem.omegas, tsystem.sigmas):
        sigma, _ = singular_values_plot(sys, omega_ref, plot=False)
        np.testing.assert_almost_equal(sigma, sigma_ref)


def test_singular_values_plot_mpl_base(ss_mimo_ct, ss_mimo_dt):
    sys_ct = ss_mimo_ct.sys
    sys_dt = ss_mimo_dt.sys
    plt.figure()
    singular_values_plot(sys_ct, plot=True)
    fig = plt.gcf()
    allaxes = fig.get_axes()
    assert(len(allaxes) == 1)
    assert(allaxes[0].get_label() == 'control-sigma')
    plt.figure()
    singular_values_plot([sys_ct, sys_dt], plot=True, Hz=True, dB=True, grid=False)
    fig = plt.gcf()
    allaxes = fig.get_axes()
    assert(len(allaxes) == 1)
    assert(allaxes[0].get_label() == 'control-sigma')


def test_singular_values_plot_mpl_superimpose_nyq(ss_mimo_ct, ss_mimo_dt):
    sys_ct = ss_mimo_ct.sys
    sys_dt = ss_mimo_dt.sys
    omega_all = np.logspace(-3, 2, 1000)
    plt.figure()
    singular_values_plot(sys_ct, omega_all, plot=True)
    singular_values_plot(sys_dt, omega_all, plot=True)
    fig = plt.gcf()
    allaxes = fig.get_axes()
    assert(len(allaxes) == 1)
    assert (allaxes[0].get_label() == 'control-sigma')
    nyquist_line = allaxes[0].lines[-1].get_data()
    assert(len(nyquist_line[0]) == 2)
    assert(nyquist_line[0][0] == nyquist_line[0][1])
    assert(nyquist_line[0][0] == np.pi/sys_dt.dt)
