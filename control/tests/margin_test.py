#!/usr/bin/env pytest
"""
margin_test.py - test suite for stability margin commands

RMM, 15 Jul 2011
BG, 30 Jun 2020 -- convert to pytest, gh-425
BG, 16 Nov 2020 -- pick from gh-438 and add discrete test
"""

import numpy as np
import pytest
from numpy import inf, nan
from numpy.testing import assert_allclose

from control import ControlMIMONotImplemented, FrequencyResponseData, \
    StateSpace, TransferFunction, margin, phase_crossover_frequencies, \
    stability_margins, disk_margins, tf, ss
from control.exception import slycot_check

s = TransferFunction.s

@pytest.fixture(params=[
    # sysfn, args,
    # stability_margins(sys),
    # stability_margins(sys, returnall=True)
    (TransferFunction, ([1, 2], [1, 2, 3]),
     (inf, inf, inf, nan, nan, nan),
     ([], [], [], [], [], [])),
    (TransferFunction, ([1], [1, 2, 3, 4]),
     (2., inf, 0.4170, 1.7321, nan, 1.6620),
     ([2.], [], [1.2500, 0.4170], [1.7321], [], [0.1690, 1.6620])),
    (StateSpace, ([[1., 4.],
                   [3., 2.]],
                  [[1.], [-4.]],
                  [[1., 0.]],
                  [[0.]]),
     (inf, 147.0743, inf, nan, 2.5483, nan),
     ([], [147.0743], [], [], [2.5483], [])),
    (None, ((8.75 * (4 * s**2 + 0.4 * s + 1))
            / ((100 * s + 1) * (s**2 + 0.22 * s + 1))
            / (s**2 / 10.**2 + 2 * 0.04 * s / 10. + 1)),
     (2.2716, 97.5941, 0.5591, 10.0053, 0.0850, 9.9918),
     ([2.2716], [97.5941, -157.7844, 134.7359], [1.0381, 0.5591],
      [10.0053], [0.0850, 0.9373, 1.0919], [0.4064, 9.9918])),
    (None, (1 / (1 + s)),  # no gain/phase crossovers
           (inf, inf, inf, nan, nan, nan),
           ([], [], [], [], [], [])),
    (None, (3 * (10 + s) / (2 + s)),  # no gain/phase crossovers
     (inf, inf, inf, nan, nan, nan),
     ([], [], [], [], [], [])),
    (None, 0.01 * (10 - s) / (2 + s) / (1 + s),  # no phase crossovers
     (300.0, inf, 0.9917, 5.6569, nan, 2.3171),
     ([300.0], [], [0.9917], [5.6569], [], 2.3171)),
])
def tsys(request):
    """Return test systems and reference data"""
    sysfn, args = request.param[:2]
    if sysfn:
        sys = sysfn(*args)
    else:
        sys = args
    return (sys,) + request.param[2:]

def compare_allmargins(actual, desired, **kwargs):
    """Compare all elements of stability_margins(returnall=True) result"""
    assert len(actual) == len(desired)
    for a, d in zip(actual, desired):
        assert_allclose(a, d, **kwargs)


def test_stability_margins(tsys):
    sys, refout, refoutall = tsys
    """Test stability_margins() function"""
    out = stability_margins(sys)
    assert_allclose(out, refout, atol=1.5e-2)
    out = stability_margins(sys, returnall=True)
    compare_allmargins(out, refoutall, atol=1.5e-2)



def test_stability_margins_omega(tsys):
    sys, refout, refoutall = tsys
    """Test stability_margins() with interpolated frequencies"""
    omega = np.logspace(-2, 2, 2000)
    out = stability_margins(FrequencyResponseData(sys, omega))
    assert_allclose(out, refout, atol=1.5e-3)


def test_stability_margins_3input(tsys):
    sys, refout, refoutall = tsys
    """Test stability_margins() function with mag, phase, omega input"""
    omega = np.logspace(-2, 2, 2000)
    mag, phase, omega_ = sys.frequency_response(omega)
    out = stability_margins((mag, phase*180/np.pi, omega_))
    assert_allclose(out, refout, atol=1.5e-3)


def test_margin_sys(tsys):
    sys, refout, refoutall = tsys
    """Test margin() function with system input"""
    out = margin(sys)
    assert_allclose(out, np.array(refout)[[0, 1, 3, 4]], atol=1.5e-3)

def test_margin_3input(tsys):
    sys, refout, refoutall = tsys
    """Test margin() function with mag, phase, omega input"""
    omega = np.logspace(-2, 2, 2000)
    mag, phase, omega_ = sys.frequency_response(omega)
    out = margin((mag, phase*180/np.pi, omega_))
    assert_allclose(out, np.array(refout)[[0, 1, 3, 4]], atol=1.5e-3)

@pytest.mark.parametrize(
    'tfargs, omega_ref, gain_ref',
    [(([1], [1, 2, 3, 4]), [1.7325, 0.], [-0.5, 0.25]),
     (([1], [1, 1]), [0.], [1.]),
     (([2], [1, 3, 3, 1]), [1.732, 0.], [-0.25, 2.]),
     ((np.array([3, 11, 3]) * 1e-4, [1., -2.7145, 2.4562, -0.7408], .1),
      [1.6235, 0.], [-0.28598, 1.88889]),
     (([200.0], [1.0, 21.0, 20.0, 0.0]),
      [4.47213595, 0], [-0.47619048, inf]),
     ])
@pytest.mark.filterwarnings("error")
def test_phase_crossover_frequencies(tfargs, omega_ref, gain_ref):
    """Test phase_crossover_frequencies() function"""
    sys = TransferFunction(*tfargs)
    omega, gain = phase_crossover_frequencies(sys)
    assert_allclose(omega, omega_ref, atol=1.5e-3)
    assert_allclose(gain, gain_ref, atol=1.5e-3)


def test_phase_crossover_frequencies_mimo():
    """Test MIMO exception"""
    tf = TransferFunction([[[1], [2]],
                           [[3], [4]]],
                          [[[1, 2, 3, 4], [1, 1]],
                           [[1, 1], [1, 1]]])
    with pytest.raises(ControlMIMONotImplemented):
        omega, gain = phase_crossover_frequencies(tf)


def test_mag_phase_omega():
    """Test for bug reported in gh-58"""
    sys = TransferFunction(15, [1, 6, 11, 6])
    out = stability_margins(sys)
    omega = np.logspace(-2, 2, 1000)
    mag, phase, omega = sys.frequency_response(omega)
    out2 = stability_margins((mag, phase*180/np.pi, omega))
    ind = [0, 1, 3, 4]   # indices of gm, pm, wg, wp -- ignore sm
    marg1 = np.array(out)[ind]
    marg2 = np.array(out2)[ind]
    assert_allclose(marg1, marg2, atol=1.5e-3)


def test_frd():
    """Test FrequencyResonseData margins"""
    f = np.array([0.005, 0.010, 0.020, 0.030, 0.040,
                  0.050, 0.060, 0.070, 0.080, 0.090,
                  0.100, 0.200, 0.300, 0.400, 0.500,
                  0.750, 1.000, 1.250, 1.500, 1.750,
                  2.000, 2.250, 2.500, 2.750, 3.000,
                  3.250, 3.500, 3.750, 4.000, 4.250,
                  4.500, 4.750, 5.000, 6.000, 7.000,
                  8.000, 9.000, 10.000])
    gain = np.array([  0.0,   0.0,   0.0,   0.0,   0.0,
                       0.0,   0.0,   0.0,   0.0,   0.0,
                       0.0,   0.1,   0.2,   0.3,   0.5,
                       0.5,  -0.4,  -2.3,  -4.8,  -7.3,
                      -9.6, -11.7, -13.6, -15.3, -16.9,
                     -18.3, -19.6, -20.8, -22.0, -23.1,
                     -24.1, -25.0, -25.9, -29.1, -31.9,
                     -34.2, -36.2, -38.1])
    phase = np.array([  0,    -1,    -2,    -3,    -4,
                       -5,    -6,    -7,    -8,    -9,
                      -10,   -19,   -29,   -40,   -51,
                      -81,  -114,  -144,  -168,  -187,
                     -202,  -214,  -224,  -233,  -240,
                     -247,  -253,  -259,  -264,  -269,
                     -273,  -277,  -280,  -292,  -301,
                     -307,  -313,  -317])
    # calculate response as complex number
    resp = 10**(gain / 20) * np.exp(1j * phase / (180./np.pi))
    # frequency response data
    fresp = FrequencyResponseData(resp, f*2*np.pi, smooth=True)
    s = TransferFunction([1, 0], [1])
    G = 1./(s**2)
    K = 1.
    C = K*(1+1.9*s)
    TFopen = fresp*C*G
    gm, pm, sm, wg, wp, ws = stability_margins(TFopen)
    assert_allclose([pm], [44.55], atol=.01)


def test_frd_indexing():
    """Test FRD edge cases

    Make sure frd objects with non benign data do not raise exceptions when
    the stability criteria evaluate at the first or last frequency point
    bug reported in gh-407
    """
    # frequency points just a little under 1. and over 2.
    w = np.linspace(.99, 2.01, 11)

    # Note: stability_margins will convert the frd with smooth=True

    # gain margins
    # p crosses -180 at w[0]=1. and w[-1]=2.
    m = 0.6
    p = -180*(2*w-1)
    d = m*np.exp(1J*np.pi/180*p)
    frd_gm = FrequencyResponseData(d, w)
    gm, _, _, wg, _, _ = stability_margins(frd_gm, returnall=True)
    assert_allclose(gm, [1/m, 1/m], atol=0.01)
    assert_allclose(wg, [1., 2.], atol=0.01)

    # phase margins
    # m crosses 1 at w[0]=1. and w[-1]=2.
    m = -(2*w-3)**4 + 2
    p = -90.
    d = m*np.exp(1J*np.pi/180*p)
    frd_pm = FrequencyResponseData(d, w)
    _, pm, _, _, wp, _ = stability_margins(frd_pm, returnall=True)
    assert_allclose(pm, [90., 90.], atol=0.01)
    assert_allclose(wp, [1., 2.], atol=0.01)

    # stability margins
    # minimum abs(d+1)=1-m at w[1]=1. and w[-2]=2., in nyquist plot
    w = np.arange(.9, 2.1, 0.1)
    m = 0.6
    p = -180*(2*w-1)
    d = m*np.exp(1J*np.pi/180*p)
    frd_sm = FrequencyResponseData(d, w)
    _, _, sm, _, _, ws = stability_margins(frd_sm, returnall=True)
    assert_allclose(sm, [1-m, 1-m], atol=0.01)
    assert_allclose(ws, [1., 2.], atol=0.01)


@pytest.fixture
def tsys_zmoresystems():
    """A cornucopia of tricky systems for phase / gain margin

    `example*` from "A note on the Gain and Phase Margin Concepts
    Journal of Control and Systems Engineering, Yazdan Bavafi-Toosi,
    Dec 2015, vol 3 iss 1, pp 51-59

    TODO: still have to convert more to tests + fix margin to handle
    also these torture cases
    """

    systems = {
        'typem1': s/(s+1),
        'type0': 1/(s+1)**3,
        'type1': (s + 0.1)/s/(s+1),
        'type2': (s + 0.1)/s**2/(s+1),
        'type3': (s + 0.1)*(s+0.1)/s**3/(s+1),
        'example21': 0.002*(s+0.02)*(s+0.05)*(s+5)*(s+10) / (
                    (s-0.0005)*(s+0.0001)*(s+0.01)*(s+0.2)*(s+1)*(s+100)**2),
        'example23': ((s+0.1)**2 + 1)*(s-0.1)/(((s+0.1)**2+4)*(s+1)),
        'example25a': s/(s**2+2*s+2)**4,
        'example26a': ((s-0.1)**2 + 1)/((s + 0.1)*((s-0.2)**2 + 4)),
        'example26b': ((s-0.1)**2 + 1)/((s - 0.3)*((s-0.2)**2 + 4))
    }
    systems['example24'] = systems['example21'] * 20000
    systems['example25b'] = systems['example25a'] * 100
    systems['example22'] = systems['example21'] * (s**2 - 2*s + 401)
    return systems


@pytest.fixture
def tsys_zmore(request, tsys_zmoresystems):
    tsys = request.param
    tsys['sys'] = tsys_zmoresystems[tsys['sysname']]
    return tsys


@pytest.mark.parametrize(
    'tsys_zmore',
    [dict(sysname='typem1', K=2.0, atol=1.5e-3,
          result=(float('Inf'), -120.0007, float('NaN'), 0.5774)),
     dict(sysname='type0', K=0.8, atol=1.5e-3,
          result=(10.0014, float('inf'), 1.7322, float('nan'))),
     dict(sysname='type0', K=2.0, atol=1e-2,
          result=(4.000, 67.6058, 1.7322, 0.7663)),
     dict(sysname='type1', K=1.0, atol=1e-4,
          result=(float('Inf'), 144.9032, float('NaN'), 0.3162)),
     dict(sysname='type2', K=1.0, atol=1e-4,
          result=(float('Inf'), 44.4594, float('NaN'), 0.7907)),
     dict(sysname='type3', K=1.0, atol=1.5e-3,
          result=(0.0626, 37.1748, 0.1119, 0.7951)),
     dict(sysname='example21', K=1.0, atol=1e-2,
          result=(0.0100, -14.5640, 0, 0.0022)),
     dict(sysname='example21', K=1000.0, atol=1e-2,
          result=(0.1793, 22.5215, 0.0243, 0.0630)),
     dict(sysname='example21', K=5000.0, atol=1.5e-3,
          result=(4.5596, 21.2101, 0.4385, 0.1868)),
     ],
    indirect=True)
def test_zmore_margin(tsys_zmore):
    """Test margins for more tricky systems

    Note
    ----
    Matlab gives gain margin 0 for system `type2`, python-control gives inf
    Difficult to argue which is right? Special case or different approach?

    Edge cases, like `type0` which approaches a gain of 1 for w -> 0, are also
    not identically indicated, Matlab gives phase margin -180, at w = 0. For
    higher or lower gains, results match.
    """

    res = margin(tsys_zmore['sys'] * tsys_zmore['K'])
    assert_allclose(res, tsys_zmore['result'], atol=tsys_zmore['atol'])


@pytest.mark.parametrize(
    'tsys_zmore',
    [dict(sysname='example21', K=1.0, rtol=1e-3, atol=1e-3,
          result=([0.01, 179.2931, 2.2798e+4, 1.5946e+07, 7.2477e+08],
                  [-14.5640],
                  [0.2496],
                  [0, 0.0243, 0.4385, 6.8640, 84.9323],
                  [0.0022],
                  [0.0022])),
    ],
    indirect=True)
def test_zmore_stability_margins(tsys_zmore):
    """Test stability_margins for more tricky systems with returnall"""
    res = stability_margins(tsys_zmore['sys'] * tsys_zmore['K'],
                            returnall=True)
    compare_allmargins(res,
                       tsys_zmore['result'],
                       atol=tsys_zmore['atol'],
                       rtol=tsys_zmore['rtol'])


@pytest.mark.parametrize(
    'cnum, cden, dt,'
    'ref,'
    'rtol, poly_is_inaccurate',
    [( # gh-465
      [2], [1, 3, 2, 0], 1e-2,
      [ 2.955761, 32.398492,  0.429535,  1.403725,  0.749367,  0.923898],
      1e-5, True),
     ( # 2/(s+1)**3
      [2], [1, 3, 3, 1], .1,
      [3.4927, 65.4212, 0.5763, 1.6283, 0.76625, 1.2019],
      1e-4, True),
     ( # gh-523 a
      [1.1 * 4 * np.pi**2], [1, 2 * 0.2 * 2 * np.pi,  4 * np.pi**2], .05,
      [2.3842, 18.161, 0.26953, 11.712, 8.7478, 9.1504],
      1e-4, False),
     ( # gh-523 b
       # H1 = w1**2 / (z**2 + 2*zt*w1 * z + w1**2)
       # H2 = w2**2 / (z**2 + 2*zt*w2 * z + w2**2)
       # H = H1 * H2
       # w1 = 1, w2 = 100, zt = 0.5
      [5e4], [1., 101., 10101., 10100., 10000.], 1e-3,
      [18.8766, 26.3564, 0.406841, 9.76358, 2.32933, 2.55986],
      1e-5, True),
     ])
@pytest.mark.filterwarnings("error")
def test_stability_margins_discrete(cnum, cden, dt,
                                    ref,
                                    rtol, poly_is_inaccurate):
    """Test stability_margins with discrete TF input"""
    tf = TransferFunction(cnum, cden).sample(dt)
    if poly_is_inaccurate:
        with pytest.warns(UserWarning, match="numerical inaccuracy in 'poly'"):
            out = stability_margins(tf)
        # cover the explicit frd branch and make sure it yields the same
        # results as the fallback mechanism
        out_frd = stability_margins(tf, method='frd')
        assert_allclose(out, out_frd)
    else:
        out = stability_margins(tf)
    assert_allclose(out, ref, rtol=rtol)

def test_siso_disk_margin():
    # Frequencies of interest
    omega = np.logspace(-1, 2, 1001)

    # Loop transfer function
    L = tf(25, [1, 10, 10, 10])

    # Balanced (S - T) disk-based stability margins
    DM, DGM, DPM = disk_margins(L, omega, skew=0.0)
    assert_allclose([DM], [0.46], atol=0.1) # disk margin of 0.46
    assert_allclose([DGM], [4.05], atol=0.1) # disk-based gain margin of 4.05 dB
    assert_allclose([DPM], [25.8], atol=0.1) # disk-based phase margin of 25.8 deg

    # For SISO systems, the S-based (S) disk margin should match the third output
    # of existing library "stability_margins", i.e., minimum distance from the
    # Nyquist plot to -1.
    _, _, SM = stability_margins(L)[:3]
    DM = disk_margins(L, omega, skew=1.0)[0]
    assert_allclose([DM], [SM], atol=0.01)

def test_mimo_disk_margin():
    # Frequencies of interest
    omega = np.logspace(-1, 3, 1001)

    # Loop transfer gain
    P = ss([[0, 10], [-10, 0]], np.eye(2), [[1, 10], [-10, 1]], 0) # plant
    K = ss([], [], [], [[1, -2], [0, 1]]) # controller
    Lo = P * K # loop transfer function, broken at plant output
    Li = K * P # loop transfer function, broken at plant input

    if slycot_check():
        # Balanced (S - T) disk-based stability margins at plant output
        DMo, DGMo, DPMo = disk_margins(Lo, omega, skew=0.0)
        assert_allclose([DMo], [0.3754], atol=0.1) # disk margin of 0.3754
        assert_allclose([DGMo], [3.3], atol=0.1) # disk-based gain margin of 3.3 dB
        assert_allclose([DPMo], [21.26], atol=0.1) # disk-based phase margin of 21.26 deg

        # Balanced (S - T) disk-based stability margins at plant input
        DMi, DGMi, DPMi = disk_margins(Li, omega, skew=0.0)
        assert_allclose([DMi], [0.3754], atol=0.1) # disk margin of 0.3754
        assert_allclose([DGMi], [3.3], atol=0.1) # disk-based gain margin of 3.3 dB
        assert_allclose([DPMi], [21.26], atol=0.1) # disk-based phase margin of 21.26 deg
    else:
        # Slycot not installed.  Should throw exception.
        with pytest.raises(ControlMIMONotImplemented,\
            match="Need slycot to compute MIMO disk_margins"):
            DMo, DGMo, DPMo = disk_margins(Lo, omega, skew=0.0)

def test_siso_disk_margin_return_all():
    # Frequencies of interest
    omega = np.logspace(-1, 2, 1001)

    # Loop transfer function
    L = tf(25, [1, 10, 10, 10])

    # Balanced (S - T) disk-based stability margins
    DM, DGM, DPM = disk_margins(L, omega, skew=0.0, returnall=True)
    assert_allclose([omega[np.argmin(DM)]], [1.94],\
        atol=0.01) # sensitivity peak at 1.94 rad/s
    assert_allclose([min(DM)], [0.46], atol=0.1) # disk margin of 0.46
    assert_allclose([DGM[np.argmin(DM)]], [4.05],\
        atol=0.1) # disk-based gain margin of 4.05 dB
    assert_allclose([DPM[np.argmin(DM)]], [25.8],\
        atol=0.1) # disk-based phase margin of 25.8 deg

def test_mimo_disk_margin_return_all():
    # Frequencies of interest
    omega = np.logspace(-1, 3, 1001)

    # Loop transfer gain
    P = ss([[0, 10], [-10, 0]], np.eye(2),\
        [[1, 10], [-10, 1]], 0) # plant
    K = ss([], [], [], [[1, -2], [0, 1]]) # controller
    Lo = P * K # loop transfer function, broken at plant output
    Li = K * P # loop transfer function, broken at plant input

    if slycot_check():
        # Balanced (S - T) disk-based stability margins at plant output
        DMo, DGMo, DPMo = disk_margins(Lo, omega, skew=0.0, returnall=True)
        assert_allclose([omega[np.argmin(DMo)]], [omega[0]],\
            atol=0.01) # sensitivity peak at 0 rad/s (or smallest provided)
        assert_allclose([min(DMo)], [0.3754], atol=0.1) # disk margin of 0.3754
        assert_allclose([DGMo[np.argmin(DMo)]], [3.3],\
            atol=0.1) # disk-based gain margin of 3.3 dB
        assert_allclose([DPMo[np.argmin(DMo)]], [21.26],\
            atol=0.1) # disk-based phase margin of 21.26 deg

        # Balanced (S - T) disk-based stability margins at plant input
        DMi, DGMi, DPMi = disk_margins(Li, omega, skew=0.0, returnall=True)
        assert_allclose([omega[np.argmin(DMi)]], [omega[0]],\
            atol=0.01) # sensitivity peak at 0 rad/s (or smallest provided)
        assert_allclose([min(DMi)], [0.3754],\
            atol=0.1) # disk margin of 0.3754
        assert_allclose([DGMi[np.argmin(DMi)]], [3.3],\
            atol=0.1) # disk-based gain margin of 3.3 dB
        assert_allclose([DPMi[np.argmin(DMi)]], [21.26],\
            atol=0.1) # disk-based phase margin of 21.26 deg
    else:
        # Slycot not installed.  Should throw exception.
        with pytest.raises(ControlMIMONotImplemented,\
            match="Need slycot to compute MIMO disk_margins"):
            DMo, DGMo, DPMo = disk_margins(Lo, omega, skew=0.0, returnall=True)
