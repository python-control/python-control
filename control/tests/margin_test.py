#!/usr/bin/env pytest
"""
margin_test.py - test suite for stability margin commands

RMM, 15 Jul 2011
BG, 30 Juin 2020 -- convert to pytest, gh-425
"""
from __future__ import print_function

import numpy as np
from numpy import inf, nan
from numpy.testing import assert_allclose
import pytest

from control.frdata import FrequencyResponseData
from control.margins import margin, phase_crossover_frequencies, \
                            stability_margins
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.exception import ControlMIMONotImplemented


s = TransferFunction([1, 0], [1])

# (system, stability_margins(sys), stability_margins(sys, returnall=True))
tsys = [(TransferFunction([1, 2], [1, 2, 3]),
         (inf, inf, inf, nan, nan, nan),
         ([], [], [], [], [], [])),
        (TransferFunction([1], [1, 2, 3, 4]),
         (2., inf, 0.4170, 1.7321, nan, 1.6620),
         ([2.],     [], [1.2500, 0.4170], [1.7321], [], [0.1690, 1.6620])),
        (StateSpace([[1., 4.], [3., 2.]], [[1.], [-4.]],
                    [[1., 0.]],           [[0.]]),
         (inf, 147.0743, inf, nan, 2.5483,   nan),
         ([], [147.0743], [], [], [2.5483],   [])),
        ((8.75*(4*s**2+0.4*s+1)) / ((100*s+1)*(s**2+0.22*s+1))
         / (s**2/(10.**2)+2*0.04*s/10.+1),
         (2.2716,  97.5941, 0.5591, 10.0053, 0.0850, 9.9918),
         ([2.2716],  [97.5941, -157.7844, 134.7359], [1.0381, 0.5591],
          [10.0053], [0.0850,     0.9373,   1.0919], [0.4064, 9.9918])),
        (1/(1+s),  # no gain/phase crossovers
         (inf, inf, inf, nan, nan, nan),
         ([], [], [], [], [], [])),
        (3*(10+s)/(2+s),  # no gain/phase crossovers
         (inf, inf, inf, nan, nan, nan),
         ([], [], [], [], [], [])),
        (0.01*(10-s)/(2+s)/(1+s),  # no phase crossovers
         (300.0, inf, 0.9917, 5.6569, nan, 2.3171),
         ([300.0], [], [0.9917], [5.6569], [], 2.3171))]


def compare_allmargins(actual, desired, **kwargs):
    """Compare all elements of stability_margins(returnall=True) result"""
    assert len(actual) == len(desired)
    for a, d in zip(actual, desired):
        assert_allclose(a, d, **kwargs)


@pytest.mark.parametrize("sys, refout, refoutall", tsys)
def test_stability_margins(sys, refout, refoutall):
    """Test stability_margins() function"""
    out = stability_margins(sys)
    assert_allclose(out, refout, atol=1.5e-2)
    out = stability_margins(sys, returnall=True)
    compare_allmargins(out, refoutall, atol=1.5e-2)


@pytest.mark.parametrize("sys, refout, refoutall", tsys)
def test_stability_margins_omega(sys, refout, refoutall):
    """Test stability_margins() with interpolated frequencies"""
    omega = np.logspace(-2, 2, 2000)
    out = stability_margins(FrequencyResponseData(sys, omega))
    assert_allclose(out, refout, atol=1.5e-3)


@pytest.mark.parametrize("sys, refout, refoutall", tsys)
def test_stability_margins_3input(sys, refout, refoutall):
    """Test stability_margins() function with mag, phase, omega input"""
    omega = np.logspace(-2, 2, 2000)
    mag, phase, omega_ = sys.freqresp(omega)
    out = stability_margins((mag, phase*180/np.pi, omega_))
    assert_allclose(out, refout, atol=1.5e-3)


@pytest.mark.parametrize("sys, refout, refoutall", tsys)
def test_margin_sys(sys, refout, refoutall):
    """Test margin() function with system input"""
    out = margin(sys)
    assert_allclose(out, np.array(refout)[[0, 1, 3, 4]], atol=1.5e-3)


@pytest.mark.parametrize("sys, refout, refoutall", tsys)
def test_margin_3input(sys, refout, refoutall):
    """Test margin() function with mag, phase, omega input"""
    omega = np.logspace(-2, 2, 2000)
    mag, phase, omega_ = sys.freqresp(omega)
    out = margin((mag, phase*180/np.pi, omega_))
    assert_allclose(out, np.array(refout)[[0, 1, 3, 4]], atol=1.5e-3)


def test_phase_crossover_frequencies():
    """Test phase_crossover_frequencies() function"""
    omega, gain = phase_crossover_frequencies(tsys[1][0])
    assert_allclose(omega, [1.73205,  0.], atol=1.5e-3)
    assert_allclose(gain, [-0.5,  0.25], atol=1.5e-3)

    tf = TransferFunction([1], [1, 1])
    omega, gain = phase_crossover_frequencies(tf)
    assert_allclose(omega, [0.], atol=1.5e-3)
    assert_allclose(gain, [1.], atol=1.5e-3)

    # MIMO
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
    mag, phase, omega = sys.freqresp(omega)
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


"""
NOTE:
Matlab gives gain margin 0 for system `type2`, python-control gives inf
Difficult to argue which is right? Special case or different approach?

Edge cases, like `type0` which approaches a gain of 1 for w -> 0, are also not
identically indicated, Matlab gives phase margin -180, at w = 0. For higher or
lower gains, results match.
"""
tzmore_sys = {
    'typem1': s/(s+1),
    'type0': 1/(s+1)**3,
    'type1': (s + 0.1)/s/(s+1),
    'type2': (s + 0.1)/s**2/(s+1),
    'type3': (s + 0.1)*(s+0.1)/s**3/(s+1)}
tzmore_margin = [
    dict(sys='typem1', K=2.0, atol=1.5e-3, result=(
        float('Inf'), -120.0007, float('NaN'), 0.5774)),
    dict(sys='type0', K=0.8, atol=1.5e-3, result=(
        10.0014, float('inf'), 1.7322, float('nan'))),
    dict(sys='type0', K=2.0, atol=1e-2, result=(
        4.000,  67.6058,  1.7322,   0.7663)),
    dict(sys='type1', K=1.0, atol=1e-4, result=(
        float('Inf'), 144.9032, float('NaN'), 0.3162)),
    dict(sys='type2', K=1.0, atol=1e-4, result=(
        float('Inf'), 44.4594, float('NaN'), 0.7907)),
    dict(sys='type3', K=1.0, atol=1.5e-3, result=(
        0.0626, 37.1748, 0.1119, 0.7951)),
    ]
tzmore_stability_margins = []

"""
from "A note on the Gain and Phase Margin Concepts
Journal of Control and Systems Engineering, Yazdan Bavafi-Toosi,
Dec 2015, vol 3 iss 1, pp 51-59

A cornucopia of tricky systems for phase / gain margin
TODO: still have to convert more to tests + fix margin to handle
also these torture cases
"""
yazdan = {
    'example21':
    0.002*(s+0.02)*(s+0.05)*(s+5)*(s+10)/(
        (s-0.0005)*(s+0.0001)*(s+0.01)*(s+0.2)*(s+1)*(s+100)**2),
    'example23':
    ((s+0.1)**2 + 1)*(s-0.1)/(
        ((s+0.1)**2+4)*(s+1)),
    'example25a':
    s/(s**2+2*s+2)**4,
    'example26a':
    ((s-0.1)**2 + 1)/(
        (s + 0.1)*((s-0.2)**2 + 4)),
    'example26b': ((s-0.1)**2 + 1)/(
        (s - 0.3)*((s-0.2)**2 + 4))
}
yazdan['example24'] = yazdan['example21']*20000
yazdan['example25b'] = yazdan['example25a']*100
yazdan['example22'] = yazdan['example21']*(s**2 - 2*s + 401)
ymargin = [
    dict(sys='example21', K=1.0, atol=1e-2,
         result=(0.0100, -14.5640,  0, 0.0022)),
    dict(sys='example21', K=1000.0, atol=1e-2,
         result=(0.1793, 22.5215, 0.0243, 0.0630)),
    dict(sys='example21', K=5000.0, atol=1.5e-3,
         result=(4.5596, 21.2101, 0.4385, 0.1868)),
    ]
ystability_margins = [
    dict(sys='example21', K=1.0, rtol=1e-3, atol=1e-3,
         result=([0.01, 179.2931, 2.2798e+4, 1.5946e+07, 7.2477e+08],
                 [-14.5640],
                 [0.2496],
                 [0, 0.0243, 0.4385, 6.8640, 84.9323],
                 [0.0022],
                 [0.0022])),
    ]

tzmore_sys.update(yazdan)
tzmore_margin += ymargin
tzmore_stability_margins += ystability_margins


@pytest.mark.parametrize('tmargin', tzmore_margin)
def test_zmore_margin(tmargin):
    """Test margins for more tricky systems"""
    res = margin(tzmore_sys[tmargin['sys']]*tmargin['K'])
    assert_allclose(res, tmargin['result'], atol=tmargin['atol'])


@pytest.mark.parametrize('tmarginall', tzmore_stability_margins)
def test_zmore_stability_margins(tmarginall):
    """Test stability_margins for more tricky systems with returnall"""
    res = stability_margins(tzmore_sys[tmarginall['sys']]*tmarginall['K'],
                            returnall=True)
    compare_allmargins(res, tmarginall['result'],
                       atol=tmarginall['atol'],
                       rtol=tmarginall['rtol'])
