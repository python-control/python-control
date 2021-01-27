"""robust_mimo.py

Demonstrate mixed-sensitivity H-infinity design for a MIMO plant.

Based on Example 3.8 from Multivariable Feedback Control, Skogestad and Postlethwaite, 1st Edition.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from control import tf, ss, mixsyn, step_response


def weighting(wb, m, a):
    """weighting(wb,m,a) -> wf
    wb - design frequency (where |wf| is approximately 1)
    m - high frequency gain of 1/wf; should be > 1
    a - low frequency gain of 1/wf; should be < 1
    wf - SISO LTI object
    """
    s = tf([1, 0], [1])
    return (s/m + wb) / (s + wb*a)


def plant():
    """plant() -> g
    g - LTI object; 2x2 plant with a RHP zero, at s=0.5.
    """
    den = [0.2, 1.2, 1]
    gtf = tf([[[1], [1]],
              [[2, 1], [2]]],
             [[den, den],
              [den, den]])
    return ss(gtf)


# as of this writing (2017-07-01), python-control doesn't have an
# equivalent to Matlab's sigma function, so use a trivial stand-in.
def triv_sigma(g, w):
    """triv_sigma(g,w) -> s
    g - LTI object, order n
    w - frequencies, length m
    s - (m,n) array of singular values of g(1j*w)"""
    m, p, _ = g.frequency_response(w)
    sjw = (m*np.exp(1j*p)).transpose(2, 0, 1)
    sv = np.linalg.svd(sjw, compute_uv=False)
    return sv


def analysis():
    """Plot open-loop responses for various inputs"""
    g = plant()

    t = np.linspace(0, 10, 101)
    _, yu1 = step_response(g, t, input=0, squeeze=True)
    _, yu2 = step_response(g, t, input=1, squeeze=True)

    # linear system, so scale and sum previous results to get the
    # [1,-1] response
    yuz = yu1 - yu2

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.plot(t, yu1[0], label='$y_1$')
    plt.plot(t, yu1[1], label='$y_2$')
    plt.xlabel('time')
    plt.ylabel('output')
    plt.ylim([-1.1, 2.1])
    plt.legend()
    plt.title('o/l response\nto input [1,0]')

    plt.subplot(1, 3, 2)
    plt.plot(t, yu2[0], label='$y_1$')
    plt.plot(t, yu2[1], label='$y_2$')
    plt.xlabel('time')
    plt.ylabel('output')
    plt.ylim([-1.1, 2.1])
    plt.legend()
    plt.title('o/l response\nto input [0,1]')

    plt.subplot(1, 3, 3)
    plt.plot(t, yuz[0], label='$y_1$')
    plt.plot(t, yuz[1], label='$y_2$')
    plt.xlabel('time')
    plt.ylabel('output')
    plt.ylim([-1.1, 2.1])
    plt.legend()
    plt.title('o/l response\nto input [1,-1]')


def synth(wb1, wb2):
    """synth(wb1,wb2) -> k,gamma
    wb1: S weighting frequency
    wb2: KS weighting frequency
    k: controller
    gamma: H-infinity norm of 'design', that is, of evaluation system
    with loop closed through design
    """
    g = plant()
    wu = ss([], [], [], np.eye(2))
    wp1 = ss(weighting(wb=wb1, m=1.5, a=1e-4))
    wp2 = ss(weighting(wb=wb2, m=1.5, a=1e-4))
    wp = wp1.append(wp2)
    k, _, info = mixsyn(g, wp, wu)
    return k, info[0]


def step_opposite(g, t):
    """reponse to step of [-1,1]"""
    _, yu1 = step_response(g, t, input=0, squeeze=True)
    _, yu2 = step_response(g, t, input=1, squeeze=True)
    return yu1 - yu2


def design():
    """Show results of designs"""
    # equal weighting on each output
    k1, gam1 = synth(0.25, 0.25)
    # increase "bandwidth" of output 2 by moving crossover weighting frequency 100 times higher
    k2, gam2 = synth(0.25, 25)
    # now weight output 1 more heavily
    # won't plot this one, just want gamma
    _, gam3 = synth(25, 0.25)

    print('design 1 gamma {:.3g} (Skogestad: 2.80)'.format(gam1))
    print('design 2 gamma {:.3g} (Skogestad: 2.92)'.format(gam2))
    print('design 3 gamma {:.3g} (Skogestad: 6.73)'.format(gam3))

    # do the designs
    g = plant()
    w = np.logspace(-2, 2, 101)
    I = ss([], [], [], np.eye(2))
    s1 = I.feedback(g*k1)
    s2 = I.feedback(g*k2)

    # frequency response
    sv1 = triv_sigma(s1, w)
    sv2 = triv_sigma(s2, w)

    plt.figure(2)

    plt.subplot(1, 2, 1)
    plt.semilogx(w, 20*np.log10(sv1[:, 0]), label=r'$\sigma_1(S_1)$')
    plt.semilogx(w, 20*np.log10(sv1[:, 1]), label=r'$\sigma_2(S_1)$')
    plt.semilogx(w, 20*np.log10(sv2[:, 0]), label=r'$\sigma_1(S_2)$')
    plt.semilogx(w, 20*np.log10(sv2[:, 1]), label=r'$\sigma_2(S_2)$')
    plt.ylim([-60, 10])
    plt.ylabel('magnitude [dB]')
    plt.xlim([1e-2, 1e2])
    plt.xlabel('freq [rad/s]')
    plt.legend()
    plt.title('Singular values of S')

    # time response

    # in design 1, both outputs have an inverse initial response; in
    # design 2, output 2 does not, and is very fast, while output 1
    # has a larger initial inverse response than in design 1
    time = np.linspace(0, 10, 301)
    t1 = (g*k1).feedback(I)
    t2 = (g*k2).feedback(I)

    y1 = step_opposite(t1, time)
    y2 = step_opposite(t2, time)

    plt.subplot(1, 2, 2)
    plt.plot(time, y1[0], label='des. 1 $y_1(t))$')
    plt.plot(time, y1[1], label='des. 1 $y_2(t))$')
    plt.plot(time, y2[0], label='des. 2 $y_1(t))$')
    plt.plot(time, y2[1], label='des. 2 $y_2(t))$')
    plt.xlabel('time [s]')
    plt.ylabel('response [1]')
    plt.legend()
    plt.title('c/l response to reference [1,-1]')


if __name__ == "__main__":
    analysis()
    design()
    if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
        plt.show()
