"""nyquist_test.py - test Nyquist plots

RMM, 30 Jan 2021

This set of unit tests covers various Nyquist plot configurations.  Because
much of the output from these tests are graphical, this file can also be run
from ipython to generate plots interactively.

"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import control as ct

pytestmark = pytest.mark.usefixtures("mplcleanup")


# Utility function for counting unstable poles of open loop (P in FBS)
def _P(sys, indent='right'):
    if indent == 'right':
        return (sys.pole().real > 0).sum()
    elif indent == 'left':
        return (sys.pole().real >= 0).sum()
    elif indent == 'none':
        if any(sys.pole().real == 0):
            raise ValueError("indent must be left or right for imaginary pole")
    else:
        raise TypeError("unknown indent value")


# Utility function for counting unstable poles of closed loop (Z in FBS)
def _Z(sys):
    return (sys.feedback().pole().real >= 0).sum()


# Basic tests
def test_nyquist_basic():
    # Simple Nyquist plot
    sys = ct.rss(5, 1, 1)
    N_sys = ct.nyquist_plot(sys)
    assert _Z(sys) == N_sys + _P(sys)

    # Unstable system
    sys = ct.tf([10], [1, 2, 2, 1])
    N_sys = ct.nyquist_plot(sys)
    assert _Z(sys) > 0
    assert _Z(sys) == N_sys + _P(sys)

    # Multiple systems - return value is final system
    sys1 = ct.rss(3, 1, 1)
    sys2 = ct.rss(4, 1, 1)
    sys3 = ct.rss(5, 1, 1)
    counts = ct.nyquist_plot([sys1, sys2, sys3])
    for N_sys, sys in zip(counts, [sys1, sys2, sys3]):
        assert _Z(sys) == N_sys + _P(sys)

    # Nyquist plot with poles at the origin, omega specified
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0])
    omega = np.linspace(0, 1e2, 100)
    count, contour = ct.nyquist_plot(sys, omega, return_contour=True)
    np.testing.assert_array_equal(
        contour[contour.real < 0], omega[contour.real < 0])

    # Make sure things match at unmodified frequencies
    np.testing.assert_almost_equal(
        contour[contour.real == 0],
        1j*np.linspace(0, 1e2, 100)[contour.real == 0])

    # Make sure that we can turn off frequency modification
    count, contour_indented = ct.nyquist_plot(
        sys, np.linspace(1e-4, 1e2, 100), return_contour=True)
    assert not all(contour_indented.real == 0)
    count, contour = ct.nyquist_plot(
        sys, np.linspace(1e-4, 1e2, 100), return_contour=True,
        indent_direction='none')
    np.testing.assert_almost_equal(contour, 1j*np.linspace(1e-4, 1e2, 100))

    # Nyquist plot with poles at the origin, omega unspecified
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0])
    count, contour = ct.nyquist_plot(sys, return_contour=True)
    assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles at the origin, return contour
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0])
    count, contour = ct.nyquist_plot(sys, return_contour=True)
    assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles on imaginary axis, omega specified
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    count = ct.nyquist_plot(sys, np.linspace(1e-3, 1e1, 1000))
    assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles on imaginary axis, omega specified, with contour
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    count, contour = ct.nyquist_plot(
        sys, np.linspace(1e-3, 1e1, 1000), return_contour=True)
    assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles on imaginary axis, return contour
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    count, contour = ct.nyquist_plot(sys, return_contour=True)
    assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles at the origin and on imaginary axis
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1]) * ct.tf([1], [1, 0])
    count, contour = ct.nyquist_plot(sys, return_contour=True)
    assert _Z(sys) == count + _P(sys)


# Some FBS examples, for comparison
def test_nyquist_fbs_examples():
    s = ct.tf('s')

    """Run through various examples from FBS2e to compare plots"""
    plt.figure()
    plt.title("Figure 10.4: L(s) = 1.4 e^{-s}/(s+1)^2")
    sys = ct.tf([1.4], [1, 2, 1]) * ct.tf(*ct.pade(1, 4))
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)

    plt.figure()
    plt.title("Figure 10.4: L(s) = 1/(s + a)^2 with a = 0.6")
    sys = 1/(s + 0.6)**3
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)

    plt.figure()
    plt.title("Figure 10.6: L(s) = 1/(s (s+1)^2) - pole at the origin")
    sys = 1/(s * (s+1)**2)
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)

    plt.figure()
    plt.title("Figure 10.10: L(s) = 3 (s+6)^2 / (s (s+1)^2)")
    sys = 3 * (s+6)**2 / (s * (s+1)**2)
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)

    plt.figure()
    plt.title("Figure 10.10: L(s) = 3 (s+6)^2 / (s (s+1)^2) [zoom]")
    count = ct.nyquist_plot(sys, omega_limits=[1.5, 1e3])
    # Frequency limits for zoom give incorrect encirclement count
    # assert _Z(sys) == count + _P(sys)
    assert count == -1


@pytest.mark.parametrize("arrows", [
    None,                       # default argument
    1, 2, 3, 4,                 # specified number of arrows
    [0.1, 0.5, 0.9],            # specify arc lengths
])
def test_nyquist_arrows(arrows):
    sys = ct.tf([1.4], [1, 2, 1]) * ct.tf(*ct.pade(1, 4))
    plt.figure();
    plt.title("L(s) = 1.4 e^{-s}/(s+1)^2 / arrows = %s" % arrows)
    count = ct.nyquist_plot(sys, arrows=arrows)
    assert _Z(sys) == count + _P(sys)


def test_nyquist_encirclements():
    # Example 14.14: effect of friction in a cart-pendulum system
    s = ct.tf('s')
    sys = (0.02 * s**3 - 0.1 * s) / (s**4 + s**3 + s**2 + 0.25 * s + 0.04)

    plt.figure();
    count = ct.nyquist_plot(sys)
    plt.title("Stable system; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys)

    plt.figure();
    count = ct.nyquist_plot(sys * 3)
    plt.title("Unstable system; encirclements = %d" % count)
    assert _Z(sys * 3) == count + _P(sys * 3)

    # System with pole at the origin
    sys = ct.tf([3], [1, 2, 2, 1, 0])

    plt.figure();
    count = ct.nyquist_plot(sys)
    plt.title("Pole at the origin; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys)


def test_nyquist_indent():
    # FBS Figure 10.10
    s = ct.tf('s')
    sys = 3 * (s+6)**2 / (s * (s+1)**2)
    # poles: [-1, -1, 0]

    plt.figure();
    count = ct.nyquist_plot(sys)
    plt.title("Pole at origin; indent_radius=default")
    assert _Z(sys) == count + _P(sys)

    # first value of default omega vector was 0.1, replaced by 0. for contour
    # indent_radius is larger than 0.1 -> no extra quater circle around origin
    count, contour = ct.nyquist_plot(sys, plot=False, indent_radius=.1007,
                                     return_contour=True)
    np.testing.assert_allclose(contour[0], .1007+0.j)
    # second value of omega_vector is larger than indent_radius: not indented
    assert np.all(contour.real[2:] == 0.)

    plt.figure();
    count, contour = ct.nyquist_plot(sys, indent_radius=0.01,
                                     return_contour=True)
    plt.title("Pole at origin; indent_radius=0.01; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys)
    # indent radius is smaller than the start of the default omega vector
    # check that a quarter circle around the pole at origin has been added.
    np.testing.assert_allclose(contour[:50].real**2 + contour[:50].imag**2,
                               0.01**2)

    plt.figure();
    count = ct.nyquist_plot(sys, indent_direction='left')
    plt.title(
        "Pole at origin; indent_direction='left'; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys, indent='left')

    # System with poles on the imaginary axis
    sys = ct.tf([1, 1], [1, 0, 1])

    # Imaginary poles with standard indentation
    plt.figure();
    count = ct.nyquist_plot(sys)
    plt.title("Imaginary poles; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys)

    # Imaginary poles with indentation to the left
    plt.figure();
    count = ct.nyquist_plot(sys, indent_direction='left', label_freq=300)
    plt.title(
        "Imaginary poles; indent_direction='left'; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys, indent='left')

    # Imaginary poles with no indentation
    plt.figure();
    count = ct.nyquist_plot(
        sys, np.linspace(0, 1e3, 1000), indent_direction='none')
    plt.title(
        "Imaginary poles; indent_direction='none'; encirclements = %d" % count)
    assert _Z(sys) == count + _P(sys)


def test_nyquist_exceptions():
    # MIMO not implemented
    sys = ct.rss(2, 2, 2)
    with pytest.raises(
            ct.exception.ControlMIMONotImplemented,
            match="only supports SISO"):
        ct.nyquist_plot(sys)

    # Legacy keywords for arrow size
    sys = ct.rss(2, 1, 1)
    with pytest.warns(FutureWarning, match="use `arrow_size` instead"):
        ct.nyquist_plot(sys, arrow_width=8, arrow_length=6)

    # Discrete time system sampled above Nyquist frequency
    sys = ct.drss(2, 1, 1)
    sys.dt = 0.01
    with pytest.warns(UserWarning, match="above Nyquist"):
        ct.nyquist_plot(sys, np.logspace(-2, 3))


if __name__ == "__main__":
    #
    # Interactive mode: generate plots for manual viewing
    #
    # Running this script in python (or better ipython) will show a collection of
    # figures that should all look OK on the screeen.
    #

    # In interactive mode, turn on ipython interactive graphics
    plt.ion()

    # Start by clearing existing figures
    plt.close('all')

    print("Nyquist examples from FBS")
    test_nyquist_fbs_examples()

    print("Arrow test")
    test_nyquist_arrows(None)
    test_nyquist_arrows(1)
    test_nyquist_arrows(3)
    test_nyquist_arrows([0.1, 0.5, 0.9])

    print("Stability checks")
    test_nyquist_encirclements()

    print("Indentation checks")
    test_nyquist_indent()

    print("Unusual Nyquist plot")
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    plt.figure()
    plt.title("Poles: %s" % np.array2string(sys.pole(), precision=2, separator=','))
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)
