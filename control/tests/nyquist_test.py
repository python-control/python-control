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
        return (sys.poles().real > 0).sum()
    elif indent == 'left':
        return (sys.poles().real >= 0).sum()
    elif indent == 'none':
        if any(sys.poles().real == 0):
            raise ValueError("indent must be left or right for imaginary pole")
    else:
        raise TypeError("unknown indent value")


# Utility function for counting unstable poles of closed loop (Z in FBS)
def _Z(sys):
    return (sys.feedback().poles().real >= 0).sum()


# Basic tests
def test_nyquist_basic():
    # Simple Nyquist plot
    sys = ct.rss(5, 1, 1)
    N_sys = ct.nyquist_plot(sys)
    assert _Z(sys) == N_sys + _P(sys)

    # Previously identified bug
    #
    # This example has an open loop pole at -0.06 and a closed loop pole at
    # 0.06, so if you use an indent_radius of larger than 0.12, then the
    # encirclements computed by nyquist_plot() will not properly predict
    # stability.  A new warning messages was added to catch this case.
    #
    A = np.array([
        [-3.56355873, -1.22980795, -1.5626527 , -0.4626829 , -0.16741484],
        [-8.52361371, -3.60331459, -3.71574266, -0.43839201,  0.41893656],
        [-2.50458726, -0.72361335, -1.77795489, -0.4038419 ,  0.52451147],
        [-0.281183  ,  0.23391825,  0.19096003, -0.9771515 ,  0.66975606],
        [-3.04982852, -1.1091943 , -1.40027242, -0.1974623 , -0.78930791]])
    B = np.array([[-0.], [-1.42827213], [ 0.76806551], [-1.07987454], [0.]])
    C = np.array([[-0.,  0.35557249,  0.35941791, -0., -1.42320969]])
    D = np.array([[0]])
    sys = ct.ss(A, B, C, D)

    # With a small indent_radius, all should be fine
    N_sys = ct.nyquist_plot(sys, indent_radius=0.001)
    assert _Z(sys) == N_sys + _P(sys)

    # With a larger indent_radius, we get a warning message + wrong answer
    with pytest.warns(UserWarning, match="contour may miss closed loop pole"):
        N_sys = ct.nyquist_plot(sys, indent_radius=0.2)
        assert _Z(sys) != N_sys + _P(sys)

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

    #
    # Make sure that we can turn off frequency modification
    #
    # Start with a case where indentation should occur
    count, contour_indented = ct.nyquist_plot(
        sys, np.linspace(1e-4, 1e2, 100), indent_radius=1e-2,
        return_contour=True)
    assert not all(contour_indented.real == 0)
    with pytest.warns(UserWarning, match="encirclements does not match"):
        count, contour = ct.nyquist_plot(
            sys, np.linspace(1e-4, 1e2, 100), indent_radius=1e-2,
            return_contour=True, indent_direction='none')
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
    # (can miss encirclements due to the imaginary poles at +/- 1j)
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    with pytest.warns(UserWarning, match="does not match") as records:
        count = ct.nyquist_plot(sys, np.linspace(1e-3, 1e1, 1000))
    if len(records) == 0:
        assert _Z(sys) == count + _P(sys)

    # Nyquist plot with poles on imaginary axis, omega specified, with contour
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    with pytest.warns(UserWarning, match="does not match") as records:
        count, contour = ct.nyquist_plot(
            sys, np.linspace(1e-3, 1e1, 1000), return_contour=True)
    if len(records) == 0:
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
    with pytest.warns(UserWarning, match="encirclements does not match"):
        count = ct.nyquist_plot(sys, omega_limits=[1.5, 1e3])
        # Frequency limits for zoom give incorrect encirclement count
        # assert _Z(sys) == count + _P(sys)
        assert count == -1


@pytest.mark.parametrize("arrows", [
    None,                       # default argument
    False,                      # no arrows
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

    # Non-integer number of encirclements
    plt.figure();
    sys = 1 / (s**2 + s + 1)
    with pytest.warns(UserWarning, match="encirclements was a non-integer"):
        count = ct.nyquist_plot(sys, omega_limits=[0.5, 1e3])
    with pytest.warns(None) as records:
        count = ct.nyquist_plot(
            sys, omega_limits=[0.5, 1e3], encirclement_threshold=0.2)
    assert len(records) == 0
    plt.title("Non-integer number of encirclements [%g]" % count)


@pytest.fixture
def indentsys():
    # FBS Figure 10.10
    # poles: [-1, -1, 0]
    s = ct.tf('s')
    return 3 * (s+6)**2 / (s * (s+1)**2)


def test_nyquist_indent_default(indentsys):
    plt.figure();
    count = ct.nyquist_plot(indentsys)
    plt.title("Pole at origin; indent_radius=default")
    assert _Z(indentsys) == count + _P(indentsys)


def test_nyquist_indent_dont(indentsys):
    # first value of default omega vector was 0.1, replaced by 0. for contour
    # indent_radius is larger than 0.1 -> no extra quater circle around origin
    with pytest.warns(UserWarning, match="encirclements does not match"):
        count, contour = ct.nyquist_plot(
            indentsys, omega=[0, 0.2, 0.3, 0.4], indent_radius=.1007,
            plot=False, return_contour=True)
    np.testing.assert_allclose(contour[0], .1007+0.j)
    # second value of omega_vector is larger than indent_radius: not indented
    assert np.all(contour.real[2:] == 0.)


def test_nyquist_indent_do(indentsys):
    plt.figure();
    count, contour = ct.nyquist_plot(
        indentsys, indent_radius=0.01, return_contour=True)
    plt.title("Pole at origin; indent_radius=0.01; encirclements = %d" % count)
    assert _Z(indentsys) == count + _P(indentsys)
    # indent radius is smaller than the start of the default omega vector
    # check that a quarter circle around the pole at origin has been added.
    np.testing.assert_allclose(contour[:50].real**2 + contour[:50].imag**2,
                               0.01**2)


def test_nyquist_indent_left(indentsys):
    plt.figure();
    count = ct.nyquist_plot(indentsys, indent_direction='left')
    plt.title(
        "Pole at origin; indent_direction='left'; encirclements = %d" % count)
    assert _Z(indentsys) == count + _P(indentsys, indent='left')


def test_nyquist_indent_im():
    """Test system with poles on the imaginary axis."""
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
    with pytest.warns(UserWarning, match="encirclements does not match"):
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

    # Unknown arrow keyword
    with pytest.raises(ValueError, match="unsupported arrow location"):
        ct.nyquist_plot(sys, arrows='uniform')

    # Bad value for indent direction
    sys = ct.tf([1], [1, 0, 1])
    with pytest.raises(ValueError, match="unknown value for indent"):
        ct.nyquist_plot(sys, indent_direction='up')

    # Discrete time system sampled above Nyquist frequency
    sys = ct.drss(2, 1, 1)
    sys.dt = 0.01
    with pytest.warns(UserWarning, match="above Nyquist"):
        ct.nyquist_plot(sys, np.logspace(-2, 3))


def test_linestyle_checks():
    sys = ct.rss(2, 1, 1)

    # Things that should work
    ct.nyquist_plot(sys, primary_style=['-', '-'], mirror_style=['-', '-'])
    ct.nyquist_plot(sys, mirror_style=None)

    with pytest.raises(ValueError, match="invalid 'primary_style'"):
        ct.nyquist_plot(sys, primary_style=False)

    with pytest.raises(ValueError, match="invalid 'mirror_style'"):
        ct.nyquist_plot(sys, mirror_style=0.2)

    # If only one line style is given use, the default value for the other
    # TODO: for now, just make sure the signature works; no correct check yet
    with pytest.warns(PendingDeprecationWarning, match="single string"):
        ct.nyquist_plot(sys, primary_style=':', mirror_style='-.')

@pytest.mark.usefixtures("editsdefaults")
def test_nyquist_legacy():
    ct.use_legacy_defaults('0.9.1')

    # Example that generated a warning using earlier defaults
    s = ct.tf('s')
    sys = (0.02 * s**3 - 0.1 * s) / (s**4 + s**3 + s**2 + 0.25 * s + 0.04)

    with pytest.warns(UserWarning, match="indented contour may miss"):
        count = ct.nyquist_plot(sys)

def test_discrete_nyquist():
    # Make sure we can handle discrete time systems with negative poles
    sys = ct.tf(1, [1, -0.1], dt=1) * ct.tf(1, [1, 0.1], dt=1)
    ct.nyquist_plot(sys)
    
if __name__ == "__main__":
    #
    # Interactive mode: generate plots for manual viewing
    #
    # Running this script in python (or better ipython) will show a
    # collection of figures that should all look OK on the screeen.
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
    s = ct.tf('s')
    indentsys = 3 * (s+6)**2 / (s * (s+1)**2)
    test_nyquist_indent_default(indentsys)
    test_nyquist_indent_do(indentsys)
    test_nyquist_indent_left(indentsys)

    # Generate a figuring showing effects of different parameters
    sys = 3 * (s+6)**2 / (s * (s**2 + 1e-4 * s + 1))
    plt.figure()
    ct.nyquist_plot(sys)
    ct.nyquist_plot(sys, max_curve_magnitude=15)
    ct.nyquist_plot(sys, indent_radius=1e-6, max_curve_magnitude=25)

    print("Unusual Nyquist plot")
    sys = ct.tf([1], [1, 3, 2]) * ct.tf([1], [1, 0, 1])
    plt.figure()
    plt.title("Poles: %s" %
              np.array2string(sys.poles(), precision=2, separator=','))
    count = ct.nyquist_plot(sys)
    assert _Z(sys) == count + _P(sys)

    print("Discrete time systems")
    sys = ct.c2d(sys, 0.01)
    plt.figure()
    plt.title("Discrete-time; poles: %s" %
              np.array2string(sys.poles(), precision=2, separator=','))
    count = ct.nyquist_plot(sys)

    

