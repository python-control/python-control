"""nyquist_test.py - test Nyquist plots

RMM, 30 Jan 2021

This set of unit tests covers various Nyquist plot configurations.  Because
much of the output from these tests are graphical, this file can also be run
from ipython to generate plots interactively.

"""

import pytest
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import control as ct

# In interactive mode, turn on ipython interactive graphics
plt.ion()


# Some FBS examples, for comparison
def test_nyquist_fbs_examples():
    s = ct.tf('s')
    
    """Run through various examples from FBS2e to compare plots"""
    plt.figure()
    plt.title("Figure 10.4: L(s) = 1.4 e^{-s}/(s+1)^2")
    sys = ct.tf([1.4], [1, 2, 1]) * ct.tf(*ct.pade(1, 4))
    ct.nyquist_plot(sys)

    plt.figure()
    plt.title("Figure 10.4: L(s) = 1/(s + a)^2 with a = 0.6")
    sys = 1/(s + 0.6)**3
    ct.nyquist_plot(sys)

    plt.figure()
    plt.title("Figure 10.6: L(s) = 1/(s (s+1)^2) - pole at the origin")
    sys = 1/(s * (s+1)**2)
    ct.nyquist_plot(sys)

    plt.figure()
    plt.title("Figure 10.10: L(s) = 3 (s+6)^2 / (s (s+1)^2)")
    sys = 3 * (s+6)**2 / (s * (s+1)**2)
    ct.nyquist_plot(sys)

    plt.figure()
    plt.title("Figure 10.10: L(s) = 3 (s+6)^2 / (s (s+1)^2) [zoom]")
    ct.nyquist_plot(sys, omega_limits=[1.5, 1e3])


@pytest.mark.parametrize("arrows", [
    None,                       # default argument
    1, 2, 3, 4,                 # specified number of arrows
    [0.1, 0.5, 0.9],            # specify arc lengths
])
def test_nyquist_arrows(arrows):
    sys = ct.tf([1.4], [1, 2, 1]) * ct.tf(*ct.pade(1, 4))
    plt.figure();
    plt.title("L(s) = 1.4 e^{-s}/(s+1)^2 / arrows = %s" % arrows)
    ct.nyquist_plot(sys, arrows=arrows)


def test_nyquist_exceptions():
    # MIMO not implemented
    sys = ct.rss(2, 2, 2)
    with pytest.raises(
            ct.exception.ControlMIMONotImplemented,
            match="only supports SISO"):
        ct.nyquist_plot(sys)


#
# Interactive mode: generate plots for manual viewing
#

print("Nyquist examples from FBS")
test_nyquist_fbs_examples()

print("Arrow test")
test_nyquist_arrows(None)
test_nyquist_arrows(1)
test_nyquist_arrows(2)
test_nyquist_arrows(3)
test_nyquist_arrows(4)
test_nyquist_arrows([0.1, 0.5, 0.9])
