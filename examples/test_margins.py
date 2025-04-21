"""test_margins.py
Demonstrate disk-based stability margin calculations.
"""

import os, sys, math
import numpy as np
import matplotlib.pyplot as plt
import control

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn

import numpy as np
import scipy as sp

def test_siso1():
    #
    # Disk-based stability margins for example
    # SISO loop transfer function(s)
    #

    # Frequencies of interest
    omega = np.logspace(-1, 2, 1001)

    # Laplace variable
    s = control.tf('s')

    # Loop transfer gain
    L = control.tf(25, [1, 10, 10, 10])

    print(f"------------- Python control built-in (S) -------------")
    GM_, PM_, SM_ = control.stability_margins(L)[:3] # python-control default (S-based...?)
    print(f"SM_ = {SM_}")
    print(f"GM_ = {GM_} dB")
    print(f"PM_ = {PM_} deg\n")

    print(f"------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 1.0, returnall = True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3,3,1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3,3,4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3,3,7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = -1.0, returnall = True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3,3,2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3,3,5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3,3,8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 0.0, returnall = True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3,3,3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3,3,6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3,3,9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    # Disk margin plot of admissible gain/phase variations for which
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew = -2.0)[0])
    DM_plot.append(control.disk_margins(L, omega, skew = 0.0)[0])
    DM_plot.append(control.disk_margins(L, omega, skew = 2.0)[0])
    plt.figure(10); plt.clf()
    control.disk_margin_plot(DM_plot, skew = [-2.0, 0.0, 2.0])

    return

def test_siso2():
    #
    # Disk-based stability margins for example
    # SISO loop transfer function(s)
    #

    # Frequencies of interest
    omega = np.logspace(-1, 2, 1001)

    # Laplace variable
    s = control.tf('s')

    # Loop transfer gain
    L = (6.25*(s + 3)*(s + 5))/(s*(s + 1)**2*(s**2 + 0.18*s + 100))

    print(f"------------- Python control built-in (S) -------------")
    GM_, PM_, SM_ = control.stability_margins(L)[:3] # python-control default (S-based...?)
    print(f"SM_ = {SM_}")
    print(f"GM_ = {GM_} dB")
    print(f"PM_ = {PM_} deg\n")

    print(f"------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 1.0, returnall = True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3,3,1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3,3,4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3,3,7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = -1.0, returnall = True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3,3,2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3,3,5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3,3,8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 0.0, returnall = True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3,3,3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3,3,6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3,3,9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    # Disk margin plot of admissible gain/phase variations for which
    # the feedback loop still remains stable, for each skew parameter
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew = -1.0)[0]) # T-based (T)
    DM_plot.append(control.disk_margins(L, omega, skew = 0.0)[0]) # balanced (S - T)
    DM_plot.append(control.disk_margins(L, omega, skew = 1.0)[0]) # S-based (S)
    plt.figure(20)
    control.disk_margin_plot(DM_plot, skew = [-1.0, 0.0, 1.0])

    return

def test_mimo():
    #
    # Disk-based stability margins for example
    # MIMO loop transfer function(s)
    #

    # Frequencies of interest
    omega = np.logspace(-1, 3, 1001)

    # Laplace variable
    s = control.tf('s')

    # Loop transfer gain
    P = control.ss([[0, 10],[-10, 0]], np.eye(2), [[1, 10], [-10, 1]], [[0, 0],[0, 0]]) # plant
    K = control.ss([],[],[], [[1, -2], [0, 1]]) # controller
    L = P*K # loop gain

    print(f"------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 1.0, returnall = True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3,3,1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3,3,4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3,3,7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = -1.0, returnall = True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3,3,2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3,3,5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3,3,8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    print(f"------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew = 0.0, returnall = True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3,3,3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.legend()
    plt.title('Disk Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3,3,6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3,3,9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])

    # Disk margin plot of admissible gain/phase variations for which
    # the feedback loop still remains stable, for each skew parameter
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew = -1.0)[0]) # T-based (T)
    DM_plot.append(control.disk_margins(L, omega, skew = 0.0)[0]) # balanced (S - T)
    DM_plot.append(control.disk_margins(L, omega, skew = 1.0)[0]) # S-based (S)
    plt.figure(30)
    control.disk_margin_plot(DM_plot, skew = [-1.0, 0.0, 1.0])

    return

if __name__ == '__main__':
    test_siso1()
    test_siso2()
    test_mimo()

    plt.show()
    plt.tight_layout()




