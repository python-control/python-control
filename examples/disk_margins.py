"""disk_margins.py

Demonstrate disk-based stability margin calculations.

References:
[1] Blight, James D., R. Lane Dailey, and Dagfinn Gangsaas. “Practical
    Control Law Design for Aircraft Using Multivariable Techniques.”
    International Journal of Control 59, no. 1 (January 1994): 93-137.
    https://doi.org/10.1080/00207179408923071.

[2] Seiler, Peter, Andrew Packard, and Pascal Gahinet. “An Introduction
    to Disk Margins [Lecture Notes].” IEEE Control Systems Magazine 40,
    no. 5 (October 2020): 78-95.

[3] P. Benner, V. Mehrmann, V. Sima, S. Van Huffel, and A. Varga, "SLICOT
    - A Subroutine Library in Systems and Control Theory", Applied and
    Computational Control, Signals, and Circuits (Birkhauser), Vol. 1, Ch.
    10, pp. 505-546, 1999.

[4] S. Van Huffel, V. Sima, A. Varga, S. Hammarling, and F. Delebecque,
    "Development of High Performance Numerical Software for Control", IEEE
    Control Systems Magazine, Vol. 24, Nr. 1, Feb., pp. 60-76, 2004.

[5] Deodhare, G., & Patel, V. (1998, August). A "Modern" Look at Gain
    and Phase Margins: An H-Infinity/mu Approach. In Guidance, Navigation,
    and Control Conference and Exhibit (p. 4134).
"""

import os
import control
import matplotlib.pyplot as plt
import numpy as np

def plot_allowable_region(alpha_max, skew, ax=None):
    """Plot region of allowable gain/phase variation, given worst-case disk margin.

    Parameters
    ----------
    alpha_max : float (scalar or list)
        worst-case disk margin(s) across all frequencies. May be a scalar or list.
    skew : float (scalar or list)
        skew parameter(s) for disk margin calculation.
        skew=0 uses the "balanced" sensitivity function 0.5*(S - T)
        skew=1 uses the sensitivity function S
        skew=-1 uses the complementary sensitivity function T
    ax : axes to plot bounding curve(s) onto

    Returns
    -------
    DM : ndarray
        1D array of frequency-dependent disk margins.  DM is the same
        size as "omega" parameter.
    GM : ndarray
        1D array of frequency-dependent disk-based gain margins, in dB.
        GM is the same size as "omega" parameter.
    PM : ndarray
        1D array of frequency-dependent disk-based phase margins, in deg.
        PM is the same size as "omega" parameter.
    """

    # Create axis if needed
    if ax is None:
        ax = plt.gca()

    # Allow scalar or vector arguments (to overlay plots)
    if np.isscalar(alpha_max):
        alpha_max = np.asarray([alpha_max])
    else:
        alpha_max = np.asarray(alpha_max)

    if np.isscalar(skew):
        skew=np.asarray([skew])
    else:
        skew=np.asarray(skew)

    # Add a plot for each (alpha, skew) pair present
    theta = np.linspace(0, np.pi, 500)
    legend_list = []
    for ii in range(0, skew.shape[0]):
        legend_str = "$\\sigma$ = %.1f, $\\alpha_{max}$ = %.2f" %(\
            skew[ii], alpha_max[ii])
        legend_list.append(legend_str)

        # Complex bounding curve of stable gain/phase variations
        f = (2 + alpha_max[ii] * (1 - skew[ii]) * np.exp(1j * theta))\
           /(2 - alpha_max[ii] * (1 + skew[ii]) * np.exp(1j * theta))

        # Allowable combined gain/phase variations
        gamma_dB = control.ctrlutil.mag2db(np.abs(f)) # gain margin (dB)
        phi_deg = np.rad2deg(np.angle(f)) # phase margin (deg)

        # Plot the allowable combined gain/phase variations
        out = ax.plot(gamma_dB, phi_deg, alpha=0.25, label='_nolegend_')
        ax.fill_between(ax.lines[ii].get_xydata()[:,0],\
            ax.lines[ii].get_xydata()[:,1], alpha=0.25)

    plt.ylabel('Phase Variation (deg)')
    plt.xlabel('Gain Variation (dB)')
    plt.title('Range of Gain and Phase Variations')
    plt.legend(legend_list)
    plt.grid()
    plt.tight_layout()

    return out

def test_siso1():
    #
    # Disk-based stability margins for example
    # SISO loop transfer function(s)
    #

    # Frequencies of interest
    omega = np.logspace(-1, 2, 1001)

    # Loop transfer gain
    L = control.tf(25, [1, 10, 10, 10])

    print("------------- Python control built-in (S) -------------")
    GM_, PM_, SM_ = control.stability_margins(L)[:3] # python-control default (S-based...?)
    print(f"SM_ = {SM_}")
    print(f"GM_ = {GM_} dB")
    print(f"PM_ = {PM_} deg\n")

    print("------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=1.0, returnall=True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3, 3, 1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('S-Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3, 3, 4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3, 3, 7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=-1.0, returnall=True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3, 3, 2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('T_Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3, 3, 5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3, 3, 8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=0.0, returnall=True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(1)
    plt.subplot(3, 3, 3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('Balanced Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(1)
    plt.subplot(3, 3, 6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(1)
    plt.subplot(3, 3, 9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    # Disk margin plot of admissible gain/phase variations for which
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew=-2.0)[0])
    DM_plot.append(control.disk_margins(L, omega, skew=0.0)[0])
    DM_plot.append(control.disk_margins(L, omega, skew=2.0)[0])
    plt.figure(10); plt.clf()
    plot_allowable_region(DM_plot, skew=[-2.0, 0.0, 2.0])

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
    L = (6.25 * (s + 3) * (s + 5)) / (s * (s + 1)**2 * (s**2 + 0.18 * s + 100))

    print("------------- Python control built-in (S) -------------")
    GM_, PM_, SM_ = control.stability_margins(L)[:3] # python-control default (S-based...?)
    print(f"SM_ = {SM_}")
    print(f"GM_ = {GM_} dB")
    print(f"PM_ = {PM_} deg\n")

    print("------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=1.0, returnall=True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3, 3, 1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('S-Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3, 3, 4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3, 3, 7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=-1.0, returnall=True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3, 3, 2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('T-Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3, 3, 5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3, 3, 8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=0.0, returnall=True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(2)
    plt.subplot(3, 3, 3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('Balanced Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(2)
    plt.subplot(3, 3, 6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(2)
    plt.subplot(3, 3, 9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    # Disk margin plot of admissible gain/phase variations for which
    # the feedback loop still remains stable, for each skew parameter
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew=-1.0)[0]) # T-based (T)
    DM_plot.append(control.disk_margins(L, omega, skew=0.0)[0]) # balanced (S - T)
    DM_plot.append(control.disk_margins(L, omega, skew=1.0)[0]) # S-based (S)
    plt.figure(20)
    plot_allowable_region(DM_plot, skew=[-1.0, 0.0, 1.0])

    return

def test_mimo():
    #
    # Disk-based stability margins for example
    # MIMO loop transfer function(s)
    #

    # Frequencies of interest
    omega = np.logspace(-1, 3, 1001)

    # Loop transfer gain
    P = control.ss([[0, 10],[-10, 0]], np.eye(2), [[1, 10], [-10, 1]], 0) # plant
    K = control.ss([], [], [], [[1, -2], [0, 1]]) # controller
    L = P * K # loop gain

    print("------------- Sensitivity function (S) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=1.0, returnall=True) # S-based (S)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3, 3, 1)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('S-Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3, 3, 4)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3, 3, 7)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Complementary sensitivity function (T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=-1.0, returnall=True) # T-based (T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3, 3, 2)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('T-Based Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3, 3, 5)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3, 3, 8)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    print("------------- Balanced sensitivity function (S - T) -------------")
    DM, GM, PM = control.disk_margins(L, omega, skew=0.0, returnall=True) # balanced (S - T)
    print(f"min(DM) = {min(DM)} (omega = {omega[np.argmin(DM)]})")
    print(f"GM = {GM[np.argmin(DM)]} dB")
    print(f"PM = {PM[np.argmin(DM)]} deg")
    print(f"min(GM) = {min(GM)} dB")
    print(f"min(PM) = {min(PM)} deg\n")

    plt.figure(3)
    plt.subplot(3, 3, 3)
    plt.semilogx(omega, DM, label='$\\alpha$')
    plt.ylabel('Disk Margin (abs)')
    plt.legend()
    plt.title('Balanced Margins')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 2])

    plt.figure(3)
    plt.subplot(3, 3, 6)
    plt.semilogx(omega, GM, label='$\\gamma_{m}$')
    plt.ylabel('Gain Margin (dB)')
    plt.legend()
    #plt.title('Gain-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 40])

    plt.figure(3)
    plt.subplot(3, 3, 9)
    plt.semilogx(omega, PM, label='$\\phi_{m}$')
    plt.ylabel('Phase Margin (deg)')
    plt.legend()
    #plt.title('Phase-Only Margin')
    plt.grid()
    plt.xlim([omega[0], omega[-1]])
    plt.ylim([0, 90])
    plt.xlabel('Frequency (rad/s)')

    # Disk margin plot of admissible gain/phase variations for which
    # the feedback loop still remains stable, for each skew parameter
    DM_plot = []
    DM_plot.append(control.disk_margins(L, omega, skew=-1.0)[0]) # T-based (T)
    DM_plot.append(control.disk_margins(L, omega, skew=0.0)[0]) # balanced (S - T)
    DM_plot.append(control.disk_margins(L, omega, skew=1.0)[0]) # S-based (S)
    plt.figure(30)
    plot_allowable_region(DM_plot, skew=[-1.0, 0.0, 1.0])

    return

if __name__ == '__main__':
    #test_siso1()
    #test_siso2()
    test_mimo()
    if 'PYCONTROL_TEST_EXAMPLES' not in os.environ:
        #plt.tight_layout()
        plt.show()
