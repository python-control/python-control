# nltools.py - nonlinear feedback analysis
#
# RMM, 23 Jan 2021
#
# This module adds functions for carrying out analysis of systems with
# static nonlinear feedback functions using the circle criterion and
# describing functions.
#

"""The :mod:~control.nltools` module contains function for performing closed
loop analysis of systems with static nonlinearities.  It is built around the
basic structure required to apply the circle criterion and describing function
analysis.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import where, dstack, diff, meshgrid
from warnings import warn

from .freqplot import nyquist_plot

__all__ = ['describing_function', 'describing_function_plot', 'sector_bounds']

def sector_bounds(fcn):
    raise NotImplementedError("function not currently implemented")


def describing_function(fcn, amp, num_points=100, zero_check=True):
    """Numerical compute the describing function of a nonlinear function

    The describing function of a static nonlinear function is given by
    magnitude and phase of the first harmonic of the function when evaluated
    along a sinusoidal input :math:`a \\sin \\omega t`.  This function returns
    the magnitude and phase of the describing function at amplitude :math:`a`.

    Parameters
    ----------
    fcn : callable
        The function fcn() should accept a scalar number as an argument and
        return a scalar number.  For compatibility with (static) nonlinear
        input/output systems, the output can also return a 1D array with a
        single element.

    amp : float or array
        The amplitude(s) at which the describing function should be calculated.

    Returns
    -------
    df : complex or array of complex
        The (complex) value of the describing fuction at the given amplitude.

    Raises
    ------
    TypeError
        If amp < 0 or if amp = 0 and the function fcn(0) is non-zero.

    """
    #
    # The describing function of a nonlinear function F() can be computed by
    # evaluating the nonlinearity over a sinusoid.  The Fourier series for a
    # static noninear function evaluated on a sinusoid can be written as
    #
    # F(a\sin\omega t) = \sum_{k=1}^\infty M_k(a) \sin(k\omega t + \phi_k(a))
    #
    # The describing function is given by the complex number
    #
    #    N(a) = M_1(a) e^{j \phi_1(a)} / a
    #
    # To compute this, we compute F(\theta) for \theta between 0 and 2 \pi,
    # use the identities
    #
    #   \sin(\theta + \phi) = \sin\theta \cos\phi + \cos\theta \sin\phi
    #   \int_0^{2\pi} \sin^2 \theta d\theta = \pi
    #   \int_0^{2\pi} \cos^2 \theta d\theta = \pi
    #
    # and then integate the product against \sin\theta and \cos\theta to obtain
    #
    #   \int_0^{2\pi} F(a\sin\theta) \sin\theta d\theta = M_1 \pi \cos\phi
    #   \int_0^{2\pi} F(a\sin\theta) \cos\theta d\theta = M_1 \pi \sin\phi
    #
    # From these we can compute M1 and \phi.
    #
    
    # Evaluate over a full range of angles
    theta = np.linspace(0, 2*np.pi, num_points)
    dtheta = theta[1] - theta[0]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Initialize any internal state by going through an initial cycle
    [fcn(x) for x in np.atleast_1d(amp).min() * sin_theta]

    # Go through all of the amplitudes we were given
    df = []
    for a in np.atleast_1d(amp):
        # Make sure we got a valid argument
        if a == 0:
            # Check to make sure the function has zero output with zero input
            if zero_check and np.squeeze(fcn(0.)) != 0:
                raise ValueError("function must evaluate to zero at zero")
            df.append(1.)
            continue
        elif a < 0:
            raise ValueError("cannot evaluate describing function for amp < 0")

        # Save the scaling factor for to make the formulas simpler
        scale = dtheta / np.pi / a

        # Evaluate the function (twice) along a sinusoid (for internal state)
        fcn_eval = np.array([fcn(x) for x in a*sin_theta]).squeeze()

        # Compute the prjections onto sine and cosine
        df_real = (fcn_eval @ sin_theta) * scale     # = M_1 \cos\phi / a
        df_imag = (fcn_eval @ cos_theta) * scale     # = M_1 \sin\phi / a

        df.append(df_real + 1j * df_imag)

    # Return the values in the same shape as they were requested
    return np.array(df).reshape(np.shape(amp))


def describing_function_plot(H, F, a, omega=None):
    """Plot a Nyquist plot with a describing function for a nonlinear system.

    This function generates a Nyquist plot for a closed loop system consisting
    of a linear system with a static nonlinear function in the feedback path.

    Parameters
    ----------
    H : LTI system
        Linear time-invariant (LTI) system (state space, transfer function, or
        FRD)
    F : static nonlinear function 
        A static nonlinearity, either a scalar function or a single-input,
        single-output, static input/output system.
    a : list
        List of amplitudes to be used for the describing function plot.
    omega : list, optional
        List of frequences to be used for the linear system Nyquist curve.

    """
    # Start by drawing a Nyquist curve
    H_real, H_imag, H_omega = nyquist_plot(H, omega, plot=True)

    # Compute the describing function
    df = describing_function(F, a)
    dfinv = -1/df

    # Now add on the describing function
    plt.plot(dfinv.real, dfinv.imag)


# Class for nonlinear functions
class NonlinearFunction():
    def sector_bounds(self, lb, ub):
        raise NotImplementedError(
            "sector bounds not implemented for this function")

    def describing_function(self, amp):
        raise NotImplementedError(
            "describing function not implemented for this function")

    # Function to compute the describing function
    def _f(self, x):
        return math.copysign(1, x) if abs(x) > 1 else \
            (math.asin(x) + x * math.sqrt(1 - x**2)) * 2 / math.pi


# Saturation nonlinearity
class saturation_nonlinearity(NonlinearFunction):
    def __init__(self, ub=1, lb=None):
        # Process arguments
        if lb == None:
            # Only received one argument; assume symmetric around zero
            lb, ub = -abs(ub), abs(ub)

        # Make sure the bounds are sensity
        if lb > 0 or ub < 0 or lb + ub != 0:
            warn("asymmetric saturation; ignoring non-zero bias term")

        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        return np.maximum(self.lb, np.minimum(x, self.ub))

    def describing_function(self, A):
        if self.lb <= A and A <= self.ub:
            return 1.
        else:
            alpha, beta = math.asin(self.ub/A), math.asin(-self.lb/A)
            return (math.sin(alpha + beta) * math.cos(alpha - beta) +
                    (alpha + beta)) / math.pi

        
# Hysteresis w/ deadzone (#40 in Gelb and Vander Velde, 1968)
class hysteresis_deadzone_nonlinearity(NonlinearFunction):
    def __init__(self, delta, D, m):
        # Initialize the state to bottom branch
        self.branch = -1        # lower branch
        self.delta = delta
        self.D = D
        self.m = m

    def __call__(self, x):
        if x > self.delta + self.D / self.m:
            y = self.m * (x - self.delta)
            self.branch = 1
        elif x < -self.delta - self.D/self.m:
            y = self.m * (x + self.delta)
            self.branch = -1
        elif self.branch == -1 and \
             x > -self.delta - self.D / self.m and \
             x < self.delta - self.D / self.m:
            y = -self.D
        elif self.branch == -1 and x >= self.delta - self.D / self.m:
            y = self.m * (x - self.delta)
        elif self.branch == 1 and \
             x > -self.delta + self.D / self.m and \
             x < self.delta + self.D / self.m:
            y = self.D
        elif self.branch == 1 and x <= -self.delta + self.D / self.m:
            y = self.m * (x + self.delta)
        return y

    def describing_function(self, A):
        def f(x):
            return math.copysign(1, x) if abs(x) > 1 else \
                (math.asin(x) + x * math.sqrt(1 - x**2)) * 2 / math.pi

        if A < self.delta + self.D/self.m:
            return np.nan
            
        df_real = self.m/2 * \
            (2 - self._f((self.D/self.m + self.delta)/A) +
             self._f((self.D/self.m - self.delta)/A))
        df_imag = -4 * self.D * self.delta / (math.pi * A**2)
        return df_real + 1j * df_imag


# Backlash nonlinearity (#48 in Gelb and Vander Velde, 1968)
class backlash_nonlinearity(NonlinearFunction):
    def __init__(self, b):
        self.b = b              # backlash distance
        self.center = 0         # current center position

    def __call__(self, x):
        # If we are outside the backlash, move and shift the center
        if x - self.center > self.b/2:
            self.center = x - self.b/2
        elif x - self.center < -self.b/2:
            self.center = x + self.b/2
        return self.center

    def describing_function(self, A):
        if A < self.b/2:
            return 0
            
        df_real = (1 + self._f(1 - self.b/A)) / 2
        df_imag = -(2 * self.b/A - (self.b/A)**2) / math.pi
        return df_real + 1j * df_imag
