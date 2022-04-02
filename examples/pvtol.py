# pvtol.py - (planar) vertical takeoff and landing system model
# RMM, 19 Jan 2022
#
# This file contains a model and utility function for a (planar)
# vertical takeoff and landing system, as described in FBS2e and OBC.
# This system is approximately differentially flat and the flat system
# mappings are included.
#

import numpy as np
import matplotlib.pyplot as plt
import control as ct
import control.flatsys as fs
from math import sin, cos
from warnings import warn

# PVTOL dynamics
def pvtol_update(t, x, u, params):
    # Get the parameter values
    m = params.get('m', 4.)             # mass of aircraft
    J = params.get('J', 0.0475)         # inertia around pitch axis
    r = params.get('r', 0.25)           # distance to center of force
    g = params.get('g', 9.8)            # gravitational constant
    c = params.get('c', 0.05)           # damping factor (estimated)

    # Get the inputs and states
    x, y, theta, xdot, ydot, thetadot = x
    F1, F2 = u

    # Constrain the inputs
    F2 = np.clip(F2, 0, 1.5 * m * g)
    F1 = np.clip(F1, -0.1 * F2, 0.1 * F2)

    # Dynamics
    xddot = (F1 * cos(theta) - F2 * sin(theta) - c * xdot) / m
    yddot = (F1 * sin(theta) + F2 * cos(theta) - m * g - c * ydot) / m
    thddot = (r * F1) / J

    return np.array([xdot, ydot, thetadot, xddot, yddot, thddot])

def pvtol_output(t, x, u, params):
    return x

# PVTOL flat system mappings
def pvtol_flat_forward(states, inputs, params={}):
    # Get the parameter values
    m = params.get('m', 4.)             # mass of aircraft
    J = params.get('J', 0.0475)         # inertia around pitch axis
    r = params.get('r', 0.25)           # distance to center of force
    g = params.get('g', 9.8)            # gravitational constant
    c = params.get('c', 0.05)           # damping factor (estimated)

    # Make sure that c is zero
    if c != 0:
        warn("System is only approximately flat (c != 0)")

    # Create a list of arrays to store the flat output and its derivatives
    zflag = [np.zeros(5), np.zeros(5)]

    # Store states and inputs in variables to make things easier to read
    x, y, theta, xdot, ydot, thdot = states
    F1, F2 = inputs

    # Use equations of motion for higher derivates
    x1ddot = (F1 * cos(theta) - F2 * sin(theta)) / m
    x2ddot = (F1 * sin(theta) + F2 * cos(theta) - m * g) / m
    thddot = (r * F1) / J

    # Flat output is a point above the vertical axis
    zflag[0][0] = x - (J / (m * r)) * sin(theta)
    zflag[1][0] = y + (J / (m * r)) * cos(theta)

    zflag[0][1] = xdot - (J / (m * r)) * cos(theta) * thdot
    zflag[1][1] = ydot - (J / (m * r)) * sin(theta) * thdot

    zflag[0][2] = (F1 * cos(theta) - F2 * sin(theta)) / m \
        + (J / (m * r)) * sin(theta) * thdot**2 \
        - (J / (m * r)) * cos(theta) * thddot
    zflag[1][2] = (F1 * sin(theta) + F2 * cos(theta) - m * g) / m \
        - (J / (m * r)) * cos(theta) * thdot**2 \
        - (J / (m * r)) * sin(theta) * thddot

    # For the third derivative, assume F1, F2 are constant (also thddot)
    zflag[0][3] = (-F1 * sin(theta) - F2 * cos(theta)) * (thdot / m) \
        + (J / (m * r)) * cos(theta) * thdot**3 \
        + 3 * (J / (m * r)) * sin(theta) * thdot * thddot
    zflag[1][3] = (F1 * cos(theta) - F2 * sin(theta)) * (thdot / m) \
        + (J / (m * r)) * sin(theta) * thdot**3 \
        - 3 * (J / (m * r)) * cos(theta) * thdot * thddot

    # For the fourth derivative, assume F1, F2 are constant (also thddot)
    zflag[0][4] = (-F1 * sin(theta) - F2 * cos(theta)) * (thddot / m) \
        + (-F1 * cos(theta) + F2 * sin(theta)) * (thdot**2 / m) \
        + 6 * (J / (m * r)) * cos(theta) * thdot**2 * thddot \
        + 3 * (J / (m * r)) * sin(theta) * thddot**2 \
        - (J / (m * r)) * sin(theta) * thdot**4
    zflag[1][4] = (F1 * cos(theta) - F2 * sin(theta)) * (thddot / m) \
        + (-F1 * sin(theta) - F2 * cos(theta)) * (thdot**2 / m) \
        - 6 * (J / (m * r)) * sin(theta) * thdot**2 * thddot \
        - 3 * (J / (m * r)) * cos(theta) * thddot**2 \
        + (J / (m * r)) * cos(theta) * thdot**4

    return zflag

def pvtol_flat_reverse(zflag, params={}):
    # Get the parameter values
    m = params.get('m', 4.)             # mass of aircraft
    J = params.get('J', 0.0475)         # inertia around pitch axis
    r = params.get('r', 0.25)           # distance to center of force
    g = params.get('g', 9.8)            # gravitational constant
    c = params.get('c', 0.05)           # damping factor (estimated)

    # Create a vector to store the state and inputs
    x = np.zeros(6)
    u = np.zeros(2)

    # Given the flat variables, solve for the state
    theta = np.arctan2(-zflag[0][2],  zflag[1][2] + g)
    x = zflag[0][0] + (J / (m * r)) * sin(theta)
    y = zflag[1][0] - (J / (m * r)) * cos(theta)

    # Solve for thdot using next derivative
    thdot = (zflag[0][3] * cos(theta) + zflag[1][3] * sin(theta)) \
        / (zflag[0][2] * sin(theta) - (zflag[1][2] + g) * cos(theta))

    # xdot and ydot can be found from first derivative of flat outputs
    xdot = zflag[0][1] + (J / (m * r)) * cos(theta) * thdot
    ydot = zflag[1][1] + (J / (m * r)) * sin(theta) * thdot

    # Solve for the input forces
    F2 = m * ((zflag[1][2] + g) * cos(theta) - zflag[0][2] * sin(theta)
              + (J / (m * r)) * thdot**2)
    F1 = (J / r) * \
        (zflag[0][4] * cos(theta) + zflag[1][4] * sin(theta)
#         - 2 * (zflag[0][3] * sin(theta) - zflag[1][3] * cos(theta)) * thdot \
         - 2 * zflag[0][3] * sin(theta) * thdot \
         + 2 * zflag[1][3] * cos(theta) * thdot \
#         - (zflag[0][2] * cos(theta)
#            + (zflag[1][2] + g) * sin(theta)) * thdot**2) \
         - zflag[0][2] * cos(theta) * thdot**2
         - (zflag[1][2] + g) * sin(theta) * thdot**2) \
        / (zflag[0][2] * sin(theta) - (zflag[1][2] + g) * cos(theta))

    return np.array([x, y, theta, xdot, ydot, thdot]), np.array([F1, F2])

pvtol = fs.FlatSystem(
    pvtol_flat_forward, pvtol_flat_reverse, name='pvtol',
    updfcn=pvtol_update, outfcn=pvtol_output,
    states = [f'x{i}' for i in range(6)],
    inputs = ['F1', 'F2'],
    outputs = [f'x{i}' for i in range(6)],
    params = {
        'm': 4.,                # mass of aircraft
        'J': 0.0475,            # inertia around pitch axis
        'r': 0.25,              # distance to center of force
        'g': 9.8,               # gravitational constant
        'c': 0.05,              # damping factor (estimated)
    }
)

#
# PVTOL dynamics with wind
# 

def windy_update(t, x, u, params):
    # Get the input vector
    F1, F2, d = u

    # Get the system response from the original dynamics
    xdot, ydot, thetadot, xddot, yddot, thddot = \
        pvtol_update(t, x, u[0:2], params)

    # Now add the wind term
    m = params.get('m', 4.)             # mass of aircraft
    xddot += d / m

    return np.array([xdot, ydot, thetadot, xddot, yddot, thddot])

windy_pvtol = ct.NonlinearIOSystem(
    windy_update, pvtol_output, name="windy_pvtol",
    states = [f'x{i}' for i in range(6)],
    inputs = ['F1', 'F2', 'd'],
    outputs = [f'x{i}' for i in range(6)]
)

#
# PVTOL dynamics with noise and disturbances
# 

def noisy_update(t, x, u, params):
    # Get the inputs
    F1, F2, Dx, Dy, Nx, Ny, Nth = u

    # Get the system response from the original dynamics
    xdot, ydot, thetadot, xddot, yddot, thddot = \
        pvtol_update(t, x, u[0:2], params)

    # Get the parameter values we need
    m = params.get('m', 4.)             # mass of aircraft
    J = params.get('J', 0.0475)         # inertia around pitch axis

    # Now add the disturbances
    xddot += Dx / m
    yddot += Dy / m

    return np.array([xdot, ydot, thetadot, xddot, yddot, thddot])

def noisy_output(t, x, u, params):
    F1, F2, dx, Dy, Nx, Ny, Nth = u
    return x + np.array([Nx, Ny, Nth, 0, 0, 0])

noisy_pvtol = ct.NonlinearIOSystem(
    noisy_update, noisy_output, name="noisy_pvtol",
    states = [f'x{i}' for i in range(6)],
    inputs = ['F1', 'F2'] + ['Dx', 'Dy'] + ['Nx', 'Ny', 'Nth'],
    outputs = pvtol.state_labels
)

# Add the linearitizations to the dynamics as additional methods
def noisy_pvtol_A(x, u, params={}):
    # Get the parameter values we need
    m = params.get('m', 4.)             # mass of aircraft
    J = params.get('J', 0.0475)         # inertia around pitch axis
    c = params.get('c', 0.05)           # damping factor (estimated)

    # Get the angle and compute sine and cosine
    theta = x[[2]]
    cth, sth = cos(theta), sin(theta)

    # Return the linearized dynamics matrix
    return np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, (-u[0] * sth - u[1] * cth)/m, -c/m, 0, 0],
        [0, 0, ( u[0] * cth - u[1] * sth)/m, 0, -c/m, 0],
        [0, 0, 0, 0, 0, 0]
    ])
pvtol.A = noisy_pvtol_A

# Plot the trajectory in xy coordinates
def plot_results(t, x, u):
    # Set the size of the figure
    plt.figure(figsize=(10, 6))

    # Top plot: xy trajectory
    plt.subplot(2, 1, 1)
    plt.plot(x[0], x[1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')

    # Time traces of the state and input
    plt.subplot(2, 4, 5)
    plt.plot(t, x[1])
    plt.xlabel('Time t [sec]')
    plt.ylabel('y [m]')

    plt.subplot(2, 4, 6)
    plt.plot(t, x[2])
    plt.xlabel('Time t [sec]')
    plt.ylabel('theta [rad]')

    plt.subplot(2, 4, 7)
    plt.plot(t, u[0])
    plt.xlabel('Time t [sec]')
    plt.ylabel('$F_1$ [N]')

    plt.subplot(2, 4, 8)
    plt.plot(t, u[1])
    plt.xlabel('Time t [sec]')
    plt.ylabel('$F_2$ [N]')
    plt.tight_layout()

#
# Additional functions for testing
#

# Check flatness calculations
def _pvtol_check_flat(test_points=None, verbose=False):
    if test_points is None:
        # If no test points, use internal set
        mg = 9.8 * 4
        test_points = [
            ([0, 0, 0, 0, 0, 0], [0, mg]),
            ([1, 0, 0, 0, 0, 0], [0, mg]),
            ([0, 1, 0, 0, 0, 0], [0, mg]),
            ([1, 1, 0.1, 0, 0, 0], [0, mg]),
            ([0, 0, 0.1, 0, 0, 0], [0, mg]),
            ([0, 0, 0, 1, 0, 0], [0, mg]),
            ([0, 0, 0, 0, 1, 0], [0, mg]),
            ([0, 0, 0, 0, 0, 0.1], [0, mg]),
            ([0, 0, 0, 1, 1, 0.1], [0, mg]),
            ([0, 0, 0, 0, 0, 0], [1, mg]),
            ([0, 0, 0, 0, 0, 0], [0, mg + 1]),
            ([0, 0, 0, 0, 0, 0.1], [1, mg]),
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, mg + 1]),
        ]
    elif isinstance(test_points, tuple):
        # If we only got one test point, convert to a list
        test_points = [test_points]

    for x, u in test_points:
        x, u = np.array(x), np.array(u)
        flag = pvtol_flat_forward(x, u)
        xc, uc = pvtol_flat_reverse(flag)
        print(f'({x}, {u}): ', end='')
        if verbose:
            print(f'\n  flag: {flag}')
            print(f'  check: ({xc}, {uc}): ', end='')
        if np.allclose(x, xc) and np.allclose(u, uc):
            print("OK")
        else:
            print("ERR")
    
