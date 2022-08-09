"""bspline_test.py - test bsplines and their use in flat system

RMM, 2 Aug 2022

This test suite checks to make sure that the bspline basic functions
supporting differential flat systetms are functioning.  It doesn't do
exhaustive testing of operations on flat systems.  Separate unit tests
should be created for that purpose.

"""

import numpy as np
import pytest
import scipy as sp

import control as ct
import control.flatsys as fs
import control.optimal as opt

def test_bspline_basis():
    Tf = 10
    degree = 5
    maxderiv = 4
    bspline = fs.BSplineFamily([0, Tf/3, Tf/2, Tf], degree, maxderiv)
    time = np.linspace(0, Tf, 100)

    # Make sure that the knotpoint vector looks right
    np.testing.assert_equal(
        bspline.knotpoints,
        [np.array([0, 0, 0, 0, 0, 0,
                  Tf/3, Tf/2,
                   Tf, Tf, Tf, Tf, Tf, Tf])])

    # Repeat with default smoothness
    bspline = fs.BSplineFamily([0, Tf/3, Tf/2, Tf], degree)
    np.testing.assert_equal(
        bspline.knotpoints,
        [np.array([0, 0, 0, 0, 0, 0,
                  Tf/3, Tf/2,
                   Tf, Tf, Tf, Tf, Tf, Tf])])

    # Sum of the B-spline curves should be one
    np.testing.assert_almost_equal(
        1, sum([bspline(i, time) for i in range(bspline.N)]))

    # Sum of derivatives should be zero
    for k in range(1, maxderiv):
        np.testing.assert_almost_equal(
            0, sum([bspline.eval_deriv(i, k, time)
                    for i in range(0, bspline.N)]))

    # Make sure that the second derivative integrates to the first
    time = np.linspace(0, Tf, 1000)
    dt = time[1] - time[0]
    for i in range(bspline.N):
        for j in range(1, maxderiv):
            np.testing.assert_allclose(
                np.diff(bspline.eval_deriv(i, j-1, time)) / dt,
                bspline.eval_deriv(i, j, time)[0:-1],
                atol=0.01, rtol=0.01)

    # Make sure that ndarrays are processed the same as integer lists
    degree = np.array(degree)
    bspline2 = fs.BSplineFamily([0, Tf/3, Tf/2, Tf], degree, maxderiv)
    np.testing.assert_equal(bspline(0, time), bspline2(0, time))

    # Exception check
    with pytest.raises(IndexError, match="out of bounds"):
        bspline.eval_deriv(bspline.N, 0, time)


@pytest.mark.parametrize(
    "xf, uf, Tf",
    [([1, 0], [0], 2),
     ([0, 1], [0], 3),
     ([1, 1], [1], 4)])
def test_double_integrator(xf, uf, Tf):
    # Define a second order integrator
    sys = ct.StateSpace([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
    flatsys = fs.LinearFlatSystem(sys)

    # Define the basis set
    bspline = fs.BSplineFamily([0, Tf/2, Tf], 4, 2)

    x0, u0, = [0, 0], [0]
    traj = fs.point_to_point(flatsys, Tf, x0, u0, xf, uf, basis=bspline)

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, 1])
    np.testing.assert_array_almost_equal(uf, u[:, 1])

    # Simulate the system and make sure we stay close to desired traj
    T = np.linspace(0, Tf, 200)
    xd, ud = traj.eval(T)

    t, y, x = ct.forced_response(sys, T, ud, x0, return_x=True)
    np.testing.assert_array_almost_equal(x, xd, decimal=3)


# Bicycle model
def vehicle_flat_forward(x, u, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    zflag = [np.zeros(3), np.zeros(3)]          # list for flag arrays
    zflag[0][0] = x[0]                          # flat outputs
    zflag[1][0] = x[1]
    zflag[0][1] = u[0] * np.cos(x[2])           # first derivatives
    zflag[1][1] = u[0] * np.sin(x[2])
    thdot = (u[0]/b) * np.tan(u[1])             # dtheta/dt
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])  # second derivatives
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])
    return zflag

def vehicle_flat_reverse(zflag, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    x = np.zeros(3); u = np.zeros(2)            # vectors to store x, u
    x[0] = zflag[0][0]                          # x position
    x[1] = zflag[1][0]                          # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1]) # angle
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)
    return x, u

def vehicle_update(t, x, u, params):
    b = params.get('wheelbase', 3.)             # get parameter values
    dx = np.array([
        np.cos(x[2]) * u[0],
        np.sin(x[2]) * u[0],
        (u[0]/b) * np.tan(u[1])
    ])
    return dx

def vehicle_output(t, x, u, params): return x

# Create differentially flat input/output system
vehicle_flat = fs.FlatSystem(
    vehicle_flat_forward, vehicle_flat_reverse, vehicle_update,
    vehicle_output, inputs=('v', 'delta'), outputs=('x', 'y', 'theta'),
    states=('x', 'y', 'theta'))

def test_kinematic_car():
    # Define the endpoints of the trajectory
    x0 = [0., -2., 0.]; u0 = [10., 0.]
    xf = [100., 2., 0.]; uf = [10., 0.]
    Tf = 10

    # Set up a basis vector
    bspline = fs.BSplineFamily([0, Tf/2, Tf], 5, 3)

    # Find trajectory between initial and final conditions
    traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=bspline)

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, 1])
    np.testing.assert_array_almost_equal(uf, u[:, 1])

def test_kinematic_car_multivar():
    # Define the endpoints of the trajectory
    x0 = [0., -2., 0.]; u0 = [10., 0.]
    xf = [100., 2., 0.]; uf = [10., 0.]
    Tf = 10

    # Set up a basis vector
    bspline = fs.BSplineFamily([0, Tf/2, Tf], [5, 6], [3, 4], vars=2)

    # Find trajectory between initial and final conditions
    traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=bspline)

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, 1])
    np.testing.assert_array_almost_equal(uf, u[:, 1])

def test_bspline_errors():
    # Breakpoints must be a 1D array, in increasing order
    with pytest.raises(NotImplementedError, match="not yet supported"):
        basis = fs.BSplineFamily([[0, 1, 3], [0, 2, 3]], [3, 3])

    with pytest.raises(ValueError,
                       match="breakpoints must be convertable to a 1D array"):
        basis = fs.BSplineFamily([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], [3, 3])

    with pytest.raises(ValueError, match="must have at least 2 values"):
        basis = fs.BSplineFamily([10], 2)

    with pytest.raises(ValueError, match="must be strictly increasing"):
        basis = fs.BSplineFamily([1, 3, 2], 2)

    # Smoothness can't be more than dimension of splines
    basis = fs.BSplineFamily([0, 1], 4, 3)      # OK
    with pytest.raises(ValueError, match="degree must be greater"):
        basis = fs.BSplineFamily([0, 1], 4, 4)  # not OK

    # nvars must be an integer
    with pytest.raises(TypeError, match="vars must be an integer"):
        basis = fs.BSplineFamily([0, 1], 4, 3, vars=['x1', 'x2'])

    # degree, smoothness must match nvars
    with pytest.raises(ValueError, match="length of 'degree' does not match"):
        basis = fs.BSplineFamily([0, 1], [4, 4, 4], 3, vars=2)

    # degree, smoothness must be list of ints
    basis = fs.BSplineFamily([0, 1], [4, 4], 3, vars=2) # OK
    with pytest.raises(ValueError, match="could not parse 'degree'"):
        basis = fs.BSplineFamily([0, 1], [4, '4'], 3, vars=2)

    # degree must be strictly positive
    with pytest.raises(ValueError, match="'degree'; must be at least 1"):
        basis = fs.BSplineFamily([0, 1], 0, 1)

    # smoothness must be non-negative
    with pytest.raises(ValueError, match="'smoothness'; must be at least 0"):
        basis = fs.BSplineFamily([0, 1], 2, -1)
