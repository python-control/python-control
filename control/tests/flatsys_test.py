"""flatsys_test.py - test flat system module

RMM, 29 Jun 2019

This test suite checks to make sure that the basic functions supporting
differential flat systetms are functioning.  It doesn't do exhaustive
testing of operations on flat systems.  Separate unit tests should be
created for that purpose.
"""

from distutils.version import StrictVersion

import numpy as np
import pytest
import scipy as sp

import control as ct
import control.flatsys as fs


class TestFlatSys:
    """Test differential flat systems"""

    @pytest.mark.parametrize(
        "xf, uf, Tf",
        [([1, 0], [0], 2),
         ([0, 1], [0], 3),
         ([1, 1], [1], 4)])
    def test_double_integrator(self, xf, uf, Tf):
        # Define a second order integrator
        sys = ct.StateSpace([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
        flatsys = fs.LinearFlatSystem(sys)

        # Define the basis set
        poly = fs.PolyFamily(6)

        x1, u1, = [0, 0], [0]
        traj = fs.point_to_point(flatsys, x1, u1, xf, uf, Tf, basis=poly)

        # Verify that the trajectory computation is correct
        x, u = traj.eval([0, Tf])
        np.testing.assert_array_almost_equal(x1, x[:, 0])
        np.testing.assert_array_almost_equal(u1, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, 1])
        np.testing.assert_array_almost_equal(uf, u[:, 1])

        # Simulate the system and make sure we stay close to desired traj
        T = np.linspace(0, Tf, 100)
        xd, ud = traj.eval(T)

        t, y, x = ct.forced_response(sys, T, ud, x1, return_x=True)
        np.testing.assert_array_almost_equal(x, xd, decimal=3)

    @pytest.mark.parametrize("poly", [fs.PolyFamily(6), fs.BezierFamily(6)])
    def test_kinematic_car(self, poly):
        """Differential flatness for a kinematic car"""
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

        # Define the endpoints of the trajectory
        x0 = [0., -2., 0.]; u0 = [10., 0.]
        xf = [100., 2., 0.]; uf = [10., 0.]
        Tf = 10

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(vehicle_flat, x0, u0, xf, uf, Tf, basis=poly)

        # Verify that the trajectory computation is correct
        x, u = traj.eval([0, Tf])
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, 1])
        np.testing.assert_array_almost_equal(uf, u[:, 1])

        # Simulate the system and make sure we stay close to desired traj
        T = np.linspace(0, Tf, 500)
        xd, ud = traj.eval(T)

        # For SciPy 1.0+, integrate equations and compare to desired
        if StrictVersion(sp.__version__) >= "1.0":
            t, y, x = ct.input_output_response(
                vehicle_flat, T, ud, x0, return_x=True)
            np.testing.assert_allclose(x, xd, atol=0.01, rtol=0.01)

    def test_bezier_basis(self):
        bezier = fs.BezierFamily(4)
        time = np.linspace(0, 1, 100)

        # Sum of the Bezier curves should be one
        np.testing.assert_almost_equal(
            1, sum([bezier(i, time) for i in range(4)]))

        # Sum of derivatives should be zero
        for k in range(1, 5):
            np.testing.assert_almost_equal(
                0, sum([bezier.eval_deriv(i, k, time) for i in range(4)]))

        # Compare derivatives to formulas
        np.testing.assert_almost_equal(
            bezier.eval_deriv(1, 0, time), 3 * time - 6 * time**2 + 3 * time**3)
        np.testing.assert_almost_equal(
            bezier.eval_deriv(1, 1, time), 3 - 12 * time + 9 * time**2)
        np.testing.assert_almost_equal(
            bezier.eval_deriv(1, 2, time), -12 + 18 * time)

        # Make sure that the second derivative integrates to the first
        time = np.linspace(0, 1, 1000)
        dt = np.diff(time)
        for i in range(4):
            for j in (2, 3, 4):
                np.testing.assert_almost_equal(
                    np.diff(bezier.eval_deriv(i, j-1, time)) / dt,
                    bezier.eval_deriv(i, j, time)[0:-1], decimal=2)

        # Exception check
        with pytest.raises(ValueError, match="index too high"):
            bezier.eval_deriv(4, 0, time)
