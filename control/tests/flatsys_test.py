#!/usr/bin/env python
#
# flatsys_test.py - test flat system module
# RMM, 29 Jun 2019
#
# This test suite checks to make sure that the basic functions supporting
# differential flat systetms are functioning.  It doesn't do exhaustive
# testing of operations on flat systems.  Separate unit tests should be
# created for that purpose.

import unittest
import numpy as np
import scipy as sp
import control as ct
import control.flatsys as fs
from distutils.version import StrictVersion


class TestFlatSys(unittest.TestCase):
    def setUp(self):
        ct.use_numpy_matrix(False)

    def test_double_integrator(self):
        # Define a second order integrator
        sys = ct.StateSpace([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
        flatsys = fs.LinearFlatSystem(sys)

        # Define the endpoints of a trajectory
        x1 = [0, 0]; u1 = [0]; T1 = 1
        x2 = [1, 0]; u2 = [0]; T2 = 2
        x3 = [0, 1]; u3 = [0]; T3 = 3
        x4 = [1, 1]; u4 = [1]; T4 = 4

        # Define the basis set
        poly = fs.PolyFamily(6)

        # Plan trajectories for various combinations
        for x0, u0, xf, uf, Tf in [
            (x1, u1, x2, u2, T2), (x1, u1, x3, u3, T3), (x1, u1, x4, u4, T4)]:
            traj = fs.point_to_point(flatsys, x0, u0, xf, uf, Tf, basis=poly)

            # Verify that the trajectory computation is correct
            x, u = traj.eval([0, Tf])
            np.testing.assert_array_almost_equal(x0, x[:, 0])
            np.testing.assert_array_almost_equal(u0, u[:, 0])
            np.testing.assert_array_almost_equal(xf, x[:, 1])
            np.testing.assert_array_almost_equal(uf, u[:, 1])

            # Simulate the system and make sure we stay close to desired traj
            T = np.linspace(0, Tf, 100)
            xd, ud = traj.eval(T)

            t, y, x = ct.forced_response(sys, T, ud, x0)
            np.testing.assert_array_almost_equal(x, xd, decimal=3)

    def test_kinematic_car(self):
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

        # Define a set of basis functions to use for the trajectories
        poly = fs.PolyFamily(6)

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

    def tearDown(self):
        ct.reset_defaults()


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestFlatSys)


if __name__ == '__main__':
    unittest.main()
