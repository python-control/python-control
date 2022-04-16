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
import control.optimal as opt

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
        traj = fs.point_to_point(flatsys, Tf, x1, u1, xf, uf, basis=poly)

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

    @pytest.fixture
    def vehicle_flat(self):
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
        return fs.FlatSystem(
            vehicle_flat_forward, vehicle_flat_reverse, vehicle_update,
            vehicle_output, inputs=('v', 'delta'), outputs=('x', 'y', 'theta'),
            states=('x', 'y', 'theta'))

    @pytest.mark.parametrize("poly", [
        fs.PolyFamily(6), fs.PolyFamily(8), fs.BezierFamily(6)])
    def test_kinematic_car(self, vehicle_flat, poly):
        # Define the endpoints of the trajectory
        x0 = [0., -2., 0.]; u0 = [10., 0.]
        xf = [100., 2., 0.]; uf = [10., 0.]
        Tf = 10

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)

        # Verify that the trajectory computation is correct
        x, u = traj.eval([0, Tf])
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, 1])
        np.testing.assert_array_almost_equal(uf, u[:, 1])

        # Simulate the system and make sure we stay close to desired traj
        T = np.linspace(0, Tf, 100)
        xd, ud = traj.eval(T)
        resp = ct.input_output_response(vehicle_flat, T, ud, x0)
        np.testing.assert_array_almost_equal(resp.states, xd, decimal=2)

        # For SciPy 1.0+, integrate equations and compare to desired
        if StrictVersion(sp.__version__) >= "1.0":
            t, y, x = ct.input_output_response(
                vehicle_flat, T, ud, x0, return_x=True)
            np.testing.assert_allclose(x, xd, atol=0.01, rtol=0.01)

    def test_flat_default_output(self, vehicle_flat):
        # Construct a flat system with the default outputs
        flatsys = fs.FlatSystem(
            vehicle_flat.forward, vehicle_flat.reverse, vehicle_flat.updfcn,
            inputs=vehicle_flat.ninputs, outputs=vehicle_flat.ninputs,
            states=vehicle_flat.nstates)

        # Define the endpoints of the trajectory
        x0 = [0., -2., 0.]; u0 = [10., 0.]
        xf = [100., 2., 0.]; uf = [10., 0.]
        Tf = 10

        # Find trajectory between initial and final conditions
        poly = fs.PolyFamily(6)
        traj1 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=poly)
        traj2 = fs.point_to_point(flatsys, Tf, x0, u0, xf, uf, basis=poly)

        # Verify that the trajectory computation is correct
        T = np.linspace(0, Tf, 10)
        x1, u1 = traj1.eval(T)
        x2, u2 = traj2.eval(T)
        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(u1, u2)

        # Run a simulation and verify that the outputs are correct
        resp1 = ct.input_output_response(vehicle_flat, T, u1, x0)
        resp2 = ct.input_output_response(flatsys, T, u1, x0)
        np.testing.assert_array_almost_equal(resp1.outputs[0:2], resp2.outputs)

    def test_flat_cost_constr(self):
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [0, 0]; uf = [0]
        Tf = 10
        T = np.linspace(0, Tf, 500)

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(
            flat_sys, Tf, x0, u0, xf, uf, basis=fs.PolyFamily(8))
        x, u = traj.eval(T)

        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Solve with a cost function
        timepts = np.linspace(0, Tf, 10)
        cost_fcn = opt.quadratic_cost(
            flat_sys, np.diag([0, 0]), 1, x0=xf, u0=uf)

        traj_cost = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            basis=fs.PolyFamily(8),
            # initial_guess='lstsq',
            # minimize_kwargs={'method': 'trust-constr'}
        )

        # Verify that the trajectory computation is correct
        x_cost, u_cost = traj_cost.eval(T)
        np.testing.assert_array_almost_equal(x0, x_cost[:, 0])
        np.testing.assert_array_almost_equal(u0, u_cost[:, 0])
        np.testing.assert_array_almost_equal(xf, x_cost[:, -1])
        np.testing.assert_array_almost_equal(uf, u_cost[:, -1])

        # Make sure that we got a different answer than before
        assert np.any(np.abs(x - x_cost) > 0.1)

        # Re-solve with constraint on the y deviation
        lb, ub = [-2, -0.1], [2, 0]
        lb, ub = [-2, np.min(x_cost[1])*0.95], [2, 1]
        constraints = [opt.state_range_constraint(flat_sys, lb, ub)]

        # Make sure that the previous solution violated at least one constraint
        assert np.any(x_cost[0, :] < lb[0]) or np.any(x_cost[0, :] > ub[0]) \
            or np.any(x_cost[1, :] < lb[1]) or np.any(x_cost[1, :] > ub[1])

        traj_const = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            constraints=constraints, basis=fs.PolyFamily(8),
        )

        # Verify that the trajectory computation is correct
        x_const, u_const = traj_const.eval(T)
        np.testing.assert_array_almost_equal(x0, x_const[:, 0])
        np.testing.assert_array_almost_equal(u0, u_const[:, 0])
        np.testing.assert_array_almost_equal(xf, x_const[:, -1])
        np.testing.assert_array_almost_equal(uf, u_const[:, -1])

        # Make sure that the solution respects the bounds (with some slop)
        for i in range(x_const.shape[0]):
            assert np.all(x_const[i] >= lb[i] * 1.02)
            assert np.all(x_const[i] <= ub[i] * 1.02)

        # Solve the same problem with a nonlinear constraint type
        nl_constraints = [
            (sp.optimize.NonlinearConstraint, lambda x, u: x, lb, ub)]
        traj_nlconst = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            constraints=nl_constraints, basis=fs.PolyFamily(8),
        )
        x_nlconst, u_nlconst = traj_nlconst.eval(T)
        np.testing.assert_almost_equal(x_const, x_nlconst)
        np.testing.assert_almost_equal(u_const, u_nlconst)

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
        for N in range(5):
            bezier = fs.BezierFamily(N)
            for i in range(N):
                for j in range(1, N+1):
                    np.testing.assert_allclose(
                        np.diff(bezier.eval_deriv(i, j-1, time)) / dt,
                        bezier.eval_deriv(i, j, time)[0:-1],
                        atol=0.01, rtol=0.01)

        # Exception check
        with pytest.raises(ValueError, match="index too high"):
            bezier.eval_deriv(4, 0, time)

    def test_point_to_point_errors(self):
        """Test error and warning conditions in point_to_point()"""
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [0, 0]; uf = [0]
        Tf = 10
        T = np.linspace(0, Tf, 500)

        # Cost function
        timepts = np.linspace(0, Tf, 10)
        cost_fcn = opt.quadratic_cost(
            flat_sys, np.diag([1, 1]), 1, x0=xf, u0=uf)

        # Solving without basis specified should be OK
        traj = fs.point_to_point(flat_sys, timepts, x0, u0, xf, uf)
        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Adding a cost function generates a warning
        with pytest.warns(UserWarning, match="optimization not possible"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn)

        # Make sure we still solved the problem
        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Try to optimize with insufficient degrees of freedom
        with pytest.warns(UserWarning, match="optimization not possible"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
                basis=fs.PolyFamily(6))

        # Make sure we still solved the problem
        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Solve with the errors in the various input arguments
        with pytest.raises(ValueError, match="Initial state: Wrong shape"):
            traj = fs.point_to_point(flat_sys, timepts, np.zeros(3), u0, xf, uf)
        with pytest.raises(ValueError, match="Initial input: Wrong shape"):
            traj = fs.point_to_point(flat_sys, timepts, x0, np.zeros(3), xf, uf)
        with pytest.raises(ValueError, match="Final state: Wrong shape"):
            traj = fs.point_to_point(flat_sys, timepts, x0, u0, np.zeros(3), uf)
        with pytest.raises(ValueError, match="Final input: Wrong shape"):
            traj = fs.point_to_point(flat_sys, timepts, x0, u0, xf, np.zeros(3))

        # Different ways of describing constraints
        constraint =  opt.input_range_constraint(flat_sys, -100, 100)

        with pytest.warns(UserWarning, match="optimization not possible"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, constraints=constraint,
                basis=fs.PolyFamily(6))

        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Constraint that isn't a constraint
        with pytest.raises(TypeError, match="must be a list"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, constraints=np.eye(2),
                basis=fs.PolyFamily(8))

        # Unknown constraint type
        with pytest.raises(TypeError, match="unknown constraint type"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf,
                constraints=[(None, 0, 0, 0)], basis=fs.PolyFamily(8))

        # Unsolvable optimization
        constraint = [opt.input_range_constraint(flat_sys, -0.01, 0.01)]
        with pytest.raises(RuntimeError, match="Unable to solve optimal"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, constraints=constraint,
                basis=fs.PolyFamily(8))

        # Method arguments, parameters
        traj_method = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            basis=fs.PolyFamily(8), minimize_method='slsqp')
        traj_kwarg = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            basis=fs.PolyFamily(8), minimize_kwargs={'method': 'slsqp'})
        np.testing.assert_allclose(
            traj_method.eval(timepts)[0], traj_kwarg.eval(timepts)[0],
            atol=1e-5)

        # Unrecognized keywords
        with pytest.raises(TypeError, match="unrecognized keyword"):
            traj_method = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, solve_ivp_method=None)

    @pytest.mark.parametrize(
        "xf, uf, Tf",
        [([1, 0], [0], 2),
         ([0, 1], [0], 3),
         ([1, 1], [1], 4)])
    def test_response(self, xf, uf, Tf):
        # Define a second order integrator
        sys = ct.StateSpace([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
        flatsys = fs.LinearFlatSystem(sys)

        # Define the basis set
        poly = fs.PolyFamily(6)

        x1, u1, = [0, 0], [0]
        traj = fs.point_to_point(flatsys, Tf, x1, u1, xf, uf, basis=poly)

        # Compute the response the regular way
        T = np.linspace(0, Tf, 10)
        x, u = traj.eval(T)

        # Recompute using response()
        response = traj.response(T, squeeze=False)
        np.testing.assert_equal(T, response.time)
        np.testing.assert_equal(u, response.inputs)
        np.testing.assert_equal(x, response.states)
