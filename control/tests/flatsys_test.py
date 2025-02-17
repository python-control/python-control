"""flatsys_test.py - test flat system module

RMM, 29 Jun 2019

This test suite checks to make sure that the basic functions supporting
differential flat systetms are functioning.  It doesn't do exhaustive
testing of operations on flat systems.  Separate unit tests should be
created for that purpose.
"""

import numpy as np
import pytest
import scipy as sp
import re
import warnings
import os
import platform

import control as ct
import control.flatsys as fs
import control.optimal as opt

# Set tolerances for lower/upper bound tests
atol = 1e-4
rtol = 1e-4

class TestFlatSys:
    """Test differential flat systems"""

    @pytest.mark.parametrize(
        " xf,     uf, Tf, basis",
        [([1, 0], [0], 2, fs.PolyFamily(6)),
         ([0, 1], [0], 3, fs.PolyFamily(6)),
         ([0, 1], [0], 3, fs.BezierFamily(6)),
         ([0, 1], [0], 3, fs.BSplineFamily([0, 1.5, 3], 4)),
         ([1, 1], [1], 4, fs.PolyFamily(6)),
         ([1, 1], [1], 4, fs.BezierFamily(6)),
         ([1, 1], [1], 4, fs.BSplineFamily([0, 1.5, 3], 4))])
    def test_double_integrator(self, xf, uf, Tf, basis):
        # Define a second order integrator
        sys = ct.StateSpace([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
        flatsys = fs.LinearFlatSystem(sys)

        x1, u1, = [0, 0], [0]
        traj = fs.point_to_point(flatsys, Tf, x1, u1, xf, uf, basis=basis)

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

    @pytest.mark.parametrize("basis", [
        fs.PolyFamily(6), fs.PolyFamily(8), fs.BezierFamily(6),
        fs.BSplineFamily([0, 10], 8),
        fs.BSplineFamily([0, 5, 10], 4)
    ])
    def test_kinematic_car(self, vehicle_flat, basis):
        # Define the endpoints of the trajectory
        x0 = [0., -2., 0.]; u0 = [10., 0.]
        xf = [100., 2., 0.]; uf = [10., 0.]
        Tf = 10

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=basis)

        # Verify that the trajectory computation is correct
        x, u = traj.eval([0, Tf])
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, 1])
        np.testing.assert_array_almost_equal(uf, u[:, 1])

        # Simulate the system and make sure we stay close to desired traj
        # Note: this can sometimes fail since system is open loop unstable
        T = np.linspace(0, Tf, 100)
        xd, ud = traj.eval(T)
        resp = ct.input_output_response(vehicle_flat, T, ud, x0)
        if not np.allclose(resp.states, xd, atol=1e-2, rtol=1e-2):
            pytest.xfail("system is open loop unstable => errors can build")

        # integrate equations and compare to desired
        t, y, x = ct.input_output_response(
            vehicle_flat, T, ud, x0, return_x=True)
        np.testing.assert_allclose(x, xd, atol=0.01, rtol=0.01)

    @pytest.mark.parametrize(
        "basis, guess, constraints, method", [
        (fs.PolyFamily(8, T=10), 'prev', None, None),
        (fs.BezierFamily(8, T=10), 'linear', None, None),
        (fs.BSplineFamily([0, 10], 8), None, None, None),
        (fs.BSplineFamily([0, 10], 8), 'prev', None, 'trust-constr'),
        (fs.BSplineFamily([0, 10], [6, 8], vars=2), 'prev', None, None),
        (fs.BSplineFamily([0, 5, 10], 5), 'linear', None, 'slsqp'),
        (fs.BSplineFamily([0, 10], 8), None, ([8, -0.1], [12, 0.1]), None),
        (fs.BSplineFamily([0, 5, 10], 5, 3), None, None, None),
    ])
    def test_kinematic_car_ocp(
            self, vehicle_flat, basis, guess, constraints, method):

        # Define the endpoints of the trajectory
        x0 = [0., -2., 0.]; u0 = [10., 0.]
        xf = [40., 2., 0.]; uf = [10., 0.]
        Tf = 4
        timepts = np.linspace(0, Tf, 10)

        # Find trajectory between initial and final conditions
        traj_p2p = fs.point_to_point(
            vehicle_flat, Tf, x0, u0, xf, uf, basis=basis)

        # Verify that the trajectory computation is correct
        x, u = traj_p2p.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        #
        # Re-solve as optimal control problem
        #

        # Define the cost function (mainly penalize steering angle)
        traj_cost = opt.quadratic_cost(
            vehicle_flat, None, np.diag([0.1, 10]), x0=xf, u0=uf)

        # Set terminal cost to bring us close to xf
        terminal_cost = opt.quadratic_cost(
            vehicle_flat, 1e3 * np.eye(3), None, x0=xf)

        # Implement terminal constraints if specified
        if constraints:
            input_constraints = opt.input_range_constraint(
                vehicle_flat, *constraints)
        else:
            input_constraints = None

        # Use a straight line as an initial guess for the trajectory
        if guess == 'prev':
            initial_guess = traj_p2p.eval(timepts)[0][0:2]
        elif guess == 'linear':
            initial_guess = np.array(
                [x0[i] + (xf[i] - x0[i]) * timepts/Tf for i in (0, 1)])
        else:
            initial_guess = None

        # Solve the optimal trajectory (allow warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message="unable to solve", category=UserWarning)
            traj_ocp = fs.solve_flat_optimal(
                vehicle_flat, timepts, x0, u0,
                trajectory_cost=traj_cost,
                trajectory_constraints=input_constraints,
                terminal_cost=terminal_cost, basis=basis,
                initial_guess=initial_guess,
                minimize_kwargs={'method': method},
            )
        xd, ud = traj_ocp.eval(timepts)

        if not traj_ocp.success:
            # Known failure cases
            if re.match(".*precision loss.*", traj_ocp.message):
                pytest.xfail("precision loss in some configurations")

            elif re.match("Iteration limit.*", traj_ocp.message) and \
                 re.match(
                     "conda ubuntu-3.* Generic", os.getenv('JOBNAME', '')) and \
                 re.match("1.24.[012]", np.__version__):
                pytest.xfail("gh820: iteration limit exceeded")

            else:
                # Dump out information to allow creation of an exception
                print("Message:", traj_ocp.message)
                print("Platform:", platform.platform())
                print("Python:", platform.python_version())
                print("NumPy version:", np.__version__)
                np.show_config()
                print("JOBNAME:", os.getenv('JOBNAME'))

                pytest.fail(
                    "unknown failure; view output to identify configuration")

        # Make sure the constraints are satisfied
        if input_constraints:
            _, _, lb, ub = input_constraints
            for i in range(ud.shape[0]):
                assert all(lb[i] - ud[i] < rtol * abs(lb[i]) + atol)
                assert all(ud[i] - ub[i] < rtol * abs(ub[i]) + atol)

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
        basis = fs.PolyFamily(6)
        traj1 = fs.point_to_point(vehicle_flat, Tf, x0, u0, xf, uf, basis=basis)
        traj2 = fs.point_to_point(flatsys, Tf, x0, u0, xf, uf, basis=basis)

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

    @pytest.mark.parametrize("basis", [
        fs.PolyFamily(8),
        fs.BSplineFamily([0, 5, 10], 6),
        fs.BSplineFamily([0, 3, 7, 10], 4, 2)
    ])
    def test_flat_cost_constr(self, basis):
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [0, 0]; uf = [0]
        Tf = 10
        T = np.linspace(0, Tf, 100)

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(
            flat_sys, Tf, x0, u0, xf, uf, basis=basis)
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
            basis=basis,
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
            constraints=constraints, basis=basis,
            # minimize_kwargs={'method': 'trust-constr'}
        )
        assert traj_const.success

        # Verify that the trajectory computation is correct
        x_cost, u_cost = traj_cost.eval(timepts)        # re-eval on timepts
        x_const, u_const = traj_const.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x_const[:, 0])
        np.testing.assert_array_almost_equal(u0, u_const[:, 0])
        np.testing.assert_array_almost_equal(xf, x_const[:, -1])
        np.testing.assert_array_almost_equal(uf, u_const[:, -1])

        # Make sure that the solution respects the bounds (with some slop)
        for i in range(x_const.shape[0]):
            assert all(lb[i] - x_const[i] < rtol * abs(lb[i]) + atol)
            assert all(x_const[i] - ub[i] < rtol * abs(ub[i]) + atol)

        # Solve the same problem with a nonlinear constraint type
        nl_constraints = [
            (sp.optimize.NonlinearConstraint, lambda x, u: x, lb, ub)]
        traj_nlconst = fs.point_to_point(
            flat_sys, timepts, x0, u0, xf, uf, cost=cost_fcn,
            constraints=nl_constraints, basis=basis,
        )
        x_nlconst, u_nlconst = traj_nlconst.eval(timepts)
        np.testing.assert_almost_equal(x_const, x_nlconst, decimal=2)
        np.testing.assert_almost_equal(u_const, u_nlconst, decimal=2)

    @pytest.mark.parametrize("basis", [
        # fs.PolyFamily(8),
        fs.BSplineFamily([0, 3, 7, 10], 5, 2)])
    def test_flat_solve_ocp(self, basis):
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [-1, 0]; uf = [0]
        Tf = 10
        T = np.linspace(0, Tf, 100)

        # Find trajectory between initial and final conditions
        traj = fs.point_to_point(
            flat_sys, Tf, x0, u0, xf, uf, basis=basis)
        x, u = traj.eval(T)

        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])
        np.testing.assert_array_almost_equal(xf, x[:, -1])
        np.testing.assert_array_almost_equal(uf, u[:, -1])

        # Solve with a terminal cost function
        timepts = np.linspace(0, Tf, 10)
        terminal_cost = opt.quadratic_cost(
            flat_sys, 1e3, 1e3, x0=xf, u0=uf)

        traj_cost = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0,
            terminal_cost=terminal_cost, basis=basis)

        # Verify that the trajectory computation is correct
        x_cost, u_cost = traj_cost.eval(T)
        np.testing.assert_array_almost_equal(x0, x_cost[:, 0])
        np.testing.assert_array_almost_equal(u0, u_cost[:, 0])
        np.testing.assert_array_almost_equal(xf, x_cost[:, -1])
        np.testing.assert_array_almost_equal(uf, u_cost[:, -1])

        # Solve with trajectory and terminal cost functions
        trajectory_cost = opt.quadratic_cost(flat_sys, 0, 1, x0=xf, u0=uf)

        traj_cost = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0, terminal_cost=terminal_cost,
            trajectory_cost=trajectory_cost, basis=basis)

        # Verify that the trajectory computation is correct
        x_cost, u_cost = traj_cost.eval(T)
        np.testing.assert_array_almost_equal(x0, x_cost[:, 0])
        np.testing.assert_array_almost_equal(u0, u_cost[:, 0])

        # Make sure we got close on the terminal condition
        assert all(np.abs(x_cost[:, -1] - xf) < 0.1)

        # Make sure that we got a different answer than before
        assert np.any(np.abs(x - x_cost) > 0.1)

        # Re-solve with constraint on the y deviation
        lb, ub = [-2, np.min(x_cost[1])*0.95], [2, 1]
        constraints = [opt.state_range_constraint(flat_sys, lb, ub)]

        # Make sure that the previous solution violated at least one constraint
        assert np.any(x_cost[0, :] < lb[0]) or np.any(x_cost[0, :] > ub[0]) \
            or np.any(x_cost[1, :] < lb[1]) or np.any(x_cost[1, :] > ub[1])

        traj_const = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0,
            terminal_cost=terminal_cost, trajectory_cost=trajectory_cost,
            trajectory_constraints=constraints, basis=basis,
        )

        # Verify that the trajectory computation is correct
        x_const, u_const = traj_const.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x_const[:, 0])
        np.testing.assert_array_almost_equal(u0, u_const[:, 0])

        # Make sure we got close on the terminal condition
        assert all(np.abs(x_cost[:, -1] - xf) < 0.1)

        # Make sure that the solution respects the bounds (with some slop)
        for i in range(x_const.shape[0]):
            assert all(lb[i] - x_const[i] < rtol * abs(lb[i]) + atol)
            assert all(x_const[i] - ub[i] < rtol * abs(ub[i]) + atol)

        # Solve the same problem with a nonlinear constraint type
        # Use alternative keywords as well
        nl_constraints = [
            (sp.optimize.NonlinearConstraint, lambda x, u: x, lb, ub)]
        traj_nlconst = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0,
            trajectory_cost=trajectory_cost, terminal_cost=terminal_cost,
            trajectory_constraints=nl_constraints, basis=basis,
        )
        x_nlconst, u_nlconst = traj_nlconst.eval(timepts)
        np.testing.assert_almost_equal(x_const, x_nlconst)
        np.testing.assert_almost_equal(u_const, u_nlconst)

    def test_solve_flat_ocp_scalar_timepts(self):
        # scalar timepts gives expected result
        f = fs.LinearFlatSystem(ct.ss(ct.tf([1],[1,1])))

        def terminal_cost(x, u):
            return (x-5).dot(x-5)+u.dot(u)

        traj1 = fs.solve_flat_ocp(f, [0, 1], x0=[23],
                                  terminal_cost=terminal_cost)

        traj2 = fs.solve_flat_ocp(f, 1, x0=[23],
                                  terminal_cost=terminal_cost)

        teval = np.linspace(0, 1, 101)

        r1 = traj1.response(teval)
        r2 = traj2.response(teval)

        np.testing.assert_array_equal(r1.x, r2.x)
        np.testing.assert_array_equal(r1.y, r2.y)
        np.testing.assert_array_equal(r1.u, r2.u)


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

    @pytest.mark.parametrize("basis, degree, T", [
        (fs.PolyFamily(4), 4, 1),
        (fs.PolyFamily(4, 100), 4, 100),
        (fs.BezierFamily(4), 4, 1),
        (fs.BezierFamily(4, 100), 4, 100),
        (fs.BSplineFamily([0, 0.5, 1], 4), 3, 1),
        (fs.BSplineFamily([0, 50, 100], 4), 3, 100),
    ])
    def test_basis_derivs(self, basis, degree, T):
        """Make sure that that basis function derivates are correct"""
        timepts = np.linspace(0, T, 10000)
        dt = timepts[1] - timepts[0]
        for i in range(basis.N):
            for j in range(degree-1):
                # Compare numerical and analytical derivative
                np.testing.assert_allclose(
                    np.diff(basis.eval_deriv(i, j, timepts)) / dt,
                    basis.eval_deriv(i, j+1, timepts)[0:-1],
                    atol=1e-2, rtol=1e-4)

    def test_point_to_point_errors(self):
        """Test error and warning conditions in point_to_point()"""
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [0, 0]; uf = [0]
        Tf = 10

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

        # too few timepoints
        with pytest.raises(ct.ControlArgument, match="at least three time points"):
            fs.point_to_point(
                flat_sys, timepts[:2], x0, u0, xf, uf, basis=fs.PolyFamily(10), cost=cost_fcn)

        # Unsolvable optimization
        constraint = [opt.input_range_constraint(flat_sys, -0.01, 0.01)]
        with pytest.warns(UserWarning, match="unable to solve"):
            traj = fs.point_to_point(
                flat_sys, timepts, x0, u0, xf, uf, constraints=constraint,
                basis=fs.PolyFamily(8))
        assert not traj.success

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

    def test_solve_flat_ocp_errors(self):
        """Test error and warning conditions in point_to_point()"""
        # Double integrator system
        sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
        flat_sys = fs.LinearFlatSystem(sys)

        # Define the endpoints of the trajectory
        x0 = [1, 0]; u0 = [0]
        xf = [0, 0]; uf = [0]
        Tf = 10

        # Cost function
        timepts = np.linspace(0, Tf, 10)
        cost_fcn = opt.quadratic_cost(
            flat_sys, np.diag([1, 1]), 1, x0=xf, u0=uf)

        # Solving without basis specified should be OK (may generate warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = fs.solve_flat_optimal(flat_sys, timepts, x0, u0, cost_fcn)
        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        if not traj.success:
            # If unsuccessful, make sure the error is just about precision
            assert re.match(".* precision loss.*", traj.message) is not None

        x, u = traj.eval(timepts)
        np.testing.assert_array_almost_equal(x0, x[:, 0])
        np.testing.assert_array_almost_equal(u0, u[:, 0])

        # Solving without a cost function generates an error
        with pytest.raises(TypeError, match="cost required"):
            traj = fs.solve_flat_optimal(flat_sys, timepts, x0, u0)

        # Try to optimize with insufficient degrees of freedom
        with pytest.raises(ValueError, match="basis set is too small"):
            traj = fs.solve_flat_optimal(
                flat_sys, timepts, x0, u0, trajectory_cost=cost_fcn,
                basis=fs.PolyFamily(2))

        # Solve with the errors in the various input arguments
        with pytest.raises(ValueError, match="Initial state: Wrong shape"):
            traj = fs.solve_flat_optimal(
                flat_sys, timepts, np.zeros(3), u0, cost_fcn)
        with pytest.raises(ValueError, match="Initial input: Wrong shape"):
            traj = fs.solve_flat_optimal(
                flat_sys, timepts, x0, np.zeros(3), cost_fcn)

        # Constraint that isn't a constraint
        with pytest.raises(TypeError, match="must be a list"):
            traj = fs.solve_flat_optimal(
                flat_sys, timepts, x0, u0, cost_fcn,
                trajectory_constraints=np.eye(2), basis=fs.PolyFamily(8))

        # Unknown constraint type
        with pytest.raises(TypeError, match="unknown constraint type"):
            traj = fs.solve_flat_optimal(
                flat_sys, timepts, x0, u0, cost_fcn,
                trajectory_constraints=[(None, 0, 0, 0)],
                basis=fs.PolyFamily(8))

        # Method arguments, parameters
        traj_method = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0, trajectory_cost=cost_fcn,
            basis=fs.PolyFamily(6), minimize_method='slsqp')
        traj_kwarg = fs.solve_flat_optimal(
            flat_sys, timepts, x0, u0, trajectory_cost=cost_fcn,
            basis=fs.PolyFamily(6), minimize_kwargs={'method': 'slsqp'})
        np.testing.assert_allclose(
            traj_method.eval(timepts)[0], traj_kwarg.eval(timepts)[0],
            atol=1e-5)

        # Unrecognized keywords
        with pytest.raises(TypeError, match="unrecognized keyword"):
            traj_method = fs.solve_flat_optimal(
                flat_sys, timepts, x0, u0, cost_fcn, solve_ivp_method=None)

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
        basis = fs.PolyFamily(6)

        x1, u1, = [0, 0], [0]
        traj = fs.point_to_point(flatsys, Tf, x1, u1, xf, uf, basis=basis)

        # Compute the response the regular way
        T = np.linspace(0, Tf, 10)
        x, u = traj.eval(T)

        # Recompute using response()
        response = traj.response(T, squeeze=False)
        np.testing.assert_array_almost_equal(T, response.time)
        np.testing.assert_array_almost_equal(u, response.inputs)
        np.testing.assert_array_almost_equal(x, response.states)

    @pytest.mark.parametrize(
        "basis",
        [fs.PolyFamily(4),
         fs.BezierFamily(4),
         fs.BSplineFamily([0, 1], 4),
         fs.BSplineFamily([0, 1], 4, vars=2),
         fs.BSplineFamily([0, 1], [4, 3], [2, 1], vars=2),
        ])
    def test_basis_class(self, basis):
        timepts = np.linspace(0, 1, 10)

        if basis.nvars is None:
            # Evaluate function on basis vectors
            for j in range(basis.N):
                coefs = np.zeros(basis.N)
                coefs[j] = 1
                np.testing.assert_array_almost_equal(
                    basis.eval(coefs, timepts),
                    basis.eval_deriv(j, 0, timepts))
        else:
            # Evaluate each variable on basis vectors
            for i in range(basis.nvars):
                for j in range(basis.var_ncoefs(i)):
                    coefs = np.zeros(basis.var_ncoefs(i))
                    coefs[j] = 1
                    np.testing.assert_array_almost_equal(
                        basis.eval(coefs, timepts, var=i),
                        basis.eval_deriv(j, 0, timepts, var=i))

            # Evaluate multi-variable output
            offset = 0
            for i in range(basis.nvars):
                for j in range(basis.var_ncoefs(i)):
                    coefs = np.zeros(basis.N)
                    coefs[offset] = 1
                    np.testing.assert_array_almost_equal(
                        basis.eval(coefs, timepts)[i],
                        basis.eval_deriv(j, 0, timepts, var=i))
                    offset += 1

    def test_flatsys_factory_function(self, vehicle_flat):
        # Basic flat system
        flatsys = fs.flatsys(
            vehicle_flat.forward, vehicle_flat.reverse,
            inputs=vehicle_flat.ninputs, outputs=vehicle_flat.ninputs,
            states=vehicle_flat.nstates)
        assert isinstance(flatsys, fs.FlatSystem)

        # Flat system with update function
        flatsys = fs.flatsys(
            vehicle_flat.forward, vehicle_flat.reverse, vehicle_flat.updfcn,
            inputs=vehicle_flat.ninputs, outputs=vehicle_flat.ninputs,
            states=vehicle_flat.nstates)
        assert isinstance(flatsys, fs.FlatSystem)
        assert flatsys.updfcn == vehicle_flat.updfcn

        # Flat system with update and output functions
        flatsys = fs.flatsys(
            vehicle_flat.forward, vehicle_flat.reverse, vehicle_flat.updfcn,
            vehicle_flat.outfcn, inputs=vehicle_flat.ninputs,
            outputs=vehicle_flat.ninputs, states=vehicle_flat.nstates)
        assert isinstance(flatsys, fs.FlatSystem)
        assert flatsys.updfcn == vehicle_flat.updfcn
        assert flatsys.outfcn == vehicle_flat.outfcn

        # Flat system with update and output functions via keywords
        flatsys = fs.flatsys(
            vehicle_flat.forward, vehicle_flat.reverse,
            updfcn=vehicle_flat.updfcn, outfcn=vehicle_flat.outfcn,
            inputs=vehicle_flat.ninputs, outputs=vehicle_flat.ninputs,
            states=vehicle_flat.nstates)
        assert isinstance(flatsys, fs.FlatSystem)
        assert flatsys.updfcn == vehicle_flat.updfcn
        assert flatsys.outfcn == vehicle_flat.outfcn

        # Linear flat system
        sys = ct.ss([[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], 0)
        flatsys = fs.flatsys(sys)
        assert isinstance(flatsys, fs.FlatSystem)
        assert isinstance(flatsys, ct.StateSpace)

        # Incorrect arguments
        with pytest.raises(TypeError, match="incorrect number or type"):
            flatsys = fs.flatsys(vehicle_flat.forward)

        with pytest.raises(TypeError, match="incorrect number or type"):
            flatsys = fs.flatsys(1, 2, 3, 4, 5)
