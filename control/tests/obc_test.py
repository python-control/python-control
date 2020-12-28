#!/usr/bin/env python
#
# obc_test.py - tests for optimization based control
# RMM, 17 Apr 2019
#
# This test suite checks the functionality for optimization based control.

import unittest
import warnings
import numpy as np
import scipy as sp
import control as ct
import control.pwa as pwa
import polytope as pc

class TestOBC(unittest.TestCase):
    def setUp(self):
        # Turn off numpy matrix warnings
        import warnings
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)

    def test_finite_horizon_mpc_simple(self):
        # Define a linear system with constraints
        # Source: https://www.mpt3.org/UI/RegulationProblem

        # LTI prediction model
        model = pwa.ConstrainedAffineSystem(
            A = [[1, 1], [0, 1]], B = [[1], [0.5]], C = np.eye(2))

        # state and input constraints
        model.add_state_constraints(pc.box2poly([[-5, 5], [-5, 5]]))
        model.add_input_constraints(pc.box2poly([-1, 1]))

        # quadratic state and input penalty
        Q = [[1, 0], [0, 1]]
        R = [[1]]

        # Create a model predictive controller system
        mpc = obc.ModelPredictiveController(
            model,
            obc.QuadraticCost(Q, R),
            horizon=5)

        # Optimal control input for a given value of the initial state:
        x0 = [4, 0]
        u = mpc.compute_input(x0)
        self.assertEqual(u, -1)

        # retrieve the full open-loop predictions
        (u_openloop, feasible, openloop) = mpc.compute_trajectory(x0)
        np.testing.assert_array_almost_equal(
            u_openloop, [-1, -1, 0.1393, 0.3361, -5.2042e-16])

        # convert it to an explicit form
        mpc_explicit = mpc.explicit();

        # Test explicit controller 
        (u_explicit, feasible, openloop) = mpc_explicit(x0)
        np.testing.assert_array_almost_equal(u_openloop, u_explicit)

    @unittest.skipIf(True, "Not yet implemented.")
    def test_finite_horizon_mpc_oscillator(self):
        # oscillator model defined in 2D
        # Source: https://www.mpt3.org/UI/RegulationProblem
        A = [[0.5403, -0.8415], [0.8415, 0.5403]]
        B = [[-0.4597], [0.8415]]
        C = [[1, 0]]
        D = [[0]]

        # Linear discrete-time model with sample time 1
        sys = ss(A, B, C, D, 1);
        model = LTISystem(sys);

        # state and input constraints
        model.add_state_constraints(pc.box2poly([[-10, 10]]))
        model.add_input_constraints(pc.box2poly([-1, 1]))

        # Include weights on states/inputs
        model.x.penalty = QuadFunction(np.eye(2));
        model.u.penalty = QuadFunction(1);

        # Compute terminal set
        Tset = model.LQRSet;

        # Compute terminal weight
        PN = model.LQRPenalty;

        # Add terminal set and terminal penalty
        # model.x.with('terminalSet');
        model.x.terminalSet = Tset;
        # model.x.with('terminalPenalty');
        model.x.terminalPenalty = PN;

        # Formulate finite horizon MPC problem
        ctrl = MPCController(model,5);

        # Add tests to make sure everything works

    # TODO: move this to examples?
    @unittest.skipIf(True, "Not yet implemented.")
    def test_finite_horizon_mpc_oscillator(self):
        # model of an aircraft discretized with 0.2s sampling time
        # Source: https://www.mpt3.org/UI/RegulationProblem
        A = [[0.99, 0.01, 0.18, -0.09,   0],
             [   0, 0.94,    0,  0.29,   0],
             [   0, 0.14, 0.81,  -0.9,   0]
             [   0, -0.2,    0,  0.95,   0],
             [   0, 0.09,    0,     0, 0.9]]
        B = [[ 0.01, -0.02],
             [-0.14,     0],
             [ 0.05,  -0.2],
             [ 0.02,     0],
             [-0.01, 0]]
        C = [[0, 1, 0, 0, -1],
             [0, 0, 1, 0,  0],
             [0, 0, 0, 1,  0],
             [1, 0, 0, 0,  0]]
        model = LTISystem('A', A, 'B', B, 'C', C, 'Ts', 0.2);

        # compute the new steady state values for a particular value
        # of the input
        us = [0.8, -0.3];
        # ys = C*( (eye(5)-A)\B*us );

        # computed values will be used as references for the desired
        # steady state which can be added using "reference" filter
        # model.u.with('reference');
        model.u.reference = us;
        # model.y.with('reference');
        model.y.reference = ys;

        # provide constraints and penalties on the system signals
        model.u.min = [-5, -6];
        model.u.max = [5, 6];

        # penalties on outputs and inputs are provided as quadratic functions
        model.u.penalty = QuadFunction( diag([3, 2]) );
        model.y.penalty = QuadFunction( diag([10, 10, 10, 10]) );

        # online MPC controller object is constructed with a horizon 6
        ctrl = MPCController(model, 6)

        # loop = ClosedLoop(ctrl, model);
        x0 = [0, 0, 0, 0, 0];
        Nsim = 30;
        data = loop.simulate(x0, Nsim);

        # Plot the results
        subplot(2,1,1)
        plot(np.range(Nsim), data.Y);
        plot(np.range(Nsim), ys*ones(1, Nsim), 'k--')
        title('outputs')
        subplot(2,1,2)
        plot(np.range(Nsim), data.U);
        plot(np.range(Nsim), us*ones(1, Nsim), 'k--')
        title('inputs')
        

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)


if __name__ == '__main__':
    unittest.main()
