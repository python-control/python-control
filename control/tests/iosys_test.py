#!/usr/bin/env python
#
# iosys_test.py - test input/output system oeprations
# RMM, 17 Apr 2019
#
# This test suite checks to make sure that basic input/output class
# operations are working.  It doesn't do exhaustive testing of
# operations on input/output systems.  Separate unit tests should be
# created for that purpose.

from __future__ import print_function
import unittest
import warnings
import numpy as np
import scipy as sp
import control as ct
import control.iosys as ios
from distutils.version import StrictVersion

class TestIOSys(unittest.TestCase):
    def setUp(self):
        # Turn off numpy matrix warnings
        import warnings
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)

        # Create a single input/single output linear system
        self.siso_linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]])

        # Create a multi input/multi output linear system
        self.mimo_linsys1 = ct.StateSpace(
            [[-1, 1], [0, -2]], [[1, 0], [0, 1]],
            [[1, 0], [0, 1]], np.zeros((2,2)))

        # Create a multi input/multi output linear system
        self.mimo_linsys2 = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0, 1], [1, 0]],
            [[1, 0], [0, 1]], np.zeros((2,2)))

        # Create simulation parameters
        self.T = np.linspace(0, 10, 100)
        self.U = np.sin(self.T)
        self.X0 = [0, 0]

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_linear_iosys(self):
        # Create an input/output system from the linear system
        linsys = self.siso_linsys
        iosys = ios.LinearIOSystem(linsys)

        # Make sure that the right hand side matches linear system
        for x, u in (([0, 0], 0), ([1, 0], 0), ([0, 1], 0), ([0, 0], 1)):
            np.testing.assert_array_almost_equal(
                np.reshape(iosys._rhs(0, x, u), (-1,1)),
                np.dot(linsys.A, np.reshape(x, (-1, 1))) + np.dot(linsys.B, u))

        # Make sure that simulations also line up
        T, U, X0 = self.T, self.U, self.X0
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_tf2io(self):
        # Create a transfer function from the state space system
        linsys = self.siso_linsys
        tfsys = ct.ss2tf(linsys)
        iosys = ct.tf2io(tfsys)

        # Verify correctness via simulation
        T, U, X0 = self.T, self.U, self.X0
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    def test_ss2io(self):
        # Create an input/output system from the linear system
        linsys = self.siso_linsys
        iosys = ct.ss2io(linsys)
        np.testing.assert_array_equal(linsys.A, iosys.A)
        np.testing.assert_array_equal(linsys.B, iosys.B)
        np.testing.assert_array_equal(linsys.C, iosys.C)
        np.testing.assert_array_equal(linsys.D, iosys.D)

        # Try adding names to things
        iosys_named = ct.ss2io(linsys, inputs='u', outputs='y',
                               states=['x1', 'x2'], name='iosys_named')
        self.assertEqual(iosys_named.find_input('u'), 0)
        self.assertEqual(iosys_named.find_input('x'), None)
        self.assertEqual(iosys_named.find_output('y'), 0)
        self.assertEqual(iosys_named.find_output('u'), None)
        self.assertEqual(iosys_named.find_state('x0'), None)
        self.assertEqual(iosys_named.find_state('x1'), 0)
        self.assertEqual(iosys_named.find_state('x2'), 1)
        np.testing.assert_array_equal(linsys.A, iosys_named.A)
        np.testing.assert_array_equal(linsys.B, iosys_named.B)
        np.testing.assert_array_equal(linsys.C, iosys_named.C)
        np.testing.assert_array_equal(linsys.D, iosys_named.D)

    # Make sure unspecified inputs/outputs/states are handled properly
    def test_iosys_unspecified(self):
        # System with unspecified inputs and outputs
        sys = ios.NonlinearIOSystem(secord_update, secord_output)
        np.testing.assert_raises(TypeError, sys.__mul__, sys)

    # Make sure we can print various types of I/O systems
    def test_iosys_print(self):
        # Send the output to /dev/null
        import os
        f = open(os.devnull,"w")

        # Simple I/O system
        iosys = ct.ss2io(self.siso_linsys)
        print(iosys, file=f)

        # I/O system without ninputs, noutputs
        ios_unspecified = ios.NonlinearIOSystem(secord_update, secord_output)
        print(ios_unspecified, file=f)

        # I/O system with derived inputs and outputs
        ios_linearized = ios.linearize(ios_unspecified, [0, 0], [0])
        print(ios_linearized, file=f)

        f.close()

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_nonlinear_iosys(self):
        # Create a simple nonlinear I/O system
        nlsys = ios.NonlinearIOSystem(predprey)
        T = self.T

        # Start by simulating from an equilibrium point
        X0 = [0, 0]
        ios_t, ios_y = ios.input_output_response(nlsys, T, 0, X0)
        np.testing.assert_array_almost_equal(ios_y, np.zeros(np.shape(ios_y)))

        # Now simulate from a nonzero point
        X0 = [0.5, 0.5]
        ios_t, ios_y = ios.input_output_response(nlsys, T, 0, X0)

        #
        # Simulate a linear function as a nonlinear function and compare
        #
        # Create a single input/single output linear system
        linsys = self.siso_linsys

        # Create a nonlinear system with the same dynamics
        nlupd = lambda t, x, u, params: \
            np.reshape(np.dot(linsys.A, np.reshape(x, (-1, 1))) + np.dot(linsys.B, u), (-1,))
        nlout = lambda t, x, u, params: \
            np.reshape(np.dot(linsys.C, np.reshape(x, (-1, 1))) + np.dot(linsys.D, u), (-1,))
        nlsys = ios.NonlinearIOSystem(nlupd, nlout)

        # Make sure that simulations also line up
        T, U, X0 = self.T, self.U, self.X0
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U, X0)
        ios_t, ios_y = ios.input_output_response(nlsys, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    def test_linearize(self):
        # Create a single input/single output linear system
        linsys = self.siso_linsys
        iosys = ios.LinearIOSystem(linsys)

        # Linearize it and make sure we get back what we started with
        linearized = iosys.linearize([0, 0], 0)
        np.testing.assert_array_almost_equal(linsys.A, linearized.A)
        np.testing.assert_array_almost_equal(linsys.B, linearized.B)
        np.testing.assert_array_almost_equal(linsys.C, linearized.C)
        np.testing.assert_array_almost_equal(linsys.D, linearized.D)

        # Create a simple nonlinear system to check (kinematic car)
        def kincar_update(t, x, u, params):
            return np.array([np.cos(x[2]) * u[0], np.sin(x[2]) * u[0], u[1]])
        def kincar_output(t, x, u, params):
            return np.array([x[0], x[1]])
        iosys = ios.NonlinearIOSystem(kincar_update, kincar_output)
        linearized = iosys.linearize([0, 0, 0], [0, 0])
        np.testing.assert_array_almost_equal(linearized.A, np.zeros((3,3)))
        np.testing.assert_array_almost_equal(
            linearized.B, [[1, 0], [0, 0], [0, 1]])
        np.testing.assert_array_almost_equal(
            linearized.C, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_almost_equal(linearized.D, np.zeros((2,2)))

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_connect(self):
        # Define a couple of (linear) systems to interconnection
        linsys1 = self.siso_linsys
        iosys1 = ios.LinearIOSystem(linsys1)
        linsys2 = self.siso_linsys
        iosys2 = ios.LinearIOSystem(linsys2)

        # Connect systems in different ways and compare to StateSpace
        linsys_series = linsys2 * linsys1
        iosys_series = ios.InterconnectedSystem(
            (iosys1, iosys2),   # systems
            ((1, 0),),          # interconnection (series)
            0,                  # input = first system
            1                   # output = second system
        )

        # Run a simulation and compare to linear response
        T, U = self.T, self.U
        X0 = np.concatenate((self.X0, self.X0))
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        lti_t, lti_y, lti_x = ct.forced_response(linsys_series, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

        # Connect systems with different timebases
        linsys2c = self.siso_linsys
        linsys2c.dt = 0         # Reset the timebase
        iosys2c = ios.LinearIOSystem(linsys2c)
        iosys_series = ios.InterconnectedSystem(
            (iosys1, iosys2c),   # systems
            ((1, 0),),          # interconnection (series)
            0,                  # input = first system
            1                   # output = second system
        )
        self.assertTrue(ct.isctime(iosys_series, strict=True))
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_series, T, U, X0, return_x=True)
        lti_t, lti_y, lti_x = ct.forced_response(linsys_series, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

        # Feedback interconnection
        linsys_feedback = ct.feedback(linsys1, linsys2)
        iosys_feedback = ios.InterconnectedSystem(
            (iosys1, iosys2),   # systems
            ((1, 0),            # input of sys2 = output of sys1
             (0, (1, 0, -1))),  # input of sys1 = -output of sys2
            0,                  # input = first system
            0                   # output = first system
        )
        ios_t, ios_y, ios_x = ios.input_output_response(
            iosys_feedback, T, U, X0, return_x=True)
        lti_t, lti_y, lti_x = ct.forced_response(linsys_feedback, T, U, X0)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_static_nonlinearity(self):
        # Linear dynamical system
        linsys = self.siso_linsys
        ioslin = ios.LinearIOSystem(linsys)

        # Nonlinear saturation
        sat = lambda u: u if abs(u) < 1 else np.sign(u)
        sat_output = lambda t, x, u, params: sat(u)
        nlsat =  ios.NonlinearIOSystem(None, sat_output, inputs=1, outputs=1)

        # Set up parameters for simulation
        T, U, X0 = self.T, 2 * self.U, self.X0
        Usat = np.vectorize(sat)(U)

        # Make sure saturation works properly by comparing linear system with
        # saturated input to nonlinear system with saturation composition
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, Usat, X0)
        ios_t, ios_y, ios_x = ios.input_output_response(
            ioslin * nlsat, T, U, X0, return_x=True)
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_array_almost_equal(lti_y, ios_y, decimal=2)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_algebraic_loop(self):
        # Create some linear and nonlinear systems to play with
        linsys = self.siso_linsys
        lnios = ios.LinearIOSystem(linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        nlios1 = nlios.copy()
        nlios2 = nlios.copy()

        # Set up parameters for simulation
        T, U, X0 = self.T, self.U, self.X0

        # Single nonlinear system - no states
        ios_t, ios_y = ios.input_output_response(nlios, T, U)
        np.testing.assert_array_almost_equal(ios_y, U*U, decimal=3)

        # Composed nonlinear system (series)
        ios_t, ios_y = ios.input_output_response(nlios1 * nlios2, T, U)
        np.testing.assert_array_almost_equal(ios_y, U**4, decimal=3)

        # Composed nonlinear system (parallel)
        ios_t, ios_y = ios.input_output_response(nlios1 + nlios2, T, U)
        np.testing.assert_array_almost_equal(ios_y, 2*U**2, decimal=3)

        # Nonlinear system composed with LTI system (series) -- with states
        ios_t, ios_y = ios.input_output_response(
            nlios * lnios * nlios, T, U, X0)
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, lti_y*lti_y, decimal=3)

        # Nonlinear system in feeback loop with LTI system
        iosys = ios.InterconnectedSystem(
            (lnios, nlios),         # linear system w/ nonlinear feedback
            ((1,),                  # feedback interconnection (sig to 0)
             (0, (1, 0, -1))),
            0,                      # input to linear system
            0                       # output from linear system
        )
        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        # No easy way to test the result

        # Algebraic loop from static nonlinear system in feedback
        # (error will be due to no states)
        iosys = ios.InterconnectedSystem(
            (nlios1, nlios2),       # two copies of a static nonlinear system
            ((0, 1),                # feedback interconnection
             (1, (0, 0, -1))),
            0, 0
        )
        args = (iosys, T, U)
        self.assertRaises(RuntimeError, ios.input_output_response, *args)

        # Algebraic loop due to feedthrough term
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[1]])
        lnios = ios.LinearIOSystem(linsys)
        iosys = ios.InterconnectedSystem(
            (nlios, lnios),         # linear system w/ nonlinear feedback
            ((0, 1),                # feedback interconnection
             (1, (0, 0, -1))),
            0, 0
        )
        args = (iosys, T, U, X0)
        # ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        self.assertRaises(RuntimeError, ios.input_output_response, *args)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_summer(self):
        # Construct a MIMO system for testing
        linsys = self.mimo_linsys1
        linio = ios.LinearIOSystem(linsys)

        linsys_parallel = linsys + linsys
        iosys_parallel = linio + linio

        # Set up parameters for simulation
        T = self.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        lin_t, lin_y, lin_x = ct.forced_response(linsys_parallel, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_parallel, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_rmul(self):
        # Test right multiplication
        # TODO: replace with better tests when conversions are implemented

        # Set up parameters for simulation
        T, U, X0 = self.T, self.U, self.X0

        # Linear system with input and output nonlinearities
        # Also creates a nested interconnected system
        ioslin = ios.LinearIOSystem(self.siso_linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        sys1 = nlios * ioslin
        sys2 = ios.InputOutputSystem.__rmul__(nlios, sys1)

        # Make sure we got the right thing (via simulation comparison)
        ios_t, ios_y = ios.input_output_response(sys2, T, U, X0)
        lti_t, lti_y, lti_x = ct.forced_response(ioslin, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, lti_y*lti_y, decimal=3)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_neg(self):
        """Test negation of a system"""

        # Set up parameters for simulation
        T, U, X0 = self.T, self.U, self.X0

        # Static nonlinear system
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1)
        ios_t, ios_y = ios.input_output_response(-nlios, T, U)
        np.testing.assert_array_almost_equal(ios_y, -U*U, decimal=3)

        # Linear system with input nonlinearity
        # Also creates a nested interconnected system
        ioslin = ios.LinearIOSystem(self.siso_linsys)
        sys = (ioslin) * (-nlios)

        # Make sure we got the right thing (via simulation comparison)
        ios_t, ios_y = ios.input_output_response(sys, T, U, X0)
        lti_t, lti_y, lti_x = ct.forced_response(ioslin, T, U*U, X0)
        np.testing.assert_array_almost_equal(ios_y, -lti_y, decimal=3)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_feedback(self):
        # Set up parameters for simulation
        T, U, X0 = self.T, self.U, self.X0

        # Linear system with constant feedback (via "nonlinear" mapping)
        ioslin = ios.LinearIOSystem(self.siso_linsys)
        nlios =  ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u, inputs=1, outputs=1)
        iosys = ct.feedback(ioslin, nlios)
        linsys = ct.feedback(self.siso_linsys, 1)

        ios_t, ios_y = ios.input_output_response(iosys, T, U, X0)
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_y, lti_y,atol=0.002,rtol=0.)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_bdalg_functions(self):
        """Test block diagram functions algebra on I/O systems"""
        # Set up parameters for simulation
        T = self.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        # Set up systems to be composed
        linsys1 = self.mimo_linsys1
        linio1 = ios.LinearIOSystem(linsys1)
        linsys2 = self.mimo_linsys2
        linio2 = ios.LinearIOSystem(linsys2)

        # Series interconnection
        linsys_series = ct.series(linsys1, linsys2)
        iosys_series = ct.series(linio1, linio2)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_series, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_series, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Make sure that systems don't commute
        linsys_series = ct.series(linsys2, linsys1)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_series, T, U, X0)
        self.assertFalse((np.abs(lin_y - ios_y) < 1e-3).all())

        # Parallel interconnection
        linsys_parallel = ct.parallel(linsys1, linsys2)
        iosys_parallel = ct.parallel(linio1, linio2)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_parallel, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_parallel, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Negation
        linsys_negate = ct.negate(linsys1)
        iosys_negate = ct.negate(linio1)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_negate, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_negate, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Feedback interconnection
        linsys_feedback = ct.feedback(linsys1, linsys2)
        iosys_feedback = ct.feedback(linio1, linio2)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_feedback, T, U, X0)
        ios_t, ios_y = ios.input_output_response(iosys_feedback, T, U, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_nonsquare_bdalg(self):
        # Set up parameters for simulation
        T = self.T
        U2 = [np.sin(T), np.cos(T)]
        U3 = [np.sin(T), np.cos(T), T]
        X0 = 0

        # Set up systems to be composed
        linsys_2i3o = ct.StateSpace(
            [[-1, 1, 0], [0, -2, 0], [0, 0, -3]], [[1, 0], [0, 1], [1, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.zeros((3, 2)))
        iosys_2i3o = ios.LinearIOSystem(linsys_2i3o)

        linsys_3i2o = ct.StateSpace(
            [[-1, 1, 0], [0, -2, 0], [0, 0, -3]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 1], [0, 1, -1]], np.zeros((2, 3)))
        iosys_3i2o = ios.LinearIOSystem(linsys_3i2o)

        # Multiplication
        linsys_multiply = linsys_3i2o * linsys_2i3o
        iosys_multiply = iosys_3i2o * iosys_2i3o
        lin_t, lin_y, lin_x = ct.forced_response(linsys_multiply, T, U2, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U2, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        linsys_multiply = linsys_2i3o * linsys_3i2o
        iosys_multiply = iosys_2i3o * iosys_3i2o
        lin_t, lin_y, lin_x = ct.forced_response(linsys_multiply, T, U3, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Right multiplication
        # TODO: add real tests once conversion from other types is supported
        iosys_multiply = ios.InputOutputSystem.__rmul__(iosys_3i2o, iosys_2i3o)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Feedback
        linsys_multiply = ct.feedback(linsys_3i2o, linsys_2i3o)
        iosys_multiply = iosys_3i2o.feedback(iosys_2i3o)
        lin_t, lin_y, lin_x = ct.forced_response(linsys_multiply, T, U3, X0)
        ios_t, ios_y = ios.input_output_response(iosys_multiply, T, U3, X0)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Mismatch should generate exception
        args = (iosys_3i2o, iosys_3i2o)
        self.assertRaises(ValueError, ct.series, *args)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_discrete(self):
        """Test discrete time functionality"""
        # Create some linear and nonlinear systems to play with
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[1, 0]], [[0]], True)
        lnios = ios.LinearIOSystem(linsys)

        # Set up parameters for simulation
        T, U, X0 = self.T, self.U, self.X0

        # Simulate and compare to LTI output
        ios_t, ios_y = ios.input_output_response(lnios, T, U, X0)
        lin_t, lin_y, lin_x = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_t, lin_t,atol=0.002,rtol=0.)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

        # Test MIMO system, converted to discrete time
        linsys = ct.StateSpace(self.mimo_linsys1)
        linsys.dt = self.T[1] - self.T[0]
        lnios = ios.LinearIOSystem(linsys)

        # Set up parameters for simulation
        T = self.T
        U = [np.sin(T), np.cos(T)]
        X0 = 0

        # Simulate and compare to LTI output
        ios_t, ios_y = ios.input_output_response(lnios, T, U, X0)
        lin_t, lin_y, lin_x = ct.forced_response(linsys, T, U, X0)
        np.testing.assert_allclose(ios_t, lin_t,atol=0.002,rtol=0.)
        np.testing.assert_allclose(ios_y, lin_y,atol=0.002,rtol=0.)

    def test_find_eqpts(self):
        """Test find_eqpt function"""
        # Simple equilibrium point with no inputs
        nlsys = ios.NonlinearIOSystem(predprey)
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [1.6, 1.2], None, return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(xeq, [1.64705879, 1.17923874])
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((2,)))

        # Ducted fan dynamics with output = velocity
        nlsys = ios.NonlinearIOSystem(pvtol, lambda t, x, u, params: x[0:2])

        # Make sure the origin is a fixed point
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0, 4*9.8], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)))
        np.testing.assert_array_almost_equal(xeq, [0, 0, 0, 0])

        # Use a small lateral force to cause motion
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Equilibrium point with fixed output
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0.1, 0.1], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Specify outputs to constrain (replicate previous)
        xeq, ueq, result = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], y0=[0.1, 0.1],
            iy = [0, 1], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Specify inputs to constrain (replicate previous), w/ no result
        xeq, ueq = ios.find_eqpt(
            nlsys, [0, 0, 0, 0], [0.01, 4*9.8], y0=[0.1, 0.1], iu = [])
        np.testing.assert_array_almost_equal(
            nlsys._out(0, xeq, ueq), [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys._rhs(0, xeq, ueq), np.zeros((4,)), decimal=5)

        # Now solve the problem with the original PVTOL variables
        # Constrain the output angle and x velocity
        nlsys_full = ios.NonlinearIOSystem(pvtol_full, None)
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0.1, 0.1, 0, 0], iy = [2, 3],
            idx=[2, 3, 4, 5], ix=[0, 1], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[[2, 3]], [0.1, 0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-4:], np.zeros((4,)), decimal=5)

        # Fix one input and vary the other
        nlsys_full = ios.NonlinearIOSystem(pvtol_full, None)
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0.1, 0.1, 0, 0], iy=[3], iu=[1],
            idx=[2, 3, 4, 5], ix=[0, 1], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_almost_equal(ueq[1], 4*9.8, decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[[3]], [0.1], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-4:], np.zeros((4,)), decimal=5)

        # PVTOL with output = y velocity
        xeq, ueq, result = ios.find_eqpt(
            nlsys_full, [0, 0, 0, 0.1, 0, 0], [0.01, 4*9.8],
            y0=[0, 0, 0, 0.1, 0, 0], iy=[3],
            dx0=[0.1, 0, 0, 0, 0, 0], idx=[1, 2, 3, 4, 5],
            ix=[0, 1], return_result=True)
        self.assertTrue(result.success)
        np.testing.assert_array_almost_equal(
            nlsys_full._out(0, xeq, ueq)[-3:], [0.1, 0, 0], decimal=5)
        np.testing.assert_array_almost_equal(
            nlsys_full._rhs(0, xeq, ueq)[-5:], np.zeros((5,)), decimal=5)

        # Unobservable system
        linsys = ct.StateSpace(
            [[-1, 1], [0, -2]], [[0], [1]], [[0, 0]], [[0]])
        lnios = ios.LinearIOSystem(linsys)

        # If result is returned, user has to check
        xeq, ueq, result = ios.find_eqpt(
            lnios, [0, 0], [0], y0=[1], return_result=True)
        self.assertFalse(result.success)

        # If result is not returned, find_eqpt should return None
        xeq, ueq = ios.find_eqpt(lnios, [0, 0], [0], y0=[1])
        self.assertEqual(xeq, None)
        self.assertEqual(ueq, None)

    @unittest.skipIf(StrictVersion(sp.__version__) < "1.0",
                     "requires SciPy 1.0 or greater")
    def test_params(self):
        # Start with the default set of parameters
        ios_secord_default = ios.NonlinearIOSystem(
            secord_update, secord_output, inputs=1, outputs=1, states=2)
        lin_secord_default = ios.linearize(ios_secord_default, [0, 0], [0])
        w_default, v_default = np.linalg.eig(lin_secord_default.A)

        # New copy, with modified parameters
        ios_secord_update = ios.NonlinearIOSystem(
            secord_update, secord_output, inputs=1, outputs=1, states=2,
            params={'omega0':2, 'zeta':0})

        # Make sure the default parameters haven't changed
        lin_secord_check = ios.linearize(ios_secord_default, [0, 0], [0])
        w, v = np.linalg.eig(lin_secord_check.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort(w_default))

        # Make sure updated system parameters got set correctly
        lin_secord_update = ios.linearize(ios_secord_update, [0, 0], [0])
        w, v = np.linalg.eig(lin_secord_update.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([2j, -2j]))

        # Change the parameters of the default sys just for the linearization
        lin_secord_local = ios.linearize(ios_secord_default, [0, 0], [0],
                                          params={'zeta':0})
        w, v = np.linalg.eig(lin_secord_local.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([1j, -1j]))

        # Change the parameters of the updated sys just for the linearization
        lin_secord_local = ios.linearize(ios_secord_update, [0, 0], [0],
                                          params={'zeta':0, 'omega0':3})
        w, v = np.linalg.eig(lin_secord_local.A)
        np.testing.assert_array_almost_equal(np.sort(w), np.sort([3j, -3j]))

        # Make sure that changes propagate through interconnections
        ios_series_default_local = ios_secord_default * ios_secord_update
        lin_series_default_local = ios.linearize(
            ios_series_default_local, [0, 0, 0, 0], [0])
        w, v = np.linalg.eig(lin_series_default_local.A)
        np.testing.assert_array_almost_equal(
            np.sort(w), np.sort(np.concatenate((w_default, [2j, -2j]))))

        # Show that we can change the parameters at linearization
        lin_series_override = ios.linearize(
            ios_series_default_local, [0, 0, 0, 0], [0],
            params={'zeta':0, 'omega0':4})
        w, v = np.linalg.eig(lin_series_override.A)
        np.testing.assert_array_almost_equal(w, [4j, -4j, 4j, -4j])

        # Check for warning if we try to set params for LinearIOSystem
        linsys = self.siso_linsys
        iosys = ios.LinearIOSystem(linsys)
        T, U, X0 = self.T, self.U, self.X0
        lti_t, lti_y, lti_x = ct.forced_response(linsys, T, U, X0)
        with warnings.catch_warnings(record=True) as warnval:
            # Turn off deprecation warnings
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=PendingDeprecationWarning)

            # Trigger a warning
            ios_t, ios_y = ios.input_output_response(
                iosys, T, U, X0, params={'something':0})

            # Verify that we got a warning
            self.assertEqual(len(warnval), 1)
            self.assertTrue(issubclass(warnval[-1].category, UserWarning))
            self.assertTrue("LinearIOSystem" in str(warnval[-1].message))
            self.assertTrue("ignored" in str(warnval[-1].message))

        # Check to make sure results are OK
        np.testing.assert_array_almost_equal(lti_t, ios_t)
        np.testing.assert_allclose(lti_y, ios_y,atol=0.002,rtol=0.)

    def test_named_signals(self):
        sys1 = ios.NonlinearIOSystem(
            updfcn = lambda t, x, u, params: np.array(
                np.dot(self.mimo_linsys1.A, np.reshape(x, (-1, 1))) \
                + np.dot(self.mimo_linsys1.B, np.reshape(u, (-1, 1)))
            ).reshape(-1,),
            outfcn = lambda t, x, u, params: np.array(
                np.dot(self.mimo_linsys1.C, np.reshape(x, (-1, 1))) \
                + np.dot(self.mimo_linsys1.D, np.reshape(u, (-1, 1)))
            ).reshape(-1,),
            inputs = ('u[0]', 'u[1]'),
            outputs = ('y[0]', 'y[1]'),
            states = self.mimo_linsys1.states,
            name = 'sys1')
        sys2 = ios.LinearIOSystem(self.mimo_linsys2,
            inputs = ('u[0]', 'u[1]'),
            outputs = ('y[0]', 'y[1]'),
            name = 'sys2')

        # Series interconnection (sys1 * sys2) using __mul__
        ios_mul = sys1 * sys2
        ss_series = self.mimo_linsys1 * self.mimo_linsys2
        lin_series = ct.linearize(ios_mul, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Series interconnection (sys1 * sys2) using series
        ios_series = ct.series(sys2, sys1)
        ss_series = ct.series(self.mimo_linsys2, self.mimo_linsys1)
        lin_series = ct.linearize(ios_series, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Series interconnection (sys1 * sys2) using named + mixed signals
        ios_connect = ios.InterconnectedSystem(
            (sys2, sys1),
            connections=(
                (('sys1', 'u[0]'), 'sys2.y[0]'),
                ('sys1.u[1]', 'sys2.y[1]')
            ),
            inplist=('sys2.u[0]', ('sys2', 1)),
            outlist=((1, 'y[0]'), 'sys1.y[1]')
        )
        lin_series = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_series.A, lin_series.A)
        np.testing.assert_array_almost_equal(ss_series.B, lin_series.B)
        np.testing.assert_array_almost_equal(ss_series.C, lin_series.C)
        np.testing.assert_array_almost_equal(ss_series.D, lin_series.D)

        # Make sure that we can use input signal names as system outputs
        ios_connect = ios.InterconnectedSystem(
            (sys1, sys2),
            connections=(
                ('sys2.u[0]', 'sys1.y[0]'), ('sys2.u[1]', 'sys1.y[1]'),
                ('sys1.u[0]', '-sys2.y[0]'), ('sys1.u[1]', '-sys2.y[1]')
            ),
            inplist=('sys1.u[0]', 'sys1.u[1]'),
            outlist=('sys2.u[0]', 'sys2.u[1]')  # = sys1.y[0], sys1.y[1]
        )
        ss_feedback = ct.feedback(self.mimo_linsys1, self.mimo_linsys2)
        lin_feedback = ct.linearize(ios_connect, 0, 0)
        np.testing.assert_array_almost_equal(ss_feedback.A, lin_feedback.A)
        np.testing.assert_array_almost_equal(ss_feedback.B, lin_feedback.B)
        np.testing.assert_array_almost_equal(ss_feedback.C, lin_feedback.C)
        np.testing.assert_array_almost_equal(ss_feedback.D, lin_feedback.D)

    def test_sys_naming_convention(self):
        """Enforce generic system names 'sys[i]' to be present when systems are created
        without explicit names."""

        ct.InputOutputSystem.idCounter = 0
        sys = ct.LinearIOSystem(self.mimo_linsys1)
        self.assertEquals(sys.name, "sys[0]")
        self.assertEquals(sys.copy().name, "copy of sys[0]")
        
        namedsys = ios.NonlinearIOSystem(
            updfcn = lambda t, x, u, params: x,
            outfcn = lambda t, x, u, params: u,
            inputs = ('u[0]', 'u[1]'),
            outputs = ('y[0]', 'y[1]'),
            states = self.mimo_linsys1.states,
            name = 'namedsys')
        unnamedsys1 = ct.NonlinearIOSystem(
            lambda t,x,u,params: x, inputs=2, outputs=2, states=2
        )
        unnamedsys2 = ct.NonlinearIOSystem(
            None, lambda t,x,u,params: u, inputs=2, outputs=2
        )
        self.assertEquals(unnamedsys2.name, "sys[2]")

        # Unnamed/unnamed connections
        uu_series = unnamedsys1 * unnamedsys2
        uu_parallel = unnamedsys1 + unnamedsys2
        u_neg = - unnamedsys1
        uu_feedback = unnamedsys2.feedback(unnamedsys1)
        uu_dup = unnamedsys1 * unnamedsys1.copy()
        uu_hierarchical = uu_series*unnamedsys1

        self.assertEquals(uu_series.name, "sys[3]")
        self.assertEquals(uu_parallel.name, "sys[4]")
        self.assertEquals(u_neg.name, "sys[5]")
        self.assertEquals(uu_feedback.name, "sys[6]")
        self.assertEquals(uu_dup.name, "sys[7]")
        self.assertEquals(uu_hierarchical.name, "sys[8]")

        # Unnamed/named connections
        un_series = unnamedsys1 * namedsys
        un_parallel = unnamedsys1 + namedsys
        un_feedback = unnamedsys2.feedback(namedsys)
        un_dup = unnamedsys1 * namedsys.copy()
        un_hierarchical = uu_series*unnamedsys1

        self.assertEquals(un_series.name, "sys[9]")
        self.assertEquals(un_parallel.name, "sys[10]")
        self.assertEquals(un_feedback.name, "sys[11]")
        self.assertEquals(un_dup.name, "sys[12]")
        self.assertEquals(un_hierarchical.name, "sys[13]")

        # Same system conflict
        with warnings.catch_warnings(record=True) as warnval:
            unnamedsys1 * unnamedsys1
            self.assertEqual(len(warnval), 1)

    def test_signals_naming_convention(self):
        """Enforce generic names to be present when systems are created
        without explicit signal names:
        input: 'u[i]'
        state: 'x[i]'
        output: 'y[i]'
        """
        ct.InputOutputSystem.idCounter = 0
        sys = ct.LinearIOSystem(self.mimo_linsys1)
        for statename in ["x[0]", "x[1]"]:
            self.assertTrue(statename in sys.state_index)
        for inputname in ["u[0]", "u[1]"]:
            self.assertTrue(inputname in sys.input_index)
        for outputname in ["y[0]", "y[1]"]:
            self.assertTrue(outputname in sys.output_index)
        self.assertEqual(len(sys.state_index), sys.nstates)
        self.assertEqual(len(sys.input_index), sys.ninputs)
        self.assertEqual(len(sys.output_index), sys.noutputs)

        namedsys = ios.NonlinearIOSystem(
            updfcn = lambda t, x, u, params: x,
            outfcn = lambda t, x, u, params: u,
            inputs = ('u0'),
            outputs = ('y0'),
            states = ('x0'),
            name = 'namedsys')
        unnamedsys = ct.NonlinearIOSystem(
            lambda t,x,u,params: x, inputs=1, outputs=1, states=1
        )
        self.assertTrue('u0' in namedsys.input_index)
        self.assertTrue('y0' in namedsys.output_index)
        self.assertTrue('x0' in namedsys.state_index)

        # Unnamed/named connections
        un_series = unnamedsys * namedsys
        un_parallel = unnamedsys + namedsys
        un_feedback = unnamedsys.feedback(namedsys)
        un_dup = unnamedsys * namedsys.copy()
        un_hierarchical = un_series*unnamedsys
        u_neg = - unnamedsys

        self.assertTrue("sys[1].x[0]" in un_series.state_index)
        self.assertTrue("namedsys.x0" in un_series.state_index)
        self.assertTrue("sys[1].x[0]" in un_parallel.state_index)
        self.assertTrue("namedsys.x0" in un_series.state_index)
        self.assertTrue("sys[1].x[0]" in un_feedback.state_index)
        self.assertTrue("namedsys.x0" in un_feedback.state_index)
        self.assertTrue("sys[1].x[0]" in un_dup.state_index)
        self.assertTrue("copy of namedsys.x0" in un_dup.state_index)
        self.assertTrue("sys[1].x[0]" in un_hierarchical.state_index)
        self.assertTrue("sys[2].sys[1].x[0]" in un_hierarchical.state_index)
        self.assertTrue("sys[1].x[0]" in u_neg.state_index)

        # Same system conflict
        with warnings.catch_warnings(record=True) as warnval:
            same_name_series = unnamedsys * unnamedsys
            self.assertEquals(len(warnval), 1)
            self.assertTrue("sys[1].x[0]" in same_name_series.state_index)
            self.assertTrue("copy of sys[1].x[0]" in same_name_series.state_index)

    def test_named_signals_linearize_inconsistent(self):
        """Mare sure that providing inputs or outputs not consistent with
           updfcn or outfcn fail
        """

        def updfcn(t, x, u, params):
            """2 inputs, 2 states"""
            return np.array(
                np.dot(self.mimo_linsys1.A, np.reshape(x, (-1, 1)))
                + np.dot(self.mimo_linsys1.B, np.reshape(u, (-1, 1)))
                ).reshape(-1,)

        def outfcn(t, x, u, params):
            """2 states, 2 outputs"""
            return np.array(
                    self.mimo_linsys1.C * np.reshape(x, (-1, 1))
                    + self.mimo_linsys1.D * np.reshape(u, (-1, 1))
                ).reshape(-1,)

        for inputs, outputs in [
                (('u[0]'), ('y[0]', 'y[1]')),  # not enough u
                (('u[0]', 'u[1]', 'u[toomuch]'), ('y[0]', 'y[1]')),
                (('u[0]', 'u[1]'), ('y[0]')),  # not enough y
                (('u[0]', 'u[1]'), ('y[0]', 'y[1]', 'y[toomuch]'))]:
            sys1 = ios.NonlinearIOSystem(updfcn=updfcn,
                                         outfcn=outfcn,
                                         inputs=inputs,
                                         outputs=outputs,
                                         states=self.mimo_linsys1.states,
                                         name='sys1')
            self.assertRaises(ValueError, sys1.linearize, [0, 0], [0, 0])

        sys2 = ios.NonlinearIOSystem(updfcn=updfcn,
                                     outfcn=outfcn,
                                     inputs=('u[0]', 'u[1]'),
                                     outputs=('y[0]', 'y[1]'),
                                     states=self.mimo_linsys1.states,
                                     name='sys1')
        for x0, u0 in [([0], [0, 0]),
                       ([0, 0, 0], [0, 0]),
                       ([0, 0], [0]),
                       ([0, 0], [0, 0, 0])]:
            self.assertRaises(ValueError, sys2.linearize, x0, u0)

    def test_lineariosys_statespace(self):
        """Make sure that a LinearIOSystem is also a StateSpace object"""
        iosys_siso = ct.LinearIOSystem(self.siso_linsys)
        self.assertTrue(isinstance(iosys_siso, ct.StateSpace))

        # Make sure that state space functions work for LinearIOSystems
        np.testing.assert_array_equal(
            iosys_siso.pole(), self.siso_linsys.pole())
        omega = np.logspace(.1, 10, 100)
        mag_io, phase_io, omega_io = iosys_siso.freqresp(omega)
        mag_ss, phase_ss, omega_ss = self.siso_linsys.freqresp(omega)
        np.testing.assert_array_equal(mag_io, mag_ss)
        np.testing.assert_array_equal(phase_io, phase_ss)
        np.testing.assert_array_equal(omega_io, omega_ss)

        # LinearIOSystem methods should override StateSpace methods
        io_mul = iosys_siso * iosys_siso
        self.assertTrue(isinstance(io_mul, ct.InputOutputSystem))

        # But also retain linear structure
        self.assertTrue(isinstance(io_mul, ct.StateSpace))

        # And make sure the systems match
        ss_series = self.siso_linsys * self.siso_linsys
        np.testing.assert_array_equal(io_mul.A, ss_series.A)
        np.testing.assert_array_equal(io_mul.B, ss_series.B)
        np.testing.assert_array_equal(io_mul.C, ss_series.C)
        np.testing.assert_array_equal(io_mul.D, ss_series.D)

        # Make sure that series does the same thing
        io_series = ct.series(iosys_siso, iosys_siso)
        self.assertTrue(isinstance(io_series, ct.InputOutputSystem))
        self.assertTrue(isinstance(io_series, ct.StateSpace))
        np.testing.assert_array_equal(io_series.A, ss_series.A)
        np.testing.assert_array_equal(io_series.B, ss_series.B)
        np.testing.assert_array_equal(io_series.C, ss_series.C)
        np.testing.assert_array_equal(io_series.D, ss_series.D)

        # Test out feedback as well
        io_feedback = ct.feedback(iosys_siso, iosys_siso)
        self.assertTrue(isinstance(io_series, ct.InputOutputSystem))

        # But also retain linear structure
        self.assertTrue(isinstance(io_series, ct.StateSpace))

        # And make sure the systems match
        ss_feedback = ct.feedback(self.siso_linsys, self.siso_linsys)
        np.testing.assert_array_equal(io_feedback.A, ss_feedback.A)
        np.testing.assert_array_equal(io_feedback.B, ss_feedback.B)
        np.testing.assert_array_equal(io_feedback.C, ss_feedback.C)
        np.testing.assert_array_equal(io_feedback.D, ss_feedback.D)

    def test_duplicates(self):
        nlios =  ios.NonlinearIOSystem(lambda t,x,u,params: x, \
                                       lambda t, x, u, params: u*u, \
                                       inputs=1, outputs=1, states=1, name="sys")

        # Turn off deprecation warnings
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)

        # Duplicate objects
        with warnings.catch_warnings(record=True) as warnval:
            # Trigger a warning
            ios_series = nlios * nlios

            # Verify that we got a warning
            self.assertEqual(len(warnval), 1)
            self.assertTrue(issubclass(warnval[-1].category, UserWarning))
            self.assertTrue("Duplicate object" in str(warnval[-1].message))

        # Nonduplicate objects
        nlios1 = nlios.copy()
        nlios2 = nlios.copy()
        with warnings.catch_warnings(record=True) as warnval:
            ios_series = nlios1 * nlios2
            self.assertEquals(len(warnval), 1)
            # when subsystems have the same name, duplicates are
            # renamed <subsysname_i>
            self.assertTrue("copy of sys_1.x[0]" in ios_series.state_index.keys())
            self.assertTrue("copy of sys.x[0]" in ios_series.state_index.keys())

        # Duplicate names
        iosys_siso = ct.LinearIOSystem(self.siso_linsys)
        nlios1 = ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1, name="sys")
        nlios2 = ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1, name="sys")
        with warnings.catch_warnings(record=True) as warnval:
            # Trigger a warning
            iosys = ct.InterconnectedSystem(
                (nlios1, iosys_siso, nlios2), inputs=0, outputs=0, states=0)

            # Verify that we got a warning
            self.assertEqual(len(warnval), 1)
            self.assertTrue(issubclass(warnval[-1].category, UserWarning))
            self.assertTrue("Duplicate name" in str(warnval[-1].message))

        # Same system, different names => everything should be OK
        nlios1 = ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1, name="nlios1")
        nlios2 = ios.NonlinearIOSystem(None, \
            lambda t, x, u, params: u*u, inputs=1, outputs=1, name="nlios2")
        with warnings.catch_warnings(record=True) as warnval:
            iosys = ct.InterconnectedSystem(
                (nlios1, iosys_siso, nlios2), inputs=0, outputs=0, states=0)
            self.assertEqual(len(warnval), 0)


# Predator prey dynamics
def predprey(t, x, u, params={}):
    r = params.get('r', 2)
    d = params.get('d', 0.7)
    b = params.get('b', 0.3)
    k = params.get('k', 10)
    a = params.get('a', 8)
    c = params.get('c', 4)

    # Dynamics for the system
    dx0 = r * x[0] * (1 - x[0]/k) - a * x[1] * x[0]/(c + x[0])
    dx1 = b * a * x[1] * x[0] / (c + x[0]) - d * x[1]

    return np.array([dx0, dx1])


# Reduced planar vertical takeoff and landing dynamics
def pvtol(t, x, u, params={}):
    from math import sin, cos
    m = params.get('m', 4.)      # kg, system mass
    J = params.get('J', 0.0475)  # kg m^2, system inertia
    r = params.get('r', 0.25)    # m, thrust offset
    g = params.get('g', 9.8)     # m/s, gravitational constant
    c = params.get('c', 0.05)    # N s/m, rotational damping
    l = params.get('c', 0.1)     # m, pivot location
    return np.array([
        x[3],
        -c/m * x[1] + 1/m * cos(x[0]) * u[0] - 1/m * sin(x[0]) * u[1],
        -g - c/m * x[2] + 1/m * sin(x[0]) * u[0] + 1/m * cos(x[0]) * u[1],
        -l/J * sin(x[0]) + r/J * u[0]
    ])

def pvtol_full(t, x, u, params={}):
    from math import sin, cos
    m = params.get('m', 4.)      # kg, system mass
    J = params.get('J', 0.0475)  # kg m^2, system inertia
    r = params.get('r', 0.25)    # m, thrust offset
    g = params.get('g', 9.8)     # m/s, gravitational constant
    c = params.get('c', 0.05)    # N s/m, rotational damping
    l = params.get('c', 0.1)     # m, pivot location
    return np.array([
        x[3], x[4], x[5],
        -c/m * x[3] + 1/m * cos(x[2]) * u[0] - 1/m * sin(x[2]) * u[1],
        -g - c/m * x[4] + 1/m * sin(x[2]) * u[0] + 1/m * cos(x[2]) * u[1],
        -l/J * sin(x[2]) + r/J * u[0]
    ])


# Second order system dynamics
def secord_update(t, x, u, params={}):
    omega0 = params.get('omega0', 1.)
    zeta = params.get('zeta', 0.5)
    u = np.array(u, ndmin=1)
    return np.array([
        x[1],
        -2 * zeta * omega0 * x[1] - omega0*omega0 * x[0] + u[0]
    ])
def secord_output(t, x, u, params={}):
    return np.array([x[0]])


if __name__ == '__main__':
    unittest.main()
