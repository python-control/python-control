#!/usr/bin/env python
#
# discrete_test.py - test discrete time classes
# RMM, 9 Sep 2012

import unittest
import numpy as np
from control import *
from control import matlab

class TestDiscrete(unittest.TestCase):
    """Tests for the DiscreteStateSpace class."""

    def setUp(self):
        """Set up a SISO and MIMO system to test operations on."""

        # Single input, single output continuous and discrete time systems
        sys = matlab.rss(3, 1, 1)
        self.siso_ss1 = StateSpace(sys.A, sys.B, sys.C, sys.D)
        self.siso_ss1c = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.0)
        self.siso_ss1d = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.1)
        self.siso_ss2d = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.2)
        self.siso_ss3d = StateSpace(sys.A, sys.B, sys.C, sys.D, True)

        # Two input, two output continuous time system
        A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.]]
        C = [[4., 2., -3.], [1., 4., 3.]]
        D = [[-2., 4.], [0., 1.]]
        self.mimo_ss1 = StateSpace(A, B, C, D)
        self.mimo_ss1c = StateSpace(A, B, C, D, 0)

        # Two input, two output discrete time system
        self.mimo_ss1d = StateSpace(A, B, C, D, 0.1)

        # Same system, but with a different sampling time
        self.mimo_ss2d = StateSpace(A, B, C, D, 0.2)

        # Single input, single output continuus and discrete transfer function
        self.siso_tf1 = TransferFunction([1, 1], [1, 2, 1])
        self.siso_tf1c = TransferFunction([1, 1], [1, 2, 1], 0)
        self.siso_tf1d = TransferFunction([1, 1], [1, 2, 1], 0.1)
        self.siso_tf2d = TransferFunction([1, 1], [1, 2, 1], 0.2)
        self.siso_tf3d = TransferFunction([1, 1], [1, 2, 1], True)

    def testTimebaseEqual(self):
        self.assertEqual(timebaseEqual(self.siso_ss1, self.siso_tf1), True)
        self.assertEqual(timebaseEqual(self.siso_ss1, self.siso_ss1c), True)
        self.assertEqual(timebaseEqual(self.siso_ss1, self.siso_ss1d), True)
        self.assertEqual(timebaseEqual(self.siso_ss1d, self.siso_ss1c), False)
        self.assertEqual(timebaseEqual(self.siso_ss1d, self.siso_ss2d), False)
        self.assertEqual(timebaseEqual(self.siso_ss1d, self.siso_ss3d), False)

    def testSystemInitialization(self):
        # Check to make sure systems are discrete time with proper variables
        self.assertEqual(self.siso_ss1.dt, None)
        self.assertEqual(self.siso_ss1c.dt, 0)
        self.assertEqual(self.siso_ss1d.dt, 0.1)
        self.assertEqual(self.siso_ss2d.dt, 0.2)
        self.assertEqual(self.siso_ss3d.dt, True)
        self.assertEqual(self.mimo_ss1c.dt, 0)
        self.assertEqual(self.mimo_ss1d.dt, 0.1)
        self.assertEqual(self.mimo_ss2d.dt, 0.2)
        self.assertEqual(self.siso_tf1.dt, None)
        self.assertEqual(self.siso_tf1c.dt, 0)
        self.assertEqual(self.siso_tf1d.dt, 0.1)
        self.assertEqual(self.siso_tf2d.dt, 0.2)
        self.assertEqual(self.siso_tf3d.dt, True)

    def testCopyConstructor(self):
        for sys in (self.siso_ss1, self.siso_ss1c, self.siso_ss1d):
            newsys = StateSpace(sys);
            self.assertEqual(sys.dt, newsys.dt)
        for sys in (self.siso_tf1, self.siso_tf1c, self.siso_tf1d):
            newsys = TransferFunction(sys);
            self.assertEqual(sys.dt, newsys.dt)

    def test_timebase(self):
        self.assertEqual(timebase(1), None);
        self.assertRaises(ValueError, timebase, [1, 2])
        self.assertEqual(timebase(self.siso_ss1, strict=False), None);
        self.assertEqual(timebase(self.siso_ss1, strict=True), None);
        self.assertEqual(timebase(self.siso_ss1c), 0);
        self.assertEqual(timebase(self.siso_ss1d), 0.1);
        self.assertEqual(timebase(self.siso_ss2d), 0.2);
        self.assertEqual(timebase(self.siso_ss3d), True);
        self.assertEqual(timebase(self.siso_ss3d, strict=False), 1);
        self.assertEqual(timebase(self.siso_tf1, strict=False), None);
        self.assertEqual(timebase(self.siso_tf1, strict=True), None);
        self.assertEqual(timebase(self.siso_tf1c), 0);
        self.assertEqual(timebase(self.siso_tf1d), 0.1);
        self.assertEqual(timebase(self.siso_tf2d), 0.2);
        self.assertEqual(timebase(self.siso_tf3d), True);
        self.assertEqual(timebase(self.siso_tf3d, strict=False), 1);

    def test_timebase_conversions(self):
        '''Check to make sure timebases transfer properly'''
        tf1 = TransferFunction([1,1],[1,2,3])       # unspecified
        tf2 = TransferFunction([1,1],[1,2,3], 0)    # cont time
        tf3 = TransferFunction([1,1],[1,2,3], True) # dtime, unspec 
        tf4 = TransferFunction([1,1],[1,2,3], 1)    # dtime, dt=1

        # Make sure unspecified timebase is converted correctly
        self.assertEqual(timebase(tf1*tf1), timebase(tf1))
        self.assertEqual(timebase(tf1*tf2), timebase(tf2))
        self.assertEqual(timebase(tf1*tf3), timebase(tf3))
        self.assertEqual(timebase(tf1*tf4), timebase(tf4))
        self.assertEqual(timebase(tf2*tf1), timebase(tf2))
        self.assertEqual(timebase(tf3*tf1), timebase(tf3))
        self.assertEqual(timebase(tf4*tf1), timebase(tf4))
        self.assertEqual(timebase(tf1+tf1), timebase(tf1))
        self.assertEqual(timebase(tf1+tf2), timebase(tf2))
        self.assertEqual(timebase(tf1+tf3), timebase(tf3))
        self.assertEqual(timebase(tf1+tf4), timebase(tf4))
        self.assertEqual(timebase(feedback(tf1, tf1)), timebase(tf1))
        self.assertEqual(timebase(feedback(tf1, tf2)), timebase(tf2))
        self.assertEqual(timebase(feedback(tf1, tf3)), timebase(tf3))
        self.assertEqual(timebase(feedback(tf1, tf4)), timebase(tf4))

        # Make sure discrete time without sampling is converted correctly
        self.assertEqual(timebase(tf3*tf3), timebase(tf3))
        self.assertEqual(timebase(tf3*tf4), timebase(tf4))
        self.assertEqual(timebase(tf3+tf3), timebase(tf3))
        self.assertEqual(timebase(tf3+tf3), timebase(tf4))
        self.assertEqual(timebase(feedback(tf3, tf3)), timebase(tf3))
        self.assertEqual(timebase(feedback(tf3, tf4)), timebase(tf4))

        # Make sure all other combinations are errors
        try:
            tf2*tf3             # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        try:
            tf2*tf4             # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        try:
            tf2+tf3             # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        try:
            tf2+tf4             # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        try:
            feedback(tf2, tf3)  # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        try:
            feedback(tf2, tf4)   # Error; incompatible timebases
            raise ValueError("incompatible operation allowed")
        except ValueError:
            pass
        
    def testisdtime(self):
        # Constant
        self.assertEqual(isdtime(1), True);
        self.assertEqual(isdtime(1, strict=True), False);

        # State space
        self.assertEqual(isdtime(self.siso_ss1), True);
        self.assertEqual(isdtime(self.siso_ss1, strict=True), False);
        self.assertEqual(isdtime(self.siso_ss1c), False);
        self.assertEqual(isdtime(self.siso_ss1c, strict=True), False);
        self.assertEqual(isdtime(self.siso_ss1d), True);
        self.assertEqual(isdtime(self.siso_ss1d, strict=True), True);
        self.assertEqual(isdtime(self.siso_ss3d, strict=True), True);

        # Transfer function
        self.assertEqual(isdtime(self.siso_tf1), True);
        self.assertEqual(isdtime(self.siso_tf1, strict=True), False);
        self.assertEqual(isdtime(self.siso_tf1c), False);
        self.assertEqual(isdtime(self.siso_tf1c, strict=True), False);
        self.assertEqual(isdtime(self.siso_tf1d), True);
        self.assertEqual(isdtime(self.siso_tf1d, strict=True), True);
        self.assertEqual(isdtime(self.siso_tf3d, strict=True), True);

    def testisctime(self):
        # Constant
        self.assertEqual(isctime(1), True);
        self.assertEqual(isctime(1, strict=True), False);

        # State Space
        self.assertEqual(isctime(self.siso_ss1), True);
        self.assertEqual(isctime(self.siso_ss1, strict=True), False);
        self.assertEqual(isctime(self.siso_ss1c), True);
        self.assertEqual(isctime(self.siso_ss1c, strict=True), True);
        self.assertEqual(isctime(self.siso_ss1d), False);
        self.assertEqual(isctime(self.siso_ss1d, strict=True), False);
        self.assertEqual(isctime(self.siso_ss3d, strict=True), False);

        # Transfer Function
        self.assertEqual(isctime(self.siso_tf1), True);
        self.assertEqual(isctime(self.siso_tf1, strict=True), False);
        self.assertEqual(isctime(self.siso_tf1c), True);
        self.assertEqual(isctime(self.siso_tf1c, strict=True), True);
        self.assertEqual(isctime(self.siso_tf1d), False);
        self.assertEqual(isctime(self.siso_tf1d, strict=True), False);
        self.assertEqual(isctime(self.siso_tf3d, strict=True), False);

    def testAddition(self):
        # State space addition
        sys = self.siso_ss1 + self.siso_ss1d
        sys = self.siso_ss1 + self.siso_ss1c
        sys = self.siso_ss1c + self.siso_ss1
        sys = self.siso_ss1d + self.siso_ss1
        sys = self.siso_ss1c + self.siso_ss1c
        sys = self.siso_ss1d + self.siso_ss1d
        sys = self.siso_ss3d + self.siso_ss3d
        self.assertRaises(ValueError, StateSpace.__add__, self.mimo_ss1c,
                          self.mimo_ss1d)
        self.assertRaises(ValueError, StateSpace.__add__, self.mimo_ss1d,
                          self.mimo_ss2d)
        self.assertRaises(ValueError, StateSpace.__add__, self.siso_ss1d,
                          self.siso_ss3d)

        # Transfer function addition
        sys = self.siso_tf1 + self.siso_tf1d
        sys = self.siso_tf1 + self.siso_tf1c
        sys = self.siso_tf1c + self.siso_tf1
        sys = self.siso_tf1d + self.siso_tf1
        sys = self.siso_tf1c + self.siso_tf1c
        sys = self.siso_tf1d + self.siso_tf1d
        sys = self.siso_tf2d + self.siso_tf2d
        self.assertRaises(ValueError, TransferFunction.__add__, self.siso_tf1c,
                          self.siso_tf1d)
        self.assertRaises(ValueError, TransferFunction.__add__, self.siso_tf1d,
                          self.siso_tf2d)
        self.assertRaises(ValueError, TransferFunction.__add__, self.siso_tf1d,
                          self.siso_tf3d)

        # State space + transfer function
        sys = self.siso_ss1c + self.siso_tf1c
        sys = self.siso_tf1c + self.siso_ss1c
        sys = self.siso_ss1d + self.siso_tf1d
        sys = self.siso_tf1d + self.siso_ss1d
        self.assertRaises(ValueError, TransferFunction.__add__, self.siso_tf1c,
                          self.siso_ss1d)

    def testMultiplication(self):
        # State space addition
        sys = self.siso_ss1 * self.siso_ss1d
        sys = self.siso_ss1 * self.siso_ss1c
        sys = self.siso_ss1c * self.siso_ss1
        sys = self.siso_ss1d * self.siso_ss1
        sys = self.siso_ss1c * self.siso_ss1c
        sys = self.siso_ss1d * self.siso_ss1d
        self.assertRaises(ValueError, StateSpace.__mul__, self.mimo_ss1c,
                          self.mimo_ss1d)
        self.assertRaises(ValueError, StateSpace.__mul__, self.mimo_ss1d,
                          self.mimo_ss2d)
        self.assertRaises(ValueError, StateSpace.__mul__, self.siso_ss1d,
                          self.siso_ss3d)

        # Transfer function addition
        sys = self.siso_tf1 * self.siso_tf1d
        sys = self.siso_tf1 * self.siso_tf1c
        sys = self.siso_tf1c * self.siso_tf1
        sys = self.siso_tf1d * self.siso_tf1
        sys = self.siso_tf1c * self.siso_tf1c
        sys = self.siso_tf1d * self.siso_tf1d
        self.assertRaises(ValueError, TransferFunction.__mul__, self.siso_tf1c,
                          self.siso_tf1d)
        self.assertRaises(ValueError, TransferFunction.__mul__, self.siso_tf1d,
                          self.siso_tf2d)
        self.assertRaises(ValueError, TransferFunction.__mul__, self.siso_tf1d,
                          self.siso_tf3d)

        # State space * transfer function
        sys = self.siso_ss1c * self.siso_tf1c
        sys = self.siso_tf1c * self.siso_ss1c
        sys = self.siso_ss1d * self.siso_tf1d
        sys = self.siso_tf1d * self.siso_ss1d
        self.assertRaises(ValueError, TransferFunction.__mul__, self.siso_tf1c,
                          self.siso_ss1d)


    def testFeedback(self):
        # State space addition
        sys = feedback(self.siso_ss1, self.siso_ss1d)
        sys = feedback(self.siso_ss1, self.siso_ss1c)
        sys = feedback(self.siso_ss1c, self.siso_ss1)
        sys = feedback(self.siso_ss1d, self.siso_ss1)
        sys = feedback(self.siso_ss1c, self.siso_ss1c)
        sys = feedback(self.siso_ss1d, self.siso_ss1d)
        self.assertRaises(ValueError, feedback, self.mimo_ss1c, self.mimo_ss1d)
        self.assertRaises(ValueError, feedback, self.mimo_ss1d, self.mimo_ss2d)
        self.assertRaises(ValueError, feedback, self.siso_ss1d, self.siso_ss3d)

        # Transfer function addition
        sys = feedback(self.siso_tf1, self.siso_tf1d)
        sys = feedback(self.siso_tf1, self.siso_tf1c)
        sys = feedback(self.siso_tf1c, self.siso_tf1)
        sys = feedback(self.siso_tf1d, self.siso_tf1)
        sys = feedback(self.siso_tf1c, self.siso_tf1c)
        sys = feedback(self.siso_tf1d, self.siso_tf1d)
        self.assertRaises(ValueError, feedback, self.siso_tf1c, self.siso_tf1d)
        self.assertRaises(ValueError, feedback, self.siso_tf1d, self.siso_tf2d)
        self.assertRaises(ValueError, feedback, self.siso_tf1d, self.siso_tf3d)

        # State space, transfer function
        sys = feedback(self.siso_ss1c, self.siso_tf1c)
        sys = feedback(self.siso_tf1c, self.siso_ss1c)
        sys = feedback(self.siso_ss1d, self.siso_tf1d)
        sys = feedback(self.siso_tf1d, self.siso_ss1d)
        self.assertRaises(ValueError, feedback, self.siso_tf1c, self.siso_ss1d)

    def testSimulation(self):
        T = range(100)
        U = np.sin(T)

        # For now, just check calling syntax
        # TODO: add checks on output of simulations
        tout, yout = step_response(self.siso_ss1d)
        tout, yout = step_response(self.siso_ss1d, T)
        tout, yout = impulse_response(self.siso_ss1d, T)
        tout, yout = impulse_response(self.siso_ss1d)
        tout, yout, xout = forced_response(self.siso_ss1d, T, U, 0)
        tout, yout, xout = forced_response(self.siso_ss2d, T, U, 0)
        tout, yout, xout = forced_response(self.siso_ss3d, T, U, 0)

    def test_sample_system(self):
        # Make sure we can convert various types of systems
        for sysc in (self.siso_tf1, self.siso_tf1c,
                     self.siso_ss1, self.siso_ss1c,
                     self.mimo_ss1, self.mimo_ss1c):
            for method in ("zoh", "bilinear", "euler", "backward_diff"):
                sysd = sample_system(sysc, 1, method=method)
                self.assertEqual(sysd.dt, 1)

        # Check "matched", defined only for SISO transfer functions
        for sysc in (self.siso_tf1, self.siso_tf1c):
            sysd = sample_system(sysc, 1, method="matched")
            self.assertEqual(sysd.dt, 1)

        # Check errors
        self.assertRaises(ValueError, sample_system, self.siso_ss1d, 1)
        self.assertRaises(ValueError, sample_system, self.siso_tf1d, 1)
        self.assertRaises(ValueError, sample_system, self.siso_ss1, 1, 'unknown')

    def test_sample_ss(self):
        # double integrators, two different ways
        sys1 = StateSpace([[0.,1.],[0.,0.]], [[0.],[1.]], [[1.,0.]], 0.)
        sys2 = StateSpace([[0.,0.],[1.,0.]], [[1.],[0.]], [[0.,1.]], 0.)
        I = np.eye(2)
        for sys in (sys1, sys2):
            for h in (0.1, 0.5, 1, 2):
                Ad = I + h * sys.A
                Bd = h * sys.B + 0.5 * h**2 * (sys.A * sys.B)
                sysd = sample_system(sys, h, method='zoh')
                np.testing.assert_array_almost_equal(sysd.A, Ad)
                np.testing.assert_array_almost_equal(sysd.B, Bd)
                np.testing.assert_array_almost_equal(sysd.C, sys.C)
                np.testing.assert_array_almost_equal(sysd.D, sys.D)
                self.assertEqual(sysd.dt, h)

    def test_sample_tf(self):
        # double integrator
        sys = TransferFunction(1, [1,0,0])
        for h in (0.1, 0.5, 1, 2):
            numd_expected = 0.5 * h**2 * np.array([1.,1.])
            dend_expected = np.array([1.,-2.,1.])
            sysd = sample_system(sys, h, method='zoh')
            self.assertEqual(sysd.dt, h)
            numd = sysd.num[0][0]
            dend = sysd.den[0][0]
            np.testing.assert_array_almost_equal(numd, numd_expected)
            np.testing.assert_array_almost_equal(dend, dend_expected)

    def test_discrete_bode(self):
        # Create a simple discrete time system and check the calculation
        sys = TransferFunction([1], [1, 0.5], 1)
        omega = [1, 2, 3]
        mag_out, phase_out, omega_out = bode(sys, omega)
        H_z = list(map(lambda w: 1./(np.exp(1.j * w) + 0.5), omega))
        np.testing.assert_array_almost_equal(omega, omega_out)
        np.testing.assert_array_almost_equal(mag_out, np.absolute(H_z))
        np.testing.assert_array_almost_equal(phase_out, np.angle(H_z))

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestDiscrete)


if __name__ == "__main__":
    unittest.main()
