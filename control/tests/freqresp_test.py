#!/usr/bin/env python
#
# freqresp_test.py - test frequency response functions
# RMM, 30 May 2016 (based on timeresp_test.py)
#
# This is a rudimentary set of tests for frequency response functions,
# including bode plots.

import unittest
import numpy as np
import control as ctrl
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from control.matlab import ss, tf, bode, rss
from control.exception import slycot_check
from control.tests.margin_test import assert_array_almost_equal
import matplotlib.pyplot as plt

class TestFreqresp(unittest.TestCase):
   def setUp(self):
      self.A = np.matrix('1,1;0,1')
      self.C = np.matrix('1,0')
      self.omega = np.linspace(10e-2,10e2,1000)

   def test_siso(self):
      B = np.matrix('0;1')
      D = 0
      sys = StateSpace(self.A,B,self.C,D)

      # test frequency response
      frq=sys.freqresp(self.omega)

      # test bode plot
      bode(sys)

      # Convert to transfer function and test bode
      systf = tf(sys)
      bode(systf)

   def test_superimpose(self):
      # Test to make sure that multiple calls to plots superimpose their
      # data on the same axes unless told to do otherwise

      # Generate two plots in a row; should be on the same axes
      plt.figure(1); plt.clf()
      ctrl.bode_plot(ctrl.tf([1], [1,2,1]))
      ctrl.bode_plot(ctrl.tf([5], [1, 1]))

      # Check to make sure there are two axes and that each axes has two lines
      self.assertEqual(len(plt.gcf().axes), 2)
      for ax in plt.gcf().axes:
         # Make sure there are 2 lines in each subplot
         assert len(ax.get_lines()) == 2
      
      # Generate two plots as a list; should be on the same axes
      plt.figure(2); plt.clf();
      ctrl.bode_plot([ctrl.tf([1], [1,2,1]), ctrl.tf([5], [1, 1])])

      # Check to make sure there are two axes and that each axes has two lines
      self.assertEqual(len(plt.gcf().axes), 2)
      for ax in plt.gcf().axes:
         # Make sure there are 2 lines in each subplot
         assert len(ax.get_lines()) == 2

      # Generate two separate plots; only the second should appear
      plt.figure(3); plt.clf();
      ctrl.bode_plot(ctrl.tf([1], [1,2,1]))
      plt.clf()
      ctrl.bode_plot(ctrl.tf([5], [1, 1]))

      # Check to make sure there are two axes and that each axes has one line
      self.assertEqual(len(plt.gcf().axes), 2)
      for ax in plt.gcf().axes:
         # Make sure there is only 1 line in the subplot
         assert len(ax.get_lines()) == 1

      # Now add a line to the magnitude plot and make sure if is there
      for ax in plt.gcf().axes:
         if ax.get_label() == 'control-bode-magnitude':
            break
      ax.semilogx([1e-2, 1e1], 20 * np.log10([1, 1]), 'k-')
      self.assertEqual(len(ax.get_lines()), 2)

   def test_doubleint(self):
      # 30 May 2016, RMM: added to replicate typecast bug in freqresp.py
      A = np.matrix('0, 1; 0, 0');
      B = np.matrix('0; 1');
      C = np.matrix('1, 0');
      D = 0;
      sys = ss(A, B, C, D);
      bode(sys);

   @unittest.skipIf(not slycot_check(), "slycot not installed")
   def test_mimo(self):
      # MIMO
      B = np.matrix('1,0;0,1')
      D = np.matrix('0,0')
      sysMIMO = ss(self.A,B,self.C,D)

      frqMIMO = sysMIMO.freqresp(self.omega)
      tfMIMO = tf(sysMIMO)

      #bode(sysMIMO) # - should throw not implemented exception
      #bode(tfMIMO) # - should throw not implemented exception

      #plt.figure(3)
      #plt.semilogx(self.omega,20*np.log10(np.squeeze(frq[0])))

      #plt.figure(4)
      #bode(sysMIMO,self.omega)

   def test_bode_margin(self):
      num = [1000]
      den = [1, 25, 100, 0]
      sys = ctrl.tf(num, den)
      plt.figure()
      ctrl.bode_plot(sys, margins=True,dB=False,deg = True, Hz=False)
      fig = plt.gcf()
      allaxes = fig.get_axes()

      mag_to_infinity = (np.array([6.07828691, 6.07828691]),
                         np.array([1.00000000e+00, 1.00000000e-08]))
      assert_array_almost_equal(mag_to_infinity, allaxes[0].lines[2].get_data())

      gm_to_infinty = (np.array([10., 10.]), np.array([4.00000000e-01, 1.00000000e-08]))
      assert_array_almost_equal(gm_to_infinty, allaxes[0].lines[3].get_data())

      one_to_gm = (np.array([10., 10.]), np.array([1., 0.4]))
      assert_array_almost_equal(one_to_gm, allaxes[0].lines[4].get_data())

      pm_to_infinity = (np.array([6.07828691, 6.07828691]),
                        np.array([100000., -157.46405841]))
      assert_array_almost_equal(pm_to_infinity, allaxes[1].lines[2].get_data())

      pm_to_phase = (np.array([6.07828691, 6.07828691]), np.array([-157.46405841, -180.]))
      assert_array_almost_equal(pm_to_phase, allaxes[1].lines[3].get_data())

      phase_to_infinity = (np.array([10., 10.]), np.array([1.00000000e-08, -1.80000000e+02]))
      assert_array_almost_equal(phase_to_infinity, allaxes[1].lines[4].get_data())

   def test_discrete(self):
      # Test discrete time frequency response

      # SISO state space systems with either fixed or unspecified sampling times
      sys = rss(3, 1, 1)
      siso_ss1d = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.1)
      siso_ss2d = StateSpace(sys.A, sys.B, sys.C, sys.D, True)

      # MIMO state space systems with either fixed or unspecified sampling times
      A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
      B = [[1., 4.], [-3., -3.], [-2., 1.]]
      C = [[4., 2., -3.], [1., 4., 3.]]
      D = [[-2., 4.], [0., 1.]]
      mimo_ss1d = StateSpace(A, B, C, D, 0.1)
      mimo_ss2d = StateSpace(A, B, C, D, True)

      # SISO transfer functions
      siso_tf1d = TransferFunction([1, 1], [1, 2, 1], 0.1)
      siso_tf2d = TransferFunction([1, 1], [1, 2, 1], True)

      # Go through each system and call the code, checking return types
      for sys in (siso_ss1d, siso_ss2d, mimo_ss1d, mimo_ss2d,
                siso_tf1d, siso_tf2d):
         # Set frequency range to just below Nyquist freq (for Bode)
         omega_ok = np.linspace(10e-4,0.99,100) * np.pi/sys.dt

         # Test frequency response
         ret = sys.freqresp(omega_ok)

         # Check for warning if frequency is out of range
         import warnings
         warnings.simplefilter('always', UserWarning)   # don't supress
         with warnings.catch_warnings(record=True) as w:
            # Set up warnings filter to only show warnings in control module
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("always", module="control")

            # Look for a warning about sampling above Nyquist frequency
            omega_bad = np.linspace(10e-4,1.1,10) * np.pi/sys.dt
            ret = sys.freqresp(omega_bad)
            print("len(w) =", len(w))
            self.assertEqual(len(w), 1)
            self.assertIn("above", str(w[-1].message))
            self.assertIn("Nyquist", str(w[-1].message))

         # Test bode plots (currently only implemented for SISO)
         if (sys.inputs == 1 and sys.outputs == 1):
            # Generic call (frequency range calculated automatically)
            ret_ss = bode(sys)

            # Convert to transfer function and test bode again
            systf = tf(sys);
            ret_tf = bode(systf)

            # Make sure we can pass a frequency range
            bode(sys, omega_ok)

         else:
            # Calling bode should generate a not implemented error
            self.assertRaises(NotImplementedError, bode, (sys,))

      def test_options(self):
         """Test ability to set parameter values"""
      # Generate a Bode plot of a transfer function
      sys = ctrl.tf([1000], [1, 25, 100, 0])
      fig1 = plt.figure()
      ctrl.bode_plot(sys, dB=False, deg = True, Hz=False)

      # Save the parameter values
      left1, right1 = fig1.axes[0].xaxis.get_data_interval()
      numpoints1 = len(fig1.axes[0].lines[0].get_data()[0])

      # Same transfer function, but add a decade on each end
      ctrl.config.set_defaults('freqplot', feature_periphery_decades=2)
      fig2 = plt.figure()
      ctrl.bode_plot(sys, dB=False, deg = True, Hz=False)
      left2, right2 = fig2.axes[0].xaxis.get_data_interval()

      # Make sure we got an extra decade on each end
      self.assertAlmostEqual(left2, 0.1 * left1)
      self.assertAlmostEqual(right2, 10 * right1)

      # Same transfer function, but add more points to the plot
      ctrl.config.set_defaults(
         'freqplot', feature_periphery_decades=2, number_of_samples=13)
      fig3 = plt.figure()
      ctrl.bode_plot(sys, dB=False, deg = True, Hz=False)
      numpoints3 = len(fig3.axes[0].lines[0].get_data()[0])

      # Make sure we got the right number of points
      self.assertNotEqual(numpoints1, numpoints3)
      self.assertEqual(numpoints3, 13)

      # Reset default parameters to avoid contamination
      ctrl.config.reset_defaults()


def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)

if __name__ == '__main__':
   unittest.main()
