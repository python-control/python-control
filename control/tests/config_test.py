#!/usr/bin/env python
#
# config_test.py - test config module
# RMM, 25 may 2019
#
# This test suite checks the functionality of the config module

import unittest
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from math import pi, log10

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a simple second order system to use for testing
        self.sys = ct.tf([10], [1, 2, 1])

    def test_fbs_bode(self):
        ct.use_fbs_defaults();

        # Generate a Bode plot
        plt.figure()
        omega = np.logspace(-3, 3, 100)
        ct.bode_plot(self.sys, omega)

        # Get the magnitude line
        mag_axis = plt.gcf().axes[0]
        mag_line = mag_axis.get_lines()
        mag_data = mag_line[0].get_data()
        mag_x, mag_y = mag_data

        # Make sure the x-axis is in rad/sec and y-axis is in natural units
        np.testing.assert_almost_equal(mag_x[0], 0.001, decimal=6)
        np.testing.assert_almost_equal(mag_y[0], 10, decimal=3)

        # Get the phase line
        phase_axis = plt.gcf().axes[1]
        phase_line = phase_axis.get_lines()
        phase_data = phase_line[0].get_data()
        phase_x, phase_y = phase_data

        # Make sure the x-axis is in rad/sec and y-axis is in degrees
        np.testing.assert_almost_equal(phase_x[-1], 1000, decimal=0)
        np.testing.assert_almost_equal(phase_y[-1], -180, decimal=0)

        # Override the defaults and make sure that works as well
        plt.figure()
        ct.bode_plot(self.sys, omega, dB=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)

        plt.figure()
        ct.bode_plot(self.sys, omega, Hz=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_x[0], 0.001 / (2*pi), decimal=6)

        plt.figure()
        ct.bode_plot(self.sys, omega, deg=False)
        phase_x, phase_y = (((plt.gcf().axes[1]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(phase_y[-1], -pi, decimal=2)
        
    def test_matlab_bode(self):
        ct.use_matlab_defaults();

        # Generate a Bode plot
        plt.figure()
        omega = np.logspace(-3, 3, 100)
        ct.bode_plot(self.sys, omega)

        # Get the magnitude line
        mag_axis = plt.gcf().axes[0]
        mag_line = mag_axis.get_lines()
        mag_data = mag_line[0].get_data()
        mag_x, mag_y = mag_data

        # Make sure the x-axis is in Hertz and y-axis is in dB
        np.testing.assert_almost_equal(mag_x[0], 0.001 / (2*pi), decimal=6)
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)

        # Get the phase line
        phase_axis = plt.gcf().axes[1]
        phase_line = phase_axis.get_lines()
        phase_data = phase_line[0].get_data()
        phase_x, phase_y = phase_data

        # Make sure the x-axis is in Hertz and y-axis is in degrees
        np.testing.assert_almost_equal(phase_x[-1], 1000 / (2*pi), decimal=1)
        np.testing.assert_almost_equal(phase_y[-1], -180, decimal=0)
        
        # Override the defaults and make sure that works as well
        plt.figure()
        ct.bode_plot(self.sys, omega, dB=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)

        plt.figure()
        ct.bode_plot(self.sys, omega, Hz=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_x[0], 0.001 / (2*pi), decimal=6)

        plt.figure()
        ct.bode_plot(self.sys, omega, deg=False)
        phase_x, phase_y = (((plt.gcf().axes[1]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(phase_y[-1], -pi, decimal=2)

    def test_custom_bode_default(self):
        ct.bode_dB = True
        ct.bode_deg = True
        ct.bode_Hz = True

        # Generate a Bode plot
        plt.figure()
        omega = np.logspace(-3, 3, 100)
        ct.bode_plot(self.sys, omega, dB=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)

        # Override defaults
        plt.figure()
        ct.bode_plot(self.sys, omega, Hz=True, deg=False, dB=True)
        mag_x, mag_y = (((plt.gcf().axes[0]).get_lines())[0]).get_data()
        phase_x, phase_y = (((plt.gcf().axes[1]).get_lines())[0]).get_data()
        np.testing.assert_almost_equal(mag_x[0], 0.001 / (2*pi), decimal=6)
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)
        np.testing.assert_almost_equal(phase_y[-1], -pi, decimal=2)

    def test_bode_number_of_samples(self):
        # Set the number of samples (default is 50, from np.logspace)
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        self.assertEqual(len(mag_ret), 87)

        # Change the default number of samples
        ct.config.bode_number_of_samples = 76
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys)
        self.assertEqual(len(mag_ret), 76)
        
        # Override the default number of samples
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        self.assertEqual(len(mag_ret), 87)

    def test_bode_feature_periphery_decade(self):
        # Generate a sample Bode plot to figure out the range it uses
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=False)
        omega_min, omega_max = omega_ret[[0,  -1]]

        # Reset the periphery decade value (should add one decade on each end)
        ct.config.bode_feature_periphery_decade = 2
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=False)
        np.testing.assert_almost_equal(omega_ret[0], omega_min/10)
        np.testing.assert_almost_equal(omega_ret[-1], omega_max * 10)

        # Make sure it also works in rad/sec, in opposite direction
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=True)
        omega_min, omega_max = omega_ret[[0,  -1]]
        ct.config.bode_feature_periphery_decade = 1
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=True)
        np.testing.assert_almost_equal(omega_ret[0], omega_min*10)
        np.testing.assert_almost_equal(omega_ret[-1], omega_max/10)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestTimeresp)


if __name__ == '__main__':
    unittest.main()
