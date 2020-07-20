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

    def test_set_defaults(self):
        ct.config.set_defaults('config', test1=1, test2=2, test3=None)
        self.assertEqual(ct.config.defaults['config.test1'], 1)
        self.assertEqual(ct.config.defaults['config.test2'], 2)
        self.assertEqual(ct.config.defaults['config.test3'], None)

    def test_get_param(self):
        self.assertEqual(
            ct.config._get_param('bode', 'dB'),
            ct.config.defaults['bode.dB'])
        self.assertEqual(ct.config._get_param('bode', 'dB', 1), 1)
        ct.config.defaults['config.test1'] = 1
        self.assertEqual(ct.config._get_param('config', 'test1', None), 1)
        self.assertEqual(ct.config._get_param('config', 'test1', None, 1), 1)
        
        ct.config.defaults['config.test3'] = None
        self.assertEqual(ct.config._get_param('config', 'test3'), None)
        self.assertEqual(ct.config._get_param('config', 'test3', 1), 1)
        self.assertEqual(
            ct.config._get_param('config', 'test3', None, 1), None)
        
        self.assertEqual(ct.config._get_param('config', 'test4'), None)
        self.assertEqual(ct.config._get_param('config', 'test4', 1), 1)
        self.assertEqual(ct.config._get_param('config', 'test4', 2, 1), 2)
        self.assertEqual(ct.config._get_param('config', 'test4', None, 3), 3)

        self.assertEqual(
            ct.config._get_param('config', 'test4', {'test4':1}, None), 1)


    def test_fbs_bode(self):
        ct.use_fbs_defaults()

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

        ct.reset_defaults()
        
    def test_matlab_bode(self):
        ct.use_matlab_defaults()

        # Generate a Bode plot
        plt.figure()
        omega = np.logspace(-3, 3, 100)
        ct.bode_plot(self.sys, omega)

        # Get the magnitude line
        mag_axis = plt.gcf().axes[0]
        mag_line = mag_axis.get_lines()
        mag_data = mag_line[0].get_data()
        mag_x, mag_y = mag_data

        # Make sure the x-axis is in rad/sec and y-axis is in dB
        np.testing.assert_almost_equal(mag_x[0], 0.001, decimal=6)
        np.testing.assert_almost_equal(mag_y[0], 20*log10(10), decimal=3)

        # Get the phase line
        phase_axis = plt.gcf().axes[1]
        phase_line = phase_axis.get_lines()
        phase_data = phase_line[0].get_data()
        phase_x, phase_y = phase_data

        # Make sure the x-axis is in rad/sec and y-axis is in degrees
        np.testing.assert_almost_equal(phase_x[-1], 1000, decimal=1)
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

        ct.reset_defaults()

    def test_custom_bode_default(self):
        ct.config.defaults['bode.dB'] = True
        ct.config.defaults['bode.deg'] = True
        ct.config.defaults['bode.Hz'] = True

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

        ct.reset_defaults()

    def test_bode_number_of_samples(self):
        # Set the number of samples (default is 50, from np.logspace)
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        self.assertEqual(len(mag_ret), 87)

        # Change the default number of samples
        ct.config.defaults['freqplot.number_of_samples'] = 76
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys)
        self.assertEqual(len(mag_ret), 76)
        
        # Override the default number of samples
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        self.assertEqual(len(mag_ret), 87)

        ct.reset_defaults()

    def test_bode_feature_periphery_decade(self):
        # Generate a sample Bode plot to figure out the range it uses
        ct.reset_defaults()     # Make sure starting state is correct
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=False)
        omega_min, omega_max = omega_ret[[0,  -1]]

        # Reset the periphery decade value (should add one decade on each end)
        ct.config.defaults['freqplot.feature_periphery_decades'] = 2
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=False)
        np.testing.assert_almost_equal(omega_ret[0], omega_min/10)
        np.testing.assert_almost_equal(omega_ret[-1], omega_max * 10)

        # Make sure it also works in rad/sec, in opposite direction
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=True)
        omega_min, omega_max = omega_ret[[0,  -1]]
        ct.config.defaults['freqplot.feature_periphery_decades'] = 1
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, Hz=True)
        np.testing.assert_almost_equal(omega_ret[0], omega_min*10)
        np.testing.assert_almost_equal(omega_ret[-1], omega_max/10)

        ct.reset_defaults()

    def test_reset_defaults(self):
        ct.use_matlab_defaults()
        ct.reset_defaults()
        self.assertEqual(ct.config.defaults['bode.dB'], False)
        self.assertEqual(ct.config.defaults['bode.deg'], True)
        self.assertEqual(ct.config.defaults['bode.Hz'], False)
        self.assertEqual(
            ct.config.defaults['freqplot.number_of_samples'], None)
        self.assertEqual(
            ct.config.defaults['freqplot.feature_periphery_decades'], 1.0)

    def test_legacy_defaults(self):
        ct.use_legacy_defaults('0.8.3')
        assert(isinstance(ct.ss(0,0,0,1).D, np.matrix))
        ct.reset_defaults()
        assert(isinstance(ct.ss(0,0,0,1).D, np.ndarray))
    
    def test_change_default_dt(self):
        # test that system with dynamics uses correct default dt
        ct.set_defaults('control', default_dt=0)
        self.assertEqual(ct.tf(1, [1,1]).dt, 0)
        self.assertEqual(ct.ss(1,0,0,1).dt, 0)
        self.assertEqual(ct.iosys.NonlinearIOSystem(
            lambda t, x, u: u*x*x, 
            lambda t, x, u: x, inputs=1, outputs=1).dt, 0)
        ct.set_defaults('control', default_dt=None)
        self.assertEqual(ct.tf(1, [1,1]).dt, None)
        self.assertEqual(ct.ss(1,0,0,1).dt, None)
        self.assertEqual(ct.iosys.NonlinearIOSystem(
            lambda t, x, u: u*x*x, 
            lambda t, x, u: x, inputs=1, outputs=1).dt, None)
        
        # test that static gain systems always have dt=None
        ct.set_defaults('control', default_dt=0)
        self.assertEqual(ct.tf(1, 1).dt, None)
        self.assertEqual(ct.ss(0,0,0,1).dt, None)        
        # TODO: add in test for static gain iosys

        ct.reset_defaults()
        
    def tearDown(self):
        # Get rid of any figures that we created
        plt.close('all')

        # Reset the configuration defaults
        ct.config.reset_defaults()


if __name__ == '__main__':
    unittest.main()
