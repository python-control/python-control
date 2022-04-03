"""config_test.py - test config module

RMM, 25 may 2019

This test suite checks the functionality of the config module
"""

from math import pi, log10

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup as mplcleanup
import numpy as np
import pytest

import control as ct


@pytest.mark.usefixtures("editsdefaults")  # makes sure to reset the defaults
                                           # to the test configuration
class TestConfig:

    # Create a simple second order system to use for testing
    sys = ct.tf([10], [1, 2, 1])

    def test_set_defaults(self):
        ct.config.set_defaults('freqplot', dB=1, deg=2, Hz=None)
        assert ct.config.defaults['freqplot.dB'] == 1
        assert ct.config.defaults['freqplot.deg'] == 2
        assert ct.config.defaults['freqplot.Hz'] is None

    @mplcleanup
    def test_get_param(self):
        assert ct.config._get_param('freqplot', 'dB')\
            == ct.config.defaults['freqplot.dB']
        assert ct.config._get_param('freqplot', 'dB', 1) == 1
        ct.config.defaults['config.test1'] = 1
        assert ct.config._get_param('config', 'test1', None) == 1
        assert ct.config._get_param('config', 'test1', None, 1) == 1

        ct.config.defaults['config.test3'] = None
        assert ct.config._get_param('config', 'test3') is None
        assert ct.config._get_param('config', 'test3', 1) == 1
        assert ct.config._get_param('config', 'test3', None, 1) is None

        assert ct.config._get_param('config', 'test4') is None
        assert ct.config._get_param('config', 'test4', 1) == 1
        assert ct.config._get_param('config', 'test4', 2, 1) == 2
        assert ct.config._get_param('config', 'test4', None, 3) == 3

        assert ct.config._get_param('config', 'test4', {'test4': 1}, None) == 1

    def test_default_deprecation(self):
        ct.config.defaults['deprecated.config.oldkey'] = 'config.newkey'
        ct.config.defaults['deprecated.config.oldmiss'] = 'config.newmiss'

        msgpattern = r'config\.oldkey.* has been renamed to .*config\.newkey'

        ct.config.defaults['config.newkey'] = 1
        with pytest.warns(FutureWarning, match=msgpattern):
            assert ct.config.defaults['config.oldkey'] == 1
        with pytest.warns(FutureWarning, match=msgpattern):
            ct.config.defaults['config.oldkey'] = 2
        with pytest.warns(FutureWarning, match=msgpattern):
            assert ct.config.defaults['config.oldkey'] == 2
        assert ct.config.defaults['config.newkey'] == 2

        ct.config.set_defaults('config', newkey=3)
        with pytest.warns(FutureWarning, match=msgpattern):
            assert ct.config._get_param('config', 'oldkey') == 3
        with pytest.warns(FutureWarning, match=msgpattern):
            ct.config.set_defaults('config', oldkey=4)
        with pytest.warns(FutureWarning, match=msgpattern):
            assert ct.config.defaults['config.oldkey'] == 4
        assert ct.config.defaults['config.newkey'] == 4

        ct.config.defaults.update({'config.newkey': 5})
        with pytest.warns(FutureWarning, match=msgpattern):
            ct.config.defaults.update({'config.oldkey': 6})
        with pytest.warns(FutureWarning, match=msgpattern):
            assert ct.config.defaults.get('config.oldkey') == 6

        with pytest.raises(KeyError):
            with pytest.warns(FutureWarning, match=msgpattern):
                ct.config.defaults['config.oldmiss']
        with pytest.raises(KeyError):
            ct.config.defaults['config.neverdefined']

        # assert that reset defaults keeps the custom type
        ct.config.reset_defaults()
        with pytest.warns(FutureWarning,
                          match='bode.* has been renamed to.*freqplot'):
            assert ct.config.defaults['bode.Hz'] \
                == ct.config.defaults['freqplot.Hz']

    @mplcleanup
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

    @mplcleanup
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

    @mplcleanup
    def test_custom_bode_default(self):
        ct.config.defaults['freqplot.dB'] = True
        ct.config.defaults['freqplot.deg'] = True
        ct.config.defaults['freqplot.Hz'] = True

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

    @mplcleanup
    def test_bode_number_of_samples(self):
        # Set the number of samples (default is 50, from np.logspace)
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        assert len(mag_ret) == 87

        # Change the default number of samples
        ct.config.defaults['freqplot.number_of_samples'] = 76
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys)
        assert len(mag_ret) == 76

        # Override the default number of samples
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, omega_num=87)
        assert len(mag_ret) == 87

    @mplcleanup
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

    def test_reset_defaults(self):
        ct.use_matlab_defaults()
        ct.reset_defaults()
        assert not ct.config.defaults['freqplot.dB']
        assert ct.config.defaults['freqplot.deg']
        assert not ct.config.defaults['freqplot.Hz']
        assert ct.config.defaults['freqplot.number_of_samples'] == 1000
        assert ct.config.defaults['freqplot.feature_periphery_decades'] == 1.0

    def test_legacy_defaults(self):
        with pytest.deprecated_call():
            ct.use_legacy_defaults('0.8.3')
            assert(isinstance(ct.ss(0, 0, 0, 1).D, np.matrix))
        ct.reset_defaults()
        assert isinstance(ct.ss(0, 0, 0, 1).D, np.ndarray)
        assert not isinstance(ct.ss(0, 0, 0, 1).D, np.matrix)

        ct.use_legacy_defaults('0.8.4')
        assert ct.config.defaults['forced_response.return_x'] is True

        ct.use_legacy_defaults('0.9.0')
        assert isinstance(ct.ss(0, 0, 0, 1).D, np.ndarray)
        assert not isinstance(ct.ss(0, 0, 0, 1).D, np.matrix)

        # test that old versions don't raise a problem
        ct.use_legacy_defaults('REL-0.1')
        ct.use_legacy_defaults('control-0.3a')
        ct.use_legacy_defaults('0.6c')
        ct.use_legacy_defaults('0.8.2')
        ct.use_legacy_defaults('0.1')

        # Make sure that nonsense versions generate an error
        with pytest.raises(ValueError):
            ct.use_legacy_defaults("a.b.c")
        with pytest.raises(ValueError):
            ct.use_legacy_defaults("1.x.3")

    @pytest.mark.parametrize("dt", [0, None])
    def test_change_default_dt(self, dt):
        """Test that system with dynamics uses correct default dt"""
        ct.set_defaults('control', default_dt=dt)
        assert ct.ss(1, 0, 0, 1).dt == dt
        assert ct.tf(1, [1, 1]).dt == dt
        nlsys = ct.iosys.NonlinearIOSystem(
            lambda t, x, u: u * x * x,
            lambda t, x, u: x, inputs=1, outputs=1)
        assert nlsys.dt == dt

    def test_change_default_dt_static(self):
        """Test that static gain systems always have dt=None"""
        ct.set_defaults('control', default_dt=0)
        assert ct.tf(1, 1).dt is None
        assert ct.ss([], [], [], 1).dt is None

        # Make sure static gain is preserved for the I/O system
        sys = ct.ss([], [], [], 1)
        sys_io = ct.ss2io(sys)
        assert sys_io.dt is None

    def test_get_param_last(self):
        """Test _get_param last keyword"""
        kwargs = {'first': 1, 'second': 2}

        with pytest.raises(TypeError, match="unrecognized keyword.*second"):
            assert ct.config._get_param(
                'config', 'first', kwargs, pop=True, last=True) == 1

        assert ct.config._get_param(
            'config', 'second', kwargs, pop=True, last=True) == 2
