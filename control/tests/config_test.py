"""config_test.py - test config module

RMM, 25 may 2019

This test suite checks the functionality of the config module
"""

from math import pi, log10

import matplotlib.pyplot as plt
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

    def test_get_param(self, mplcleanup):
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
        msgmisspattern = r'config\.oldmiss.* has been renamed to .*config\.newmiss'

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
            with pytest.warns(FutureWarning, match=msgmisspattern):
                ct.config.defaults['config.oldmiss']
        with pytest.raises(KeyError):
            ct.config.defaults['config.neverdefined']

        # assert that reset defaults keeps the custom type
        ct.config.reset_defaults()
        with pytest.raises(KeyError):
            assert ct.config.defaults['bode.Hz'] \
                == ct.config.defaults['freqplot.Hz']

    @pytest.mark.usefixtures("legacy_plot_signature")
    def test_fbs_bode(self, mplcleanup):
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

        # Make sure x-axis label is Gain
        assert mag_axis.get_ylabel() == "Gain"

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

    @pytest.mark.usefixtures("legacy_plot_signature")
    def test_matlab_bode(self, mplcleanup):
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

        # Make sure x-axis label is Gain
        assert mag_axis.get_ylabel() == "Magnitude [dB]"

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

    @pytest.mark.usefixtures("legacy_plot_signature")
    def test_custom_bode_default(self, mplcleanup):
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

    @pytest.mark.usefixtures("legacy_plot_signature")
    def test_bode_number_of_samples(self, mplcleanup):
        # Set the number of samples (default is 50, from np.logspace)
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, omega_num=87, plot=True)
        assert len(mag_ret) == 87

        # Change the default number of samples
        ct.config.defaults['freqplot.number_of_samples'] = 76
        mag_ret, phase_ret, omega_ret = ct.bode_plot(self.sys, plot=True)
        assert len(mag_ret) == 76

        # Override the default number of samples
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, omega_num=87, plot=True)
        assert len(mag_ret) == 87

    @pytest.mark.usefixtures("legacy_plot_signature")
    def test_bode_feature_periphery_decade(self, mplcleanup):
        # Generate a sample Bode plot to figure out the range it uses
        ct.reset_defaults()     # Make sure starting state is correct
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, Hz=False, plot=True)
        omega_min, omega_max = omega_ret[[0,  -1]]

        # Reset the periphery decade value (should add one decade on each end)
        ct.config.defaults['freqplot.feature_periphery_decades'] = 2
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, Hz=False, plot=True)
        np.testing.assert_almost_equal(omega_ret[0], omega_min/10)
        np.testing.assert_almost_equal(omega_ret[-1], omega_max * 10)

        # Make sure it also works in rad/sec, in opposite direction
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, Hz=True, plot=True)
        omega_min, omega_max = omega_ret[[0,  -1]]
        ct.config.defaults['freqplot.feature_periphery_decades'] = 1
        mag_ret, phase_ret, omega_ret = ct.bode_plot(
            self.sys, Hz=True, plot=True)
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
        with pytest.warns(UserWarning, match="NumPy matrix class no longer"):
            ct.use_legacy_defaults('0.8.3')
            ct.reset_defaults()

        with pytest.warns(UserWarning, match="NumPy matrix class no longer"):
            ct.use_legacy_defaults('0.8.4')
            assert ct.config.defaults['forced_response.return_x'] is True

        ct.use_legacy_defaults('0.9.0')
        assert isinstance(ct.ss(0, 0, 0, 1).D, np.ndarray)
        assert not isinstance(ct.ss(0, 0, 0, 1).D, np.matrix)

        # test that old versions don't raise a problem (besides Numpy warning)
        for ver in ['REL-0.1', 'control-0.3a', '0.6c', '0.8.2', '0.1']:
            with pytest.warns(
                    UserWarning, match="NumPy matrix class no longer"):
                ct.use_legacy_defaults(ver)

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
        nlsys = ct.NonlinearIOSystem(
            lambda t, x, u: u * x * x,
            lambda t, x, u: x, inputs=1, outputs=1)
        assert nlsys.dt == dt

    def test_change_default_dt_static(self):
        """Test that static gain systems always have dt=None"""
        ct.set_defaults('control', default_dt=0)
        assert ct.tf(1, 1).dt is None
        assert ct.ss([], [], [], 1).dt is None

    def test_get_param_last(self):
        """Test _get_param last keyword"""
        kwargs = {'first': 1, 'second': 2}

        with pytest.raises(TypeError, match="unrecognized keyword.*second"):
            assert ct.config._get_param(
                'config', 'first', kwargs, pop=True, last=True) == 1

        assert ct.config._get_param(
            'config', 'second', kwargs, pop=True, last=True) == 2

    def test_system_indexing(self):
        # Default renaming
        sys = ct.TransferFunction(
            [ [   [1],    [2],    [3]], [   [3],    [4],    [5]] ],
            [ [[1, 2], [1, 3], [1, 4]], [[1, 4], [1, 5], [1, 6]] ], 0.5)
        sys1 = sys[1:, 1:]
        assert sys1.name == sys.name + '$indexed'

        # Reset the format
        ct.config.set_defaults(
            'iosys', indexed_system_name_prefix='PRE',
            indexed_system_name_suffix='POST')
        sys2 = sys[1:, 1:]
        assert sys2.name == 'PRE' + sys.name + 'POST'

    @pytest.mark.parametrize("kwargs", [
        {},
        {'name': 'mysys'},
        {'inputs': 1},
        {'inputs': 'u'},
        {'outputs': 1},
        {'outputs': 'y'},
        {'states': 1},
        {'states': 'x'},
        {'inputs': 1, 'outputs': 'y', 'states': 'x'},
        {'dt': 0.1}
    ])
    def test_repr_format(self, kwargs):
        sys = ct.ss([[1]], [[1]], [[1]], [[0]], **kwargs)
        new = eval(repr(sys), None, {'StateSpace':ct.StateSpace, 'array':np.array})
        for attr in ['A', 'B', 'C', 'D']:
            assert getattr(new, attr) == getattr(sys, attr)
        for prop in ['input_labels', 'output_labels', 'state_labels']:
            assert getattr(new, attr) == getattr(sys, attr)
        if 'name' in kwargs:
            assert new.name == sys.name


def test_config_context_manager():
    # Make sure we can temporarily set the value of a parameter
    default_val = ct.config.defaults['statesp.latex_repr_type']
    with ct.config.defaults({'statesp.latex_repr_type': 'new value'}):
        assert ct.config.defaults['statesp.latex_repr_type'] != default_val
        assert ct.config.defaults['statesp.latex_repr_type'] == 'new value'
    assert ct.config.defaults['statesp.latex_repr_type'] == default_val

    # OK to call the context manager and not do anything with it
    ct.config.defaults({'statesp.latex_repr_type': 'new value'})
    assert ct.config.defaults['statesp.latex_repr_type'] == default_val

    with pytest.raises(ValueError, match="unknown parameter 'unknown'"):
        with ct.config.defaults({'unknown': 'new value'}):
            pass
