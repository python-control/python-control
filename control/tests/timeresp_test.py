"""timeresp_test.py - test time response functions"""

from copy import copy
from math import isclose

import numpy as np
import pytest

import control as ct
from control import StateSpace, TransferFunction, c2d, isctime, ss2tf, tf2ss
from control.exception import pandas_check, slycot_check
from control.tests.conftest import slycotonly
from control.timeresp import _default_time_vector, _ideal_tfinal_and_dt, \
    forced_response, impulse_response, initial_response, step_info, \
    step_response


class TSys:
    """Struct of test system"""
    def __init__(self, sys=None, call_kwargs=None):
        self.sys = sys
        self.kwargs = call_kwargs if call_kwargs else {}

    def __repr__(self):
        """Show system when debugging"""
        return self.sys.__repr__()


class TestTimeresp:

    @pytest.fixture
    def tsystem(self, request):
        """Define some test systems"""

        """continuous"""
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        siso_ss1 = TSys(StateSpace(A, B, C, D, 0))
        siso_ss1.t = np.linspace(0, 1, 10)
        siso_ss1.ystep = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234,
                                   39.1165, 42.3227, 44.9694, 47.1599,
                                   48.9776])
        siso_ss1.X0 = np.array([[.5], [1.]])
        siso_ss1.yinitial = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118,
                                      1.9092, 1.1508, 0.5833, 0.1645, -0.1391])
        ss1 = siso_ss1.sys

        """D=0, continuous"""
        siso_ss2 = TSys(StateSpace(ss1.A, ss1.B, ss1.C, 0, 0))
        siso_ss2.t = siso_ss1.t
        siso_ss2.ystep = siso_ss1.ystep - 9
        siso_ss2.initial = siso_ss1.yinitial - 9
        siso_ss2.yimpulse = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766,
                                      31.7344, 26.1668, 21.6292, 17.9245,
                                      14.8945])

        """System with unspecified timebase"""
        siso_ss2_dtnone = TSys(StateSpace(ss1.A, ss1.B, ss1.C, 0, None))
        siso_ss2_dtnone.t = np.arange(0, 10, 1.)
        siso_ss2_dtnone.ystep = np.array([0., 86., -72., 230., -360.,  806.,
                                          -1512.,  3110., -6120., 12326.])

        siso_tf1 = TSys(TransferFunction([1], [1, 2, 1], 0))

        siso_tf2 = copy(siso_ss1)
        siso_tf2.sys = ss2tf(siso_ss1.sys)

        """MIMO system, contains `siso_ss1` twice"""
        mimo_ss1 = copy(siso_ss1)
        A = np.zeros((4, 4))
        A[:2, :2] = siso_ss1.sys.A
        A[2:, 2:] = siso_ss1.sys.A
        B = np.zeros((4, 2))
        B[:2, :1] = siso_ss1.sys.B
        B[2:, 1:] = siso_ss1.sys.B
        C = np.zeros((2, 4))
        C[:1, :2] = siso_ss1.sys.C
        C[1:, 2:] = siso_ss1.sys.C
        D = np.zeros((2, 2))
        D[:1, :1] = siso_ss1.sys.D
        D[1:, 1:] = siso_ss1.sys.D
        mimo_ss1.sys = StateSpace(A, B, C, D)

        """MIMO system, contains `siso_ss2` twice"""
        mimo_ss2 = copy(siso_ss2)
        A = np.zeros((4, 4))
        A[:2, :2] = siso_ss2.sys.A
        A[2:, 2:] = siso_ss2.sys.A
        B = np.zeros((4, 2))
        B[:2, :1] = siso_ss2.sys.B
        B[2:, 1:] = siso_ss2.sys.B
        C = np.zeros((2, 4))
        C[:1, :2] = siso_ss2.sys.C
        C[1:, 2:] = siso_ss2.sys.C
        D = np.zeros((2, 2))
        mimo_ss2.sys = StateSpace(A, B, C, D, 0)

        """discrete"""
        siso_dtf0 = TSys(TransferFunction([1.], [1., 0.], 1.))
        siso_dtf0.t = np.arange(4)
        siso_dtf0.yimpulse = [0., 1., 0., 0.]

        siso_dtf1 =  TSys(TransferFunction([1], [1, 1, 0.25], True))
        siso_dtf1.t = np.arange(0, 5, 1)
        siso_dtf1.ystep = np.array([0.  , 0.  , 1.  , 0.  , 0.75])

        siso_dtf2 = TSys(TransferFunction([1], [1, 1, 0.25], 0.2))
        siso_dtf2.t = np.arange(0, 5, 0.2)
        siso_dtf2.ystep = np.array(
            [0.    , 0.    , 1.    , 0.    , 0.75  , 0.25  ,
             0.5625, 0.375 , 0.4844, 0.4219, 0.457 , 0.4375,
             0.4482, 0.4424, 0.4456, 0.4438, 0.4448, 0.4443,
             0.4445, 0.4444, 0.4445, 0.4444, 0.4445, 0.4444,
             0.4444])

        """Time step which leads to rounding errors for time vector length"""
        num = [-0.10966442, 0.12431949]
        den = [1., -1.86789511, 0.88255018]
        dt = 0.12493963338370018
        siso_dtf3 = TSys(TransferFunction(num, den, dt))
        siso_dtf3.t = np.linspace(0, 9*dt, 10)
        siso_dtf3.ystep = np.array(
            [ 0.    , -0.1097, -0.1902, -0.2438, -0.2729,
             -0.2799, -0.2674, -0.2377, -0.1934, -0.1368])

        """dtf1 converted statically, because Slycot and Scipy produce
        different realizations, wich means different initial condtions,"""
        siso_dss1 = copy(siso_dtf1)
        siso_dss1.sys = StateSpace([[-1., -0.25],
                                    [ 1.,  0.]],
                                   [[1.],
                                    [0.]],
                                   [[0., 1.]],
                                   [[0.]],
                                   True)
        siso_dss1.X0 = [0.5, 1.]
        siso_dss1.yinitial = np.array([1., 0.5, -0.75, 0.625, -0.4375])

        siso_dss2 = copy(siso_dtf2)
        siso_dss2.sys = tf2ss(siso_dtf2.sys)

        mimo_dss1 = TSys(StateSpace(ss1.A, ss1.B, ss1.C, ss1.D, True))
        mimo_dss1.t = np.arange(0, 5, 0.2)

        mimo_dss2 = copy(mimo_ss1)
        mimo_dss2.sys = c2d(mimo_ss1.sys, mimo_ss1.t[1]-mimo_ss1.t[0])

        mimo_tf2 = copy(mimo_ss2)
        tf_ = ss2tf(siso_ss2.sys)
        mimo_tf2.sys = TransferFunction(
            [[tf_.num[0][0], [0]], [[0], tf_.num[0][0]]],
            [[tf_.den[0][0], [1]], [[1], tf_.den[0][0]]],
            0)

        mimo_dtf1 = copy(siso_dtf1)
        tf_ = siso_dtf1.sys
        mimo_dtf1.sys = TransferFunction(
            [[tf_.num[0][0], [0]], [[0], tf_.num[0][0]]],
            [[tf_.den[0][0], [1]], [[1], tf_.den[0][0]]],
            True)

        # for pole cancellation tests
        pole_cancellation = TSys(TransferFunction(
            [1.067e+05, 5.791e+04],
            [10.67, 1.067e+05, 5.791e+04]))

        no_pole_cancellation = TSys(TransferFunction(
            [1.881e+06],
            [188.1, 1.881e+06]))

        # System Type 1 - Step response not stationary:  G(s)=1/s(s+1)
        siso_tf_type1 = TSys(TransferFunction(1, [1, 1, 0]))
        siso_tf_type1.step_info = {
             'RiseTime': np.nan,
             'SettlingTime': np.nan,
             'SettlingMin': np.nan,
             'SettlingMax': np.nan,
             'Overshoot': np.nan,
             'Undershoot': np.nan,
             'Peak': np.inf,
             'PeakTime': np.inf,
             'SteadyStateValue': np.nan}

        # SISO under shoot response and positive final value
        # G(s)=(-s+1)/(s²+s+1)
        siso_tf_kpos = TSys(TransferFunction([-1, 1], [1, 1, 1]))
        siso_tf_kpos.step_info = {
             'RiseTime': 1.242,
             'SettlingTime': 9.110,
             'SettlingMin': 0.90,
             'SettlingMax': 1.208,
             'Overshoot': 20.840,
             'Undershoot': 28.0,
             'Peak': 1.208,
             'PeakTime': 4.282,
             'SteadyStateValue': 1.0}

        # SISO under shoot response and negative final value
        # k=-1 G(s)=-(-s+1)/(s²+s+1)
        siso_tf_kneg = TSys(TransferFunction([1, -1], [1, 1, 1]))
        siso_tf_kneg.step_info = {
             'RiseTime': 1.242,
             'SettlingTime': 9.110,
             'SettlingMin': -1.208,
             'SettlingMax': -0.90,
             'Overshoot': 20.840,
             'Undershoot': 28.0,
             'Peak': 1.208,
             'PeakTime': 4.282,
             'SteadyStateValue': -1.0}

        siso_tf_asymptotic_from_neg1 = TSys(TransferFunction([-1, 1], [1, 1]))
        siso_tf_asymptotic_from_neg1.step_info = {
            'RiseTime': 2.197,
            'SettlingTime': 4.605,
            'SettlingMin': 0.9,
            'SettlingMax': 1.0,
            'Overshoot': 0,
            'Undershoot': 100.0,
            'Peak': 1.0,
            'PeakTime': 0.0,
            'SteadyStateValue': 1.0}
        siso_tf_asymptotic_from_neg1.kwargs = {
            'step_info': {'T': np.arange(0, 5, 1e-3)}}

        # example from matlab online help
        # https://www.mathworks.com/help/control/ref/stepinfo.html
        siso_tf_step_matlab = TSys(TransferFunction([1, 5, 5],
                                                    [1, 1.65, 5, 6.5, 2]))
        siso_tf_step_matlab.step_info = {
            'RiseTime': 3.8456,
            'SettlingTime': 27.9762,
            'SettlingMin': 2.0689,
            'SettlingMax': 2.6873,
            'Overshoot': 7.4915,
            'Undershoot': 0,
            'Peak': 2.6873,
            'PeakTime': 8.0530,
            'SteadyStateValue': 2.5}

        A = [[0.68, -0.34],
             [0.34, 0.68]]
        B = [[0.18, -0.05],
             [0.04, 0.11]]
        C = [[0, -1.53],
             [-1.12, -1.10]]
        D = [[0, 0],
             [0.06, -0.37]]
        mimo_ss_step_matlab = TSys(StateSpace(A, B, C, D, 0.2))
        mimo_ss_step_matlab.kwargs['step_info'] = {'T': 4.6}
        mimo_ss_step_matlab.step_info = [[
             {'RiseTime': 0.6000,
              'SettlingTime': 3.0000,
              'SettlingMin': -0.5999,
              'SettlingMax': -0.4689,
              'Overshoot': 15.5072,
              'Undershoot': 0.,
              'Peak': 0.5999,
              'PeakTime': 1.4000,
              'SteadyStateValue': -0.5193},
             {'RiseTime': 0.,
              'SettlingTime': 3.6000,
              'SettlingMin': -0.2797,
              'SettlingMax': -0.1043,
              'Overshoot': 118.9918,
              'Undershoot': 0,
              'Peak': 0.2797,
              'PeakTime': .6000,
              'SteadyStateValue': -0.1277}],
            [{'RiseTime': 0.4000,
              'SettlingTime': 2.8000,
              'SettlingMin': -0.6724,
              'SettlingMax': -0.5188,
              'Overshoot': 24.6476,
              'Undershoot': 11.1224,
              'Peak': 0.6724,
              'PeakTime': 1,
              'SteadyStateValue': -0.5394},
              {'RiseTime': 0.0000, # (*)
              'SettlingTime': 3.4000,
              'SettlingMin': -0.4350,  # (*)
              'SettlingMax': -0.1485,
              'Overshoot': 132.0170,
              'Undershoot': 0.,
              'Peak': 0.4350,
              'PeakTime': .2,
              'SteadyStateValue': -0.1875}]]
              # (*): MATLAB gives 0.4 for RiseTime and -0.1034 for
              # SettlingMin, but it is unclear what 10% and 90% of
              # the steady state response mean, when the step for
              # this channel does not start a 0.

        siso_ss_step_matlab = copy(mimo_ss_step_matlab)
        siso_ss_step_matlab.sys = siso_ss_step_matlab.sys[1, 0]
        siso_ss_step_matlab.step_info = siso_ss_step_matlab.step_info[1][0]

        Ta = [[siso_tf_kpos, siso_tf_kneg, siso_tf_step_matlab],
              [siso_tf_step_matlab, siso_tf_kpos, siso_tf_kneg]]
        mimo_tf_step_info = TSys(TransferFunction(
            [[Ti.sys.num[0][0] for Ti in Tr] for Tr in Ta],
            [[Ti.sys.den[0][0] for Ti in Tr] for Tr in Ta]))
        mimo_tf_step_info.step_info = [[Ti.step_info for Ti in Tr]
                                       for Tr in Ta]
        # enforce enough sample points for all channels (they have different
        # characteristics)
        mimo_tf_step_info.kwargs['step_info'] = {'T_num': 2000}

        systems = locals()
        if isinstance(request.param, str):
            return systems[request.param]
        else:
            return [systems[sys] for sys in request.param]

    @pytest.mark.parametrize(
        "kwargs",
        [{},
         {'X0': 0},
         {'X0': np.array([0, 0])},
         {'X0': 0, 'return_x': True},
         ])
    @pytest.mark.parametrize("tsystem", ["siso_ss1"], indirect=True)
    def test_step_response_siso(self, tsystem, kwargs):
        """Test SISO system step response"""
        sys = tsystem.sys
        t = tsystem.t
        yref = tsystem.ystep
        # SISO call
        out = step_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    @pytest.mark.parametrize("tsystem", ["mimo_ss1"], indirect=True)
    def test_step_response_mimo(self, tsystem):
        """Test MIMO system, which contains `siso_ss1` twice."""
        sys = tsystem.sys
        t = tsystem.t
        yref = tsystem.ystep
        _t, y_00 = step_response(sys, T=t, input=0, output=0)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)

        _t, y_11 = step_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_11, yref, decimal=4)

        _t, y_01 = step_response(
            sys, T=t, input_indices=[0], output_indices=[1])
        np.testing.assert_array_almost_equal(y_01, 0 * yref, decimal=4)

        # Make sure we get the same result using MIMO step response
        response = step_response(sys, T=t)
        np.testing.assert_allclose(response.y[0, 0, :], y_00)
        np.testing.assert_allclose(response.y[1, 1, :], y_11)
        np.testing.assert_allclose(response.u[0, 0, :], 1)
        np.testing.assert_allclose(response.u[1, 0, :], 0)
        np.testing.assert_allclose(response.u[0, 1, :], 0)
        np.testing.assert_allclose(response.u[1, 1, :], 1)

        # Index lists not yet implemented
        with pytest.raises(NotImplementedError, match="list of .* indices"):
            step_response(
                sys, timepts=t, input_indices=[0, 1], output_indices=[1])

    @pytest.mark.parametrize("tsystem", ["mimo_ss1"], indirect=True)
    def test_step_response_return(self, tsystem):
        """Verify continuous and discrete time use same return conventions."""
        sysc = tsystem.sys
        sysd = c2d(sysc, 1)            # discrete-time system
        Tvec = np.linspace(0, 10, 11)  # make sure to use integer times 0..10
        Tc, youtc = step_response(sysc, Tvec, input=0)
        Td, youtd = step_response(sysd, Tvec, input=0)
        np.testing.assert_array_equal(Tc.shape, Td.shape)
        np.testing.assert_array_equal(youtc.shape, youtd.shape)

    @pytest.mark.parametrize("dt", [0, 1], ids=["continuous", "discrete"])
    def test_step_nostates(self, dt):
        """Constant system, continuous and discrete time.

        gh-374 "Bug in step_response()"
        """
        sys = TransferFunction([1], [1], dt)
        t, y = step_response(sys)
        np.testing.assert_allclose(y, np.ones(len(t)))

    def assert_step_info_match(self, sys, info, info_ref):
        """Assert reasonable step_info accuracy."""
        if sys.isdtime(strict=True):
            dt = sys.dt
        else:
            _, dt = _ideal_tfinal_and_dt(sys, is_step=True)

        for k in ['RiseTime', 'SettlingTime', 'PeakTime']:
            np.testing.assert_allclose(info[k], info_ref[k], atol=dt,
                                       err_msg=f"{k} does not match")
        for k in ['Overshoot', 'Undershoot', 'Peak', 'SteadyStateValue']:
            np.testing.assert_allclose(info[k], info_ref[k], rtol=5e-3,
                                       err_msg=f"{k} does not match")

        # steep gradient right after RiseTime
        absrefinf = np.abs(info_ref['SteadyStateValue'])
        if info_ref['RiseTime'] > 0:
            y_next_sample_max = 0.8*absrefinf/info_ref['RiseTime']*dt
        else:
            y_next_sample_max = 0
        for k in ['SettlingMin', 'SettlingMax']:
            if (np.abs(info_ref[k]) - 0.9 * absrefinf) > y_next_sample_max:
                # local min/max peak well after signal has risen
                np.testing.assert_allclose(info[k], info_ref[k], rtol=1e-3)

    @pytest.mark.parametrize(
        "yfinal", [True, False], ids=["yfinal", "no yfinal"])
    @pytest.mark.parametrize(
        "systype, time_2d",
        [("ltisys", False),
         ("time response", False),
         ("time response", True),
         ],
        ids=["ltisys", "time response (n,)", "time response (1,n)"])
    @pytest.mark.parametrize(
        "tsystem",
        ["siso_tf_step_matlab",
         "siso_ss_step_matlab",
         "siso_tf_kpos",
         "siso_tf_kneg",
         "siso_tf_type1",
         "siso_tf_asymptotic_from_neg1"],
        indirect=["tsystem"])
    def test_step_info(self, tsystem, systype, time_2d, yfinal):
        """Test step info for SISO systems."""
        step_info_kwargs = tsystem.kwargs.get('step_info', {})
        if systype == "time response":
            # simulate long enough for steady state value
            tfinal = 3 * tsystem.step_info['SettlingTime']
            if np.isnan(tfinal):
                pytest.skip("test system does not settle")
            t, y = step_response(tsystem.sys, T=tfinal, T_num=5000)
            sysdata = y
            step_info_kwargs['T'] = t[np.newaxis, :] if time_2d else t
        else:
            sysdata = tsystem.sys
        if yfinal:
            step_info_kwargs['yfinal'] = tsystem.step_info['SteadyStateValue']

        info = step_info(sysdata, **step_info_kwargs)

        self.assert_step_info_match(tsystem.sys, info, tsystem.step_info)

    @pytest.mark.parametrize(
        "yfinal", [True, False], ids=["yfinal", "no_yfinal"])
    @pytest.mark.parametrize(
        "systype", ["ltisys", "time response"])
    @pytest.mark.parametrize(
        "tsystem",
        ['mimo_ss_step_matlab',
         pytest.param('mimo_tf_step_info', marks=slycotonly)],
        indirect=["tsystem"])
    def test_step_info_mimo(self, tsystem, systype, yfinal):
        """Test step info for MIMO systems."""
        step_info_kwargs = tsystem.kwargs.get('step_info', {})
        if systype == "time response":
            tfinal = 3 * max([S['SettlingTime']
                              for Srow in tsystem.step_info for S in Srow])
            t, y = step_response(tsystem.sys, T=tfinal, T_num=5000)
            sysdata = y
            step_info_kwargs['T'] = t
        else:
            sysdata = tsystem.sys
        if yfinal:
            step_info_kwargs['yfinal'] = [[S['SteadyStateValue']
                                           for S in Srow]
                                          for Srow in tsystem.step_info]

        info_dict = step_info(sysdata, **step_info_kwargs)

        for i, row in enumerate(info_dict):
            for j, info in enumerate(row):
                self.assert_step_info_match(tsystem.sys,
                                            info, tsystem.step_info[i][j])

    def test_step_info_invalid(self):
        """Call step_info with invalid parameters."""
        with pytest.raises(ValueError, match="time series data convention"):
            step_info(["not numeric data"])
        with pytest.raises(ValueError, match="time series data convention"):
            step_info(np.ones((10, 15)))                     # invalid shape
        with pytest.raises(ValueError, match="matching time vector"):
            step_info(np.ones(15), T=np.linspace(0, 1, 20))  # time too long
        with pytest.raises(ValueError, match="matching time vector"):
            step_info(np.ones((2, 2, 15)))                   # no time vector

    @pytest.mark.parametrize("tsystem",
                             [("no_pole_cancellation", "pole_cancellation")],
                             indirect=True)
    def test_step_pole_cancellation(self, tsystem):
        # confirm that pole-zero cancellation doesn't perturb results
        # https://github.com/python-control/python-control/issues/440
        step_info_no_cancellation = step_info(tsystem[0].sys)
        step_info_cancellation = step_info(tsystem[1].sys)
        self.assert_step_info_match(tsystem[0].sys,
                                    step_info_no_cancellation,
                                    step_info_cancellation)

    @pytest.mark.parametrize(
        "tsystem, kwargs",
        [("siso_ss2", {}),
         ("siso_ss2", {'return_x': True}),
         ("siso_dtf0", {})],
        indirect=["tsystem"])
    def test_impulse_response_siso(self, tsystem, kwargs):
        """Test impulse response of SISO systems."""
        sys = tsystem.sys
        t = tsystem.t
        yref = tsystem.yimpulse

        out = impulse_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    @pytest.mark.parametrize("tsystem", ["mimo_ss2"], indirect=True)
    def test_impulse_response_mimo(self, tsystem):
        """"Test impulse response of MIMO systems."""
        sys = tsystem.sys
        t = tsystem.t

        yref = tsystem.yimpulse
        _t, y_00 = impulse_response(sys, T=t, input=0, output=0)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)

        _t, y_11 = impulse_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_11, yref, decimal=4)

        _t, y_01 = impulse_response(
            sys, T=t, input_indices=[0], output_indices=[1])
        np.testing.assert_array_almost_equal(y_01, 0 * yref, decimal=4)

        yref_notrim = np.zeros((2, len(t)))
        yref_notrim[:1, :] = yref
        _t, yy = impulse_response(sys, T=t, input=0)
        np.testing.assert_array_almost_equal(yy[:,0,:], yref_notrim, decimal=4)

        # Index lists not yet implemented
        with pytest.raises(NotImplementedError, match="list of .* indices"):
            impulse_response(
                sys, timepts=t, input_indices=[0, 1], output_indices=[1])

    @pytest.mark.parametrize("tsystem", ["siso_tf1"], indirect=True)
    def test_discrete_time_impulse(self, tsystem):
        # discrete-time impulse sampled version should match cont time
        dt = 0.1
        t = np.arange(0, 3, dt)
        sys = tsystem.sys
        sysdt = sys.sample(dt, 'impulse')
        np.testing.assert_array_almost_equal(impulse_response(sys, t)[1],
                                             impulse_response(sysdt, t)[1])

    def test_discrete_time_impulse_input(self):
        # discrete-time impulse input, Only one active input for each trace
        A = [[.5, 0.25],[.0, .5]]
        B = [[1., 0,],[0., 1.]]
        C = [[1., 0.],[0., 1.]]
        D = [[0., 0.],[0., 0.]]
        dt = True
        sysd = ct.ss(A,B,C,D, dt=dt)
        response = ct.impulse_response(sysd,T=dt*3)

        Uexpected = np.zeros((2,2,4), dtype=float).astype(object)
        Uexpected[0,0,0] = 1./dt
        Uexpected[1,1,0] = 1./dt

        np.testing.assert_array_equal(response.inputs,Uexpected)

        dt = 0.5
        sysd = ct.ss(A,B,C,D, dt=dt)
        response = ct.impulse_response(sysd,T=dt*3)

        Uexpected = np.zeros((2,2,4), dtype=float).astype(object)
        Uexpected[0,0,0] = 1./dt
        Uexpected[1,1,0] = 1./dt

        np.testing.assert_array_equal(response.inputs,Uexpected)

    @pytest.mark.parametrize("tsystem", ["siso_ss1"], indirect=True)
    def test_impulse_response_warnD(self, tsystem):
        """Test warning about direct feedthrough"""
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            _ = impulse_response(tsystem.sys, tsystem.t)

    @pytest.mark.parametrize(
        "kwargs",
        [{},
         {'X0': 0},
         {'X0': np.array([0.5, 1])},
         {'X0': np.array([[0.5], [1]])},
         {'X0': np.array([0.5, 1]), 'return_x': True},
         ])
    @pytest.mark.parametrize("tsystem", ["siso_ss1"], indirect=True)
    def test_initial_response(self, tsystem, kwargs):
        """Test initial response of SISO system"""
        sys = tsystem.sys
        t = tsystem.t
        x0 = kwargs.get('X0', 0)
        yref = tsystem.yinitial if np.any(x0) else np.zeros_like(t)

        out = initial_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    @pytest.mark.parametrize("tsystem", ["mimo_ss1"], indirect=True)
    def test_initial_response_mimo(self, tsystem):
        """Test initial response of MIMO system"""
        sys = tsystem.sys
        t = tsystem.t
        x0 = np.array([[.5], [1.], [.5], [1.]])
        yref = tsystem.yinitial
        yref_notrim = np.broadcast_to(yref, (2, len(t)))

        _t, y_00 = initial_response(sys, T=t, X0=x0, output=0)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)
        _t, y_11 = initial_response(sys, T=t, X0=x0, output=1)
        np.testing.assert_array_almost_equal(y_11, yref, decimal=4)
        _t, yy = initial_response(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yy, yref_notrim, decimal=4)

    @pytest.mark.parametrize("tsystem",
                             ["siso_ss1", "siso_tf2"],
                             indirect=True)
    def test_forced_response_step(self, tsystem):
        """Test forced response of SISO systems as step response"""
        sys = tsystem.sys
        t = tsystem.t
        u = np.ones_like(t, dtype=float)
        yref = tsystem.ystep

        tout, yout = forced_response(sys, t, u)
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    @pytest.mark.parametrize("u",
                             [np.zeros((10,), dtype=float),
                              0]  # special algorithm
                             )
    @pytest.mark.parametrize("tsystem", ["siso_ss1", "siso_tf2"],
                             indirect=True)
    def test_forced_response_initial(self, tsystem, u):
        """Test forced response of SISO system as intitial response."""
        sys = tsystem.sys
        t = tsystem.t
        x0 = tsystem.X0
        yref = tsystem.yinitial

        if isinstance(sys, StateSpace):
            tout, yout = forced_response(sys, t, u, X0=x0)
            np.testing.assert_array_almost_equal(tout, t)
            np.testing.assert_array_almost_equal(yout, yref, decimal=4)
        else:
            with pytest.warns(UserWarning, match="Non-zero initial condition "
                              "given for transfer function"):
                tout, yout = forced_response(sys, t, u, X0=x0)

    @pytest.mark.parametrize("tsystem, useT",
                             [("mimo_ss1", True),
                              ("mimo_dss2", True),
                              ("mimo_dss2", False)],
                             indirect=["tsystem"])
    def test_forced_response_mimo(self, tsystem, useT):
        """Test forced response of MIMO system"""
        # first system: initial value, second system: step response
        sys = tsystem.sys
        t = tsystem.t
        u = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1., 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        x0 = np.array([[.5], [1], [0], [0]])
        yref = np.vstack([tsystem.yinitial, tsystem.ystep])

        if useT:
            _t, yout = forced_response(sys, t, u, x0)
        else:
            _t, yout = forced_response(sys, U=u, X0=x0)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    @pytest.mark.usefixtures("editsdefaults")
    def test_forced_response_legacy(self):
        # Define a system for testing
        sys = ct.rss(2, 1, 1)
        T = np.linspace(0, 10, 10)
        U = np.sin(T)

        """Make sure that legacy version of forced_response works"""
        with pytest.warns(
                UserWarning, match="NumPy matrix class no longer"):
            ct.config.use_legacy_defaults("0.8.4")
        # forced_response returns x by default
        t, y = ct.step_response(sys, T)
        t, y, x = ct.forced_response(sys, T, U)

        ct.config.use_legacy_defaults("0.9.0")
        # forced_response returns input/output by default
        t, y = ct.step_response(sys, T)
        t, y = ct.forced_response(sys, T, U)
        t, y, x = ct.forced_response(sys, T, U, return_x=True)

    @pytest.mark.parametrize(
        "tsystem, fr_kwargs, refattr",
        [pytest.param("siso_ss1",
                      {'T':  np.linspace(0, 1, 10)}, 'yinitial',
                      id="ctime no U"),
         pytest.param("siso_dss1",
                      {'T': np.arange(0, 5, 1,)}, 'yinitial',
                      id="dt=True, no U"),
         pytest.param("siso_dtf1",
                      {'U': np.ones(5,)}, 'ystep',
                      id="dt=True, no T"),
         pytest.param("siso_dtf2",
                      {'U': np.ones(25,)}, 'ystep',
                      id="dt=0.2, no T"),
         pytest.param("siso_ss2_dtnone",
                      {'U': np.ones(10,)}, 'ystep',
                      id="dt=None, no T"),
         pytest.param("siso_dtf3",
                      {'U': np.ones(10,)}, 'ystep',
                      id="dt with rounding error, no T"),
         ],
        indirect=["tsystem"])
    def test_forced_response_T_U(self, tsystem, fr_kwargs, refattr):
        """Test documented forced_response behavior for parameters T and U."""
        if refattr == 'yinitial':
            fr_kwargs['X0'] = tsystem.X0
        t, y = forced_response(tsystem.sys, **fr_kwargs)
        np.testing.assert_allclose(t, tsystem.t)
        np.testing.assert_allclose(y, getattr(tsystem, refattr),
                                   rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("tsystem", ["siso_ss1"], indirect=True)
    def test_forced_response_invalid_c(self, tsystem):
        """Test invalid parameters."""
        with pytest.raises(TypeError,
                           match="StateSpace.*or.*TransferFunction"):
            forced_response("not a system")
        with pytest.raises(ValueError, match="T.*is mandatory for continuous"):
            forced_response(tsystem.sys)
        with pytest.raises(ValueError, match="time values must be equally "
                                             "spaced"):
            forced_response(tsystem.sys, [0, 0.1, 0.12, 0.4])

    @pytest.mark.parametrize("tsystem", ["siso_dss2"], indirect=True)
    def test_forced_response_invalid_d(self, tsystem):
        """Test invalid parameters dtime with sys.dt > 0."""
        with pytest.raises(ValueError, match="can't both be zero"):
            forced_response(tsystem.sys)
        with pytest.raises(ValueError, match="Parameter `U`: Wrong shape"):
            forced_response(tsystem.sys,
                            T=tsystem.t, U=np.random.randn(1, 12))
        with pytest.raises(ValueError, match="Parameter `U`: Wrong shape"):
            forced_response(tsystem.sys,
                            T=tsystem.t, U=np.random.randn(12))
        with pytest.raises(ValueError, match="must match sampling time"):
            forced_response(tsystem.sys, T=tsystem.t*0.9)
        with pytest.raises(ValueError, match="must be multiples of "
                                             "sampling time"):
            forced_response(tsystem.sys, T=tsystem.t*1.1)
        # but this is ok
        forced_response(tsystem.sys, T=tsystem.t*2)

    @pytest.mark.parametrize("u, x0, xtrue",
                             [(np.zeros((10,)),
                               np.array([2., 3.]),
                               np.vstack([np.linspace(2, 5, 10),
                                          np.full((10,), 3)])),
                              (np.ones((10,)),
                               np.array([0., 0.]),
                               np.vstack([0.5 * np.linspace(0, 1, 10)**2,
                                          np.linspace(0, 1, 10)])),
                              (np.linspace(0, 1, 10),
                               np.array([0., 0.]),
                               np.vstack([np.linspace(0, 1, 10)**3 / 6.,
                                          np.linspace(0, 1, 10)**2 / 2.]))],
                             ids=["zeros", "ones", "linear"])
    def test_lsim_double_integrator(self, u, x0, xtrue):
        """Test forced response of double integrator"""
        # Note: scipy.signal.lsim fails if A is not invertible
        A = np.array([[0., 1.],
                      [0., 0.]])
        B = np.array([[0.],
                      [1.]])
        C = np.array([[1., 0.]])
        D = 0.
        sys = StateSpace(A, B, C, D)
        t = np.linspace(0, 1, 10)

        _t, yout, xout = forced_response(sys, t, u, x0, return_x=True)
        np.testing.assert_array_almost_equal(xout, xtrue, decimal=6)
        ytrue = np.squeeze(np.asarray(C @ xtrue))
        np.testing.assert_array_almost_equal(yout, ytrue, decimal=6)


    @slycotonly
    def test_step_robustness(self):
        "Test robustness os step_response against denomiantors: gh-240"
        # Create 2 input, 2 output system
        num =  [[[0], [1]],           [[1],    [0]]]

        den1 = [[[1], [1,1]],         [[1, 4], [1]]]
        sys1 = TransferFunction(num, den1)

        den2 = [[[1], [1e-10, 1, 1]], [[1, 4], [1]]]   # slight perturbation
        sys2 = TransferFunction(num, den2)

        t1, y1 = step_response(sys1, input=0, T=2, T_num=100)
        t2, y2 = step_response(sys2, input=0, T=2, T_num=100)
        np.testing.assert_array_almost_equal(y1, y2)


    @pytest.mark.parametrize(
        "tfsys, tfinal",
        [(TransferFunction(1, [1, .5]), 13.81551),        #  pole at 0.5
         (TransferFunction(1, [1, .5]).sample(.1), 25),  # discrete pole at 0.5
         (TransferFunction(1, [1, .5, 0]), 25)])         # poles at 0.5 and 0
    def test_auto_generated_time_vector_tfinal(self, tfsys, tfinal):
        """Confirm a TF with a pole at p simulates for tfinal seconds"""
        ideal_tfinal, ideal_dt = _ideal_tfinal_and_dt(tfsys)
        np.testing.assert_allclose(ideal_tfinal, tfinal, rtol=1e-4)
        T = _default_time_vector(tfsys)
        np.testing.assert_allclose(T[-1], tfinal, atol=0.5*ideal_dt)

    @pytest.mark.parametrize("wn, zeta", [(10, 0), (100, 0), (100, .1)])
    def test_auto_generated_time_vector_dt_cont1(self, wn, zeta):
        """Confirm a TF with a natural frequency of wn rad/s gets a
        dt of 1/(ratio*wn)"""

        dtref = 0.25133 / wn

        tfsys = TransferFunction(1, [1, 2*zeta*wn, wn**2])
        np.testing.assert_almost_equal(_ideal_tfinal_and_dt(tfsys)[1], dtref,
                                       decimal=5)

    def test_auto_generated_time_vector_dt_cont2(self):
        """A sampled tf keeps its dt"""
        wn = 100
        zeta = .1
        tfsys = TransferFunction(1, [1, 2*zeta*wn, wn**2]).sample(.1)
        tfinal, dt = _ideal_tfinal_and_dt(tfsys)
        np.testing.assert_almost_equal(dt, .1)
        T, _ = initial_response(tfsys)
        np.testing.assert_almost_equal(np.diff(T[:2]), [.1])


    def test_default_timevector_long(self):
        """Test long time vector"""

        # TF with fast oscillations simulates only 5000 time steps
        # even with long tfinal
        wn = 100
        tfsys = TransferFunction(1, [1, 0, wn**2])
        tout = _default_time_vector(tfsys, tfinal=100)
        assert len(tout) == 5000

    @pytest.mark.parametrize("fun", [step_response,
                                     impulse_response,
                                     initial_response])
    def test_default_timevector_functions_c(self, fun):
        """Test that functions can calculate the time vector automatically"""
        sys = TransferFunction(1, [1, .5, 0])
        _tfinal, _dt = _ideal_tfinal_and_dt(sys)

        # test impose number of time steps
        tout, _ = fun(sys, T_num=10)
        assert len(tout) == 10

        # test impose final time
        tout, _ = fun(sys, T=100.)
        np.testing.assert_allclose(tout[-1], 100., atol=0.5*_dt)

    @pytest.mark.parametrize("fun", [step_response,
                                     impulse_response,
                                     initial_response])
    @pytest.mark.parametrize("dt", [0.1, 0.112])
    def test_default_timevector_functions_d(self, fun, dt):
        """Test that functions can calculate the time vector automatically"""
        sys = TransferFunction(1, [1, .5, 0], dt)

        # test impose number of time steps is ignored with dt given
        tout, _ = fun(sys, T_num=15)
        assert len(tout) != 15

        # test impose final time
        tout, _ = fun(sys, 100)
        np.testing.assert_allclose(tout[-1], 100., atol=0.5*dt)


    @pytest.mark.parametrize("tsystem",
                             ["siso_ss2",   # continuous
                              "siso_tf1",
                              "siso_dss1",  # unspecified sampling time
                              "siso_dtf1",
                              "siso_dss2",  # matching timebase
                              "siso_dtf2",
                              "siso_ss2_dtnone",  # undetermined timebase
                              "mimo_ss2",   # MIMO
                              pytest.param("mimo_tf2", marks=slycotonly),
                              "mimo_dss1",
                              pytest.param("mimo_dtf1", marks=slycotonly),
                              ],
                             indirect=True)
    @pytest.mark.parametrize("fun", [step_response,
                                     impulse_response,
                                     initial_response,
                                     forced_response])
    @pytest.mark.parametrize("squeeze", [None, True, False])
    def test_time_vector(self, tsystem, fun, squeeze):
        """Test time vector handling and correct output convention

        gh-239, gh-295
        """
        sys = tsystem.sys

        kw = {}
        if hasattr(tsystem, "t"):
            t = tsystem.t
            kw['T'] = t
            if fun == forced_response:
                kw['U'] = np.vstack([np.sin(t) for i in range(sys.ninputs)])
        elif fun == forced_response and isctime(sys, strict=True):
            pytest.skip("No continuous forced_response without time vector.")
        if hasattr(sys, "nstates") and sys.nstates is not None and \
           fun != impulse_response:
            kw['X0'] = np.arange(sys.nstates) + 1
        if sys.ninputs > 1 and fun in [step_response, impulse_response]:
            kw['input'] = 1
        if squeeze is not None:
            kw['squeeze'] = squeeze

        out = fun(sys, **kw)
        tout, yout = out[:2]

        assert tout.ndim == 1
        if hasattr(tsystem, 't'):
            # tout should always match t, which has shape (n, )
            np.testing.assert_allclose(tout, tsystem.t)
        elif fun == forced_response and sys.dt in [None, True]:
            np.testing.assert_allclose(np.diff(tout), 1.)

        if squeeze is False or not sys.issiso():
            assert yout.shape[0] == sys.noutputs
            assert yout.shape[-1] == tout.shape[0]
        else:
            assert yout.shape == tout.shape

        if sys.isdtime(strict=True) and sys.dt is not True and not \
                np.isclose(sys.dt, 0.5):
            kw['T'] = np.arange(0, 5, 0.5)  # incompatible timebase
            with pytest.raises(ValueError):
                fun(sys, **kw)

    @pytest.mark.parametrize("squeeze", [None, True, False])
    @pytest.mark.parametrize("tsystem", ["siso_dtf2"], indirect=True)
    def test_time_vector_interpolation(self, tsystem, squeeze):
        """Test time vector handling in case of interpolation.

        Interpolation of the input (to match scipy.signal.dlsim)

        gh-239, gh-295
        """
        sys = tsystem.sys
        t = np.arange(0, 10, 1.)
        u = np.sin(t)
        x0 = 0

        squeezekw = {} if squeeze is None else {"squeeze": squeeze}

        tout, yout = forced_response(sys, t, u, x0,
                                     interpolate=True, **squeezekw)
        if squeeze is False or sys.noutputs > 1:
            assert yout.shape[0] == sys.noutputs
            assert yout.shape[1] == tout.shape[0]
        else:
            assert yout.shape == tout.shape
        assert np.allclose(tout[1:] - tout[:-1], sys.dt)

    @pytest.mark.parametrize("tsystem", ["siso_dtf2"], indirect=True)
    def test_discrete_time_steps(self, tsystem):
        """Make sure rounding errors in sample time are handled properly

        These tests play around with the input time vector to make sure that
        small rounding errors don't generate spurious errors.

        gh-332
        """
        sys = tsystem.sys

        # Set up a time range and simulate
        T = np.arange(0, 100, 0.2)
        tout1, yout1 = step_response(sys, T)

        # Simulate every other time step
        T = np.arange(0, 100, 0.4)
        tout2, yout2 = step_response(sys, T)
        np.testing.assert_array_almost_equal(tout1[::2], tout2)
        np.testing.assert_array_almost_equal(yout1[::2], yout2)

        # Add a small error into some of the time steps
        T = np.arange(0, 100, 0.2)
        T[1:-2:2] -= 1e-12      # tweak second value and a few others
        tout3, yout3 = step_response(sys, T)
        np.testing.assert_array_almost_equal(tout1, tout3)
        np.testing.assert_array_almost_equal(yout1, yout3)

        # Add a small error into some of the time steps (w/ skipping)
        T = np.arange(0, 100, 0.4)
        T[1:-2:2] -= 1e-12      # tweak second value and a few others
        tout4, yout4 = step_response(sys, T)
        np.testing.assert_array_almost_equal(tout2, tout4)
        np.testing.assert_array_almost_equal(yout2, yout4)

        # Make sure larger errors *do* generate an error
        T = np.arange(0, 100, 0.2)
        T[1:-2:2] -= 1e-3      # change second value and a few others
        with pytest.raises(ValueError):
            step_response(sys, T)

    @pytest.mark.parametrize("tsystem", ["siso_ss1"], indirect=True)
    def test_time_series_data_convention_2D(self, tsystem):
        """Allow input time as 2D array (output should be 1D)"""
        tin = np.array(np.linspace(0, 10, 100), ndmin=2)
        t, y = step_response(tsystem.sys, tin)
        assert isinstance(t, np.ndarray) and not isinstance(t, np.matrix)
        assert t.ndim == 1
        assert y.ndim == 1  # SISO returns "scalar" output
        assert t.shape == y.shape  # Allows direct plotting of output

    @pytest.mark.usefixtures("editsdefaults")
    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf])
    @pytest.mark.parametrize("nstate, nout, ninp, squeeze, shape1, shape2", [
    #  state  out   in   squeeze  in/out      out-only
        [1,    1,    1,  None,   (8,),       (8,)],
        [2,    1,    1,  True,   (8,),       (8,)],
        [3,    1,    1,  False,  (1, 1, 8),  (1, 8)],
        [3,    2,    1,  None,   (2, 1, 8),  (2, 8)],
        [4,    2,    1,  True,   (2, 8),     (2, 8)],
        [5,    2,    1,  False,  (2, 1, 8),  (2, 8)],
        [3,    1,    2,  None,   (1, 2, 8),  (1, 8)],
        [4,    1,    2,  True,   (2, 8),     (8,)],
        [5,    1,    2,  False,  (1, 2, 8),  (1, 8)],
        [4,    2,    2,  None,   (2, 2, 8),  (2, 8)],
        [5,    2,    2,  True,   (2, 2, 8),  (2, 8)],
        [6,    2,    2,  False,  (2, 2, 8),  (2, 8)],
    ])
    def test_squeeze(self, fcn, nstate, nout, ninp, squeeze, shape1, shape2):
        # Define the system
        if fcn == ct.tf and (nout > 1 or ninp > 1) and not slycot_check():
            pytest.skip("Conversion of MIMO systems to transfer functions "
                        "requires slycot.")
        else:
            sys = fcn(ct.rss(nstate, nout, ninp, strictly_proper=True))

        # Generate the time and input vectors
        tvec = np.linspace(0, 1, 8)
        uvec = np.ones((sys.ninputs, 1)) @ np.reshape(np.sin(tvec), (1, 8))

        #
        # Pass squeeze argument and make sure the shape is correct
        #
        # For responses that are indexed by the input, check against shape1
        # For responses that have no/fixed input, check against shape2
        #

        # Impulse response
        if isinstance(sys, StateSpace):
            # Check the states as well
            _, yvec, xvec = ct.impulse_response(
                sys, tvec, squeeze=squeeze, return_x=True)
            if sys.issiso():
                assert xvec.shape == (sys.nstates, 8)
            else:
                assert xvec.shape == (sys.nstates, sys.ninputs, 8)
        else:
            _, yvec = ct.impulse_response(sys, tvec, squeeze=squeeze)
        assert yvec.shape == shape1

        # Step response
        if isinstance(sys, StateSpace):
            # Check the states as well
            _, yvec, xvec = ct.step_response(
                sys, tvec, squeeze=squeeze, return_x=True)
            if sys.issiso():
                assert xvec.shape == (sys.nstates, 8)
            else:
                assert xvec.shape == (sys.nstates, sys.ninputs, 8)
        else:
            _, yvec = ct.step_response(sys, tvec, squeeze=squeeze)
        assert yvec.shape == shape1

        # Initial response (only indexed by output)
        if isinstance(sys, StateSpace):
            # Check the states as well
            _, yvec, xvec = ct.initial_response(
                sys, tvec, 1, squeeze=squeeze, return_x=True)
            assert xvec.shape == (sys.nstates, 8)
        elif isinstance(sys, TransferFunction):
            with pytest.warns(UserWarning, match="may not be consistent"):
                _, yvec = ct.initial_response(sys, tvec, 1, squeeze=squeeze)
        assert yvec.shape == shape2

        # Forced response (only indexed by output)
        if isinstance(sys, StateSpace):
            # Check the states as well
            _, yvec, xvec = ct.forced_response(
                sys, tvec, uvec, 0, return_x=True, squeeze=squeeze)
            assert xvec.shape == (sys.nstates, 8)
        else:
            # Just check the input/output response
            _, yvec = ct.forced_response(sys, tvec, uvec, 0, squeeze=squeeze)
        assert yvec.shape == shape2

        # Test cases where we choose a subset of inputs and outputs
        _, yvec = ct.step_response(
            sys, tvec, input=ninp-1, output=nout-1, squeeze=squeeze)
        if squeeze is False:
            # Shape should be unsqueezed
            assert yvec.shape == (1, 1, 8)
        else:
            # Shape should be squeezed
            assert yvec.shape == (8, )

        # For NonlinearIOSystem, also test input/output response
        if isinstance(sys, ct.NonlinearIOSystem):
            _, yvec = ct.input_output_response(sys, tvec, uvec, squeeze=squeeze)
            assert yvec.shape == shape2

        #
        # Changing config.default to False should return 3D frequency response
        #
        ct.config.set_defaults('control', squeeze_time_response=False)

        _, yvec = ct.impulse_response(sys, tvec)
        if squeeze is not True or sys.ninputs > 1 or sys.noutputs > 1:
            assert yvec.shape == (sys.noutputs, sys.ninputs, 8)

        _, yvec = ct.step_response(sys, tvec)
        if squeeze is not True or sys.ninputs > 1 or sys.noutputs > 1:
            assert yvec.shape == (sys.noutputs, sys.ninputs, 8)

        if isinstance(sys, TransferFunction):
            with pytest.warns(UserWarning, match="may not be consistent"):
                _, yvec = ct.initial_response(sys, tvec, 1)
        else:
            _, yvec = ct.initial_response(sys, tvec, 1)
        if squeeze is not True or sys.noutputs > 1:
            assert yvec.shape == (sys.noutputs, 8)

        if isinstance(sys, ct.StateSpace):
            _, yvec, xvec = ct.forced_response(
                sys, tvec, uvec, 0, return_x=True)
            assert xvec.shape == (sys.nstates, 8)
        else:
            _, yvec = ct.forced_response(sys, tvec, uvec, 0)
        if squeeze is not True or sys.noutputs > 1:
            assert yvec.shape == (sys.noutputs, 8)

        # For NonlinearIOSystems, also test input_output_response
        if isinstance(sys, ct.NonlinearIOSystem):
            _, yvec = ct.input_output_response(sys, tvec, uvec)
            if squeeze is not True or sys.noutputs > 1:
                assert yvec.shape == (sys.noutputs, 8)

    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf])
    def test_squeeze_exception(self, fcn):
        sys = fcn(ct.rss(2, 1, 1))
        with pytest.raises(ValueError, match="Unknown squeeze value"):
            step_response(sys, squeeze=1)

    @pytest.mark.usefixtures("editsdefaults")
    @pytest.mark.parametrize("nstate, nout, ninp, squeeze, shape", [
        [1, 1, 1, None, (8,)],
        [2, 1, 1, True, (8,)],
        [3, 1, 1, False, (1, 8)],
        [1, 2, 1, None, (2, 8)],
        [2, 2, 1, True, (2, 8)],
        [3, 2, 1, False, (2, 8)],
        [1, 1, 2, None, (8,)],
        [2, 1, 2, True, (8,)],
        [3, 1, 2, False, (1, 8)],
        [1, 2, 2, None, (2, 8)],
        [2, 2, 2, True, (2, 8)],
        [3, 2, 2, False, (2, 8)],
    ])
    def test_squeeze_0_8_4(self, nstate, nout, ninp, squeeze, shape):
        # Set defaults to match release 0.8.4
        with pytest.warns(UserWarning, match="NumPy matrix class no longer"):
            ct.config.use_legacy_defaults('0.8.4')

        # Generate system, time, and input vectors
        sys = ct.rss(nstate, nout, ninp, strictly_proper=True)
        tvec = np.linspace(0, 1, 8)

        _, yvec = ct.initial_response(sys, tvec, 1, squeeze=squeeze)
        assert yvec.shape == shape

    @pytest.mark.parametrize(
        "nstate, nout, ninp, squeeze, ysh_in, ysh_no, xsh_in", [
        [4,    1,    1,  None,   (8,),       (8,),    (8, 4)],
        [4,    1,    1,  True,   (8,),       (8,),    (8, 4)],
        [4,    1,    1,  False,  (8, 1, 1),  (8, 1),  (8, 4)],
        [4,    2,    1,  None,   (8, 2, 1),  (8, 2),  (8, 4, 1)],
        [4,    2,    1,  True,   (8, 2),     (8, 2),  (8, 4, 1)],
        [4,    2,    1,  False,  (8, 2, 1),  (8, 2),  (8, 4, 1)],
        [4,    1,    2,  None,   (8, 1, 2),  (8, 1),  (8, 4, 2)],
        [4,    1,    2,  True,   (8, 2),     (8,),    (8, 4, 2)],
        [4,    1,    2,  False,  (8, 1, 2),  (8, 1),  (8, 4, 2)],
        [4,    2,    2,  None,   (8, 2, 2),  (8, 2),  (8, 4, 2)],
        [4,    2,    2,  True,   (8, 2, 2),  (8, 2),  (8, 4, 2)],
        [4,    2,    2,  False,  (8, 2, 2),  (8, 2),  (8, 4, 2)],
    ])
    def test_response_transpose(
            self, nstate, nout, ninp, squeeze, ysh_in, ysh_no, xsh_in):
        sys = ct.rss(nstate, nout, ninp)
        T = np.linspace(0, 1, 8)

        # Step response - input indexed
        t, y, x = ct.step_response(
            sys, T, transpose=True, return_x=True, squeeze=squeeze)
        assert t.shape == (T.size, )
        assert y.shape == ysh_in
        assert x.shape == xsh_in

        # Initial response - no input indexing
        t, y, x = ct.initial_response(
            sys, T, 1, transpose=True, return_x=True, squeeze=squeeze)
        assert t.shape == (T.size, )
        assert y.shape == ysh_no
        assert x.shape == (T.size, sys.nstates)


@pytest.mark.skipif(not pandas_check(), reason="pandas not installed")
def test_to_pandas():
    # Create a SISO time response
    sys = ct.rss(2, 1, 1)
    timepts = np.linspace(0, 10, 10)
    resp = ct.input_output_response(sys, timepts, 1)

    # Convert to pandas
    df = resp.to_pandas()

    # Check to make sure the data make senses
    np.testing.assert_equal(df['time'], resp.time)
    np.testing.assert_equal(df['u[0]'], resp.inputs)
    np.testing.assert_equal(df['y[0]'], resp.outputs)
    np.testing.assert_equal(df['x[0]'], resp.states[0])
    np.testing.assert_equal(df['x[1]'], resp.states[1])

    # Create a MIMO time response
    sys = ct.rss(2, 2, 1)
    resp = ct.input_output_response(sys, timepts, np.sin(timepts))
    df = resp.to_pandas()
    np.testing.assert_equal(df['time'], resp.time)
    np.testing.assert_equal(df['u[0]'], resp.inputs[0])
    np.testing.assert_equal(df['y[0]'], resp.outputs[0])
    np.testing.assert_equal(df['y[1]'], resp.outputs[1])
    np.testing.assert_equal(df['x[0]'], resp.states[0])
    np.testing.assert_equal(df['x[1]'], resp.states[1])

    # Change the time points
    sys = ct.rss(2, 1, 2)
    T = np.linspace(0, timepts[-1]/2, timepts.size * 2)
    resp = ct.input_output_response(
        sys, timepts, [np.sin(timepts), 0], t_eval=T)
    df = resp.to_pandas()
    np.testing.assert_equal(df['time'], resp.time)
    np.testing.assert_equal(df['u[0]'], resp.inputs[0])
    np.testing.assert_equal(df['y[0]'], resp.outputs[0])
    np.testing.assert_equal(df['x[0]'], resp.states[0])
    np.testing.assert_equal(df['x[1]'], resp.states[1])

    # System with no states
    sys = ct.ss([], [], [], 5)
    resp = ct.input_output_response(sys, timepts, np.sin(timepts), t_eval=T)
    df = resp.to_pandas()
    np.testing.assert_equal(df['time'], resp.time)
    np.testing.assert_equal(df['u[0]'], resp.inputs)
    np.testing.assert_equal(df['y[0]'], resp.inputs * 5)

    # Multi-trace data
    # https://github.com/python-control/python-control/issues/1087
    model = ct.rss(
        states=['x0', 'x1'], outputs=['y0', 'y1'],
        inputs=['u0', 'u1'], name='My Model')
    T = np.linspace(0, 10, 100, endpoint=False)
    X0 = np.zeros(model.nstates)

    res = ct.step_response(model, T=T, X0=X0, input=0)  # extract single trace
    df = res.to_pandas()
    np.testing.assert_equal(
        df[df['trace'] == 'From u0']['time'], res.time)
    np.testing.assert_equal(
        df[df['trace'] == 'From u0']['u0'], res.inputs['u0', 0])
    np.testing.assert_equal(
        df[df['trace'] == 'From u0']['y1'], res.outputs['y1', 0])

    res = ct.step_response(model, T=T, X0=X0)           # all traces
    df = res.to_pandas()
    for i, label in enumerate(res.trace_labels):
        np.testing.assert_equal(
            df[df['trace'] == label]['time'], res.time)
        np.testing.assert_equal(
            df[df['trace'] == label]['u1'], res.inputs['u1', i])
        np.testing.assert_equal(
            df[df['trace'] == label]['y0'], res.outputs['y0', i])


@pytest.mark.skipif(pandas_check(), reason="pandas installed")
def test_no_pandas():
    # Create a SISO time response
    sys = ct.rss(2, 1, 1)
    timepts = np.linspace(0, 10, 10)
    resp = ct.input_output_response(sys, timepts, 1)

    # Convert to pandas
    with pytest.raises(ImportError, match="pandas"):
        resp.to_pandas()


# https://github.com/python-control/python-control/issues/1014
def test_step_info_nonstep():
    # Pass a constant input
    timepts = np.linspace(0, 10, endpoint=False)
    y_const = np.ones_like(timepts)

    # Constant value of 1
    step_info = ct.step_info(y_const, timepts)
    assert step_info['RiseTime'] == 0
    assert step_info['SettlingTime'] == 0
    assert step_info['SettlingMin'] == 1
    assert step_info['SettlingMax'] == 1
    assert step_info['Overshoot'] == 0
    assert step_info['Undershoot'] == 0
    assert step_info['Peak'] == 1
    assert step_info['PeakTime'] == 0
    assert step_info['SteadyStateValue'] == 1

    # Constant value of -1
    step_info = ct.step_info(-y_const, timepts)
    assert step_info['RiseTime'] == 0
    assert step_info['SettlingTime'] == 0
    assert step_info['SettlingMin'] == -1
    assert step_info['SettlingMax'] == -1
    assert step_info['Overshoot'] == 0
    assert step_info['Undershoot'] == 0
    assert step_info['Peak'] == 1
    assert step_info['PeakTime'] == 0
    assert step_info['SteadyStateValue'] == -1

    # Ramp from -1 to 1
    step_info = ct.step_info(-1 + 2 * timepts/10, timepts)
    assert step_info['RiseTime'] == 3.8
    assert step_info['SettlingTime'] == 9.8
    assert isclose(step_info['SettlingMin'], 0.88)
    assert isclose(step_info['SettlingMax'], 0.96)
    assert step_info['Overshoot'] == 0
    assert step_info['Peak'] == 1
    assert step_info['PeakTime'] == 0
    assert isclose(step_info['SteadyStateValue'], 0.96)


def test_signal_labels():
    # Create a system response for a SISO system
    sys = ct.rss(4, 1, 1)
    response = ct.step_response(sys)

    # Make sure access via strings works
    np.testing.assert_equal(response.states['x[2]'], response.states[2])

    # Make sure access via lists of strings works
    np.testing.assert_equal(
        response.states[['x[1]', 'x[2]']], response.states[[1, 2]])

    # Make sure errors are generated if key is unknown
    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        response.inputs['bad']

    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        response.states[['x[1]', 'bad']]

    # Create a system response for a MIMO system
    sys = ct.rss(4, 2, 2)
    response = ct.step_response(sys)

    # Make sure access via strings works
    np.testing.assert_equal(
        response.outputs['y[0]', 'u[1]'],
        response.outputs[0, 1])
    np.testing.assert_equal(
        response.states['x[2]', 'u[0]'], response.states[2, 0])

    # Make sure access via lists of strings works
    np.testing.assert_equal(
        response.states[['x[1]', 'x[2]'], 'u[0]'],
        response.states[[1, 2], 0])

    np.testing.assert_equal(
        response.outputs[['y[1]'], ['u[1]', 'u[0]']],
        response.outputs[[1], [1, 0]])

    # Make sure errors are generated if key is unknown
    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        response.inputs['bad']

    with pytest.raises(ValueError, match="unknown signal name 'bad'"):
        response.states[['x[1]', 'bad']]

    with pytest.raises(ValueError, match=r"unknown signal name 'x\[2\]'"):
        response.states['x[1]', 'x[2]']         # second index = input name


def test_timeresp_aliases():
    sys = ct.rss(2, 1, 1)
    timepts = np.linspace(0, 10, 10)
    resp_long = ct.input_output_response(sys, timepts, 1, initial_state=[1, 1])

    # Positional usage
    resp_posn = ct.input_output_response(sys, timepts, 1, [1, 1])
    np.testing.assert_allclose(resp_long.states, resp_posn.states)

    # Aliases
    resp_short = ct.input_output_response(sys, timepts, 1, X0=[1, 1])
    np.testing.assert_allclose(resp_long.states, resp_short.states)

    # Legacy
    with pytest.warns(PendingDeprecationWarning, match="legacy"):
        resp_legacy = ct.input_output_response(sys, timepts, 1, x0=[1, 1])
    np.testing.assert_allclose(resp_long.states, resp_legacy.states)

    # Check for multiple values: full keyword and alias
    with pytest.raises(TypeError, match="multiple"):
        ct.input_output_response(
            sys, timepts, 1, initial_state=[1, 2], X0=[1, 1])

    # Check for multiple values: positional and keyword
    with pytest.raises(TypeError, match="multiple"):
        ct.input_output_response(
            sys, timepts, 1, [1, 2], initial_state=[1, 1])

    # Check for multiple values: positional and alias
    with pytest.raises(TypeError, match="multiple"):
        ct.input_output_response(
            sys, timepts, 1, [1, 2], X0=[1, 1])

    # Make sure that LTI functions check for keywords
    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.forced_response(sys, timepts, 1, unknown=True)

    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.impulse_response(sys, timepts, unknown=True)

    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.initial_response(sys, timepts, [1, 2], unknown=True)

    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.step_response(sys, timepts, unknown=True)

    with pytest.raises(TypeError, match="unrecognized keyword"):
        ct.step_info(sys, timepts, unknown=True)
