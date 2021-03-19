"""timeresp_test.py - test time response functions

RMM, 17 Jun 2011 (based on TestMatlab from v0.4c)

This test suite just goes through and calls all of the MATLAB
functions using different systems and arguments to make sure that
nothing crashes.  It doesn't test actual functionality; the module
specific unit tests will do that.
"""

from copy import copy
from distutils.version import StrictVersion

import numpy as np
import pytest
import scipy as sp

import control as ct
from control import StateSpace, TransferFunction, c2d, isctime, ss2tf, tf2ss
from control.exception import slycot_check
from control.tests.conftest import slycotonly
from control.timeresp import (_default_time_vector, _ideal_tfinal_and_dt,
                              forced_response, impulse_response,
                              initial_response, step_info, step_response)


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
    def siso_ss1(self):

        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        T = TSys(StateSpace(A, B, C, D, 0))

        T.t = np.linspace(0, 1, 10)
        T.ystep = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234,
                            39.1165, 42.3227, 44.9694, 47.1599, 48.9776])

        T.yinitial = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118,
                               1.9092, 1.1508, 0.5833, 0.1645, -0.1391])

        return T

    @pytest.fixture
    def siso_ss2(self, siso_ss1):
        """System siso_ss2 with D=0"""
        ss1 = siso_ss1.sys
        T = TSys(StateSpace(ss1.A, ss1.B, ss1.C, 0, 0))
        T.t = siso_ss1.t
        T.ystep = siso_ss1.ystep - 9
        T.initial = siso_ss1.yinitial - 9
        T.yimpulse = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766,
                               31.7344, 26.1668, 21.6292, 17.9245, 14.8945])

        return T

    @pytest.fixture
    def siso_tf1(self):
        # Create some transfer functions
        return TSys(TransferFunction([1], [1, 2, 1], 0))

    @pytest.fixture
    def siso_tf2(self, siso_ss1):
        T = copy(siso_ss1)
        T.sys = ss2tf(siso_ss1.sys)
        return T

    @pytest.fixture
    def mimo_ss1(self, siso_ss1):
        # Create MIMO system, contains ``siso_ss1`` twice
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
        T = copy(siso_ss1)
        T.sys = StateSpace(A, B, C, D)
        return T

    @pytest.fixture
    def mimo_ss2(self, siso_ss2):
        # Create MIMO system, contains ``siso_ss2`` twice
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
        T = copy(siso_ss2)
        T.sys = StateSpace(A, B, C, D, 0)
        return T

    # Create discrete time systems

    @pytest.fixture
    def siso_dtf0(self):
        T = TSys(TransferFunction([1.], [1., 0.], 1.))
        T.t = np.arange(4)
        T.yimpulse = [0., 1., 0., 0.]
        return T

    @pytest.fixture
    def siso_dtf1(self):
        T =  TSys(TransferFunction([1], [1, 1, 0.25], True))
        T.t = np.arange(0, 5, 1)
        return T

    @pytest.fixture
    def siso_dtf2(self):
        T = TSys(TransferFunction([1], [1, 1, 0.25], 0.2))
        T.t = np.arange(0, 5, 0.2)
        return T

    @pytest.fixture
    def siso_dss1(self, siso_dtf1):
        T = copy(siso_dtf1)
        T.sys = tf2ss(siso_dtf1.sys)
        return T

    @pytest.fixture
    def siso_dss2(self, siso_dtf2):
        T = copy(siso_dtf2)
        T.sys = tf2ss(siso_dtf2.sys)
        return T

    @pytest.fixture
    def mimo_dss1(self, mimo_ss1):
        ss1 = mimo_ss1.sys
        T = TSys(
            StateSpace(ss1.A, ss1.B, ss1.C, ss1.D, True))
        T.t = np.arange(0, 5, 0.2)
        return T

    @pytest.fixture
    def mimo_dss2(self, mimo_ss1):
        T = copy(mimo_ss1)
        T.sys = c2d(mimo_ss1.sys, T.t[1]-T.t[0])
        return T

    @pytest.fixture
    def mimo_tf2(self, siso_ss2, mimo_ss2):
        T = copy(mimo_ss2)
        # construct from siso to avoid slycot during fixture setup
        tf_ = ss2tf(siso_ss2.sys)
        T.sys = TransferFunction([[tf_.num[0][0], [0]], [[0], tf_.num[0][0]]],
                                 [[tf_.den[0][0], [1]], [[1], tf_.den[0][0]]],
                                 0)
        return T

    @pytest.fixture
    def mimo_dtf1(self, siso_dtf1):
        T = copy(siso_dtf1)
        # construct from siso to avoid slycot during fixture setup
        tf_ = siso_dtf1.sys
        T.sys = TransferFunction([[tf_.num[0][0], [0]], [[0], tf_.num[0][0]]],
                                 [[tf_.den[0][0], [1]], [[1], tf_.den[0][0]]],
                                 True)
        return T

    @pytest.fixture
    def pole_cancellation(self):
        # for pole cancellation tests
        return TransferFunction([1.067e+05, 5.791e+04],
                                [10.67, 1.067e+05, 5.791e+04])

    @pytest.fixture
    def no_pole_cancellation(self):
        return TransferFunction([1.881e+06],
                                [188.1, 1.881e+06])

    @pytest.fixture
    def siso_tf_type1(self):
        # System Type 1 - Step response not stationary:  G(s)=1/s(s+1)
        T = TSys(TransferFunction(1, [1, 1, 0]))
        T.step_info = {
             'RiseTime': np.NaN,
             'SettlingTime': np.NaN,
             'SettlingMin': np.NaN,
             'SettlingMax': np.NaN,
             'Overshoot': np.NaN,
             'Undershoot': np.NaN,
             'Peak': np.Inf,
             'PeakTime': np.Inf,
             'SteadyStateValue': np.NaN}
        return T

    @pytest.fixture
    def siso_tf_kpos(self):
        # SISO under shoot response and positive final value
        # G(s)=(-s+1)/(s²+s+1)
        T = TSys(TransferFunction([-1, 1], [1, 1, 1]))
        T.step_info = {
             'RiseTime': 1.242,
             'SettlingTime': 9.110,
             'SettlingMin': 0.90,
             'SettlingMax': 1.208,
             'Overshoot': 20.840,
             'Undershoot': 28.0,
             'Peak': 1.208,
             'PeakTime': 4.282,
             'SteadyStateValue': 1.0}
        return T

    @pytest.fixture
    def siso_tf_kneg(self):
        # SISO under shoot response and negative final value
        # k=-1 G(s)=-(-s+1)/(s²+s+1)
        T = TSys(TransferFunction([1, -1], [1, 1, 1]))
        T.step_info = {
             'RiseTime': 1.242,
             'SettlingTime': 9.110,
             'SettlingMin': -1.208,
             'SettlingMax': -0.90,
             'Overshoot': 20.840,
             'Undershoot': 28.0,
             'Peak': 1.208,
             'PeakTime': 4.282,
             'SteadyStateValue': -1.0}
        return T

    @pytest.fixture
    def siso_tf_step_matlab(self):
        # example from matlab online help
        # https://www.mathworks.com/help/control/ref/stepinfo.html
        T = TSys(TransferFunction([1, 5, 5], [1, 1.65, 5, 6.5, 2]))
        T.step_info = {
            'RiseTime': 3.8456,
            'SettlingTime': 27.9762,
            'SettlingMin': 2.0689,
            'SettlingMax': 2.6873,
            'Overshoot': 7.4915,
            'Undershoot': 0,
            'Peak': 2.6873,
            'PeakTime': 8.0530,
            'SteadyStateValue': 2.5}
        return T

    @pytest.fixture
    def mimo_ss_step_matlab(self):
        A = [[0.68, -0.34],
             [0.34, 0.68]]
        B = [[0.18, -0.05],
             [0.04, 0.11]]
        C = [[0, -1.53],
             [-1.12, -1.10]]
        D = [[0, 0],
             [0.06, -0.37]]
        T = TSys(StateSpace(A, B, C, D, 0.2))
        T.kwargs['step_info'] = {'T': 4.6}
        T.step_info = [[{'RiseTime': 0.6000,
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
                         'SettlingMin': -0.1034,
                         'SettlingMax': -0.1485,
                         'Overshoot': 132.0170,
                         'Undershoot': 79.222, # 0. in MATLAB
                         'Peak': 0.4350,
                         'PeakTime': .2,
                         'SteadyStateValue': -0.1875}]]
                         # (*): MATLAB gives 0.4 here, but it is unclear what
                         # 10% and 90% of the steady state response mean, when
                         # the step for this channel does not start a 0 for
                         # 0 initial conditions
        return T

    @pytest.fixture
    def siso_ss_step_matlab(self, mimo_ss_step_matlab):
        T = copy(mimo_ss_step_matlab)
        T.sys = T.sys[1, 0]
        T.step_info = T.step_info[1][0]
        return T

    @pytest.fixture
    def mimo_tf_step_info(self,
                          siso_tf_kpos, siso_tf_kneg,
                          siso_tf_step_matlab):
        Ta = [[siso_tf_kpos, siso_tf_kneg, siso_tf_step_matlab],
               [siso_tf_step_matlab, siso_tf_kpos, siso_tf_kneg]]
        T = TSys(TransferFunction(
            [[Ti.sys.num[0][0] for Ti in Tr] for Tr in Ta],
            [[Ti.sys.den[0][0] for Ti in Tr] for Tr in Ta]))
        T.step_info = [[Ti.step_info for Ti in Tr] for Tr in Ta]
        # enforce enough sample points for all channels (they have different
        # characteristics)
        T.kwargs['step_info'] = {'T_num': 2000}
        return T


    @pytest.fixture
    def tsystem(self,
                request,
                siso_ss1, siso_ss2, siso_tf1, siso_tf2,
                mimo_ss1, mimo_ss2, mimo_tf2,
                siso_dtf0, siso_dtf1, siso_dtf2,
                siso_dss1, siso_dss2,
                mimo_dss1, mimo_dss2, mimo_dtf1,
                pole_cancellation, no_pole_cancellation, siso_tf_type1,
                siso_tf_kpos, siso_tf_kneg,
                siso_tf_step_matlab, siso_ss_step_matlab,
                mimo_ss_step_matlab, mimo_tf_step_info):
        systems = {"siso_ss1": siso_ss1,
                   "siso_ss2": siso_ss2,
                   "siso_tf1": siso_tf1,
                   "siso_tf2": siso_tf2,
                   "mimo_ss1": mimo_ss1,
                   "mimo_ss2": mimo_ss2,
                   "mimo_tf2": mimo_tf2,
                   "siso_dtf0": siso_dtf0,
                   "siso_dtf1": siso_dtf1,
                   "siso_dtf2": siso_dtf2,
                   "siso_dss1": siso_dss1,
                   "siso_dss2": siso_dss2,
                   "mimo_dss1": mimo_dss1,
                   "mimo_dss2": mimo_dss2,
                   "mimo_dtf1": mimo_dtf1,
                   "pole_cancellation": pole_cancellation,
                   "no_pole_cancellation": no_pole_cancellation,
                   "siso_tf_type1": siso_tf_type1,
                   "siso_tf_kpos": siso_tf_kpos,
                   "siso_tf_kneg": siso_tf_kneg,
                   "siso_tf_step_matlab": siso_tf_step_matlab,
                   "siso_ss_step_matlab": siso_ss_step_matlab,
                   "mimo_ss_step_matlab": mimo_ss_step_matlab,
                   "mimo_tf_step": mimo_tf_step_info,
                   }
        return systems[request.param]

    @pytest.mark.parametrize(
        "kwargs",
        [{},
         {'X0': 0},
         {'X0': np.array([0, 0])},
         {'X0': 0, 'return_x': True},
         ])
    def test_step_response_siso(self, siso_ss1, kwargs):
        """Test SISO system step response"""
        sys = siso_ss1.sys
        t = siso_ss1.t
        yref = siso_ss1.ystep
        # SISO call
        out = step_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    def test_step_response_mimo(self, mimo_ss1):
        """Test MIMO system, which contains ``siso_ss1`` twice"""
        sys = mimo_ss1.sys
        t = mimo_ss1.t
        yref = mimo_ss1.ystep
        _t, y_00 = step_response(sys, T=t, input=0, output=0)
        _t, y_11 = step_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)
        np.testing.assert_array_almost_equal(y_11, yref, decimal=4)

    def test_step_response_return(self, mimo_ss1):
        """Verify continuous and discrete time use same return conventions"""
        sysc = mimo_ss1.sys
        sysd = c2d(sysc, 1)            # discrete time system
        Tvec = np.linspace(0, 10, 11)  # make sure to use integer times 0..10
        Tc, youtc = step_response(sysc, Tvec, input=0)
        Td, youtd = step_response(sysd, Tvec, input=0)
        np.testing.assert_array_equal(Tc.shape, Td.shape)
        np.testing.assert_array_equal(youtc.shape, youtd.shape)


    @pytest.mark.parametrize("dt", [0, 1], ids=["continuous", "discrete"])
    def test_step_nostates(self, dt):
        """Constant system, continuous and discrete time

        gh-374 "Bug in step_response()"
        """
        sys = TransferFunction([1], [1], dt)
        t, y = step_response(sys)
        np.testing.assert_array_equal(y, np.ones(len(t)))

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
         "siso_tf_type1"],
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
         pytest.param('mimo_tf_step', marks=slycotonly)],
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

    def test_step_pole_cancellation(self, pole_cancellation,
                                    no_pole_cancellation):
        # confirm that pole-zero cancellation doesn't perturb results
        # https://github.com/python-control/python-control/issues/440
        step_info_no_cancellation = step_info(no_pole_cancellation)
        step_info_cancellation = step_info(pole_cancellation)
        self.assert_step_info_match(no_pole_cancellation,
                                    step_info_no_cancellation,
                                    step_info_cancellation)

    @pytest.mark.parametrize(
        "tsystem, kwargs",
        [("siso_ss2", {}),
         ("siso_ss2", {'X0': 0}),
         ("siso_ss2", {'X0': np.array([0, 0])}),
         ("siso_ss2", {'X0': 0, 'return_x': True}),
         ("siso_dtf0", {})],
        indirect=["tsystem"])
    def test_impulse_response_siso(self, tsystem, kwargs):
        """Test impulse response of SISO systems"""
        sys = tsystem.sys
        t = tsystem.t
        yref = tsystem.yimpulse

        out = impulse_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    def test_impulse_response_mimo(self, mimo_ss2):
        """"Test impulse response of MIMO systems"""
        sys = mimo_ss2.sys
        t = mimo_ss2.t

        yref = mimo_ss2.yimpulse
        _t, y_00 = impulse_response(sys, T=t, input=0, output=0)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)
        _t, y_11 = impulse_response(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_11, yref, decimal=4)

        yref_notrim = np.zeros((2, len(t)))
        yref_notrim[:1, :] = yref
        _t, yy = impulse_response(sys, T=t, input=0)
        np.testing.assert_array_almost_equal(yy[:,0,:], yref_notrim, decimal=4)

    @pytest.mark.skipif(StrictVersion(sp.__version__) < "1.3",
                        reason="requires SciPy 1.3 or greater")
    def test_discrete_time_impulse(self, siso_tf1):
        # discrete time impulse sampled version should match cont time
        dt = 0.1
        t = np.arange(0, 3, dt)
        sys = siso_tf1.sys
        sysdt = sys.sample(dt, 'impulse')
        np.testing.assert_array_almost_equal(impulse_response(sys, t)[1],
                                             impulse_response(sysdt, t)[1])

    def test_impulse_response_warnD(self, siso_ss1):
        """Test warning about direct feedthrough"""
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            _ = impulse_response(siso_ss1.sys, siso_ss1.t)

    @pytest.mark.parametrize(
        "kwargs",
        [{},
         {'X0': 0},
         {'X0': np.array([0.5, 1])},
         {'X0': np.array([[0.5], [1]])},
         {'X0': np.array([0.5, 1]), 'return_x': True},
         ])
    def test_initial_response(self, siso_ss1, kwargs):
        """Test initial response of SISO system"""
        sys = siso_ss1.sys
        t = siso_ss1.t
        x0 = kwargs.get('X0', 0)
        yref = siso_ss1.yinitial if np.any(x0) else np.zeros_like(t)

        out = initial_response(sys, T=t, **kwargs)
        tout, yout = out[:2]
        assert len(out) == 3 if ('return_x', True) in kwargs.items() else 2
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

    def test_initial_response_mimo(self, mimo_ss1):
        """Test initial response of MIMO system"""
        sys = mimo_ss1.sys
        t = mimo_ss1.t
        x0 = np.array([[.5], [1.], [.5], [1.]])
        yref = mimo_ss1.yinitial
        yref_notrim = np.broadcast_to(yref, (2, len(t)))

        _t, y_00 = initial_response(sys, T=t, X0=x0, input=0, output=0)
        np.testing.assert_array_almost_equal(y_00, yref, decimal=4)
        _t, y_11 = initial_response(sys, T=t, X0=x0, input=0, output=1)
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
    def test_forced_response_initial(self, siso_ss1, u):
        """Test forced response of SISO system as intitial response"""
        sys = siso_ss1.sys
        t = siso_ss1.t
        x0 = np.array([[.5], [1.]])
        yref = siso_ss1.yinitial

        tout, yout = forced_response(sys, t, u, X0=x0)
        np.testing.assert_array_almost_equal(tout, t)
        np.testing.assert_array_almost_equal(yout, yref, decimal=4)

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
        ct.config.use_legacy_defaults("0.8.4")
        # forced_response returns x by default
        t, y = ct.step_response(sys, T)
        t, y, x = ct.forced_response(sys, T, U)

        ct.config.use_legacy_defaults("0.9.0")
        # forced_response returns input/output by default
        t, y = ct.step_response(sys, T)
        t, y = ct.forced_response(sys, T, U)
        t, y, x = ct.forced_response(sys, T, U, return_x=True)


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
        ytrue = np.squeeze(np.asarray(C.dot(xtrue)))
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
                              "siso_dss1",  # no timebase
                              "siso_dtf1",
                              "siso_dss2",  # matching timebase
                              "siso_dtf2",
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
    def test_time_vector(self, tsystem, fun, squeeze, matarrayout):
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
        elif fun == forced_response and isctime(sys):
            pytest.skip("No continuous forced_response without time vector.")
        if hasattr(tsystem.sys, "nstates"):
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

        if squeeze is False or not sys.issiso():
            assert yout.shape[0] == sys.noutputs
            assert yout.shape[-1] == tout.shape[0]
        else:
            assert yout.shape == tout.shape

        if sys.dt > 0 and sys.dt is not True and not np.isclose(sys.dt, 0.5):
            kw['T'] = np.arange(0, 5, 0.5)  # incompatible timebase
            with pytest.raises(ValueError):
                fun(sys, **kw)

    @pytest.mark.parametrize("squeeze", [None, True, False])
    def test_time_vector_interpolation(self, siso_dtf2, squeeze):
        """Test time vector handling in case of interpolation

        Interpolation of the input (to match scipy.signal.dlsim)

        gh-239, gh-295
        """
        sys = siso_dtf2.sys
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

    def test_discrete_time_steps(self, siso_dtf2):
        """Make sure rounding errors in sample time are handled properly

        These tests play around with the input time vector to make sure that
        small rounding errors don't generate spurious errors.

        gh-332
        """
        sys = siso_dtf2.sys

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

    def test_time_series_data_convention_2D(self, siso_ss1):
        """Allow input time as 2D array (output should be 1D)"""
        tin = np.array(np.linspace(0, 10, 100), ndmin=2)
        t, y = step_response(siso_ss1.sys, tin)
        assert isinstance(t, np.ndarray) and not isinstance(t, np.matrix)
        assert t.ndim == 1
        assert y.ndim == 1  # SISO returns "scalar" output
        assert t.shape == y.shape  # Allows direct plotting of output

    @pytest.mark.usefixtures("editsdefaults")
    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.ss2io])
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
        # Figure out if we have SciPy 1+
        scipy0 = StrictVersion(sp.__version__) < '1.0'

        # Define the system
        if fcn == ct.tf and (nout > 1 or ninp > 1) and not slycot_check():
            pytest.skip("Conversion of MIMO systems to transfer functions "
                        "requires slycot.")
        else:
            sys = fcn(ct.rss(nstate, nout, ninp, strictly_proper=True))

        # Generate the time and input vectors
        tvec = np.linspace(0, 1, 8)
        uvec = np.dot(
            np.ones((sys.ninputs, 1)),
            np.reshape(np.sin(tvec), (1, 8)))

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
        else:
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

        # For InputOutputSystems, also test input/output response
        if isinstance(sys, ct.InputOutputSystem) and not scipy0:
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

        # For InputOutputSystems, also test input_output_response
        if isinstance(sys, ct.InputOutputSystem) and not scipy0:
            _, yvec = ct.input_output_response(sys, tvec, uvec)
            if squeeze is not True or sys.noutputs > 1:
                assert yvec.shape == (sys.noutputs, 8)

    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.ss2io])
    def test_squeeze_exception(self, fcn):
        sys = fcn(ct.rss(2, 1, 1))
        with pytest.raises(ValueError, match="unknown squeeze value"):
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
        ct.config.use_legacy_defaults('0.8.4')
        ct.config.use_numpy_matrix(False)

        # Generate system, time, and input vectors
        sys = ct.rss(nstate, nout, ninp, strictly_proper=True)
        tvec = np.linspace(0, 1, 8)
        uvec = np.dot(
            np.ones((sys.ninputs, 1)),
            np.reshape(np.sin(tvec), (1, 8)))

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
