"""lti_test.py"""

import numpy as np
import pytest
from .conftest import editsdefaults

import control as ct
from control import c2d, tf, ss, tf2ss, NonlinearIOSystem
from control.lti import LTI, evalfr, damp, dcgain, zeros, poles, bandwidth
from control import common_timebase, isctime, isdtime, issiso, timebaseEqual
from control.tests.conftest import slycotonly
from control.exception import slycot_check

class TestLTI:
    @pytest.mark.parametrize("fun, args", [
        [tf, (126, [-1, 42])],
        [ss, ([[42]], [[1]], [[1]], 0)]
    ])
    def test_poles(self, fun, args):
        sys = fun(*args)
        np.testing.assert_allclose(sys.poles(), 42)
        np.testing.assert_allclose(poles(sys), 42)

        with pytest.warns(PendingDeprecationWarning):
            pole_list = sys.pole()
            assert pole_list == sys.poles()

        with pytest.warns(PendingDeprecationWarning):
            pole_list = ct.pole(sys)
            assert pole_list == sys.poles()

    @pytest.mark.parametrize("fun, args", [
        [tf, (126, [-1, 42])],
        [ss, ([[42]], [[1]], [[1]], 0)]
    ])
    def test_zero(self, fun, args):
        sys = fun(*args)
        np.testing.assert_allclose(sys.zeros(), 42)
        np.testing.assert_allclose(zeros(sys), 42)

        with pytest.warns(PendingDeprecationWarning):
            sys.zero()

        with pytest.warns(PendingDeprecationWarning):
            ct.zero(sys)

    def test_issiso(self):
        assert issiso(1)
        with pytest.raises(ValueError):
            issiso(1, strict=True)

        # SISO transfer function
        sys = tf([-1, 42], [1, 10])
        assert issiso(sys)
        assert issiso(sys, strict=True)

        # SISO state space system
        sys = tf2ss(sys)
        assert issiso(sys)
        assert issiso(sys, strict=True)

    @slycotonly
    def test_issiso_mimo(self):
        # MIMO transfer function
        sys = tf([[[-1, 41], [1]], [[1, 2], [3, 4]]],
                 [[[1, 10], [1, 20]], [[1, 30], [1, 40]]]);
        assert not issiso(sys)
        assert not issiso(sys, strict=True)

        # MIMO state space system
        sys = tf2ss(sys)
        assert not issiso(sys)
        assert not issiso(sys, strict=True)

    def test_damp(self):
        # Test the continuous time case.
        zeta = 0.1
        wn = 42
        p = -wn * zeta + 1j * wn * np.sqrt(1 - zeta**2)
        sys = tf(1, [1, 2 * zeta * wn, wn**2])
        expected = ([wn, wn], [zeta, zeta], [p, p.conjugate()])
        np.testing.assert_allclose(sys.damp(), expected)
        np.testing.assert_allclose(damp(sys), expected)

        # Also test the discrete time case.
        dt = 0.001
        sys_dt = c2d(sys, dt, method='matched')
        p_zplane = np.exp(p*dt)
        expected_dt = ([wn, wn], [zeta, zeta],
                       [p_zplane, p_zplane.conjugate()])
        np.testing.assert_almost_equal(sys_dt.damp(), expected_dt)
        np.testing.assert_almost_equal(damp(sys_dt), expected_dt)

        #also check that for a discrete system with a negative real pole the damp function can extract wn and zeta.
        p2_zplane = -0.2
        sys_dt2 = tf(1, [1, -p2_zplane], dt)
        wn2, zeta2, p2 = sys_dt2.damp()
        p2_splane = -wn2 * zeta2 + 1j * wn2 * np.sqrt(1 - zeta2**2)
        p2_zplane = np.exp(p2_splane * dt)
        np.testing.assert_almost_equal(p2, p2_zplane)

    def test_dcgain(self):
        sys = tf(84, [1, 2])
        np.testing.assert_allclose(sys.dcgain(), 42)
        np.testing.assert_allclose(dcgain(sys), 42)

    def test_bandwidth(self):
        # test a first-order system, compared with matlab
        sys1 = tf(0.1, [1, 0.1])
        np.testing.assert_allclose(sys1.bandwidth(), 0.099762834511098)
        np.testing.assert_allclose(bandwidth(sys1), 0.099762834511098)

        # test a second-order system, compared with matlab
        wn2 = 1
        zeta2 = 0.001
        sys2 = sys1 * tf(wn2**2, [1, 2*zeta2*wn2, wn2**2])
        np.testing.assert_allclose(sys2.bandwidth(), 0.101848388240241)
        np.testing.assert_allclose(bandwidth(sys2), 0.101848388240241)

        # test constant gain, bandwidth should be infinity
        sysAP = tf(1,1)
        np.testing.assert_allclose(bandwidth(sysAP), np.inf)

        # test integrator, bandwidth should return np.nan
        sysInt = tf(1, [1, 0])
        np.testing.assert_allclose(bandwidth(sysInt), np.nan)

        # test exception for system other than LTI
        np.testing.assert_raises(TypeError, bandwidth, 1)

        # test exception for system other than SISO system
        sysMIMO = tf([[[-1, 41], [1]], [[1, 2], [3, 4]]], 
                     [[[1, 10], [1, 20]], [[1, 30], [1, 40]]])
        np.testing.assert_raises(TypeError, bandwidth, sysMIMO)

        # test if raise exception if dbdrop is positive scalar
        np.testing.assert_raises(ValueError, bandwidth, sys1, 3)

    @pytest.mark.parametrize("dt1, dt2, expected",
                             [(None, None, True),
                              (None, 0, True),
                              (None, 1, True),
                              pytest.param(None, True, True,
                                           marks=pytest.mark.xfail(
                                               reason="returns false")),
                              (0, 0, True),
                              (0, 1, False),
                              (0, True, False),
                              (1, 1, True),
                              (1, 2, False),
                              (1, True, False),
                              (True, True, True)])
    def test_timebaseEqual_deprecated(self, dt1, dt2, expected):
        """Test that timbaseEqual throws a warning and returns as documented"""
        sys1 = tf([1], [1, 2, 3], dt1)
        sys2 = tf([1], [1, 4, 5], dt2)

        print(sys1.dt)
        print(sys2.dt)

        with pytest.deprecated_call():
            assert timebaseEqual(sys1, sys2) is expected
        # Make sure behaviour is symmetric
        with pytest.deprecated_call():
            assert timebaseEqual(sys2, sys1) is expected

    @pytest.mark.parametrize("dt1, dt2, expected",
                             [(None, None, None),
                              (None, 0, 0),
                              (None, 1, 1),
                              (None, True, True),
                              (True, True, True),
                              (True, 1, 1),
                              (1, 1, 1),
                              (0, 0, 0),
                              ])
    @pytest.mark.parametrize("sys1", [True, False])
    @pytest.mark.parametrize("sys2", [True, False])
    def test_common_timebase(self, dt1, dt2, expected, sys1, sys2):
        """Test that common_timbase adheres to :ref:`conventions-ref`"""
        i1 = tf([1], [1, 2, 3], dt1) if sys1 else dt1
        i2 = tf([1], [1, 4, 5], dt2) if sys2 else dt2
        assert common_timebase(i1, i2) == expected
        # Make sure behaviour is symmetric
        assert common_timebase(i2, i1) == expected

    @pytest.mark.parametrize("i1, i2",
                             [(True, 0),
                              (0, 1),
                              (1, 2)])
    def test_common_timebase_errors(self, i1, i2):
        """Test that common_timbase raises errors on invalid combinations"""
        with pytest.raises(ValueError):
            common_timebase(i1, i2)
        # Make sure behaviour is symmetric
        with pytest.raises(ValueError):
            common_timebase(i2, i1)

    @pytest.mark.parametrize("dt, ref, strictref",
                             [(None, True, False),
                              (0, False, False),
                              (1, True, True),
                              (True, True, True)])
    @pytest.mark.parametrize("objfun, arg",
                             [(LTI, ()),
                              (NonlinearIOSystem, (lambda x: x, ))])
    def test_isdtime(self, objfun, arg, dt, ref, strictref):
        """Test isdtime and isctime functions to follow convention"""
        obj = objfun(*arg, dt=dt)

        assert isdtime(obj) == ref
        assert isdtime(obj, strict=True) == strictref

        if dt is not None:
            ref = not ref
            strictref = not strictref
        assert isctime(obj) == ref
        assert isctime(obj, strict=True) == strictref

    @pytest.mark.usefixtures("editsdefaults")
    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd, ct.ss2io])
    @pytest.mark.parametrize("nstate, nout, ninp, omega, squeeze, shape", [
        [1, 1, 1, 0.1,          None,  ()],             # SISO
        [1, 1, 1, [0.1],        None,  (1,)],
        [1, 1, 1, [0.1, 1, 10], None,  (3,)],
        [2, 1, 1, 0.1,          True,  ()],
        [2, 1, 1, [0.1],        True,  ()],
        [2, 1, 1, [0.1, 1, 10], True,  (3,)],
        [3, 1, 1, 0.1,          False, (1, 1)],
        [3, 1, 1, [0.1],        False, (1, 1, 1)],
        [3, 1, 1, [0.1, 1, 10], False, (1, 1, 3)],
        [1, 2, 1, 0.1,          None,  (2, 1)],         # SIMO
        [1, 2, 1, [0.1],        None,  (2, 1, 1)],
        [1, 2, 1, [0.1, 1, 10], None,  (2, 1, 3)],
        [2, 2, 1, 0.1,          True,  (2,)],
        [2, 2, 1, [0.1],        True,  (2,)],
        [3, 2, 1, 0.1,          False, (2, 1)],
        [3, 2, 1, [0.1],        False, (2, 1, 1)],
        [3, 2, 1, [0.1, 1, 10], False, (2, 1, 3)],
        [1, 1, 2, [0.1, 1, 10], None, (1, 2, 3)],       # MISO
        [2, 1, 2, [0.1, 1, 10], True, (2, 3)],
        [3, 1, 2, [0.1, 1, 10], False, (1, 2, 3)],
        [1, 1, 2, 0.1,          None, (1, 2)],
        [1, 1, 2, 0.1,          True, (2,)],
        [1, 1, 2, 0.1,          False, (1, 2)],
        [1, 2, 2, [0.1, 1, 10], None, (2, 2, 3)],       # MIMO
        [2, 2, 2, [0.1, 1, 10], True, (2, 2, 3)],
        [3, 2, 2, [0.1, 1, 10], False, (2, 2, 3)],
        [1, 2, 2, 0.1, None, (2, 2)],
        [2, 2, 2, 0.1, True, (2, 2)],
        [3, 2, 2, 0.1, False, (2, 2)],
    ])
    @pytest.mark.parametrize("omega_type", ["numpy", "native"])
    def test_squeeze(self, fcn, nstate, nout, ninp, omega, squeeze, shape,
                     omega_type):
        """Test correct behavior of frequencey response squeeze parameter."""
        # Create the system to be tested
        if fcn == ct.frd:
            sys = fcn(ct.rss(nstate, nout, ninp), [1e-2, 1e-1, 1, 1e1, 1e2])
        elif fcn == ct.tf and (nout > 1 or ninp > 1) and not slycot_check():
            pytest.skip("Conversion of MIMO systems to transfer functions "
                        "requires slycot.")
        else:
            sys = fcn(ct.rss(nstate, nout, ninp))

        if omega_type == "numpy":
            omega = np.asarray(omega)
            isscalar = omega.ndim == 0
            # keep the ndarray type even for scalars
            s = np.asarray(omega * 1j)
        else:
            isscalar = not hasattr(omega, '__len__')
            if isscalar:
                s = omega*1J
            else:
                s = [w*1J for w in omega]

        # Call the transfer function directly and make sure shape is correct
        assert sys(s, squeeze=squeeze).shape == shape

        # Make sure that evalfr also works as expected
        assert ct.evalfr(sys, s, squeeze=squeeze).shape == shape

        # Check frequency response
        mag, phase, _ = sys.frequency_response(omega, squeeze=squeeze)
        if isscalar and squeeze is not True:
            # sys.frequency_response() expects a list as an argument
            # Add the shape of the input to the expected shape
            assert mag.shape == shape + (1,)
            assert phase.shape == shape + (1,)
        else:
            assert mag.shape == shape
            assert phase.shape == shape

        # Make sure the default shape lines up with squeeze=None case
        if squeeze is None:
            assert sys(s).shape == shape

        # Changing config.default to False should return 3D frequency response
        ct.config.set_defaults('control', squeeze_frequency_response=False)
        mag, phase, _ = sys.frequency_response(omega)
        if isscalar:
            assert mag.shape == (sys.noutputs, sys.ninputs, 1)
            assert phase.shape == (sys.noutputs, sys.ninputs, 1)
            assert sys(s).shape == (sys.noutputs, sys.ninputs)
            assert ct.evalfr(sys, s).shape == (sys.noutputs, sys.ninputs)
        else:
            assert mag.shape == (sys.noutputs, sys.ninputs, len(omega))
            assert phase.shape == (sys.noutputs, sys.ninputs, len(omega))
            assert sys(s).shape == \
                (sys.noutputs, sys.ninputs, len(omega))
            assert ct.evalfr(sys, s).shape == \
                (sys.noutputs, sys.ninputs, len(omega))

    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd, ct.ss2io])
    def test_squeeze_exceptions(self, fcn):
        if fcn == ct.frd:
            sys = fcn(ct.rss(2, 1, 1), [1e-2, 1e-1, 1, 1e1, 1e2])
        else:
            sys = fcn(ct.rss(2, 1, 1))

        with pytest.raises(ValueError, match="unknown squeeze value"):
            resp = sys.frequency_response([1], squeeze='siso')
        with pytest.raises(ValueError, match="unknown squeeze value"):
            sys([1j], squeeze='siso')
        with pytest.raises(ValueError, match="unknown squeeze value"):
            evalfr(sys, [1j], squeeze='siso')

        with pytest.raises(ValueError, match="must be 1D"):
            sys.frequency_response([[0.1, 1], [1, 10]])
        with pytest.raises(ValueError, match="must be 1D"):
            sys([[0.1j, 1j], [1j, 10j]])
        with pytest.raises(ValueError, match="must be 1D"):
            evalfr(sys, [[0.1j, 1j], [1j, 10j]])

        with pytest.warns(DeprecationWarning, match="LTI `inputs`"):
            ninputs = sys.inputs
        assert ninputs == sys.ninputs

        with pytest.warns(DeprecationWarning, match="LTI `outputs`"):
            noutputs = sys.outputs
        assert noutputs == sys.noutputs

        if isinstance(sys, ct.StateSpace):
            with pytest.warns(
                    DeprecationWarning, match="StateSpace `states`"):
                nstates = sys.states
            assert nstates == sys.nstates
