"""lti_test.py"""

import re

import numpy as np
import pytest

import control as ct
from control import NonlinearIOSystem, c2d, common_timebase, isctime, \
    isdtime, issiso, ss, tf, tf2ss
from control.exception import slycot_check
from control.lti import LTI, bandwidth, damp, dcgain, evalfr, poles, zeros
from control.tests.conftest import slycotonly


class TestLTI:
    @pytest.mark.parametrize("fun, args", [
        [tf, (126, [-1, 42])],
        [ss, ([[42]], [[1]], [[1]], 0)]
    ])
    def test_poles(self, fun, args):
        sys = fun(*args)
        np.testing.assert_allclose(sys.poles(), 42)
        np.testing.assert_allclose(poles(sys), 42)

        with pytest.raises(AttributeError, match="no attribute 'pole'"):
            sys.pole()

        with pytest.raises(AttributeError, match="no attribute 'pole'"):
            ct.pole(sys)

    @pytest.mark.parametrize("fun, args", [
        [tf, (126, [-1, 42])],
        [ss, ([[42]], [[1]], [[1]], 0)]
    ])
    def test_zeros(self, fun, args):
        sys = fun(*args)
        np.testing.assert_allclose(sys.zeros(), 42)
        np.testing.assert_allclose(zeros(sys), 42)

        with pytest.raises(AttributeError, match="no attribute 'zero'"):
            sys.zero()

        with pytest.raises(AttributeError, match="no attribute 'zero'"):
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
        # Test the continuous-time case.
        zeta = 0.1
        wn = 42
        p = -wn * zeta + 1j * wn * np.sqrt(1 - zeta**2)
        sys = tf(1, [1, 2 * zeta * wn, wn**2])
        expected = ([wn, wn], [zeta, zeta], [p, p.conjugate()])
        np.testing.assert_allclose(sys.damp(), expected)
        np.testing.assert_allclose(damp(sys), expected)

        # Also test the discrete-time case.
        dt = 0.001
        sys_dt = c2d(sys, dt, method='matched')
        p_zplane = np.exp(p*dt)
        expected_dt = ([wn, wn], [zeta, zeta],
                       [p_zplane, p_zplane.conjugate()])
        np.testing.assert_almost_equal(sys_dt.damp(), expected_dt)
        np.testing.assert_almost_equal(damp(sys_dt), expected_dt)

        # also check that for a discrete system with a negative real pole
        # the damp function can extract wn and zeta.
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
    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd])
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

    @pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd])
    def test_squeeze_exceptions(self, fcn):
        if fcn == ct.frd:
            sys = fcn(ct.rss(2, 1, 1), [1e-2, 1e-1, 1, 1e1, 1e2])
        else:
            sys = fcn(ct.rss(2, 1, 1))

        with pytest.raises(ValueError, match="unknown squeeze value"):
            sys.frequency_response([1], squeeze='siso')
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


@pytest.mark.parametrize(
    "outdx, inpdx, key",
    [('y[0]', 'u[1]', (0, 1)),
     (['y[0]'], ['u[1]'], (0, 1)),
     (slice(0, 1, 1), slice(1, 2, 1), (0, 1)),
     (['y[0]', 'y[1]'], ['u[1]', 'u[2]'], ([0, 1], [1, 2])),
     ([0, 'y[1]'], ['u[1]', 2], ([0, 1], [1, 2])),
     (slice(0, 2, 1), slice(1, 3, 1), ([0, 1], [1, 2])),
     (['y[2]', 'y[1]'], ['u[2]', 'u[0]'], ([2, 1], [2, 0])),
    ])
@pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd])
def test_subsys_indexing(fcn, outdx, inpdx, key):
    # Construct the base system and subsystem
    sys = ct.rss(4, 3, 3)
    subsys = sys[key]

    # Construct the system to be tested
    match fcn:
        case ct.frd:
            omega = np.logspace(-1, 1)
            sys = fcn(sys, omega)
            subsys_chk = fcn(subsys, omega)
        case _:
            sys = fcn(sys)
            subsys_chk = fcn(subsys)

    # Construct the subsystem
    subsys_fcn = sys[outdx, inpdx]

    # Check to make sure everythng matches up
    match fcn:
        case ct.frd:
            np.testing.assert_almost_equal(
                subsys_fcn.complex, subsys_chk.complex)
        case ct.ss:
            np.testing.assert_almost_equal(subsys_fcn.A, subsys_chk.A)
            np.testing.assert_almost_equal(subsys_fcn.B, subsys_chk.B)
            np.testing.assert_almost_equal(subsys_fcn.C, subsys_chk.C)
            np.testing.assert_almost_equal(subsys_fcn.D, subsys_chk.D)
        case ct.tf:
            omega = np.logspace(-1, 1)
            np.testing.assert_almost_equal(
                subsys_fcn.frequency_response(omega).complex,
                subsys_chk.frequency_response(omega).complex)


@pytest.mark.parametrize("op", [
    '__mul__', '__rmul__', '__add__', '__radd__', '__sub__', '__rsub__'])
@pytest.mark.parametrize("fcn", [ct.ss, ct.tf, ct.frd])
def test_scalar_algebra(op, fcn):
    sys_ss = ct.rss(4, 2, 2)
    match fcn:
        case ct.ss:
            sys = sys_ss
        case ct.tf:
            sys = ct.tf(sys_ss)
        case ct.frd:
            sys = ct.frd(sys_ss, [0.1, 1, 10])

    scaled = getattr(sys, op)(2)
    np.testing.assert_almost_equal(getattr(sys(1j), op)(2), scaled(1j))


@pytest.mark.parametrize(
    "fcn, args, kwargs, suppress, " +
    "repr_expected, str_expected, latex_expected", [
    (ct.ss, (-1e-12, 1, 2, 3), {}, False,
     r"StateSpace\([\s]*array\(\[\[-1.e-12\]\]\).*",
     None,                      # standard Numpy formatting
     r"10\^\{-12\}"),
    (ct.ss, (-1e-12, 1, 3, 3), {}, True,
     r"StateSpace\([\s]*array\(\[\[-0\.\]\]\).*",
     None,                      # standard Numpy formatting
     r"-0"),
    (ct.tf, ([1, 1e-12, 1], [1, 2, 1]), {}, False,
     r"\[1\.e\+00, 1\.e-12, 1.e\+00\]",
     r"s\^2 \+ 1e-12 s \+ 1",
     r"1 \\times 10\^\{-12\}"),
    (ct.tf, ([1, 1e-12, 1], [1, 2, 1]), {}, True,
     r"\[1\., 0., 1.\]",
     r"s\^2 \+ 1",
     r"\{s\^2 \+ 1\}"),
])
@pytest.mark.usefixtures("editsdefaults")
def test_printoptions(
        fcn, args, kwargs, suppress,
        repr_expected, str_expected, latex_expected):
    sys = fcn(*args, **kwargs)

    with np.printoptions(suppress=suppress):
        # Test loadable representation
        assert re.search(repr_expected, ct.iosys_repr(sys, 'eval')) is not None

        # Test string representation
        if str_expected is not None:
            assert re.search(str_expected, str(sys)) is not None

        # Test LaTeX/HTML representation
        assert re.search(latex_expected, sys._repr_html_()) is not None
