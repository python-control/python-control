"""matlab_test.py - test MATLAB compatibility

RMM, 30 Mar 2011 (based on TestMatlab from v0.4a)

This test suite just goes through and calls all of the MATLAB
functions using different systems and arguments to make sure that
nothing crashes.  Many test don't test actual functionality; the module
specific unit tests will do that.
"""

import numpy as np
import pytest
import scipy as sp
from scipy.linalg import eigvals

from control.matlab import ss, ss2tf, ssdata, tf, tf2ss, tfdata, rss, drss, frd
from control.matlab import parallel, series, feedback
from control.matlab import pole, zero, damp
from control.matlab import step, stepinfo, impulse, initial, lsim
from control.matlab import margin, dcgain
from control.matlab import linspace, logspace
from control.matlab import bode, rlocus, nyquist, nichols, ngrid, pzmap
from control.matlab import freqresp, evalfr
from control.matlab import hsvd, balred, modred, minreal
from control.matlab import place, place_varga, acker
from control.matlab import lqr, ctrb, obsv, gram
from control.matlab import pade
from control.matlab import unwrap, c2d, isctime, isdtime
from control.matlab import connect, append
from control.exception import ControlArgument

from control.frdata import FRD
from control.tests.conftest import slycotonly

# for running these through Matlab or Octave
'''
siso_ss1 = ss([1. -2.; 3. -4.], [5.; 7.], [6. 8.], [0])

siso_tf1 = tf([1], [1, 2, 1])
siso_tf2 = tf([1, 1], [1, 2, 3, 1])

siso_tf3 = tf(siso_ss1)
siso_ss2 = ss(siso_tf2)
siso_ss3 = ss(siso_tf3)
siso_tf4 = tf(siso_ss2)

A =[ 1. -2. 0.  0.;
     3. -4. 0.  0.;
     0.  0. 1. -2.;
     0.  0. 3. -4. ]
B = [ 5. 0.;
      7. 0.;
      0. 5.;
      0. 7. ]
C = [ 6. 8. 0. 0.;
      0. 0. 6. 8. ]
D = [ 9. 0.;
      0. 9. ]
mimo_ss1 = ss(A, B, C, D)

% all boring, since no cross-over
margin(siso_tf1)
margin(siso_tf2)
margin(siso_ss1)
margin(siso_ss2)

% make a bit better
[gm, pm, gmc, pmc] = margin(siso_ss2*siso_ss2*2)

'''


@pytest.fixture(scope="class")
def fixedseed():
    """Get consistent test results"""
    np.random.seed(0)


class tsystems:
    """struct for test systems"""

    pass


@pytest.mark.usefixtures("fixedseed")
class TestMatlab:
    """Test matlab style functions"""

    @pytest.fixture
    def siso(self):
        """Set up some systems for testing out MATLAB functions"""
        s = tsystems()

        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        s.ss1 = ss(A, B, C, D)

        # Create some transfer functions
        s.tf1 = tf([1], [1, 2, 1])
        s.tf2 = tf([1, 1], [1, 2, 3, 1])

        # Conversions
        s.tf3 = tf(s.ss1)
        s.ss2 = ss(s.tf2)
        s.ss3 = tf2ss(s.tf3)
        s.tf4 = ss2tf(s.ss2)
        return s

    @pytest.fixture
    def mimo(self):
        """Create MIMO system, contains ``siso_ss1`` twice"""
        m = tsystems()
        A = np.array([[1., -2., 0., 0.],
                      [3., -4., 0., 0.],
                      [0., 0., 1., -2.],
                      [0., 0., 3., -4.]])
        B = np.array([[5., 0.],
                      [7., 0.],
                      [0., 5.],
                      [0., 7.]])
        C = np.array([[6., 8., 0., 0.],
                      [0., 0., 6., 8.]])
        D = np.array([[9., 0.],
                      [0., 9.]])
        m.ss1 = ss(A, B, C, D)
        return m

    def testParallel(self, siso):
        """Call parallel()"""
        sys1 = parallel(siso.ss1, siso.ss2)
        sys1 = parallel(siso.ss1, siso.tf2)
        sys1 = parallel(siso.tf1, siso.ss2)
        sys1 = parallel(1, siso.ss2)
        sys1 = parallel(1, siso.tf2)
        sys1 = parallel(siso.ss1, 1)
        sys1 = parallel(siso.tf1, 1)

    def testSeries(self, siso):
        """Call series()"""
        sys1 = series(siso.ss1, siso.ss2)
        sys1 = series(siso.ss1, siso.tf2)
        sys1 = series(siso.tf1, siso.ss2)
        sys1 = series(1, siso.ss2)
        sys1 = series(1, siso.tf2)
        sys1 = series(siso.ss1, 1)
        sys1 = series(siso.tf1, 1)

    def testFeedback(self, siso):
        """Call feedback()"""
        sys1 = feedback(siso.ss1, siso.ss2)
        sys1 = feedback(siso.ss1, siso.tf2)
        sys1 = feedback(siso.tf1, siso.ss2)
        sys1 = feedback(1, siso.ss2)
        sys1 = feedback(1, siso.tf2)
        sys1 = feedback(siso.ss1, 1)
        sys1 = feedback(siso.tf1, 1)

    def testPoleZero(self, siso):
        """Call pole() and zero()"""
        pole(siso.ss1)
        pole(siso.tf1)
        pole(siso.tf2)
        zero(siso.ss1)
        zero(siso.tf1)
        zero(siso.tf2)

    @pytest.mark.parametrize(
        "subsys", ["tf1", "tf2"])
    def testPZmap(self, siso, subsys, mplcleanup):
        """Call pzmap()"""
        # pzmap(siso.ss1);         not implemented
        # pzmap(siso.ss2);         not implemented
        pzmap(getattr(siso, subsys))
        pzmap(getattr(siso, subsys), plot=False)

    def testStep(self, siso):
        """Test step()"""
        t = np.linspace(0, 1, 10)
        # Test transfer function
        yout, tout = step(siso.tf1, T=t)
        youttrue = np.array([0, 0.0057, 0.0213, 0.0446, 0.0739,
                             0.1075, 0.1443, 0.1832, 0.2235, 0.2642])
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Test SISO system with direct feedthrough
        sys = siso.ss1
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                             42.3227, 44.9694, 47.1599, 48.9776])

        yout, tout = step(sys, T=t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        yout, tout = step(sys, T=t, X0=0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        X0 = np.array([0, 0])
        yout, tout = step(sys, T=t, X0=X0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        yout, tout, xout = step(sys, T=t, X0=0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

    @slycotonly
    def testStep_mimo(self, mimo):
        """Test step for MIMO system"""
        sys = mimo.ss1
        t = np.linspace(0, 1, 10)
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                             42.3227, 44.9694, 47.1599, 48.9776])

        y_00, _t = step(sys, T=t, input=0, output=0)
        y_11, _t = step(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testStepinfo(self, siso):
        """Test the stepinfo function (no return value check)"""
        infodict = stepinfo(siso.ss1)
        assert isinstance(infodict, dict)
        assert len(infodict) == 9

    def testImpulse(self, siso):
        """Test impulse()"""
        t = np.linspace(0, 1, 10)
        # test transfer function
        yout, tout = impulse(siso.tf1, T=t)
        youttrue = np.array([0., 0.0994, 0.1779, 0.2388, 0.2850, 0.3188,
                             0.3423, 0.3573, 0.3654, 0.3679])
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        sys = siso.ss1
        youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344,
                             26.1668, 21.6292, 17.9245, 14.8945])
        # produce a warning for a system with direct feedthrough
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            # Test SISO system
            yout, tout = impulse(sys, T=t)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

        # produce a warning for a system with direct feedthrough
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            # Play with arguments
            yout, tout = impulse(sys, T=t, X0=0)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

        # produce a warning for a system with direct feedthrough
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            X0 = np.array([0, 0])
            yout, tout = impulse(sys, T=t, X0=X0)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

        # produce a warning for a system with direct feedthrough
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            yout, tout, xout = impulse(sys, T=t, X0=0, return_x=True)
            np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
            np.testing.assert_array_almost_equal(tout, t)

    @slycotonly
    def testImpulse_mimo(self, mimo):
        """Test impulse() for MIMO system"""
        t = np.linspace(0, 1, 10)
        youttrue = np.array([86., 70.1808, 57.3753, 46.9975, 38.5766, 31.7344,
                             26.1668, 21.6292, 17.9245, 14.8945])
        sys = mimo.ss1
        with pytest.warns(UserWarning, match="System has direct feedthrough"):
            y_00, _t = impulse(sys, T=t, input=0, output=0)
            y_11, _t = impulse(sys, T=t, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testInitial(self, siso):
        """Test initial() for SISO system"""
        t = np.linspace(0, 1, 10)
        x0 = np.array([[.5], [1.]])
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        sys = siso.ss1
        yout, tout = initial(sys, T=t, X0=x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

        # Play with arguments
        yout, tout, xout = initial(sys, T=t, X0=x0, return_x=True)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)

    @slycotonly
    def testInitial_mimo(self, mimo):
        """Test initial() for MIMO system"""
        t = np.linspace(0, 1, 10)
        x0 = np.array([[.5], [1.], [.5], [1.]])
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        sys = mimo.ss1
        y_00, _t = initial(sys, T=t, X0=x0, input=0, output=0)
        y_11, _t = initial(sys, T=t, X0=x0, input=1, output=1)
        np.testing.assert_array_almost_equal(y_00, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(y_11, youttrue, decimal=4)

    def testLsim(self, siso):
        """Test lsim() for SISO system"""
        t = np.linspace(0, 1, 10)

        # compute step response - test with state space, and transfer function
        # objects
        u = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1])
        youttrue = np.array([9., 17.6457, 24.7072, 30.4855, 35.2234, 39.1165,
                             42.3227, 44.9694, 47.1599, 48.9776])
        yout, tout, _xout = lsim(siso.ss1, u, t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)
        np.testing.assert_array_almost_equal(tout, t)
        with pytest.warns(UserWarning, match="Internal conversion"):
            yout, _t, _xout = lsim(siso.tf3, u, t)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

        # test with initial value and special algorithm for ``U=0``
        u = 0
        x0 = np.array([[.5], [1.]])
        youttrue = np.array([11., 8.1494, 5.9361, 4.2258, 2.9118, 1.9092,
                             1.1508, 0.5833, 0.1645, -0.1391])
        yout, _t, _xout = lsim(siso.ss1, u, t, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

    @slycotonly
    def testLsim_mimo(self, mimo):
        """Test lsim() for MIMO system.

        first system: initial value, second system: step response
        """
        t = np.linspace(0, 1, 10)

        u = np.array([[0., 1.], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        x0 = np.array([[.5], [1], [0], [0]])
        youttrue = np.array([[11., 9.], [8.1494, 17.6457],
                             [5.9361, 24.7072], [4.2258, 30.4855],
                             [2.9118, 35.2234], [1.9092, 39.1165],
                             [1.1508, 42.3227], [0.5833, 44.9694],
                             [0.1645, 47.1599], [-0.1391, 48.9776]])
        yout, _t, _xout = lsim(mimo.ss1, u, t, x0)
        np.testing.assert_array_almost_equal(yout, youttrue, decimal=4)

    def testMargin(self, siso):
        """Test margin()"""
        #! TODO: check results to make sure they are OK
        gm, pm, wcg, wcp = margin(siso.tf1)
        gm, pm, wcg, wcp = margin(siso.tf2)
        gm, pm, wcg, wcp = margin(siso.ss1)
        gm, pm, wcg, wcp = margin(siso.ss2)
        gm, pm, wcg, wcp = margin(siso.ss2 * siso.ss2 * 2)
        np.testing.assert_array_almost_equal(
            [gm, pm, wcg, wcp], [1.5451, 75.9933, 1.2720, 0.6559], decimal=3)

    def testDcgain(self, siso):
        """Test dcgain() for SISO system"""
        # Create different forms of a SISO system using scipy.signal
        A, B, C, D = siso.ss1.A, siso.ss1.B, siso.ss1.C, siso.ss1.D
        Z, P, k = sp.signal.ss2zpk(A, B, C, D)
        num, den = sp.signal.ss2tf(A, B, C, D)
        sys_ss = siso.ss1

        # Compute the gain with ``dcgain``
        gain_abcd = dcgain(A, B, C, D)
        gain_zpk = dcgain(Z, P, k)
        gain_numden = dcgain(np.squeeze(num), den)
        gain_sys_ss = dcgain(sys_ss)
        # print('\ngain_abcd:', gain_abcd, 'gain_zpk:', gain_zpk)
        # print('gain_numden:', gain_numden, 'gain_sys_ss:', gain_sys_ss)

        # Compute the gain with a long simulation
        t = linspace(0, 1000, 1000)
        y, _t = step(sys_ss, t)
        gain_sim = y[-1]
        # print('gain_sim:', gain_sim)

        # All gain values must be approximately equal to the known gain
        np.testing.assert_array_almost_equal(
            [gain_abcd, gain_zpk, gain_numden, gain_sys_ss, gain_sim],
            [59, 59, 59, 59, 59])

    def testDcgain_mimo(self, mimo):
        """Test dcgain() for MIMO system"""
        gain_mimo = dcgain(mimo.ss1)
        # print('gain_mimo: \n', gain_mimo)
        np.testing.assert_array_almost_equal(gain_mimo, [[59., 0],
                                                         [0, 59.]])

    def testBode(self, siso, mplcleanup):
        """Call bode()"""
        bode(siso.ss1)
        bode(siso.tf1)
        bode(siso.tf2)
        (mag, phase, freq) = bode(siso.tf2, plot=False)
        bode(siso.tf1, siso.tf2)
        w = logspace(-3, 3)
        bode(siso.ss1, w)
        bode(siso.ss1, siso.tf2, w)
        # Not yet implemented
        #  bode(siso.ss1, '-', siso.tf1, 'b--', siso.tf2, 'k.')

    @pytest.mark.parametrize("subsys", ["ss1", "tf1", "tf2"])
    def testRlocus(self, siso, subsys, mplcleanup):
        """Call rlocus()"""
        rlocus(getattr(siso, subsys))

    def testRlocus_list(self, siso, mplcleanup):
        """Test rlocus() with list"""
        klist = [1, 10, 100]
        rlist, klist_out = rlocus(siso.tf2, klist, plot=False)
        np.testing.assert_equal(len(rlist), len(klist))
        np.testing.assert_allclose(klist, klist_out)

    def testNyquist(self, siso):
        """Call nyquist()"""
        nyquist(siso.ss1)
        nyquist(siso.tf1)
        nyquist(siso.tf2)
        w = logspace(-3, 3)
        nyquist(siso.tf2, w)
        (real, imag, freq) = nyquist(siso.tf2, w, plot=False)

    @pytest.mark.parametrize("subsys", ["ss1", "tf1", "tf2"])
    def testNichols(self, siso, subsys, mplcleanup):
        """Call nichols()"""
        nichols(getattr(siso, subsys))

    def testNichols_logspace(self, siso, mplcleanup):
        """Call nichols() with logspace w"""
        w = logspace(-3, 3)
        nichols(siso.tf2, w)

    def testNichols_ngrid(self, siso, mplcleanup):
        """Call nichols() and ngrid()"""
        nichols(siso.tf2, grid=False)
        ngrid()

    def testFreqresp(self, siso):
        """Call freqresp()"""
        w = logspace(-3, 3)
        freqresp(siso.ss1, w)
        freqresp(siso.ss2, w)
        freqresp(siso.ss3, w)
        freqresp(siso.tf1, w)
        freqresp(siso.tf2, w)
        freqresp(siso.tf3, w)

    def testEvalfr(self, siso):
        """Call evalfr()"""
        w = 1j
        np.testing.assert_almost_equal(evalfr(siso.ss1, w), 44.8 - 21.4j)
        evalfr(siso.ss2, w)
        evalfr(siso.ss3, w)
        evalfr(siso.tf1, w)
        evalfr(siso.tf2, w)
        evalfr(siso.tf3, w)

    def testEvalfr_mimo(self, mimo):
        """Test evalfr() MIMO"""
        fr = evalfr(mimo.ss1, 1j)
        ref = np.array([[44.8 - 21.4j, 0.], [0., 44.8 - 21.4j]])
        np.testing.assert_array_almost_equal(fr, ref)

    @slycotonly
    def testHsvd(self, siso):
        """Call hsvd()"""
        hsvd(siso.ss1)
        hsvd(siso.ss2)
        hsvd(siso.ss3)

    @slycotonly
    def testBalred(self, siso):
        """Call balred()"""
        balred(siso.ss1, 1)
        balred(siso.ss2, 2)
        balred(siso.ss3, [2, 2])

    @slycotonly
    def testModred(self, siso):
        """Call modred()"""
        modred(siso.ss1, [1])
        modred(siso.ss2 * siso.ss1, [0, 1])
        modred(siso.ss1, [1], 'matchdc')
        modred(siso.ss1, [1], 'truncate')

    @slycotonly
    def testPlace_varga(self, siso):
        """Call place_varga()"""
        place_varga(siso.ss1.A, siso.ss1.B, [-2, -2])

    def testPlace(self, siso):
        """Call place()"""
        place(siso.ss1.A, siso.ss1.B, [-2, -2.5])

    def testAcker(self, siso):
        """Call acker()"""
        acker(siso.ss1.A, siso.ss1.B, [-2, -2.5])

    def testLQR(self, siso):
        """Call lqr()"""
        (K, S, E) = lqr(siso.ss1.A, siso.ss1.B, np.eye(2), np.eye(1))

        # Should work if [Q N;N' R] is positive semi-definite
        (K, S, E) = lqr(siso.ss2.A, siso.ss2.B, 10 * np.eye(3), np.eye(1),
                        [[1], [1], [2]])

    def testRss(self):
        """Call rss()"""
        rss(1)
        rss(2)
        rss(2, 1, 3)

    def testDrss(self):
        """Call drss()"""
        drss(1)
        drss(2)
        drss(2, 1, 3)

    def testCtrb(self, siso):
        """Call ctrb()"""
        ctrb(siso.ss1.A, siso.ss1.B)
        ctrb(siso.ss2.A, siso.ss2.B)

    def testObsv(self, siso):
        """Call obsv()"""
        obsv(siso.ss1.A, siso.ss1.C)
        obsv(siso.ss2.A, siso.ss2.C)

    @slycotonly
    def testGram(self, siso):
        """Call gram()"""
        gram(siso.ss1, 'c')
        gram(siso.ss2, 'c')
        gram(siso.ss1, 'o')
        gram(siso.ss2, 'o')

    def testPade(self):
        """Call pade()"""
        pade(1, 1)
        pade(1, 2)
        pade(5, 4)

    def testOpers(self, siso):
        """Use arithmetic operators"""
        siso.ss1 + siso.ss2
        siso.tf1 + siso.tf2
        siso.ss1 + siso.tf2
        siso.tf1 + siso.ss2
        siso.ss1 * siso.ss2
        siso.tf1 * siso.tf2
        siso.ss1 * siso.tf2
        siso.tf1 * siso.ss2
        # siso.ss1 / siso.ss2         not implemented yet
        # siso.tf1 / siso.tf2
        # siso.ss1 / siso.tf2
        # siso.tf1 / siso.ss2

    def testUnwrap(self):
        """Call unwrap()"""
        phase = np.array(range(1, 100)) / 10.
        wrapped = phase % (2 * np.pi)
        unwrapped = unwrap(wrapped)

    def testSISOssdata(self, siso):
        """Call ssdata()

        At least test for consistency between ss and tf
        """
        ssdata_1 = ssdata(siso.ss2)
        ssdata_2 = ssdata(siso.tf2)
        for i in range(len(ssdata_1)):
            np.testing.assert_array_almost_equal(ssdata_1[i], ssdata_2[i])

    @slycotonly
    def testMIMOssdata(self, mimo):
        """Test ssdata() MIMO"""
        m = (mimo.ss1.A, mimo.ss1.B, mimo.ss1.C, mimo.ss1.D)
        ssdata_1 = ssdata(mimo.ss1)
        for i in range(len(ssdata_1)):
            np.testing.assert_array_almost_equal(ssdata_1[i], m[i])

    def testSISOtfdata(self, siso):
        """Call tfdata()"""
        tfdata_1 = tfdata(siso.tf2)
        tfdata_2 = tfdata(siso.tf2)
        for i in range(len(tfdata_1)):
            np.testing.assert_array_almost_equal(tfdata_1[i], tfdata_2[i])

    def testDamp(self):
        """Test damp()"""
        A = np.array([[-0.2, 0.06, 0, -1],
                      [0, 0, 1, 0],
                      [-17, 0, -3.8, 1],
                      [9.4, 0, -0.4, -0.6]])
        B = np.array([[-0.01, 0.06],
                      [0, 0],
                      [-32, 5.4],
                      [2.6, -7]])
        C = np.eye(4)
        D = np.zeros((4, 2))
        sys = ss(A, B, C, D)
        wn, Z, p = damp(sys, False)
        # print (wn)
        np.testing.assert_array_almost_equal(
            wn, np.array([4.07381994, 3.28874827, 3.28874827,
                          1.08937685e-03]))
        np.testing.assert_array_almost_equal(
            Z, np.array([1.0, 0.07983139, 0.07983139, 1.0]))

    def testConnect(self):
        """Test append() and  connect()"""
        sys1 = ss([[1., -2],
                   [3., -4]],
                  [[5.],
                   [7]],
                  [[6, 8]],
                  [[9.]])
        sys2 = ss(-1., 1., 1., 0.)
        sys = append(sys1, sys2)
        Q = np.array([[1, 2],  # basically feedback, output 2 in 1
                      [2, -1]])
        sysc = connect(sys, Q, [2], [1, 2])
        # print(sysc)
        np.testing.assert_array_almost_equal(
            sysc.A, np.array([[1, -2, 5], [3, -4, 7], [-6, -8, -10]]))
        np.testing.assert_array_almost_equal(
            sysc.B, np.array([[0], [0], [1]]))
        np.testing.assert_array_almost_equal(
            sysc.C, np.array([[6, 8, 9], [0, 0, 1]]))
        np.testing.assert_array_almost_equal(
            sysc.D, np.array([[0], [0]]))

    def testConnect2(self):
        """Test append and connect() case 2"""
        sys = append(ss([[-5, -2.25],
                         [4, 0]],
                        [[2],
                         [0]],
                        [[0, 1.125]],
                        [[0]]),
                     ss([[-1.6667, 0],
                         [1, 0]],
                        [[2], [0]],
                        [[0, 3.3333]], [[0]]),
                     1)
        Q = [[1, 3],
             [2, 1],
             [3, -2]]
        sysc = connect(sys, Q, [3], [3, 1, 2])
        np.testing.assert_array_almost_equal(
            sysc.A, np.array([[-5, -2.25, 0, -6.6666],
                              [4, 0, 0, 0],
                              [0, 2.25, -1.6667, 0],
                              [0, 0, 1, 0]]))
        np.testing.assert_array_almost_equal(
            sysc.B, np.array([[2], [0], [0], [0]]))
        np.testing.assert_array_almost_equal(
            sysc.C, np.array([[0, 0, 0, -3.3333],
                              [0, 1.125, 0, 0],
                              [0, 0, 0, 3.3333]]))
        np.testing.assert_array_almost_equal(
            sysc.D, np.array([[1], [0], [0]]))

    def testFRD(self):
        """Test frd()"""
        h = tf([1], [1, 2, 2])
        omega = np.logspace(-1, 2, 10)
        frd1 = frd(h, omega)
        assert isinstance(frd1, FRD)
        frd2 = frd(frd1.fresp[0, 0, :], omega)
        assert isinstance(frd2, FRD)

    @slycotonly
    def testMinreal(self, verbose=False):
        """Test a minreal model reduction"""
        # A = [-2, 0.5, 0; 0.5, -0.3, 0; 0, 0, -0.1]
        A = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        # B = [0.3, -1.3; 0.1, 0; 1, 0]
        B = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        # C = [0, 0.1, 0; -0.3, -0.2, 0]
        C = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        # D = [0 -0.8; -0.3 0]
        D = [[0., -0.8], [-0.3, 0.]]
        # sys = ss(A, B, C, D)

        sys = ss(A, B, C, D)
        sysr = minreal(sys, verbose=verbose)
        assert sysr.nstates == 2
        assert sysr.ninputs == sys.ninputs
        assert sysr.noutputs == sys.noutputs
        np.testing.assert_array_almost_equal(
            eigvals(sysr.A), [-2.136154, -0.1638459])

        s = tf([1, 0], [1])
        h = (s+1)*(s+2.00000000001)/(s+2)/(s**2+s+1)
        hm = minreal(h, verbose=verbose)
        hr = (s+1)/(s**2+s+1)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(hm.den[0][0], hr.den[0][0])

    def testSS2cont(self):
        """Test c2d()"""
        sys = ss(
            np.array([[-3, 4, 2], [-1, -3, 0], [2, 5, 3]]),
            np.array([[1, 4], [-3, -3], [-2, 1]]),
            np.array([[4, 2, -3], [1, 4, 3]]),
            np.array([[-2, 4], [0, 1]]))
        sysd = c2d(sys, 0.1)
        np.testing.assert_array_almost_equal(
            np.array(
                [[ 0.742840837331905, 0.342242024293711,  0.203124211149560],
                 [-0.074130792143890, 0.724553295044645, -0.009143771143630],
                 [ 0.180264783290485, 0.544385612448419,  1.370501013067845]]),
            sysd.A)
        np.testing.assert_array_almost_equal(
            np.array([[ 0.012362066084719,  0.301932197918268],
                      [-0.260952977031384, -0.274201791021713],
                      [-0.304617775734327,  0.075182622718853]]),
            sysd.B)

    def testCombi01(self):
        """Test from a "real" case, combines tf, ss, connect and margin.

        This is a type 2 system, with phase starting at -180. The
        margin command should remove the solution for w = nearly zero.
        """
        # Example is a concocted two-body satellite with flexible link
        Jb = 400
        Jp = 1000
        k = 10
        b = 5

        # can now define an "s" variable, to make TF's
        s = tf([1, 0], [1])
        hb1 = 1/(Jb*s)
        hb2 = 1/s
        hp1 = 1/(Jp*s)
        hp2 = 1/s

        # convert to ss and append
        sat0 = append(ss(hb1), ss(hb2), k, b, ss(hp1), ss(hp2))

        # connection of the elements with connect call
        Q = [[1, -3, -4],  # link moment (spring, damper), feedback to body
             [2,  1,  0],  # link integrator to body velocity
             [3,  2, -6],  # spring input, th_b - th_p
             [4,  1, -5],  # damper input
             [5,  3,  4],  # link moment, acting on payload
             [6,  5,  0]]
        inputs = [1]
        outputs = [1, 2, 5, 6]
        sat1 = connect(sat0, Q, inputs, outputs)

        # matched notch filter
        wno = 0.19
        z1 = 0.05
        z2 = 0.7
        Hno = (1+2*z1/wno*s+s**2/wno**2)/(1+2*z2/wno*s+s**2/wno**2)

        # the controller, Kp = 1 for now
        Kp = 1.64
        tau_PD = 50.
        Hc = (1 + tau_PD*s)*Kp

        # start with the basic satellite model sat1, and get the
        # payload attitude response
        Hp = tf(np.array([0, 0, 0, 1])*sat1)

        # total open loop
        Hol = Hc*Hno*Hp

        gm, pm, wcg, wcp = margin(Hol)
        # print("%f %f %f %f" % (gm, pm, wcg, wcp))
        np.testing.assert_allclose(gm, 3.32065569155)
        np.testing.assert_allclose(pm, 46.9740430224)
        np.testing.assert_allclose(wcg, 0.176469728448)
        np.testing.assert_allclose(wcp, 0.0616288455466)

    def test_tf_string_args(self):
        """Make sure s and z are defined properly"""
        s = tf('s')
        G = (s + 1)/(s**2 + 2*s + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        assert isctime(G, strict=True)

        z = tf('z')
        G = (z + 1)/(z**2 + 2*z + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        assert isdtime(G, strict=True)

    def test_matlab_wrapper_exceptions(self):
        """Test out exceptions in matlab/wrappers.py"""
        sys = tf([1], [1, 2, 1])

        # Extra arguments in bode
        with pytest.raises(ControlArgument, match="not all arguments"):
            bode(sys, 'r-', [1e-2, 1e2], 5.0)

        # Multiple plot styles
        with pytest.warns(UserWarning, match="plot styles not implemented"):
            bode(sys, 'r-', sys, 'b--', [1e-2, 1e2])

        # Incorrect number of arguments to dcgain
        with pytest.raises(ValueError, match="needs either 1, 2, 3 or 4"):
            dcgain(1, 2, 3, 4, 5)

    def test_matlab_freqplot_passthru(self, mplcleanup):
        """Test nyquist and bode to make sure the pass arguments through"""
        sys = tf([1], [1, 2, 1])
        bode((sys,))            # Passing tuple will call bode_plot
        nyquist((sys,))         # Passing tuple will call nyquist_plot


#! TODO: not yet implemented
#    def testMIMOtfdata(self):
#        sisotf = ss2tf(siso.ss1)
#        tfdata_1 = tfdata(sisotf)
#        tfdata_2 = tfdata(mimo.ss1, input=0, output=0)
#        for i in range(len(tfdata)):
#            np.testing.assert_array_almost_equal(tfdata_1[i], tfdata_2[i])
