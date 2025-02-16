"""modelsimp_array_test.py - test model reduction functions

RMM, 30 Mar 2011 (based on TestModelSimp from v0.4a)
"""

import warnings

import numpy as np
import pytest

import control as ct
from control import StateSpace, TimeResponseData, c2d, forced_response, \
    impulse_response, rss, step_response, tf
from control.exception import ControlArgument, ControlDimension
from control.modelsimp import balred, eigensys_realization, hsvd, markov, \
    modred
from control.tests.conftest import slycotonly


class TestModelsimp:
    """Test model reduction functions"""

    @slycotonly
    def testHSVD(self):
        A = np.array([[1., -2.], [3., -4.]])
        B = np.array([[5.], [7.]])
        C = np.array([[6., 8.]])
        D = np.array([[9.]])
        sys = StateSpace(A, B, C, D)
        hsv = hsvd(sys)
        hsvtrue = np.array([24.42686, 0.5731395])  # from MATLAB
        np.testing.assert_array_almost_equal(hsv, hsvtrue)

        # test for correct return type: ALWAYS return ndarray, even when
        # use_numpy_matrix(True) was used
        assert isinstance(hsv, np.ndarray)
        assert not isinstance(hsv, np.matrix)

    def testMarkovSignature(self):
        U = np.array([[1., 1., 1., 1., 1., 1., 1.]])
        Y = U
        response = TimeResponseData(time=np.arange(U.shape[-1]),
                                    outputs=Y,
                                    output_labels='y',
                                    inputs=U,
                                    input_labels='u',
                                    )

        # setup
        m = 3
        Htrue = np.array([1., 0., 0.])
        Htrue_l = np.array([1., 0., 0., 0., 0., 0., 0.])

        # test not enough input arguments
        with pytest.raises(ControlArgument):
            H = markov(Y)
        with pytest.raises(ControlArgument):
            H = markov()

        # too many positional arguments
        with pytest.raises(ControlArgument):
            H = markov(Y,U,m,1)
        with pytest.raises(ControlArgument):
            H = markov(response,m,1)

        # too many positional arguments
        with pytest.raises(ControlDimension):
            U2 = np.hstack([U,U])
            H = markov(Y,U2,m)

        # not enough data
        with pytest.warns(Warning):
            H = markov(Y,U,8)

        # Basic Usage, m=l
        H = markov(Y, U)
        np.testing.assert_array_almost_equal(H, Htrue_l)

        H = markov(response)
        np.testing.assert_array_almost_equal(H, Htrue_l)

        # Basic Usage, m
        H = markov(Y, U, m)
        np.testing.assert_array_almost_equal(H, Htrue)

        H = markov(response, m)
        np.testing.assert_array_almost_equal(H, Htrue)

        H = markov(Y, U, m=m)
        np.testing.assert_array_almost_equal(H, Htrue)

        H = markov(response, m=m)
        np.testing.assert_array_almost_equal(H, Htrue)

        response.transpose=False
        H = markov(response, m=m)
        np.testing.assert_array_almost_equal(H, Htrue)

        # Make sure that transposed data also works, siso
        HT = markov(Y.T, U.T, m, transpose=True)
        np.testing.assert_array_almost_equal(HT, np.transpose(Htrue))

        response.transpose = True
        HT = markov(response, m)
        np.testing.assert_array_almost_equal(HT, np.transpose(Htrue))
        response.transpose=False

        # Test example from docstring
        # TODO: There is a problem here, last markov parameter does not fit
        # the approximation error could be to big
        Htrue = np.array([0, 1., -0.5])
        T = np.linspace(0, 10, 100)
        U = np.ones((1, 100))
        T, Y = forced_response(tf([1], [1, 0.5], True), T, U)
        H = markov(Y, U, 4, dt=True)
        np.testing.assert_array_almost_equal(H[:3], Htrue[:3])

        response = forced_response(tf([1], [1, 0.5], True), T, U)
        H = markov(response, 4, dt=True)
        np.testing.assert_array_almost_equal(H[:3], Htrue[:3])

        # Test example from issue #395
        inp = np.array([1, 2])
        outp = np.array([2, 4])
        mrk = markov(outp, inp, 1, transpose=False)
        np.testing.assert_almost_equal(mrk, 2.)

        # Test mimo example
        # Mechanical Vibrations: Theory and Application, SI Edition, 1st ed.
        # Figure 6.5 / Example 6.7
        m1, k1, c1 = 1., 4., 1.
        m2, k2, c2 = 2., 2., 1.
        k3, c3 = 6., 2.

        A = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [-(k1+k2)/m1, (k2)/m1, -(c1+c2)/m1, c2/m1],
            [(k2)/m2, -(k2+k3)/m2, c2/m2, -(c2+c3)/m2]
        ])
        B = np.array([[0.,0.],[0.,0.],[1/m1,0.],[0.,1/m2]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0]])
        D = np.zeros((2,2))

        sys = StateSpace(A, B, C, D)
        dt = 0.25
        sysd = sys.sample(dt, method='zoh')

        T = np.arange(0,100,dt)
        U = np.random.randn(sysd.B.shape[-1], len(T))
        response = forced_response(sysd, U=U)
        Y = response.outputs

        m = 100
        _, Htrue = impulse_response(sysd, T=dt*(m-1))


        # test array_like
        H = markov(Y, U, m, dt=dt)
        np.testing.assert_array_almost_equal(H, Htrue)

        # test array_like, truncate
        H = markov(Y, U, m, dt=dt, truncate=True)
        np.testing.assert_array_almost_equal(H, Htrue)

        # test array_like, transpose
        HT = markov(Y.T, U.T, m, dt=dt, transpose=True)
        np.testing.assert_array_almost_equal(HT, np.transpose(Htrue))

        # test response data
        H = markov(response, m, dt=dt)
        np.testing.assert_array_almost_equal(H, Htrue)

        # test response data
        H = markov(response, m, dt=dt, truncate=True)
        np.testing.assert_array_almost_equal(H, Htrue)

        # test response data, transpose
        response.transpose = True
        HT = markov(response, m, dt=dt)
        np.testing.assert_array_almost_equal(HT, np.transpose(Htrue))


    # Make sure markov() returns the right answer
    @pytest.mark.parametrize("k, m, n",
                             [(2, 2, 2),
                              (2, 5, 5),
                              (5, 2, 2),
                              (5, 5, 5),
                              (5, 10, 10)])
    def testMarkovResults(self, k, m, n):
        #
        # Test over a range of parameters
        #
        # k = order of the system
        # m = number of Markov parameters
        # n = size of the data vector
        #
        # Values *should* match exactly for n = m, otherewise you get a
        # close match but errors due to the assumption that C A^k B =
        # 0 for k > m-2 (see modelsimp.py).
        #

        # Generate stable continuous-time system
        Hc = rss(k, 1, 1)

        # Choose sampling time based on fastest time constant / 10
        w, _ = np.linalg.eig(Hc.A)
        Ts = np.min(-np.real(w)) / 10.

        # Convert to a discrete-time system via sampling
        Hd = c2d(Hc, Ts, 'zoh')

        # Compute the Markov parameters from state space
        Mtrue = np.hstack([Hd.D] + [
            Hd.C @ np.linalg.matrix_power(Hd.A, i) @ Hd.B
            for i in range(m-1)])

        Mtrue = np.squeeze(Mtrue)

        # Generate input/output data
        T = np.array(range(n)) * Ts
        U = np.cos(T) + np.sin(T/np.pi)

        ir_true = impulse_response(Hd,T)
        Mtrue_scaled = ir_true[1][:m]

        # Compare to results from markov()
        # experimentally determined probability to get non matching results
        # with rtot=1e-6 and atol=1e-8 due to numerical errors
        # for k=5, m=n=10: 0.015 %
        T, Y = forced_response(Hd, T, U, squeeze=True)
        Mcomp = markov(Y, U, m, dt=True)
        Mcomp_scaled = markov(Y, U, m, dt=Ts)

        np.testing.assert_allclose(Mtrue, Mcomp, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(Mtrue_scaled, Mcomp_scaled, rtol=1e-6, atol=1e-8)

        response = forced_response(Hd, T, U, squeeze=True)
        Mcomp = markov(response, m, dt=True)
        Mcomp_scaled = markov(response, m, dt=Ts)

        np.testing.assert_allclose(Mtrue, Mcomp, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            Mtrue_scaled, Mcomp_scaled, rtol=1e-6, atol=1e-8)

    def testERASignature(self):

        # test siso
        # Katayama, Subspace Methods for System Identification
        # Example 6.1, Fibonacci sequence
        H_true = np.array([0.,1.,1.,2.,3.,5.,8.,13.,21.,34.])

        # A realization of fibonacci impulse response
        A = np.array([[0., 1.],[1., 1.,]])
        B = np.array([[1.],[1.,]])
        C = np.array([[1., 0.,]])
        D = np.array([[0.,]])

        T = np.arange(0,10,1)
        sysd_true = StateSpace(A,B,C,D,True)
        ir_true = impulse_response(sysd_true,T=T)

        # test TimeResponseData
        sysd_est, _  = eigensys_realization(ir_true,r=2)
        ir_est = impulse_response(sysd_est, T=T)
        _, H_est = ir_est

        np.testing.assert_allclose(H_true, H_est, rtol=1e-6, atol=1e-8)

        # test ndarray
        _, YY_true = ir_true
        sysd_est, _  = eigensys_realization(YY_true,r=2)
        ir_est = impulse_response(sysd_est, T=T)
        _, H_est = ir_est

        np.testing.assert_allclose(H_true, H_est, rtol=1e-6, atol=1e-8)

        # test mimo
        # Mechanical Vibrations: Theory and Application, SI Edition, 1st ed.
        # Figure 6.5 / Example 6.7
        # m q_dd + c q_d + k q = f
        m1, k1, c1 = 1., 4., 1.
        m2, k2, c2 = 2., 2., 1.
        k3, c3 = 6., 2.

        A = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [-(k1+k2)/m1, (k2)/m1, -(c1+c2)/m1, c2/m1],
            [(k2)/m2, -(k2+k3)/m2, c2/m2, -(c2+c3)/m2]
        ])
        B = np.array([[0.,0.],[0.,0.],[1/m1,0.],[0.,1/m2]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0]])
        D = np.zeros((2,2))

        sys = StateSpace(A, B, C, D)

        dt = 0.1
        T = np.arange(0,10,dt)
        sysd_true = sys.sample(dt, method='zoh')
        ir_true = impulse_response(sysd_true, T=T)

        # test TimeResponseData
        sysd_est, _ = eigensys_realization(ir_true,r=4,dt=dt)

        step_true = step_response(sysd_true)
        step_est = step_response(sysd_est)

        np.testing.assert_allclose(step_true.outputs,
                                   step_est.outputs,
                                   rtol=1e-6, atol=1e-8)

        # test ndarray
        _, YY_true = ir_true
        sysd_est, _  = eigensys_realization(YY_true,r=4,dt=dt)

        step_true = step_response(sysd_true, T=T)
        step_est = step_response(sysd_est, T=T)

        np.testing.assert_allclose(step_true.outputs,
                                   step_est.outputs,
                                   rtol=1e-6, atol=1e-8)


    def testModredMatchDC(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = np.array([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = np.array([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = np.array([[0.]])
        sys = StateSpace(A, B, C, D)
        rsys = modred(sys,[2, 3],'matchdc')
        Artrue = np.array([[-4.431, -4.552], [-4.552, -5.361]])
        Brtrue = np.array([[-1.362], [-1.031]])
        Crtrue = np.array([[-1.362, -1.031]])
        Drtrue = np.array([[-0.08384]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue, decimal=3)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue, decimal=2)

    def testModredUnstable(self):
        """Check if warning is issued when an unstable system is given"""
        A = np.array(
            [[4.5418, 3.3999, 5.0342, 4.3808],
             [0.3890, 0.3599, 0.4195, 0.1760],
             [-4.2117, -3.2395, -4.6760, -4.2180],
             [0.0052, 0.0429, 0.0155, 0.2743]])
        B = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        C = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        D = np.array([[0.0, 0.0], [0.0, 0.0]])
        sys = StateSpace(A, B, C, D)

        # Make sure we get a warning message
        with pytest.warns(UserWarning, match="System is unstable"):
            newsys1 = modred(sys, [2, 3])

        # Make sure we can turn the warning off
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            newsys2 = ct.model_reduction(sys, [2, 3], warn_unstable=False)
            np.testing.assert_equal(newsys1.A, newsys2.A)

    def testModredTruncate(self):
        #balanced realization computed in matlab for the transfer function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-1.958, -1.194, 1.824, -1.464],
             [-1.194, -0.8344, 2.563, -1.351],
             [-1.824, -2.563, -1.124, 2.704],
             [-1.464, -1.351, -2.704, -11.08]])
        B = np.array([[-0.9057], [-0.4068], [-0.3263], [-0.3474]])
        C = np.array([[-0.9057, -0.4068, 0.3263, -0.3474]])
        D = np.array([[0.]])
        sys = StateSpace(A, B, C, D)
        rsys = modred(sys,[2, 3],'truncate')
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[-0.9057], [-0.4068]])
        Crtrue = np.array([[-0.9057, -0.4068]])
        Drtrue = np.array([[0.]])
        np.testing.assert_array_almost_equal(rsys.A, Artrue)
        np.testing.assert_array_almost_equal(rsys.B, Brtrue)
        np.testing.assert_array_almost_equal(rsys.C, Crtrue)
        np.testing.assert_array_almost_equal(rsys.D, Drtrue)


    @slycotonly
    def testBalredTruncate(self):
        # controlable canonical realization computed in matlab for the transfer
        # function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = np.array([[2.], [0.], [0.], [0.]])
        C = np.array([[0.5, 0.6875, 0.7031, 0.5]])
        D = np.array([[0.]])

        sys = StateSpace(A, B, C, D)
        orders = 2
        rsys = balred(sys, orders, method='truncate')
        Ar, Br, Cr, Dr = rsys.A, rsys.B, rsys.C, rsys.D

        # Result from MATLAB
        Artrue = np.array([[-1.958, -1.194], [-1.194, -0.8344]])
        Brtrue = np.array([[0.9057], [0.4068]])
        Crtrue = np.array([[0.9057, 0.4068]])
        Drtrue = np.array([[0.]])

        # Look for possible changes in state in slycot
        T1 = np.array([[1, 0], [0, -1]])
        T2 = np.array([[-1, 0], [0, 1]])
        T3 = np.array([[0, 1], [1, 0]])
        for T in (T1, T2, T3):
            if np.allclose(T @ Ar @ T, Artrue, atol=1e-2, rtol=1e-2):
                # Apply a similarity transformation
                Ar, Br, Cr = T @ Ar @ T, T @ Br, Cr @ T
                break

        # Make sure we got the correct answer
        np.testing.assert_array_almost_equal(Ar, Artrue, decimal=2)
        np.testing.assert_array_almost_equal(Br, Brtrue, decimal=4)
        np.testing.assert_array_almost_equal(Cr, Crtrue, decimal=4)
        np.testing.assert_array_almost_equal(Dr, Drtrue, decimal=4)

    @slycotonly
    def testBalredMatchDC(self):
        # controlable canonical realization computed in matlab for the transfer
        # function:
        # num = [1 11 45 32], den = [1 15 60 200 60]
        A = np.array(
            [[-15., -7.5, -6.25, -1.875],
             [8., 0., 0., 0.],
             [0., 4., 0., 0.],
             [0., 0., 1., 0.]])
        B = np.array([[2.], [0.], [0.], [0.]])
        C = np.array([[0.5, 0.6875, 0.7031, 0.5]])
        D = np.array([[0.]])

        sys = StateSpace(A, B, C, D)
        orders = 2
        rsys = balred(sys,orders,method='matchdc')
        Ar, Br, Cr, Dr = rsys.A, rsys.B, rsys.C, rsys.D

        # Result from MATLAB
        Artrue = np.array(
            [[-4.43094773, -4.55232904],
             [-4.55232904, -5.36195206]])
        Brtrue = np.array([[1.36235673], [1.03114388]])
        Crtrue = np.array([[1.36235673, 1.03114388]])
        Drtrue = np.array([[-0.08383902]])

        # Look for possible changes in state in slycot
        T1 = np.array([[1, 0], [0, -1]])
        T2 = np.array([[-1, 0], [0, 1]])
        T3 = np.array([[0, 1], [1, 0]])
        for T in (T1, T2, T3):
            if np.allclose(T @ Ar @ T, Artrue, atol=1e-2, rtol=1e-2):
                # Apply a similarity transformation
                Ar, Br, Cr = T @ Ar @ T, T @ Br, Cr @ T
                break

        # Make sure we got the correct answer
        np.testing.assert_array_almost_equal(Ar, Artrue, decimal=2)
        np.testing.assert_array_almost_equal(Br, Brtrue, decimal=4)
        np.testing.assert_array_almost_equal(Cr, Crtrue, decimal=4)
        np.testing.assert_array_almost_equal(Dr, Drtrue, decimal=4)


@pytest.mark.parametrize("kwargs, nstates, noutputs, ninputs", [
    ({'elim_states': [1, 3]}, 3, 3, 3),
    ({'elim_inputs': [1, 2], 'keep_states': [1, 3]}, 2, 3, 1),
    ({'elim_outputs': [1, 2], 'keep_inputs': [0, 1],}, 5, 1, 2),
    ({'keep_states': [2, 0], 'keep_outputs': [0, 1]}, 2, 2, 3),
    ({'keep_states': slice(0, 4, 2), 'keep_outputs': slice(None, 2)}, 2, 2, 3),
    ({'keep_states': ['x[0]', 'x[3]'], 'keep_inputs': 'u[0]'}, 2, 3, 1),
    ({'elim_inputs': [0, 1, 2]}, 5, 3, 0),              # no inputs
    ({'elim_outputs': [0, 1, 2]}, 5, 0, 3),             # no outputs
    ({'elim_states': [0, 1, 2, 3, 4]}, 0, 3, 3),        # no states
    ({'elim_states': [0, 1], 'keep_states': [1, 2]}, None, None, None),
])
@pytest.mark.parametrize("method", ['truncate', 'matchdc'])
def test_model_reduction(method, kwargs, nstates, noutputs, ninputs):
    sys = ct.rss(5, 3, 3)

    if nstates is None:
        # Arguments should generate an error
        with pytest.raises(ValueError, match="can't provide both"):
            red = ct.model_reduction(sys, **kwargs, method=method)
        return
    else:
        red = ct.model_reduction(sys, **kwargs, method=method)

    assert red.nstates == nstates
    assert red.ninputs == ninputs
    assert red.noutputs == noutputs

    if method == 'matchdc':
        # Define a new system with truncated inputs and outputs
        # (assumes we always keep the initial inputs and outputs)
        chk = ct.ss(
            sys.A, sys.B[:, :ninputs], sys.C[:noutputs, :],
            sys.D[:noutputs, :][:, :ninputs])
        np.testing.assert_allclose(red(0), chk(0))
