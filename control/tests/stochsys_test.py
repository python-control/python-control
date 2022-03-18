# stochsys_test.py - test stochastic system operations
# RMM, 16 Mar 2022

import numpy as np
import pytest
from control.tests.conftest import asmatarrayout

import control as ct
from control import lqe, dlqe, rss, drss, tf, ss, ControlArgument, slycot_check

# Utility function to check LQE answer
def check_LQE(L, P, poles, G, QN, RN):
    P_expected = asmatarrayout(np.sqrt(G @ QN @ G @ RN))
    L_expected = asmatarrayout(P_expected / RN)
    poles_expected = -np.squeeze(np.asarray(L_expected))
    np.testing.assert_almost_equal(P, P_expected)
    np.testing.assert_almost_equal(L, L_expected)
    np.testing.assert_almost_equal(poles, poles_expected)

# Utility function to check discrete LQE solutions
def check_DLQE(L, P, poles, G, QN, RN):
    P_expected = asmatarrayout(G.dot(QN).dot(G))
    L_expected = asmatarrayout(0)
    poles_expected = -np.squeeze(np.asarray(L_expected))
    np.testing.assert_almost_equal(P, P_expected)
    np.testing.assert_almost_equal(L, L_expected)
    np.testing.assert_almost_equal(poles, poles_expected)

@pytest.mark.parametrize("method", [None, 'slycot', 'scipy'])
def test_LQE(matarrayin, method):
    if method == 'slycot' and not slycot_check():
        return

    A, G, C, QN, RN = (matarrayin([[X]]) for X in [0., .1, 1., 10., 2.])
    L, P, poles = lqe(A, G, C, QN, RN, method=method)
    check_LQE(L, P, poles, G, QN, RN)

@pytest.mark.parametrize("cdlqe", [lqe, dlqe])
def test_lqe_call_format(cdlqe):
    # Create a random state space system for testing
    sys = rss(4, 3, 2)
    sys.dt = None           # treat as either continuous or discrete time

    # Covariance matrices
    Q = np.eye(sys.ninputs)
    R = np.eye(sys.noutputs)
    N = np.zeros((sys.ninputs, sys.noutputs))

    # Standard calling format
    Lref, Pref, Eref = cdlqe(sys.A, sys.B, sys.C, Q, R)
    
    # Call with system instead of matricees
    L, P, E = cdlqe(sys, Q, R)
    np.testing.assert_almost_equal(Lref, L)
    np.testing.assert_almost_equal(Pref, P)
    np.testing.assert_almost_equal(Eref, E)

    # Make sure we get an error if we specify N
    with pytest.raises(ct.ControlNotImplemented):
        L, P, E = cdlqe(sys, Q, R, N)
        
    # Inconsistent system dimensions
    with pytest.raises(ct.ControlDimension, match="Incompatible"):
        L, P, E = cdlqe(sys.A, sys.C, sys.B, Q, R)
        
    # Incorrect covariance matrix dimensions
    with pytest.raises(ct.ControlDimension, match="Incompatible"):
        L, P, E = cdlqe(sys.A, sys.B, sys.C, R, Q)
        
    # Too few input arguments
    with pytest.raises(ct.ControlArgument, match="not enough input"):
        L, P, E = cdlqe(sys.A, sys.C)

    # First argument is the wrong type (use SISO for non-slycot tests)
    sys_tf = tf(rss(3, 1, 1))
    sys_tf.dt = None        # treat as either continuous or discrete time
    with pytest.raises(ct.ControlArgument, match="LTI system must be"):
        L, P, E = cdlqe(sys_tf, Q, R)

@pytest.mark.parametrize("method", [None, 'slycot', 'scipy'])
def test_DLQE(matarrayin, method):
    if method == 'slycot' and not slycot_check():
        return

    A, G, C, QN, RN = (matarrayin([[X]]) for X in [0., .1, 1., 10., 2.])
    L, P, poles = dlqe(A, G, C, QN, RN, method=method)
    check_DLQE(L, P, poles, G, QN, RN)

def test_lqe_discrete():
    """Test overloading of lqe operator for discrete time systems"""
    csys = ct.rss(2, 1, 1)
    dsys = ct.drss(2, 1, 1)
    Q = np.eye(1)
    R = np.eye(1)

    # Calling with a system versus explicit A, B should be the sam
    K_csys, S_csys, E_csys = ct.lqe(csys, Q, R)
    K_expl, S_expl, E_expl = ct.lqe(csys.A, csys.B, csys.C, Q, R)
    np.testing.assert_almost_equal(K_csys, K_expl)
    np.testing.assert_almost_equal(S_csys, S_expl)
    np.testing.assert_almost_equal(E_csys, E_expl)
    
    # Calling lqe() with a discrete time system should call dlqe()
    K_lqe, S_lqe, E_lqe = ct.lqe(dsys, Q, R)
    K_dlqe, S_dlqe, E_dlqe = ct.dlqe(dsys, Q, R)
    np.testing.assert_almost_equal(K_lqe, K_dlqe)
    np.testing.assert_almost_equal(S_lqe, S_dlqe)
    np.testing.assert_almost_equal(E_lqe, E_dlqe)
    
    # Calling lqe() with no timebase should call lqe()
    asys = ct.ss(csys.A, csys.B, csys.C, csys.D, dt=None)
    K_asys, S_asys, E_asys = ct.lqe(asys, Q, R)
    K_expl, S_expl, E_expl = ct.lqe(csys.A, csys.B, csys.C, Q, R)
    np.testing.assert_almost_equal(K_asys, K_expl)
    np.testing.assert_almost_equal(S_asys, S_expl)
    np.testing.assert_almost_equal(E_asys, E_expl)
    
    # Calling dlqe() with a continuous time system should raise an error
    with pytest.raises(ControlArgument, match="called with a continuous"):
        K, S, E = ct.dlqe(csys, Q, R)
        
def test_estimator_iosys():
    sys = ct.drss(4, 2, 2, strictly_proper=True)

    Q, R = np.eye(sys.nstates), np.eye(sys.ninputs)
    K, _, _ = ct.dlqr(sys, Q, R)

    P0 = np.eye(sys.nstates)
    QN = np.eye(sys.ninputs)
    RN = np.eye(sys.noutputs)
    estim = ct.create_estimator_iosystem(sys, QN, RN, P0)
        
    ctrl, clsys = ct.create_statefbk_iosystem(sys, K, estimator=estim)

    # Extract the elements of the estimator
    est = estim.linearize(0, 0)
    Be1 = est.B[:sys.nstates, :sys.noutputs]
    Be2 = est.B[:sys.nstates, sys.noutputs:]
    A_clchk = np.block([
        [sys.A, -sys.B @ K],
        [Be1 @ sys.C, est.A[:sys.nstates, :sys.nstates] - Be2 @ K]
    ])
    B_clchk = np.block([
        [sys.B @ K, sys.B],
        [Be2 @ K, Be2]
    ])
    C_clchk = np.block([
        [sys.C, np.zeros((sys.noutputs, sys.nstates))],
        [np.zeros_like(K), -K]
    ])
    D_clchk = np.block([
        [np.zeros((sys.noutputs, sys.nstates + sys.ninputs))],
        [K, np.eye(sys.ninputs)]
    ])

    # Check to make sure everything matches
    cls = clsys.linearize(0, 0)
    nstates = sys.nstates
    np.testing.assert_almost_equal(cls.A[:2*nstates, :2*nstates], A_clchk)
    np.testing.assert_almost_equal(cls.B[:2*nstates, :], B_clchk)
    np.testing.assert_almost_equal(cls.C[:, :2*nstates], C_clchk)
    np.testing.assert_almost_equal(cls.D, D_clchk)


def test_estimator_errors():
    sys = ct.drss(4, 2, 2, strictly_proper=True)
    P0 = np.eye(sys.nstates)
    QN = np.eye(sys.ninputs)
    RN = np.eye(sys.noutputs)

    with pytest.raises(ct.ControlArgument, match="Input system must be I/O"):
        sys_tf = ct.tf([1], [1, 1], dt=True)
        estim = ct.create_estimator_iosystem(sys_tf, QN, RN)
            
    with pytest.raises(NotImplementedError, match="continuous time not"):
        sys_ct = ct.rss(4, 2, 2, strictly_proper=True)
        estim = ct.create_estimator_iosystem(sys_ct, QN, RN)
            
    with pytest.raises(ValueError, match="output must be full state"):
        C = np.eye(2, 4)
        estim = ct.create_estimator_iosystem(sys, QN, RN, C=C)

    with pytest.raises(ValueError, match="output is the wrong size"):
        sys_fs = ct.drss(4, 4, 2, strictly_proper=True)
        sys_fs.C = np.eye(4)
        C = np.eye(1, 4)
        estim = ct.create_estimator_iosystem(sys_fs, QN, RN, C=C)


def test_white_noise():
    # Scalar white noise signal
    T = np.linspace(0, 1000, 1000)
    R = 0.5
    V = ct.white_noise(T, R)
    assert abs(np.mean(V)) < 0.1                # can occassionally fail
    assert abs(np.cov(V) - 0.5) < 0.1           # can occassionally fail

    # Vector white noise signal
    R = [[0.5, 0], [0, 0.1]]
    V = ct.white_noise(T, R)
    assert abs(np.mean(V)) < 0.1                # can occassionally fail
    assert np.all(abs(np.cov(V) - R) < 0.1)     # can occassionally fail

    # Make sure time scaling works properly
    T = T / 10
    V = ct.white_noise(T, R)
    assert abs(np.mean(V)) < np.sqrt(10)        # can occassionally fail
    assert np.all(abs(np.cov(V) - R) < 10)      # can occassionally fail

    # Make sure discrete time works properly
    V = ct.white_noise(T, R, dt=T[1] - T[0])
    assert abs(np.mean(V)) < 0.1                # can occassionally fail
    assert np.all(abs(np.cov(V) - R) < 0.1)     # can occassionally fail

    # Test error conditions
    with pytest.raises(ValueError, match="T must be 1D"):
        V = ct.white_noise(R, R)

    with pytest.raises(ValueError, match="Q must be square"):
        R = np.outer(np.eye(2, 3), np.ones_like(T))
        V = ct.white_noise(T, R)

    with pytest.raises(ValueError, match="Time values must be equally"):
        T = np.logspace(0, 2, 100)
        R = [[0.5, 0], [0, 0.1]]
        V = ct.white_noise(T, R)


def test_correlation():
    # Create an uncorrelated random sigmal
    T = np.linspace(0, 1000, 1000)
    R = 0.5
    V = ct.white_noise(T, R)

    # Compute the correlation
    tau, Rtau = ct.correlation(T, V)

    # Make sure the correlation makes sense
    zero_index = np.where(tau == 0)
    np.testing.assert_almost_equal(Rtau[zero_index], np.cov(V), decimal=2)
    for i, t in enumerate(tau):
        if i == zero_index:
            continue
    assert abs(Rtau[i]) < 0.01

    # Try passing a second argument
    tau, Rneg = ct.correlation(T, V, -V)
    np.testing.assert_equal(Rtau, -Rneg)
    
    # Test error conditions
    with pytest.raises(ValueError, match="Time vector T must be 1D"):
        tau, Rtau = ct.correlation(V, V)

    with pytest.raises(ValueError, match="X and Y must be 2D"):
        tau, Rtau = ct.correlation(T, np.zeros((3, T.size, 2)))

    with pytest.raises(ValueError, match="X and Y must have same length as T"):
        tau, Rtau = ct.correlation(T, V[:, 0:-1])

    with pytest.raises(ValueError, match="Time values must be equally"):
        T = np.logspace(0, 2, T.size)
        tau, Rtau = ct.correlation(T, V)
