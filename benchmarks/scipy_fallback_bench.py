# scipy_fallback_bench.py - benchmarks for the SLICOT-free (scipy) fallbacks
# KL, 1 Jul 2026
#
# This benchmark compares the pure scipy/numpy fallbacks against the SLICOT
# (slycot) implementations for the matrix-equation routines that gained a
# ``method`` argument:
#
#   * generalized continuous Lyapunov   lyap(A, Q, E=E)
#   * generalized discrete   Lyapunov   dlyap(A, Q, E=E)
#   * discrete Sylvester (Stein)        dlyap(A, Q, C)
#
# The ``time_*`` methods time each (routine, size, method) combination.  The
# ``track_*`` method records the accuracy of the generalized-Lyapunov solution
# as a function of cond(E).  Every problem is constructed from a known solution
# ``X`` so that both speed and accuracy are measured against ground truth; the
# ``setup`` methods therefore build the matrices *outside* the timed region.
#
# When slycot is not installed the ``method='slycot'`` parameterizations are
# skipped (asv treats NotImplementedError raised in setup() as "skip"), so the
# suite runs with or without slycot.
#
# A single deterministic seed is used per problem, so runs are reproducible and
# comparable across commits.  (The tables discussed in PR #1234 were medians
# over several seeds; the ratios here match, as asv's repeated sampling
# averages the timing.)
#
# Run, e.g.:
#
#   PYTHONPATH=`pwd` asv run --python=python --bench scipy_fallback
#
# or, since these are plain classes, call the methods directly to reproduce the
# numbers without asv.

import numpy as np

import control as ct

# Fixed seed: deterministic, reproducible problems across runs and commits.
SEED = 20260627


def _slycot_available():
    try:
        return ct.slycot_check()
    except Exception:
        return False


def _spd(rng, n):
    """Return a symmetric positive-definite n-by-n matrix."""
    P = rng.standard_normal((n, n))
    return P @ P.T + n * np.eye(n)


def _make_gen_cont_lyap(rng, n):
    # A X E' + E X A' + Q = 0, built from a known SPD solution X (A Hurwitz).
    E = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    M = rng.standard_normal((n, n))
    S = M - (np.linalg.norm(M, 2) + 1.0) * np.eye(n)
    A = E @ S
    X = _spd(rng, n)
    Q = -(A @ X @ E.T + E @ X @ A.T)
    Q = 0.5 * (Q + Q.T)
    return ct.lyap, (A, Q), dict(E=E), X


def _make_gen_disc_lyap(rng, n):
    # A X A' - E X E' + Q = 0, built from a known SPD solution X (A Schur).
    E = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    M = rng.standard_normal((n, n))
    S = M / (np.linalg.norm(M, 2) + 1.0)
    A = E @ S
    X = _spd(rng, n)
    Q = -(A @ X @ A.T - E @ X @ E.T)
    Q = 0.5 * (Q + Q.T)
    return ct.dlyap, (A, Q), dict(E=E), X


def _make_disc_sylvester(rng, n):
    # A X Q' - X + C = 0 (discrete Sylvester / Stein), from a known X.
    MA = rng.standard_normal((n, n))
    MQ = rng.standard_normal((n, n))
    A = MA / (np.linalg.norm(MA, 2) + 1.0)
    Q = MQ / (np.linalg.norm(MQ, 2) + 1.0)
    X = rng.standard_normal((n, n))
    C = X - A @ X @ Q.T
    return ct.dlyap, (A, Q, C), dict(), X


_MAKERS = {
    'gen_cont_lyap': _make_gen_cont_lyap,
    'gen_disc_lyap': _make_gen_disc_lyap,
    'disc_sylvester': _make_disc_sylvester,
}


class MatrixEquationTiming:
    """Time the scipy fallback against slycot for the ``method=`` routines."""

    params = (
        ['gen_cont_lyap', 'gen_disc_lyap', 'disc_sylvester'],
        [10, 50, 100, 200, 400],
        ['scipy', 'slycot'],
    )
    param_names = ['routine', 'n', 'method']
    timeout = 120

    def setup(self, routine, n, method):
        if method == 'slycot' and not _slycot_available():
            raise NotImplementedError("slycot not available")
        rng = np.random.default_rng(SEED)
        self.func, self.args, self.kwargs, X = _MAKERS[routine](rng, n)
        # Confirm the method actually solves the problem before timing it.
        Xhat = self.func(*self.args, method=method, **self.kwargs)
        relerr = np.linalg.norm(Xhat - X, 'fro') / np.linalg.norm(X, 'fro')
        assert relerr < 1e-6, f"{routine} {method} n={n}: relerr={relerr:.1e}"

    def time_solve(self, routine, n, method):
        self.func(*self.args, method=method, **self.kwargs)


class GenLyapAccuracy:
    """Track generalized continuous Lyapunov accuracy versus cond(E).

    Both the scipy and slycot paths require E nonsingular and degrade together
    as E becomes ill-conditioned (the problem is itself about cond(E)**2
    conditioned); this benchmark records that, rather than timing.
    """

    params = (
        [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12],
        ['scipy', 'slycot'],
    )
    param_names = ['cond_E', 'method']
    unit = "relative error"
    n = 100

    def setup(self, cond_E, method):
        if method == 'slycot' and not _slycot_available():
            raise NotImplementedError("slycot not available")
        n = self.n
        rng = np.random.default_rng(SEED)
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        M = rng.standard_normal((n, n))
        S = M - (np.linalg.norm(M, 2) + 1.0) * np.eye(n)
        X = _spd(rng, n)
        # E with prescribed condition number: singular values spanning cond_E.
        E = (U * np.logspace(0, -np.log10(cond_E), n)) @ V.T
        A = E @ S
        Q = -(A @ X @ E.T + E @ X @ A.T)
        self.A, self.Q, self.E, self.X = A, 0.5 * (Q + Q.T), E, X

    def track_relerr(self, cond_E, method):
        import warnings
        with warnings.catch_warnings():
            # Ill-conditioned E deliberately triggers the accuracy warning.
            warnings.simplefilter("ignore")
            Xhat = ct.lyap(self.A, self.Q, E=self.E, method=method)
        return float(np.linalg.norm(Xhat - self.X, 'fro')
                     / np.linalg.norm(self.X, 'fro'))
