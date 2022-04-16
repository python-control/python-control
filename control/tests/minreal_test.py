"""minreal_test.py - test state space class

Rvp, 13 Jun 2013
"""

import numpy as np
from scipy.linalg import eigvals
import pytest

from control import rss, ss, zeros
from control.statesp import StateSpace
from control.xferfcn import TransferFunction
from itertools import permutations
from control.tests.conftest import slycotonly


@pytest.fixture
def fixedseed(scope="class"):
    np.random.seed(5)


@slycotonly
@pytest.mark.usefixtures("fixedseed")
class TestMinreal:
    """Tests for the StateSpace class."""

    def assert_numden_almost_equal(self, n1, n2, d1, d2):
        n1[np.abs(n1) < 1e-10] = 0.
        n1 = np.trim_zeros(n1)
        d1[np.abs(d1) < 1e-10] = 0.
        d1 = np.trim_zeros(d1)
        n2[np.abs(n2) < 1e-10] = 0.
        n2 = np.trim_zeros(n2)
        d2[np.abs(d2) < 1e-10] = 0.
        d2 = np.trim_zeros(d2)
        np.testing.assert_array_almost_equal(n1, n2)
        np.testing.assert_array_almost_equal(d2, d2)

    def testMinrealBrute(self):

        # depending on the seed and minreal performance, a number of
        # reductions is produced. If random gen or minreal change, this
        # will be likely to fail
        nreductions = 0

        for n, m, p in permutations(range(1,6), 3):
            s = rss(n, p, m)
            sr = s.minreal()
            if s.nstates > sr.nstates:
                nreductions += 1
            else:
                # Check to make sure that poles and zeros match

                # For poles, just look at eigenvalues of A
                np.testing.assert_array_almost_equal(
                    np.sort(eigvals(s.A)), np.sort(eigvals(sr.A)))

                # For zeros, need to extract SISO systems
                for i in range(m):
                    for j in range(p):
                        # Extract SISO dynamixs from input i to output j
                        s1 = ss(s.A, s.B[:,i], s.C[j,:], s.D[j,i])
                        s2 = ss(sr.A, sr.B[:,i], sr.C[j,:], sr.D[j,i])

                        # Check that the zeros match
                        # Note: sorting doesn't work => have to do the hard way
                        z1 = zeros(s1)
                        z2 = zeros(s2)

                        # Start by making sure we have the same # of zeros
                        assert len(z1) == len(z2)

                        # Make sure all zeros in s1 are in s2
                        for z in z1:
                            # Find the closest zero TODO: find proper bounds
                            assert min(abs(z2 - z)) <= 1e-7

                        # Make sure all zeros in s2 are in s1
                        for z in z2:
                            # Find the closest zero
                            assert min(abs(z1 - z)) <= 1e-7

        # Make sure that the number of systems reduced is as expected
        # (Need to update this number if you change the seed at top of file)
        assert nreductions == 2

    def testMinrealSS(self):
        """Test a minreal model reduction"""
        #A = [-2, 0.5, 0; 0.5, -0.3, 0; 0, 0, -0.1]
        A = [[-2, 0.5, 0], [0.5, -0.3, 0], [0, 0, -0.1]]
        #B = [0.3, -1.3; 0.1, 0; 1, 0]
        B = [[0.3, -1.3], [0.1, 0.], [1.0, 0.0]]
        #C = [0, 0.1, 0; -0.3, -0.2, 0]
        C = [[0., 0.1, 0.0], [-0.3, -0.2, 0.0]]
        #D = [0 -0.8; -0.3 0]
        D = [[0., -0.8], [-0.3, 0.]]
        # sys = ss(A, B, C, D)

        sys = StateSpace(A, B, C, D)
        sysr = sys.minreal()
        assert sysr.nstates == 2
        assert sysr.ninputs == sys.ninputs
        assert sysr.noutputs == sys.noutputs
        np.testing.assert_array_almost_equal(
            eigvals(sysr.A), [-2.136154, -0.1638459])

    def testMinrealtf(self):
        """Try the minreal function, and also test easy entry by creation
        of a Laplace variable s"""
        s = TransferFunction([1, 0], [1])
        h = (s+1)*(s+2.00000000001)/(s+2)/(s**2+s+1)
        hm = h.minreal()
        hr = (s+1)/(s**2+s+1)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(hm.den[0][0], hr.den[0][0])

