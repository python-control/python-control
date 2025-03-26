# -*- coding: utf-8 -*-
"""Test Pade approx

Primitive; ideally test to numerical limits
"""

import numpy as np
import pytest

from control.delay import pade

from control.delaylti import delay
from control.xferfcn import tf


class TestPade:
    """Test Pade approx

    Reference data from Miklos Vajta's paper "Some remarks on
    Padé-approximations", Table 1, with corrections.  The
    corrections are to highest power coeff in numerator for
    (ddeg,ndeg)=(4,3) and (5,4); use Eq (12) in the paper to verify
    """
    # all for T = 1
    ref = [
        # dendeg   numdeg   den        num
        ( 1,       1,       [1,2],     [-1,2]),
        ( 1,       0,       [1,1],     [1]),
        ( 2,       2,       [1,6,12],  [1,-6,12]),
        ( 2,       1,       [1,4,6],   [-2,6]),
        ( 3,       3,       [1,12,60,120], [-1,12,-60,120]),
        ( 3,       2,       [1,9,36,60], [3,-24,60]),
        ( 4,       4,       [1,20,180,840,1680], [1,-20,180,-840,1680]),
        ( 4,       3,       [1,16,120,480,840], [-4,60,-360,840]),
        ( 5,       5,       [1,30,420,3360,15120,30240], [-1,30,-420,3360,-15120,30240]),
        ( 5,       4,       [1,25,300,2100,8400,15120,], [5,-120,1260,-6720,15120]),
    ]

    @pytest.mark.parametrize("dendeg, numdeg, refden, refnum", ref)
    def testRefs(self, dendeg, numdeg, refden, refnum):
        "test reference cases for T=1"
        T = 1
        num, den = pade(T, dendeg, numdeg)
        np.testing.assert_array_almost_equal_nulp(
            np.array(refden), den, nulp=2)
        np.testing.assert_array_almost_equal_nulp(
                np.array(refnum), num, nulp=2)

    @pytest.mark.parametrize("dendeg, numdeg, baseden, basenum", ref)
    @pytest.mark.parametrize("T", [1/53, 21.95])
    def testTvalues(self, T, dendeg, numdeg, baseden, basenum):
        "test reference cases for T!=1"
        refden = T**np.arange(dendeg, -1, -1)*baseden
        refnum = T**np.arange(numdeg, -1, -1)*basenum
        refnum /= refden[0]
        refden /= refden[0]
        num, den = pade(T, dendeg, numdeg)
        np.testing.assert_array_almost_equal_nulp(refden, den, nulp=4)
        np.testing.assert_array_almost_equal_nulp(refnum, num, nulp=4)

    def testErrors(self):
        "ValueError raised for invalid arguments"
        with pytest.raises(ValueError):
            pade(-1, 1)  # T<0
        with pytest.raises(ValueError):
            pade(1, -1)  # dendeg < 0
        with pytest.raises(ValueError):
            pade(1, 2, -3)  # numdeg < 0
        with pytest.raises(ValueError):
            pade(1, 2, 3)  # numdeg > dendeg

    def testNumdeg(self):
        "numdeg argument follows docs"
        # trivialish - interface check, not math check
        T = 1
        dendeg = 5
        ref = [pade(T,dendeg,numdeg)
               for numdeg in range(0,dendeg+1)]
        testneg = [pade(T,dendeg,numdeg)
                   for numdeg in range(-dendeg,0)]
        assert ref[:-1] == testneg
        assert ref[-1] == pade(T,dendeg,dendeg)
        assert ref[-1] == pade(T,dendeg,None)
        assert ref[-1] == pade(T,dendeg)

    def testT0(self):
        "T=0 always returns [1],[1]"
        T = 0
        refnum = [1.0]
        refden = [1.0]
        for dendeg in range(1, 6):
            for numdeg in range(0, dendeg+1):
                num, den = pade(T, dendeg, numdeg)
                np.testing.assert_array_almost_equal_nulp(
                    np.array(refnum), np.array(num))
                np.testing.assert_array_almost_equal_nulp(
                    np.array(refden), np.array(den))


class TestDelayLTI:
    # comparison with julia controlSystems package
    def test_pure_delay(self):
        pure_delay = delay(1)
        A = []
        B = np.zeros((0,2))
        C = []
        D = [[0., 1.], [1., 0.]]

        assert(np.all(pure_delay.P.A == A))
        assert(np.all(pure_delay.P.B == B))
        assert(np.all(pure_delay.P.C == C))
        assert(np.all(pure_delay.P.D == D))
        

    def test_delay(self):
        tf_test = tf([1], [1,1]) * delay(1)
        A = [[-1]]
        B = [[0,1]]
        C = [[1],[0]]
        D = [[0,0], [1,0]]

        assert(np.all(tf_test.P.A == A))
        assert(np.all(tf_test.P.B == B))
        assert(np.all(tf_test.P.C == C))
        assert(np.all(tf_test.P.D == D))

    
    def test_forced_response(self):
        from control.delaylti import DelayLTISystem, tf2dlti
        from control.xferfcn import tf
        from control.timeresp import forced_response
        import matplotlib.pyplot as plt

        tf1 = tf([1], [1.5,1]) * delay(1)

        timepts = np.linspace(0, 10, 100)
        input1 = np.zeros(100)
        input1[31:] = 1
        resp = forced_response(tf1, timepts=timepts, inputs=input1)
        t, y = resp.t, resp.y[0] 

        #plt.figure()
        #plt.plot(t, input1)
        #plt.plot(t, y)
        #plt.show()


    def test_exp(self):
        from control.delaylti import exp
        from control.xferfcn import tf

        s = tf([1,0], [1])
        exp_delay = exp(-2*s)
        print(exp_delay)


        


        