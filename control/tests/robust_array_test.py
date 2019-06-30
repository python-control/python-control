import unittest
import numpy as np
import control
import control.robust
from control.exception import slycot_check

class TestHinf(unittest.TestCase):
    def setUp(self):
        # Use array instead of matrix (and save old value to restore at end)
        control.use_numpy_matrix(False)
        
    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testHinfsyn(self):
        """Test hinfsyn"""
        p = control.ss(-1, [[1, 1]], [[1], [1]], [[0, 1], [1, 0]])
        k, cl, gam, rcond = control.robust.hinfsyn(p, 1, 1)
        # from Octave, which also uses SB10AD:
        #   a= -1; b1= 1; b2= 1; c1= 1; c2= 1; d11= 0; d12= 1; d21= 1; d22= 0;
        #   g = ss(a,[b1,b2],[c1;c2],[d11,d12;d21,d22]);
        #   [k,cl] = hinfsyn(g,1,1);
        np.testing.assert_array_almost_equal(k.A, [[-3]])
        np.testing.assert_array_almost_equal(k.B, [[1]])
        np.testing.assert_array_almost_equal(k.C, [[-1]])
        np.testing.assert_array_almost_equal(k.D, [[0]])
        np.testing.assert_array_almost_equal(cl.A, [[-1, -1], [1, -3]])
        np.testing.assert_array_almost_equal(cl.B, [[1], [1]])
        np.testing.assert_array_almost_equal(cl.C, [[1, -1]])
        np.testing.assert_array_almost_equal(cl.D, [[0]])

    # TODO: add more interesting examples

    def tearDown(self):
        control.config.reset_defaults()


class TestH2(unittest.TestCase):
    def setUp(self):
        # Use array instead of matrix (and save old value to restore at end)
        control.use_numpy_matrix(False)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testH2syn(self):
        """Test h2syn"""
        p = control.ss(-1, [[1, 1]], [[1], [1]], [[0, 1], [1, 0]])
        k = control.robust.h2syn(p, 1, 1)
        # from Octave, which also uses SB10HD for H-2 synthesis:
        #   a= -1; b1= 1; b2= 1; c1= 1; c2= 1; d11= 0; d12= 1; d21= 1; d22= 0;
        #   g = ss(a,[b1,b2],[c1;c2],[d11,d12;d21,d22]);
        #   k = h2syn(g,1,1);
        # the solution is the same as for the hinfsyn test
        np.testing.assert_array_almost_equal(k.A, [[-3]])
        np.testing.assert_array_almost_equal(k.B, [[1]])
        np.testing.assert_array_almost_equal(k.C, [[-1]])
        np.testing.assert_array_almost_equal(k.D, [[0]])

    def tearDown(self):
        control.config.reset_defaults()


class TestAugw(unittest.TestCase):
    """Test control.robust.augw"""
    def setUp(self):
        # Use array instead of matrix (and save old value to restore at end)
        control.use_numpy_matrix(False)

    # tolerance for system equality
    TOL = 1e-8

    def siso_almost_equal(self, g, h):
        """siso_almost_equal(g,h) -> None
        Raises AssertionError if g and h, two SISO LTI objects, are not almost equal"""
        from control import tf, minreal
        gmh = tf(minreal(g - h, verbose=False))
        if not (gmh.num[0][0] < self.TOL).all():
            maxnum = max(abs(gmh.num[0][0]))
            raise AssertionError(
                'systems not approx equal; max num. coeff is {}\nsys 1:\n{}\nsys 2:\n{}'.format(
                    maxnum, g, h))

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testSisoW1(self):
        """SISO plant with S weighting"""
        from control import augw, ss
        g = ss([-1.], [1.], [1.], [1.])
        w1 = ss([-2], [2.], [1.], [2.])
        p = augw(g, w1)
        self.assertEqual(2, p.outputs)
        self.assertEqual(2, p.inputs)
        # w->z1 should be w1
        self.siso_almost_equal(w1, p[0, 0])
        # w->v should be 1
        self.siso_almost_equal(ss([], [], [], [1]), p[1, 0])
        # u->z1 should be -w1*g
        self.siso_almost_equal(-w1 * g, p[0, 1])
        # u->v should be -g
        self.siso_almost_equal(-g, p[1, 1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testSisoW2(self):
        """SISO plant with KS weighting"""
        from control import augw, ss
        g = ss([-1.], [1.], [1.], [1.])
        w2 = ss([-2], [1.], [1.], [2.])
        p = augw(g, w2=w2)
        self.assertEqual(2, p.outputs)
        self.assertEqual(2, p.inputs)
        # w->z2 should be 0
        self.siso_almost_equal(ss([], [], [], 0), p[0, 0])
        # w->v should be 1
        self.siso_almost_equal(ss([], [], [], [1]), p[1, 0])
        # u->z2 should be w2
        self.siso_almost_equal(w2, p[0, 1])
        # u->v should be -g
        self.siso_almost_equal(-g, p[1, 1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testSisoW3(self):
        """SISO plant with T weighting"""
        from control import augw, ss
        g = ss([-1.], [1.], [1.], [1.])
        w3 = ss([-2], [1.], [1.], [2.])
        p = augw(g, w3=w3)
        self.assertEqual(2, p.outputs)
        self.assertEqual(2, p.inputs)
        # w->z3 should be 0
        self.siso_almost_equal(ss([], [], [], 0), p[0, 0])
        # w->v should be 1
        self.siso_almost_equal(ss([], [], [], [1]), p[1, 0])
        # u->z3 should be w3*g
        self.siso_almost_equal(w3 * g, p[0, 1])
        # u->v should be -g
        self.siso_almost_equal(-g, p[1, 1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testSisoW123(self):
        """SISO plant with all weights"""
        from control import augw, ss
        g = ss([-1.], [1.], [1.], [1.])
        w1 = ss([-2.], [2.], [1.], [2.])
        w2 = ss([-3.], [3.], [1.], [3.])
        w3 = ss([-4.], [4.], [1.], [4.])
        p = augw(g, w1, w2, w3)
        self.assertEqual(4, p.outputs)
        self.assertEqual(2, p.inputs)
        # w->z1 should be w1
        self.siso_almost_equal(w1, p[0, 0])
        # w->z2 should be 0
        self.siso_almost_equal(0, p[1, 0])
        # w->z3 should be 0
        self.siso_almost_equal(0, p[2, 0])
        # w->v should be 1
        self.siso_almost_equal(ss([], [], [], [1]), p[3, 0])
        # u->z1 should be -w1*g
        self.siso_almost_equal(-w1 * g, p[0, 1])
        # u->z2 should be w2
        self.siso_almost_equal(w2, p[1, 1])
        # u->z3 should be w3*g
        self.siso_almost_equal(w3 * g, p[2, 1])
        # u->v should be -g
        self.siso_almost_equal(-g, p[3, 1])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMimoW1(self):
        """MIMO plant with S weighting"""
        from control import augw, ss
        g = ss([[-1., -2], [-3, -4]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]])
        w1 = ss([-2], [2.], [1.], [2.])
        p = augw(g, w1)
        self.assertEqual(4, p.outputs)
        self.assertEqual(4, p.inputs)
        # w->z1 should be diag(w1,w1)
        self.siso_almost_equal(w1, p[0, 0])
        self.siso_almost_equal(0, p[0, 1])
        self.siso_almost_equal(0, p[1, 0])
        self.siso_almost_equal(w1, p[1, 1])
        # w->v should be I
        self.siso_almost_equal(1, p[2, 0])
        self.siso_almost_equal(0, p[2, 1])
        self.siso_almost_equal(0, p[3, 0])
        self.siso_almost_equal(1, p[3, 1])
        # u->z1 should be -w1*g
        self.siso_almost_equal(-w1 * g[0, 0], p[0, 2])
        self.siso_almost_equal(-w1 * g[0, 1], p[0, 3])
        self.siso_almost_equal(-w1 * g[1, 0], p[1, 2])
        self.siso_almost_equal(-w1 * g[1, 1], p[1, 3])
        # # u->v should be -g
        self.siso_almost_equal(-g[0, 0], p[2, 2])
        self.siso_almost_equal(-g[0, 1], p[2, 3])
        self.siso_almost_equal(-g[1, 0], p[3, 2])
        self.siso_almost_equal(-g[1, 1], p[3, 3])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMimoW2(self):
        """MIMO plant with KS weighting"""
        from control import augw, ss
        g = ss([[-1., -2], [-3, -4]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]])
        w2 = ss([-2], [2.], [1.], [2.])
        p = augw(g, w2=w2)
        self.assertEqual(4, p.outputs)
        self.assertEqual(4, p.inputs)
        # w->z2 should be 0
        self.siso_almost_equal(0, p[0, 0])
        self.siso_almost_equal(0, p[0, 1])
        self.siso_almost_equal(0, p[1, 0])
        self.siso_almost_equal(0, p[1, 1])
        # w->v should be I
        self.siso_almost_equal(1, p[2, 0])
        self.siso_almost_equal(0, p[2, 1])
        self.siso_almost_equal(0, p[3, 0])
        self.siso_almost_equal(1, p[3, 1])
        # u->z2 should be w2
        self.siso_almost_equal(w2, p[0, 2])
        self.siso_almost_equal(0, p[0, 3])
        self.siso_almost_equal(0, p[1, 2])
        self.siso_almost_equal(w2, p[1, 3])
        # # u->v should be -g
        self.siso_almost_equal(-g[0, 0], p[2, 2])
        self.siso_almost_equal(-g[0, 1], p[2, 3])
        self.siso_almost_equal(-g[1, 0], p[3, 2])
        self.siso_almost_equal(-g[1, 1], p[3, 3])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMimoW3(self):
        """MIMO plant with T weighting"""
        from control import augw, ss
        g = ss([[-1., -2], [-3, -4]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]])
        w3 = ss([-2], [2.], [1.], [2.])
        p = augw(g, w3=w3)
        self.assertEqual(4, p.outputs)
        self.assertEqual(4, p.inputs)
        # w->z3 should be 0
        self.siso_almost_equal(0, p[0, 0])
        self.siso_almost_equal(0, p[0, 1])
        self.siso_almost_equal(0, p[1, 0])
        self.siso_almost_equal(0, p[1, 1])
        # w->v should be I
        self.siso_almost_equal(1, p[2, 0])
        self.siso_almost_equal(0, p[2, 1])
        self.siso_almost_equal(0, p[3, 0])
        self.siso_almost_equal(1, p[3, 1])
        # u->z3 should be w3*g
        self.siso_almost_equal(w3 * g[0, 0], p[0, 2])
        self.siso_almost_equal(w3 * g[0, 1], p[0, 3])
        self.siso_almost_equal(w3 * g[1, 0], p[1, 2])
        self.siso_almost_equal(w3 * g[1, 1], p[1, 3])
        # # u->v should be -g
        self.siso_almost_equal(-g[0, 0], p[2, 2])
        self.siso_almost_equal(-g[0, 1], p[2, 3])
        self.siso_almost_equal(-g[1, 0], p[3, 2])
        self.siso_almost_equal(-g[1, 1], p[3, 3])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testMimoW123(self):
        """MIMO plant with all weights"""
        from control import augw, ss, append
        g = ss([[-1., -2], [-3, -4]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]],
               [[1., 0.], [0., 1.]])
        # this should be expaned to w1*I
        w1 = ss([-2.], [2.], [1.], [2.])
        # diagonal weighting
        w2 = append(ss([-3.], [3.], [1.], [3.]), ss([-4.], [4.], [1.], [4.]))
        # full weighting
        w3 = ss([[-4., -5], [-6, -7]],
                [[2., 3.], [5., 7.]],
                [[11., 13.], [17., 19.]],
                [[23., 29.], [31., 37.]])
        p = augw(g, w1, w2, w3)
        self.assertEqual(8, p.outputs)
        self.assertEqual(4, p.inputs)
        # w->z1 should be w1
        self.siso_almost_equal(w1, p[0, 0])
        self.siso_almost_equal(0, p[0, 1])
        self.siso_almost_equal(0, p[1, 0])
        self.siso_almost_equal(w1, p[1, 1])
        # w->z2 should be 0
        self.siso_almost_equal(0, p[2, 0])
        self.siso_almost_equal(0, p[2, 1])
        self.siso_almost_equal(0, p[3, 0])
        self.siso_almost_equal(0, p[3, 1])
        # w->z3 should be 0
        self.siso_almost_equal(0, p[4, 0])
        self.siso_almost_equal(0, p[4, 1])
        self.siso_almost_equal(0, p[5, 0])
        self.siso_almost_equal(0, p[5, 1])
        # w->v should be I
        self.siso_almost_equal(1, p[6, 0])
        self.siso_almost_equal(0, p[6, 1])
        self.siso_almost_equal(0, p[7, 0])
        self.siso_almost_equal(1, p[7, 1])

        # u->z1 should be -w1*g
        self.siso_almost_equal(-w1 * g[0, 0], p[0, 2])
        self.siso_almost_equal(-w1 * g[0, 1], p[0, 3])
        self.siso_almost_equal(-w1 * g[1, 0], p[1, 2])
        self.siso_almost_equal(-w1 * g[1, 1], p[1, 3])
        # u->z2 should be w2
        self.siso_almost_equal(w2[0, 0], p[2, 2])
        self.siso_almost_equal(w2[0, 1], p[2, 3])
        self.siso_almost_equal(w2[1, 0], p[3, 2])
        self.siso_almost_equal(w2[1, 1], p[3, 3])
        # u->z3 should be w3*g
        w3g = w3 * g;
        self.siso_almost_equal(w3g[0, 0], p[4, 2])
        self.siso_almost_equal(w3g[0, 1], p[4, 3])
        self.siso_almost_equal(w3g[1, 0], p[5, 2])
        self.siso_almost_equal(w3g[1, 1], p[5, 3])
        # u->v should be -g
        self.siso_almost_equal(-g[0, 0], p[6, 2])
        self.siso_almost_equal(-g[0, 1], p[6, 3])
        self.siso_almost_equal(-g[1, 0], p[7, 2])
        self.siso_almost_equal(-g[1, 1], p[7, 3])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testErrors(self):
        """Error cases handled"""
        from control import augw, ss
        # no weights
        g1by1 = ss(-1, 1, 1, 0)
        g2by2 = ss(-np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
        self.assertRaises(ValueError, augw, g1by1)
        # mismatched size of weight and plant
        self.assertRaises(ValueError, augw, g1by1, w1=g2by2)
        self.assertRaises(ValueError, augw, g1by1, w2=g2by2)
        self.assertRaises(ValueError, augw, g1by1, w3=g2by2)

    def tearDown(self):
        control.config.reset_defaults()


class TestMixsyn(unittest.TestCase):
    """Test control.robust.mixsyn"""
    def setUp(self):
        # Use array instead of matrix (and save old value to restore at end)
        control.use_numpy_matrix(False)
        
    # it's a relatively simple wrapper; compare results with augw, hinfsyn
    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testSiso(self):
        """mixsyn with SISO system"""
        from control import tf, augw, hinfsyn, mixsyn
        from control import ss
        # Skogestad+Postlethwaite, Multivariable Feedback Control, 1st Ed., Example 2.11
        s = tf([1, 0], 1)
        # plant
        g = 200 / (10 * s + 1) / (0.05 * s + 1) ** 2
        # sensitivity weighting
        M = 1.5
        wb = 10
        A = 1e-4
        w1 = (s / M + wb) / (s + wb * A)
        # KS weighting
        w2 = tf(1, 1)

        p = augw(g, w1, w2)
        kref, clref, gam, rcond = hinfsyn(p, 1, 1)
        ktest, cltest, info = mixsyn(g, w1, w2)
        # check similar to S+P's example
        np.testing.assert_allclose(gam, 1.37, atol=1e-2)

        # mixsyn is a convenience wrapper around augw and hinfsyn, so
        # results will be exactly the same.  Given than, use the lazy
        # but fragile testing option.
        np.testing.assert_allclose(ktest.A, kref.A)
        np.testing.assert_allclose(ktest.B, kref.B)
        np.testing.assert_allclose(ktest.C, kref.C)
        np.testing.assert_allclose(ktest.D, kref.D)

        np.testing.assert_allclose(cltest.A, clref.A)
        np.testing.assert_allclose(cltest.B, clref.B)
        np.testing.assert_allclose(cltest.C, clref.C)
        np.testing.assert_allclose(cltest.D, clref.D)

        np.testing.assert_allclose(gam, info[0])

        np.testing.assert_allclose(rcond, info[1])

    def tearDown(self):
        control.config.reset_defaults()

if __name__ == "__main__":
    unittest.main()
