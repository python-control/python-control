import unittest
import numpy as np
import control
import control.robust
from control.exception import slycot_check


class TestHinf(unittest.TestCase):
    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def testHinfsyn(self):
        "Test hinfsyn"
        p = control.ss(-1, [1, 1], [[1], [1]], [[0, 1], [1, 0]])
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


if __name__ == "__main__":
    unittest.main()
