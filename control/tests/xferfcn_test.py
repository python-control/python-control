#!/usr/bin/env python
#
# xferfcn_test.py - test TransferFunction class
# RMM, 30 Mar 2011 (based on TestXferFcn from v0.4a)

import unittest
import sys as pysys
import numpy as np
from control.statesp import StateSpace, _convertToStateSpace, rss
from control.xferfcn import TransferFunction, _convert_to_transfer_function, \
    ss2tf
from control.lti import evalfr
from control.exception import slycot_check
from control.lti import isctime, isdtime
from control.dtime import sample_system


class TestXferFcn(unittest.TestCase):
    """These are tests for functionality and correct reporting of the transfer
    function class.  Throughout these tests, we will give different input
    formats to the xTranferFunction constructor, to try to break it.  These
    tests have been verified in MATLAB."""

    # Tests for raising exceptions.

    def test_constructor_bad_input_type(self):
        """Give the constructor invalid input types."""

        # MIMO requires lists of lists of vectors (not lists of vectors)
        self.assertRaises(
            TypeError,
            TransferFunction, [[0., 1.], [2., 3.]], [[5., 2.], [3., 0.]])
        TransferFunction([[ [0., 1.], [2., 3.] ]], [[ [5., 2.], [3., 0.] ]])

        # Single argument of the wrong type
        self.assertRaises(TypeError, TransferFunction, [1])

        # Too many arguments
        self.assertRaises(ValueError, TransferFunction, 1, 2, 3, 4)

        # Different numbers of elements in numerator rows
        self.assertRaises(
            ValueError,
            TransferFunction, [ [[0, 1], [2, 3]],
                                [[4, 5]] ],
                              [ [[6, 7], [4, 5]],
                                [[2, 3], [0, 1]] ])
        self.assertRaises(
            ValueError,
            TransferFunction, [ [[0, 1], [2, 3]],
                                [[4, 5], [6, 7]] ],
                              [ [[6, 7], [4, 5]],
                                [[2, 3]] ])
        TransferFunction(       # This version is OK
            [ [[0, 1], [2, 3]], [[4, 5], [6, 7]] ],
            [ [[6, 7], [4, 5]], [[2, 3], [0, 1]] ])

    def test_constructor_inconsistent_dimension(self):
        """Give constructor numerators, denominators of different sizes."""

        self.assertRaises(ValueError, TransferFunction,
            [[[1.]]], [[[1.], [2., 3.]]])
        self.assertRaises(ValueError, TransferFunction,
            [[[1.]]], [[[1.]], [[2., 3.]]])
        self.assertRaises(ValueError, TransferFunction,
            [[[1.]]], [[[1.], [1., 2.]], [[5., 2.], [2., 3.]]])

    def test_constructor_inconsistent_columns(self):
        """Give the constructor inputs that do not have the same number of
        columns in each row."""

        self.assertRaises(ValueError, TransferFunction,
                          1., [[[1.]], [[2.], [3.]]])
        self.assertRaises(ValueError, TransferFunction,
                          [[[1.]], [[2.], [3.]]], 1.)

    def test_constructor_zero_denominator(self):
        """Give the constructor a transfer function with a zero denominator."""

        self.assertRaises(ValueError, TransferFunction, 1., 0.)
        self.assertRaises(ValueError, TransferFunction,
                          [[[1.], [2., 3.]], [[-1., 4.], [3., 2.]]],
                          [[[1., 0.], [0.]], [[0., 0.], [2.]]])

    def test_add_inconsistent_dimension(self):
        """Add two transfer function matrices of different sizes."""

        sys1 = TransferFunction([[[1., 2.]]], [[[4., 5.]]])
        sys2 = TransferFunction([[[4., 3.]], [[1., 2.]]],
                                [[[1., 6.]], [[2., 4.]]])
        self.assertRaises(ValueError, sys1.__add__, sys2)
        self.assertRaises(ValueError, sys1.__sub__, sys2)
        self.assertRaises(ValueError, sys1.__radd__, sys2)
        self.assertRaises(ValueError, sys1.__rsub__, sys2)

    def test_mul_inconsistent_dimension(self):
        """Multiply two transfer function matrices of incompatible sizes."""

        sys1 = TransferFunction([[[1., 2.], [4., 5.]], [[2., 5.], [4., 3.]]],
                                [[[6., 2.], [4., 1.]], [[6., 7.], [2., 4.]]])
        sys2 = TransferFunction([[[1.]], [[2.]], [[3.]]],
                                [[[4.]], [[5.]], [[6.]]])
        self.assertRaises(ValueError, sys1.__mul__, sys2)
        self.assertRaises(ValueError, sys2.__mul__, sys1)
        self.assertRaises(ValueError, sys1.__rmul__, sys2)
        self.assertRaises(ValueError, sys2.__rmul__, sys1)

    # Tests for TransferFunction._truncatecoeff

    def test_truncate_coefficients_non_null_numerator(self):
        """Remove extraneous zeros in polynomial representations."""

        sys1 = TransferFunction([0., 0., 1., 2.], [[[0., 0., 0., 3., 2., 1.]]])

        np.testing.assert_array_equal(sys1.num, [[[1., 2.]]])
        np.testing.assert_array_equal(sys1.den, [[[3., 2., 1.]]])

    def test_truncate_coefficients_null_numerator(self):
        """Remove extraneous zeros in polynomial representations."""

        sys1 = TransferFunction([0., 0., 0.], 1.)

        np.testing.assert_array_equal(sys1.num, [[[0.]]])
        np.testing.assert_array_equal(sys1.den, [[[1.]]])

    # Tests for TransferFunction.__neg__

    def test_reverse_sign_scalar(self):
        """Negate a direct feedthrough system."""

        sys1 = TransferFunction(2., np.array([-3.]))
        sys2 = - sys1

        np.testing.assert_array_equal(sys2.num, [[[-2.]]])
        np.testing.assert_array_equal(sys2.den, [[[-3.]]])

    def test_reverse_sign_siso(self):
        """Negate a SISO system."""

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1.])
        sys2 = - sys1

        np.testing.assert_array_equal(sys2.num, [[[-1., -3., -5.]]])
        np.testing.assert_array_equal(sys2.den, [[[1., 6., 2., -1.]]])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_reverse_sign_mimo(self):
        """Negate a MIMO system."""

        num1 = [[[1., 2.], [0., 3.], [2., -1.]],
                [[1.], [4., 0.], [1., -4., 3.]]]
        num3 = [[[-1., -2.], [0., -3.], [-2., 1.]],
                [[-1.], [-4., 0.], [-1., 4., -3.]]]
        den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
                [[3., 0., .0], [2., -1., -1.], [1.]]]

        sys1 = TransferFunction(num1, den1)
        sys2 = - sys1
        sys3 = TransferFunction(num3, den1)

        for i in range(sys3.outputs):
            for j in range(sys3.inputs):
                np.testing.assert_array_equal(sys2.num[i][j], sys3.num[i][j])
                np.testing.assert_array_equal(sys2.den[i][j], sys3.den[i][j])

    # Tests for TransferFunction.__add__

    def test_add_scalar(self):
        """Add two direct feedthrough systems."""

        sys1 = TransferFunction(1., [[[1.]]])
        sys2 = TransferFunction(np.array([2.]), [1.])
        sys3 = sys1 + sys2

        np.testing.assert_array_equal(sys3.num, 3.)
        np.testing.assert_array_equal(sys3.den, 1.)

    def test_add_siso(self):
        """Add two SISO systems."""

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1])
        sys2 = TransferFunction([[np.array([-1., 3.])]], [[[1., 0., -1.]]])
        sys3 = sys1 + sys2

        # If sys3.num is [[[0., 20., 4., -8.]]], then this is wrong!
        np.testing.assert_array_equal(sys3.num, [[[20., 4., -8]]])
        np.testing.assert_array_equal(sys3.den, [[[1., 6., 1., -7., -2., 1.]]])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_add_mimo(self):
        """Add two MIMO systems."""

        num1 = [[[1., 2.], [0., 3.], [2., -1.]],
                [[1.], [4., 0.], [1., -4., 3.]]]
        den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
                [[3., 0., .0], [2., -1., -1.], [1.]]]
        num2 = [[[0., 0., -1], [2.], [-1., -1.]],
                [[1., 2.], [-1., -2.], [4.]]]
        den2 = [[[-1.], [1., 2., 3.], [-1., -1.]],
                [[-4., -3., 2.], [0., 1.], [1., 0.]]]
        num3 = [[[3., -3., -6], [5., 6., 9.], [-4., -2., 2]],
                [[3., 2., -3., 2], [-2., -3., 7., 2.], [1., -4., 3., 4]]]
        den3 = [[[3., -2., -4.], [1., 2., 3., 0., 0.], [-2., -1., 1.]],
                [[-12., -9., 6., 0., 0.], [2., -1., -1.], [1., 0.]]]

        sys1 = TransferFunction(num1, den1)
        sys2 = TransferFunction(num2, den2)
        sys3 = sys1 + sys2

        for i in range(sys3.outputs):
            for j in range(sys3.inputs):
                np.testing.assert_array_equal(sys3.num[i][j], num3[i][j])
                np.testing.assert_array_equal(sys3.den[i][j], den3[i][j])

    # Tests for TransferFunction.__sub__

    def test_subtract_scalar(self):
        """Subtract two direct feedthrough systems."""

        sys1 = TransferFunction(1., [[[1.]]])
        sys2 = TransferFunction(np.array([2.]), [1.])
        sys3 = sys1 - sys2

        np.testing.assert_array_equal(sys3.num, -1.)
        np.testing.assert_array_equal(sys3.den, 1.)

    def test_subtract_siso(self):
        """Subtract two SISO systems."""

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1])
        sys2 = TransferFunction([[np.array([-1., 3.])]], [[[1., 0., -1.]]])
        sys3 = sys1 - sys2
        sys4 = sys2 - sys1

        np.testing.assert_array_equal(sys3.num, [[[2., 6., -12., -10., -2.]]])
        np.testing.assert_array_equal(sys3.den, [[[1., 6., 1., -7., -2., 1.]]])
        np.testing.assert_array_equal(sys4.num, [[[-2., -6., 12., 10., 2.]]])
        np.testing.assert_array_equal(sys4.den, [[[1., 6., 1., -7., -2., 1.]]])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_subtract_mimo(self):
        """Subtract two MIMO systems."""

        num1 = [[[1., 2.], [0., 3.], [2., -1.]],
                [[1.], [4., 0.], [1., -4., 3.]]]
        den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
                [[3., 0., .0], [2., -1., -1.], [1.]]]
        num2 = [[[0., 0., -1], [2.], [-1., -1.]],
                [[1., 2.], [-1., -2.], [4.]]]
        den2 = [[[-1.], [1., 2., 3.], [-1., -1.]],
                [[-4., -3., 2.], [0., 1.], [1., 0.]]]
        num3 = [[[-3., 1., 2.], [1., 6., 9.], [0.]],
                [[-3., -10., -3., 2], [2., 3., 1., -2], [1., -4., 3., -4]]]
        den3 = [[[3., -2., -4], [1., 2., 3., 0., 0.], [1]],
                [[-12., -9., 6., 0., 0.], [2., -1., -1], [1., 0.]]]

        sys1 = TransferFunction(num1, den1)
        sys2 = TransferFunction(num2, den2)
        sys3 = sys1 - sys2

        for i in range(sys3.outputs):
            for j in range(sys3.inputs):
                np.testing.assert_array_equal(sys3.num[i][j], num3[i][j])
                np.testing.assert_array_equal(sys3.den[i][j], den3[i][j])

    # Tests for TransferFunction.__mul__

    def test_multiply_scalar(self):
        """Multiply two direct feedthrough systems."""

        sys1 = TransferFunction(2., [1.])
        sys2 = TransferFunction(1., 4.)
        sys3 = sys1 * sys2
        sys4 = sys1 * sys2

        np.testing.assert_array_equal(sys3.num, [[[2.]]])
        np.testing.assert_array_equal(sys3.den, [[[4.]]])
        np.testing.assert_array_equal(sys3.num, sys4.num)
        np.testing.assert_array_equal(sys3.den, sys4.den)

    def test_multiply_siso(self):
        """Multiply two SISO systems."""

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1])
        sys2 = TransferFunction([[[-1., 3.]]], [[[1., 0., -1.]]])
        sys3 = sys1 * sys2
        sys4 = sys2 * sys1

        np.testing.assert_array_equal(sys3.num, [[[-1., 0., 4., 15.]]])
        np.testing.assert_array_equal(sys3.den, [[[1., 6., 1., -7., -2., 1.]]])
        np.testing.assert_array_equal(sys3.num, sys4.num)
        np.testing.assert_array_equal(sys3.den, sys4.den)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_multiply_mimo(self):
        """Multiply two MIMO systems."""

        num1 = [[[1., 2.], [0., 3.], [2., -1.]],
                [[1.], [4., 0.], [1., -4., 3.]]]
        den1 = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
                [[3., 0., .0], [2., -1., -1.], [1.]]]
        num2 = [[[0., 1., 2.]],
                [[1., -5.]],
                [[-2., 1., 4.]]]
        den2 = [[[1., 0., 0., 0.]],
                [[-2., 1., 3.]],
                [[4., -1., -1., 0.]]]
        num3 = [[[-24., 52., -14., 245., -490., -115., 467., -95., -56., 12.,
                  0., 0., 0.]],
                [[24., -132., 138., 345., -768., -106., 510., 41., -79., -69.,
                 -23., 17., 6., 0.]]]
        den3 = [[[48., -92., -84., 183., 44., -97., -2., 12., 0., 0., 0., 0.,
                  0., 0.]],
                [[-48., 60., 84., -81., -45., 21., 9., 0., 0., 0., 0., 0., 0.]]]

        sys1 = TransferFunction(num1, den1)
        sys2 = TransferFunction(num2, den2)
        sys3 = sys1 * sys2

        for i in range(sys3.outputs):
            for j in range(sys3.inputs):
                np.testing.assert_array_equal(sys3.num[i][j], num3[i][j])
                np.testing.assert_array_equal(sys3.den[i][j], den3[i][j])

    # Tests for TransferFunction.__div__

    def test_divide_scalar(self):
        """Divide two direct feedthrough systems."""

        sys1 = TransferFunction(np.array([3.]), -4.)
        sys2 = TransferFunction(5., 2.)
        sys3 = sys1 / sys2

        np.testing.assert_array_equal(sys3.num, [[[6.]]])
        np.testing.assert_array_equal(sys3.den, [[[-20.]]])

    def test_divide_siso(self):
        """Divide two SISO systems."""

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1])
        sys2 = TransferFunction([[[-1., 3.]]], [[[1., 0., -1.]]])
        sys3 = sys1 / sys2
        sys4 = sys2 / sys1

        np.testing.assert_array_equal(sys3.num, [[[1., 3., 4., -3., -5.]]])
        np.testing.assert_array_equal(sys3.den, [[[-1., -3., 16., 7., -3.]]])
        np.testing.assert_array_equal(sys4.num, sys3.den)
        np.testing.assert_array_equal(sys4.den, sys3.num)

    def test_div(self):
        # Make sure that sampling times work correctly
        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1], None)
        sys2 = TransferFunction([[[-1., 3.]]], [[[1., 0., -1.]]], True)
        sys3 = sys1 / sys2
        self.assertEqual(sys3.dt, True)

        sys2 = TransferFunction([[[-1., 3.]]], [[[1., 0., -1.]]], 0.5)
        sys3 = sys1 / sys2
        self.assertEqual(sys3.dt, 0.5)

        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1], 0.1)
        self.assertRaises(ValueError, TransferFunction.__truediv__, sys1, sys2)

        sys1 = sample_system(rss(4, 1, 1), 0.5)
        sys3 = TransferFunction.__rtruediv__(sys2, sys1)
        self.assertEqual(sys3.dt, 0.5)

    def test_pow(self):
        sys1 = TransferFunction([1., 3., 5], [1., 6., 2., -1])
        self.assertRaises(ValueError, TransferFunction.__pow__, sys1, 0.5)

    def test_slice(self):
        sys = TransferFunction(
            [ [   [1],    [2],    [3]], [   [3],    [4],    [5]] ],
            [ [[1, 2], [1, 3], [1, 4]], [[1, 4], [1, 5], [1, 6]] ])
        sys1 = sys[1:, 1:]
        self.assertEqual((sys1.inputs, sys1.outputs), (2, 1))

        sys2 = sys[:2, :2]
        self.assertEqual((sys2.inputs, sys2.outputs), (2, 2))

        sys = TransferFunction(
            [ [   [1],    [2],    [3]], [   [3],    [4],    [5]] ],
            [ [[1, 2], [1, 3], [1, 4]], [[1, 4], [1, 5], [1, 6]] ], 0.5)
        sys1 = sys[1:, 1:]
        self.assertEqual((sys1.inputs, sys1.outputs), (2, 1))
        self.assertEqual(sys1.dt, 0.5)

    def test_evalfr_siso(self):
        """Evaluate the frequency response of a SISO system at one frequency."""

        sys = TransferFunction([1., 3., 5], [1., 6., 2., -1])

        np.testing.assert_array_almost_equal(evalfr(sys, 1j),
                                             np.array([[-0.5 - 0.5j]]))
        np.testing.assert_array_almost_equal(
            evalfr(sys, 32j),
            np.array([[0.00281959302585077 - 0.030628473607392j]]))

        # Test call version as well
        np.testing.assert_almost_equal(sys(1.j), -0.5 - 0.5j)
        np.testing.assert_almost_equal(
            sys(32.j), 0.00281959302585077 - 0.030628473607392j)

        # Test internal version (with real argument)
        np.testing.assert_array_almost_equal(
            sys._evalfr(1.), np.array([[-0.5 - 0.5j]]))
        np.testing.assert_array_almost_equal(
            sys._evalfr(32.),
            np.array([[0.00281959302585077 - 0.030628473607392j]]))

    def test_is_static_gain(self):
        numstatic = 1.1
        denstatic = 1.2
        numdynamic = [1,1]
        dendynamic = [2,1]
        numstaticmimo = [[[1.1,], [1.2,]], [[1.2,], [0.8,]]]        
        denstaticmimo = [[[1.9,], [1.2,]], [[1.2,], [0.8,]]]
        numdynamicmimo = [[[1.1, 0.9], [1.2]], [[1.2], [0.8]]]        
        dendynamicmimo = [[[1.1, 0.7], [0.2]], [[1.2], [0.8]]]
        self.assertTrue(TransferFunction(numstatic,denstatic).is_static_gain()) # True
        self.assertTrue(TransferFunction(numstaticmimo,denstaticmimo).is_static_gain()) # True
        
        self.assertFalse(TransferFunction(numstatic,dendynamic).is_static_gain()) # False
        self.assertFalse(TransferFunction(numdynamic,dendynamic).is_static_gain()) # False
        self.assertFalse(TransferFunction(numdynamic,denstatic).is_static_gain()) # False
        self.assertFalse(TransferFunction(numstatic,dendynamic).is_static_gain()) # False
        
        self.assertFalse(TransferFunction(numstaticmimo,dendynamicmimo).is_static_gain()) # False
        self.assertFalse(TransferFunction(numdynamicmimo,denstaticmimo).is_static_gain()) # False
        
    # This test only works in Python 3 due to a conflict with the same
    # warning type in other test modules (frd_test.py).  See
    # https://bugs.python.org/issue4180 for more details
    @unittest.skipIf(pysys.version_info < (3, 0), "test requires Python 3+")
    def test_evalfr_deprecated(self):
        sys = TransferFunction([1., 3., 5], [1., 6., 2., -1])

        # Deprecated version of the call (should generate warning)
        import warnings
        with warnings.catch_warnings():
            # Make warnings generate an exception
            warnings.simplefilter('error')

            # Make sure that we get a pending deprecation warning
            self.assertRaises(PendingDeprecationWarning, sys.evalfr, 1.)

    @unittest.skipIf(pysys.version_info < (3, 0), "test requires Python 3+")
    def test_evalfr_dtime(self):
        sys = TransferFunction([1., 3., 5], [1., 6., 2., -1], 0.1)
        np.testing.assert_array_almost_equal(sys(1j), -0.5 - 0.5j)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_evalfr_mimo(self):
        """Evaluate the frequency response of a MIMO system at one frequency."""

        num = [[[1., 2.], [0., 3.], [2., -1.]],
               [[1.], [4., 0.], [1., -4., 3.]]]
        den = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
               [[3., 0., .0], [2., -1., -1.], [1.]]]
        sys = TransferFunction(num, den)
        resp = [[0.147058823529412 + 0.0882352941176471j, -0.75, 1.],
                [-0.083333333333333, -0.188235294117647 - 0.847058823529412j,
                 -1. - 8.j]]

        np.testing.assert_array_almost_equal(sys._evalfr(2.), resp)

        # Test call version as well
        np.testing.assert_array_almost_equal(sys(2.j), resp)

    def test_freqresp_siso(self):
        """Evaluate the magnitude and phase of a SISO system at
        multiple frequencies."""

        sys = TransferFunction([1., 3., 5], [1., 6., 2., -1])

        truemag = [[[4.63507337473906, 0.707106781186548, 0.0866592803995351]]]
        truephase = [[[-2.89596891081488, -2.35619449019234,
                       -1.32655885133871]]]
        trueomega = [0.1, 1., 10.]

        mag, phase, omega = sys.freqresp(trueomega)

        np.testing.assert_array_almost_equal(mag, truemag)
        np.testing.assert_array_almost_equal(phase, truephase)
        np.testing.assert_array_almost_equal(omega, trueomega)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_freqresp_mimo(self):
        """Evaluate the magnitude and phase of a MIMO system at
        multiple frequencies."""

        num = [[[1., 2.], [0., 3.], [2., -1.]],
               [[1.], [4., 0.], [1., -4., 3.]]]
        den = [[[-3., 2., 4.], [1., 0., 0.], [2., -1.]],
               [[3., 0., .0], [2., -1., -1.], [1.]]]
        sys = TransferFunction(num, den)

        true_omega = [0.1, 1., 10.]
        true_mag = [[[0.49628709, 0.30714755, 0.03347381],
                    [300., 3., 0.03], [1., 1., 1.]],
                    [[33.333333, 0.33333333, 0.00333333],
                     [0.39028569, 1.26491106, 0.19875914],
                    [3.01663720, 4.47213595, 104.92378186]]]
        true_phase = [[[3.7128711e-4, 0.18534794,
                        1.30770596], [-np.pi, -np.pi, -np.pi],
                       [0., 0., 0.]],
                      [[-np.pi, -np.pi, -np.pi],
                       [-1.66852323, -1.89254688, -1.62050658],
                       [-0.13298964, -1.10714871, -2.75046720]]]

        mag, phase, omega = sys.freqresp(true_omega)

        np.testing.assert_array_almost_equal(mag, true_mag)
        np.testing.assert_array_almost_equal(phase, true_phase)
        np.testing.assert_array_equal(omega, true_omega)

    # Tests for TransferFunction.pole and TransferFunction.zero.
    def test_common_den(self):
        """ Test the helper function to compute common denomitators."""

        # _common_den() computes the common denominator per input/column.
        # The testing columns are:
        # 0: no common poles
        # 1: regular common poles
        # 2: poles with multiplicity,
        # 3: complex poles
        # 4: complex poles below threshold

        eps = np.finfo(float).eps
        tol_imag = np.sqrt(eps*5*2*2)*0.9

        numin = [[[1.], [1.], [1.], [1.], [1.]],
                 [[1.], [1.], [1.], [1.], [1.]]]
        denin = [[[1., 3., 2.],          # 0: poles: [-1, -2]
                  [1., 6., 11., 6.],     # 1: poles: [-1, -2, -3]
                  [1., 6., 11., 6.],     # 2: poles: [-1, -2, -3]
                  [1., 6., 11., 6.],     # 3: poles: [-1, -2, -3]
                  [1., 6., 11., 6.]],    # 4: poles: [-1, -2, -3],
                 [[1., 12., 47., 60.],   # 0: poles: [-3, -4, -5]
                  [1., 9., 26., 24.],    # 1: poles: [-2, -3, -4]
                  [1., 7., 16., 12.],    # 2: poles: [-2, -2, -3]
                  [1., 7., 17., 15.],    # 3: poles: [-2+1J, -2-1J, -3],
                  np.poly([-2 + tol_imag * 1J, -2 - tol_imag * 1J, -3])]]
        numref = np.array([
                [[0.,  0.,  1., 12., 47., 60.],
                 [0.,  0.,  0.,  1.,  4.,  0.],
                 [0.,  0.,  0.,  1.,  2.,  0.],
                 [0.,  0.,  0.,  1.,  4.,  5.],
                 [0.,  0.,  0.,  1.,  2.,  0.]],
                [[0.,  0.,  0.,  1.,  3.,  2.],
                 [0.,  0.,  0.,  1.,  1.,  0.],
                 [0.,  0.,  0.,  1.,  1.,  0.],
                 [0.,  0.,  0.,  1.,  3.,  2.],
                 [0.,  0.,  0.,  1.,  1.,  0.]]])
        denref = np.array(
                [[1., 15., 85., 225., 274., 120.],
                 [1., 10., 35., 50., 24.,  0.],
                 [1.,  8., 23., 28., 12.,  0.],
                 [1., 10., 40., 80., 79., 30.],
                 [1.,  8., 23., 28., 12.,  0.]])
        sys = TransferFunction(numin, denin)
        num, den, denorder = sys._common_den()
        np.testing.assert_array_almost_equal(num[:2, :, :], numref)
        np.testing.assert_array_almost_equal(num[2:, :, :],
                                             np.zeros((3, 5, 6)))
        np.testing.assert_array_almost_equal(den, denref)

    def test_common_den_nonproper(self):
        """ Test _common_den with order(num)>order(den) """

        tf1 = TransferFunction(
                [[[1., 2., 3.]], [[1., 2.]]],
                [[[1., -2.]], [[1., -3.]]])
        tf2 = TransferFunction(
                [[[1., 2.]], [[1., 2., 3.]]],
                [[[1., -2.]], [[1., -3.]]])

        common_den_ref = np.array([[1., -5., 6.]])

        np.testing.assert_raises(ValueError, tf1._common_den)
        np.testing.assert_raises(ValueError, tf2._common_den)

        _, den1, _ = tf1._common_den(allow_nonproper=True)
        np.testing.assert_array_almost_equal(den1, common_den_ref)
        _, den2, _ = tf2._common_den(allow_nonproper=True)
        np.testing.assert_array_almost_equal(den2, common_den_ref)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_pole_mimo(self):
        """Test for correct MIMO poles."""

        sys = TransferFunction(
            [[[1.], [1.]], [[1.], [1.]]],
            [[[1., 2.], [1., 3.]], [[1., 4., 4.], [1., 9., 14.]]])
        p = sys.pole()

        np.testing.assert_array_almost_equal(p, [-2., -2., -7., -3., -2.])

        # non proper transfer function
        sys2 = TransferFunction(
            [[[1., 2., 3., 4.], [1.]], [[1.], [1.]]],
            [[[1., 2.], [1., 3.]], [[1., 4., 4.], [1., 9., 14.]]])
        p2 = sys2.pole()

        np.testing.assert_array_almost_equal(p2, [-2., -2., -7., -3., -2.])

    def test_double_cancelling_poles_siso(self):

        H = TransferFunction([1, 1], [1, 2, 1])
        p = H.pole()
        np.testing.assert_array_almost_equal(p, [-1, -1])

    # Tests for TransferFunction.feedback
    def test_feedback_siso(self):
        """Test for correct SISO transfer function feedback."""

        sys1 = TransferFunction([-1., 4.], [1., 3., 5.])
        sys2 = TransferFunction([2., 3., 0.], [1., -3., 4., 0])

        sys3 = sys1.feedback(sys2)
        sys4 = sys1.feedback(sys2, 1)

        np.testing.assert_array_equal(sys3.num, [[[-1., 7., -16., 16., 0.]]])
        np.testing.assert_array_equal(sys3.den, [[[1., 0., -2., 2., 32., 0.]]])
        np.testing.assert_array_equal(sys4.num, [[[-1., 7., -16., 16., 0.]]])
        np.testing.assert_array_equal(sys4.den, [[[1., 0., 2., -8., 8., 0.]]])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_convert_to_transfer_function(self):
        """Test for correct state space to transfer function conversion."""

        A = [[1., -2.], [-3., 4.]]
        B = [[6., 5.], [4., 3.]]
        C = [[1., -2.], [3., -4.], [5., -6.]]
        D = [[1., 0.], [0., 1.], [1., 0.]]
        sys = StateSpace(A, B, C, D)

        tfsys = _convert_to_transfer_function(sys)

        num = [[np.array([1., -7., 10.]), np.array([-1., 10.])],
               [np.array([2., -8.]), np.array([1., -2., -8.])],
               [np.array([1., 1., -30.]), np.array([7., -22.])]]
        den = [[np.array([1., -5., -2.]) for _ in range(sys.inputs)]
               for _ in range(sys.outputs)]

        for i in range(sys.outputs):
            for j in range(sys.inputs):
                np.testing.assert_array_almost_equal(tfsys.num[i][j], num[i][j])
                np.testing.assert_array_almost_equal(tfsys.den[i][j], den[i][j])

    def test_minreal(self):
        """Try the minreal function, and also test easy entry by creation
        of a Laplace variable s"""
        s = TransferFunction([1, 0], [1])
        h = (s + 1) * (s + 2.00000000001) / (s + 2) / (s**2 + s + 1)
        hm = h.minreal()
        hr = (s + 1) / (s**2 + s + 1)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(hm.den[0][0], hr.den[0][0])
        np.testing.assert_equal(hm.dt, hr.dt)

    def test_minreal_2(self):
        """This one gave a problem, due to poly([]) giving simply 1
        instead of numpy.array([1])"""
        s = TransferFunction([1, 0], [1])
        G = 6205/(s*(s**2 + 13*s + 1281))
        Heq = G.feedback(1)
        H1 = 1/(s+5)
        H2a = Heq/H1
        H2b = H2a.minreal()
        hr = 6205/(s**2+8*s+1241)
        np.testing.assert_array_almost_equal(H2b.num[0][0], hr.num[0][0])
        np.testing.assert_array_almost_equal(H2b.den[0][0], hr.den[0][0])
        np.testing.assert_equal(H2b.dt, hr.dt)

    def test_minreal_3(self):
        """Regression test for minreal of tf([1,1],[1,1])"""
        g = TransferFunction([1,1],[1,1]).minreal()
        np.testing.assert_array_almost_equal(1.0, g.num[0][0])
        np.testing.assert_array_almost_equal(1.0, g.den[0][0])

    def test_minreal_4(self):
        """Check minreal on discrete TFs."""
        T = 0.01
        z = TransferFunction([1, 0], [1], T)
        h = (z - 1.00000000001) * (z + 1.0000000001) / (z**2 - 1)
        hm = h.minreal()
        hr = TransferFunction([1], [1], T)
        np.testing.assert_array_almost_equal(hm.num[0][0], hr.num[0][0])
        np.testing.assert_equal(hr.dt, hm.dt)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_state_space_conversion_mimo(self):
        """Test conversion of a single input, two-output state-space
        system against the same TF"""
        s = TransferFunction([1, 0], [1])
        b0 = 0.2
        b1 = 0.1
        b2 = 0.5
        a0 = 2.3
        a1 = 6.3
        a2 = 3.6
        a3 = 1.0
        h = (b0 + b1*s + b2*s**2)/(a0 + a1*s + a2*s**2 + a3*s**3)
        H = TransferFunction([[h.num[0][0]], [(h*s).num[0][0]]],
                             [[h.den[0][0]], [h.den[0][0]]])
        sys = _convertToStateSpace(H)
        H2 = _convert_to_transfer_function(sys)
        np.testing.assert_array_almost_equal(H.num[0][0], H2.num[0][0])
        np.testing.assert_array_almost_equal(H.den[0][0], H2.den[0][0])
        np.testing.assert_array_almost_equal(H.num[1][0], H2.num[1][0])
        np.testing.assert_array_almost_equal(H.den[1][0], H2.den[1][0])

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_indexing(self):
        tm = ss2tf(rss(5, 3, 3))

        # scalar indexing
        sys01 = tm[0, 1]
        np.testing.assert_array_almost_equal(sys01.num[0][0], tm.num[0][1])
        np.testing.assert_array_almost_equal(sys01.den[0][0], tm.den[0][1])

        # slice indexing
        sys = tm[:2, 1:3]
        np.testing.assert_array_almost_equal(sys.num[0][0], tm.num[0][1])
        np.testing.assert_array_almost_equal(sys.den[0][0], tm.den[0][1])
        np.testing.assert_array_almost_equal(sys.num[0][1], tm.num[0][2])
        np.testing.assert_array_almost_equal(sys.den[0][1], tm.den[0][2])
        np.testing.assert_array_almost_equal(sys.num[1][0], tm.num[1][1])
        np.testing.assert_array_almost_equal(sys.den[1][0], tm.den[1][1])
        np.testing.assert_array_almost_equal(sys.num[1][1], tm.num[1][2])
        np.testing.assert_array_almost_equal(sys.den[1][1], tm.den[1][2])

    def test_matrix_multiply(self):
        """MIMO transfer functions should be multiplyable by constant
        matrices"""
        s = TransferFunction([1, 0], [1])
        b0 = 0.2
        b1 = 0.1
        b2 = 0.5
        a0 = 2.3
        a1 = 6.3
        a2 = 3.6
        a3 = 1.0
        h = (b0 + b1*s + b2*s**2)/(a0 + a1*s + a2*s**2 + a3*s**3)
        H = TransferFunction([[h.num[0][0]], [(h*s).num[0][0]]],
                             [[h.den[0][0]], [h.den[0][0]]])
        H1 = (np.matrix([[1.0, 0]])*H).minreal()
        H2 = (np.matrix([[0, 1.0]])*H).minreal()
        np.testing.assert_array_almost_equal(H.num[0][0], H1.num[0][0])
        np.testing.assert_array_almost_equal(H.den[0][0], H1.den[0][0])
        np.testing.assert_array_almost_equal(H.num[1][0], H2.num[0][0])
        np.testing.assert_array_almost_equal(H.den[1][0], H2.den[0][0])

    def test_dcgain_cont(self):
        """Test DC gain for continuous-time transfer functions"""
        sys = TransferFunction(6, 3)
        np.testing.assert_equal(sys.dcgain(), 2)

        sys2 = TransferFunction(6, [1, 3])
        np.testing.assert_equal(sys2.dcgain(), 2)

        sys3 = TransferFunction(6, [1, 0])
        np.testing.assert_equal(sys3.dcgain(), np.inf)

        num = [[[15], [21], [33]], [[10], [14], [22]]]
        den = [[[1, 3], [2, 3], [3, 3]], [[1, 5], [2, 7], [3, 11]]]
        sys4 = TransferFunction(num, den)
        expected = [[5, 7, 11], [2, 2, 2]]
        np.testing.assert_array_equal(sys4.dcgain(), expected)

    def test_dcgain_discr(self):
        """Test DC gain for discrete-time transfer functions"""
        # static gain
        sys = TransferFunction(6, 3, True)
        np.testing.assert_equal(sys.dcgain(), 2)

        # averaging filter
        sys = TransferFunction(0.5, [1, -0.5], True)
        np.testing.assert_almost_equal(sys.dcgain(), 1)

        # differencer
        sys = TransferFunction(1, [1, -1], True)
        np.testing.assert_equal(sys.dcgain(), np.inf)

        # summer
        # causes a RuntimeWarning due to the divide by zero
        sys = TransferFunction([1, -1], [1], True)
        np.testing.assert_equal(sys.dcgain(), 0)

    def test_ss2tf(self):
        A = np.array([[-4, -1], [-1, -4]])
        B = np.array([[1], [3]])
        C = np.array([[3, 1]])
        D = 0
        sys = ss2tf(A, B, C, D)
        true_sys = TransferFunction([6., 14.], [1., 8., 15.])
        np.testing.assert_almost_equal(sys.num, true_sys.num)
        np.testing.assert_almost_equal(sys.den, true_sys.den)

    def test_class_constants(self):
        # Make sure that the 's' variable is defined properly
        s = TransferFunction.s
        G = (s + 1)/(s**2 + 2*s + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        self.assertTrue(isctime(G, strict=True))

        # Make sure that the 'z' variable is defined properly
        z = TransferFunction.z
        G = (z + 1)/(z**2 + 2*z + 1)
        np.testing.assert_array_almost_equal(G.num, [[[1, 1]]])
        np.testing.assert_array_almost_equal(G.den, [[[1, 2, 1]]])
        self.assertTrue(isdtime(G, strict=True))

    def test_printing(self):
        # SISO, continuous time
        sys = ss2tf(rss(4, 1, 1))
        self.assertTrue(isinstance(str(sys), str))
        self.assertTrue(isinstance(sys._repr_latex_(), str))

        # SISO, discrete time
        sys = sample_system(sys, 1)
        self.assertTrue(isinstance(str(sys), str))
        self.assertTrue(isinstance(sys._repr_latex_(), str))

    def test_printing_polynomial(self):
        """Cover all _tf_polynomial_to_string code branches"""
        # Note: the assertions below use plain assert statements instead of
        # unittest methods so that debugging with pytest is easier

        assert str(TransferFunction([0], [1])) == "\n0\n-\n1\n"
        assert str(TransferFunction([1.0001], [-1.1111])) == \
            "\n  1\n------\n-1.111\n"
        assert str(TransferFunction([0, 1], [0, 1.])) == "\n1\n-\n1\n"
        for var, dt, dtstring in zip(["s", "z", "z"],
                                     [None, True, 1],
                                     ['', '', '\ndt = 1\n']):
            assert str(TransferFunction([1, 0], [2, 1], dt)) == \
                "\n   {var}\n-------\n2 {var} + 1\n{dtstring}".format(
                    var=var, dtstring=dtstring)
            assert str(TransferFunction([2, 0, -1], [1, 0, 0, 1.2], dt)) == \
                "\n2 {var}^2 - 1\n---------\n{var}^3 + 1.2\n{dtstring}".format(
                    var=var, dtstring=dtstring)

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_printing_mimo(self):
        # MIMO, continuous time
        sys = ss2tf(rss(4, 2, 3))
        self.assertTrue(isinstance(str(sys), str))
        self.assertTrue(isinstance(sys._repr_latex_(), str))

    @unittest.skipIf(not slycot_check(), "slycot not installed")
    def test_size_mismatch(self):
        sys1 = ss2tf(rss(2, 2, 2))

        # Different number of inputs
        sys2 = ss2tf(rss(3, 1, 2))
        self.assertRaises(ValueError, TransferFunction.__add__, sys1, sys2)

        # Different number of outputs
        sys2 = ss2tf(rss(3, 2, 1))
        self.assertRaises(ValueError, TransferFunction.__add__, sys1, sys2)

        # Inputs and outputs don't match
        self.assertRaises(ValueError, TransferFunction.__mul__, sys2, sys1)

        # Feedback mismatch (MIMO not implemented)
        self.assertRaises(NotImplementedError,
                          TransferFunction.feedback, sys2, sys1)

    def test_latex_repr(self):
        """ Test latex printout for TransferFunction """
        Hc = TransferFunction([1e-5, 2e5, 3e-4],
                              [1.2e34, 2.3e-4, 2.3e-45])
        Hd = TransferFunction([1e-5, 2e5, 3e-4],
                              [1.2e34, 2.3e-4, 2.3e-45],
                              .1)
        # TODO: make the multiplication sign configurable
        expmul = r'\times'
        for var, H, suffix in zip(['s', 'z'],
                                  [Hc, Hd],
                                  ['', r'\quad dt = 0.1']):
            ref = (r'$$\frac{'
                   r'1 ' + expmul + ' 10^{-5} ' + var + '^2 '
                   r'+ 2 ' + expmul + ' 10^{5} ' + var + ' + 0.0003'
                   r'}{'
                   r'1.2 ' + expmul + ' 10^{34} ' + var + '^2 '
                   r'+ 0.00023 ' + var + ' '
                   r'+ 2.3 ' + expmul + ' 10^{-45}'
                   r'}' + suffix + '$$')
            self.assertEqual(H._repr_latex_(), ref)

    def test_repr(self):
        """Test __repr__ printout."""
        Hc = TransferFunction([-1., 4.], [1., 3., 5.])
        Hd = TransferFunction([2., 3., 0.], [1., -3., 4., 0], 2.0)
        Hcm = TransferFunction(
            [ [[0, 1], [2, 3]], [[4, 5], [6, 7]] ],
            [ [[6, 7], [4, 5]], [[2, 3], [0, 1]] ])
        Hdm = TransferFunction(
            [ [[0, 1], [2, 3]], [[4, 5], [6, 7]] ],
            [ [[6, 7], [4, 5]], [[2, 3], [0, 1]] ], 0.5)

        refs = [
            "TransferFunction(array([-1.,  4.]), array([1., 3., 5.]))",
            "TransferFunction(array([2., 3., 0.]),"
            " array([ 1., -3.,  4.,  0.]), 2.0)",
            "TransferFunction([[array([1]), array([2, 3])],"
            " [array([4, 5]), array([6, 7])]],"
            " [[array([6, 7]), array([4, 5])],"
            " [array([2, 3]), array([1])]])",
            "TransferFunction([[array([1]), array([2, 3])],"
            " [array([4, 5]), array([6, 7])]],"
            " [[array([6, 7]), array([4, 5])],"
            " [array([2, 3]), array([1])]], 0.5)" ]
        self.assertEqual(repr(Hc), refs[0])
        self.assertEqual(repr(Hd), refs[1])
        self.assertEqual(repr(Hcm), refs[2])
        self.assertEqual(repr(Hdm), refs[3])

        # and reading back
        array = np.array
        for H in (Hc, Hd, Hcm, Hdm):
            H2 = eval(H.__repr__())
            for p in range(len(H.num)):
                for m in range(len(H.num[0])):
                    np.testing.assert_array_almost_equal(
                        H.num[p][m], H2.num[p][m])
                    np.testing.assert_array_almost_equal(
                        H.den[p][m], H2.den[p][m])
            self.assertEqual(H.dt, H2.dt)
    
if __name__ == "__main__":
    unittest.main()
