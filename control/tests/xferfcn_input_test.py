#!/usr/bin/env python
#
# xferfcn_input_test.py - test inputs to TransferFunction class
# jed-frey, 18 Feb 2017 (based on xferfcn_test.py)

import unittest
import numpy as np

from numpy import int, int8, int16, int32, int64
from numpy import float, float16, float32, float64, longdouble
from numpy import all, ndarray, array

from control.xferfcn import _clean_part


class TestXferFcnInput(unittest.TestCase):
    """These are tests for functionality of cleaning and validating XferFcnInput."""

    # Tests for raising exceptions.
    def test_clean_part_bad_input_type(self):
        """Give the part cleaner invalid input type."""

        self.assertRaises(TypeError, _clean_part, [[0., 1.], [2., 3.]])

    def test_clean_part_bad_input_type2(self):
        """Give the part cleaner another invalid input type."""
        self.assertRaises(TypeError, _clean_part, [1, "a"])

    def test_clean_part_scalar(self):
        """Test single scalar value."""
        num = 1
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_list_scalar(self):
        """Test single scalar value in list."""
        num = [1]
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_tuple_scalar(self):
        """Test single scalar value in tuple."""
        num = (1)
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_list(self):
        """Test multiple values in a list."""
        num = [1, 2]
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0, 2.0], dtype=float))

    def test_clean_part_tuple(self):
        """Test multiple values in tuple."""
        num = (1, 2)
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0, 2.0], dtype=float))

    def test_clean_part_all_scalar_types(self):
        """Test single scalar value for all valid data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = dtype(1)
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_np_array(self):
        """Test multiple values in numpy array."""
        num = np.array([1, 2])
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0, 2.0], dtype=float))

    def test_clean_part_all_np_array_types(self):
        """Test scalar value in numpy array of ndim=0 for all data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = np.array(1, dtype=dtype)
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_all_np_array_types2(self):
        """Test numpy array for all types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = np.array([1, 2], dtype=dtype)
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 2.0], dtype=float))

    def test_clean_part_list_all_types(self):
        """Test list of a single value for all data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = [dtype(1)]
            num_ = _clean_part(num)
            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_list_all_types2(self):
        """List of list of numbers of all data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = [dtype(1), dtype(2)]
            num_ = _clean_part(num)
            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 2.0], dtype=float))

    def test_clean_part_tuple_all_types(self):
        """Test tuple of a single value for all data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = (dtype(1),)
            num_ = _clean_part(num)
            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_tuple_all_types2(self):
        """Test tuple of a single value for all data types."""
        for dtype in [int, int8, int16, int32, int64, float, float16, float32, float64, longdouble]:
            num = (dtype(1), dtype(2))
            num_ = _clean_part(num)
            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1, 2], dtype=float))

    def test_clean_part_list_list_list_int(self):
        """ Test an int in a list of a list of a list."""
        num = [[[1]]]
        num_ = _clean_part(num)
        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_list_list_list_float(self):
        """ Test a float in a list of a list of a list."""
        num = [[[1.0]]]
        num_ = _clean_part(num)
        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0], dtype=float))

    def test_clean_part_list_list_list_ints(self):
        """Test 2 lists of ints in a list in a list."""
        num = [[[1, 1], [2, 2]]]
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
        np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_list_list_list_floats(self):
        """Test 2 lists of ints in a list in a list."""
        num = [[[1.0, 1.0], [2.0, 2.0]]]
        num_ = _clean_part(num)

        assert isinstance(num_, list)
        assert np.all([isinstance(part, list) for part in num_])
        np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
        np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_list_list_array(self):
        """List of list of numpy arrays for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = [[array([1, 1], dtype=dtype), array([2, 2], dtype=dtype)]]
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_tuple_list_array(self):
        """Tuple of list of numpy arrays for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = ([array([1, 1], dtype=dtype), array([2, 2], dtype=dtype)],)
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_list_tuple_array(self):
        """List of tuple of numpy array for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = [(array([1, 1], dtype=dtype), array([2, 2], dtype=dtype))]
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_tuple_tuples_arrays(self):
        """Tuple of tuples of numpy arrays for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = ((array([1, 1], dtype=dtype), array([2, 2], dtype=dtype)),
                   (array([3, 4], dtype=dtype), array([4, 4], dtype=dtype)))
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_list_tuples_arrays(self):
        """List of tuples of numpy arrays for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = [(array([1, 1], dtype=dtype), array([2, 2], dtype=dtype)),
                   (array([3, 4], dtype=dtype), array([4, 4], dtype=dtype))]
            num_ = _clean_part(num)

            assert isinstance(num_, list)
            assert np.all([isinstance(part, list) for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))

    def test_clean_part_list_list_arrays(self):
        """List of list of numpy arrays for all valid types."""
        for dtype in int, int8, int16, int32, int64, float, float16, float32, float64, longdouble:
            num = [[array([1, 1], dtype=dtype), array([2, 2], dtype=dtype)],
                   [array([3, 3], dtype=dtype), array([4, 4], dtype=dtype)]]
            num_ = _clean_part(num)

            assert len(num_) == 2
            assert np.all([isinstance(part, list) for part in num_])
            assert np.all([len(part) == 2 for part in num_])
            np.testing.assert_array_equal(num_[0][0], array([1.0, 1.0], dtype=float))
            np.testing.assert_array_equal(num_[0][1], array([2.0, 2.0], dtype=float))
            np.testing.assert_array_equal(num_[1][0], array([3.0, 3.0], dtype=float))
            np.testing.assert_array_equal(num_[1][1], array([4.0, 4.0], dtype=float))


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestXferFcnInput)


if __name__ == "__main__":
    unittest.main()
