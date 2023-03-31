"""ctrlutil_test.py"""

import numpy as np
import pytest
import control as ct
from control.ctrlutil import db2mag, mag2db, unwrap

class TestUtils:

    mag = np.array([1, 10, 100, 2, 0.1, 0.01])
    db = np.array([0, 20, 40, 6.0205999, -20, -40])

    def check_unwrap_array(self, angle, period=None):
        if period is None:
            angle_mod = angle % (2 * np.pi)
            angle_unwrap = unwrap(angle_mod)
        else:
            angle_mod = angle % period
            angle_unwrap = unwrap(angle_mod, period)
        np.testing.assert_array_almost_equal(angle_unwrap, angle)

    def test_unwrap_increasing(self):
        angle = np.linspace(0, 20, 50)
        self.check_unwrap_array(angle)

    def test_unwrap_decreasing(self):
        angle = np.linspace(0, -20, 50)
        self.check_unwrap_array(angle)

    def test_unwrap_inc_degrees(self):
        angle = np.linspace(0, 720, 50)
        self.check_unwrap_array(angle, 360)

    def test_unwrap_dec_degrees(self):
        angle = np.linspace(0, -720, 50)
        self.check_unwrap_array(angle, 360)

    def test_unwrap_large_skips(self):
        angle = np.array([0., 4 * np.pi, -2 * np.pi])
        np.testing.assert_array_almost_equal(unwrap(angle), [0., 0., 0.])

    def test_unwrap_list(self):
        angle = [0, 2.2, 5.4, -0.4]
        angle_unwrapped = [0, 0.2, 0.4, 0.6]
        np.testing.assert_array_almost_equal(unwrap(angle, 1.0), angle_unwrapped)

    def test_db2mag(self):
        for mag, db in zip(self.mag, self.db):
            np.testing.assert_almost_equal(mag, db2mag(db))

    def test_db2mag_array(self):
        mag_array = db2mag(self.db)
        np.testing.assert_array_almost_equal(mag_array, self.mag)

    def test_mag2db(self):
        for db, mag in zip(self.db, self.mag):
            np.testing.assert_almost_equal(db, mag2db(mag))

    def test_mag2db_array(self):
        db_array = mag2db(self.mag)
        np.testing.assert_array_almost_equal(db_array, self.db)

    def test_issys(self):
        sys = ct.rss(2, 1, 1)
        with pytest.warns(FutureWarning, match="deprecated; use isinstance"):
            ct.issys(sys)
