"""discrete_test.py - test discrete time classes

RMM, 9 Sep 2012
"""

import numpy as np
import pytest

from control import (StateSpace, TransferFunction, bode, common_timebase,
                     feedback, forced_response, impulse_response,
                     isctime, isdtime, rss, c2d, sample_system, step_response,
                     timebase)


class TestDiscrete:
    """Tests for the system classes with discrete timebase."""

    @pytest.fixture
    def tsys(self):
        """Create some systems for testing"""
        class Tsys:
            pass
        T = Tsys()
        # Single input, single output continuous and discrete time systems
        sys = rss(3, 1, 1)
        T.siso_ss1 = StateSpace(sys.A, sys.B, sys.C, sys.D, None)
        T.siso_ss1c = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.0)
        T.siso_ss1d = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.1)
        T.siso_ss2d = StateSpace(sys.A, sys.B, sys.C, sys.D, 0.2)
        T.siso_ss3d = StateSpace(sys.A, sys.B, sys.C, sys.D, True)

        # Two input, two output continuous time system
        A = [[-3., 4., 2.], [-1., -3., 0.], [2., 5., 3.]]
        B = [[1., 4.], [-3., -3.], [-2., 1.]]
        C = [[4., 2., -3.], [1., 4., 3.]]
        D = [[-2., 4.], [0., 1.]]
        T.mimo_ss1 = StateSpace(A, B, C, D, None)
        T.mimo_ss1c = StateSpace(A, B, C, D, 0)

        # Two input, two output discrete time system
        T.mimo_ss1d = StateSpace(A, B, C, D, 0.1)

        # Same system, but with a different sampling time
        T.mimo_ss2d = StateSpace(A, B, C, D, 0.2)

        # Single input, single output continuus and discrete transfer function
        T.siso_tf1 = TransferFunction([1, 1], [1, 2, 1], None)
        T.siso_tf1c = TransferFunction([1, 1], [1, 2, 1], 0)
        T.siso_tf1d = TransferFunction([1, 1], [1, 2, 1], 0.1)
        T.siso_tf2d = TransferFunction([1, 1], [1, 2, 1], 0.2)
        T.siso_tf3d = TransferFunction([1, 1], [1, 2, 1], True)

        return T

    def testCompatibleTimebases(self, tsys):
        """test that compatible timebases don't throw errors and vice versa"""
        common_timebase(tsys.siso_ss1.dt, tsys.siso_tf1.dt)
        common_timebase(tsys.siso_ss1.dt, tsys.siso_ss1c.dt)
        common_timebase(tsys.siso_ss1d.dt, tsys.siso_ss1.dt)
        common_timebase(tsys.siso_ss1.dt, tsys.siso_ss1d.dt)
        common_timebase(tsys.siso_ss1.dt, tsys.siso_ss1d.dt)
        common_timebase(tsys.siso_ss1d.dt, tsys.siso_ss3d.dt)
        common_timebase(tsys.siso_ss3d.dt, tsys.siso_ss1d.dt)
        with pytest.raises(ValueError):
            # cont + discrete
            common_timebase(tsys.siso_ss1d.dt, tsys.siso_ss1c.dt)
        with pytest.raises(ValueError):
            # incompatible discrete
            common_timebase(tsys.siso_ss1d.dt, tsys.siso_ss2d.dt)

    def testSystemInitialization(self, tsys):
        # Check to make sure systems are discrete time with proper variables
        assert tsys.siso_ss1.dt is None
        assert tsys.siso_ss1c.dt == 0
        assert tsys.siso_ss1d.dt == 0.1
        assert tsys.siso_ss2d.dt == 0.2
        assert tsys.siso_ss3d.dt is True
        assert tsys.mimo_ss1c.dt == 0
        assert tsys.mimo_ss1d.dt == 0.1
        assert tsys.mimo_ss2d.dt == 0.2
        assert tsys.siso_tf1.dt is None
        assert tsys.siso_tf1c.dt == 0
        assert tsys.siso_tf1d.dt == 0.1
        assert tsys.siso_tf2d.dt == 0.2
        assert tsys.siso_tf3d.dt is True

        # keyword argument check
        # dynamic systems
        assert TransferFunction(1, [1, 1], dt=0.1).dt == 0.1
        assert TransferFunction(1, [1, 1], 0.1).dt == 0.1
        assert StateSpace(1,1,1,1, dt=0.1).dt == 0.1
        assert StateSpace(1,1,1,1, 0.1).dt == 0.1
        # static gain system, dt argument should still override default dt
        assert TransferFunction(1, [1,], dt=0.1).dt == 0.1
        assert TransferFunction(1, [1,], 0.1).dt == 0.1
        assert StateSpace(0,0,1,1, dt=0.1).dt == 0.1
        assert StateSpace(0,0,1,1, 0.1).dt == 0.1

    def testCopyConstructor(self, tsys):
        for sys in (tsys.siso_ss1, tsys.siso_ss1c, tsys.siso_ss1d):
            newsys = StateSpace(sys)
            assert sys.dt == newsys.dt
        for sys in (tsys.siso_tf1, tsys.siso_tf1c, tsys.siso_tf1d):
            newsys = TransferFunction(sys)
            assert sys.dt == newsys.dt

    def test_timebase(self, tsys):
        assert timebase(1) is None
        with pytest.raises(ValueError):
            timebase([1, 2])
        assert timebase(tsys.siso_ss1, strict=False) is None
        assert timebase(tsys.siso_ss1, strict=True) is None
        assert timebase(tsys.siso_ss1c) == 0
        assert timebase(tsys.siso_ss1d) == 0.1
        assert timebase(tsys.siso_ss2d) == 0.2
        assert timebase(tsys.siso_ss3d)
        assert timebase(tsys.siso_ss3d, strict=False) == 1
        assert timebase(tsys.siso_tf1, strict=False) is None
        assert timebase(tsys.siso_tf1, strict=True) is None
        assert timebase(tsys.siso_tf1c) == 0
        assert timebase(tsys.siso_tf1d) == 0.1
        assert timebase(tsys.siso_tf2d) == 0.2
        assert timebase(tsys.siso_tf3d)
        assert timebase(tsys.siso_tf3d, strict=False) == 1

    def test_timebase_conversions(self, tsys):
        '''Check to make sure timebases transfer properly'''
        tf1 = TransferFunction([1, 1], [1, 2, 3], None)  # unspecified
        tf2 = TransferFunction([1, 1], [1, 2, 3], 0)     # cont time
        tf3 = TransferFunction([1, 1], [1, 2, 3], True)  # dtime, unspec
        tf4 = TransferFunction([1, 1], [1, 2, 3], .1)    # dtime, dt=.1

        # Make sure unspecified timebase is converted correctly
        assert timebase(tf1*tf1) == timebase(tf1)
        assert timebase(tf1*tf2) == timebase(tf2)
        assert timebase(tf1*tf3) == timebase(tf3)
        assert timebase(tf1*tf4) == timebase(tf4)
        assert timebase(tf3*tf4) == timebase(tf4)
        assert timebase(tf2*tf1) == timebase(tf2)
        assert timebase(tf3*tf1) == timebase(tf3)
        assert timebase(tf4*tf1) == timebase(tf4)
        assert timebase(tf1+tf1) == timebase(tf1)
        assert timebase(tf1+tf2) == timebase(tf2)
        assert timebase(tf1+tf3) == timebase(tf3)
        assert timebase(tf1+tf4) == timebase(tf4)
        assert timebase(feedback(tf1, tf1)) == timebase(tf1)
        assert timebase(feedback(tf1, tf2)) == timebase(tf2)
        assert timebase(feedback(tf1, tf3)) == timebase(tf3)
        assert timebase(feedback(tf1, tf4)) == timebase(tf4)

        # Make sure discrete time without sampling is converted correctly
        assert timebase(tf3*tf3) == timebase(tf3)
        assert timebase(tf3*tf4) == timebase(tf4)
        assert timebase(tf3+tf3) == timebase(tf3)
        assert timebase(tf3+tf4) == timebase(tf4)
        assert timebase(feedback(tf3, tf3)) == timebase(tf3)
        assert timebase(feedback(tf3, tf4)) == timebase(tf4)

        # Make sure all other combinations are errors
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf2 * tf3
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf3 * tf2
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf2 * tf4
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf4 * tf2
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf2 + tf3
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf3 + tf2
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf2 + tf4
        with pytest.raises(ValueError, match="incompatible timebases"):
            tf4 + tf2
        with pytest.raises(ValueError, match="incompatible timebases"):
            feedback(tf2, tf3)
        with pytest.raises(ValueError, match="incompatible timebases"):
            feedback(tf3, tf2)
        with pytest.raises(ValueError, match="incompatible timebases"):
            feedback(tf2, tf4)
        with pytest.raises(ValueError, match="incompatible timebases"):
            feedback(tf4, tf2)

    def testisdtime(self, tsys):
        # Constant
        assert isdtime(1)
        assert not isdtime(1, strict=True)

        # State space
        assert isdtime(tsys.siso_ss1)
        assert not isdtime(tsys.siso_ss1, strict=True)
        assert not isdtime(tsys.siso_ss1c)
        assert not isdtime(tsys.siso_ss1c, strict=True)
        assert isdtime(tsys.siso_ss1d)
        assert isdtime(tsys.siso_ss1d, strict=True)
        assert isdtime(tsys.siso_ss3d, strict=True)

        # Transfer function
        assert isdtime(tsys.siso_tf1)
        assert not isdtime(tsys.siso_tf1, strict=True)
        assert not isdtime(tsys.siso_tf1c)
        assert not isdtime(tsys.siso_tf1c, strict=True)
        assert isdtime(tsys.siso_tf1d)
        assert isdtime(tsys.siso_tf1d, strict=True)
        assert isdtime(tsys.siso_tf3d, strict=True)

    def testisctime(self, tsys):
        # Constant
        assert isctime(1)
        assert not isctime(1, strict=True)

        # State Space
        assert isctime(tsys.siso_ss1)
        assert not isctime(tsys.siso_ss1, strict=True)
        assert isctime(tsys.siso_ss1c)
        assert isctime(tsys.siso_ss1c, strict=True)
        assert not isctime(tsys.siso_ss1d)
        assert not isctime(tsys.siso_ss1d, strict=True)
        assert not isctime(tsys.siso_ss3d, strict=True)

        # Transfer Function
        assert isctime(tsys.siso_tf1)
        assert not isctime(tsys.siso_tf1, strict=True)
        assert isctime(tsys.siso_tf1c)
        assert isctime(tsys.siso_tf1c, strict=True)
        assert not isctime(tsys.siso_tf1d)
        assert not isctime(tsys.siso_tf1d, strict=True)
        assert not isctime(tsys.siso_tf3d, strict=True)

    def testAddition(self, tsys):
        # State space addition
        sys = tsys.siso_ss1 + tsys.siso_ss1d
        sys = tsys.siso_ss1 + tsys.siso_ss1c
        sys = tsys.siso_ss1c + tsys.siso_ss1
        sys = tsys.siso_ss1d + tsys.siso_ss1
        sys = tsys.siso_ss1c + tsys.siso_ss1c
        sys = tsys.siso_ss1d + tsys.siso_ss1d
        sys = tsys.siso_ss3d + tsys.siso_ss3d
        sys = tsys.siso_ss1d + tsys.siso_ss3d

        with pytest.raises(ValueError):
            StateSpace.__add__(tsys.mimo_ss1c, tsys.mimo_ss1d)
        with pytest.raises(ValueError):
            StateSpace.__add__(tsys.mimo_ss1d, tsys.mimo_ss2d)

        # Transfer function addition
        sys = tsys.siso_tf1 + tsys.siso_tf1d
        sys = tsys.siso_tf1 + tsys.siso_tf1c
        sys = tsys.siso_tf1c + tsys.siso_tf1
        sys = tsys.siso_tf1d + tsys.siso_tf1
        sys = tsys.siso_tf1c + tsys.siso_tf1c
        sys = tsys.siso_tf1d + tsys.siso_tf1d
        sys = tsys.siso_tf2d + tsys.siso_tf2d
        sys = tsys.siso_tf1d + tsys.siso_tf3d

        with pytest.raises(ValueError):
            TransferFunction.__add__(tsys.siso_tf1c, tsys.siso_tf1d)
        with pytest.raises(ValueError):
            TransferFunction.__add__(tsys.siso_tf1d, tsys.siso_tf2d)

        # State space + transfer function
        sys = tsys.siso_ss1c + tsys.siso_tf1c
        sys = tsys.siso_tf1c + tsys.siso_ss1c
        sys = tsys.siso_ss1d + tsys.siso_tf1d
        sys = tsys.siso_tf1d + tsys.siso_ss1d
        with pytest.raises(ValueError):
            TransferFunction.__add__(tsys.siso_tf1c, tsys.siso_ss1d)

    def testMultiplication(self, tsys):
        # State space multiplication
        sys = tsys.siso_ss1 * tsys.siso_ss1d
        sys = tsys.siso_ss1 * tsys.siso_ss1c
        sys = tsys.siso_ss1c * tsys.siso_ss1
        sys = tsys.siso_ss1d * tsys.siso_ss1
        sys = tsys.siso_ss1c * tsys.siso_ss1c
        sys = tsys.siso_ss1d * tsys.siso_ss1d
        sys = tsys.siso_ss1d * tsys.siso_ss3d

        with pytest.raises(ValueError):
            StateSpace.__mul__(tsys.mimo_ss1c, tsys.mimo_ss1d)
        with pytest.raises(ValueError):
            StateSpace.__mul__(tsys.mimo_ss1d, tsys.mimo_ss2d)

        # Transfer function multiplication
        sys = tsys.siso_tf1 * tsys.siso_tf1d
        sys = tsys.siso_tf1 * tsys.siso_tf1c
        sys = tsys.siso_tf1c * tsys.siso_tf1
        sys = tsys.siso_tf1d * tsys.siso_tf1
        sys = tsys.siso_tf1c * tsys.siso_tf1c
        sys = tsys.siso_tf1d * tsys.siso_tf1d
        sys = tsys.siso_tf1d * tsys.siso_tf3d

        with pytest.raises(ValueError):
            TransferFunction.__mul__(tsys.siso_tf1c, tsys.siso_tf1d)
        with pytest.raises(ValueError):
            TransferFunction.__mul__(tsys.siso_tf1d, tsys.siso_tf2d)

        # State space * transfer function
        sys = tsys.siso_ss1c * tsys.siso_tf1c
        sys = tsys.siso_tf1c * tsys.siso_ss1c
        sys = tsys.siso_ss1d * tsys.siso_tf1d
        sys = tsys.siso_tf1d * tsys.siso_ss1d
        with pytest.raises(ValueError):
            TransferFunction.__mul__(tsys.siso_tf1c,
                          tsys.siso_ss1d)


    def testFeedback(self, tsys):
        # State space feedback
        sys = feedback(tsys.siso_ss1, tsys.siso_ss1d)
        sys = feedback(tsys.siso_ss1, tsys.siso_ss1c)
        sys = feedback(tsys.siso_ss1c, tsys.siso_ss1)
        sys = feedback(tsys.siso_ss1d, tsys.siso_ss1)
        sys = feedback(tsys.siso_ss1c, tsys.siso_ss1c)
        sys = feedback(tsys.siso_ss1d, tsys.siso_ss1d)
        sys = feedback(tsys.siso_ss1d, tsys.siso_ss3d)

        with pytest.raises(ValueError):
            feedback(tsys.mimo_ss1c, tsys.mimo_ss1d)
        with pytest.raises(ValueError):
            feedback(tsys.mimo_ss1d, tsys.mimo_ss2d)

        # Transfer function feedback
        sys = feedback(tsys.siso_tf1, tsys.siso_tf1d)
        sys = feedback(tsys.siso_tf1, tsys.siso_tf1c)
        sys = feedback(tsys.siso_tf1c, tsys.siso_tf1)
        sys = feedback(tsys.siso_tf1d, tsys.siso_tf1)
        sys = feedback(tsys.siso_tf1c, tsys.siso_tf1c)
        sys = feedback(tsys.siso_tf1d, tsys.siso_tf1d)
        sys = feedback(tsys.siso_tf1d, tsys.siso_tf3d)

        with pytest.raises(ValueError):
            feedback(tsys.siso_tf1c, tsys.siso_tf1d)
        with pytest.raises(ValueError):
            feedback(tsys.siso_tf1d, tsys.siso_tf2d)

        # State space, transfer function
        sys = feedback(tsys.siso_ss1c, tsys.siso_tf1c)
        sys = feedback(tsys.siso_tf1c, tsys.siso_ss1c)
        sys = feedback(tsys.siso_ss1d, tsys.siso_tf1d)
        sys = feedback(tsys.siso_tf1d, tsys.siso_ss1d)
        with pytest.raises(ValueError):
            feedback(tsys.siso_tf1c, tsys.siso_ss1d)

    def testSimulation(self, tsys):
        T = range(100)
        U = np.sin(T)

        # For now, just check calling syntax
        # TODO: add checks on output of simulations
        tout, yout = step_response(tsys.siso_ss1d)
        tout, yout = step_response(tsys.siso_ss1d, T)
        tout, yout = impulse_response(tsys.siso_ss1d)
        tout, yout = impulse_response(tsys.siso_ss1d, T)
        tout, yout = forced_response(tsys.siso_ss1d, T, U, 0)
        tout, yout = forced_response(tsys.siso_ss2d, T, U, 0)
        tout, yout = forced_response(tsys.siso_ss3d, T, U, 0)
        tout, yout, xout = forced_response(tsys.siso_ss1d, T, U, 0,
                                           return_x=True)

    def test_sample_system(self, tsys):
        # Make sure we can convert various types of systems
        for sysc in (tsys.siso_tf1, tsys.siso_tf1c,
                     tsys.siso_ss1, tsys.siso_ss1c,
                     tsys.mimo_ss1, tsys.mimo_ss1c):
            for method in ("zoh", "bilinear", "euler", "backward_diff"):
                sysd = sample_system(sysc, 1, method=method)
                assert sysd.dt == 1

        # Check "matched", defined only for SISO transfer functions
        for sysc in (tsys.siso_tf1, tsys.siso_tf1c):
            sysd = sample_system(sysc, 1, method="matched")
            assert sysd.dt == 1

    @pytest.mark.parametrize("plantname",
                             ["siso_ss1c",
                              "siso_tf1c"])
    def test_sample_system_prewarp(self, tsys, plantname):
        """bilinear approximation with prewarping test"""
        wwarp = 50
        Ts = 0.025
        # test state space version
        plant = getattr(tsys, plantname)
        plant_fr = plant(wwarp * 1j)

        plant_d_warped = plant.sample(Ts, 'bilinear', prewarp_frequency=wwarp)
        dt = plant_d_warped.dt
        plant_d_fr = plant_d_warped(np.exp(wwarp * 1.j * dt))
        np.testing.assert_array_almost_equal(plant_fr, plant_d_fr)

        plant_d_warped = sample_system(plant, Ts, 'bilinear',
                prewarp_frequency=wwarp)
        plant_d_fr = plant_d_warped(np.exp(wwarp * 1.j * dt))
        np.testing.assert_array_almost_equal(plant_fr, plant_d_fr)

        plant_d_warped = c2d(plant, Ts, 'bilinear', prewarp_frequency=wwarp)
        plant_d_fr = plant_d_warped(np.exp(wwarp * 1.j * dt))
        np.testing.assert_array_almost_equal(plant_fr, plant_d_fr)

    def test_sample_system_errors(self, tsys):
        # Check errors
        with pytest.raises(ValueError):
            sample_system(tsys.siso_ss1d, 1)
        with pytest.raises(ValueError):
            sample_system(tsys.siso_tf1d, 1)
        with pytest.raises(ValueError):
            sample_system(tsys.siso_ss1, 1, 'unknown')


    def test_sample_ss(self, tsys):
        # double integrators, two different ways
        sys1 = StateSpace([[0.,1.],[0.,0.]], [[0.],[1.]], [[1.,0.]], 0.)
        sys2 = StateSpace([[0.,0.],[1.,0.]], [[1.],[0.]], [[0.,1.]], 0.)
        I = np.eye(2)
        for sys in (sys1, sys2):
            for h in (0.1, 0.5, 1, 2):
                Ad = I + h * sys.A
                Bd = h * sys.B + 0.5 * h**2 * sys.A @ sys.B
                sysd = sample_system(sys, h, method='zoh')
                np.testing.assert_array_almost_equal(sysd.A, Ad)
                np.testing.assert_array_almost_equal(sysd.B, Bd)
                np.testing.assert_array_almost_equal(sysd.C, sys.C)
                np.testing.assert_array_almost_equal(sysd.D, sys.D)
                assert sysd.dt == h

    def test_sample_tf(self, tsys):
        # double integrator
        sys = TransferFunction(1, [1,0,0])
        for h in (0.1, 0.5, 1, 2):
            numd_expected = 0.5 * h**2 * np.array([1.,1.])
            dend_expected = np.array([1.,-2.,1.])
            sysd = sample_system(sys, h, method='zoh')
            assert sysd.dt == h
            numd = sysd.num[0][0]
            dend = sysd.den[0][0]
            np.testing.assert_array_almost_equal(numd, numd_expected)
            np.testing.assert_array_almost_equal(dend, dend_expected)

    def test_discrete_bode(self, tsys):
        # Create a simple discrete time system and check the calculation
        sys = TransferFunction([1], [1, 0.5], 1)
        omega = [1, 2, 3]
        mag_out, phase_out, omega_out = bode(sys, omega)
        H_z = list(map(lambda w: 1./(np.exp(1.j * w) + 0.5), omega))
        np.testing.assert_array_almost_equal(omega, omega_out)
        np.testing.assert_array_almost_equal(mag_out, np.absolute(H_z))
        np.testing.assert_array_almost_equal(phase_out, np.angle(H_z))
