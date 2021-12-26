"""matlab2_test.py

Test the control.matlab toolbox.

Copyright (C) 2011 by Eike Welk.
"""


from matplotlib.pyplot import figure, plot, legend, subplot2grid
import numpy as np
from numpy import array, matrix, zeros, linspace, r_
from numpy.testing import assert_array_almost_equal

import pytest
import scipy.signal

from control.matlab import ss, step, impulse, initial, lsim, dcgain, ss2tf
from control.statesp import _mimo2siso
from control.timeresp import _check_convert_array
from control.tests.conftest import slycotonly


class TestControlMatlab:
    """Test the control.matlab toolbox."""

    @pytest.fixture
    def SISO_mats(self):
        """Return matrices for a SISO system"""
        A =  array([[-81.82, -45.45],
                    [ 10.,    -1.  ]])
        B =  array([[9.09],
                    [0.  ]])
        C =  array([[0, 0.159]])
        D = zeros((1, 1))
        return A, B, C, D

    @pytest.fixture
    def MIMO_mats(self):
        """Return matrices for a MIMO system"""
        A = array([[-81.82, -45.45,   0,      0   ],
                   [ 10,     -1,      0,      0   ],
                   [  0,      0,    -81.82, -45.45],
                   [  0,      0,     10,     -1,  ]])
        B = array([[9.09, 0   ],
                   [0   , 0   ],
                   [0   , 9.09],
                   [0   , 0   ]])
        C = array([[0, 0.159, 0, 0    ],
                   [0, 0,     0, 0.159]])
        D = zeros((2, 2))
        return A, B, C, D

    @slycotonly
    def test_dcgain_mimo(self, MIMO_mats):
        """Test function dcgain with MIMO systems"""
        #Test MIMO systems
        A, B, C, D = MIMO_mats

        gain1 = dcgain(ss(A, B, C, D))
        gain2 = dcgain(A, B, C, D)
        sys_tf = ss2tf(A, B, C, D)
        gain3 = dcgain(sys_tf)
        gain4 = dcgain(sys_tf.num, sys_tf.den)
        #print("gain1:", gain1)

        assert_array_almost_equal(gain1,
                                  array([[0.0269, 0.    ],
                                         [0.    , 0.0269]]),
                                  decimal=4)
        assert_array_almost_equal(gain1, gain2)
        assert_array_almost_equal(gain3, gain4)
        assert_array_almost_equal(gain1, gain4)

    def test_dcgain_siso(self, SISO_mats):
        """Test function dcgain with SISO systems"""
        A, B, C, D = SISO_mats

        gain1 = dcgain(ss(A, B, C, D))
        assert_array_almost_equal(gain1,
                                  array([[0.0269]]),
                                  decimal=4)

    def test_dcgain_2(self, SISO_mats):
        """Test function dcgain with different systems"""
        #Create different forms of a SISO system
        A, B, C, D = SISO_mats
        num, den = scipy.signal.ss2tf(A, B, C, D)
        # numerator is only a constant here; pick it out to avoid numpy warning
        Z, P, k = scipy.signal.tf2zpk(num[0][-1], den)
        sys_ss = ss(A, B, C, D)

        #Compute the gain with ``dcgain``
        gain_abcd = dcgain(A, B, C, D)
        gain_zpk = dcgain(Z, P, k)
        gain_numden = dcgain(np.squeeze(num), den)
        gain_sys_ss = dcgain(sys_ss)
        # print('gain_abcd:', gain_abcd, 'gain_zpk:', gain_zpk)
        # print('gain_numden:', gain_numden, 'gain_sys_ss:', gain_sys_ss)

        #Compute the gain with a long simulation
        t = linspace(0, 1000, 1000)
        y, _t = step(sys_ss, t)
        gain_sim = y[-1]
        # print('gain_sim:', gain_sim)

        #All gain values must be approximately equal to the known gain
        assert_array_almost_equal([gain_abcd,   gain_zpk,
                                   gain_numden, gain_sys_ss, gain_sim],
                                  [0.026948, 0.026948, 0.026948, 0.026948,
                                   0.026948],
                                  decimal=6)

    def test_step(self, SISO_mats, MIMO_mats, mplcleanup):
        """Test function ``step``."""
        figure(); plot_shape = (1, 3)

        #Test SISO system
        A, B, C, D = SISO_mats
        sys = ss(A, B, C, D)
        #print(sys)
        #print("gain:", dcgain(sys))

        subplot2grid(plot_shape, (0, 0))
        y, t = step(sys)
        plot(t, y)

        subplot2grid(plot_shape, (0, 1))
        T = linspace(0, 2, 100)
        X0 = array([1, 1])
        y, t = step(sys, T, X0)
        plot(t, y)

        # Test output of state vector
        y, t, x = step(sys, return_x=True)

        #Test MIMO system
        A, B, C, D = MIMO_mats
        sys = ss(A, B, C, D)

        subplot2grid(plot_shape, (0, 2))
        y, t = step(sys)
        plot(t, y[:, 0, 0])

    def test_impulse(self, SISO_mats, mplcleanup):
        A, B, C, D = SISO_mats
        sys = ss(A, B, C, D)

        figure()

        #everything automatically
        t, y = impulse(sys)
        plot(t, y, label='Simple Case')

        #supply time and X0
        T = linspace(0, 2, 100)
        X0 = [0.2, 0.2]
        t, y = impulse(sys, T, X0)
        plot(t, y, label='t=0..2, X0=[0.2, 0.2]')

        #Test system with direct feed-though, the function should print a warning.
        D = [[0.5]]
        sys_ft = ss(A, B, C, D)
        with pytest.warns(UserWarning, match="has direct feedthrough"):
            t, y = impulse(sys_ft)
            plot(t, y, label='Direct feedthrough D=[[0.5]]')

    def test_impulse_mimo(self, MIMO_mats, mplcleanup):
        #Test MIMO system
        A, B, C, D = MIMO_mats
        sys = ss(A, B, C, D)
        y, t = impulse(sys)
        plot(t, y[:, :, 0], label='MIMO System')

        legend(loc='best')
        #show()


    def test_initial(self, SISO_mats, MIMO_mats, mplcleanup):
        A, B, C, D = SISO_mats
        sys = ss(A, B, C, D)

        figure(); plot_shape = (1, 3)

        #X0=0 : must produce line at 0
        subplot2grid(plot_shape, (0, 0))
        t, y = initial(sys)
        plot(t, y)

        #X0=[1,1] : produces a spike
        subplot2grid(plot_shape, (0, 1))
        t, y = initial(sys, X0=array([[1], [1]]))
        plot(t, y)

        A, B, C, D = MIMO_mats
        sys = ss(A, B, C, D)
        #X0=[1,1] : produces same spike as above spike
        subplot2grid(plot_shape, (0, 2))
        t, y = initial(sys, X0=[1, 1, 0, 0])
        plot(t, y)

        #show()

    #! Old test; no longer functional?? (RMM, 3 Nov 2012)
    @pytest.mark.skip(
        reason="skipping test_check_convert_shape, need to update test")
    def test_check_convert_shape(self):
        #TODO: check if shape is correct everywhere.
        #Correct input ---------------------------------------------
        #Recognize correct shape
        #Input is array, shape (3,), single legal shape
        arr = _check_convert_array(array([1., 2, 3]), [(3,)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Input is array, shape (3,), two legal shapes
        arr = _check_convert_array(array([1., 2, 3]), [(3,), (1,3)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Input is array, 2D, shape (1,3)
        arr = _check_convert_array(array([[1., 2, 3]]), [(3,), (1,3)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Test special value any
        #Input is array, 2D, shape (1,3)
        arr = _check_convert_array(array([[1., 2, 3]]), [(4,), (1,"any")], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Input is array, 2D, shape (3,1)
        arr = _check_convert_array(array([[1.], [2], [3]]), [(4,), ("any", 1)],
                                   'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Convert array-like objects to arrays
        #Input is matrix, shape (1,3), must convert to array
        arr = _check_convert_array(matrix("1. 2 3"), [(3,), (1,3)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Input is list, shape (1,3), must convert to array
        arr = _check_convert_array([[1., 2, 3]], [(3,), (1,3)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)

        #Special treatment of scalars and zero dimensional arrays:
        #They are converted to an array of a legal shape, filled with the scalar
        #value
        arr = _check_convert_array(5, [(3,), (1,3)], 'Test: ')
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert_array_almost_equal(arr, [5, 5, 5])

        #Squeeze shape
        #Input is array, 2D, shape (1,3)
        arr = _check_convert_array(array([[1., 2, 3]]), [(3,), (1,3)],
                                        'Test: ', squeeze=True)
        assert isinstance(arr, np.ndarray)
        assert not isinstance(arr, matrix)
        assert arr.shape == (3,)  #Shape must be squeezed. (1,3) -> (3,)

        #Erroneous input -----------------------------------------------------
        #test wrong element data types
        #Input is array of functions, 2D, shape (1,3)
        self.assertRaises(TypeError, _check_convert_array(array([[min, max, all]]),
            [(3,), (1,3)], 'Test: ', squeeze=True))

        #Test wrong shapes
        #Input has shape (4,) but (3,) or (1,3) are legal shapes
        self.assertRaises(ValueError, _check_convert_array(array([1., 2, 3, 4]),
            [(3,), (1,3)], 'Test: '))

    @pytest.mark.skip(reason="need to update test")
    def test_lsim(self, SISO_mats, MIMO_mats):
        A, B, C, D = SISO_mats
        sys = ss(A, B, C, D)

        figure(); plot_shape = (2, 2)

        #Test with arrays
        subplot2grid(plot_shape, (0, 0))
        t = linspace(0, 1, 100)
        u = r_[1:1:50j, 0:0:50j]
        y, _t, _x = lsim(sys, u, t)
        plot(t, y, label='y')
        plot(t, u/10, label='u/10')
        legend(loc='best')

        #Test with U=None - uses 2nd algorithm which is much faster.
        subplot2grid(plot_shape, (0, 1))
        t = linspace(0, 1, 100)
        x0 = [-1, -1]
        y, _t, _x = lsim(sys, U=None, T=t, X0=x0)
        plot(t, y, label='y')
        legend(loc='best')

        #Test with U=0, X0=0
        #Correct reaction to zero dimensional special values
        subplot2grid(plot_shape, (0, 1))
        t = linspace(0, 1, 100)
        y, _t, _x = lsim(sys, U=0, T=t, X0=0)
        plot(t, y, label='y')
        legend(loc='best')

        #Test with MIMO system
        subplot2grid(plot_shape, (1, 1))
        A, B, C, D = MIMO_mats
        sys = ss(A, B, C, D)
        t =  array(linspace(0, 1, 100))
        u = array([r_[1:1:50j, 0:0:50j],
                   r_[0:1:50j, 0:0:50j]])
        x0 = [0, 0, 0, 0]
        y, t_out, _x = lsim(sys, u, t, x0)
        plot(t_out, y[0], label='y[0]')
        plot(t_out, y[1], label='y[1]')
        plot(t_out, u[0]/10, label='u[0]/10')
        plot(t_out, u[1]/10, label='u[1]/10')
        legend(loc='best')


        #Test with wrong values for t
        #T is None; - special handling: Value error
        self.assertRaises(ValueError, lsim(sys, U=0, T=None, x0=0))
        #T="hello" : Wrong type
        #TODO: better wording of error messages of ``lsim`` and
        #      ``_check_convert_array``, when wrong type is given.
        #      Current error message is too cryptic.
        self.assertRaises(TypeError, lsim(sys, U=0, T="hello", x0=0))
        #T=0; - T can not be zero dimensional, it determines the size of the
        #       input vector ``U``
        self.assertRaises(ValueError, lsim(sys, U=0, T=0, x0=0))
        #T is not monotonically increasing
        self.assertRaises(ValueError, lsim(sys, U=0, T=[0., 1., 2., 2., 3.], x0=0))
        #show()

    def assert_systems_behave_equal(self, sys1, sys2):
        '''
        Test if the behavior of two LTI systems is equal. Raises ``AssertionError``
        if the systems are not equal.

        Works only for SISO systems.

        Currently computes dcgain, and computes step response.
        '''
        #gain of both systems must be the same
        assert_array_almost_equal(dcgain(sys1), dcgain(sys2))

        #Results of ``step`` simulation must be the same too
        y1, t1 = step(sys1)
        y2, t2 = step(sys2, t1)
        assert_array_almost_equal(y1, y2)

    def test_convert_MIMO_to_SISO(self, SISO_mats, MIMO_mats):
        '''Convert mimo to siso systems'''
        #Test with our usual systems --------------------------------------------
        #SISO PT2 system
        As, Bs, Cs, Ds = SISO_mats
        sys_siso = ss(As, Bs, Cs, Ds)
        #MIMO system that contains two independent copies of the SISO system above
        Am, Bm, Cm, Dm = MIMO_mats
        sys_mimo = ss(Am, Bm, Cm, Dm)
        #    t, y = step(sys_siso)
        #    plot(t, y, label='sys_siso d=0')

        sys_siso_00 = _mimo2siso(sys_mimo, input=0, output=0,
                                         warn_conversion=False)
        sys_siso_11 = _mimo2siso(sys_mimo, input=1, output=1,
                                         warn_conversion=False)
        #print("sys_siso_00 ---------------------------------------------")
        #print(sys_siso_00)
        #print("sys_siso_11 ---------------------------------------------")
        #print(sys_siso_11)

        #gain of converted system and equivalent SISO system must be the same
        self.assert_systems_behave_equal(sys_siso, sys_siso_00)
        self.assert_systems_behave_equal(sys_siso, sys_siso_11)

        #Test with additional systems --------------------------------------------
        #They have crossed inputs and direct feedthrough
        #SISO system
        As =  array([[-81.82, -45.45],
                     [ 10.,    -1.  ]])
        Bs =  array([[9.09],
                     [0.  ]])
        Cs =  array([[0, 0.159]])
        Ds =  array([[0.02]])
        sys_siso = ss(As, Bs, Cs, Ds)
        #    t, y = step(sys_siso)
        #    plot(t, y, label='sys_siso d=0.02')
        #    legend(loc='best')

        #MIMO system
        #The upper left sub-system uses : input 0, output 1
        #The lower right sub-system uses: input 1, output 0
        Am = array([[-81.82, -45.45,   0,      0   ],
                    [ 10,     -1,      0,      0   ],
                    [  0,      0,    -81.82, -45.45],
                    [  0,      0,     10,     -1,  ]])
        Bm = array([[9.09, 0   ],
                    [0   , 0   ],
                    [0   , 9.09],
                    [0   , 0   ]])
        Cm = array([[0, 0,     0, 0.159],
                    [0, 0.159, 0, 0    ]])
        Dm =  array([[0,   0.02],
                     [0.02, 0  ]])
        sys_mimo = ss(Am, Bm, Cm, Dm)


        sys_siso_01 = _mimo2siso(sys_mimo, input=0, output=1,
                                         warn_conversion=False)
        sys_siso_10 = _mimo2siso(sys_mimo, input=1, output=0,
                                         warn_conversion=False)
        # print("sys_siso_01 ---------------------------------------------")
        # print(sys_siso_01)
        # print("sys_siso_10 ---------------------------------------------")
        # print(sys_siso_10)

        #gain of converted system and equivalent SISO system must be the same
        self.assert_systems_behave_equal(sys_siso, sys_siso_01)
        self.assert_systems_behave_equal(sys_siso, sys_siso_10)

