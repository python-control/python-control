# -*- coding: utf-8 -*-
"""
Tests for sysnorm module.

Created on Mon Jan  8 11:31:46 2024
Author: Henrik Sandberg
"""

import control as ct
import numpy as np
import pytest


def test_norm_1st_order_stable_system():
    """First-order stable continuous-time system"""
    s = ct.tf('s')

    G1 = 1/(s+1)
    assert np.allclose(ct.norm(G1, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(G1, p=2), 0.707106781186547) # Comparison to norm computed in MATLAB

    Gd1 = ct.sample_system(G1, 0.1)
    assert np.allclose(ct.norm(Gd1, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(Gd1, p=2), 0.223513699524858) # Comparison to norm computed in MATLAB


def test_norm_1st_order_unstable_system():
    """First-order unstable continuous-time system"""
    s = ct.tf('s')

    G2 = 1/(1-s)
    assert np.allclose(ct.norm(G2, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="System is unstable!"):
        assert ct.norm(G2, p=2) == float('inf') # Comparison to norm computed in MATLAB

    Gd2 = ct.sample_system(G2, 0.1)
    assert np.allclose(ct.norm(Gd2, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="System is unstable!"):
        assert ct.norm(Gd2, p=2) == float('inf') # Comparison to norm computed in MATLAB

def test_norm_2nd_order_system_imag_poles():
    """Second-order continuous-time system with poles on imaginary axis"""
    s = ct.tf('s')

    G3 = 1/(s**2+1)
    with pytest.warns(UserWarning, match="Poles close to, or on, the imaginary axis."):
        assert ct.norm(G3, p='inf') == float('inf') # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="Poles close to, or on, the imaginary axis."):
        assert ct.norm(G3, p=2) == float('inf') # Comparison to norm computed in MATLAB

    Gd3 = ct.sample_system(G3, 0.1)
    with pytest.warns(UserWarning, match="Poles close to, or on, the complex unit circle."):
        assert ct.norm(Gd3, p='inf') == float('inf') # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="Poles close to, or on, the complex unit circle."):
        assert ct.norm(Gd3, p=2) == float('inf') # Comparison to norm computed in MATLAB

def test_norm_3rd_order_mimo_system():
    """Third-order stable MIMO continuous-time system"""
    A = np.array([[-1.017041847539126,  -0.224182952826418,   0.042538079149249],
                  [-0.310374015319095,  -0.516461581407780,  -0.119195790221750],
                  [-1.452723568727942,   1.7995860837102088,  -1.491935830615152]])
    B = np.array([[0.312858596637428,  -0.164879019209038],
                  [-0.864879917324456,   0.627707287528727],
                  [-0.030051296196269,   1.093265669039484]])
    C = np.array([[1.109273297614398,   0.077359091130425,  -1.113500741486764],
                  [-0.863652821988714,  -1.214117043615409,  -0.006849328103348]])
    D = np.zeros((2,2))
    G4 = ct.ss(A,B,C,D) # Random system generated in MATLAB
    assert np.allclose(ct.norm(G4, p='inf'), 4.276759162964244) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(G4, p=2), 2.237461821810309) # Comparison to norm computed in MATLAB

    Gd4 = ct.sample_system(G4, 0.1)
    assert np.allclose(ct.norm(Gd4, p='inf'), 4.276759162964228) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(Gd4, p=2), 0.707434962289554) # Comparison to norm computed in MATLAB
