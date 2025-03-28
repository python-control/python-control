import numpy as np
import pytest

from dataclasses import dataclass
from control.delaylti import delay, tf2dlti, exp, hcat, vcat, mimo_delay
from control.statesp import ss
from control.xferfcn import tf

@dataclass
class JuliaResults:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    tau: np.ndarray
    
s = tf('s')

@pytest.fixture
def simple_siso_tf():
    return tf([1], [1, 1])

@pytest.fixture
def tf_one():
    return tf([1], [1])

@pytest.fixture
def pure_delay():
    return delay(1)


@pytest.fixture
def delay_siso_tf():
    P_tf = 1 / (s + 1)  
    D = delay(1.5)  
    return P_tf * D

@pytest.fixture
def delay_siso_tf2():
    P_tf = 3 / (2*s + 5)
    D = delay(0.5)
    return P_tf * D

@pytest.fixture
def wood_berry():
    P11_tf = 12.8 / (16.7*s + 1)
    P12_tf = -18.9 / (21.0*s + 1)
    P21_tf = 6.6 / (10.9*s + 1)
    P22_tf = -19.4 / (14.4*s + 1)
    G11 = P11_tf * delay(1.0)
    G12 = P12_tf * delay(3.0)
    G21 = P21_tf * delay(7.0)
    G22 = P22_tf * delay(3.0)
    Row1 = hcat(G11, G12)
    Row2 = hcat(G21, G22)
    G_wb = vcat(Row1, Row2) 
    return G_wb

@pytest.fixture
def array_wood_berry():
    G_wb = mimo_delay([
        [12.8 / (16.7*s + 1) * exp(-s),  -18.9 / (21.0*s + 1) * exp(-3*s)],
        [6.6 / (10.9*s + 1) * exp(-7*s), -19.4 / (14.4*s + 1) * exp(-3*s)]
    ])
    return G_wb
    
@pytest.fixture
def julia_wood_berry():
    return JuliaResults(
        A = [[-0.059880239520958084, 0.0, 0.0, 0.0],
            [0.0, -0.047619047619047616, 0.0 , 0.0],
            [0.0, 0.0, -0.09174311926605504, 0.0],
            [0.0, 0.0, 0.0, -0.06944444444444445]],
        B = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        C = [[0.7664670658682635, -0.8999999999999999, 0.0, 0.0],
            [0.0, 0.0, 0.6055045871559632, -1.347222222222222],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]],
        D = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
        tau= [1.0, 3.0, 7.0, 3.0]
    )


class TestConstructors:
    def test_from_ss(self, simple_siso_tf):
        pass

    def test_from_tf(self, tf_one):
        delay_lti_tf_one = tf2dlti(tf_one)
        julia_A = []
        julia_B = []
        julia_C = []
        julia_D = [[1.]]
        assert(np.allclose(delay_lti_tf_one.A, julia_A))
        assert(np.allclose(delay_lti_tf_one.B, julia_B))
        assert(np.allclose(delay_lti_tf_one.C, julia_C))
        assert(np.allclose(delay_lti_tf_one.D, julia_D))

    def test_hcat_vcat(self, wood_berry, julia_wood_berry):
        assert(np.allclose(wood_berry.A, julia_wood_berry.A))
        assert(np.allclose(wood_berry.B, julia_wood_berry.B))
        assert(np.allclose(wood_berry.C, julia_wood_berry.C))
        assert(np.allclose(wood_berry.D, julia_wood_berry.D))
        assert(np.allclose(wood_berry.tau, julia_wood_berry.tau))

    def test_mimo_delay(self, array_wood_berry, julia_wood_berry):
        assert(np.allclose(array_wood_berry.A, julia_wood_berry.A))
        assert(np.allclose(array_wood_berry.B, julia_wood_berry.B))
        assert(np.allclose(array_wood_berry.C, julia_wood_berry.C))
        assert(np.allclose(array_wood_berry.D, julia_wood_berry.D))
        assert(np.allclose(array_wood_berry.tau, julia_wood_berry.tau))

class TestOperators: 
    def test_add(self, delay_siso_tf, delay_siso_tf2):
        G = delay_siso_tf + delay_siso_tf2
        julia_A = [[-1., 0.], [0., -2.5]]
        julia_B = [[0., 1., 0.], [0., 0., 1.]]
        julia_C = [[1., 1.5], [0., 0.], [0., 0.]]
        julia_D = [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]]
        julia_delays = [1.5, 0.5]
        assert(np.allclose(G.A , julia_A))
        assert(np.allclose(G.B , julia_B))
        assert(np.allclose(G.C , julia_C))
        assert(np.allclose(G.D , julia_D))
        assert(np.allclose(G.tau , julia_delays))

    def test_add_constant(self, delay_siso_tf):
        G = delay_siso_tf + 2.5
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[1.], [0.]]
        julia_D = [[2.5, 0.], [1., 0.]]
        julia_delays = [1.5]
        assert(np.allclose(G.A , julia_A))
        assert(np.allclose(G.B , julia_B))
        assert(np.allclose(G.C , julia_C))
        assert(np.allclose(G.D , julia_D))
        assert(np.allclose(G.tau , julia_delays))
        
    def test_mul(self, delay_siso_tf, delay_siso_tf2):
        G = delay_siso_tf * delay_siso_tf2
        julia_A = [[-1., 0.], [0., -2.5]]
        julia_B = [[0., 1., 0.], [0., 0., 1.]]
        julia_C = [[1., 0.], [0., 1.5], [0., 0.]]
        julia_D = [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]
        julia_delays = [1.5, 0.5]
        assert(np.allclose(G.A , julia_A))
        assert(np.allclose(G.B , julia_B))
        assert(np.allclose(G.C , julia_C))
        assert(np.allclose(G.D , julia_D))
        assert(np.allclose(G.tau , julia_delays))

    def test_gain_mul(self, delay_siso_tf):
        G2 = 2 * delay_siso_tf
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[2.], [0.]]
        julia_D = [[0., 0.], [1., 0.]]
        julia_delays = [1.5]
        assert(np.allclose(G2.A , julia_A))
        assert(np.allclose(G2.B , julia_B))
        assert(np.allclose(G2.C , julia_C))
        assert(np.allclose(G2.D , julia_D))
        assert(np.allclose(G2.tau , julia_delays))

    def test_tf_mul_delay(self, simple_siso_tf):
        G = simple_siso_tf * delay(0.5)
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[1.], [0.]]
        julia_D = [[0., 0.], [1., 0.]]
        julia_delays = [0.5]
        assert(np.allclose(G.A , julia_A))
        assert(np.allclose(G.B , julia_B))
        assert(np.allclose(G.C , julia_C))
        assert(np.allclose(G.D , julia_D))
        assert(np.allclose(G.tau , julia_delays))


class TestFeedback:
    def test_feedback_one(self, delay_siso_tf):
        G = delay_siso_tf
        H = G.feedback()
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[1.], [-1.]]
        julia_D = [[0., 0.], [1., 0.]]
        julia_delays = [1.5]
        assert(np.allclose(H.A , julia_A))
        assert(np.allclose(H.B , julia_B))
        assert(np.allclose(H.C , julia_C))
        assert(np.allclose(H.D , julia_D))
        assert(np.allclose(H.tau , julia_delays))

    def test_feedback_tf_one(self, delay_siso_tf, tf_one):
        G = delay_siso_tf
        H = G.feedback(tf_one)
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[1.], [-1.]]
        julia_D = [[0., 0.], [1., 0.]]
        julia_delays = [1.5]
        assert(np.allclose(H.A , julia_A))
        assert(np.allclose(H.B , julia_B))
        assert(np.allclose(H.C , julia_C))
        assert(np.allclose(H.D , julia_D))
        assert(np.allclose(H.tau , julia_delays))

    
    def test_complex_feedback(self, delay_siso_tf):
        G = delay_siso_tf
        H = G.feedback(G)
        julia_A = [[-1., 0.], [0., -1.]]
        julia_B = [[0., 1., 0.], [0., 0., 1.]]
        julia_C = [[1., 0.], [0., -1.], [1., 0.]]
        julia_D = [[0., 0., 0.], [1., 0., 0], [0., 0., 0.]]
        julia_delays = [1.5, 1.5]
        assert(np.allclose(H.A , julia_A))
        assert(np.allclose(H.B , julia_B))
        assert(np.allclose(H.C , julia_C))
        assert(np.allclose(H.D , julia_D))
        assert(np.allclose(H.tau , julia_delays))


class TestPureDelay:
    def test_unit_delay(self, pure_delay):
        A = np.empty((0,0))
        B = np.empty((0,2))
        C = np.empty((2,0))
        D = [[0., 1.], [1., 0.]]
        julia_delays = [1.0]
        assert(np.allclose(pure_delay.A , A))
        assert(np.allclose(pure_delay.B , B))
        assert(np.allclose(pure_delay.C , C))
        assert(np.allclose(pure_delay.D , D))
        assert(np.allclose(pure_delay.tau , julia_delays))

    def test_exp_delay(self):
        G = exp(-2*s)
        A = np.empty((0,0))
        B = np.empty((0,2))
        C = np.empty((2,0))
        D = [[0., 1.], [1., 0.]]
        julia_delays = [2.0]
        assert(np.allclose(G.A , A))
        assert(np.allclose(G.B , B))
        assert(np.allclose(G.C , C))
        assert(np.allclose(G.D , D))
        assert(np.allclose(G.tau , julia_delays))


class TestDelayLTI:
    # comparison with julia controlSystems package

    def test_delay(self):
        tf_test = tf([1], [1,1]) * delay(1)
        A = [[-1]]
        B = [[0,1]]
        C = [[1],[0]]
        D = [[0,0], [1,0]]

        assert(np.allclose(tf_test.P.A , A))
        assert(np.allclose(tf_test.P.B , B))
        assert(np.allclose(tf_test.P.C , C))
        assert(np.allclose(tf_test.P.D , D))

    
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
