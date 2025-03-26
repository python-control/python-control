import numpy as np
import pytest

from control.delaylti import delay, tf2dlti
from control.statesp import ss
from control.xferfcn import tf

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

    #@pytest.mark.skip()
    def test_feedback_tf_one(self, delay_siso_tf, tf_one):
        G = delay_siso_tf
        H = G.feedback(tf_one)
        julia_A = [[-1.]]
        julia_B = [[0., 1.]]
        julia_C = [[1.], [-1.]]
        julia_D = [[0., 0.], [1., 0.]]
        julia_delays = [1.5]

        print("HA", H.A)
        print("HB", H.B)
        print("HC", H.C)
        print("HD", H.D)
        print("Htau", H.tau)
        
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
        A = []
        B = np.zeros((0,2))
        C = []
        D = [[0., 1.], [1., 0.]]

        assert(np.allclose(pure_delay.A , A))
        assert(np.allclose(pure_delay.B , B))
        assert(np.allclose(pure_delay.C , C))
        assert(np.allclose(pure_delay.D , D))



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
