import numpy as np
import pytest
import matplotlib.pyplot as plt

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

    def test_mimo_delay(self, wood_berry, julia_wood_berry):
        assert(np.allclose(wood_berry.A, julia_wood_berry.A))
        assert(np.allclose(wood_berry.B, julia_wood_berry.B))
        assert(np.allclose(wood_berry.C, julia_wood_berry.C))
        assert(np.allclose(wood_berry.D, julia_wood_berry.D))
        assert(np.allclose(wood_berry.tau, julia_wood_berry.tau))

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


class TestFreqResp:
    def test_siso_freq_resp(self, delay_siso_tf):
        from control.lti import frequency_response
        w = np.logspace(-2, 2, 10, base=10)
        resp = delay_siso_tf.frequency_response(w)
        fun_resp = frequency_response(delay_siso_tf, w)
        julia_freq_resp = np.array([
            0.9996375439798979 - 0.024995812946127068*1j,
            0.9971959288676081 - 0.06947384247065888*1j,
            0.9784258086709292 - 0.1916345960427577*1j,
            0.8407906760670428 - 0.4987123630856*1j,
            0.11248651027122411 - 0.8502796738335039*1j,
            -0.47530358539375506 + 0.19610650928623855*1j,
            -0.09481872294498477 - 0.18805963874096887*1j,
            -0.033328025868165176 - 0.06963017462205673*1j,
            0.01265424121150209 + 0.02476963547555173*1j,
            0.007217967580181465 - 0.006920328388981937*1j
        ], dtype=complex)
        assert(np.allclose(np.array(resp.complex), julia_freq_resp))
        assert(np.allclose(np.array(fun_resp.complex), julia_freq_resp))

    def test_bode_plot(self, delay_siso_tf):
        from control.freqplot import bode_plot
        import matplotlib.pyplot as plt
        w = np.logspace(-2, 2, 100, base=10)
        #bode_plot(delay_siso_tf, w)
        #plt.show() comparison with julia notebook

    def test_mimo_freq_resp(self, wood_berry):
        from control.lti import frequency_response
        w = np.logspace(-2, 2, 10, base=10)
        fresp = frequency_response(wood_berry, w)
        julia_freq_resp = np.array([
            [[12.4313-2.20402*1j, 10.3867-5.1827*1j, 4.29712-6.54633*1j,  0.190665-3.42239*1j, -0.609851-1.11652*1j,
              -0.458323+0.0281868*1j, 0.164539+0.0138042*1j, -0.0200415-0.0558575*1j, 0.0209361+0.0040665*1j, 0.00388508-0.00660706*1j],
             [-17.9795+4.34262*1j, -13.3537+9.37895*1j, -3.10627+9.40135*1j, 1.69596+3.70969*1j, 1.48013-0.221262*1j,
              -0.520719+0.140405*1j, 0.189103+0.0428153*1j, 0.0602249+0.035053*1j, 0.0210585+0.0135532*1j, -0.00899771-0.000203154*1j]],
            [[6.45681-1.16541*1j, 5.57492-2.9683*1j, 1.62412-4.7752*1j, -2.31095-1.16016*1j, 0.783909+0.618326*1j,
              0.293676-0.212413*1j, -0.113486-0.0642809*1j, -0.0303737+0.0357106*1j, -0.00395567-0.0163775*1j, -0.00329842+0.00507779*1j],
             [-18.9152+3.30571*1j, -16.0995+8.06846*1j, -6.19676+11.3748*1j, 1.954+5.6218*1j, 2.2183-0.250236*1j,
              -0.781793+0.199879*1j, 0.282749+0.0654171*1j, 0.090062+0.0526232*1j, 0.0315105+0.0203071*1j, -0.0134687-0.000307044*1j]]
        ],dtype=complex)
        assert(np.allclose(fresp.complex, julia_freq_resp))



class TestTimeResp:
    def test_siso_step_response(self, delay_siso_tf):
        from control.timeresp import step_response
        timepts1 = np.arange(0, 11, 1)
        t1, y1 = step_response(delay_siso_tf, timepts=timepts1)
        timepts2 = np.arange(0, 10.1, 0.1)
        t2, y2 = step_response(delay_siso_tf, timepts=timepts2)
        julia_resp = [0.0, 0.0, 0.3934701171707952, 0.7768701219836066, 0.917915094373695, 0.9698026491618251, 0.9888910143620181, 0.9959132295791787, 0.9984965605514581, 0.9994469148973136, 0.9997965302422651]
        plt.figure()
        plt.plot(t1, y1, label="python step 1")
        plt.plot(timepts1, julia_resp, label="julia")
        plt.plot(t2, y2, label="python step 0.1")
        plt.legend()
        plt.show()
        



    def test_forced_response(self, delay_siso_tf):
        from control.timeresp import forced_response

        timepts = np.arange(0, 10.1, 0.1)
        input1 = np.zeros(101)
        input1[31:] = 1
        resp = forced_response(delay_siso_tf, timepts=timepts, inputs=input1)
        t, y = resp.t, resp.y[0] 
        #plt.figure()
        #plt.plot(t, input1)
        #plt.plot(t, y)
        #plt.show()