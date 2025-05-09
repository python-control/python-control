import numpy as np
import pytest
import matplotlib.pyplot as plt

from control.xferfcn import tf
from control.delaylti import delay, exp, mimo_delay
from control.julia.utils import julia_json

s = tf('s')

@pytest.fixture
def simple_siso_tf():
    return tf([1], [1, 1])

@pytest.fixture
def delay_siso_tf():
    P_tf = 1 / (s + 1)
    D = delay(1.5)
    return P_tf * D

@pytest.fixture
def wood_berry():
    # Construct a 2x2 MIMO system with delays
    G_wb = mimo_delay([
        [12.8 / (16.7 * s + 1) * exp(-s),  -18.9 / (21.0 * s + 1) * exp(-3 * s)],
        [6.6 / (10.9 * s + 1) * exp(-7 * s), -19.4 / (14.4 * s + 1) * exp(-3 * s)]
    ])
    return G_wb


class TestTimeResp:
    def test_siso_delayed_step_response(self, delay_siso_tf, simple_siso_tf):
        from control.timeresp import step_response
        timepts = np.linspace(0, 10, 1001)
        step = step_response(simple_siso_tf, timepts=timepts)
        delay_step = step_response(delay_siso_tf, timepts=timepts)
        # Construct a manually delayed step response by shifting the step response
        hand_delayed_step = np.zeros_like(step.y[0][0])
        count = 0
        for i, t in enumerate(step.t):
            if t >= 1.5:
                hand_delayed_step[i] = step.y[0][0][count]
                count += 1
        plt.figure()
        plt.plot(delay_step.y[0][0] - hand_delayed_step)
        plt.legend()
        plt.show()
        assert np.allclose(delay_step.y[0][0], hand_delayed_step)

    def test_siso_delayed_step_response_mos(self, delay_siso_tf, simple_siso_tf):
        from control.timeresp import step_response
        timepts = np.linspace(0, 10, 1001)
        step = step_response(simple_siso_tf, timepts=timepts)
        delay_step = step_response(delay_siso_tf, timepts=timepts)
        # Construct a manually delayed step response by shifting the step response
        hand_delayed_step = np.zeros_like(step.y[0][0])
        count = 0
        for i, t in enumerate(step.t):
            if t >= 1.5:
                hand_delayed_step[i] = step.y[0][0][count]
                count += 1

        plt.figure()
        plt.plot(delay_step.y[0][0] - hand_delayed_step)
        plt.legend()
        plt.show()
        assert np.allclose(delay_step.y[0][0], hand_delayed_step, atol=1e-5)

    # wood berry step response compared to julia
    def test_mimo_step_response(self, wood_berry, plot=False):
        from control.timeresp import step_response
        import matplotlib.pyplot as plt
        timepts = np.linspace(0, 100, 1001)
        step = step_response(wood_berry, timepts=timepts)
        print(step.y[0].shape)

        plot = True
        if plot:
            plt.figure()
            plt.plot(step.y[0][0] - julia_json["TestTimeResp"]["test_mimo_step_response"]["y11"])
            plt.plot(step.y[1][0] - julia_json["TestTimeResp"]["test_mimo_step_response"]["y21"])
            plt.plot(step.y[0][1] - julia_json["TestTimeResp"]["test_mimo_step_response"]["y12"])
            plt.plot(step.y[1][1] - julia_json["TestTimeResp"]["test_mimo_step_response"]["y22"])
            plt.title("Step response")
            plt.show()

        # Precision is currently between 1e-5 and 1e-6 compared to julia solver for mimo
        assert np.allclose(step.y[0][0], julia_json["TestTimeResp"]["test_mimo_step_response"]["y11"], atol=1e-5)
        assert np.allclose(step.y[0][1], julia_json["TestTimeResp"]["test_mimo_step_response"]["y12"], atol=1e-5)
        assert np.allclose(step.y[1][1], julia_json["TestTimeResp"]["test_mimo_step_response"]["y22"], atol=1e-5)
        assert np.allclose(step.y[1][0], julia_json["TestTimeResp"]["test_mimo_step_response"]["y21"], atol=1e-5)
        
    def test_forced_response(self, delay_siso_tf, simple_siso_tf, plot=False):
        from control.timeresp import forced_response

        timepts = np.linspace(0, 10, 1001)    
        inputs = np.sin(timepts)
        resp = forced_response(simple_siso_tf, timepts=timepts, inputs=inputs)
        delay_resp = forced_response(delay_siso_tf, timepts=timepts, inputs=inputs)
        hand_delayed_resp = np.zeros_like(resp.y[0])
        count = 0
        for i, t in enumerate(resp.t):
            if t >= 1.5:
                hand_delayed_resp[i] = resp.y[0][count]
                count += 1

        # Optionally, inspect the plot:
        #plot = True
        if plot:
            plt.figure()
            plt.plot(resp.t, inputs, label="input")
            plt.plot(resp.t, resp.y[0], label="response")
            plt.plot(resp.t, delay_resp.y[0], label="delay LTI")
            plt.plot(resp.t, hand_delayed_resp, label="hand delay")
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(resp.t, delay_resp.y[0] - hand_delayed_resp)
            plt.show()

        assert np.allclose(delay_resp.y[0], hand_delayed_resp, atol=1e-5)

    def test_mimo_forced_response(self, wood_berry, plot=False):
        from control.timeresp import forced_response
        timepts = np.linspace(0, 100, 10001)
        inputs = np.array([np.sin(timepts), np.cos(timepts)])
        resp = forced_response(wood_berry, timepts=timepts, inputs=inputs)

        resp_wb_11 = forced_response(12.8 / (16.7 * s + 1), timepts=timepts, inputs=inputs[0])
        resp_wb_12 = forced_response(-18.9 / (21.0 * s + 1), timepts=timepts, inputs=inputs[1])
        resp_wb_21 = forced_response(6.6 / (10.9 * s + 1), timepts=timepts, inputs=inputs[0])
        resp_wb_22 = forced_response(-19.4 / (14.4 * s + 1), timepts=timepts, inputs=inputs[1])
        
        hand_delayed_resp_y1 = np.zeros_like(resp.y[0])
        hand_delayed_resp_y2 = np.zeros_like(resp.y[1])
        count11 = 0
        count12 = 0
        count21 = 0
        count22 = 0
        for i, t in enumerate(resp.t):
            if t >= 1:
                hand_delayed_resp_y1[i] = resp_wb_11.y[0][count11]
                count11 += 1
                if t >= 3:
                    hand_delayed_resp_y1[i] += resp_wb_12.y[0][count12]
                    count12 += 1
                    hand_delayed_resp_y2[i] += resp_wb_22.y[0][count22]
                    count22 += 1
                    if t >= 7:
                        hand_delayed_resp_y2[i] += resp_wb_21.y[0][count21]
                        count21 += 1
        #plot = True     
        if plot:          
            plt.figure()
            plt.plot(resp.t, resp.y[0] - hand_delayed_resp_y1, label="y1 - hand y1")
            plt.plot(resp.t, resp.y[1] - hand_delayed_resp_y2, label="y2 - hand y2")

            plt.legend()
            plt.show()

        assert np.allclose(resp.y[0], hand_delayed_resp_y1, atol=1e-5)
        assert np.allclose(resp.y[1], hand_delayed_resp_y2, atol=1e-5)

