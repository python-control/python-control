import numpy as np
import pytest

from control.delaylti import (
    delay, tf2dlti, exp, hcat,
    vcat, mimo_delay, ss2dlti
)
from control.statesp import ss
from control.xferfcn import tf
from control.julia.utils import julia_json, assert_delayLTI

s = tf("s")


@pytest.fixture
def simple_siso_tf():
    return tf([1], [1, 1])


@pytest.fixture
def tf_one():
    return tf([1], [1])


@pytest.fixture
def delay_siso_tf():
    P_tf = 1 / (s + 1)
    D = delay(1.5)
    return P_tf * D


@pytest.fixture
def delay_siso_tf2():
    P_tf = 3 / (2 * s + 5)
    D = delay(0.5)
    return P_tf * D


@pytest.fixture
def wood_berry():
    # Construct a 2x2 MIMO system with delays
    G_wb = mimo_delay(
        [
            [
                12.8 / (16.7 * s + 1) * exp(-s),
                -18.9 / (21.0 * s + 1) * exp(-3 * s)
            ],
            [
                6.6 / (10.9 * s + 1) * exp(-7 * s),
                -19.4 / (14.4 * s + 1) * exp(-3 * s)
            ]
        ]
    )
    return G_wb


class TestConstructors:
    @pytest.mark.parametrize("key", ["simple_siso_tf", "tf_one"])
    def test_tf2dlti(self, request, key):
        tf = request.getfixturevalue(key)
        delay_lti = tf2dlti(tf)
        julia_results = julia_json["TestConstructors"]["test_tf2dlti"]
        assert_delayLTI(delay_lti, julia_results[key])

    @pytest.mark.parametrize("tau", [1, 1.5, 10])
    def test_delay_function(self, tau):
        dlti = delay(tau)
        julia_results = julia_json["TestConstructors"]["test_delay_function"]
        assert_delayLTI(dlti, julia_results[str(tau)])

    @pytest.mark.parametrize("tau", [1, 1.5, 10])
    def test_exp_delay(self, tau):
        G = exp(-tau * s)
        julia_results = julia_json["TestConstructors"]["test_exp_delay"]
        assert_delayLTI(G, julia_results[str(tau)])

    @pytest.mark.parametrize("tau", [1, 1.5, 10])
    def test_two_ways_delay(self, tau):
        delay_exp = exp(-tau * s)
        delay_pure = delay(tau)
        assert delay_exp == delay_pure

    def test_siso_delay(self, delay_siso_tf):
        assert_delayLTI(
            delay_siso_tf, julia_json["TestConstructors"]["test_siso_delay"]
        )

    def test_build_wood_berry(self, wood_berry):
        assert_delayLTI(
            wood_berry, julia_json["TestConstructors"]["test_build_wood_berry"]
        )


class TestOperators:
    def test_siso_add(self, delay_siso_tf, delay_siso_tf2):
        assert_delayLTI(
            delay_siso_tf + delay_siso_tf2,
            julia_json["TestOperators"]["test_siso_add"]
        )

    def test_siso_add_constant(self, delay_siso_tf):
        assert_delayLTI(
            delay_siso_tf + 2.5,
            julia_json["TestOperators"]["test_siso_add_constant"]
        )

    def test_siso_sub(self, delay_siso_tf, delay_siso_tf2):
        assert_delayLTI(
            delay_siso_tf - delay_siso_tf2,
            julia_json["TestOperators"]["test_siso_sub"]
        )

    def test_siso_sub_constant(self, delay_siso_tf):
        assert_delayLTI(
            delay_siso_tf - 2.5,
            julia_json["TestOperators"]["test_siso_sub_constant"]
        )

    def test_siso_mul(self, delay_siso_tf, delay_siso_tf2):
        assert_delayLTI(
            delay_siso_tf * delay_siso_tf2,
            julia_json["TestOperators"]["test_siso_mul"]
        )

    def test_siso_mul_constant(self, delay_siso_tf):
        assert_delayLTI(
            delay_siso_tf * 2.0,
            julia_json["TestOperators"]["test_siso_mul_constant"]
        )

    def test_siso_rmul_constant(self, delay_siso_tf):
        assert_delayLTI(
            2.0 * delay_siso_tf,
            julia_json["TestOperators"]["test_siso_rmul_constant"]
        )

    def test_mimo_add(self, wood_berry):
        assert_delayLTI(
            wood_berry + wood_berry,
            julia_json["TestOperators"]["test_mimo_add"]
        )

    def test_mimo_add_constant(self, wood_berry):
        assert_delayLTI(
            wood_berry + 2.7,
            julia_json["TestOperators"]["test_mimo_add_constant"]
        )

    def test_mimo_mul(self, wood_berry):
        assert_delayLTI(
            wood_berry * wood_berry,
            julia_json["TestOperators"]["test_mimo_mul"]
        )

    def test_mimo_mul_constant(self, wood_berry):
        assert_delayLTI(
            wood_berry * 2.7,
            julia_json["TestOperators"]["test_mimo_mul_constant"]
        )


class TestDelayLtiMethods:
    @pytest.mark.parametrize("key", ["empty", "tf_one", "delay_siso_tf"])
    def test_feedback(self, request, key, delay_siso_tf):
        G = delay_siso_tf
        julia_results = julia_json["TestDelayLtiMethods"]["test_feedback"]
        if key == "empty":
            H = G.feedback()
        else:
            tf = request.getfixturevalue(key)
            H = G.feedback(tf)
        assert_delayLTI(H, julia_results[key])

    def test_mimo_feedback(self, wood_berry):
        G = wood_berry.feedback(wood_berry)
        assert_delayLTI(
            G, julia_json["TestDelayLtiMethods"]["test_mimo_feedback"]
        )

    def test_siso_freq_resp(self, delay_siso_tf):
        from control.lti import frequency_response

        w = np.logspace(-2, 2, 100, base=10)
        resp = frequency_response(delay_siso_tf, w).complex
        assert np.allclose(
            np.real(resp),
            julia_json["TestDelayLtiMethods"]["test_siso_freq_resp"]["real"],
        )
        assert np.allclose(
            np.imag(resp),
            julia_json["TestDelayLtiMethods"]["test_siso_freq_resp"]["imag"],
        )

    def test_tito_freq_response(self, wood_berry):
        from control.lti import frequency_response

        w = np.logspace(-2, 2, 100, base=10)
        resp = frequency_response(wood_berry, w).complex
        assert np.allclose(
            np.real(resp[0][0]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r11"]["real"],
        )
        assert np.allclose(
            np.imag(resp[0][0]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r11"]["imag"],
        )

        assert np.allclose(
            np.real(resp[0][1]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r12"]["real"],
        )
        assert np.allclose(
            np.imag(resp[0][1]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r12"]["imag"],
        )

        assert np.allclose(
            np.real(resp[1][0]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r21"]["real"],
        )
        assert np.allclose(
            np.imag(resp[1][0]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r21"]["imag"],
        )

        assert np.allclose(
            np.real(resp[1][1]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r22"]["real"],
        )
        assert np.allclose(
            np.imag(resp[1][1]),
            julia_json["TestDelayLtiMethods"]
            ["test_tito_freq_response"]["r22"]["imag"],
        )

    def test_call(self):
        sys = tf([1], [1, 1])
        dlti = tf2dlti(sys) * delay(1)
        freq_resp = dlti(1j * 2 * np.pi)
        expected = 1 / (2 * np.pi * 1j + 1)
        assert np.allclose(freq_resp, expected)

    @pytest.mark.parametrize(
        "cat_func, expected_A, expected_tau",
        [(vcat, [[-1, 0], [0, -1]], [1, 2]), (hcat, [[-1, 0], [0, -1]], [1, 2])],
    )
    def test_cat(self, cat_func, expected_A, expected_tau):
        # Create two simple delayed state-space systems
        sys1 = ss([[-1]], [[1]], [[1]], [[0]])
        dlti1 = ss2dlti(sys1) * delay(1)
        dlti2 = ss2dlti(sys1) * delay(2)
        dlti_cat = cat_func(dlti1, dlti2)
        assert np.allclose(dlti_cat.P.A, np.array(expected_A))
        assert np.allclose(dlti_cat.tau, np.array(expected_tau))

    def test_issiso(self, delay_siso_tf, wood_berry):
        assert delay_siso_tf.issiso()
        assert not wood_berry.issiso()
