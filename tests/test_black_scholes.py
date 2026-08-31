import numpy as np

from src.models.black_scholes import (
    call_price,
    put_price,
    call_delta,
    put_delta,
    gamma,
    vega,
)


def test_put_call_parity():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1e-8)


def test_delta_relation():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    call_d = call_delta(S, K, T, r, sigma)
    put_d = put_delta(S, K, T, r, sigma)

    assert np.isclose(call_d - put_d, 1.0, atol=1e-8)


def test_delta_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    h = 1e-4

    numerical_delta = (
        call_price(S + h, K, T, r, sigma)
        - call_price(S - h, K, T, r, sigma)
    ) / (2 * h)

    analytical_delta = call_delta(S, K, T, r, sigma)

    assert np.isclose(
        numerical_delta,
        analytical_delta,
        atol=1e-5
    )


def test_gamma_finite_difference():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    h = 1e-2

    numerical_gamma = (
        call_price(S + h, K, T, r, sigma)
        - 2 * call_price(S, K, T, r, sigma)
        + call_price(S - h, K, T, r, sigma)
    ) / (h ** 2)

    analytical_gamma = gamma(S, K, T, r, sigma)

    assert np.isclose(
        numerical_gamma,
        analytical_gamma,
        atol=1e-4
    )


def test_vega_positive():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    assert vega(S, K, T, r, sigma) > 0
