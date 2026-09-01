import numpy as np

from src.models.black_scholes import call_price, put_price
from src.pricing.monte_carlo import (
    simulate_terminal_prices,
    monte_carlo_price,
)


def test_terminal_prices_positive():
    ST = simulate_terminal_prices(
        S0=100,
        T=1,
        r=0.05,
        sigma=0.20,
        n_paths=10_000,
        seed=42,
    )

    assert np.all(ST > 0)


def test_monte_carlo_call_close_to_black_scholes():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    mc_price, se = monte_carlo_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        n_paths=200_000,
        seed=42,
    )

    bs_price = call_price(S0, K, T, r, sigma)

    assert abs(mc_price - bs_price) < 3 * se


def test_monte_carlo_put_close_to_black_scholes():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    mc_price, se = monte_carlo_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        n_paths=200_000,
        seed=42,
    )

    bs_price = put_price(S0, K, T, r, sigma)

    assert abs(mc_price - bs_price) < 3 * se


def test_standard_error_positive():
    _, se = monte_carlo_price(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.20,
        n_paths=10_000,
        seed=42,
    )

    assert se > 0