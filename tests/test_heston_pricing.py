import numpy as np

from src.pricing.heston_pricing import (
    heston_characteristic_function,
    heston_call_price,
    heston_put_price,
    heston_monte_carlo_price,
)


def test_heston_characteristic_function_at_zero():

    phi_zero = (
        heston_characteristic_function(
            u=0.0,
            S0=100.0,
            T=1.0,
            r=0.05,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.30,
            rho=-0.70,
        )
    )

    assert np.isclose(
        phi_zero,
        1.0 + 0.0j,
        atol=1e-12,
    )


def test_heston_characteristic_function_first_moment():

    S0 = 100.0
    T = 1.0
    r = 0.05

    phi_minus_i = (
        heston_characteristic_function(
            u=-1j,
            S0=S0,
            T=T,
            r=r,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.30,
            rho=-0.70,
        )
    )

    expected_stock = (
        S0
        * np.exp(r * T)
    )

    assert np.isclose(
        np.real(phi_minus_i),
        expected_stock,
        rtol=1e-10,
        atol=1e-10,
    )

    assert abs(
        np.imag(phi_minus_i)
    ) < 1e-10

def test_heston_call_price_no_arbitrage_bounds():

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05

    call = heston_call_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.30,
        rho=-0.70,
    )

    lower_bound = max(
        S0 - K * np.exp(-r * T),
        0.0,
    )

    upper_bound = S0

    assert (
        lower_bound
        <= call
        <= upper_bound
    )


def test_heston_put_call_parity():

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05

    parameters = {
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.30,
        "rho": -0.70,
    }

    call = heston_call_price(
        **parameters
    )

    put = heston_put_price(
        **parameters
    )

    lhs = call - put

    rhs = (
        S0
        - K * np.exp(-r * T)
    )

    assert np.isclose(
        lhs,
        rhs,
        atol=1e-10,
    )


def test_heston_monte_carlo_standard_error_positive():

    price, standard_error = (
        heston_monte_carlo_price(
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.30,
            rho=-0.70,
            option_type="call",
            n_paths=2_000,
            n_steps=100,
            seed=42,
        )
    )

    assert price > 0.0
    assert standard_error > 0.0


def test_heston_monte_carlo_seed_reproducibility():

    parameters = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "v0": 0.04,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.30,
        "rho": -0.70,
        "option_type": "call",
        "n_paths": 2_000,
        "n_steps": 100,
        "seed": 42,
    }

    price_1, se_1 = (
        heston_monte_carlo_price(
            **parameters
        )
    )

    price_2, se_2 = (
        heston_monte_carlo_price(
            **parameters
        )
    )

    assert np.isclose(
        price_1,
        price_2,
    )

    assert np.isclose(
        se_1,
        se_2,
    )