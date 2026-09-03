import numpy as np

from src.models.black_scholes import (
    call_price,
    put_price,
)

from src.pricing.finite_difference import (
    crank_nicolson_price,
)


def test_cn_call_close_to_black_scholes():

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    cn_price = crank_nicolson_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        M=400,
        N=400,
    )

    bs_price = call_price(
        S0,
        K,
        T,
        r,
        sigma,
    )

    assert abs(cn_price - bs_price) < 0.02


def test_cn_put_close_to_black_scholes():

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    cn_price = crank_nicolson_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="put",
        M=400,
        N=400,
    )

    bs_price = put_price(
        S0,
        K,
        T,
        r,
        sigma,
    )

    assert abs(cn_price - bs_price) < 0.02


def test_cn_put_call_parity():

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    call = crank_nicolson_price(
        S0,
        K,
        T,
        r,
        sigma,
        option_type="call",
        M=400,
        N=400,
    )

    put = crank_nicolson_price(
        S0,
        K,
        T,
        r,
        sigma,
        option_type="put",
        M=400,
        N=400,
    )

    lhs = call - put

    rhs = (
        S0
        - K * np.exp(-r * T)
    )

    assert abs(lhs - rhs) < 0.02


def test_grid_refinement_improves_accuracy():

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    bs_price = call_price(
        S0,
        K,
        T,
        r,
        sigma,
    )

    coarse_price = crank_nicolson_price(
        S0,
        K,
        T,
        r,
        sigma,
        option_type="call",
        M=100,
        N=100,
    )

    fine_price = crank_nicolson_price(
        S0,
        K,
        T,
        r,
        sigma,
        option_type="call",
        M=400,
        N=400,
    )

    coarse_error = abs(
        coarse_price - bs_price
    )

    fine_error = abs(
        fine_price - bs_price
    )

    assert fine_error < coarse_error