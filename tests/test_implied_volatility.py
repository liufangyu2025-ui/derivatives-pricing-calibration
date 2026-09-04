import pytest

from src.models.black_scholes import (
    call_price,
    put_price,
)

from src.volatility.implied_volatility import (
    implied_volatility,
)


def test_recover_call_implied_volatility():
    S = 100
    K = 100
    T = 1.0
    r = 0.05

    sigma_true = 0.20

    market_price = call_price(
        S,
        K,
        T,
        r,
        sigma_true,
    )

    sigma_implied = implied_volatility(
        market_price,
        S,
        K,
        T,
        r,
        option_type="call",
    )

    assert abs(
        sigma_implied - sigma_true
    ) < 1e-8


def test_recover_put_implied_volatility():
    S = 100
    K = 110
    T = 0.75
    r = 0.03

    sigma_true = 0.35

    market_price = put_price(
        S,
        K,
        T,
        r,
        sigma_true,
    )

    sigma_implied = implied_volatility(
        market_price,
        S,
        K,
        T,
        r,
        option_type="put",
    )

    assert abs(
        sigma_implied - sigma_true
    ) < 1e-8


def test_invalid_call_price_above_upper_bound():
    S = 100
    K = 100
    T = 1.0
    r = 0.05

    with pytest.raises(ValueError):
        implied_volatility(
            market_price=120,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type="call",
        )


def test_invalid_option_type():
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=10,
            S=100,
            K=100,
            T=1.0,
            r=0.05,
            option_type="invalid",
        )