import numpy as np

from src.comparison.model_comparison import (
    black_scholes_call_price_vector,
    calibrate_black_scholes_constant_vol,
)


def test_black_scholes_vector_prices_positive():

    prices = (
        black_scholes_call_price_vector(
            sigma=0.20,
            S0=100.0,
            r=0.05,
            strikes=np.array(
                [
                    90.0,
                    100.0,
                    110.0,
                ]
            ),
            maturities=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
        )
    )

    assert prices.shape == (3,)

    assert np.all(
        prices > 0.0
    )


def test_black_scholes_calibration_recovers_sigma():

    true_sigma = 0.25

    strikes = np.array(
        [
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
        ]
    )

    maturities = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )

    market_prices = (
        black_scholes_call_price_vector(
            sigma=true_sigma,
            S0=100.0,
            r=0.05,
            strikes=strikes,
            maturities=maturities,
        )
    )

    result = (
        calibrate_black_scholes_constant_vol(
            market_prices=market_prices,
            S0=100.0,
            r=0.05,
            strikes=strikes,
            maturities=maturities,
        )
    )

    assert result["success"]

    assert np.isclose(
        result["sigma"],
        true_sigma,
        atol=1e-6,
    )

    assert result["rmse"] < 1e-7