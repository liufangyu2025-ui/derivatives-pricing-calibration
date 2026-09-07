import numpy as np
import pytest

from src.calibration.heston_calibration import (
    heston_call_price_vector,
    heston_price_residuals,
)


def test_weighted_residuals_use_sqrt_weights():
    parameters = np.array(
        [0.04, 2.0, 0.04, 0.30, -0.70],
        dtype=float,
    )

    S0 = 100.0
    r = 0.05
    strikes = np.array([95.0, 105.0])
    maturities = np.array([0.5, 1.0])

    model_prices = heston_call_price_vector(
        parameters=parameters,
        S0=S0,
        r=r,
        strikes=strikes,
        maturities=maturities,
        integration_limit=100.0,
    )

    market_prices = model_prices - np.array([0.10, -0.20])
    weights = np.array([4.0, 9.0])

    unweighted = heston_price_residuals(
        parameters=parameters,
        market_prices=market_prices,
        S0=S0,
        r=r,
        strikes=strikes,
        maturities=maturities,
        integration_limit=100.0,
    )

    weighted = heston_price_residuals(
        parameters=parameters,
        market_prices=market_prices,
        S0=S0,
        r=r,
        strikes=strikes,
        maturities=maturities,
        weights=weights,
        integration_limit=100.0,
    )

    assert np.allclose(
        weighted,
        unweighted * np.sqrt(weights),
        atol=1e-12,
    )


def test_nonpositive_weights_raise_value_error():
    parameters = np.array(
        [0.04, 2.0, 0.04, 0.30, -0.70],
        dtype=float,
    )

    S0 = 100.0
    r = 0.05
    strikes = np.array([95.0, 105.0])
    maturities = np.array([0.5, 1.0])

    market_prices = heston_call_price_vector(
        parameters=parameters,
        S0=S0,
        r=r,
        strikes=strikes,
        maturities=maturities,
        integration_limit=100.0,
    )

    with pytest.raises(ValueError):
        heston_price_residuals(
            parameters=parameters,
            market_prices=market_prices,
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
            weights=np.array([1.0, 0.0]),
            integration_limit=100.0,
        )
