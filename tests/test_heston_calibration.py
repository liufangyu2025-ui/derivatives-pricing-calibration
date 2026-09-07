import numpy as np
import pytest

from src.calibration.heston_calibration import (
    parameter_vector_to_dict,
    heston_call_price_vector,
    heston_price_residuals,
)


def test_parameter_vector_to_dict():

    parameters = np.array(
        [
            0.04,
            2.0,
            0.04,
            0.30,
            -0.70,
        ]
    )

    result = (
        parameter_vector_to_dict(
            parameters
        )
    )

    assert result["v0"] == 0.04
    assert result["kappa"] == 2.0
    assert result["theta"] == 0.04
    assert result["xi"] == 0.30
    assert result["rho"] == -0.70


def test_parameter_vector_wrong_length():

    with pytest.raises(
        ValueError
    ):
        parameter_vector_to_dict(
            [
                0.04,
                2.0,
                0.04,
            ]
        )


def test_true_parameters_give_zero_residuals():

    true_parameters = np.array(
        [
            0.04,
            2.0,
            0.04,
            0.30,
            -0.70,
        ]
    )

    S0 = 100.0
    r = 0.05

    strikes = np.array(
        [
            90.0,
            100.0,
            110.0,
            100.0,
        ]
    )

    maturities = np.array(
        [
            0.5,
            0.5,
            0.5,
            1.0,
        ]
    )

    synthetic_prices = (
        heston_call_price_vector(
            parameters=true_parameters,
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
            integration_limit=100.0,
        )
    )

    residuals = (
        heston_price_residuals(
            parameters=true_parameters,
            market_prices=(
                synthetic_prices
            ),
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
            integration_limit=100.0,
        )
    )

    assert np.allclose(
        residuals,
        0.0,
        atol=1e-10,
    )