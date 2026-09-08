import numpy as np

from scipy.optimize import minimize_scalar

from src.models.black_scholes import (
    call_price,
)

def black_scholes_call_price_vector(
    sigma,
    S0,
    r,
    strikes,
    maturities,
):
    """
    Compute Black-Scholes European call prices
    for a collection of strike-maturity pairs
    using one constant volatility.
    """

    if S0 <= 0:
        raise ValueError(
            "S0 must be positive."
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be positive."
        )

    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    maturities = np.asarray(
        maturities,
        dtype=float,
    )

    if strikes.ndim != 1:
        raise ValueError(
            "strikes must be one-dimensional."
        )

    if maturities.ndim != 1:
        raise ValueError(
            "maturities must be one-dimensional."
        )

    if strikes.shape != maturities.shape:
        raise ValueError(
            "strikes and maturities must "
            "have the same shape."
        )

    if np.any(strikes <= 0):
        raise ValueError(
            "All strikes must be positive."
        )

    if np.any(maturities <= 0):
        raise ValueError(
            "All maturities must be positive."
        )

    prices = np.empty(
        len(strikes),
        dtype=float,
    )

    for index, (
        strike,
        maturity,
    ) in enumerate(
        zip(
            strikes,
            maturities,
        )
    ):

        prices[index] = call_price(
            S0,
            float(strike),
            float(maturity),
            r,
            sigma,
        )

    return prices

def calibrate_black_scholes_constant_vol(
    market_prices,
    S0,
    r,
    strikes,
    maturities,
    sigma_lower=0.01,
    sigma_upper=2.0,
):
    """
    Calibrate one constant Black-Scholes
    volatility to a collection of option prices
    by minimizing mean squared pricing error.
    """

    market_prices = np.asarray(
        market_prices,
        dtype=float,
    )

    strikes = np.asarray(
        strikes,
        dtype=float,
    )

    maturities = np.asarray(
        maturities,
        dtype=float,
    )

    if market_prices.ndim != 1:
        raise ValueError(
            "market_prices must be "
            "one-dimensional."
        )

    if not (
        market_prices.shape
        == strikes.shape
        == maturities.shape
    ):
        raise ValueError(
            "market_prices, strikes, and "
            "maturities must have the same shape."
        )

    if np.any(
        market_prices < 0
    ):
        raise ValueError(
            "market_prices must be non-negative."
        )

    if sigma_lower <= 0:
        raise ValueError(
            "sigma_lower must be positive."
        )

    if sigma_upper <= sigma_lower:
        raise ValueError(
            "sigma_upper must exceed sigma_lower."
        )

    def objective(
        sigma,
    ):

        model_prices = (
            black_scholes_call_price_vector(
                sigma=sigma,
                S0=S0,
                r=r,
                strikes=strikes,
                maturities=maturities,
            )
        )

        residuals = (
            model_prices
            - market_prices
        )

        return np.mean(
            residuals**2
        )

    result = minimize_scalar(
        objective,
        bounds=(
            sigma_lower,
            sigma_upper,
        ),
        method="bounded",
        options={
            "xatol": 1e-10,
        },
    )

    fitted_sigma = float(
        result.x
    )

    fitted_prices = (
        black_scholes_call_price_vector(
            sigma=fitted_sigma,
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
        )
    )

    residuals = (
        fitted_prices
        - market_prices
    )

    rmse = np.sqrt(
        np.mean(
            residuals**2
        )
    )

    mae = np.mean(
        np.abs(
            residuals
        )
    )

    return {
        "sigma":
            fitted_sigma,
        "success":
            bool(result.success),
        "message":
            result.message,
        "rmse":
            float(rmse),
        "mae":
            float(mae),
        "fitted_prices":
            fitted_prices,
        "residuals":
            residuals,
        "optimizer_result":
            result,
    }