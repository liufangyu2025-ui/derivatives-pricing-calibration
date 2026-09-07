import numpy as np

from scipy.optimize import least_squares

from src.pricing.heston_pricing import (
    heston_call_price,
)


PARAMETER_NAMES = (
    "v0",
    "kappa",
    "theta",
    "xi",
    "rho",
)


DEFAULT_LOWER_BOUNDS = np.array(
    [
        1e-4,    # v0
        0.05,    # kappa
        1e-4,    # theta
        0.01,    # xi
        -0.999,  # rho
    ],
    dtype=float,
)


DEFAULT_UPPER_BOUNDS = np.array(
    [
        1.00,   # v0
        10.00,  # kappa
        1.00,   # theta
        5.00,   # xi
        0.999,  # rho
    ],
    dtype=float,
)


def parameter_vector_to_dict(
    parameters,
):
    """
    Convert a five-element parameter vector into
    a named Heston parameter dictionary.
    """

    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    if parameters.shape != (5,):
        raise ValueError(
            "Heston parameter vector must contain "
            "exactly five elements."
        )

    if not np.all(
        np.isfinite(parameters)
    ):
        raise ValueError(
            "Heston parameters must be finite."
        )

    return dict(
        zip(
            PARAMETER_NAMES,
            parameters,
        )
    )


def _validate_option_data(
    strikes,
    maturities,
    market_prices=None,
):
    """
    Validate flattened option observations.
    """

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
            "strikes and maturities must have "
            "the same shape."
        )

    if np.any(strikes <= 0):
        raise ValueError(
            "All strikes must be positive."
        )

    if np.any(maturities <= 0):
        raise ValueError(
            "All maturities must be positive."
        )

    if market_prices is not None:

        market_prices = np.asarray(
            market_prices,
            dtype=float,
        )

        if (
            market_prices.shape
            != strikes.shape
        ):
            raise ValueError(
                "market_prices must have the "
                "same shape as strikes."
            )

        if np.any(
            market_prices < 0
        ):
            raise ValueError(
                "market_prices must be "
                "non-negative."
            )

    return (
        strikes,
        maturities,
    )


def heston_call_price_vector(
    parameters,
    S0,
    r,
    strikes,
    maturities,
    integration_limit=100.0,
):
    """
    Compute Heston European call prices for
    a collection of strike-maturity pairs.
    """

    if S0 <= 0:
        raise ValueError(
            "S0 must be positive."
        )

    strikes, maturities = (
        _validate_option_data(
            strikes=strikes,
            maturities=maturities,
        )
    )

    parameter_dict = (
        parameter_vector_to_dict(
            parameters
        )
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

        prices[index] = (
            heston_call_price(
                S0=S0,
                K=float(strike),
                T=float(maturity),
                r=r,
                v0=parameter_dict["v0"],
                kappa=parameter_dict[
                    "kappa"
                ],
                theta=parameter_dict[
                    "theta"
                ],
                xi=parameter_dict["xi"],
                rho=parameter_dict["rho"],
                integration_limit=(
                    integration_limit
                ),
            )
        )

    return prices


def heston_price_residuals(
    parameters,
    market_prices,
    S0,
    r,
    strikes,
    maturities,
    weights=None,
    integration_limit=100.0,
):
    """
    Return model-minus-market pricing residuals.

    Optional weights are interpreted as
    weighted least-squares weights, so the
    returned residual is multiplied by
    sqrt(weight).
    """

    strikes, maturities = (
        _validate_option_data(
            strikes=strikes,
            maturities=maturities,
            market_prices=market_prices,
        )
    )

    market_prices = np.asarray(
        market_prices,
        dtype=float,
    )

    model_prices = (
        heston_call_price_vector(
            parameters=parameters,
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
            integration_limit=(
                integration_limit
            ),
        )
    )

    residuals = (
        model_prices
        - market_prices
    )

    if weights is not None:

        weights = np.asarray(
            weights,
            dtype=float,
        )

        if weights.shape != residuals.shape:
            raise ValueError(
                "weights must have the same "
                "shape as market_prices."
            )

        if np.any(weights <= 0):
            raise ValueError(
                "weights must be positive."
            )

        residuals = (
            residuals
            * np.sqrt(weights)
        )

    return residuals


def calibrate_heston(
    market_prices,
    S0,
    r,
    strikes,
    maturities,
    initial_guess,
    lower_bounds=None,
    upper_bounds=None,
    weights=None,
    integration_limit=100.0,
    max_nfev=250,
    verbose=0,
):
    """
    Calibrate Heston parameters using bounded
    nonlinear least squares.
    """

    strikes, maturities = (
        _validate_option_data(
            strikes=strikes,
            maturities=maturities,
            market_prices=market_prices,
        )
    )

    market_prices = np.asarray(
        market_prices,
        dtype=float,
    )

    initial_guess = np.asarray(
        initial_guess,
        dtype=float,
    )

    if initial_guess.shape != (5,):
        raise ValueError(
            "initial_guess must contain "
            "five Heston parameters."
        )

    if lower_bounds is None:
        lower_bounds = (
            DEFAULT_LOWER_BOUNDS.copy()
        )
    else:
        lower_bounds = np.asarray(
            lower_bounds,
            dtype=float,
        )

    if upper_bounds is None:
        upper_bounds = (
            DEFAULT_UPPER_BOUNDS.copy()
        )
    else:
        upper_bounds = np.asarray(
            upper_bounds,
            dtype=float,
        )

    if lower_bounds.shape != (5,):
        raise ValueError(
            "lower_bounds must contain "
            "five elements."
        )

    if upper_bounds.shape != (5,):
        raise ValueError(
            "upper_bounds must contain "
            "five elements."
        )

    if np.any(
        lower_bounds
        >= upper_bounds
    ):
        raise ValueError(
            "Each lower bound must be "
            "strictly below its upper bound."
        )

    if np.any(
        initial_guess
        <= lower_bounds
    ) or np.any(
        initial_guess
        >= upper_bounds
    ):
        raise ValueError(
            "initial_guess must lie strictly "
            "inside the parameter bounds."
        )

    result = least_squares(
        fun=heston_price_residuals,
        x0=initial_guess,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        args=(
            market_prices,
            S0,
            r,
            strikes,
            maturities,
            weights,
            integration_limit,
        ),
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=verbose,
    )

    calibrated_parameters = (
        parameter_vector_to_dict(
            result.x
        )
    )

    fitted_prices = (
        heston_call_price_vector(
            parameters=result.x,
            S0=S0,
            r=r,
            strikes=strikes,
            maturities=maturities,
            integration_limit=(
                integration_limit
            ),
        )
    )

    raw_residuals = (
        fitted_prices
        - market_prices
    )

    rmse = np.sqrt(
        np.mean(
            raw_residuals**2
        )
    )

    mae = np.mean(
        np.abs(
            raw_residuals
        )
    )

    return {
        "parameters":
            calibrated_parameters,
        "x":
            result.x.copy(),
        "success":
            bool(result.success),
        "message":
            result.message,
        "nfev":
            int(result.nfev),
        "cost":
            float(result.cost),
        "rmse":
            float(rmse),
        "mae":
            float(mae),
        "fitted_prices":
            fitted_prices,
        "residuals":
            raw_residuals,
        "optimizer_result":
            result,
    }