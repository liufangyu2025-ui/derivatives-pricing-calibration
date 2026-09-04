from math import exp

from scipy.optimize import brentq

from src.models.black_scholes import (
    call_price,
    put_price,
)


def _option_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
):
    """
    Return the Black-Scholes price for a European call or put.
    """

    if option_type == "call":
        return call_price(
            S,
            K,
            T,
            r,
            sigma,
        )

    if option_type == "put":
        return put_price(
            S,
            K,
            T,
            r,
            sigma,
        )

    raise ValueError(
        "option_type must be either 'call' or 'put'"
    )


def _arbitrage_bounds(
    S,
    K,
    T,
    r,
    option_type,
):
    """
    Return no-arbitrage lower and upper bounds
    for a European option without dividends.
    """

    discounted_strike = K * exp(-r * T)

    if option_type == "call":
        lower_bound = max(
            S - discounted_strike,
            0.0,
        )
        upper_bound = S

    elif option_type == "put":
        lower_bound = max(
            discounted_strike - S,
            0.0,
        )
        upper_bound = discounted_strike

    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    return lower_bound, upper_bound


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    sigma_lower=1e-8,
    sigma_upper=5.0,
    tol=1e-10,
):
    """
    Compute Black-Scholes implied volatility
    using Brent's root-finding method.

    Parameters
    ----------
    market_price : float
        Observed option price.

    S : float
        Current underlying price.

    K : float
        Strike price.

    T : float
        Time to maturity in years.

    r : float
        Continuously compounded risk-free rate.

    option_type : {"call", "put"}
        European option type.

    sigma_lower : float
        Lower volatility search bound.

    sigma_upper : float
        Upper volatility search bound.

    tol : float
        Numerical tolerance.

    Returns
    -------
    float
        Black-Scholes implied volatility.
    """

    if S <= 0:
        raise ValueError("S must be positive")

    if K <= 0:
        raise ValueError("K must be positive")

    if T <= 0:
        raise ValueError("T must be positive")

    if market_price < 0:
        raise ValueError(
            "market_price must be non-negative"
        )

    lower_bound, upper_bound = _arbitrage_bounds(
        S,
        K,
        T,
        r,
        option_type,
    )

    if market_price < lower_bound - tol:
        raise ValueError(
            "Option price is below the no-arbitrage lower bound"
        )

    if market_price > upper_bound + tol:
        raise ValueError(
            "Option price is above the no-arbitrage upper bound"
        )

    # At the lower arbitrage bound,
    # implied volatility approaches zero.
    if abs(market_price - lower_bound) <= tol:
        return 0.0

    def objective(sigma):
        model_price = _option_price(
            S,
            K,
            T,
            r,
            sigma,
            option_type,
        )

        return model_price - market_price

    f_lower = objective(sigma_lower)
    f_upper = objective(sigma_upper)

    if f_lower * f_upper > 0:
        raise ValueError(
            "Unable to bracket the implied volatility. "
            "Try increasing sigma_upper."
        )

    sigma_implied = brentq(
        objective,
        sigma_lower,
        sigma_upper,
        xtol=tol,
        rtol=tol,
    )

    return sigma_implied