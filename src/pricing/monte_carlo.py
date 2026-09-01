import numpy as np


def simulate_terminal_prices(
    S0,
    T,
    r,
    sigma,
    n_paths=100_000,
    seed=None,
):
    """
    Simulate terminal stock prices under the risk-neutral measure.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    n_paths : int
        Number of Monte Carlo simulations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated terminal stock prices.
    """

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp(
        (r - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )

    return ST

def monte_carlo_price(
    S0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_paths=100_000,
    seed=None,
):
    """
    Price a European option using Monte Carlo simulation.

    Returns
    -------
    price : float
        Monte Carlo option price.
    standard_error : float
        Standard error of the Monte Carlo estimator.
    """

    ST = simulate_terminal_prices(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_paths=n_paths,
        seed=seed,
    )

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)

    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )

    discounted_payoff = np.exp(-r * T) * payoff

    price = np.mean(discounted_payoff)

    standard_error = (
        np.std(discounted_payoff, ddof=1)
        / np.sqrt(n_paths)
    )

    return price, standard_error

def monte_carlo_price(
    S0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_paths=100_000,
    seed=None,
):
    """
    Price a European option using Monte Carlo simulation.

    Returns
    -------
    price : float
        Monte Carlo option price.
    standard_error : float
        Standard error of the Monte Carlo estimator.
    """

    ST = simulate_terminal_prices(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_paths=n_paths,
        seed=seed,
    )

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)

    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)

    else:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )

    discounted_payoff = np.exp(-r * T) * payoff

    price = np.mean(discounted_payoff)

    standard_error = (
        np.std(discounted_payoff, ddof=1)
        / np.sqrt(n_paths)
    )

    return price, standard_error

def monte_carlo_price_antithetic(
    S0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_paths=100_000,
    seed=None,
):
    rng = np.random.default_rng(seed)

    n_pairs = n_paths // 2
    Z = rng.standard_normal(n_pairs)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    ST_positive = S0 * np.exp(
        drift + diffusion * Z
    )

    ST_negative = S0 * np.exp(
        drift - diffusion * Z
    )

    if option_type == "call":
        payoff_positive = np.maximum(ST_positive - K, 0.0)
        payoff_negative = np.maximum(ST_negative - K, 0.0)

    elif option_type == "put":
        payoff_positive = np.maximum(K - ST_positive, 0.0)
        payoff_negative = np.maximum(K - ST_negative, 0.0)

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    pair_payoff = (
        payoff_positive + payoff_negative
    ) / 2.0

    discounted_pair_payoff = (
        np.exp(-r * T) * pair_payoff
    )

    price = np.mean(discounted_pair_payoff)

    standard_error = (
        np.std(discounted_pair_payoff, ddof=1)
        / np.sqrt(n_pairs)
    )

    return price, standard_error