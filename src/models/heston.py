import numpy as np


def feller_condition(
    kappa,
    theta,
    xi,
):
    """
    Check the Feller condition for the Heston variance process.

    The sufficient condition for strict positivity is

        2 * kappa * theta > xi**2

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.

    theta : float
        Long-run variance.

    xi : float
        Volatility of variance (vol-of-vol).

    Returns
    -------
    bool
        True if the Feller condition is satisfied.
    """

    return (
        2.0 * kappa * theta
        >
        xi**2
    )


def expected_variance(
    t,
    v0,
    kappa,
    theta,
):
    """
    Compute the theoretical expected Heston variance.

    E[v_t] =
        theta
        + (v0 - theta) * exp(-kappa * t)
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    return (
        theta
        +
        (v0 - theta)
        * np.exp(-kappa * t)
    )


def simulate_heston_paths(
    S0,
    v0,
    T,
    r,
    kappa,
    theta,
    xi,
    rho,
    n_paths=10_000,
    n_steps=252,
    seed=None,
):
    """
    Simulate Heston stock-price and variance paths.

    A positivity-truncated Euler approximation is used
    for the variance process, while the stock process
    is updated using a log-Euler step.

    Parameters
    ----------
    S0 : float
        Initial stock price.

    v0 : float
        Initial variance.

    T : float
        Time horizon in years.

    r : float
        Continuously compounded risk-free rate.

    kappa : float
        Variance mean-reversion speed.

    theta : float
        Long-run variance.

    xi : float
        Volatility of variance.

    rho : float
        Correlation between stock and variance shocks.

    n_paths : int
        Number of simulated paths.

    n_steps : int
        Number of time steps.

    seed : int or None
        Random seed.

    Returns
    -------
    times : ndarray
        Time grid with shape (n_steps + 1,).

    stock_paths : ndarray
        Simulated stock paths with shape
        (n_paths, n_steps + 1).

    variance_paths : ndarray
        Simulated variance paths with shape
        (n_paths, n_steps + 1).
    """

    if S0 <= 0:
        raise ValueError(
            "S0 must be positive"
        )

    if v0 < 0:
        raise ValueError(
            "v0 must be non-negative"
        )

    if T <= 0:
        raise ValueError(
            "T must be positive"
        )

    if kappa < 0:
        raise ValueError(
            "kappa must be non-negative"
        )

    if theta < 0:
        raise ValueError(
            "theta must be non-negative"
        )

    if xi < 0:
        raise ValueError(
            "xi must be non-negative"
        )

    if not -1.0 <= rho <= 1.0:
        raise ValueError(
            "rho must lie between -1 and 1"
        )

    if n_paths <= 0:
        raise ValueError(
            "n_paths must be positive"
        )

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    dt = T / n_steps

    times = np.linspace(
        0.0,
        T,
        n_steps + 1,
    )

    stock_paths = np.empty(
        (
            n_paths,
            n_steps + 1,
        ),
        dtype=float,
    )

    variance_paths = np.empty(
        (
            n_paths,
            n_steps + 1,
        ),
        dtype=float,
    )

    stock_paths[:, 0] = S0
    variance_paths[:, 0] = v0

    sqrt_dt = np.sqrt(dt)

    for step in range(n_steps):

        z_stock = rng.standard_normal(
            n_paths
        )

        z_independent = (
            rng.standard_normal(
                n_paths
            )
        )

        z_variance = (
            rho * z_stock
            +
            np.sqrt(
                1.0 - rho**2
            )
            * z_independent
        )

        current_variance = np.maximum(
            variance_paths[:, step],
            0.0,
        )

        stock_paths[:, step + 1] = (
            stock_paths[:, step]
            * np.exp(
                (
                    r
                    - 0.5
                    * current_variance
                )
                * dt
                +
                np.sqrt(
                    current_variance
                )
                * sqrt_dt
                * z_stock
            )
        )

        next_variance = (
            variance_paths[:, step]
            +
            kappa
            * (
                theta
                - current_variance
            )
            * dt
            +
            xi
            * np.sqrt(
                current_variance
            )
            * sqrt_dt
            * z_variance
        )

        variance_paths[:, step + 1] = (
            np.maximum(
                next_variance,
                0.0,
            )
        )

    return (
        times,
        stock_paths,
        variance_paths,
    )