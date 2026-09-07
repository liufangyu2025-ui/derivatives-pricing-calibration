import numpy as np
from scipy.integrate import quad
from src.models.heston import simulate_heston_paths


def heston_characteristic_function(
    u,
    S0,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
):
    """
    Compute the risk-neutral characteristic function
    of log(S_T) under the Heston model.

    The characteristic function is defined as

        phi(u) = E^Q[exp(i * u * log(S_T))]

    Parameters
    ----------
    u : complex or array-like
        Fourier argument.

    S0 : float
        Initial stock price.

    T : float
        Time to maturity.

    r : float
        Continuously compounded risk-free rate.

    v0 : float
        Initial variance.

    kappa : float
        Mean-reversion speed of variance.

    theta : float
        Long-run variance.

    xi : float
        Volatility of variance.

    rho : float
        Correlation between stock and variance shocks.

    Returns
    -------
    complex or ndarray
        Characteristic-function value.
    """

    if S0 <= 0:
        raise ValueError(
            "S0 must be positive"
        )

    if T < 0:
        raise ValueError(
            "T must be non-negative"
        )

    if v0 < 0:
        raise ValueError(
            "v0 must be non-negative"
        )

    if kappa <= 0:
        raise ValueError(
            "kappa must be positive"
        )

    if theta < 0:
        raise ValueError(
            "theta must be non-negative"
        )

    if xi <= 0:
        raise ValueError(
            "xi must be positive"
        )

    if not -1.0 <= rho <= 1.0:
        raise ValueError(
            "rho must lie between -1 and 1"
        )

    u = np.asarray(
        u,
        dtype=np.complex128,
    )

    i = 1j

    beta = (
        kappa
        - rho * xi * i * u
    )

    d = np.sqrt(
        beta**2
        +
        xi**2
        * (
            u**2
            + i * u
        )
    )

    # Use the branch with non-negative real part.
    d = np.where(
        np.real(d) < 0,
        -d,
        d,
    )

    g = (
        beta - d
    ) / (
        beta + d
    )

    exp_minus_dT = np.exp(
        -d * T
    )

    log_term = np.log(
        (
            1.0
            - g * exp_minus_dT
        )
        /
        (
            1.0 - g
        )
    )

    C = (
        r * i * u * T
        +
        (
            kappa
            * theta
            / xi**2
        )
        *
        (
            (
                beta - d
            )
            * T
            -
            2.0
            * log_term
        )
    )

    D = (
        (
            beta - d
        )
        / xi**2
        *
        (
            1.0
            - exp_minus_dT
        )
        /
        (
            1.0
            - g
            * exp_minus_dT
        )
    )

    return np.exp(
        i
        * u
        * np.log(S0)
        +
        C
        +
        D * v0
    )

def heston_call_price(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    integration_limit=150.0,
):
    """
    Price a European call option under the Heston model
    using characteristic-function Fourier inversion.

    Parameters
    ----------
    S0 : float
        Initial stock price.

    K : float
        Strike price.

    T : float
        Time to maturity.

    r : float
        Continuously compounded risk-free rate.

    v0 : float
        Initial variance.

    kappa : float
        Variance mean-reversion speed.

    theta : float
        Long-run variance.

    xi : float
        Volatility of variance.

    rho : float
        Correlation between stock and variance shocks.

    integration_limit : float
        Upper limit used to approximate the
        infinite Fourier integral.

    Returns
    -------
    float
        Heston European call price.
    """

    if K <= 0:
        raise ValueError(
            "K must be positive"
        )

    if T <= 0:
        raise ValueError(
            "T must be positive"
        )

    if integration_limit <= 0:
        raise ValueError(
            "integration_limit must be positive"
        )

    log_strike = np.log(K)

    phi_minus_i = (
        heston_characteristic_function(
            u=-1j,
            S0=S0,
            T=T,
            r=r,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
        )
    )

    def integrand_p1(u):

        shifted_cf = (
            heston_characteristic_function(
                u=u - 1j,
                S0=S0,
                T=T,
                r=r,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
            )
        )

        value = (
            np.exp(
                -1j * u * log_strike
            )
            * shifted_cf
            /
            (
                1j
                * u
                * phi_minus_i
            )
        )

        return float(
            np.real(value)
        )

    def integrand_p2(u):

        cf_value = (
            heston_characteristic_function(
                u=u,
                S0=S0,
                T=T,
                r=r,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
            )
        )

        value = (
            np.exp(
                -1j * u * log_strike
            )
            * cf_value
            /
            (
                1j * u
            )
        )

        return float(
            np.real(value)
        )

    lower_limit = 1e-8

    integral_p1 = quad(
        integrand_p1,
        lower_limit,
        integration_limit,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=500,
    )[0]

    integral_p2 = quad(
        integrand_p2,
        lower_limit,
        integration_limit,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=500,
    )[0]

    P1 = (
        0.5
        +
        integral_p1 / np.pi
    )

    P2 = (
        0.5
        +
        integral_p2 / np.pi
    )

    call_price = (
        S0 * P1
        -
        K
        * np.exp(-r * T)
        * P2
    )

    return float(
        np.real(call_price)
    )

def heston_put_price(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    integration_limit=150.0,
):
    """
    Price a European put option under the Heston model
    using put-call parity.
    """

    call_price = heston_call_price(
        S0=S0,
        K=K,
        T=T,
        r=r,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        integration_limit=integration_limit,
    )

    put_price = (
        call_price
        - S0
        + K * np.exp(-r * T)
    )

    return float(
        put_price
    )

def heston_monte_carlo_price(
    S0,
    K,
    T,
    r,
    v0,
    kappa,
    theta,
    xi,
    rho,
    option_type="call",
    n_paths=10_000,
    n_steps=252,
    seed=None,
):
    """
    Price a European option under the Heston model
    using Monte Carlo simulation.

    Returns
    -------
    price : float
        Monte Carlo price estimate.

    standard_error : float
        Standard error of the discounted payoff mean.
    """

    if K <= 0:
        raise ValueError(
            "K must be positive"
        )

    if n_paths < 2:
        raise ValueError(
            "n_paths must be at least 2"
        )

    option_type = option_type.lower()

    if option_type not in {
        "call",
        "put",
    }:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )

    _, stock_paths, _ = simulate_heston_paths(
        S0=S0,
        v0=v0,
        T=T,
        r=r,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )

    terminal_prices = (
        stock_paths[:, -1]
    )

    if option_type == "call":

        payoff = np.maximum(
            terminal_prices - K,
            0.0,
        )

    else:

        payoff = np.maximum(
            K - terminal_prices,
            0.0,
        )

    discounted_payoff = (
        np.exp(-r * T)
        * payoff
    )

    price = np.mean(
        discounted_payoff
    )

    standard_error = (
        np.std(
            discounted_payoff,
            ddof=1,
        )
        /
        np.sqrt(n_paths)
    )

    return (
        float(price),
        float(standard_error),
    )