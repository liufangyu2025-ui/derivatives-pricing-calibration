import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_price(
    S0,
    K,
    T,
    r,
    sigma,
    option_type="call",
    S_max=None,
    M=400,
    N=400,
    return_grid=False,
):
    """
    Price a European option using the Crank-Nicolson
    finite-difference method.

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    option_type : {"call", "put"}
        Type of European option.
    S_max : float or None
        Upper boundary of the stock-price grid.
    M : int
        Number of stock-price intervals.
    N : int
        Number of time intervals.
    return_grid : bool
        If True, also return the stock grid and option values.

    Returns
    -------
    float
        Option price at S0.

    If return_grid=True:
        price, S_grid, V
    """

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be 'call' or 'put'"
        )

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if M < 3 or N < 1:
        raise ValueError("M must be >= 3 and N must be >= 1.")

    if S_max is None:
        S_max = 4 * max(S0, K)

    if S0 >= S_max:
        raise ValueError("S_max must be greater than S0.")

    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(
        0.0,
        S_max,
        M + 1,
    )

    # Terminal payoff at tau = 0
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)

    # Interior grid indices: i = 1, ..., M-1
    i = np.arange(
        1,
        M,
        dtype=float,
    )

    alpha = (
        0.25
        * dt
        * (
            sigma**2 * i**2
            - r * i
        )
    )

    beta = (
        -0.5
        * dt
        * (
            sigma**2 * i**2
            + r
        )
    )

    gamma = (
        0.25
        * dt
        * (
            sigma**2 * i**2
            + r * i
        )
    )

    # Left-hand-side tridiagonal matrix
    main_diag = 1.0 - beta
    lower_diag = -alpha[1:]
    upper_diag = -gamma[:-1]

    # Banded matrix format required by scipy.linalg.solve_banded
    ab = np.zeros(
        (3, M - 1)
    )

    ab[0, 1:] = upper_diag
    ab[1, :] = main_diag
    ab[2, :-1] = lower_diag

    def boundary_values(tau):
        discount = np.exp(-r * tau)

        if option_type == "call":
            left = 0.0
            right = (
                S_max
                - K * discount
            )
        else:
            left = K * discount
            right = 0.0

        return left, right

    # March forward in tau:
    # tau = 0 corresponds to maturity,
    # tau = T corresponds to today.
    for n in range(N):

        tau_now = n * dt
        tau_next = (n + 1) * dt

        left_now, right_now = (
            boundary_values(tau_now)
        )

        left_next, right_next = (
            boundary_values(tau_next)
        )

        V[0] = left_now
        V[-1] = right_now

        rhs = (
            alpha * V[:-2]
            + (1.0 + beta) * V[1:-1]
            + gamma * V[2:]
        )

        # Boundary contribution from the next time layer
        rhs[0] += alpha[0] * left_next
        rhs[-1] += gamma[-1] * right_next

        V[1:-1] = solve_banded(
            (1, 1),
            ab,
            rhs,
        )

        V[0] = left_next
        V[-1] = right_next

    price = np.interp(
        S0,
        S_grid,
        V,
    )

    if return_grid:
        return price, S_grid, V

    return price