import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def call_price(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def put_price(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )

def call_delta(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def put_delta(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def call_theta(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)

    first_term = (
        -S * norm.pdf(d1) * sigma
        / (2 * np.sqrt(T))
    )

    second_term = (
        -r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    return first_term + second_term


def put_theta(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)

    first_term = (
        -S * norm.pdf(d1) * sigma
        / (2 * np.sqrt(T))
    )

    second_term = (
        r * K * np.exp(-r * T) * norm.cdf(-d2)
    )

    return first_term + second_term


def call_rho(S, K, T, r, sigma):
    _, d2 = d1_d2(S, K, T, r, sigma)

    return (
        K * T * np.exp(-r * T) * norm.cdf(d2)
    )


def put_rho(S, K, T, r, sigma):
    _, d2 = d1_d2(S, K, T, r, sigma)

    return (
        -K * T * np.exp(-r * T) * norm.cdf(-d2)
    )

