import numpy as np

from src.models.heston import (
    expected_variance,
    feller_condition,
    simulate_heston_paths,
)


def test_heston_path_shapes_and_initial_values():
    (
        times,
        stock_paths,
        variance_paths,
    ) = simulate_heston_paths(
        S0=100.0,
        v0=0.04,
        T=1.0,
        r=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        n_paths=100,
        n_steps=50,
        seed=42,
    )

    assert times.shape == (51,)

    assert stock_paths.shape == (
        100,
        51,
    )

    assert variance_paths.shape == (
        100,
        51,
    )

    assert np.all(
        stock_paths[:, 0] == 100.0
    )

    assert np.all(
        variance_paths[:, 0] == 0.04
    )


def test_heston_variance_nonnegative():
    _, _, variance_paths = (
        simulate_heston_paths(
            S0=100.0,
            v0=0.04,
            T=1.0,
            r=0.05,
            kappa=2.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            n_paths=500,
            n_steps=100,
            seed=1,
        )
    )

    assert np.all(
        variance_paths >= 0.0
    )


def test_heston_seed_reproducibility():
    result_1 = simulate_heston_paths(
        S0=100.0,
        v0=0.04,
        T=1.0,
        r=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        n_paths=50,
        n_steps=20,
        seed=123,
    )

    result_2 = simulate_heston_paths(
        S0=100.0,
        v0=0.04,
        T=1.0,
        r=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        n_paths=50,
        n_steps=20,
        seed=123,
    )

    assert np.allclose(
        result_1[1],
        result_2[1],
    )

    assert np.allclose(
        result_1[2],
        result_2[2],
    )


def test_zero_vol_of_vol_gives_deterministic_variance():
    (
        times,
        _,
        variance_paths,
    ) = simulate_heston_paths(
        S0=100.0,
        v0=0.09,
        T=1.0,
        r=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.0,
        rho=0.0,
        n_paths=20,
        n_steps=100,
        seed=10,
    )

    first_path = variance_paths[0]

    for path in variance_paths:
        assert np.allclose(
            path,
            first_path,
        )

    theoretical_final = (
        expected_variance(
            times[-1],
            v0=0.09,
            kappa=2.0,
            theta=0.04,
        )
    )

    assert abs(
        first_path[-1]
        - theoretical_final
    ) < 0.001


def test_feller_condition():
    assert feller_condition(
        kappa=3.0,
        theta=0.04,
        xi=0.30,
    )

    assert not feller_condition(
        kappa=1.0,
        theta=0.04,
        xi=0.50,
    )