"""Option pricing model implementations.

Provides Black-Scholes, Binomial Tree (CRR), and Monte Carlo
simulation for European-style vanilla options.
"""

from typing import Literal

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


def _validate_inputs(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
) -> None:
    """Validate common pricing inputs."""
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"Strike price must be positive, got {strike}")
    if volatility < 0:
        raise ValueError(f"Volatility must be non-negative, got {volatility}")
    if expiry_years <= 0:
        raise ValueError(f"Time to expiry must be positive, got {expiry_years}")


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    option_type: OptionType = "call",
) -> float:
    """Price a European option using the Black-Scholes closed-form solution.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized, as a decimal).
        volatility: Annualized volatility of the underlying (as a decimal).
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years)

    d1 = (
        np.log(spot / strike) + (rate + 0.5 * volatility**2) * expiry_years
    ) / (volatility * np.sqrt(expiry_years))
    d2 = d1 - volatility * np.sqrt(expiry_years)

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * expiry_years) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * expiry_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return float(price)


def binomial_tree(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    option_type: OptionType = "call",
    steps: int = 200,
    american: bool = False,
) -> float:
    """Price an option using the Cox-Ross-Rubinstein binomial tree model.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        steps: Number of time steps in the tree (higher = more accurate).
        american: If True, allows early exercise (American-style).

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years)
    if steps < 1:
        raise ValueError(f"Steps must be at least 1, got {steps}")

    dt = expiry_years / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(rate * dt) - d) / (u - d)
    discount = np.exp(-rate * dt)

    # Build terminal payoffs
    asset_prices = spot * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)

    if option_type == "call":
        option_values = np.maximum(asset_prices - strike, 0.0)
    else:
        option_values = np.maximum(strike - asset_prices, 0.0)

    # Backward induction
    for step in range(steps - 1, -1, -1):
        option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])

        if american:
            asset_at_step = spot * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
            if option_type == "call":
                intrinsic = np.maximum(asset_at_step - strike, 0.0)
            else:
                intrinsic = np.maximum(strike - asset_at_step, 0.0)
            option_values = np.maximum(option_values, intrinsic)

    return float(option_values[0])


def monte_carlo(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    option_type: OptionType = "call",
    simulations: int = 100_000,
    seed: int | None = None,
) -> float:
    """Price a European option using Monte Carlo simulation with antithetic variates.

    Uses geometric Brownian motion paths and antithetic sampling
    for variance reduction.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        simulations: Number of simulation paths (half regular, half antithetic).
        seed: Optional random seed for reproducibility.

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years)
    if simulations < 100:
        raise ValueError(f"Simulations must be at least 100, got {simulations}")

    rng = np.random.default_rng(seed)
    half = simulations // 2

    z = rng.standard_normal(half)
    z_combined = np.concatenate([z, -z])  # antithetic variates

    drift = (rate - 0.5 * volatility**2) * expiry_years
    diffusion = volatility * np.sqrt(expiry_years) * z_combined
    terminal_prices = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_prices, 0.0)

    discounted_mean = np.exp(-rate * expiry_years) * np.mean(payoffs)
    return float(discounted_mean)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    expiry_years: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility using Newton-Raphson iteration.

    Args:
        market_price: Observed market price of the option.
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        tol: Convergence tolerance for the IV solution.
        max_iter: Maximum number of Newton-Raphson iterations.

    Returns:
        Implied volatility as a decimal.

    Raises:
        RuntimeError: If the solver does not converge within max_iter steps.
    """
    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    sigma = 0.25  # initial guess

    for i in range(max_iter):
        price = black_scholes(spot, strike, rate, sigma, expiry_years, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega: dPrice/dSigma
        d1 = (
            np.log(spot / strike) + (rate + 0.5 * sigma**2) * expiry_years
        ) / (sigma * np.sqrt(expiry_years))
        vega = spot * norm.pdf(d1) * np.sqrt(expiry_years)

        if vega < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(sigma, 1e-6)

    raise RuntimeError(
        f"Implied volatility solver did not converge after {max_iter} iterations "
        f"(last sigma={sigma:.6f}, price diff={diff:.6f})"
    )
