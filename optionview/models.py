"""Option pricing model implementations.

Provides Black-Scholes (Merton 1973 continuous-dividend extension),
Binomial Tree (CRR), and Monte Carlo simulation for European-style
vanilla options. All models accept an optional continuous dividend
yield so that equity options with known yield assumptions can be
priced consistently across methods.

Design note: the dividend yield enters via cost-of-carry. A stock
paying continuous yield q is equivalent to a forward priced at
S * exp((r - q) * T) under risk-neutral measure, which is the
substitution applied throughout.
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
    dividend_yield: float = 0.0,
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
    if dividend_yield < 0:
        raise ValueError(f"Dividend yield must be non-negative, got {dividend_yield}")


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option using the Black-Scholes/Merton closed-form solution.

    Implements the Merton (1973) continuous dividend yield extension. When
    dividend_yield=0 the result is identical to the original Black-Scholes
    formula. For equity options, pass the trailing or implied continuous
    dividend yield (as a decimal, e.g. 0.015 for 1.5%).

    The cost-of-carry adjustment replaces spot S with the dividend-discounted
    forward S * exp(-q * T) in all payoff calculations, which shifts d1:

        d1 = [log(S/K) + (r - q + 0.5*sigma^2)*T] / (sigma*sqrt(T))

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized, as a decimal).
        volatility: Annualized volatility of the underlying (as a decimal).
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        dividend_yield: Continuous dividend yield (annualized, as a decimal).
            Defaults to 0.0 (no dividends). For index options, this is
            typically the index dividend yield; for single stocks, use
            the trailing annual yield from fetch_dividend_yield().

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years, dividend_yield)

    sqrt_t = np.sqrt(expiry_years)
    d1 = (
        np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * expiry_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discount_r = np.exp(-rate * expiry_years)
    discount_q = np.exp(-dividend_yield * expiry_years)

    if option_type == "call":
        price = spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2)
    else:
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)

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
    dividend_yield: float = 0.0,
) -> float:
    """Price an option using the Cox-Ross-Rubinstein binomial tree model.

    With a continuous dividend yield, the risk-neutral up-probability
    uses cost-of-carry (r - q) in place of the risk-free rate:

        p = [exp((r - q) * dt) - d] / (u - d)

    This maintains put-call parity and no-arbitrage conditions while
    correctly discounting the expected dividend stream from the stock.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        steps: Number of time steps in the tree (higher = more accurate).
        american: If True, allows early exercise (American-style).
        dividend_yield: Continuous dividend yield (annualized, as a decimal).

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years, dividend_yield)
    if steps < 1:
        raise ValueError(f"Steps must be at least 1, got {steps}")

    dt = expiry_years / steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((rate - dividend_yield) * dt) - d) / (u - d)
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
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option using Monte Carlo simulation with antithetic variates.

    Uses geometric Brownian motion paths with the dividend-adjusted drift
    (r - q - 0.5*sigma^2). Antithetic sampling halves variance relative
    to naive Monte Carlo at no additional cost.

    The dividend yield reduces the expected stock growth rate, lowering
    call prices and raising put prices relative to the no-dividend case.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        simulations: Number of simulation paths (half regular, half antithetic).
        seed: Optional random seed for reproducibility.
        dividend_yield: Continuous dividend yield (annualized, as a decimal).

    Returns:
        Theoretical option price.
    """
    _validate_inputs(spot, strike, rate, volatility, expiry_years, dividend_yield)
    if simulations < 100:
        raise ValueError(f"Simulations must be at least 100, got {simulations}")

    rng = np.random.default_rng(seed)
    half = simulations // 2

    z = rng.standard_normal(half)
    z_combined = np.concatenate([z, -z])  # antithetic variates

    drift = (rate - dividend_yield - 0.5 * volatility**2) * expiry_years
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
    dividend_yield: float = 0.0,
) -> float:
    """Solve for implied volatility using Newton-Raphson iteration.

    Inverts the Black-Scholes/Merton formula to find the volatility
    sigma that reproduces the observed market_price. The vega
    used for Newton updates accounts for the dividend yield:

        vega = S * exp(-q * T) * N'(d1) * sqrt(T)

    Args:
        market_price: Observed market price of the option.
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        tol: Convergence tolerance for the IV solution.
        max_iter: Maximum number of Newton-Raphson iterations.
        dividend_yield: Continuous dividend yield (annualized, as a decimal).

    Returns:
        Implied volatility as a decimal.

    Raises:
        ValueError: If market_price is non-positive or below the arbitrage floor.
        RuntimeError: If the solver does not converge within max_iter steps.
    """
    if market_price <= 0:
        raise ValueError(f"Market price must be positive, got {market_price}")

    discount_q = np.exp(-dividend_yield * expiry_years)
    discount_r = np.exp(-rate * expiry_years)

    # Arbitrage floor check: intrinsic value
    if option_type == "call":
        floor = max(spot * discount_q - strike * discount_r, 0.0)
    else:
        floor = max(strike * discount_r - spot * discount_q, 0.0)

    if market_price < floor - tol:
        raise ValueError(
            f"Market price {market_price:.4f} is below the no-arbitrage floor "
            f"{floor:.4f} for this option. Check inputs."
        )

    sigma = 0.25  # initial guess

    for i in range(max_iter):
        price = black_scholes(spot, strike, rate, sigma, expiry_years, option_type, dividend_yield)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Dividend-adjusted vega: dPrice/dSigma
        sqrt_t = np.sqrt(expiry_years)
        d1 = (
            np.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma**2) * expiry_years
        ) / (sigma * sqrt_t)
        vega = spot * discount_q * norm.pdf(d1) * sqrt_t

        if vega < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(sigma, 1e-6)

    raise RuntimeError(
        f"Implied volatility solver did not converge after {max_iter} iterations "
        f"(last sigma={sigma:.6f}, price diff={diff:.6f})"
    )
