"""Option pricing model implementations.

Provides Black-Scholes (Merton 1973 continuous-dividend extension),
Binomial Tree (CRR), Monte Carlo simulation, and the Heston stochastic
volatility model for European-style vanilla options. All models accept an optional continuous dividend
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


# ---------------------------------------------------------------------------
# Heston stochastic volatility model
# ---------------------------------------------------------------------------
import cmath as _cmath
import math as _math
from scipy import integrate as _integrate


def _heston_cf(
    phi: float,
    j: int,
    spot: float,
    rate: float,
    dividend_yield: float,
    expiry_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> complex:
    """Heston characteristic function f_j(phi) for computing risk-adjusted probabilities.

    Implements the 'little trap' sign convention from Albrecher, Mayer, Schoutens,
    and Tistaert (2007). The original Heston (1993) formula uses exp(-d*T) in the
    g_j denominator, which causes the argument of the complex log to cross the
    negative real axis for certain (phi, kappa, rho) combinations, introducing a
    branch-cut discontinuity. The fix uses +d in the g_j numerator/denominator,
    so the log argument stays in the right half-plane for all valid inputs.

    Args:
        phi: Real integration variable (strictly positive).
        j: 1 for the stock-price measure (computes P1), 2 for risk-neutral (P2).
    """
    u_j = 0.5 if j == 1 else -0.5
    b_j = kappa - rho * sigma if j == 1 else kappa

    ip = 1j * phi

    d_j = _cmath.sqrt(
        (rho * sigma * ip - b_j) ** 2 - sigma ** 2 * (2.0 * u_j * ip - phi ** 2)
    )

    # Little-trap sign: +d in numerator keeps the log argument bounded away
    # from the negative real axis, eliminating the branch-cut problem.
    g_j = (b_j - rho * sigma * ip + d_j) / (b_j - rho * sigma * ip - d_j)

    exp_d_tau = _cmath.exp(d_j * expiry_years)

    C_j = (rate - dividend_yield) * ip * expiry_years + (kappa * theta / sigma ** 2) * (
        (b_j - rho * sigma * ip + d_j) * expiry_years
        - 2.0 * _cmath.log((1.0 - g_j * exp_d_tau) / (1.0 - g_j))
    )
    D_j = ((b_j - rho * sigma * ip + d_j) / sigma ** 2) * (
        (1.0 - exp_d_tau) / (1.0 - g_j * exp_d_tau)
    )

    return _cmath.exp(C_j + D_j * v0 + ip * _cmath.log(spot))


def heston(
    spot: float,
    strike: float,
    rate: float,
    expiry_years: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option under the Heston stochastic volatility model.

    The Heston model extends geometric Brownian motion with a mean-reverting
    variance process, capturing the implied vol smile that constant-vol models
    cannot reproduce:

        dS = (r - q) S dt + sqrt(v) S dW_S
        dv = kappa * (theta - v) dt + sigma * sqrt(v) dW_v
        corr(dW_S, dW_v) = rho dt

    The European price is computed via Gil-Pelaez inversion of the characteristic
    function. The formula decomposes the call price into two risk-adjusted
    probabilities P1 and P2:

        C = S * exp(-q*T) * P1 - K * exp(-r*T) * P2

    Each probability is recovered from a single numerical integral over the
    real line using scipy.integrate.quad. The 'little trap' sign convention
    (Albrecher et al. 2007) is used to prevent branch-cut discontinuities
    that arise in the original Heston (1993) formula for certain parameter
    combinations (large kappa, long expiry, or extreme rho).

    The key inputs that control the smile shape are:
      - sigma (vol-of-vol): governs smile curvature; higher sigma widens the smile.
      - rho: governs skew direction; negative rho creates the equity-style
        downward slope (put wings are more expensive than call wings).
      - kappa and theta: control variance mean reversion; fast reversion (high
        kappa) collapses the term structure of the smile.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price. Must be positive.
        rate: Continuously compounded risk-free rate (annualized).
        expiry_years: Time to expiration in years. Must be positive.
        v0: Initial instantaneous variance (annualized). Equal to vol^2, not vol.
            For example, a 25% initial vol corresponds to v0 = 0.0625.
        kappa: Mean-reversion speed for the variance process. Must be positive.
        theta: Long-run variance (the variance level v(t) reverts to). Must be
            positive. The long-run implied vol is approximately sqrt(theta).
        sigma: Volatility of variance (vol-of-vol). Controls smile curvature.
            Typical equity values are in the range 0.2-0.8. Must be positive.
        rho: Correlation between the spot and variance Brownian motions.
            Equity indices typically have rho in (-0.9, -0.3). Must be in (-1, 1).
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield (annualized, as a decimal).
            Defaults to 0.0.

    Returns:
        European option price. Always non-negative.

    Raises:
        ValueError: If spot, strike, v0, kappa, theta, or sigma are non-positive;
            if rho is outside the open interval (-1, 1); if expiry_years is
            non-positive; or if dividend_yield is negative.

    Note:
        The Feller condition (2 * kappa * theta > sigma^2) guarantees that the
        variance process stays strictly positive. When the condition is violated,
        v(t) can touch zero, which may reduce numerical precision for long
        expiries or large sigma values. Prices remain valid but interpret results
        with care outside the Feller regime.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0.0:
        raise ValueError(f"strike must be positive, got {strike}")
    if expiry_years <= 0.0:
        raise ValueError(f"expiry_years must be positive, got {expiry_years}")
    if v0 <= 0.0:
        raise ValueError(f"v0 must be positive, got {v0}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if theta <= 0.0:
        raise ValueError(f"theta must be positive, got {theta}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if dividend_yield < 0.0:
        raise ValueError(f"dividend_yield must be non-negative, got {dividend_yield}")

    log_k = _math.log(strike)

    def _integrand(phi: float, j: int) -> float:
        cf = _heston_cf(
            phi, j, spot, rate, dividend_yield, expiry_years,
            v0, kappa, theta, sigma, rho,
        )
        return (_cmath.exp(-1j * phi * log_k) * cf / (1j * phi)).real

    # Integrate from a small epsilon to avoid the well-defined but numerically
    # sensitive limit at phi=0. Upper limit 1000 is conservative: the integrand
    # decays exponentially and is negligible well before phi reaches this value
    # for all parameter combinations encountered in practice.
    eps = 1e-4
    upper = 1000.0

    p1_int, _ = _integrate.quad(
        _integrand, eps, upper, args=(1,), limit=300, epsabs=1.5e-8, epsrel=1.5e-8,
    )
    p2_int, _ = _integrate.quad(
        _integrand, eps, upper, args=(2,), limit=300, epsabs=1.5e-8, epsrel=1.5e-8,
    )

    p1 = max(0.0, min(1.0, 0.5 + p1_int / _math.pi))
    p2 = max(0.0, min(1.0, 0.5 + p2_int / _math.pi))

    call_price = (
        spot * _math.exp(-dividend_yield * expiry_years) * p1
        - strike * _math.exp(-rate * expiry_years) * p2
    )

    if option_type == "call":
        return max(call_price, 0.0)

    # Derive put via put-call parity: P = C - S*exp(-q*T) + K*exp(-r*T)
    put_price = (
        call_price
        - spot * _math.exp(-dividend_yield * expiry_years)
        + strike * _math.exp(-rate * expiry_years)
    )
    return max(put_price, 0.0)
