"""Analytical Greeks calculations for European options.

Computes first-order sensitivities (Delta, Gamma, Theta, Vega, Rho, Epsilon)
and two cross-sensitivities (Vanna, Charm) using the Black-Scholes/Merton
framework with continuous dividend yield.

Second-order Greeks included:

  Vanna  (d^2V / dS d_sigma): sensitivity of delta to volatility, and
    equivalently of vega to spot. Critical for delta-hedging books with
    significant skew exposure.

  Charm  (d^2V / dS dt): the rate of change of delta over time, also
    called "delta decay." Useful when managing overnight or weekend
    delta exposure without rehedging.

All Greeks default to dividend_yield=0.0, preserving backward compatibility
with code that does not pass a yield.
"""

from typing import Literal

import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


def _bs_d1_d2(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Compute d1 and d2 terms used across all Greeks.

    With continuous dividend yield q, d1 incorporates the cost-of-carry
    adjustment (r - q) rather than r alone.
    """
    sqrt_t = np.sqrt(expiry_years)
    d1 = (
        np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * expiry_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return float(d1), float(d2)


def compute_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    expiry_years: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Compute first-order Greeks, Gamma, Vanna, and Charm for a European option.

    All Greeks use the Merton (1973) continuous-dividend-yield extension.
    Setting dividend_yield=0 reproduces the standard Black-Scholes Greeks.

    Returned quantities:

      delta  - Option price change per unit change in spot. Range [-1, 1].
      gamma  - Rate of change of delta with respect to spot. Always positive.
      theta  - Daily time decay of option value (per calendar day, not
               trading day). Expressed as a loss in dollar terms.
      vega   - Option price change per 1% absolute move in implied volatility.
      rho    - Option price change per 1% absolute move in the risk-free rate.
      epsilon - Option price change per 1% absolute move in the dividend yield
               (sometimes called "psi" or "phi"). Always negative for calls,
               positive for puts, because higher yield reduces the forward.
      vanna  - dDelta/dSigma = dVega/dSpot. Measures how much delta shifts
               when volatility moves. Key for skew-sensitive hedging.
      charm  - dDelta/dt. Rate of delta decay over one calendar day. Expressed
               per day so that overnight exposure can be estimated directly.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".
        dividend_yield: Continuous dividend yield (annualized, as a decimal).
            Defaults to 0.0 (no dividends).

    Returns:
        Dictionary with keys: delta, gamma, theta, vega, rho, epsilon,
        vanna, charm.
    """
    d1, d2 = _bs_d1_d2(spot, strike, rate, volatility, expiry_years, dividend_yield)
    sqrt_t = np.sqrt(expiry_years)
    discount_r = np.exp(-rate * expiry_years)
    discount_q = np.exp(-dividend_yield * expiry_years)
    pdf_d1 = float(norm.pdf(d1))

    # Gamma: identical for calls and puts, dividend-adjusted
    gamma = float(discount_q * pdf_d1 / (spot * volatility * sqrt_t))

    # Vega: per 1% absolute vol move, dividend-adjusted
    vega = float(spot * discount_q * pdf_d1 * sqrt_t * 0.01)

    # Vanna: d^2V/(dS d_sigma) = -exp(-q*T) * N'(d1) * d2 / sigma
    # Same sign/magnitude for calls and puts.
    vanna = float(-discount_q * pdf_d1 * d2 / volatility)

    if option_type == "call":
        delta = float(discount_q * norm.cdf(d1))

        theta_annual = (
            -(spot * volatility * discount_q * pdf_d1) / (2.0 * sqrt_t)
            + dividend_yield * spot * discount_q * norm.cdf(d1)
            - rate * strike * discount_r * norm.cdf(d2)
        )
        rho = float(strike * expiry_years * discount_r * norm.cdf(d2) * 0.01)
        epsilon = float(-spot * expiry_years * discount_q * norm.cdf(d1) * 0.01)

        # Charm (call): -exp(-q*T) * [N'(d1) * (2*(r-q)*T - d2*sigma*sqrt(T))
        #               / (2*T*sigma*sqrt(T)) - q*N(d1)]
        charm_annual = -discount_q * (
            pdf_d1 * ((2.0 * (rate - dividend_yield) * expiry_years - d2 * volatility * sqrt_t)
                      / (2.0 * expiry_years * volatility * sqrt_t))
            - dividend_yield * float(norm.cdf(d1))
        )

    else:
        delta = float(discount_q * (norm.cdf(d1) - 1.0))

        theta_annual = (
            -(spot * volatility * discount_q * pdf_d1) / (2.0 * sqrt_t)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
            + rate * strike * discount_r * norm.cdf(-d2)
        )
        rho = float(-strike * expiry_years * discount_r * norm.cdf(-d2) * 0.01)
        epsilon = float(spot * expiry_years * discount_q * norm.cdf(-d1) * 0.01)

        # Charm (put): -exp(-q*T) * [N'(d1) * (2*(r-q)*T - d2*sigma*sqrt(T))
        #              / (2*T*sigma*sqrt(T)) + q*N(-d1)]
        charm_annual = -discount_q * (
            pdf_d1 * ((2.0 * (rate - dividend_yield) * expiry_years - d2 * volatility * sqrt_t)
                      / (2.0 * expiry_years * volatility * sqrt_t))
            + dividend_yield * float(norm.cdf(-d1))
        )

    # Theta: annual rate divided by 365 for per-calendar-day decay
    theta_daily = float(theta_annual / 365.0)

    # Charm: annual rate divided by 365 for per-calendar-day delta drift
    charm_daily = float(charm_annual / 365.0)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_daily,
        "vega": vega,
        "rho": rho,
        "epsilon": epsilon,
        "vanna": vanna,
        "charm": charm_daily,
    }
