"""Analytical Greeks calculations for European options.

Computes first-order and second-order sensitivities using
the Black-Scholes framework.
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
) -> tuple[float, float]:
    """Compute d1 and d2 terms used across all Greeks."""
    sqrt_t = np.sqrt(expiry_years)
    d1 = (
        np.log(spot / strike) + (rate + 0.5 * volatility**2) * expiry_years
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
) -> dict[str, float]:
    """Compute all first-order Greeks plus Gamma for a European option.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized).
        volatility: Annualized volatility of the underlying.
        expiry_years: Time to expiration in years.
        option_type: Either "call" or "put".

    Returns:
        Dictionary with keys: delta, gamma, theta, vega, rho.
        Theta is expressed as daily decay (divided by 365).
        Vega is per 1% move in volatility.
    """
    d1, d2 = _bs_d1_d2(spot, strike, rate, volatility, expiry_years)
    sqrt_t = np.sqrt(expiry_years)
    discount = np.exp(-rate * expiry_years)
    pdf_d1 = norm.pdf(d1)

    # Gamma (same for calls and puts)
    gamma = pdf_d1 / (spot * volatility * sqrt_t)

    # Vega (same for calls and puts, per 1% vol move)
    vega = spot * pdf_d1 * sqrt_t * 0.01

    if option_type == "call":
        delta = float(norm.cdf(d1))
        theta_annual = (
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            - rate * strike * discount * norm.cdf(d2)
        )
        rho = strike * expiry_years * discount * norm.cdf(d2) * 0.01
    else:
        delta = float(norm.cdf(d1) - 1)
        theta_annual = (
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            + rate * strike * discount * norm.cdf(-d2)
        )
        rho = -strike * expiry_years * discount * norm.cdf(-d2) * 0.01

    # Convert annual theta to daily
    theta_daily = theta_annual / 365.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_daily),
        "vega": float(vega),
        "rho": float(rho),
    }
