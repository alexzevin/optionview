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


# ---------------------------------------------------------------------------
# SABR stochastic volatility model (Hagan, Kumar, Lesniewski, Woodward 2002)
# ---------------------------------------------------------------------------

def sabr_implied_vol(
    forward: float,
    strike: float,
    expiry_years: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute SABR implied volatility using the Hagan et al. (2002) approximation.

    The SABR model (Stochastic Alpha Beta Rho) describes the risk-neutral
    dynamics of a forward price F and its instantaneous volatility:

        dF = alpha * F^beta * dW_F
        d_alpha = nu * alpha * dW_alpha
        corr(dW_F, dW_alpha) = rho * dt

    The Hagan approximation gives a closed-form Black-Scholes implied vol
    without numerical integration, accurate to O(T^2). It is the industry
    standard for interest rate swaptions and FX options because it calibrates
    rapidly to multi-strike slices and naturally reproduces the skew (via rho)
    and curvature (via nu) observed in liquid option markets.

    The implementation handles three structurally distinct cases:

    1. ATM (F ~ K) or no stochastic vol (nu=0): the z/chi(z) ratio limits
       to 1 via L'Hopital and the formula collapses to the pure-CEV level.
    2. Off-ATM with nu > 0: the full z/chi(z) factor modulates the slope
       and curvature of the smile across strikes.
    3. Degenerate chi numerator: raises ValueError to signal that the
       parameter combination is outside the stable approximation region.

    The time-correction term (1 + expansion * T) captures the leading-order
    effect of mean reversion in alpha. For T > 2 years the O(T^2) truncation
    error grows and a more exact formula (Obloj 2008 correction) may be
    preferable.

    Args:
        forward: Forward price of the underlying (F = S * exp((r-q)*T)).
        strike: Option strike price. Must be positive.
        expiry_years: Time to expiration in years. Must be positive.
        alpha: Initial instantaneous volatility. Sets the overall vol level
            for this expiry slice. Must be positive.
        beta: CEV elasticity exponent in [0, 1]. beta=0 is the normal
            (Bachelier) backbone; beta=1 is the lognormal (Black-76)
            backbone. For equities beta=1.0 is common; for interest rates
            beta=0.5 is the standard choice.
        rho: Correlation between forward and vol Brownian motions.
            Negative rho produces downward-sloping skew (equity-like put
            premium). Must be strictly inside (-1, 1).
        nu: Vol-of-vol parameter. Controls smile curvature. nu=0 degenerates
            to a flat (CEV) smile. Must be non-negative.

    Returns:
        Annualized implied Black-Scholes volatility as a decimal.

    Raises:
        ValueError: If forward, strike, or expiry_years are non-positive; if
            alpha is non-positive; if beta is outside [0, 1]; if rho is
            outside (-1, 1); if nu is negative; or if the approximation
            yields a non-positive or degenerate implied vol.

    Note:
        The Hagan approximation can produce slightly negative implied vols
        for very deep OTM strikes when nu is large and T is long. This
        raises ValueError. In practice, such strikes carry negligible
        liquidity and the approximation breaks down before the vol goes
        negative for sensible calibrations.
    """
    if forward <= 0.0:
        raise ValueError(f"forward must be positive, got {forward}")
    if strike <= 0.0:
        raise ValueError(f"strike must be positive, got {strike}")
    if expiry_years <= 0.0:
        raise ValueError(f"expiry_years must be positive, got {expiry_years}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative, got {nu}")

    one_minus_beta = 1.0 - beta
    log_fk = _math.log(forward / strike)
    fk_mid = (forward * strike) ** (one_minus_beta / 2.0)

    # Denominator: CEV backbone expansion in log(F/K).
    # Accounts for the curvature of the F^beta payoff between F and K.
    fk_denom = fk_mid * (
        1.0
        + (one_minus_beta ** 2 / 24.0) * log_fk ** 2
        + (one_minus_beta ** 4 / 1920.0) * log_fk ** 4
    )

    # Time-correction: leading-order O(T) adjustment for vol-of-vol effects.
    # This is the term inside the braces in equation (2.17) of Hagan et al.
    time_correction = 1.0 + (
        (one_minus_beta ** 2 / 24.0) * (alpha ** 2) / (fk_mid ** 2)
        + (rho * beta * nu * alpha / (4.0 * fk_mid))
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * expiry_years

    if nu == 0.0 or abs(log_fk) < 1e-7:
        # ATM or no stochastic vol: z/chi(z) -> 1, formula simplifies.
        sigma_b = (alpha / fk_denom) * time_correction
    else:
        # z parameterizes the distance from ATM in the SABR metric.
        z = (nu / alpha) * fk_mid * log_fk

        # chi(z): log-ratio of the two roots of the chi equation.
        disc = _math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0))
        chi_num = disc + z - rho
        if chi_num <= 0.0:
            raise ValueError(
                f"SABR approximation is degenerate for these parameters "
                f"(chi numerator={chi_num:.4f}). Reduce |rho| or |z| "
                f"(large log(F/K) combined with nu near |rho|*alpha can cause this)."
            )
        chi_z = _math.log(chi_num / (1.0 - rho))

        # z/chi(z): the ratio that corrects the flat-smile vol for skew.
        # The limit as chi_z -> 0 is 1.0 (handled above for near-ATM).
        z_over_chi = z / chi_z if abs(chi_z) > 1e-14 else 1.0

        sigma_b = (alpha / fk_denom) * z_over_chi * time_correction

    if sigma_b <= 0.0:
        raise ValueError(
            f"SABR approximation produced non-positive implied vol ({sigma_b:.8f}). "
            f"The parameter combination is likely outside the valid region for "
            f"this expiry (forward={forward}, strike={strike}, T={expiry_years}, "
            f"alpha={alpha}, beta={beta}, rho={rho}, nu={nu})."
        )

    return float(sigma_b)


def sabr(
    spot: float,
    strike: float,
    rate: float,
    expiry_years: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option under the SABR stochastic volatility model.

    Computes the SABR implied volatility via the Hagan et al. (2002)
    analytical approximation, then prices using the Black-76 formula on the
    risk-neutral forward price F = S * exp((r - q) * T):

        C = exp(-r*T) * [F * N(d1) - K * N(d2)]
        P = exp(-r*T) * [K * N(-d2) - F * N(-d1)]

    where d1 and d2 are computed from the SABR-derived implied vol.

    The SABR model is the industry standard for interest rate swaptions and
    FX options because calibrating four parameters (alpha, beta, rho, nu) to
    a single expiry slice is fast, reliable, and produces a smooth vol smile
    consistent with put-call parity. The model natively captures the negative
    skew (negative rho) and curvature (positive nu) observed in equity and
    rate option markets without the multi-dimensional numerical integration
    required by Heston.

    Beta controls the backbone: beta=1 gives a lognormal backbone (prices
    proportional to spot, consistent with Black-Scholes for single-strike
    fitting) while beta=0 gives a normal backbone (prices independent of
    spot level, more appropriate for near-zero rates). For equities, beta=1
    is the natural choice; for rate options, beta=0.5 is conventional.

    Args:
        spot: Current price of the underlying asset.
        strike: Option strike price. Must be positive.
        rate: Risk-free interest rate (annualized, as a decimal).
        expiry_years: Time to expiration in years. Must be positive.
        alpha: Initial instantaneous volatility level. Not directly comparable
            to the Black-Scholes vol: the relationship to ATM implied vol
            depends on beta and the ATM forward level.
        beta: CEV elasticity exponent in [0, 1].
        rho: Spot-vol correlation. Negative rho creates downward-sloping
            skew (OTM puts more expensive than OTM calls). Must be in (-1, 1).
        nu: Vol-of-vol. Controls smile curvature. nu=0 is a flat smile.
            Must be non-negative.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield (annualized, as a decimal).
            Defaults to 0.0.

    Returns:
        European option price. Always non-negative.

    Raises:
        ValueError: If any input fails validation or if the SABR
            approximation is degenerate for the given parameters.
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot}")
    if dividend_yield < 0.0:
        raise ValueError(f"dividend_yield must be non-negative, got {dividend_yield}")

    forward = spot * _math.exp((rate - dividend_yield) * expiry_years)

    # SABR implied vol (validates alpha, beta, rho, nu, strike, expiry internally)
    sigma_bs = sabr_implied_vol(forward, strike, expiry_years, alpha, beta, rho, nu)

    # Black-76 pricing: discount factor and standardized moneyness
    from scipy.stats import norm as _norm

    discount = _math.exp(-rate * expiry_years)
    sqrt_t = _math.sqrt(expiry_years)
    vol_sqrt_t = sigma_bs * sqrt_t

    if vol_sqrt_t < 1e-12:
        # Degenerate limit: return discounted intrinsic value
        if option_type == "call":
            return float(max(discount * (forward - strike), 0.0))
        return float(max(discount * (strike - forward), 0.0))

    d1 = (_math.log(forward / strike) + 0.5 * sigma_bs ** 2 * expiry_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "call":
        return float(discount * (forward * _norm.cdf(d1) - strike * _norm.cdf(d2)))
    return float(discount * (strike * _norm.cdf(-d2) - forward * _norm.cdf(-d1)))


# ---------------------------------------------------------------------------
# SABR calibration: fit (alpha, rho, nu) to a market-implied vol slice
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass
from typing import Sequence as _Sequence
from scipy import optimize as _optimize


@_dataclass(frozen=True)
class SABRCalibration:
    """Fitted SABR parameters from calibrating to a market implied vol slice.

    Attributes:
        alpha: Fitted initial volatility level. Sets the overall vol scale for
            this expiry slice; not directly comparable to Black-Scholes vol
            because the relationship depends on beta and the forward level.
        beta: Fixed CEV exponent supplied to calibrate_sabr (not optimized).
        rho: Fitted spot-vol correlation. Negative values produce a downward-
            sloping skew (OTM puts richer than OTM calls), typical for equity
            and FX options.
        nu: Fitted vol-of-vol. Controls smile curvature; nu=0 collapses to a
            flat CEV smile. Typical equity values are in the range 0.1 to 1.5.
        rmse: Root mean squared error between fitted SABR IVs and market IVs,
            in the same units as the input market_iv values (decimal vol,
            e.g. 0.005 means 0.5 vol-point RMSE).
        max_abs_error: Largest single-strike absolute IV error in decimal vol.
            Useful for detecting outlier strikes that the model fits poorly.
        n_points: Number of market quotes included in the calibration.
    """

    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float
    max_abs_error: float
    n_points: int


def calibrate_sabr(
    market_quotes: "_Sequence[tuple[float, float, float, float]]",
    beta: float = 1.0,
    alpha_init: float = 0.25,
    rho_init: float = -0.3,
    nu_init: float = 0.4,
) -> "SABRCalibration":
    """Fit SABR (alpha, rho, nu) to observed Black-Scholes implied volatilities.

    Holding beta fixed, solves:

        min_{alpha, rho, nu}  sum_i [sigma_SABR(F_i, K_i, T, alpha, beta, rho, nu)
                                     - sigma_market_i]^2

    using scipy.optimize.minimize with the L-BFGS-B method and explicit parameter
    bounds. Beta is fixed as a prior choice based on the underlying asset class;
    calibrating beta jointly with the other three parameters is ill-conditioned
    because rho and nu can partially compensate for changes in beta.

    Parameter values that produce degenerate SABR approximations (e.g. extreme
    rho combined with large log(F/K)) are penalized with a large surrogate error
    rather than propagating exceptions, so the optimizer steers away from invalid
    regions without aborting.

    Typical workflow: build a VolatilitySurface, extract a single-expiry slice
    with surface.smile(expiration), convert to (forward, strike, expiry_years,
    iv) tuples, and pass to this function. All quotes in market_quotes should
    share the same expiry; mixing expirations in a single call produces a
    meaningless fit because the time-correction term is expiry-specific.

    Args:
        market_quotes: Sequence of (forward, strike, expiry_years, market_iv)
            tuples. forward is the risk-neutral forward price for this expiry:
            F = S * exp((r - q) * T). market_iv must be a positive decimal
            (e.g. 0.25 for 25% implied vol). All tuples should share the same
            expiry_years within floating-point precision.
        beta: Fixed CEV exponent in [0, 1]. Defaults to 1.0 (lognormal backbone,
            appropriate for equities and FX). Use 0.5 for interest rate options
            or 0.0 for the normal backbone near zero rates.
        alpha_init: Initial guess for alpha. A good starting point is the ATM
            implied vol for the target expiry. Defaults to 0.25.
        rho_init: Initial guess for rho. Defaults to -0.3 (mild negative skew).
        nu_init: Initial guess for nu. Defaults to 0.4 (moderate curvature).

    Returns:
        SABRCalibration with fitted alpha, rho, nu and fit quality metrics
        (rmse, max_abs_error, n_points).

    Raises:
        ValueError: If market_quotes is empty; if beta is outside [0, 1]; or if
            any individual quote contains a non-positive forward, strike,
            expiry_years, or market_iv.
        RuntimeError: If the optimizer converges to a point with RMSE above 1%
            (100 bps), indicating a genuine calibration failure rather than
            routine optimizer tolerance noise.

    Example:
        Calibrate to a SPY near-dated expiry after building a surface:

            from optionview.fetcher import fetch_option_chain, fetch_spot_price
            from optionview.surface import build_surface
            from optionview.models import calibrate_sabr
            import math

            records = fetch_option_chain("SPY")
            spot = fetch_spot_price("SPY")
            rate = 0.05
            div_yield = 0.013

            surface = build_surface(records, spot, rate, div_yield)
            exp = surface.expirations[0]
            t = [p.expiry_years for p in surface.smile(exp)][0]
            forward = spot * math.exp((rate - div_yield) * t)

            quotes = [
                (forward, p.strike, p.expiry_years, p.iv)
                for p in surface.smile(exp)
            ]

            fit = calibrate_sabr(quotes, beta=1.0)
            print(f"alpha={fit.alpha:.4f}  rho={fit.rho:.4f}  nu={fit.nu:.4f}")
            print(f"RMSE={fit.rmse:.4f}  max_err={fit.max_abs_error:.4f}")
    """
    if not market_quotes:
        raise ValueError("market_quotes must be non-empty.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    for i, (fwd, k, t, iv) in enumerate(market_quotes):
        if fwd <= 0.0:
            raise ValueError(f"market_quotes[{i}]: forward must be positive, got {fwd}")
        if k <= 0.0:
            raise ValueError(f"market_quotes[{i}]: strike must be positive, got {k}")
        if t <= 0.0:
            raise ValueError(f"market_quotes[{i}]: expiry_years must be positive, got {t}")
        if iv <= 0.0:
            raise ValueError(f"market_quotes[{i}]: market_iv must be positive, got {iv}")

    n = len(market_quotes)
    # Large per-point penalty for parameter combinations that produce degenerate
    # SABR approximations; chosen to be 10x larger than a typical IV error.
    _PENALTY = 1.0

    def _objective(params: "np.ndarray") -> float:
        a, rho_, nu_ = float(params[0]), float(params[1]), float(params[2])
        sse = 0.0
        for fwd, k, t, market_iv in market_quotes:
            try:
                model_iv = sabr_implied_vol(fwd, k, t, a, beta, rho_, nu_)
                sse += (model_iv - market_iv) ** 2
            except (ValueError, ZeroDivisionError):
                sse += _PENALTY ** 2
        return sse

    bounds = [
        (1e-4, 5.0),     # alpha: strictly positive, cap prevents runaway fits
        (-0.999, 0.999), # rho: open interval (-1, 1)
        (1e-6, 5.0),     # nu: non-negative (small epsilon avoids ATM degeneracy test)
    ]
    x0 = np.array([alpha_init, rho_init, nu_init])

    result = _optimize.minimize(
        _objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-8, "maxiter": 2000},
    )

    # Accept the result unless RMSE exceeds 1% (100 bps), which signals a genuine
    # calibration failure rather than normal optimizer convergence noise.
    if not result.success and result.fun > n * 0.01 ** 2:
        raise RuntimeError(
            f"SABR calibration did not converge: {result.message}. "
            f"Final SSE={result.fun:.6f}. Verify that market_quotes spans a single "
            f"expiry and that market_iv values are in decimal form (e.g. 0.25, not 25)."
        )

    alpha_fit = float(result.x[0])
    rho_fit = float(result.x[1])
    nu_fit = float(result.x[2])

    errors = []
    for fwd, k, t, market_iv in market_quotes:
        try:
            model_iv = sabr_implied_vol(fwd, k, t, alpha_fit, beta, rho_fit, nu_fit)
            errors.append(abs(model_iv - market_iv))
        except (ValueError, ZeroDivisionError):
            errors.append(float(_PENALTY))

    rmse = float(_math.sqrt(sum(e ** 2 for e in errors) / n))
    max_abs = float(max(errors))

    return SABRCalibration(
        alpha=alpha_fit,
        beta=beta,
        rho=rho_fit,
        nu=nu_fit,
        rmse=rmse,
        max_abs_error=max_abs,
        n_points=n,
    )


# ---------------------------------------------------------------------------
# Heston calibration: fit all five parameters to a market-implied vol slice
# ---------------------------------------------------------------------------


@_dataclass(frozen=True)
class HestonCalibration:
    """Fitted Heston model parameters from calibrating to market implied volatilities.

    Attributes:
        v0: Fitted initial variance. The initial implied vol level is approximately
            sqrt(v0). For a 20% vol environment, expect v0 near 0.04.
        kappa: Fitted mean-reversion speed for the variance process. Higher kappa
            means variance reverts to theta faster. Typical equity values are 1-5.
        theta: Fitted long-run variance. The long-run implied vol is approximately
            sqrt(theta). For equity indices with 15-25% realized vol, expect
            theta near 0.02-0.06.
        sigma: Fitted vol-of-vol (volatility of the variance process). Controls
            smile curvature. Typical equity values are 0.2-0.8.
        rho: Fitted spot-variance correlation. Negative values produce the
            downward-sloping skew typical of equity index options.
            Equity indices typically calibrate rho in (-0.9, -0.3).
        rmse: Root mean squared error between Heston-implied IVs and input market
            IVs, in decimal vol units (e.g. 0.005 = 0.5 vol-point RMSE).
        max_abs_error: Largest single-strike absolute IV error in decimal vol.
            Useful for identifying strikes the model cannot fit well.
        n_points: Number of market quotes used in calibration.
        feller_satisfied: Whether the Feller condition (2 * kappa * theta > sigma^2)
            holds for the fitted parameters. When True, the variance process stays
            strictly positive. When False, v(t) can touch zero, reducing numerical
            precision for long expiries.
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    rmse: float
    max_abs_error: float
    n_points: int
    feller_satisfied: bool


def calibrate_heston(
    market_quotes: "_Sequence[tuple[float, float, float, float]]",
    v0_init: float = 0.04,
    kappa_init: float = 2.0,
    theta_init: float = 0.04,
    sigma_init: float = 0.3,
    rho_init: float = -0.5,
) -> "HestonCalibration":
    """Fit all five Heston model parameters to observed Black-Scholes implied volatilities.

    Minimizes the sum of squared price errors between Heston model prices and
    market prices derived from the input implied volatilities:

        min_{v0, kappa, theta, sigma, rho}
            sum_i [HestonPrice(F_i, K_i, T_i; params) - Black76Price(F_i, K_i, T_i; iv_i)]^2

    The objective uses price space (not IV space) to avoid the computational cost
    of a nested IV solve per optimizer iteration. After convergence, IV-space errors
    are computed once at the fitted parameters for reporting.

    Unlike SABR, which uses a closed-form IV approximation, Heston pricing requires
    numerical integration (Gil-Pelaez characteristic function inversion) per quote
    per optimizer iteration. Calibration is therefore slower than SABR for the same
    number of quotes. For a 10-strike smile slice, expect 10-60 seconds depending
    on integration tolerance and optimizer convergence speed.

    The Heston calibration surface is non-convex and the L-BFGS-B optimizer finds
    a local minimum from the supplied initial guess. For high-quality fits, consider
    running from multiple starting points (different v0_init and rho_init values)
    and selecting the result with the lowest RMSE. A common starting strategy is to
    set v0_init = theta_init = (ATM_vol)^2 for the nearest expiry.

    The pricing convention follows Black-76: each quote is represented as
    (forward, strike, expiry_years, market_iv) where forward = S * exp((r - q) * T).
    Heston is priced with rate=0 and spot=forward, which is correct under the
    forward measure and consistent with the calibrate_sabr interface.

    Unlike calibrate_sabr, which calibrates to a single expiry slice, calibrate_heston
    accepts quotes across multiple expiries. The Heston model prices each (T, K) pair
    independently via its characteristic function, so mixing expiries gives the
    optimizer more information about both the smile shape and term structure
    simultaneously.

    Args:
        market_quotes: Sequence of (forward, strike, expiry_years, market_iv) tuples.
            forward is the risk-neutral forward price for each expiry:
            F = S * exp((r - q) * T). market_iv must be a positive decimal
            (e.g. 0.25 for 25% implied vol). Quotes may span multiple expiries.
        v0_init: Initial guess for v0 (initial variance). A reasonable starting
            point is the square of the ATM implied vol for the nearest expiry.
            Defaults to 0.04 (equivalent to a 20% initial vol).
        kappa_init: Initial guess for the mean-reversion speed. Defaults to 2.0.
        theta_init: Initial guess for the long-run variance. Setting theta_init
            equal to v0_init is a natural starting point when the variance process
            is near its long-run level. Defaults to 0.04.
        sigma_init: Initial guess for vol-of-vol. Defaults to 0.3.
        rho_init: Initial guess for spot-variance correlation. Defaults to -0.5,
            which reflects mild negative skew typical of equity index options.

    Returns:
        HestonCalibration with fitted parameters and IV-space fit quality metrics.

    Raises:
        ValueError: If market_quotes is empty, or if any quote contains a
            non-positive forward, strike, expiry_years, or market_iv.
        RuntimeError: If the optimizer converges to a fit with RMSE above 5%
            (500 bps), indicating a genuine calibration failure. Common causes:
            very sparse quote coverage, market_iv values in percentage form
            (e.g. 25.0 instead of 0.25), or a market regime that requires
            better initial guesses.

    Example:
        Calibrate to a near-dated SPY smile slice::

            from optionview.fetcher import fetch_option_chain, fetch_spot_price
            from optionview.surface import build_surface
            from optionview.models import calibrate_heston
            import math

            records = fetch_option_chain("SPY")
            spot = fetch_spot_price("SPY")
            rate = 0.05
            div_yield = 0.013

            surface = build_surface(records, spot, rate, div_yield, min_open_interest=50)
            exp = surface.expirations[0]
            t = next(p.expiry_years for p in surface.smile(exp))
            forward = spot * math.exp((rate - div_yield) * t)

            quotes = [
                (forward, p.strike, p.expiry_years, p.iv)
                for p in surface.smile(exp)
            ]

            fit = calibrate_heston(quotes)
            print(f"v0={fit.v0:.4f}  kappa={fit.kappa:.4f}  theta={fit.theta:.4f}")
            print(f"sigma={fit.sigma:.4f}  rho={fit.rho:.4f}")
            print(f"RMSE={fit.rmse:.4f}  max_err={fit.max_abs_error:.4f}")
            print(f"Feller condition satisfied: {fit.feller_satisfied}")
    """
    if not market_quotes:
        raise ValueError("market_quotes must be non-empty.")

    for i, (fwd, k, t, iv) in enumerate(market_quotes):
        if fwd <= 0.0:
            raise ValueError(f"market_quotes[{i}]: forward must be positive, got {fwd}")
        if k <= 0.0:
            raise ValueError(f"market_quotes[{i}]: strike must be positive, got {k}")
        if t <= 0.0:
            raise ValueError(f"market_quotes[{i}]: expiry_years must be positive, got {t}")
        if iv <= 0.0:
            raise ValueError(f"market_quotes[{i}]: market_iv must be positive, got {iv}")

    n = len(market_quotes)

    # Pre-compute Black-76 market call prices from market IVs once.
    # Setting rate=0 with spot=forward converts black_scholes to Black-76:
    #   C = forward * N(d1) - strike * N(d2)
    # This is the forward-measure pricing convention that matches Heston at rate=0.
    market_prices: list[float] = [
        black_scholes(fwd, k, 0.0, iv, t, "call")
        for fwd, k, t, iv in market_quotes
    ]

    # Price-space penalty for parameter regions where Heston pricing raises an
    # exception. Set to 1.0 (roughly the full price of a near-ATM option at
    # typical strike/vol/expiry combinations), which strongly steers the optimizer
    # away from degenerate regions without dominating the loss landscape.
    _PRICE_PENALTY = 1.0

    def _objective(params: "np.ndarray") -> float:
        v0_, kappa_, theta_, sigma_, rho_ = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            float(params[4]),
        )
        sse = 0.0
        for (fwd, k, t, _), mkt_price in zip(market_quotes, market_prices):
            try:
                h_price = heston(
                    spot=fwd,
                    strike=k,
                    rate=0.0,
                    expiry_years=t,
                    v0=v0_,
                    kappa=kappa_,
                    theta=theta_,
                    sigma=sigma_,
                    rho=rho_,
                    option_type="call",
                )
                sse += (h_price - mkt_price) ** 2
            except Exception:
                sse += _PRICE_PENALTY ** 2
        return sse

    bounds = [
        (1e-6, 4.0),     # v0: variance; 4.0 corresponds to 200% vol (extreme upper bound)
        (0.1, 20.0),     # kappa: mean-reversion speed
        (1e-6, 4.0),     # theta: long-run variance
        (0.05, 3.0),     # sigma: vol-of-vol; lower bound prevents CF degeneracy
        (-0.999, 0.999), # rho: spot-variance correlation
    ]

    x0 = np.array([v0_init, kappa_init, theta_init, sigma_init, rho_init])
    for idx, (lo, hi) in enumerate(bounds):
        x0[idx] = float(np.clip(x0[idx], lo, hi))

    result = _optimize.minimize(
        _objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 3000},
    )

    v0_fit = float(result.x[0])
    kappa_fit = float(result.x[1])
    theta_fit = float(result.x[2])
    sigma_fit = float(result.x[3])
    rho_fit = float(result.x[4])

    # Compute IV-space errors at the fitted parameters.
    # The price-space objective is numerically efficient during optimization;
    # vol-point errors are more interpretable for users inspecting fit quality.
    errors: list[float] = []
    for fwd, k, t, market_iv in market_quotes:
        try:
            h_price = heston(
                spot=fwd,
                strike=k,
                rate=0.0,
                expiry_years=t,
                v0=v0_fit,
                kappa=kappa_fit,
                theta=theta_fit,
                sigma=sigma_fit,
                rho=rho_fit,
                option_type="call",
            )
            h_iv = implied_volatility(h_price, fwd, k, 0.0, t, "call")
            errors.append(abs(h_iv - market_iv))
        except (ValueError, RuntimeError):
            errors.append(float(_PRICE_PENALTY))

    rmse = float(_math.sqrt(sum(e ** 2 for e in errors) / n))
    max_abs = float(max(errors))

    # 5% (500 bps) RMSE threshold: beyond this, the calibration has failed in a
    # meaningful way. Normal L-BFGS-B convergence noise is well below this level.
    if rmse > 0.05:
        raise RuntimeError(
            f"Heston calibration did not converge to an acceptable fit: "
            f"RMSE={rmse:.4f} (threshold=0.05). Verify that market_iv values "
            f"are in decimal form (e.g. 0.25, not 25) and that the quote set "
            f"spans a liquid strike range. Consider supplying better initial "
            f"guesses via v0_init and rho_init for the target market."
        )

    feller = 2.0 * kappa_fit * theta_fit > sigma_fit ** 2

    return HestonCalibration(
        v0=v0_fit,
        kappa=kappa_fit,
        theta=theta_fit,
        sigma=sigma_fit,
        rho=rho_fit,
        rmse=rmse,
        max_abs_error=max_abs,
        n_points=n,
        feller_satisfied=feller,
    )
