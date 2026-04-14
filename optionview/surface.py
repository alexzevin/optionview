"""Implied volatility surface construction and analysis.

Organizes option chain data into a structured volatility surface indexed
by (expiration, log-moneyness) and provides tools for analyzing the shape
of the volatility smile and term structure.

The surface is constructed from OptionRecord objects returned by
fetch_option_chain() and supports both raw market-IV mode (using the
implied_volatility field directly from the data source) and mid-price
re-solve mode (recomputing IV from the bid/ask midpoint via Newton-Raphson).
Records with zero or missing quotes, insufficient open interest, or
near-expiry dates are filtered out for surface quality.

Key concepts:

    Log-moneyness: k = log(K / F) where F = S * exp((r - q) * T) is the
    at-the-money-forward price. Negative k means the strike is below the
    forward (ITM call / OTM put); positive k means the strike is above
    (OTM call / ITM put). Using the forward rather than spot makes IV
    levels comparable across expirations without spot-scaling bias.

    ATM point: the IVPoint with the smallest absolute log-moneyness for
    each expiration. Used as the reference level for smile construction
    and term structure extraction.

    Smile slope: linear regression slope of IV on log-moneyness across
    all contracts at a given expiry. Negative slope (IV decreasing with
    strike) is typical for equity index options and reflects the put
    premium or volatility skew. Positive slope can appear in commodity
    or single-stock options with strong upside demand.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Sequence

from optionview.fetcher import OptionRecord
from optionview.models import implied_volatility as _solve_iv


@dataclass(frozen=True)
class IVPoint:
    """A single point on the implied volatility surface.

    Attributes:
        expiration: Option expiration date.
        expiry_years: Fractional years to expiration from the reference date.
        strike: Option strike price.
        log_moneyness: log(K / F) where F is the risk-neutral forward price
            for this expiry. Zero means the strike is exactly at the forward.
        option_type: "call" or "put".
        iv: Implied volatility as a decimal (e.g. 0.25 for 25%).
        open_interest: Number of open contracts at this strike and expiry.
        bid: Market bid price.
        ask: Market ask price.
    """

    expiration: date
    expiry_years: float
    strike: float
    log_moneyness: float
    option_type: str
    iv: float
    open_interest: int
    bid: float
    ask: float


@dataclass(frozen=True)
class SmileSummary:
    """Summary statistics for the volatility smile at a single expiration.

    Attributes:
        expiration: Option expiration date.
        expiry_years: Fractional years to expiration.
        atm_iv: Implied volatility at the point closest to log-moneyness = 0.
        smile_slope: OLS slope of IV regressed on log-moneyness across all
            points for this expiry. Negative means IV is higher for low
            strikes (typical put-skew / downside-demand structure). None if
            fewer than two points are present.
        n_points: Number of IVPoints included in this smile.
        iv_range: (min_iv, max_iv) across all strikes for this expiry.
    """

    expiration: date
    expiry_years: float
    atm_iv: float
    smile_slope: float | None
    n_points: int
    iv_range: tuple[float, float]


@dataclass
class VolatilitySurface:
    """Implied volatility surface organized by expiration and log-moneyness.

    The surface holds all valid IVPoints filtered from a raw option chain
    and provides methods to inspect the shape of the smile and term
    structure. Points are stored sorted by (expiration, log_moneyness).

    Attributes:
        points: All IVPoints on the surface in (expiration, log_moneyness)
            order.
        spot: Spot price used when building the surface.
        rate: Risk-free rate used when building the surface.
        dividend_yield: Continuous dividend yield used when building the
            surface.
        built_at: UTC timestamp when the surface was constructed.
        n_filtered: Number of records discarded during surface construction
            (zero quotes, low OI, near expiry, zero IV, or solver failure).
    """

    points: list[IVPoint]
    spot: float
    rate: float
    dividend_yield: float
    built_at: datetime
    n_filtered: int

    @property
    def expirations(self) -> list[date]:
        """Sorted list of unique expiration dates present on the surface."""
        seen: set[date] = set()
        result: list[date] = []
        for p in self.points:
            if p.expiration not in seen:
                seen.add(p.expiration)
                result.append(p.expiration)
        return sorted(result)

    def smile(self, expiration: date) -> list[IVPoint]:
        """All IVPoints for a single expiration, sorted by log-moneyness.

        Args:
            expiration: The target expiration date.

        Returns:
            IVPoints for that expiry sorted from most negative to most
            positive log-moneyness (low strike to high strike in forward
            space). Returns an empty list if the expiration is absent.
        """
        return sorted(
            (p for p in self.points if p.expiration == expiration),
            key=lambda p: p.log_moneyness,
        )

    def atm_term_structure(self) -> dict[date, float]:
        """ATM implied volatility indexed by expiration date.

        For each expiry, selects the IVPoint with the smallest absolute
        log-moneyness and returns its IV as the ATM level. This is the
        at-the-money-forward vol; for most practical purposes it equals
        the vol observed near the current spot.

        Returns:
            Mapping from expiration date to ATM IV, sorted chronologically.
        """
        result: dict[date, float] = {}
        for exp in self.expirations:
            pts = self.smile(exp)
            if not pts:
                continue
            atm_pt = min(pts, key=lambda p: abs(p.log_moneyness))
            result[exp] = atm_pt.iv
        return result

    def smile_summary(self) -> list[SmileSummary]:
        """Summary statistics for each expiry's volatility smile.

        For every expiration on the surface, computes the ATM IV, OLS
        slope of IV on log-moneyness, point count, and IV range (min, max).

        The smile slope captures the overall tilt of the volatility smile.
        A strongly negative slope across all expirations is characteristic
        of equity index options, where downside puts are bid relative to
        upside calls.

        Returns:
            List of SmileSummary objects, one per expiration, sorted
            chronologically by expiration date.
        """
        summaries: list[SmileSummary] = []
        for exp in self.expirations:
            pts = self.smile(exp)
            if not pts:
                continue

            atm_pt = min(pts, key=lambda p: abs(p.log_moneyness))
            ivs = [p.iv for p in pts]
            moneynesses = [p.log_moneyness for p in pts]

            slope: float | None = None
            if len(pts) >= 2:
                try:
                    slope = _ols_slope(moneynesses, ivs)
                except ValueError:
                    slope = None

            summaries.append(
                SmileSummary(
                    expiration=exp,
                    expiry_years=atm_pt.expiry_years,
                    atm_iv=atm_pt.iv,
                    smile_slope=slope,
                    n_points=len(pts),
                    iv_range=(min(ivs), max(ivs)),
                )
            )
        return summaries


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute the ordinary-least-squares slope of ys regressed on xs.

    Uses the closed-form estimator: slope = cov(x, y) / var(x).
    This is equivalent to fitting a line through the (x, y) scatter
    but avoids unnecessary overhead for the one-variable case.

    Args:
        xs: Independent variable values (log-moneyness).
        ys: Dependent variable values (implied volatility).

    Returns:
        Slope coefficient (dIV / d_log_moneyness).

    Raises:
        ValueError: If xs has zero variance (all values identical), which
            would make the slope undefined.
    """
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)

    if var_x < 1e-14:
        raise ValueError(
            "xs has zero variance; OLS slope is undefined when all "
            "log-moneyness values are identical."
        )

    return cov_xy / var_x


def build_surface(
    records: list[OptionRecord],
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    min_open_interest: int = 0,
    min_expiry_days: int = 1,
    use_market_iv: bool = True,
    reference_date: date | None = None,
) -> VolatilitySurface:
    """Build an implied volatility surface from a list of option records.

    Filters out low-quality contracts, computes log-moneyness relative
    to the at-the-money forward for each expiry, and organizes the
    remaining IVPoints into a VolatilitySurface sorted by (expiration,
    log_moneyness).

    Filter order (each filtered record increments n_filtered):
      1. Zero-wide market: bid == 0 and ask == 0.
      2. Open interest below min_open_interest.
      3. Expiry in fewer than min_expiry_days calendar days.
      4. Non-positive forward price or non-positive strike.
      5. Zero or negative IV (in use_market_iv=True mode).
      6. Non-positive mid-price (in use_market_iv=False mode).
      7. IV solver failure (in use_market_iv=False mode).

    Args:
        records: Option records from fetch_option_chain(), optionally
            spanning multiple expirations.
        spot: Current spot price of the underlying.
        rate: Annualized risk-free rate as a decimal (e.g. 0.05 for 5%).
        dividend_yield: Continuous annualized dividend yield as a decimal.
            Defaults to 0.0.
        min_open_interest: Discard records with open_interest strictly below
            this value. Set to a positive value (e.g. 100) to exclude
            illiquid strikes that can distort the surface. Defaults to 0.
        min_expiry_days: Discard records expiring in fewer than this many
            calendar days. Must be at least 1 to avoid T = 0 degeneration.
            Defaults to 1.
        use_market_iv: If True, use the implied_volatility field from each
            OptionRecord directly. This is fast and suitable when the data
            source provides reliable IV. If False, solve IV from the
            bid/ask midpoint using Newton-Raphson; slower but more precise
            when market IV data is stale or absent. Defaults to True.
        reference_date: The date from which time-to-expiry is measured.
            Defaults to today's UTC date when None.

    Returns:
        VolatilitySurface with all valid IVPoints filtered, normalized
        to log-moneyness, and sorted by (expiration, log_moneyness).

    Raises:
        ValueError: If spot <= 0, rate < 0, or dividend_yield < 0.
    """
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if rate < 0:
        raise ValueError(f"Rate must be non-negative, got {rate}")
    if dividend_yield < 0:
        raise ValueError(f"Dividend yield must be non-negative, got {dividend_yield}")

    min_expiry_days = max(min_expiry_days, 1)
    today = reference_date if reference_date is not None else datetime.now(timezone.utc).date()

    points: list[IVPoint] = []
    n_filtered = 0

    for rec in records:
        # Filter 1: zero-wide market
        if rec.bid == 0.0 and rec.ask == 0.0:
            n_filtered += 1
            continue

        # Filter 2: open interest floor
        if rec.open_interest < min_open_interest:
            n_filtered += 1
            continue

        # Filter 3: near expiry
        days_to_exp = (rec.expiration - today).days
        if days_to_exp < min_expiry_days:
            n_filtered += 1
            continue

        expiry_years = days_to_exp / 365.0

        # Filter 4: degenerate forward or strike
        forward = spot * math.exp((rate - dividend_yield) * expiry_years)
        if forward <= 0.0 or rec.strike <= 0.0:
            n_filtered += 1
            continue

        log_moneyness = math.log(rec.strike / forward)

        if use_market_iv:
            # Filter 5: zero or missing IV in market-IV mode
            if rec.implied_volatility <= 0.0:
                n_filtered += 1
                continue
            iv = rec.implied_volatility
        else:
            mid = (rec.bid + rec.ask) / 2.0
            # Filter 6: non-positive mid
            if mid <= 0.0:
                n_filtered += 1
                continue
            # Filter 7: IV solver failure
            try:
                iv = _solve_iv(
                    market_price=mid,
                    spot=spot,
                    strike=rec.strike,
                    rate=rate,
                    expiry_years=expiry_years,
                    option_type=rec.option_type,  # type: ignore[arg-type]
                    dividend_yield=dividend_yield,
                )
            except (ValueError, RuntimeError):
                n_filtered += 1
                continue

        points.append(
            IVPoint(
                expiration=rec.expiration,
                expiry_years=expiry_years,
                strike=rec.strike,
                log_moneyness=log_moneyness,
                option_type=rec.option_type,
                iv=iv,
                open_interest=rec.open_interest,
                bid=rec.bid,
                ask=rec.ask,
            )
        )

    points.sort(key=lambda p: (p.expiration, p.log_moneyness))

    return VolatilitySurface(
        points=points,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        built_at=datetime.now(timezone.utc),
        n_filtered=n_filtered,
    )
