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




@dataclass(frozen=True)
class ForwardVolPoint:
    """Implied forward volatility between two adjacent expiry slices.

    Derives the forward implied variance between two consecutive expirations
    from their ATM volatilities:

        var_fwd(T1, T2) = (sigma_far^2 * T_far - sigma_near^2 * T_near)
                          / (T_far - T_near)

    This is the unique constant variance that, compounded with the near-expiry
    variance, reproduces the far-expiry total variance. It is the vol-surface
    analog of the instantaneous forward rate in interest rate term structures.

    A non-negative forward variance is a necessary condition for the ATM term
    structure to be free of calendar spread arbitrage. If var_fwd < 0, a
    static arbitrage exists: buying the near-expiry straddle and selling the
    far-expiry straddle at the same strike locks in a risk-free profit
    regardless of the realized path.

    Attributes:
        near_expiry: Earlier expiration date.
        far_expiry: Later expiration date.
        near_atm_vol: ATM implied vol at near_expiry, as a decimal.
        far_atm_vol: ATM implied vol at far_expiry, as a decimal.
        near_years: Fractional years to near_expiry from the surface build date.
        far_years: Fractional years to far_expiry from the surface build date.
        forward_variance: Raw implied forward variance for the period
            [near_years, far_years]. May be negative when the term structure
            is inverted strongly enough to violate no-arbitrage.
        forward_vol: Square root of forward_variance, as a decimal.
            None if forward_variance is non-positive (arbitrage violation or
            degenerate case).
        is_arbitrage_free: True if forward_variance is strictly positive.
    """

    near_expiry: date
    far_expiry: date
    near_atm_vol: float
    far_atm_vol: float
    near_years: float
    far_years: float
    forward_variance: float
    forward_vol: float | None
    is_arbitrage_free: bool

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

    def interpolate_iv(self, expiration: date, log_moneyness: float) -> float:
        """Interpolate implied volatility at an arbitrary log-moneyness for a given expiry.

        Uses piecewise-linear interpolation across smile points sorted by log-moneyness.
        When the query falls outside the observed strike range, the IV of the nearest
        boundary point is returned (flat extrapolation). Flat extrapolation is the most
        conservative choice: it avoids introducing artificial slope at the tails that
        would understate the true cost of far-OTM options, and it makes no distributional
        assumptions beyond what the observed market quotes support.

        Piecewise-linear interpolation is preferred over parametric forms (SVI, SSVI)
        when the smile is used for pricing rather than calibration: it reproduces
        observed market IVs exactly at grid points, introduces no free parameters, and
        cannot create local extrema between data points that would generate static
        arbitrage.

        Args:
            expiration: The expiry slice to interpolate. Must be present in
                self.expirations.
            log_moneyness: Target log(K / F) value. Negative means the strike is
                below the risk-neutral forward (ITM call / OTM put); zero is
                ATM-forward; positive means the strike is above the forward.

        Returns:
            Interpolated implied volatility as a decimal (e.g. 0.22 for 22%).

        Raises:
            ValueError: If no surface points exist for the given expiration.
            ValueError: If fewer than two points are available for the expiration,
                which makes interpolation undefined (a single quote has no spread
                to interpolate across).
        """
        pts = self.smile(expiration)
        if not pts:
            raise ValueError(
                f"No surface points for expiration {expiration}. "
                f"Available expirations: {self.expirations}"
            )
        if len(pts) < 2:
            raise ValueError(
                f"Interpolation requires at least two smile points for expiration "
                f"{expiration}; only {len(pts)} available. Use atm_term_structure() "
                f"to read the single observed IV directly."
            )

        # Flat extrapolation beyond the observed strike range
        if log_moneyness <= pts[0].log_moneyness:
            return pts[0].iv
        if log_moneyness >= pts[-1].log_moneyness:
            return pts[-1].iv

        # Binary search for the bracketing interval [pts[lo], pts[hi]]
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid].log_moneyness <= log_moneyness:
                lo = mid
            else:
                hi = mid

        left, right = pts[lo], pts[hi]
        span = right.log_moneyness - left.log_moneyness

        # Guard against degenerate case: two distinct IVPoints at the same log-moneyness
        # (e.g. a call and put at the identical strike). Return the mean of their IVs.
        if span < 1e-12:
            return (left.iv + right.iv) * 0.5

        t = (log_moneyness - left.log_moneyness) / span
        return left.iv + t * (right.iv - left.iv)

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

    def forward_vol_curve(self) -> list[ForwardVolPoint]:
        """Compute implied forward volatilities across adjacent expiry pairs.

        Extracts ATM implied vol at each expiry using atm_term_structure()
        and derives the forward variance between every consecutive pair of
        expirations:

            var_fwd(T1, T2) = (sigma2^2 * T2 - sigma1^2 * T1) / (T2 - T1)

        The forward vol for each interval is sqrt(var_fwd) when var_fwd > 0.
        When var_fwd is zero or negative, forward_vol is set to None and
        is_arbitrage_free is False, indicating a calendar spread arbitrage
        in the observed ATM term structure.

        A common source of negative forward variance in practice is stale
        market data: if the near expiry is quoted with an artificially high
        IV but the far expiry reflects a normal level, the inferred forward
        variance can go negative. Filtering with min_open_interest during
        surface construction reduces this noise.

        The method reads expiry_years directly from the surface points rather
        than recomputing from today's date, so results are consistent with
        the surface's built_at timestamp.

        Returns:
            List of ForwardVolPoint objects for each adjacent expiry pair,
            ordered chronologically by near_expiry. Returns an empty list
            when fewer than two expirations are present on the surface.
        """
        atm = self.atm_term_structure()
        exps = sorted(atm.keys())
        if len(exps) < 2:
            return []

        exp_to_years: dict[date, float] = {}
        for pt in self.points:
            if pt.expiration not in exp_to_years:
                exp_to_years[pt.expiration] = pt.expiry_years

        result: list[ForwardVolPoint] = []
        for i in range(len(exps) - 1):
            near_exp = exps[i]
            far_exp = exps[i + 1]

            sigma_near = atm[near_exp]
            sigma_far = atm[far_exp]
            t_near = exp_to_years[near_exp]
            t_far = exp_to_years[far_exp]

            dt = t_far - t_near
            if dt <= 0.0:
                continue

            var_fwd = (sigma_far ** 2 * t_far - sigma_near ** 2 * t_near) / dt
            is_arb_free = var_fwd > 0.0
            fwd_vol: float | None = math.sqrt(var_fwd) if is_arb_free else None

            result.append(
                ForwardVolPoint(
                    near_expiry=near_exp,
                    far_expiry=far_exp,
                    near_atm_vol=sigma_near,
                    far_atm_vol=sigma_far,
                    near_years=t_near,
                    far_years=t_far,
                    forward_variance=var_fwd,
                    forward_vol=fwd_vol,
                    is_arbitrage_free=is_arb_free,
                )
            )

        return result



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
