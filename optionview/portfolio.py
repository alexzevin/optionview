"""Portfolio-level Greeks aggregation and scenario P&L analysis.

Aggregates analytical Greeks across a set of option positions, accounting
for position size and direction (long or short). Each position is described
by its contract parameters and a signed quantity: positive quantities
represent long positions (bought options), negative quantities represent
short positions (sold options).

Aggregation procedure:
  1. Compute unit Greeks for each contract using the Black-Scholes/Merton model.
  2. Scale each Greek by the signed quantity for that position.
  3. Sum scaled Greeks across all positions to obtain net portfolio exposure.

Dollar Greeks are provided alongside unit Greeks:
  - Dollar delta: delta * spot * quantity. Approximates P&L (in dollars per
    contract) for a $1 absolute move in spot, ignoring higher-order terms.
  - Dollar gamma: 0.5 * gamma * spot^2 * quantity. The second-order P&L
    contribution for a $1 absolute move in spot. A large positive dollar
    gamma means the portfolio benefits disproportionately from large moves.

Scenario P&L:
  scenario_pnl applies a second-order Taylor expansion to estimate portfolio
  P&L for a hypothetical simultaneous move in spot, implied volatility, and
  time. The expansion includes delta, gamma, vega, theta, and two cross-
  sensitivities (vanna and charm) for a more complete attribution than a
  pure first-order estimate. This is useful for stress testing and
  "what-if" analysis across Greek components.

These conventions assume all positions share the same underlying. For mixed
underlyings, net dollar figures and scenario P&L should be interpreted with
caution since they aggregate sensitivities to different spot prices.
"""

from __future__ import annotations

from dataclasses import dataclass

from optionview.greeks import compute_greeks
from optionview.models import OptionType


_GREEK_KEYS: tuple[str, ...] = (
    "delta", "gamma", "theta", "vega", "rho", "epsilon", "vanna", "charm"
)


@dataclass(frozen=True)
class Position:
    """A single option position defined by its contract parameters and quantity.

    Attributes:
        spot: Current price of the underlying asset.
        strike: Option strike price.
        rate: Risk-free interest rate (annualized, as a decimal).
        volatility: Implied or assumed annualized volatility (as a decimal).
        expiry_years: Time to expiration in years.
        option_type: "call" or "put".
        quantity: Signed position size. Positive means long (bought options),
            negative means short (sold options). For example, quantity=10
            means 10 contracts long; quantity=-5 means 5 contracts short.
        dividend_yield: Continuous dividend yield (annualized, as a decimal).
            Defaults to 0.0 (no dividends). Pass fetch_dividend_yield() output
            for equity options with known yield assumptions.
        label: Optional human-readable identifier for this position. Used in
            per-position output for logging or display; has no effect on
            computation.
    """

    spot: float
    strike: float
    rate: float
    volatility: float
    expiry_years: float
    option_type: OptionType
    quantity: float
    dividend_yield: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class PositionRisk:
    """Risk metrics for a single option position, scaled by quantity.

    Attributes:
        position: The source Position object.
        unit_greeks: Greeks for a single contract at quantity=1. Keys match
            compute_greeks output: delta, gamma, theta, vega, rho, epsilon,
            vanna, charm.
        scaled_greeks: unit_greeks multiplied element-wise by position.quantity.
            Negative quantities invert sign, reflecting the mirror-image risk
            of short positions (e.g., short call has negative delta and vega).
        dollar_delta: unit_delta * spot * quantity. P&L for a $1 move in spot.
        dollar_gamma: 0.5 * unit_gamma * spot^2 * quantity. P&L due to the
            quadratic spot term for a $1 absolute move.
    """

    position: Position
    unit_greeks: dict[str, float]
    scaled_greeks: dict[str, float]
    dollar_delta: float
    dollar_gamma: float


@dataclass(frozen=True)
class PortfolioRisk:
    """Aggregated risk for a collection of option positions.

    Attributes:
        positions: Per-position risk objects in input order, one per Position
            passed to aggregate_greeks.
        net_greeks: Sum of scaled_greeks across all positions. Same keys as
            compute_greeks: delta, gamma, theta, vega, rho, epsilon, vanna,
            charm. Represents total Greek exposure of the portfolio.
        net_dollar_delta: Sum of dollar_delta across all positions.
        net_dollar_gamma: Sum of dollar_gamma across all positions.
        n_positions: Number of positions in the portfolio (equals
            len(positions)).
    """

    positions: tuple[PositionRisk, ...]
    net_greeks: dict[str, float]
    net_dollar_delta: float
    net_dollar_gamma: float
    n_positions: int


@dataclass(frozen=True)
class ScenarioPnL:
    """Estimated portfolio P&L for a hypothetical market scenario.

    Computed via a second-order Taylor expansion of portfolio value with
    respect to spot (delta, gamma), implied volatility (vega), calendar
    time (theta), and two cross-sensitivities (vanna, charm).

    The expansion captures the dominant sources of P&L over short horizons
    without requiring a full reprice of every position. Cross-sensitivity
    terms (vanna and charm) are small for small moves but become material
    when both dimensions move simultaneously (e.g., large spot gap AND vol
    shift, common during macro events).

    Attributes:
        delta_pnl: P&L from the first-order spot move: delta * ds.
        gamma_pnl: P&L from second-order convexity: 0.5 * gamma * ds^2.
            Always non-negative for a long-gamma book, regardless of move
            direction. Positive gamma benefits from both up and down moves.
        vega_pnl: P&L from the implied vol shift. Vega is stored per 1%
            absolute vol move, so this scales as vega * (dvol / 0.01).
        theta_pnl: P&L from time decay: theta * dt_days. Theta is negative
            for net long option books; dt_days positive means losing value.
        vanna_pnl: Cross-sensitivity P&L: vanna * ds * dvol. Measures how
            much P&L arises when both spot and vol move together. Relevant
            for books with significant skew exposure.
        charm_pnl: Cross-sensitivity P&L: charm * ds * dt_days. Measures
            the additional spot sensitivity accumulated as delta drifts over
            dt_days calendar days, applied to the spot move ds.
        total_pnl: Arithmetic sum of all six components.
        ds: Spot shift used (same units as Position.spot).
        dvol: Implied vol shift in decimal units (e.g. 0.01 = +1 vol point).
        dt_days: Calendar days elapsed (non-negative).
    """

    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    vanna_pnl: float
    charm_pnl: float
    total_pnl: float
    ds: float
    dvol: float
    dt_days: float


def scenario_pnl(
    risk: PortfolioRisk,
    ds: float = 0.0,
    dvol: float = 0.0,
    dt_days: float = 0.0,
) -> ScenarioPnL:
    """Estimate portfolio P&L under a simultaneous market scenario.

    Applies a second-order Taylor expansion using the net Greeks from a
    PortfolioRisk object. The expansion decomposes P&L into six named
    components, each attributable to a specific Greek:

        P&L = delta*ds
            + 0.5*gamma*ds^2
            + vega*(dvol/0.01)
            + theta*dt_days
            + vanna*ds*dvol
            + charm*ds*dt_days

    Greek unit conventions (inherited from compute_greeks):
      - delta: price change per $1 spot move (dimensionless fraction).
      - gamma: delta change per $1 spot move.
      - vega: price change per 1% absolute vol move (0.01 in decimal).
        Dividing dvol by 0.01 converts the shift to the same scale.
      - theta: price change per calendar day (negative for long options).
      - vanna: delta change per unit sigma (decimal). Multiplied by ds*dvol.
      - charm: delta change per calendar day. Multiplied by ds*dt_days.

    The expansion is accurate to first order in dvol and dt_days and to
    second order in ds. It does not account for vol-of-vol (volga),
    higher-order spot terms, or carry effects. For large moves (ds/spot > 5%)
    or long horizons (dt_days > 30), a full reprice is more reliable.

    Args:
        risk: Portfolio risk from aggregate_greeks, providing net Greeks.
        ds: Spot price shift in the same units as Position.spot. Positive
            for a spot increase, negative for a decrease. Defaults to 0.
        dvol: Implied volatility shift in decimal units (e.g. 0.01 for +1
            vol point, -0.02 for -2 vol points). Defaults to 0.
        dt_days: Calendar days elapsed. Must be non-negative. Defaults to 0.

    Returns:
        ScenarioPnL with per-Greek P&L contributions and total.

    Raises:
        ValueError: If dt_days is negative.

    Example:
        Estimate the P&L of a straddle if spot rises $5, vol drops 2%,
        and one day passes::

            from optionview.portfolio import Position, aggregate_greeks, scenario_pnl

            positions = [
                Position(spot=100, strike=100, rate=0.05, volatility=0.25,
                         expiry_years=0.25, option_type="call", quantity=10),
                Position(spot=100, strike=100, rate=0.05, volatility=0.25,
                         expiry_years=0.25, option_type="put", quantity=10),
            ]
            risk = aggregate_greeks(positions)
            pnl = scenario_pnl(risk, ds=5.0, dvol=-0.02, dt_days=1.0)
            print(f"Delta P&L:  {pnl.delta_pnl:+.2f}")
            print(f"Gamma P&L:  {pnl.gamma_pnl:+.2f}")
            print(f"Vega P&L:   {pnl.vega_pnl:+.2f}")
            print(f"Theta P&L:  {pnl.theta_pnl:+.2f}")
            print(f"Total P&L:  {pnl.total_pnl:+.2f}")
    """
    if dt_days < 0:
        raise ValueError(f"dt_days must be non-negative, got {dt_days}")

    g = risk.net_greeks

    delta_pnl = g["delta"] * ds
    gamma_pnl = 0.5 * g["gamma"] * ds ** 2
    # vega stored per 1% absolute vol move (0.01 decimal); scale to actual dvol
    vega_pnl = g["vega"] * (dvol / 0.01)
    theta_pnl = g["theta"] * dt_days
    vanna_pnl = g["vanna"] * ds * dvol
    charm_pnl = g["charm"] * ds * dt_days

    total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl + charm_pnl

    return ScenarioPnL(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        vanna_pnl=vanna_pnl,
        charm_pnl=charm_pnl,
        total_pnl=total,
        ds=ds,
        dvol=dvol,
        dt_days=dt_days,
    )


def aggregate_greeks(positions: list[Position]) -> PortfolioRisk:
    """Compute and aggregate Greeks across a portfolio of option positions.

    For each position, unit Greeks are computed via Black-Scholes/Merton and
    scaled by the signed quantity. Net portfolio Greeks are the element-wise
    sum of scaled Greeks across all positions. An empty input list returns a
    PortfolioRisk with zero net exposure.

    The function supports portfolios where each position may have a different
    strike, expiry, option type, or volatility assumption. This makes it
    suitable for analyzing realistic multi-leg strategies such as straddles,
    strangles, verticals, calendars, and delta-hedged books.

    Args:
        positions: List of Position objects defining the portfolio. An empty
            list is valid and returns all-zero net Greeks.

    Returns:
        PortfolioRisk with per-position detail and aggregated net exposure.

    Raises:
        ValueError: If any position has invalid parameters (non-positive spot
            or strike, negative volatility or dividend yield, non-positive
            expiry). The error is raised immediately on the offending position
            without processing subsequent positions.
    """
    net: dict[str, float] = {k: 0.0 for k in _GREEK_KEYS}
    net_dollar_delta = 0.0
    net_dollar_gamma = 0.0
    position_risks: list[PositionRisk] = []

    for pos in positions:
        unit = compute_greeks(
            spot=pos.spot,
            strike=pos.strike,
            rate=pos.rate,
            volatility=pos.volatility,
            expiry_years=pos.expiry_years,
            option_type=pos.option_type,
            dividend_yield=pos.dividend_yield,
        )
        scaled = {k: unit[k] * pos.quantity for k in _GREEK_KEYS}

        dollar_delta = unit["delta"] * pos.spot * pos.quantity
        dollar_gamma = 0.5 * unit["gamma"] * pos.spot ** 2 * pos.quantity

        position_risks.append(
            PositionRisk(
                position=pos,
                unit_greeks=unit,
                scaled_greeks=scaled,
                dollar_delta=dollar_delta,
                dollar_gamma=dollar_gamma,
            )
        )

        for k in _GREEK_KEYS:
            net[k] += scaled[k]
        net_dollar_delta += dollar_delta
        net_dollar_gamma += dollar_gamma

    return PortfolioRisk(
        positions=tuple(position_risks),
        net_greeks=net,
        net_dollar_delta=net_dollar_delta,
        net_dollar_gamma=net_dollar_gamma,
        n_positions=len(positions),
    )
