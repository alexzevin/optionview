"""Portfolio-level Greeks aggregation for collections of option positions.

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

These conventions assume all positions share the same underlying. For mixed
underlyings, the net dollar figures should be interpreted with caution since
they aggregate sensitivities to different spot prices.
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
