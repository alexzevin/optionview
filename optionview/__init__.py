"""optionview - Compare option pricing models against live market data."""

__version__ = "0.1.7"

from optionview.models import black_scholes, binomial_tree, monte_carlo, implied_volatility, heston, sabr, sabr_implied_vol
from optionview.greeks import compute_greeks
from optionview.fetcher import (
    fetch_option_chain,
    fetch_spot_price,
    fetch_dividend_yield,
    list_expirations,
)
from optionview.compare import (
    compare_to_market,
    ComparisonResult,
    ComparisonReport,
    SkippedRecord,
)
from optionview.surface import (
    build_surface,
    VolatilitySurface,
    IVPoint,
    SmileSummary,
)
from optionview.portfolio import (
    aggregate_greeks,
    scenario_pnl,
    Position,
    PositionRisk,
    PortfolioRisk,
    ScenarioPnL,
)

__all__ = [
    "black_scholes",
    "binomial_tree",
    "monte_carlo",
    "implied_volatility",
    "heston",
    "sabr",
    "sabr_implied_vol",
    "compute_greeks",
    "fetch_option_chain",
    "fetch_spot_price",
    "fetch_dividend_yield",
    "list_expirations",
    "compare_to_market",
    "ComparisonResult",
    "ComparisonReport",
    "SkippedRecord",
    "build_surface",
    "VolatilitySurface",
    "IVPoint",
    "SmileSummary",
    "aggregate_greeks",
    "scenario_pnl",
    "Position",
    "PositionRisk",
    "PortfolioRisk",
    "ScenarioPnL",
]
