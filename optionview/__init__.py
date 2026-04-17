"""optionview - Compare option pricing models against live market data."""

__version__ = "0.1.4"

from optionview.models import black_scholes, binomial_tree, monte_carlo, implied_volatility
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
    Position,
    PositionRisk,
    PortfolioRisk,
)

__all__ = [
    "black_scholes",
    "binomial_tree",
    "monte_carlo",
    "implied_volatility",
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
    "Position",
    "PositionRisk",
    "PortfolioRisk",
]
