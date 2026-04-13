"""optionview - Compare option pricing models against live market data."""

__version__ = "0.1.2"

from optionview.models import black_scholes, binomial_tree, monte_carlo, implied_volatility
from optionview.greeks import compute_greeks
from optionview.fetcher import fetch_option_chain, fetch_spot_price, fetch_dividend_yield
from optionview.compare import (
    compare_to_market,
    ComparisonResult,
    ComparisonReport,
    SkippedRecord,
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
    "compare_to_market",
    "ComparisonResult",
    "ComparisonReport",
    "SkippedRecord",
]
