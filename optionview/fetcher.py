"""Market data retrieval from free public APIs.

Uses Yahoo Finance (via yfinance) to fetch live option chains
and underlying asset data without requiring authentication.
"""

from dataclasses import dataclass
from datetime import date

import yfinance as yf


@dataclass(frozen=True)
class OptionRecord:
    """A single option contract with market data and metadata."""

    symbol: str
    expiration: date
    strike: float
    option_type: str
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float


def fetch_option_chain(
    ticker: str,
    expiration: str | None = None,
) -> list[OptionRecord]:
    """Fetch the option chain for a given ticker symbol.

    Retrieves call and put data from Yahoo Finance. No API key
    or login is required.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "SPY").
        expiration: Specific expiration date as "YYYY-MM-DD".
            If None, uses the nearest available expiration.

    Returns:
        List of OptionRecord dataclass instances, one per contract.

    Raises:
        ValueError: If the ticker is invalid or no expirations are found.
    """
    asset = yf.Ticker(ticker)
    expirations = asset.options

    if not expirations:
        raise ValueError(
            f"No option expirations found for ticker '{ticker}'. "
            "Verify the symbol is correct and has listed options."
        )

    if expiration is not None:
        if expiration not in expirations:
            raise ValueError(
                f"Expiration '{expiration}' not available for '{ticker}'. "
                f"Available: {', '.join(expirations[:5])}..."
            )
        target_exp = expiration
    else:
        target_exp = expirations[0]

    chain = asset.option_chain(target_exp)
    records: list[OptionRecord] = []

    for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
        for _, row in df.iterrows():
            records.append(
                OptionRecord(
                    symbol=ticker.upper(),
                    expiration=date.fromisoformat(target_exp),
                    strike=float(row.get("strike", 0)),
                    option_type=opt_type,
                    last_price=float(row.get("lastPrice", 0)),
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    volume=int(row.get("volume", 0) or 0),
                    open_interest=int(row.get("openInterest", 0) or 0),
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                )
            )

    return records


def fetch_spot_price(ticker: str) -> float:
    """Fetch the current spot price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Current market price as a float.

    Raises:
        ValueError: If the ticker is invalid or price is unavailable.
    """
    asset = yf.Ticker(ticker)
    info = asset.fast_info

    price = getattr(info, "last_price", None)
    if price is None:
        raise ValueError(f"Could not retrieve spot price for '{ticker}'")

    return float(price)


def list_expirations(ticker: str) -> list[str]:
    """List all available option expiration dates for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of expiration date strings in "YYYY-MM-DD" format.
    """
    asset = yf.Ticker(ticker)
    return list(asset.options)
