"""Market data retrieval from free public APIs.

Uses Yahoo Finance (via yfinance) to fetch live option chains,
spot prices, and dividend yield data without requiring authentication.
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
                f"Available: {list(expirations[:5])}..."
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


def fetch_dividend_yield(ticker: str) -> float:
    """Fetch the trailing annual dividend yield for a ticker as a decimal.

    Returns the trailing twelve-month dividend yield expressed as a
    continuous rate approximation, suitable for use as the dividend_yield
    parameter in the pricing models and Greeks functions.

    The value is sourced from Yahoo Finance's reported trailing annual
    dividend yield (e.g. 0.015 represents 1.5% annual yield). For stocks
    that pay no dividends, 0.0 is returned rather than raising an error.

    Note: Yahoo Finance reports a discrete annualized yield; for high-yield
    securities over short time horizons the continuous approximation is close
    but not exact. Convert with q_continuous = -log(1 - q_discrete * T) / T
    if precision over a specific horizon matters.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "SPY").

    Returns:
        Trailing annual dividend yield as a decimal (e.g. 0.015 for 1.5%).
        Returns 0.0 if the ticker pays no dividend or data is unavailable.

    Raises:
        ValueError: If the ticker symbol is not recognized by Yahoo Finance.
    """
    asset = yf.Ticker(ticker)

    try:
        info = asset.info
    except Exception as exc:
        raise ValueError(
            f"Could not retrieve data for ticker '{ticker}': {exc}"
        ) from exc

    if not info or info.get("quoteType") is None:
        raise ValueError(
            f"Ticker '{ticker}' not recognized. Verify the symbol is valid."
        )

    # Yahoo Finance may report the yield under either key depending on version
    yield_value = (
        info.get("trailingAnnualDividendYield")
        or info.get("dividendYield")
        or 0.0
    )

    return float(yield_value)


def list_expirations(ticker: str) -> list[str]:
    """List all available option expiration dates for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        List of expiration date strings in "YYYY-MM-DD" format.
    """
    asset = yf.Ticker(ticker)
    return list(asset.options)
