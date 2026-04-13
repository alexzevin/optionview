"""Model-vs-market comparison utilities for option chains.

Provides a single-pass comparison of all three pricing models against
observed market mid-prices, returning structured pricing errors and
relative deviations for each contract.

The mid-price (bid + ask) / 2 is used as the benchmark rather than the
last trade price, because the mid is a cleaner measure of prevailing
consensus value for liquid options. Last-trade prices can lag by hours in
low-volume strikes and introduce systematic noise into error metrics.

Filtering is applied before comparison to remove contracts where market
microstructure dominates the observable price:

  - Zero-wide markets (bid == 0 and ask == 0): no reliable quote.
  - Open interest below the caller-specified floor: stale or illiquid.
  - Expiry within one calendar day: near-zero T makes all models
    numerically degenerate and the market price reflects pin risk,
    not theoretical value.
  - Zero implied volatility (when using per-contract vol mode): the
    model cannot be evaluated without a vol input.

Convergence mode vs. surface mode
----------------------------------
When no explicit volatility is provided (the default), each contract is
evaluated at its own market-implied volatility. In this mode Black-Scholes
should reproduce the market price exactly (by definition of IV), so the
comparison quantifies how well BT and MC converge to the analytical
solution as a function of their discretization parameters. This is useful
for selecting step counts and simulation sizes.

When a single volatility is passed (surface mode), all contracts are
evaluated at the same vol. The comparison then shows which model better
fits market prices across strikes and expiries for that vol assumption,
useful for diagnosing skew and term-structure effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from optionview.fetcher import OptionRecord
from optionview.models import black_scholes, binomial_tree, monte_carlo


SkipReason = Literal[
    "zero_market",
    "low_open_interest",
    "expired_or_imminent",
    "zero_implied_volatility",
]


@dataclass(frozen=True)
class ComparisonResult:
    """Pricing comparison for a single option contract.

    Holds the market mid-price alongside theoretical prices from three
    models and the corresponding absolute and relative errors. Relative
    error is (model_price - market_mid) / market_mid, so a positive
    value means the model overprices the contract relative to the market.

    Attributes:
        record: Original OptionRecord from the fetcher.
        market_mid: Mid-point of the bid/ask spread, used as the market
            benchmark. Equal to (bid + ask) / 2.
        volatility_used: The annualized volatility passed to all three
            models. In per-contract mode this equals the OptionRecord's
            implied_volatility; in surface mode it is the caller-supplied
            override.
        expiry_years: Fractional years to expiry at the time of comparison.
        bs_price: Black-Scholes/Merton theoretical price.
        bt_price: Binomial tree (CRR) theoretical price.
        mc_price: Monte Carlo theoretical price.
        bs_error: Absolute error for Black-Scholes: bs_price - market_mid.
        bt_error: Absolute error for Binomial Tree: bt_price - market_mid.
        mc_error: Absolute error for Monte Carlo: mc_price - market_mid.
        bs_rel_error: Relative error for Black-Scholes.
        bt_rel_error: Relative error for Binomial Tree.
        mc_rel_error: Relative error for Monte Carlo.
        best_model: Name of the model with the smallest absolute error
            among the three. Ties broken in the order: bs, bt, mc.
    """

    record: OptionRecord
    market_mid: float
    volatility_used: float
    expiry_years: float
    bs_price: float
    bt_price: float
    mc_price: float
    bs_error: float
    bt_error: float
    mc_error: float
    bs_rel_error: float
    bt_rel_error: float
    mc_rel_error: float
    best_model: str


@dataclass(frozen=True)
class SkippedRecord:
    """A contract excluded from comparison with an explanation.

    Attributes:
        record: The original OptionRecord.
        reason: A SkipReason code indicating why the contract was excluded.
    """

    record: OptionRecord
    reason: SkipReason


@dataclass
class ComparisonReport:
    """Full comparison results for an option chain.

    Attributes:
        results: Successfully compared contracts, sorted by strike
            (ascending) then by option_type (call before put).
        skipped: Contracts excluded from comparison, in original order.
        spot: Spot price used for all model calls.
        rate: Risk-free rate used for all model calls.
        dividend_yield: Continuous dividend yield used.
        surface_vol: The uniform volatility override if one was supplied,
            or None when per-contract implied volatility was used.
        run_at: UTC timestamp when the comparison was performed.
        mean_abs_error: Mean absolute pricing error per model across all
            compared contracts. Keys are "bs", "bt", "mc".
        median_rel_error: Median relative pricing error per model, where
            the relative error is abs(model - market) / market. Keys are
            "bs", "bt", "mc". Reported as unsigned magnitude to avoid
            cancellation of over- and under-pricing.
        best_model_counts: Number of contracts where each model achieved
            the smallest absolute error. Keys are "bs", "bt", "mc".
    """

    results: list[ComparisonResult]
    skipped: list[SkippedRecord]
    spot: float
    rate: float
    dividend_yield: float
    surface_vol: float | None
    run_at: datetime
    mean_abs_error: dict[str, float]
    median_rel_error: dict[str, float]
    best_model_counts: dict[str, int]


def compare_to_market(
    chain: list[OptionRecord],
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    volatility: float | None = None,
    min_open_interest: int = 0,
    bt_steps: int = 200,
    mc_simulations: int = 100_000,
    mc_seed: int | None = None,
    reference_date: date | None = None,
) -> ComparisonReport:
    """Compare Black-Scholes, Binomial Tree, and Monte Carlo against market mids.

    Iterates over the provided option chain, applies quality filters, and
    runs all three pricing models against each surviving contract. Results
    are returned as a ComparisonReport with per-contract errors and
    aggregate statistics.

    Filter order (contracts failing any filter are recorded in
    ComparisonReport.skipped with an explicit reason):

      1. Zero-wide market (bid == 0 and ask == 0) -> "zero_market"
      2. Open interest below min_open_interest -> "low_open_interest"
      3. Expiry within one calendar day -> "expired_or_imminent"
      4. Per-contract IV == 0.0 when volatility is None -> "zero_implied_volatility"

    Args:
        chain: List of OptionRecord instances from fetch_option_chain().
        spot: Current underlying price. Use fetch_spot_price() for live data.
        rate: Annualized risk-free rate as a decimal (e.g. 0.05 for 5%).
        dividend_yield: Continuous annualized dividend yield as a decimal.
            Defaults to 0.0. Use fetch_dividend_yield() for live data.
        volatility: If provided, this single vol is used for every contract
            (surface mode). If None, each contract's own implied_volatility
            field is used (convergence mode). Defaults to None.
        min_open_interest: Contracts with open_interest strictly below this
            threshold are skipped. Defaults to 0 (only zero-OI contracts
            are skipped when min_open_interest=0).
        bt_steps: Number of time steps for the binomial tree. Higher values
            give tighter convergence to the Black-Scholes price at the cost
            of O(steps^2) work per contract. Defaults to 200.
        mc_simulations: Total simulation paths for Monte Carlo (half regular,
            half antithetic). Defaults to 100_000.
        mc_seed: Optional random seed for Monte Carlo reproducibility.
            Pass an integer for deterministic output. Defaults to None.
        reference_date: The date from which time-to-expiry is measured.
            Defaults to today's UTC date if None.

    Returns:
        ComparisonReport with per-contract results and aggregate statistics.

    Raises:
        ValueError: If spot or rate are non-positive, or if dividend_yield
            or volatility (when supplied) is negative.
    """
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    if rate < 0:
        raise ValueError(f"Risk-free rate must be non-negative, got {rate}")
    if dividend_yield < 0:
        raise ValueError(f"Dividend yield must be non-negative, got {dividend_yield}")
    if volatility is not None and volatility < 0:
        raise ValueError(f"Volatility must be non-negative, got {volatility}")

    today = reference_date if reference_date is not None else datetime.now(timezone.utc).date()

    results: list[ComparisonResult] = []
    skipped: list[SkippedRecord] = []

    for record in chain:
        # Filter 1: zero-wide market
        if record.bid == 0.0 and record.ask == 0.0:
            skipped.append(SkippedRecord(record=record, reason="zero_market"))
            continue

        # Filter 2: open interest floor
        if record.open_interest < min_open_interest:
            skipped.append(SkippedRecord(record=record, reason="low_open_interest"))
            continue

        # Filter 3: near-expiry (less than one calendar day)
        days_to_expiry = (record.expiration - today).days
        if days_to_expiry < 1:
            skipped.append(SkippedRecord(record=record, reason="expired_or_imminent"))
            continue

        expiry_years = days_to_expiry / 365.0

        # Resolve per-contract or uniform volatility
        if volatility is not None:
            vol = volatility
        else:
            vol = record.implied_volatility
            # Filter 4: zero IV in per-contract mode
            if vol == 0.0:
                skipped.append(
                    SkippedRecord(record=record, reason="zero_implied_volatility")
                )
                continue

        market_mid = (record.bid + record.ask) / 2.0

        bs = black_scholes(
            spot=spot,
            strike=record.strike,
            rate=rate,
            volatility=vol,
            expiry_years=expiry_years,
            option_type=record.option_type,  # type: ignore[arg-type]
            dividend_yield=dividend_yield,
        )
        bt = binomial_tree(
            spot=spot,
            strike=record.strike,
            rate=rate,
            volatility=vol,
            expiry_years=expiry_years,
            option_type=record.option_type,  # type: ignore[arg-type]
            steps=bt_steps,
            dividend_yield=dividend_yield,
        )
        mc = monte_carlo(
            spot=spot,
            strike=record.strike,
            rate=rate,
            volatility=vol,
            expiry_years=expiry_years,
            option_type=record.option_type,  # type: ignore[arg-type]
            simulations=mc_simulations,
            seed=mc_seed,
            dividend_yield=dividend_yield,
        )

        bs_err = bs - market_mid
        bt_err = bt - market_mid
        mc_err = mc - market_mid

        # Relative error: signed, so callers can distinguish over- vs. under-pricing.
        # Guard against zero mid (shouldn't occur after filter 1, but defensive).
        if market_mid > 0:
            bs_rel = bs_err / market_mid
            bt_rel = bt_err / market_mid
            mc_rel = mc_err / market_mid
        else:
            bs_rel = bt_rel = mc_rel = 0.0

        abs_errors = {"bs": abs(bs_err), "bt": abs(bt_err), "mc": abs(mc_err)}
        best = min(abs_errors, key=lambda k: abs_errors[k])

        results.append(
            ComparisonResult(
                record=record,
                market_mid=market_mid,
                volatility_used=vol,
                expiry_years=expiry_years,
                bs_price=bs,
                bt_price=bt,
                mc_price=mc,
                bs_error=bs_err,
                bt_error=bt_err,
                mc_error=mc_err,
                bs_rel_error=bs_rel,
                bt_rel_error=bt_rel,
                mc_rel_error=mc_rel,
                best_model=best,
            )
        )

    # Sort by strike ascending, then call before put
    _type_order = {"call": 0, "put": 1}
    results.sort(key=lambda r: (r.record.strike, _type_order.get(r.record.option_type, 2)))

    # Aggregate statistics
    n = len(results)
    if n > 0:
        import statistics

        mean_abs: dict[str, float] = {
            "bs": sum(abs(r.bs_error) for r in results) / n,
            "bt": sum(abs(r.bt_error) for r in results) / n,
            "mc": sum(abs(r.mc_error) for r in results) / n,
        }
        median_rel: dict[str, float] = {
            "bs": statistics.median(abs(r.bs_rel_error) for r in results),
            "bt": statistics.median(abs(r.bt_rel_error) for r in results),
            "mc": statistics.median(abs(r.mc_rel_error) for r in results),
        }
        counts: dict[str, int] = {"bs": 0, "bt": 0, "mc": 0}
        for r in results:
            counts[r.best_model] += 1
    else:
        mean_abs = {"bs": 0.0, "bt": 0.0, "mc": 0.0}
        median_rel = {"bs": 0.0, "bt": 0.0, "mc": 0.0}
        counts = {"bs": 0, "bt": 0, "mc": 0}

    return ComparisonReport(
        results=results,
        skipped=skipped,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        surface_vol=volatility,
        run_at=datetime.now(timezone.utc),
        mean_abs_error=mean_abs,
        median_rel_error=median_rel,
        best_model_counts=counts,
    )
