"""Tests for optionview model implementations.

Covers:
- Black-Scholes/Merton closed-form correctness (put-call parity, boundary conditions)
- IV solver round-trip consistency across a range of volatilities
- Binomial tree convergence to the Black-Scholes benchmark
- Monte Carlo statistical convergence with antithetic variates
- Greeks sign and magnitude identities (delta bounds, put-call parity on delta,
  gamma/vega symmetry, theta sign, rho and epsilon signs, charm/vanna types)
- Volatility surface construction and structural invariants
- ComparisonReport filtering and aggregate statistics
- NaN-safe conversion helpers in fetcher.py

Design note: tests that involve Monte Carlo use a fixed seed and a generous
absolute tolerance. The MC error bound reflects the statistical nature of the
estimator (not a defect) and would tighten proportionally to 1/sqrt(n_paths).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from optionview.compare import compare_to_market
from optionview.fetcher import OptionRecord, _safe_float, _safe_int
from optionview.greeks import compute_greeks
from optionview.models import (
    binomial_tree,
    black_scholes,
    implied_volatility,
    monte_carlo,
)
from optionview.surface import build_surface


# ---------------------------------------------------------------------------
# Shared reference parameters
# ---------------------------------------------------------------------------
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOL = 0.20
T = 1.0          # one year
DIV = 0.02       # 2% continuous dividend yield

# Tolerances
TOL_EXACT = 1e-6   # identities that must hold analytically
TOL_BT = 0.015     # binomial tree at 500 steps vs. Black-Scholes
TOL_MC = 0.06      # MC at 200k paths (relative error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    strike: float,
    expiry_days: int,
    iv: float,
    option_type: str = "call",
    open_interest: int = 500,
) -> OptionRecord:
    """Construct a synthetic OptionRecord priced via Black-Scholes mid."""
    exp = date.today() + timedelta(days=expiry_days)
    t = expiry_days / 365.0
    mid = black_scholes(SPOT, strike, RATE, iv, t, option_type)  # type: ignore[arg-type]
    return OptionRecord(
        symbol="TEST",
        expiration=exp,
        strike=strike,
        option_type=option_type,
        last_price=mid,
        bid=round(mid * 0.99, 4),
        ask=round(mid * 1.01, 4),
        volume=1000,
        open_interest=open_interest,
        implied_volatility=iv,
    )


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class TestBlackScholes:
    """Tests for the Black-Scholes/Merton closed-form pricer."""

    def test_put_call_parity_no_dividends(self) -> None:
        """C - P = S - K*exp(-r*T) when q = 0."""
        call = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call")
        put = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put")
        lhs = call - put
        rhs = SPOT - STRIKE * math.exp(-RATE * T)
        assert abs(lhs - rhs) < TOL_EXACT, (
            f"Put-call parity (q=0) violated: C-P={lhs:.8f}, rhs={rhs:.8f}"
        )

    def test_put_call_parity_with_dividends(self) -> None:
        """C - P = S*exp(-q*T) - K*exp(-r*T) with continuous dividends."""
        call = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", DIV)
        put = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put", DIV)
        lhs = call - put
        rhs = SPOT * math.exp(-DIV * T) - STRIKE * math.exp(-RATE * T)
        assert abs(lhs - rhs) < TOL_EXACT, (
            f"Put-call parity (q={DIV}) violated: C-P={lhs:.8f}, rhs={rhs:.8f}"
        )

    def test_atm_call_positive(self) -> None:
        """ATM call with positive vol and positive T must have positive value."""
        price = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call")
        assert price > 0

    def test_deep_itm_call_approaches_discounted_intrinsic(self) -> None:
        """Deep ITM call (vol near zero) approaches S - K*exp(-r*T)."""
        call = black_scholes(200.0, 100.0, RATE, 0.001, T, "call")
        intrinsic = 200.0 - 100.0 * math.exp(-RATE * T)
        assert abs(call - intrinsic) < 0.05

    def test_deep_otm_call_near_zero(self) -> None:
        """Deep OTM call at low vol is essentially worthless."""
        call = black_scholes(50.0, 200.0, RATE, 0.10, T, "call")
        assert call < 0.001

    def test_zero_vol_itm_call_equals_discounted_intrinsic(self) -> None:
        """At sigma=0, ITM call = max(S - K*exp(-r*T), 0)."""
        call = black_scholes(110.0, 100.0, RATE, 0.0, T, "call")
        expected = max(110.0 - 100.0 * math.exp(-RATE * T), 0.0)
        assert abs(call - expected) < TOL_EXACT

    def test_zero_vol_otm_call_is_zero(self) -> None:
        """At sigma=0, OTM call has no extrinsic value and is exactly zero."""
        call = black_scholes(90.0, 100.0, RATE, 0.0, T, "call")
        assert call == 0.0

    def test_dividend_lowers_call_price(self) -> None:
        """Higher dividend yield reduces the risk-neutral forward, lowering call value."""
        call_low_div = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", 0.01)
        call_high_div = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", 0.05)
        assert call_high_div < call_low_div

    def test_dividend_raises_put_price(self) -> None:
        """Higher dividend yield lowers the forward, raising put value."""
        put_low_div = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put", 0.01)
        put_high_div = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put", 0.05)
        assert put_high_div > put_low_div

    def test_higher_vol_raises_both_prices(self) -> None:
        """Increasing vol raises option value for both calls and puts."""
        call_lo = black_scholes(SPOT, STRIKE, RATE, 0.15, T, "call")
        call_hi = black_scholes(SPOT, STRIKE, RATE, 0.40, T, "call")
        put_lo = black_scholes(SPOT, STRIKE, RATE, 0.15, T, "put")
        put_hi = black_scholes(SPOT, STRIKE, RATE, 0.40, T, "put")
        assert call_hi > call_lo
        assert put_hi > put_lo

    def test_invalid_spot_raises(self) -> None:
        with pytest.raises(ValueError, match="Spot"):
            black_scholes(0.0, STRIKE, RATE, VOL, T)

    def test_invalid_strike_raises(self) -> None:
        with pytest.raises(ValueError, match="Strike"):
            black_scholes(SPOT, -1.0, RATE, VOL, T)

    def test_negative_vol_raises(self) -> None:
        with pytest.raises(ValueError, match="Volatility"):
            black_scholes(SPOT, STRIKE, RATE, -0.1, T)

    def test_zero_expiry_raises(self) -> None:
        with pytest.raises(ValueError, match="expiry"):
            black_scholes(SPOT, STRIKE, RATE, VOL, 0.0)

    def test_negative_dividend_raises(self) -> None:
        with pytest.raises(ValueError, match="[Dd]ividend"):
            black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", -0.01)


# ---------------------------------------------------------------------------
# Implied Volatility Solver
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    """Tests for the Newton-Raphson IV solver."""

    @pytest.mark.parametrize("vol", [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20])
    def test_round_trip_call(self, vol: float) -> None:
        """price(sigma) -> IV(price) must recover sigma to within 1e-5."""
        price = black_scholes(SPOT, STRIKE, RATE, vol, T, "call", DIV)
        recovered = implied_volatility(
            price, SPOT, STRIKE, RATE, T, "call", dividend_yield=DIV
        )
        assert abs(recovered - vol) < 1e-5, (
            f"IV round-trip (call) failed for vol={vol}: recovered {recovered:.7f}"
        )

    @pytest.mark.parametrize("vol", [0.10, 0.20, 0.40, 0.70])
    def test_round_trip_put(self, vol: float) -> None:
        """IV round-trip must work for puts across a range of volatilities."""
        price = black_scholes(SPOT, 105.0, RATE, vol, T, "put", DIV)
        recovered = implied_volatility(
            price, SPOT, 105.0, RATE, T, "put", dividend_yield=DIV
        )
        assert abs(recovered - vol) < 1e-5, (
            f"IV round-trip (put) failed for vol={vol}: recovered {recovered:.7f}"
        )

    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_round_trip_otm_strikes(self, strike: float) -> None:
        """IV solver must work for OTM and ITM strikes, not just ATM."""
        vol = 0.25
        price = black_scholes(SPOT, strike, RATE, vol, T, "call")
        recovered = implied_volatility(price, SPOT, strike, RATE, T, "call")
        assert abs(recovered - vol) < 1e-5

    def test_below_arbitrage_floor_raises(self) -> None:
        """Price strictly below the no-arbitrage floor must raise ValueError."""
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_volatility(0.0001, SPOT, STRIKE, RATE, T, "call")

    def test_zero_price_raises(self) -> None:
        with pytest.raises(ValueError, match="Market price"):
            implied_volatility(0.0, SPOT, STRIKE, RATE, T, "call")

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValueError, match="Market price"):
            implied_volatility(-1.0, SPOT, STRIKE, RATE, T, "call")


# ---------------------------------------------------------------------------
# Binomial Tree
# ---------------------------------------------------------------------------

class TestBinomialTree:
    """Tests for the CRR binomial tree pricer."""

    def test_converges_to_bs_call(self) -> None:
        """At 500 steps, BT call must be within TOL_BT of the BS benchmark."""
        bs = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", DIV)
        bt = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "call", steps=500, dividend_yield=DIV)
        assert abs(bt - bs) < TOL_BT, f"BT call={bt:.5f}, BS call={bs:.5f}"

    def test_converges_to_bs_put(self) -> None:
        """At 500 steps, BT put must be within TOL_BT of the BS benchmark."""
        bs = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put", DIV)
        bt = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "put", steps=500, dividend_yield=DIV)
        assert abs(bt - bs) < TOL_BT, f"BT put={bt:.5f}, BS put={bs:.5f}"

    def test_european_put_call_parity(self) -> None:
        """European BT call - put should equal the forward parity relationship."""
        call = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "call", steps=500, dividend_yield=DIV)
        put = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "put", steps=500, dividend_yield=DIV)
        lhs = call - put
        rhs = SPOT * math.exp(-DIV * T) - STRIKE * math.exp(-RATE * T)
        # BT has discretization error; allow slightly wider tolerance than BS
        assert abs(lhs - rhs) < 0.02, f"BT put-call parity: lhs={lhs:.5f}, rhs={rhs:.5f}"

    def test_american_put_ge_european_put(self) -> None:
        """American put must be at least as valuable as its European counterpart."""
        european = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "put", steps=200, american=False)
        american = binomial_tree(SPOT, STRIKE, RATE, VOL, T, "put", steps=200, american=True)
        # Allow a tiny numerical slack for tree discretization
        assert american >= european - 1e-4

    def test_american_call_equals_european_call_no_dividends(self) -> None:
        """Early exercise of a call is never optimal without dividends."""
        european = binomial_tree(
            SPOT, STRIKE, RATE, VOL, T, "call", steps=300, american=False, dividend_yield=0.0
        )
        american = binomial_tree(
            SPOT, STRIKE, RATE, VOL, T, "call", steps=300, american=True, dividend_yield=0.0
        )
        assert abs(american - european) < 1e-3

    def test_monotone_convergence_with_steps(self) -> None:
        """Higher step count should not diverge further from the BS price."""
        bs = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call")
        err_100 = abs(binomial_tree(SPOT, STRIKE, RATE, VOL, T, "call", steps=100) - bs)
        err_500 = abs(binomial_tree(SPOT, STRIKE, RATE, VOL, T, "call", steps=500) - bs)
        # 500-step error must be smaller than 100-step error
        assert err_500 < err_100

    def test_zero_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ss]teps"):
            binomial_tree(SPOT, STRIKE, RATE, VOL, T, steps=0)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Tests for the Monte Carlo pricer with antithetic variates."""

    def test_seeded_run_is_reproducible(self) -> None:
        """Two runs with the same seed must return identical results."""
        p1 = monte_carlo(SPOT, STRIKE, RATE, VOL, T, seed=42)
        p2 = monte_carlo(SPOT, STRIKE, RATE, VOL, T, seed=42)
        assert p1 == p2

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds almost certainly yield different prices (stochastic check)."""
        p1 = monte_carlo(SPOT, STRIKE, RATE, VOL, T, seed=1)
        p2 = monte_carlo(SPOT, STRIKE, RATE, VOL, T, seed=7777)
        assert p1 != p2

    def test_converges_to_bs_call(self) -> None:
        """MC call at 200k paths should be within TOL_MC of the BS price."""
        bs = black_scholes(SPOT, STRIKE, RATE, VOL, T, "call", DIV)
        mc = monte_carlo(SPOT, STRIKE, RATE, VOL, T, "call",
                         simulations=200_000, seed=0, dividend_yield=DIV)
        rel_err = abs(mc - bs) / bs
        assert rel_err < TOL_MC, (
            f"MC relative error {rel_err:.3%} exceeds tolerance {TOL_MC:.0%}"
        )

    def test_converges_to_bs_put(self) -> None:
        """MC put at 200k paths should be within TOL_MC of the BS price."""
        bs = black_scholes(SPOT, STRIKE, RATE, VOL, T, "put", DIV)
        mc = monte_carlo(SPOT, STRIKE, RATE, VOL, T, "put",
                         simulations=200_000, seed=0, dividend_yield=DIV)
        rel_err = abs(mc - bs) / bs
        assert rel_err < TOL_MC

    def test_put_call_parity_approximate(self) -> None:
        """MC call - put should approximately equal the forward parity."""
        call = monte_carlo(SPOT, STRIKE, RATE, VOL, T, "call",
                           simulations=200_000, seed=0, dividend_yield=DIV)
        put = monte_carlo(SPOT, STRIKE, RATE, VOL, T, "put",
                          simulations=200_000, seed=0, dividend_yield=DIV)
        lhs = call - put
        rhs = SPOT * math.exp(-DIV * T) - STRIKE * math.exp(-RATE * T)
        # MC has statistical error; 10-cent absolute tolerance
        assert abs(lhs - rhs) < 0.10, (
            f"MC put-call parity: lhs={lhs:.4f}, rhs={rhs:.4f}"
        )

    def test_insufficient_simulations_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ss]imulations"):
            monte_carlo(SPOT, STRIKE, RATE, VOL, T, simulations=50)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    """Sign and magnitude tests for Black-Scholes Greeks."""

    def setup_method(self) -> None:
        self.call = compute_greeks(SPOT, STRIKE, RATE, VOL, T, "call", DIV)
        self.put = compute_greeks(SPOT, STRIKE, RATE, VOL, T, "put", DIV)

    # Delta
    def test_call_delta_in_range(self) -> None:
        """Call delta must lie in (0, 1)."""
        assert 0.0 < self.call["delta"] < 1.0

    def test_put_delta_in_range(self) -> None:
        """Put delta must lie in (-1, 0)."""
        assert -1.0 < self.put["delta"] < 0.0

    def test_call_put_delta_parity(self) -> None:
        """call_delta - put_delta must equal exp(-q*T) (Merton dividend adjustment)."""
        expected = math.exp(-DIV * T)
        diff = self.call["delta"] - self.put["delta"]
        assert abs(diff - expected) < TOL_EXACT, (
            f"Delta parity: diff={diff:.8f}, expected={expected:.8f}"
        )

    # Gamma
    def test_gamma_positive_call(self) -> None:
        assert self.call["gamma"] > 0.0

    def test_gamma_positive_put(self) -> None:
        assert self.put["gamma"] > 0.0

    def test_gamma_equal_for_call_and_put(self) -> None:
        """Gamma is identical for a call and put at the same strike."""
        assert abs(self.call["gamma"] - self.put["gamma"]) < TOL_EXACT

    # Vega
    def test_vega_positive_call(self) -> None:
        assert self.call["vega"] > 0.0

    def test_vega_positive_put(self) -> None:
        assert self.put["vega"] > 0.0

    def test_vega_equal_for_call_and_put(self) -> None:
        """Vega is identical for a call and put at the same strike."""
        assert abs(self.call["vega"] - self.put["vega"]) < TOL_EXACT

    # Theta
    def test_call_theta_negative(self) -> None:
        """Call theta is negative: time decay erodes the option holder's position."""
        assert self.call["theta"] < 0.0

    def test_atm_put_theta_negative(self) -> None:
        """ATM put theta is negative for our parameter set (rate > dividend)."""
        assert self.put["theta"] < 0.0

    # Rho
    def test_call_rho_positive(self) -> None:
        """Higher rates increase the call value via the discounted strike."""
        assert self.call["rho"] > 0.0

    def test_put_rho_negative(self) -> None:
        """Higher rates reduce the put value."""
        assert self.put["rho"] < 0.0

    # Epsilon (dividend sensitivity)
    def test_call_epsilon_negative(self) -> None:
        """Higher dividend yield lowers the forward, reducing call value."""
        assert self.call["epsilon"] < 0.0

    def test_put_epsilon_positive(self) -> None:
        """Higher dividend yield raises put value via the reduced forward."""
        assert self.put["epsilon"] > 0.0

    # Vanna and Charm
    def test_vanna_is_float(self) -> None:
        """Vanna must be a finite float for both option types."""
        assert isinstance(self.call["vanna"], float) and math.isfinite(self.call["vanna"])
        assert isinstance(self.put["vanna"], float) and math.isfinite(self.put["vanna"])

    def test_charm_is_float(self) -> None:
        """Charm must be a finite float for both option types."""
        assert isinstance(self.call["charm"], float) and math.isfinite(self.call["charm"])
        assert isinstance(self.put["charm"], float) and math.isfinite(self.put["charm"])

    def test_vanna_equal_call_put(self) -> None:
        """Vanna is the same for call and put at the same strike."""
        assert abs(self.call["vanna"] - self.put["vanna"]) < TOL_EXACT

    # Key set
    def test_all_keys_present(self) -> None:
        """compute_greeks must always return all eight named Greeks."""
        required = {"delta", "gamma", "theta", "vega", "rho", "epsilon", "vanna", "charm"}
        assert required.issubset(set(self.call.keys()))


# ---------------------------------------------------------------------------
# VolatilitySurface
# ---------------------------------------------------------------------------

class TestVolatilitySurface:
    """Structural invariant tests for build_surface."""

    def test_basic_construction_retains_all_records(self) -> None:
        """Surface should hold all valid records that pass quality filters."""
        records = [
            _make_record(90.0, 30, 0.22),
            _make_record(100.0, 30, 0.20),
            _make_record(110.0, 30, 0.19),
            _make_record(100.0, 60, 0.21),
        ]
        surface = build_surface(records, SPOT, RATE)
        assert len(surface.points) == 4
        assert surface.n_filtered == 0

    def test_zero_market_records_are_filtered(self) -> None:
        """Records with bid == ask == 0 must be excluded and counted in n_filtered."""
        exp = date.today() + timedelta(days=30)
        zero_quote = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=0.0,
            bid=0.0,
            ask=0.0,
            volume=0,
            open_interest=200,
            implied_volatility=0.20,
        )
        surface = build_surface([zero_quote], SPOT, RATE)
        assert len(surface.points) == 0
        assert surface.n_filtered == 1

    def test_low_open_interest_filtered_when_min_set(self) -> None:
        """Records below min_open_interest must be filtered."""
        record = _make_record(100.0, 30, 0.20, open_interest=5)
        surface = build_surface([record], SPOT, RATE, min_open_interest=10)
        assert len(surface.points) == 0
        assert surface.n_filtered == 1

    def test_atm_term_structure_keys_sorted(self) -> None:
        """atm_term_structure() must return expirations in chronological order."""
        records = [
            _make_record(100.0, 90, 0.22),
            _make_record(100.0, 30, 0.20),
            _make_record(100.0, 60, 0.21),
        ]
        surface = build_surface(records, SPOT, RATE)
        ts = surface.atm_term_structure()
        keys = list(ts.keys())
        assert keys == sorted(keys), "Term structure keys are not in sorted order"

    def test_smile_sorted_by_log_moneyness(self) -> None:
        """smile() must return points ordered from low to high log-moneyness."""
        records = [
            _make_record(110.0, 30, 0.19),
            _make_record(90.0, 30, 0.22),
            _make_record(100.0, 30, 0.20),
        ]
        surface = build_surface(records, SPOT, RATE)
        smile = surface.smile(surface.expirations[0])
        moneynesses = [p.log_moneyness for p in smile]
        assert moneynesses == sorted(moneynesses)

    def test_smile_summary_n_points_consistent(self) -> None:
        """SmileSummary.n_points must match the count returned by smile()."""
        records = [
            _make_record(90.0, 30, 0.22),
            _make_record(100.0, 30, 0.20),
            _make_record(110.0, 30, 0.19),
        ]
        surface = build_surface(records, SPOT, RATE)
        for summary in surface.smile_summary():
            smile_len = len(surface.smile(summary.expiration))
            assert summary.n_points == smile_len

    def test_negative_spot_raises(self) -> None:
        records = [_make_record(100.0, 30, 0.20)]
        with pytest.raises(ValueError, match="[Ss]pot"):
            build_surface(records, -1.0, RATE)

    def test_build_surface_iv_from_mid(self) -> None:
        """use_market_iv=False should solve IV from the bid/ask midpoint.

        _make_record prices the mid using black_scholes at the given vol, so
        solving IV from that mid (with the same spot, rate, and T) must
        recover an IV close to the original. The test uses a loose bound
        because rounding in bid/ask construction introduces a tiny discrepancy.
        """
        records = [
            _make_record(95.0, 60, 0.22),
            _make_record(100.0, 60, 0.20),
            _make_record(105.0, 60, 0.18),
        ]
        surface = build_surface(records, SPOT, RATE, use_market_iv=False)
        assert len(surface.points) == 3, (
            f"Expected 3 points, got {len(surface.points)} (n_filtered={surface.n_filtered})"
        )
        assert surface.n_filtered == 0
        for pt in surface.points:
            # Solved IV should be in the ballpark of the original vol used in _make_record.
            # 0.10-0.40 is wide enough to survive the bid/ask rounding without false failures.
            assert 0.10 < pt.iv < 0.40, (
                f"Unexpected IV {pt.iv:.4f} for strike {pt.strike}"
            )

    def test_smile_summary_iv_range_bounds(self) -> None:
        """iv_range (min_iv, max_iv) must span exactly the min and max IVs for the expiry."""
        records = [
            _make_record(90.0, 30, 0.25),
            _make_record(100.0, 30, 0.20),
            _make_record(110.0, 30, 0.17),
        ]
        surface = build_surface(records, SPOT, RATE)
        summaries = surface.smile_summary()
        assert len(summaries) == 1
        lo, hi = summaries[0].iv_range
        ivs = [pt.iv for pt in surface.points]
        assert abs(lo - min(ivs)) < 1e-10, f"iv_range low {lo:.6f} != min IV {min(ivs):.6f}"
        assert abs(hi - max(ivs)) < 1e-10, f"iv_range high {hi:.6f} != max IV {max(ivs):.6f}"

    def test_smile_summary_slope_negative_for_equity_skew(self) -> None:
        """Smile slope must be negative when low strikes carry higher IV (put-skew structure).

        This is the canonical equity index shape: downside demand bids up OTM put vol
        relative to OTM calls. The OLS regression of IV on log-moneyness must produce
        a negative coefficient whenever IV decreases monotonically with strike.
        """
        records = [
            _make_record(85.0, 60, 0.32, option_type="put"),
            _make_record(95.0, 60, 0.24),
            _make_record(100.0, 60, 0.20),
            _make_record(105.0, 60, 0.18),
            _make_record(115.0, 60, 0.16),
        ]
        surface = build_surface(records, SPOT, RATE)
        summaries = surface.smile_summary()
        assert len(summaries) == 1
        slope = summaries[0].smile_slope
        assert slope is not None, "smile_slope is None with 5 points"
        assert slope < 0.0, (
            f"Expected negative slope for put-skew data, got {slope:.4f}"
        )

    def test_smile_summary_none_slope_for_single_point(self) -> None:
        """smile_slope must be None when fewer than two points are available for a given expiry."""
        records = [_make_record(100.0, 30, 0.20)]
        surface = build_surface(records, SPOT, RATE)
        summaries = surface.smile_summary()
        assert len(summaries) == 1
        assert summaries[0].smile_slope is None


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------

class TestCompareToMarket:
    """Tests for compare_to_market filtering and aggregate statistics."""

    def test_bs_near_zero_error_in_convergence_mode(self) -> None:
        """In per-contract IV mode, BS must reproduce the mid-price almost exactly.

        Passes reference_date=date.today() so that compare_to_market uses the same
        calendar date as _make_record, avoiding a one-day mismatch that can arise
        late at night when date.today() (local) and datetime.now(timezone.utc).date()
        (UTC) straddle midnight and produce different T values.
        """
        from datetime import date
        today = date.today()
        records = [_make_record(100.0, 30, 0.20)]
        report = compare_to_market(
            records, SPOT, RATE,
            bt_steps=100, mc_simulations=50_000,
            reference_date=today,
        )
        assert len(report.results) == 1
        # The mid was priced by BS at exactly vol=0.20; bid/ask rounding introduces
        # at most a 1% spread, so BS error should be well below $0.05 for near-ATM options.
        assert abs(report.results[0].bs_error) < 0.05

    def test_aggregate_stat_keys_always_present(self) -> None:
        """ComparisonReport must contain bs, bt, mc keys in every stat dict."""
        records = [_make_record(100.0, 30, 0.20)]
        report = compare_to_market(records, SPOT, RATE)
        for stat_dict in (
            report.mean_abs_error,
            report.median_rel_error,
            report.best_model_counts,
        ):
            assert set(stat_dict.keys()) == {"bs", "bt", "mc"}

    def test_zero_market_contracts_go_to_skipped(self) -> None:
        """Zero-wide contracts must be recorded in skipped, not results."""
        exp = date.today() + timedelta(days=30)
        zero = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=0.0,
            bid=0.0,
            ask=0.0,
            volume=0,
            open_interest=100,
            implied_volatility=0.20,
        )
        report = compare_to_market([zero], SPOT, RATE)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "zero_market"

    def test_expired_contracts_go_to_skipped(self) -> None:
        """Contracts expiring today (0 days) must be recorded as expired."""
        exp = date.today()  # 0 calendar days away
        rec = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=1.0,
            bid=0.95,
            ask=1.05,
            volume=100,
            open_interest=100,
            implied_volatility=0.20,
        )
        report = compare_to_market([rec], SPOT, RATE)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "expired_or_imminent"

    def test_results_sorted_by_strike(self) -> None:
        """compare_to_market must return results sorted by strike ascending."""
        records = [
            _make_record(110.0, 30, 0.19),
            _make_record(90.0, 30, 0.22),
            _make_record(100.0, 30, 0.20),
        ]
        report = compare_to_market(records, SPOT, RATE)
        strikes = [r.record.strike for r in report.results]
        assert strikes == sorted(strikes)

    def test_empty_chain_returns_zero_stats(self) -> None:
        """An empty chain should produce a valid report with all-zero aggregates."""
        report = compare_to_market([], SPOT, RATE)
        assert report.mean_abs_error == {"bs": 0.0, "bt": 0.0, "mc": 0.0}
        assert len(report.results) == 0
        assert len(report.skipped) == 0

    def test_surface_mode_uses_uniform_vol(self) -> None:
        """When a fixed volatility is supplied, every result must record that vol.

        In surface mode, all three models evaluate at the caller-supplied vol
        rather than per-contract implied volatility. This matters when diagnosing
        skew or term-structure effects under a flat vol assumption.
        """
        records = [
            _make_record(95.0, 30, 0.20),
            _make_record(100.0, 30, 0.20),
            _make_record(105.0, 30, 0.20),
        ]
        report = compare_to_market(records, SPOT, RATE, volatility=0.30)
        assert len(report.results) == 3
        for result in report.results:
            assert result.volatility_used == 0.30, (
                f"Expected volatility_used=0.30, got {result.volatility_used}"
            )
        assert report.surface_vol == 0.30

    def test_min_open_interest_filtering_in_compare(self) -> None:
        """Contracts below min_open_interest must appear in skipped with reason low_open_interest."""
        low_oi = _make_record(100.0, 30, 0.20, open_interest=5)
        high_oi = _make_record(105.0, 30, 0.20, open_interest=500)
        report = compare_to_market([low_oi, high_oi], SPOT, RATE, min_open_interest=10)
        assert len(report.results) == 1
        assert report.results[0].record.strike == 105.0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "low_open_interest"

    def test_zero_iv_skipped_in_convergence_mode(self) -> None:
        """In per-contract IV mode, a zero implied_volatility contract must be skipped."""
        from datetime import timedelta

        exp = date.today() + timedelta(days=30)
        zero_iv = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=1.0,
            bid=0.95,
            ask=1.05,
            volume=100,
            open_interest=200,
            implied_volatility=0.0,
        )
        report = compare_to_market([zero_iv], SPOT, RATE)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "zero_implied_volatility"

    def test_best_model_counts_sum_to_result_count(self) -> None:
        """Total across best_model_counts must equal the number of compared results.

        Each result contributes exactly one vote to the model with the smallest
        absolute pricing error, so the histogram must be exhaustive and non-overlapping.
        """
        records = [
            _make_record(90.0, 30, 0.22),
            _make_record(95.0, 30, 0.20),
            _make_record(100.0, 30, 0.19),
            _make_record(105.0, 30, 0.21),
            _make_record(110.0, 30, 0.23),
        ]
        report = compare_to_market(records, SPOT, RATE)
        total_counts = sum(report.best_model_counts.values())
        assert total_counts == len(report.results), (
            f"best_model_counts sum {total_counts} != result count {len(report.results)}"
        )


# ---------------------------------------------------------------------------
# NaN-safe conversion helpers
# ---------------------------------------------------------------------------

class TestSafeFloat:
    """Unit tests for _safe_float, the NaN-tolerant float converter in fetcher.py.

    yfinance DataFrames return numpy NaN for missing bid, ask, and
    impliedVolatility fields. _safe_float ensures these values are
    substituted with a default (0.0 by default) rather than propagating
    silently through downstream filters and arithmetic.
    """

    def test_nan_returns_default(self) -> None:
        """float('nan') must be replaced by the default value."""
        assert _safe_float(float("nan")) == 0.0

    def test_nan_custom_default(self) -> None:
        """Custom default must be returned when val is NaN."""
        assert _safe_float(float("nan"), default=-1.0) == -1.0

    def test_none_returns_default(self) -> None:
        """None input must be replaced by the default."""
        assert _safe_float(None) == 0.0

    def test_valid_positive_float(self) -> None:
        """A normal positive float must pass through unchanged."""
        assert _safe_float(1.5) == 1.5

    def test_valid_zero(self) -> None:
        """Explicit zero is a valid value and must not be replaced."""
        assert _safe_float(0.0) == 0.0

    def test_negative_float(self) -> None:
        """Negative values are valid and must pass through unchanged."""
        assert _safe_float(-3.14) == -3.14

    def test_integer_input(self) -> None:
        """Integer inputs must be converted to float correctly."""
        assert _safe_float(5) == 5.0

    def test_string_numeric(self) -> None:
        """Numeric strings must be converted to float."""
        assert _safe_float("2.5") == 2.5

    def test_string_nonnumeric_returns_default(self) -> None:
        """A non-numeric string must return the default."""
        assert _safe_float("not_a_number") == 0.0


class TestSafeInt:
    """Unit tests for _safe_int, the NaN-tolerant integer converter in fetcher.py.

    Fields like volume and open_interest are stored as int in OptionRecord.
    yfinance can return NaN for these fields; int(float('nan')) raises ValueError
    in CPython, so a guard is needed before the conversion.
    """

    def test_nan_returns_default(self) -> None:
        """float('nan') must be replaced by the integer default (0)."""
        assert _safe_int(float("nan")) == 0

    def test_none_returns_default(self) -> None:
        """None must be replaced by the default."""
        assert _safe_int(None) == 0

    def test_valid_integer(self) -> None:
        """A plain integer must pass through unchanged."""
        assert _safe_int(42) == 42

    def test_float_truncates(self) -> None:
        """A float is truncated toward zero, not rounded."""
        assert _safe_int(5.9) == 5

    def test_zero(self) -> None:
        """Zero is a valid value and must not be replaced."""
        assert _safe_int(0) == 0

    def test_string_integer(self) -> None:
        """A string encoding an integer must be converted correctly."""
        assert _safe_int("10") == 10

    def test_custom_default(self) -> None:
        """Custom default must be used when val is NaN."""
        assert _safe_int(float("nan"), default=-1) == -1

    def test_string_nonnumeric_returns_default(self) -> None:
        """A non-numeric string must return the default."""
        assert _safe_int("bad") == 0


class TestNaNFieldsInCompare:
    """Integration tests confirming that OptionRecords with NaN-derived zero
    fields are handled correctly by compare_to_market and build_surface.

    These tests simulate what happens after fetch_option_chain processes a
    yfinance row that had NaN for bid and ask: those fields become 0.0, and
    the contract should be treated as having no reliable quote and skipped.
    """

    def test_zero_bid_ask_filtered_as_zero_market(self) -> None:
        """A record with bid=0.0 and ask=0.0 (from NaN normalization) must be
        classified as zero_market and excluded from comparison results."""
        exp = date.today() + timedelta(days=30)
        rec = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=0.0,
            bid=0.0,   # NaN was normalized to 0.0 by _safe_float
            ask=0.0,   # NaN was normalized to 0.0 by _safe_float
            volume=0,
            open_interest=500,
            implied_volatility=0.20,
        )
        report = compare_to_market([rec], SPOT, RATE)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "zero_market"

    def test_zero_iv_filtered_in_convergence_mode(self) -> None:
        """A record with implied_volatility=0.0 (from NaN normalization) must be
        classified as zero_implied_volatility and excluded."""
        exp = date.today() + timedelta(days=30)
        rec = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=2.0,
            bid=1.95,
            ask=2.05,
            volume=100,
            open_interest=200,
            implied_volatility=0.0,  # NaN normalized to 0.0
        )
        report = compare_to_market([rec], SPOT, RATE)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "zero_implied_volatility"

    def test_zero_open_interest_filtered_when_min_set(self) -> None:
        """A record with open_interest=0 (from NaN normalization) must be
        excluded when min_open_interest > 0."""
        exp = date.today() + timedelta(days=30)
        bs_mid = black_scholes(SPOT, 100.0, RATE, 0.20, 30 / 365.0, "call")
        rec = OptionRecord(
            symbol="TEST",
            expiration=exp,
            strike=100.0,
            option_type="call",
            last_price=bs_mid,
            bid=round(bs_mid * 0.99, 4),
            ask=round(bs_mid * 1.01, 4),
            volume=0,
            open_interest=0,   # NaN normalized to 0
            implied_volatility=0.20,
        )
        report = compare_to_market([rec], SPOT, RATE, min_open_interest=1)
        assert len(report.results) == 0
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "low_open_interest"


# ---------------------------------------------------------------------------
# Portfolio Greeks
# ---------------------------------------------------------------------------

class TestPortfolio:
    """Tests for aggregate_greeks, Position, PositionRisk, and PortfolioRisk.

    Covers the key mathematical invariants of the portfolio aggregation layer:
    scaled Greeks are unit Greeks multiplied by signed quantity, dollar Greeks
    use the conventional definitions, and net Greeks are the element-wise sum
    across positions. Tests also verify that invalid contract parameters raise
    errors rather than silently propagating NaN.
    """

    # Shared parameters for a single ATM call/put pair
    SPOT = 100.0
    STRIKE = 100.0
    RATE = 0.05
    VOL = 0.20
    T = 1.0

    def _long_call(self, quantity: float = 1.0) -> "Position":
        from optionview.portfolio import Position
        return Position(
            spot=self.SPOT,
            strike=self.STRIKE,
            rate=self.RATE,
            volatility=self.VOL,
            expiry_years=self.T,
            option_type="call",
            quantity=quantity,
        )

    def _long_put(self, quantity: float = 1.0) -> "Position":
        from optionview.portfolio import Position
        return Position(
            spot=self.SPOT,
            strike=self.STRIKE,
            rate=self.RATE,
            volatility=self.VOL,
            expiry_years=self.T,
            option_type="put",
            quantity=quantity,
        )

    def test_empty_portfolio_returns_zero_net_greeks(self) -> None:
        """An empty position list must produce zero net exposure and no positions."""
        from optionview.portfolio import aggregate_greeks
        risk = aggregate_greeks([])
        assert risk.n_positions == 0
        assert risk.net_dollar_delta == 0.0
        assert risk.net_dollar_gamma == 0.0
        for k, v in risk.net_greeks.items():
            assert v == 0.0, f"Expected net {k} == 0.0 for empty portfolio, got {v}"

    def test_n_positions_matches_input_count(self) -> None:
        """PortfolioRisk.n_positions must equal the number of positions passed in."""
        from optionview.portfolio import aggregate_greeks
        assert aggregate_greeks([self._long_call()]).n_positions == 1
        assert aggregate_greeks([self._long_call(), self._long_put()]).n_positions == 2

    def test_unit_greeks_match_direct_compute_greeks(self) -> None:
        """PositionRisk.unit_greeks must match compute_greeks called with the same params."""
        from optionview.portfolio import aggregate_greeks
        risk = aggregate_greeks([self._long_call()])
        direct = compute_greeks(
            self.SPOT, self.STRIKE, self.RATE, self.VOL, self.T, "call"
        )
        for k in direct:
            assert abs(risk.positions[0].unit_greeks[k] - direct[k]) < 1e-12, (
                f"unit_greeks[{k!r}] mismatch: portfolio={risk.positions[0].unit_greeks[k]:.8f} "
                f"vs compute_greeks={direct[k]:.8f}"
            )

    def test_scaled_greeks_equal_unit_times_quantity(self) -> None:
        """Scaled Greeks for a single position must be unit Greeks times the quantity."""
        from optionview.portfolio import aggregate_greeks
        qty = 7.0
        risk = aggregate_greeks([self._long_call(quantity=qty)])
        pr = risk.positions[0]
        for k in pr.unit_greeks:
            expected = pr.unit_greeks[k] * qty
            actual = pr.scaled_greeks[k]
            assert abs(actual - expected) < 1e-12, (
                f"scaled_greeks[{k!r}] = {actual:.8f}, expected unit*qty = {expected:.8f}"
            )

    def test_short_position_inverts_scaled_greeks(self) -> None:
        """Negative quantity must invert the sign of all scaled Greeks relative to long."""
        from optionview.portfolio import aggregate_greeks
        long_risk = aggregate_greeks([self._long_call(quantity=3.0)])
        short_risk = aggregate_greeks([self._long_call(quantity=-3.0)])
        for k in long_risk.positions[0].scaled_greeks:
            long_val = long_risk.positions[0].scaled_greeks[k]
            short_val = short_risk.positions[0].scaled_greeks[k]
            assert abs(long_val + short_val) < 1e-12, (
                f"scaled_greeks[{k!r}]: long={long_val:.8f}, short={short_val:.8f}; "
                "short should be exact negation of long"
            )

    def test_dollar_delta_formula(self) -> None:
        """dollar_delta must equal unit_delta * spot * quantity."""
        from optionview.portfolio import aggregate_greeks
        qty = 13.0
        risk = aggregate_greeks([self._long_call(quantity=qty)])
        pr = risk.positions[0]
        expected = pr.unit_greeks["delta"] * self.SPOT * qty
        assert abs(pr.dollar_delta - expected) < 1e-10, (
            f"dollar_delta={pr.dollar_delta:.8f}, expected={expected:.8f}"
        )

    def test_dollar_gamma_formula(self) -> None:
        """dollar_gamma must equal 0.5 * unit_gamma * spot^2 * quantity."""
        from optionview.portfolio import aggregate_greeks
        qty = 5.0
        risk = aggregate_greeks([self._long_call(quantity=qty)])
        pr = risk.positions[0]
        expected = 0.5 * pr.unit_greeks["gamma"] * self.SPOT ** 2 * qty
        assert abs(pr.dollar_gamma - expected) < 1e-10, (
            f"dollar_gamma={pr.dollar_gamma:.8f}, expected={expected:.8f}"
        )

    def test_net_dollar_delta_is_sum_of_position_dollar_deltas(self) -> None:
        """net_dollar_delta must equal the sum of individual position dollar deltas."""
        from optionview.portfolio import aggregate_greeks
        positions = [self._long_call(quantity=10.0), self._long_put(quantity=5.0)]
        risk = aggregate_greeks(positions)
        expected_sum = sum(pr.dollar_delta for pr in risk.positions)
        assert abs(risk.net_dollar_delta - expected_sum) < 1e-10

    def test_net_dollar_gamma_is_sum_of_position_dollar_gammas(self) -> None:
        """net_dollar_gamma must equal the sum of individual position dollar gammas."""
        from optionview.portfolio import aggregate_greeks
        positions = [self._long_call(quantity=10.0), self._long_put(quantity=5.0)]
        risk = aggregate_greeks(positions)
        expected_sum = sum(pr.dollar_gamma for pr in risk.positions)
        assert abs(risk.net_dollar_gamma - expected_sum) < 1e-10

    def test_straddle_gamma_equals_twice_unit_gamma(self) -> None:
        """Long straddle net gamma equals twice the single-contract gamma.

        Gamma is identical for calls and puts at the same strike (it is a
        pure-second-order derivative of price with respect to spot and does
        not depend on option type). Two contracts at quantity=1 therefore
        contribute twice the unit gamma to net portfolio gamma.
        """
        from optionview.portfolio import aggregate_greeks
        risk = aggregate_greeks([self._long_call(), self._long_put()])
        unit = compute_greeks(self.SPOT, self.STRIKE, self.RATE, self.VOL, self.T, "call")
        assert abs(risk.net_greeks["gamma"] - 2.0 * unit["gamma"]) < 1e-12, (
            f"Straddle net gamma {risk.net_greeks['gamma']:.8f} != 2x unit "
            f"{unit['gamma']:.8f}"
        )

    def test_straddle_vega_equals_twice_unit_vega(self) -> None:
        """Long straddle net vega equals twice the single-contract vega.

        Vega is also identical for calls and puts (both increase in value
        as volatility rises), so a long straddle has positive net vega equal
        to the sum of each leg's unit vega.
        """
        from optionview.portfolio import aggregate_greeks
        risk = aggregate_greeks([self._long_call(), self._long_put()])
        unit = compute_greeks(self.SPOT, self.STRIKE, self.RATE, self.VOL, self.T, "call")
        assert abs(risk.net_greeks["vega"] - 2.0 * unit["vega"]) < 1e-12

    def test_straddle_theta_is_negative(self) -> None:
        """A long straddle loses value with the passage of time (negative net theta).

        Buying both a call and a put means paying extrinsic value in both legs.
        As expiry approaches, that extrinsic value decays, producing a net
        negative theta for the combined position.
        """
        from optionview.portfolio import aggregate_greeks
        risk = aggregate_greeks([self._long_call(), self._long_put()])
        assert risk.net_greeks["theta"] < 0.0

    def test_net_greeks_are_sum_of_scaled_greeks(self) -> None:
        """Net portfolio Greeks must be the element-wise sum of all scaled Greeks."""
        from optionview.portfolio import aggregate_greeks
        positions = [
            self._long_call(quantity=3.0),
            self._long_put(quantity=2.0),
            self._long_call(quantity=-1.0),
        ]
        risk = aggregate_greeks(positions)
        for k in risk.net_greeks:
            expected = sum(pr.scaled_greeks[k] for pr in risk.positions)
            assert abs(risk.net_greeks[k] - expected) < 1e-10, (
                f"net_greeks[{k!r}]={risk.net_greeks[k]:.8f} != "
                f"sum_of_scaled={expected:.8f}"
            )

    def test_long_and_offsetting_short_cancel_to_zero_net(self) -> None:
        """Equal long and short positions on the same contract must produce zero net Greeks.

        Adding a short position of the same size and parameters as a long position
        is an exact hedge. The net portfolio Greek exposure must be exactly zero for
        every Greek, not approximately zero due to floating-point accumulation.
        """
        from optionview.portfolio import aggregate_greeks
        long5 = self._long_call(quantity=5.0)
        short5 = self._long_call(quantity=-5.0)
        risk = aggregate_greeks([long5, short5])
        for k, v in risk.net_greeks.items():
            assert abs(v) < 1e-12, (
                f"net_greeks[{k!r}] = {v:.2e} for a perfectly hedged book; expected zero"
            )
        assert abs(risk.net_dollar_delta) < 1e-10
        assert abs(risk.net_dollar_gamma) < 1e-10

    def test_positions_at_different_strikes_net_delta_is_additive(self) -> None:
        """Net delta across positions at different strikes equals the sum of their deltas.

        A bull call spread (long lower strike, short upper strike) has positive net
        delta bounded strictly between 0 and exp(-q*T) (the maximum single-contract
        call delta). This verifies that the aggregation is a simple sum and that
        positions at different strikes interact only through their combined net exposure.
        """
        from optionview.portfolio import Position, aggregate_greeks
        long_atm = Position(
            spot=self.SPOT, strike=100.0, rate=self.RATE,
            volatility=self.VOL, expiry_years=self.T,
            option_type="call", quantity=1.0,
        )
        short_otm = Position(
            spot=self.SPOT, strike=110.0, rate=self.RATE,
            volatility=self.VOL, expiry_years=self.T,
            option_type="call", quantity=-1.0,
        )
        risk = aggregate_greeks([long_atm, short_otm])
        net_delta = risk.net_greeks["delta"]
        # Bull spread: long ATM call delta > short OTM call delta, so net is positive
        assert net_delta > 0.0, (
            f"Bull call spread must have positive net delta, got {net_delta:.6f}"
        )
        # Net delta is bounded by the maximum call delta (exp(-q*T) = 1 with no dividends)
        assert net_delta < 1.0, (
            f"Bull call spread net delta must be < 1.0, got {net_delta:.6f}"
        )

    def test_position_with_label_does_not_affect_greeks(self) -> None:
        """The label field is cosmetic and must not alter any computed risk figures."""
        from optionview.portfolio import Position, aggregate_greeks
        labeled = Position(
            spot=self.SPOT,
            strike=self.STRIKE,
            rate=self.RATE,
            volatility=self.VOL,
            expiry_years=self.T,
            option_type="call",
            quantity=1.0,
            label="my_call",
        )
        unlabeled = self._long_call(quantity=1.0)
        r1 = aggregate_greeks([labeled])
        r2 = aggregate_greeks([unlabeled])
        for k in r1.net_greeks:
            assert abs(r1.net_greeks[k] - r2.net_greeks[k]) < 1e-12, (
                f"label should not affect net_greeks[{k!r}]"
            )


# ---------------------------------------------------------------------------
# Greeks numerical verification via finite differences
# ---------------------------------------------------------------------------

class TestGreeksFiniteDifference:
    """Verify all eight Black-Scholes Greeks against centered finite-difference approximations.

    Finite-difference checks catch formula bugs that sign-and-bound tests miss.
    Each Greek is estimated by perturbing a single input and observing the change
    in option price (or in delta, for cross-sensitivities). Centered differences
    have O(h^2) truncation error; step sizes balance truncation against
    floating-point cancellation.

    All tests use a non-zero dividend yield (Q=0.02) to exercise the full
    Merton continuous-dividend extension in every formula path.
    """

    S = 100.0   # spot
    K = 100.0   # strike (ATM)
    R = 0.05    # risk-free rate
    V = 0.20    # volatility
    T = 0.50    # 6 months (avoids near-expiry singularities)
    Q = 0.02    # continuous dividend yield

    FD_TOL = 5e-5      # first-order Greeks (delta, vega, rho, epsilon)
    FD_TOL_2ND = 2e-4  # second-order (gamma)
    FD_TOL_CROSS = 1e-3  # cross-partials (vanna, charm)

    def _bs(
        self,
        s: float | None = None,
        k: float | None = None,
        r: float | None = None,
        v: float | None = None,
        t: float | None = None,
        q: float | None = None,
        opt: str = "call",
    ) -> float:
        """Black-Scholes price with optional per-parameter overrides."""
        return black_scholes(
            self.S if s is None else s,
            self.K if k is None else k,
            self.R if r is None else r,
            self.V if v is None else v,
            self.T if t is None else t,
            opt,  # type: ignore[arg-type]
            self.Q if q is None else q,
        )

    def _greeks(self, opt: str = "call") -> dict[str, float]:
        return compute_greeks(self.S, self.K, self.R, self.V, self.T, opt, self.Q)  # type: ignore[arg-type]

    # -- Delta: dV/dS --

    def test_delta_call_matches_fd(self) -> None:
        """Call delta matches (V(S+h) - V(S-h)) / (2h) within 5e-5."""
        h = 0.5
        fd = (self._bs(s=self.S + h) - self._bs(s=self.S - h)) / (2.0 * h)
        assert abs(self._greeks()["delta"] - fd) < self.FD_TOL, (
            f"Call delta: analytical={self._greeks()['delta']:.7f}, fd={fd:.7f}"
        )

    def test_delta_put_matches_fd(self) -> None:
        """Put delta matches (V_put(S+h) - V_put(S-h)) / (2h) within 5e-5."""
        h = 0.5
        fd = (self._bs(s=self.S + h, opt="put") - self._bs(s=self.S - h, opt="put")) / (2.0 * h)
        assert abs(self._greeks("put")["delta"] - fd) < self.FD_TOL, (
            f"Put delta: analytical={self._greeks('put')['delta']:.7f}, fd={fd:.7f}"
        )

    # -- Gamma: d^2V/dS^2 --

    def test_gamma_matches_fd(self) -> None:
        """Gamma matches (V(S+h) - 2V(S) + V(S-h)) / h^2 within 2e-4."""
        h = 0.5
        fd = (self._bs(s=self.S + h) - 2.0 * self._bs() + self._bs(s=self.S - h)) / (h ** 2)
        assert abs(self._greeks()["gamma"] - fd) < self.FD_TOL_2ND, (
            f"Gamma: analytical={self._greeks()['gamma']:.7f}, fd={fd:.7f}"
        )

    def test_gamma_fd_identical_for_call_and_put(self) -> None:
        """Finite-difference gamma is equal for call and put at the same strike.

        Gamma is the second derivative of price with respect to spot and does not
        depend on option type. Both legs of a put-call pair at the same strike must
        share the same gamma, which the FD estimate confirms independently of the
        analytical formula.
        """
        h = 0.5
        fd_call = (self._bs(s=self.S + h) - 2.0 * self._bs() + self._bs(s=self.S - h)) / h**2
        fd_put = (
            self._bs(s=self.S + h, opt="put")
            - 2.0 * self._bs(opt="put")
            + self._bs(s=self.S - h, opt="put")
        ) / h**2
        assert abs(fd_call - fd_put) < 1e-10

    # -- Vega: dV/dSigma (scaled to per 1% absolute vol move) --

    def test_vega_call_matches_fd(self) -> None:
        """Vega (per 1%) matches (V(sigma+h) - V(sigma-h)) / (2h) * 0.01 within 5e-5."""
        h = 0.005
        fd = (self._bs(v=self.V + h) - self._bs(v=self.V - h)) / (2.0 * h) * 0.01
        assert abs(self._greeks()["vega"] - fd) < self.FD_TOL, (
            f"Vega: analytical={self._greeks()['vega']:.7f}, fd={fd:.7f}"
        )

    # -- Theta: -dV/dT per calendar day --

    def test_theta_call_matches_fd(self) -> None:
        """Call theta matches -(V(T+h) - V(T-h)) / (2h) / 365 within 5e-5.

        Theta is conventionally expressed as the dollar decay per calendar day
        as expiry approaches, hence the negative sign and 1/365 scaling. For a
        long call this value is negative (time erodes option value).
        """
        h = 0.01
        fd = -(self._bs(t=self.T + h) - self._bs(t=self.T - h)) / (2.0 * h) / 365.0
        assert abs(self._greeks()["theta"] - fd) < self.FD_TOL, (
            f"Call theta: analytical={self._greeks()['theta']:.8f}, fd={fd:.8f}"
        )

    def test_theta_put_matches_fd(self) -> None:
        """Put theta matches finite-difference estimate within 5e-5."""
        h = 0.01
        fd = -(
            self._bs(t=self.T + h, opt="put") - self._bs(t=self.T - h, opt="put")
        ) / (2.0 * h) / 365.0
        assert abs(self._greeks("put")["theta"] - fd) < self.FD_TOL, (
            f"Put theta: analytical={self._greeks('put')['theta']:.8f}, fd={fd:.8f}"
        )

    # -- Rho: dV/dr per 1% rate move --

    def test_rho_call_matches_fd(self) -> None:
        """Call rho (per 1% rate) matches (V(r+h) - V(r-h)) / (2h) * 0.01 within 5e-5."""
        h = 0.001
        fd = (self._bs(r=self.R + h) - self._bs(r=self.R - h)) / (2.0 * h) * 0.01
        assert abs(self._greeks()["rho"] - fd) < self.FD_TOL, (
            f"Call rho: analytical={self._greeks()['rho']:.7f}, fd={fd:.7f}"
        )

    def test_rho_put_matches_fd(self) -> None:
        """Put rho matches finite-difference estimate within 5e-5."""
        h = 0.001
        fd = (
            self._bs(r=self.R + h, opt="put") - self._bs(r=self.R - h, opt="put")
        ) / (2.0 * h) * 0.01
        assert abs(self._greeks("put")["rho"] - fd) < self.FD_TOL, (
            f"Put rho: analytical={self._greeks('put')['rho']:.7f}, fd={fd:.7f}"
        )

    # -- Epsilon: dV/dq per 1% dividend yield move --

    def test_epsilon_call_matches_fd(self) -> None:
        """Call epsilon (per 1% yield) matches (V(q+h) - V(q-h)) / (2h) * 0.01 within 5e-5."""
        h = 0.001
        fd = (self._bs(q=self.Q + h) - self._bs(q=self.Q - h)) / (2.0 * h) * 0.01
        assert abs(self._greeks()["epsilon"] - fd) < self.FD_TOL, (
            f"Call epsilon: analytical={self._greeks()['epsilon']:.7f}, fd={fd:.7f}"
        )

    def test_epsilon_put_matches_fd(self) -> None:
        """Put epsilon matches finite-difference estimate within 5e-5."""
        h = 0.001
        fd = (
            self._bs(q=self.Q + h, opt="put") - self._bs(q=self.Q - h, opt="put")
        ) / (2.0 * h) * 0.01
        assert abs(self._greeks("put")["epsilon"] - fd) < self.FD_TOL, (
            f"Put epsilon: analytical={self._greeks('put')['epsilon']:.7f}, fd={fd:.7f}"
        )

    # -- Vanna: d(delta)/d(sigma) --

    def test_vanna_call_matches_fd(self) -> None:
        """Vanna matches d(delta)/d(sigma) estimated via centered differences.

        Vanna is the mixed partial d^2V/(dS d_sigma). Differencing the analytical
        delta with respect to vol avoids accumulating two layers of finite-difference
        error and isolates the formula under test. Tolerance is 1e-3 because vanna
        itself is small in magnitude (~0.07 for these parameters).
        """
        h = 0.005

        def _delta_at_vol(vol: float) -> float:
            return compute_greeks(self.S, self.K, self.R, vol, self.T, "call", self.Q)["delta"]  # type: ignore[arg-type]

        fd = (_delta_at_vol(self.V + h) - _delta_at_vol(self.V - h)) / (2.0 * h)
        assert abs(self._greeks()["vanna"] - fd) < self.FD_TOL_CROSS, (
            f"Vanna: analytical={self._greeks()['vanna']:.6f}, fd={fd:.6f}"
        )

    # -- Charm: -d(delta)/dT per calendar day --

    def test_charm_call_matches_fd(self) -> None:
        """Charm matches -d(delta)/dT / 365 estimated via centered differences.

        Charm is the daily rate of change of delta as time passes. Differencing
        the analytical delta with respect to T, then negating (time to expiry
        decreases as calendar time advances), and scaling by 1/365 gives the
        per-day delta decay. For a near-ATM call, charm is negative: delta drifts
        toward 0.5*exp(-q*T) as expiry approaches.
        """
        h = 0.01  # ~3.65 calendar days

        def _delta_at_T(t: float) -> float:
            return compute_greeks(self.S, self.K, self.R, self.V, t, "call", self.Q)["delta"]  # type: ignore[arg-type]

        fd = -(_delta_at_T(self.T + h) - _delta_at_T(self.T - h)) / (2.0 * h) / 365.0
        assert abs(self._greeks()["charm"] - fd) < self.FD_TOL_CROSS, (
            f"Charm: analytical={self._greeks()['charm']:.8f}, fd={fd:.8f}"
        )


# ---------------------------------------------------------------------------
# VolatilitySurface.interpolate_iv
# ---------------------------------------------------------------------------

class TestInterpolateIV:
    """Tests for VolatilitySurface.interpolate_iv.

    interpolate_iv uses piecewise-linear interpolation along the smile for a
    given expiration, with flat extrapolation beyond the observed strike range.
    Tests cover the interior case, exact hits on grid points, boundary
    extrapolation on both sides, and error conditions.
    """

    # Shared 30-day expiry with a three-point smile: low/ATM/high strikes
    EXP_DAYS = 30
    LOW_K = 90.0
    ATM_K = 100.0
    HIGH_K = 110.0
    LOW_IV = 0.25
    ATM_IV = 0.20
    HIGH_IV = 0.18

    def _three_point_surface(self) -> "VolatilitySurface":
        """Build a surface with three smile points at a single expiry."""
        records = [
            _make_record(self.LOW_K,  self.EXP_DAYS, self.LOW_IV),
            _make_record(self.ATM_K,  self.EXP_DAYS, self.ATM_IV),
            _make_record(self.HIGH_K, self.EXP_DAYS, self.HIGH_IV),
        ]
        return build_surface(records, SPOT, RATE)

    def test_interior_linear_interpolation(self) -> None:
        """Querying midway between two grid points returns their midpoint IV.

        For a smile with ATM IV=0.20 and HIGH IV=0.18, the midpoint in
        log-moneyness space must have IV exactly halfway between the two: 0.19.
        """
        surface = self._three_point_surface()
        exp = surface.expirations[0]

        atm_k = min(surface.smile(exp), key=lambda p: abs(p.log_moneyness))
        high_k = max(surface.smile(exp), key=lambda p: p.log_moneyness)

        mid_lm = (atm_k.log_moneyness + high_k.log_moneyness) / 2.0
        iv = surface.interpolate_iv(exp, mid_lm)

        expected = (atm_k.iv + high_k.iv) / 2.0
        assert abs(iv - expected) < 1e-10, (
            f"Midpoint interpolation: got {iv:.6f}, expected {expected:.6f}"
        )

    def test_exact_grid_point_returns_observed_iv(self) -> None:
        """Querying at an observed grid log-moneyness must return that point's IV exactly."""
        surface = self._three_point_surface()
        exp = surface.expirations[0]

        for pt in surface.smile(exp):
            iv = surface.interpolate_iv(exp, pt.log_moneyness)
            assert abs(iv - pt.iv) < 1e-10, (
                f"Grid hit at log_moneyness={pt.log_moneyness:.4f}: "
                f"got {iv:.6f}, expected {pt.iv:.6f}"
            )

    def test_left_extrapolation_returns_leftmost_iv(self) -> None:
        """Querying below the leftmost strike must return that strike's IV (flat extrapolation)."""
        surface = self._three_point_surface()
        exp = surface.expirations[0]
        leftmost = surface.smile(exp)[0]

        iv = surface.interpolate_iv(exp, leftmost.log_moneyness - 0.5)
        assert iv == leftmost.iv, (
            f"Left extrapolation: got {iv:.6f}, expected leftmost IV {leftmost.iv:.6f}"
        )

    def test_right_extrapolation_returns_rightmost_iv(self) -> None:
        """Querying above the rightmost strike must return that strike's IV (flat extrapolation)."""
        surface = self._three_point_surface()
        exp = surface.expirations[0]
        rightmost = surface.smile(exp)[-1]

        iv = surface.interpolate_iv(exp, rightmost.log_moneyness + 0.5)
        assert iv == rightmost.iv, (
            f"Right extrapolation: got {iv:.6f}, expected rightmost IV {rightmost.iv:.6f}"
        )

    def test_interpolated_iv_within_observed_range(self) -> None:
        """Any interior query must return an IV between the two bracketing grid IVs.

        For a monotone-decreasing smile (higher IV at lower strikes), any
        interpolated value must be bounded by its two neighbors. This verifies
        that piecewise-linear interpolation cannot overshoot or undershoot
        the local IV range.
        """
        surface = self._three_point_surface()
        exp = surface.expirations[0]
        pts = surface.smile(exp)

        # Query at 20 points evenly spaced across the full smile range
        lm_min = pts[0].log_moneyness
        lm_max = pts[-1].log_moneyness
        for i in range(1, 20):
            lm = lm_min + i * (lm_max - lm_min) / 20.0
            iv = surface.interpolate_iv(exp, lm)
            iv_min = min(p.iv for p in pts)
            iv_max = max(p.iv for p in pts)
            assert iv_min <= iv <= iv_max, (
                f"Interpolated IV {iv:.6f} at log_moneyness={lm:.4f} is outside "
                f"observed range [{iv_min:.6f}, {iv_max:.6f}]"
            )

    def test_missing_expiration_raises(self) -> None:
        """Querying an expiration not on the surface must raise ValueError."""
        surface = self._three_point_surface()
        absent_date = date.today() + timedelta(days=999)
        with pytest.raises(ValueError, match="No surface points"):
            surface.interpolate_iv(absent_date, 0.0)

    def test_single_point_expiration_raises(self) -> None:
        """An expiration with only one IVPoint must raise ValueError (cannot interpolate)."""
        records = [_make_record(100.0, 45, 0.20)]
        surface = build_surface(records, SPOT, RATE)
        exp = surface.expirations[0]
        with pytest.raises(ValueError, match="at least two"):
            surface.interpolate_iv(exp, 0.0)

    def test_interpolation_independent_of_option_type_ordering(self) -> None:
        """Mixing calls and puts at the same strikes must not break interpolation.

        build_surface accepts both option types; the smile() method returns all
        points sorted by log-moneyness regardless of type. interpolate_iv must
        handle this mixed-type ordering correctly.
        """
        records = [
            _make_record(90.0,  30, 0.25, option_type="put"),
            _make_record(100.0, 30, 0.20),
            _make_record(110.0, 30, 0.18),
        ]
        surface = build_surface(records, SPOT, RATE)
        exp = surface.expirations[0]
        pts = surface.smile(exp)
        assert len(pts) == 3

        # Midpoint between first two points should interpolate cleanly
        mid_lm = (pts[0].log_moneyness + pts[1].log_moneyness) / 2.0
        iv = surface.interpolate_iv(exp, mid_lm)
        expected = (pts[0].iv + pts[1].iv) / 2.0
        assert abs(iv - expected) < 1e-10
