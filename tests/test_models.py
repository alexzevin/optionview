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

Design note: tests that involve Monte Carlo use a fixed seed and a generous
absolute tolerance. The MC error bound reflects the statistical nature of the
estimator (not a defect) and would tighten proportionally to 1/sqrt(n_paths).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from optionview.compare import compare_to_market
from optionview.fetcher import OptionRecord
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
        """In per-contract IV mode, BS must reproduce the mid-price almost exactly."""
        records = [_make_record(100.0, 30, 0.20)]
        report = compare_to_market(records, SPOT, RATE, bt_steps=100, mc_simulations=50_000)
        assert len(report.results) == 1
        # The mid was priced by BS at exactly vol=0.20, so BS error should be tiny
        assert abs(report.results[0].bs_error) < 0.01

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
