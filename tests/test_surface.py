"""Tests for forward_vol_curve() and ForwardVolPoint on VolatilitySurface.

These tests exercise the implied forward volatility computation in isolation
from network I/O by building VolatilitySurface objects directly from
synthetic IVPoint lists. Each test targets a specific property of the
forward variance formula:

    var_fwd(T1, T2) = (sigma2^2 * T2 - sigma1^2 * T1) / (T2 - T1)

The key correctness properties are:
  - Flat term structure: forward vol equals spot vol everywhere.
  - Monotone increasing term structure: forward vol > far ATM vol > near ATM vol.
  - Arbitrage detection: inverted structure with sufficiently high near vol
    produces negative forward variance, flagged by is_arbitrage_free=False.
  - Exact formula recovery: round-trip from known vols to forward vol and back.
  - Single expiry: returns empty list (no adjacent pairs).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import pytest

from optionview.surface import (
    ForwardVolPoint,
    IVPoint,
    VolatilitySurface,
    build_surface,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_surface(
    exp_vol_years: list[tuple[date, float, float]],
    spot: float = 100.0,
    rate: float = 0.05,
) -> VolatilitySurface:
    """Build a VolatilitySurface from (expiry, atm_iv, expiry_years) triples.

    Constructs a single IVPoint at log-moneyness=0 per expiry so that
    atm_term_structure() returns the supplied vol for each date.
    """
    points: list[IVPoint] = []
    today = date(2025, 1, 1)
    for exp, iv, t_years in exp_vol_years:
        points.append(
            IVPoint(
                expiration=exp,
                expiry_years=t_years,
                strike=spot,  # ATM: K == S -> log(K/F) ~ 0 for small r*T
                log_moneyness=0.0,
                option_type="call",
                iv=iv,
                open_interest=500,
                bid=5.0,
                ask=5.1,
            )
        )
    return VolatilitySurface(
        points=points,
        spot=spot,
        rate=rate,
        dividend_yield=0.0,
        built_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        n_filtered=0,
    )


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

class TestForwardVolCurveStructure:
    def test_empty_list_for_single_expiry(self) -> None:
        """A surface with one expiry has no adjacent pairs."""
        surface = _make_surface([(date(2025, 3, 21), 0.20, 0.25)])
        result = surface.forward_vol_curve()
        assert result == []

    def test_length_is_n_expirations_minus_one(self) -> None:
        """forward_vol_curve returns one entry per adjacent pair."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
            (date(2025, 9, 19), 0.21, 0.75),
            (date(2025, 12, 19), 0.23, 1.00),
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()
        assert len(result) == 3

    def test_returns_forward_vol_point_instances(self) -> None:
        """Each element is a ForwardVolPoint."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()
        assert len(result) == 1
        assert isinstance(result[0], ForwardVolPoint)

    def test_near_and_far_expiry_fields_ordered(self) -> None:
        """near_expiry < far_expiry for each ForwardVolPoint."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
            (date(2025, 9, 19), 0.21, 0.75),
        ]
        surface = _make_surface(triples)
        for pt in surface.forward_vol_curve():
            assert pt.near_expiry < pt.far_expiry

    def test_atm_vols_match_surface_term_structure(self) -> None:
        """near_atm_vol and far_atm_vol match the ATM term structure."""
        triples = [
            (date(2025, 3, 21), 0.18, 0.25),
            (date(2025, 6, 20), 0.24, 0.50),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        assert abs(pt.near_atm_vol - 0.18) < 1e-10
        assert abs(pt.far_atm_vol - 0.24) < 1e-10

    def test_expiry_years_fields_match_surface_points(self) -> None:
        """near_years and far_years are taken from the surface IVPoints."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.2466),
            (date(2025, 6, 20), 0.22, 0.4959),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        assert abs(pt.near_years - 0.2466) < 1e-10
        assert abs(pt.far_years - 0.4959) < 1e-10

    def test_curve_ordered_chronologically(self) -> None:
        """Results are sorted by near_expiry (earliest pair first)."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
            (date(2025, 9, 19), 0.21, 0.75),
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()
        assert result[0].near_expiry == date(2025, 3, 21)
        assert result[1].near_expiry == date(2025, 6, 20)


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------

class TestForwardVolFormula:
    def test_flat_term_structure_forward_vol_equals_spot_vol(self) -> None:
        """Flat term structure implies forward vol == spot vol everywhere.

        If sigma_near == sigma_far == sigma, then:
            var_fwd = (sigma^2 * T_far - sigma^2 * T_near) / (T_far - T_near)
                    = sigma^2
        so forward_vol = sigma.
        """
        sigma = 0.25
        triples = [
            (date(2025, 3, 21), sigma, 0.25),
            (date(2025, 6, 20), sigma, 0.50),
            (date(2025, 9, 19), sigma, 0.75),
        ]
        surface = _make_surface(triples)
        for pt in surface.forward_vol_curve():
            assert pt.forward_vol is not None
            assert abs(pt.forward_vol - sigma) < 1e-12, (
                f"Expected {sigma}, got {pt.forward_vol}"
            )

    def test_forward_variance_formula_manual(self) -> None:
        """Verify var_fwd against the closed-form expression by hand."""
        sigma_near, sigma_far = 0.20, 0.30
        t_near, t_far = 0.25, 0.50
        expected_var = (sigma_far ** 2 * t_far - sigma_near ** 2 * t_near) / (t_far - t_near)

        triples = [
            (date(2025, 3, 21), sigma_near, t_near),
            (date(2025, 6, 20), sigma_far, t_far),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]

        assert abs(pt.forward_variance - expected_var) < 1e-12
        assert pt.forward_vol is not None
        assert abs(pt.forward_vol - math.sqrt(expected_var)) < 1e-12

    def test_forward_vol_recovers_far_vol_from_near_and_forward(self) -> None:
        """Compounding near variance and forward variance reproduces far total variance.

        The forward variance identity:
            sigma_near^2 * T_near + var_fwd * (T_far - T_near) == sigma_far^2 * T_far
        """
        sigma_near, sigma_far = 0.18, 0.22
        t_near, t_far = 0.25, 1.00
        triples = [
            (date(2025, 3, 21), sigma_near, t_near),
            (date(2025, 3, 20) .replace(year=2026), sigma_far, t_far),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]

        recomposed = (
            sigma_near ** 2 * t_near
            + pt.forward_variance * (t_far - t_near)
        )
        assert abs(recomposed - sigma_far ** 2 * t_far) < 1e-12

    def test_increasing_term_structure_forward_vol_above_far_atm(self) -> None:
        """Steeply rising term structure yields forward vol above both ATM vols.

        When the far vol rises faster than the total variance accumulates,
        the marginal forward variance exceeds the far total variance.
        """
        sigma_near, sigma_far = 0.15, 0.30
        t_near, t_far = 0.25, 0.50
        var_fwd = (sigma_far ** 2 * t_far - sigma_near ** 2 * t_near) / (t_far - t_near)

        triples = [
            (date(2025, 3, 21), sigma_near, t_near),
            (date(2025, 6, 20), sigma_far, t_far),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]

        assert pt.is_arbitrage_free
        assert pt.forward_vol is not None
        assert pt.forward_vol > sigma_far

    def test_multiple_pairs_independent_computation(self) -> None:
        """Each adjacent pair uses only its own near/far vols and years."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.25, 0.50),
            (date(2025, 9, 19), 0.22, 0.75),
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()

        # Compute expected forward vols manually for each pair
        pairs = [
            (0.20, 0.25, 0.25, 0.50),
            (0.25, 0.22, 0.50, 0.75),
        ]
        for i, (sn, sf, tn, tf) in enumerate(pairs):
            expected_var = (sf ** 2 * tf - sn ** 2 * tn) / (tf - tn)
            assert abs(result[i].forward_variance - expected_var) < 1e-12


# ---------------------------------------------------------------------------
# Arbitrage detection
# ---------------------------------------------------------------------------

class TestForwardVolArbitrageDetection:
    def test_arbitrage_free_flag_true_for_normal_structure(self) -> None:
        """Normal upward-sloping term structure is arbitrage-free."""
        triples = [
            (date(2025, 3, 21), 0.18, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        assert pt.is_arbitrage_free is True
        assert pt.forward_vol is not None

    def test_arbitrage_detected_when_forward_variance_negative(self) -> None:
        """Strongly inverted term structure flags is_arbitrage_free=False.

        If sigma_near is large and sigma_far is small, var_fwd can be
        negative. Example: near=40%, far=20%, T_near=0.25, T_far=0.50.

            var_fwd = (0.20^2 * 0.50 - 0.40^2 * 0.25) / 0.25
                    = (0.020 - 0.040) / 0.25
                    = -0.08  (negative: arbitrage)
        """
        triples = [
            (date(2025, 3, 21), 0.40, 0.25),
            (date(2025, 6, 20), 0.20, 0.50),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        assert pt.is_arbitrage_free is False
        assert pt.forward_vol is None
        assert pt.forward_variance < 0.0

    def test_forward_variance_value_correct_for_negative_case(self) -> None:
        """forward_variance field holds the exact negative value."""
        sigma_near, sigma_far = 0.40, 0.20
        t_near, t_far = 0.25, 0.50
        expected = (sigma_far ** 2 * t_far - sigma_near ** 2 * t_near) / (t_far - t_near)

        triples = [
            (date(2025, 3, 21), sigma_near, t_near),
            (date(2025, 6, 20), sigma_far, t_far),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        assert abs(pt.forward_variance - expected) < 1e-12

    def test_mild_inversion_does_not_trigger_arbitrage_flag(self) -> None:
        """Mild inversion where far vol is still above breakeven is arb-free.

        With T_far = 2*T_near, breakeven holds when sigma_far >= sigma_near/sqrt(2).
        Above that threshold, var_fwd > 0.
        """
        sigma_near = 0.30
        # far vol just above the breakeven: sigma_near / sqrt(2) ~ 0.212
        sigma_far = 0.22
        t_near, t_far = 0.25, 0.50
        var_fwd = (sigma_far ** 2 * t_far - sigma_near ** 2 * t_near) / (t_far - t_near)

        triples = [
            (date(2025, 3, 21), sigma_near, t_near),
            (date(2025, 6, 20), sigma_far, t_far),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]

        if var_fwd > 0:
            assert pt.is_arbitrage_free is True
            assert pt.forward_vol is not None
        else:
            assert pt.is_arbitrage_free is False

    def test_mixed_surface_some_pairs_arbitrage_free_some_not(self) -> None:
        """Each pair is evaluated independently; flags can differ across pairs."""
        triples = [
            (date(2025, 3, 21), 0.40, 0.25),   # near: very high vol
            (date(2025, 6, 20), 0.20, 0.50),   # far: low vol  => pair 0 violates
            (date(2025, 9, 19), 0.30, 0.75),   # farther: higher vol => pair 1 ok
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()

        # Pair (March, June): sigma_near=0.40 >> sigma_far=0.20 -> arbitrage
        assert result[0].is_arbitrage_free is False
        assert result[0].forward_vol is None

        # Pair (June, Sep): var_fwd = (0.30^2*0.75 - 0.20^2*0.50)/0.25 = (0.0675-0.02)/0.25 = 0.19 > 0
        assert result[1].is_arbitrage_free is True
        assert result[1].forward_vol is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestForwardVolEdgeCases:
    def test_two_expiry_surface_returns_one_point(self) -> None:
        """Minimum non-empty case: two expirations yield one ForwardVolPoint."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
        ]
        surface = _make_surface(triples)
        result = surface.forward_vol_curve()
        assert len(result) == 1

    def test_forward_vol_is_positive_for_all_arb_free_points(self) -> None:
        """forward_vol is always strictly positive when is_arbitrage_free=True."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
            (date(2025, 9, 19), 0.23, 0.75),
            (date(2025, 12, 19), 0.24, 1.00),
        ]
        surface = _make_surface(triples)
        for pt in surface.forward_vol_curve():
            if pt.is_arbitrage_free:
                assert pt.forward_vol is not None
                assert pt.forward_vol > 0.0

    def test_forward_vol_none_iff_not_arbitrage_free(self) -> None:
        """forward_vol and is_arbitrage_free are always consistent."""
        triples = [
            (date(2025, 3, 21), 0.40, 0.25),   # triggers arbitrage
            (date(2025, 6, 20), 0.20, 0.50),
            (date(2025, 9, 19), 0.30, 0.75),
        ]
        surface = _make_surface(triples)
        for pt in surface.forward_vol_curve():
            if pt.is_arbitrage_free:
                assert pt.forward_vol is not None
            else:
                assert pt.forward_vol is None

    def test_forward_vol_point_is_immutable(self) -> None:
        """ForwardVolPoint is a frozen dataclass."""
        triples = [
            (date(2025, 3, 21), 0.20, 0.25),
            (date(2025, 6, 20), 0.22, 0.50),
        ]
        surface = _make_surface(triples)
        pt = surface.forward_vol_curve()[0]
        with pytest.raises((AttributeError, TypeError)):
            pt.forward_vol = 99.0  # type: ignore[misc]

    def test_empty_surface_returns_empty_list(self) -> None:
        """Surface with no points returns empty curve without raising."""
        surface = VolatilitySurface(
            points=[],
            spot=100.0,
            rate=0.05,
            dividend_yield=0.0,
            built_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            n_filtered=0,
        )
        result = surface.forward_vol_curve()
        assert result == []
