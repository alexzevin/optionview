# optionview

A Python toolkit for comparing option pricing models against real-time market data. Built for traders and quant enthusiasts who want to evaluate how theoretical models stack up against live prices, all using freely available data sources.

## Features

- **Multiple Pricing Models**: Black-Scholes, Binomial Tree, Monte Carlo simulation, Heston stochastic volatility, and SABR (Hagan et al. 2002)
- **Live Market Data**: Fetches real-time option chains from Yahoo Finance (no API key required)
- **Model Comparison**: Side-by-side comparison of theoretical vs. market prices
- **Volatility Surface Construction**: Builds an implied volatility surface from multi-expiry option chains, with per-expiry smile analysis (OLS slope and IV range), ATM term structure extraction, log-moneyness normalized IVPoints, piecewise-linear IV interpolation across strikes, and implied forward volatility extraction with calendar spread arbitrage detection
- **Greeks Calculation**: Full analytical Greeks including Delta, Gamma, Theta, Vega, Rho, Epsilon (dividend sensitivity), Vanna, Charm, and Volga (vomma)
- **Implied Volatility Solver**: Newton-Raphson IV solver with configurable tolerance
- **Portfolio Greeks Aggregation**: Aggregate Greeks across a collection of long and short option positions, with per-position and net portfolio risk
- **Scenario P&L**: Estimate portfolio P&L for a simultaneous move in spot, implied vol, and time using a second-order Taylor expansion (delta, gamma, vega, theta, vanna, charm, volga), useful for stress testing and what-if analysis without a full reprice
- **Clean API**: Simple, composable functions for scripting and analysis

## Installation

```bash
pip install -e .
```

### Dependencies

- numpy
- scipy
- yfinance

## Quick Start

```python
from optionview.models import black_scholes, binomial_tree, monte_carlo, heston
from optionview.greeks import compute_greeks
from optionview.fetcher import fetch_option_chain

# Price a call option using Black-Scholes
price = black_scholes(
    spot=150.0,
    strike=155.0,
    rate=0.05,
    volatility=0.25,
    expiry_years=0.25,
    option_type="call"
)
print(f"Black-Scholes price: ${price:.2f}")

# Compare all three constant-vol models
spot, strike, rate, vol, T = 150.0, 155.0, 0.05, 0.25, 0.25

bs_price = black_scholes(spot, strike, rate, vol, T, "call")
bt_price = binomial_tree(spot, strike, rate, vol, T, "call", steps=200)
mc_price = monte_carlo(spot, strike, rate, vol, T, "call", simulations=100_000)

print(f"Black-Scholes:  ${bs_price:.4f}")
print(f"Binomial Tree:  ${bt_price:.4f}")
print(f"Monte Carlo:    ${mc_price:.4f}")

# Fetch live option chain data
chain = fetch_option_chain("AAPL")
print(f"Retrieved {len(chain)} contracts")
for record in chain[:3]:
    print(f"  {record.option_type} K={record.strike:.0f} exp={record.expiration} IV={record.implied_volatility:.2%}")

# Compute Greeks
greeks = compute_greeks(spot, strike, rate, vol, T, "call")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega:  {greeks['vega']:.4f}")
```

## Model Comparison

Compare all three pricing models against live market mid-prices and identify
which model best fits each contract:

```python
from optionview.compare import compare_to_market
from optionview.fetcher import fetch_option_chain, fetch_spot_price, fetch_dividend_yield

ticker = "AAPL"
chain = fetch_option_chain(ticker)
spot = fetch_spot_price(ticker)
div_yield = fetch_dividend_yield(ticker)
rate = 0.05

report = compare_to_market(
    chain,
    spot=spot,
    rate=rate,
    dividend_yield=div_yield,
    min_open_interest=50,  # exclude illiquid strikes
)

print(f"Compared {len(report.results)} contracts, skipped {len(report.skipped)}")
print(f"Mean absolute error (Black-Scholes): ${report.mean_abs_error['bs']:.4f}")
print(f"Mean absolute error (Binomial Tree): ${report.mean_abs_error['bt']:.4f}")
print(f"Mean absolute error (Monte Carlo):   ${report.mean_abs_error['mc']:.4f}")
print(f"Best model wins: {report.best_model_counts}")

# Inspect individual results
for result in report.results[:5]:
    rec = result.record
    print(
        f"  {rec.option_type} K={rec.strike:.0f}: "
        f"market=${result.market_mid:.2f} "
        f"BS=${result.bs_price:.2f} (err={result.bs_rel_error:+.2%}) "
        f"best={result.best_model}"
    )
```

By default, each contract is evaluated at its own market-implied volatility.
In this convergence mode, Black-Scholes reproduces the market price exactly
(by definition of IV), so the comparison quantifies how well Binomial Tree
and Monte Carlo converge to the analytical benchmark. Pass a uniform
`volatility` override to evaluate all contracts at the same vol assumption,
which instead reveals pricing errors driven by the volatility smile.

## Volatility Surface

Build an implied volatility surface from an option chain and analyze the smile structure:

```python
from optionview.fetcher import fetch_option_chain, fetch_spot_price, list_expirations
from optionview.surface import build_surface

# Fetch contracts across multiple expirations for a richer surface
ticker = "SPY"
expirations = list_expirations(ticker)[:4]  # nearest four expiries

all_records = []
for exp in expirations:
    all_records.extend(fetch_option_chain(ticker, expiration=exp))

spot = fetch_spot_price(ticker)
rate = 0.05
dividend_yield = 0.013  # ~1.3% SPY yield

surface = build_surface(all_records, spot, rate, dividend_yield, min_open_interest=50)

print(f"Surface built: {len(surface.points)} points across {len(surface.expirations)} expiries")
print(f"Filtered out: {surface.n_filtered} low-quality records")

# ATM vol term structure
atm = surface.atm_term_structure()
for exp, vol in atm.items():
    print(f"  {exp}: ATM IV = {vol:.1%}")

# Smile analysis per expiry
for summary in surface.smile_summary():
    slope_str = f"{summary.smile_slope:+.3f}" if summary.smile_slope is not None else "n/a"
    print(
        f"  {summary.expiration}: ATM={summary.atm_iv:.1%} "
        f"slope={slope_str} pts={summary.n_points}"
    )

# Inspect individual smile for a specific expiry
first_exp = surface.expirations[0]
for pt in surface.smile(first_exp):
    print(f"  k={pt.log_moneyness:+.3f}  IV={pt.iv:.1%}  {pt.option_type}  K={pt.strike:.0f}")
```

The smile slope is the OLS slope of IV regressed on log-moneyness. A negative slope
(IV decreasing with strike) is characteristic of equity index options and reflects
downside demand. A strongly negative slope across all expirations signals pronounced
skew, which has implications for delta-hedging books with large gamma exposure.

## Volatility Interpolation

Query the IV at any arbitrary strike for a given expiry using piecewise-linear
interpolation across smile points. Flat extrapolation is used beyond the
observed strike range, which is the conservative choice: it avoids introducing
artificial slope at the tails and makes no distributional assumptions beyond
what the observed market quotes support.

```python
from optionview.fetcher import fetch_option_chain, fetch_spot_price
from optionview.surface import build_surface

ticker = "SPY"
records = fetch_option_chain(ticker)
spot = fetch_spot_price(ticker)
rate = 0.05

surface = build_surface(records, spot, rate, min_open_interest=50)
exp = surface.expirations[0]

# Interpolate IV at a specific log-moneyness (log(K / F))
import math
forward = spot  # approximate for near-dated options
target_strike = spot * 0.97  # 3% OTM put
log_moneyness = math.log(target_strike / forward)

iv = surface.interpolate_iv(exp, log_moneyness)
print(f"Interpolated IV at K={target_strike:.1f}: {iv:.1%}")

# Grid points return their observed IV exactly
for pt in surface.smile(exp)[:3]:
    iv_at_grid = surface.interpolate_iv(exp, pt.log_moneyness)
    assert abs(iv_at_grid - pt.iv) < 1e-10   # exact at grid points
```

The interpolation uses binary search to locate the bracketing interval, runs
in O(log n) per query, and cannot produce values outside the observed IV range.
Mixing calls and puts at the same strikes is handled correctly since
`smile()` returns all points sorted by log-moneyness regardless of option type.

`interpolate_iv` raises `ValueError` if the expiration is not on the surface
or if fewer than two points are present for that expiry (use
`atm_term_structure()` for single-point expirations).

## Forward Volatility Curve

`forward_vol_curve()` extracts the implied forward volatility between each pair of adjacent
expirations on the surface. The forward variance for the interval [T1, T2] is derived from
the ATM implied vols at the two expiries:

    var_fwd(T1, T2) = (sigma_far^2 * T_far - sigma_near^2 * T_near) / (T_far - T_near)

This is the unique constant variance that, when compounded with the near-expiry variance,
reproduces the far-expiry total variance. It is the volatility analog of the instantaneous
forward rate in interest rate term structures. A rising ATM term structure (far vol above
near vol) implies a forward vol above the far ATM level, because the far slice must
"account for" both the near period and the steeper forward period together.

A negative forward variance indicates a calendar spread arbitrage: buying the near-expiry
straddle and selling the far-expiry straddle at the same strike would produce a risk-free
profit regardless of the realized path. In practice this usually reflects stale quotes on a
near expiry rather than a genuine market dislocation. Building the surface with a
`min_open_interest` filter reduces the frequency of these cases.

```python
from optionview.fetcher import fetch_option_chain, fetch_spot_price, list_expirations
from optionview.surface import build_surface

ticker = "SPY"
expirations = list_expirations(ticker)[:5]  # five nearest expiries

all_records = []
for exp in expirations:
    all_records.extend(fetch_option_chain(ticker, expiration=exp))

spot = fetch_spot_price(ticker)
rate = 0.05
div_yield = 0.013

surface = build_surface(all_records, spot, rate, div_yield, min_open_interest=50)

for pt in surface.forward_vol_curve():
    arb_flag = "" if pt.is_arbitrage_free else "  [calendar arb]"
    fwd_str = f"{pt.forward_vol:.1%}" if pt.forward_vol is not None else "n/a"
    print(
        f"  {pt.near_expiry} -> {pt.far_expiry}: "
        f"near={pt.near_atm_vol:.1%}  far={pt.far_atm_vol:.1%}  "
        f"fwd={fwd_str}{arb_flag}"
    )
```

`forward_vol_curve()` returns one `ForwardVolPoint` per adjacent expiry pair. Each object
carries the raw `forward_variance` alongside the derived `forward_vol` (None when the
variance is non-positive) and an explicit `is_arbitrage_free` flag. Because the flag and
`forward_vol` are always consistent, callers can filter arbitrage-violating intervals with
a single attribute check rather than recomputing variance.

```python
# Count intervals that violate calendar no-arbitrage
violations = [pt for pt in surface.forward_vol_curve() if not pt.is_arbitrage_free]
print(f"{len(violations)} calendar arb violation(s) detected")
for pt in violations:
    print(
        f"  {pt.near_expiry} -> {pt.far_expiry}: "
        f"forward_variance={pt.forward_variance:.6f} (near={pt.near_atm_vol:.1%}, "
        f"far={pt.far_atm_vol:.1%})"
    )
```

The method returns an empty list when fewer than two expirations are present on the surface.

## Heston Stochastic Volatility

Unlike Black-Scholes, the Heston model treats variance as a mean-reverting
stochastic process. This lets it produce the implied vol smile and skew that
constant-vol models cannot reproduce, making it a better fit for multi-strike
comparisons and stress scenarios.

The model dynamics:

```
dS = (r - q) S dt + sqrt(v) S dW_S
dv = kappa * (theta - v) dt + sigma * sqrt(v) dW_v
corr(dW_S, dW_v) = rho
```

Key parameters:
- `v0`: initial variance (equals vol^2; a 25% initial vol is v0=0.0625)
- `kappa`: mean-reversion speed
- `theta`: long-run variance (sqrt(theta) is the long-run implied vol)
- `sigma`: vol-of-vol; controls smile curvature
- `rho`: spot-variance correlation; negative values produce the equity downward skew

The price is computed via Gil-Pelaez inversion of the characteristic function,
using the little-trap sign convention (Albrecher et al. 2007) to prevent the
branch-cut discontinuity in the original Heston (1993) formula:

```python
from optionview.models import heston

# Typical equity-index Heston parameters
call = heston(
    spot=100.0,
    strike=105.0,
    rate=0.05,
    expiry_years=0.5,
    v0=0.04,        # initial variance: sqrt(0.04) = 20% initial vol
    kappa=3.0,      # mean-reversion speed
    theta=0.04,     # long-run variance: 20% long-run vol
    sigma=0.4,      # vol-of-vol
    rho=-0.7,       # negative skew (equity-like)
    option_type="call",
)
print(f"Heston call: ${call:.4f}")

# Puts via put-call parity
put = heston(
    spot=100.0, strike=105.0, rate=0.05, expiry_years=0.5,
    v0=0.04, kappa=3.0, theta=0.04, sigma=0.4, rho=-0.7,
    option_type="put",
)
print(f"Heston put:  ${put:.4f}")

# Observe the smile: OTM put is richer than OTM call (negative rho effect)
for strike in [90, 95, 100, 105, 110]:
    c = heston(100.0, strike, 0.05, 0.25, 0.04, 3.0, 0.04, 0.4, -0.7)
    print(f"  K={strike}: call=${c:.4f}")
```

The Feller condition (2 * kappa * theta > sigma^2) ensures the variance
process stays strictly positive. When violated, v(t) can touch zero, which
reduces numerical precision for long expiries.

## SABR Stochastic Volatility

The SABR model (Hagan, Kumar, Lesniewski, Woodward 2002) is the industry standard for
interest rate swaptions and FX options. It models the forward price and its instantaneous
volatility as correlated stochastic processes:

```
dF = alpha * F^beta * dW_F
d_alpha = nu * alpha * dW_alpha
corr(dW_F, dW_alpha) = rho * dt
```

The key advantage over Heston is speed: the Hagan approximation gives an analytical formula
for implied vol without numerical integration, making calibration to a full strike slice
orders of magnitude faster. The tradeoff is that the approximation is O(T^2) accurate and
can break down for very long expiries (T > 5 years) or extreme parameters.

Four parameters control the shape of the smile for a given expiry:
- `alpha`: overall vol level (not directly comparable to Black-Scholes vol)
- `beta`: CEV backbone (1.0 for equities/FX, 0.5 for rates)
- `rho`: skew direction; negative rho creates the equity-style downward slope
- `nu`: curvature; nu=0 is a flat (CEV) smile, larger nu widens the wings

`sabr_implied_vol` takes a forward price directly and returns the Black-Scholes equivalent
implied vol. `sabr` wraps it with Black-76 pricing and accepts the same spot/rate/dividend
inputs as the other models in this library.

```python
from optionview.models import sabr, sabr_implied_vol
import math

# Typical equity SABR calibration (beta=1 lognormal backbone)
spot = 100.0
rate = 0.05
T = 0.5

# Parameters calibrated to match a 25% ATM vol with mild negative skew
alpha = 0.25
beta = 1.0
rho = -0.3
nu = 0.4

call = sabr(spot, strike=100.0, rate=rate, expiry_years=T,
            alpha=alpha, beta=beta, rho=rho, nu=nu, option_type="call")
put = sabr(spot, strike=100.0, rate=rate, expiry_years=T,
           alpha=alpha, beta=beta, rho=rho, nu=nu, option_type="put")
print(f"SABR ATM call: ${call:.4f}")
print(f"SABR ATM put:  ${put:.4f}")

# Observe the skew: OTM puts are richer than OTM calls (negative rho effect)
forward = spot * math.exp(rate * T)
for strike in [85, 90, 95, 100, 105, 110, 115]:
    iv = sabr_implied_vol(forward, strike, T, alpha, beta, rho, nu)
    print(f"  K={strike}: SABR IV = {iv:.1%}")
```

Compare SABR against Heston for multi-strike fitting to understand where each model
fits the market smile better:

```python
from optionview.models import sabr_implied_vol, heston

# For a given expiry, SABR produces the smile analytically
# while Heston requires a numerical integral per strike
forward = 100.0
T = 0.5
rate = 0.05

# SABR parameters
alpha, beta, rho_s, nu = 0.25, 1.0, -0.3, 0.4

# Heston parameters (calibrated to similar ATM vol and skew)
v0, kappa, theta, sigma_h, rho_h = 0.0625, 3.0, 0.04, 0.4, -0.7

print(f"{'Strike':>8}  {'SABR IV':>8}  {'Heston price':>12}")
for K in [85, 90, 95, 100, 105, 110, 115]:
    sabr_iv = sabr_implied_vol(forward, K, T, alpha, beta, rho_s, nu)
    from optionview.models import heston
    h_price = heston(forward, K, rate, T, v0, kappa, theta, sigma_h, rho_h)
    print(f"  K={K:>3}:  SABR={sabr_iv:.2%}")
```

The beta parameter controls the relationship between price level and vol level. With
beta=1 (lognormal backbone), the model behaves like a stochastic-vol extension of
Black-Scholes and is the natural choice when vol is roughly proportional to price.
With beta=0 (normal backbone), the model is better suited for low or near-zero forward
rates where a lognormal assumption is problematic.


## SABR Calibration

Given an observed implied vol smile, `calibrate_sabr` fits the SABR parameters (alpha, rho, nu)
to the market quotes while holding beta fixed. The result is a `SABRCalibration` dataclass
containing the fitted parameters and fit quality metrics (RMSE and max absolute error per strike).

Beta is held constant as a prior choice based on the underlying asset class: use beta=1.0 for
equities and FX (lognormal backbone), beta=0.5 for interest rate swaptions (conventional), or
beta=0.0 for the normal backbone near zero rates. Calibrating beta jointly with the other three
parameters is ill-conditioned because rho and nu can partially absorb changes in beta.

The typical workflow is to build a surface first, extract a single-expiry smile slice, convert
each IVPoint to a (forward, strike, expiry_years, iv) tuple, and pass the list to `calibrate_sabr`.
All quotes should share the same expiry; mixing expirations in a single call produces a meaningless
fit because the SABR time-correction term is expiry-specific.

```python
import math
from optionview.fetcher import fetch_option_chain, fetch_spot_price
from optionview.surface import build_surface
from optionview.models import calibrate_sabr

ticker = "SPY"
records = fetch_option_chain(ticker)
spot = fetch_spot_price(ticker)
rate = 0.05
div_yield = 0.013

surface = build_surface(records, spot, rate, div_yield, min_open_interest=50)
exp = surface.expirations[0]

# Compute the risk-neutral forward for this expiry
t = next(p.expiry_years for p in surface.smile(exp))
forward = spot * math.exp((rate - div_yield) * t)

# Build (forward, strike, expiry_years, market_iv) tuples from one smile slice
quotes = [
    (forward, p.strike, p.expiry_years, p.iv)
    for p in surface.smile(exp)
]

fit = calibrate_sabr(quotes, beta=1.0)

print(f"Fitted parameters for {exp}:")
print(f"  alpha = {fit.alpha:.4f}  (overall vol level)")
print(f"  beta  = {fit.beta:.1f}    (fixed, lognormal backbone)")
print(f"  rho   = {fit.rho:.4f}  (skew direction)")
print(f"  nu    = {fit.nu:.4f}   (smile curvature / vol-of-vol)")
print(f"  RMSE  = {fit.rmse:.4f}  ({fit.rmse * 100:.2f} vol points)")
print(f"  max_abs_error = {fit.max_abs_error:.4f}")
print(f"  calibrated to {fit.n_points} strikes")
```

Once calibrated, the fitted parameters can be used to interpolate the smile at any strike,
including strikes not in the original option chain, without building a full surface:

```python
from optionview.models import sabr_implied_vol, sabr, calibrate_sabr

# Re-use the calibrated parameters to price options at arbitrary strikes
for K in [400, 420, 440, 460, 480, 500, 520, 540]:
    iv = sabr_implied_vol(forward, K, t, fit.alpha, fit.beta, fit.rho, fit.nu)
    call = sabr(spot, K, rate, t, fit.alpha, fit.beta, fit.rho, fit.nu,
                option_type="call", dividend_yield=div_yield)
    print(f"  K={K}: IV={iv:.2%}  call=${call:.4f}")
```

Calibration across multiple expirations produces a term structure of SABR parameters:

```python
# Calibrate one SABR slice per expiry to build the full parameter term structure
for exp in surface.expirations:
    t_exp = next(p.expiry_years for p in surface.smile(exp))
    fwd_exp = spot * math.exp((rate - div_yield) * t_exp)
    qs = [(fwd_exp, p.strike, p.expiry_years, p.iv) for p in surface.smile(exp)]
    if len(qs) < 3:
        continue  # need at least 3 strikes for a meaningful 3-parameter fit
    fit_exp = calibrate_sabr(qs, beta=1.0)
    print(
        f"  {exp}  T={t_exp:.3f}  alpha={fit_exp.alpha:.3f}  "
        f"rho={fit_exp.rho:.3f}  nu={fit_exp.nu:.3f}  "
        f"RMSE={fit_exp.rmse:.4f}"
    )
```

A common calibration failure mode is supplying market IVs in percentage form (e.g. 25.0)
rather than decimal form (0.25). The optimizer will converge, but alpha will be approximately
100 times too large, and RMSE will be reported in the same percentage-form units. Always verify
that input market_iv values are decimals before calling `calibrate_sabr`.

## Portfolio Greeks

Aggregate Greeks across a collection of option positions to compute net portfolio
risk. Each position carries a signed quantity: positive means long, negative means
short.

```python
from optionview.portfolio import Position, aggregate_greeks

# A delta-neutral straddle: long call + long put at the same strike
positions = [
    Position(
        spot=150.0, strike=150.0, rate=0.05, volatility=0.25,
        expiry_years=0.25, option_type="call", quantity=10, label="long call"
    ),
    Position(
        spot=150.0, strike=150.0, rate=0.05, volatility=0.25,
        expiry_years=0.25, option_type="put", quantity=10, label="long put"
    ),
]

risk = aggregate_greeks(positions)

print(f"Net delta:  {risk.net_greeks['delta']:+.4f}")   # near zero for ATM straddle
print(f"Net gamma:  {risk.net_greeks['gamma']:+.4f}")   # positive (long gamma)
print(f"Net vega:   {risk.net_greeks['vega']:+.4f}")    # positive (long vol)
print(f"Net theta:  {risk.net_greeks['theta']:+.4f}")   # negative (time decay)
print(f"Dollar delta: ${risk.net_dollar_delta:+.2f}")   # net $ exposure per $1 spot move
print(f"Dollar gamma: ${risk.net_dollar_gamma:+.2f}")   # $ convexity per $1 spot move

# Per-position breakdown
for pr in risk.positions:
    pos = pr.position
    print(
        f"  [{pos.label}] qty={pos.quantity:+g} "
        f"delta={pr.scaled_greeks['delta']:+.4f} "
        f"gamma={pr.scaled_greeks['gamma']:+.4f}"
    )
```

Dollar delta (`unit_delta * spot * quantity`) normalizes position size across
different strikes and expirations, making it easier to compare exposure contributions
when all positions share the same underlying.

## Scenario P&L

`scenario_pnl` applies a second-order Taylor expansion to estimate how a portfolio
gains or loses under a simultaneous move in spot (`ds`), implied volatility (`dvol`),
and calendar time (`dt_days`). It decomposes total P&L into seven named components:

```
P&L = delta*ds + 0.5*gamma*ds^2 + vega*(dvol/0.01) + theta*dt_days
      + vanna*ds*dvol + charm*ds*dt_days + 0.5*volga*(dvol/0.01)^2
```

This is faster than a full reprice and gives a clear attribution across Greek factors.
Cross-sensitivity terms (vanna, charm) are small for modest moves but become material
when spot and vol shift together, which is common around macro events. The volga term
captures the curvature of option value with respect to vol: it is positive for OTM and
ITM positions (which gain vega as vol moves toward the money) and negligible for small
vol moves but meaningful for shifts of 3 or more vol points.

```python
from optionview.portfolio import Position, aggregate_greeks, scenario_pnl

# A long straddle: equal long call and long put at the same strike
positions = [
    Position(spot=100, strike=100, rate=0.05, volatility=0.25,
             expiry_years=0.25, option_type="call", quantity=10),
    Position(spot=100, strike=100, rate=0.05, volatility=0.25,
             expiry_years=0.25, option_type="put", quantity=10),
]
risk = aggregate_greeks(positions)

# Estimate P&L if spot rises $5, vol drops 2 points, and one day passes
pnl = scenario_pnl(risk, ds=5.0, dvol=-0.02, dt_days=1.0)

print(f"Delta P&L:  {pnl.delta_pnl:+.2f}")   # first-order spot contribution
print(f"Gamma P&L:  {pnl.gamma_pnl:+.2f}")   # convexity benefit (always >= 0 for long gamma)
print(f"Vega P&L:   {pnl.vega_pnl:+.2f}")    # vol compression hurts long vol book
print(f"Theta P&L:  {pnl.theta_pnl:+.2f}")   # time decay cost
print(f"Vanna P&L:  {pnl.vanna_pnl:+.2f}")   # spot-vol cross term
print(f"Charm P&L:  {pnl.charm_pnl:+.2f}")   # spot-time cross term
print(f"Volga P&L:  {pnl.volga_pnl:+.2f}")   # second-order vol curvature
print(f"Total P&L:  {pnl.total_pnl:+.2f}")
```

For large moves (more than roughly 5% of spot) or horizons beyond 30 days, the
expansion accuracy degrades and a full reprice using the pricing models is more
reliable. The expansion does not capture higher-order spot terms (speed, color) or
carry effects, so it works best for short-horizon stress scenarios. The included
volga term improves vol-move accuracy relative to a pure first-order vega estimate,
particularly for OTM positions experiencing vol expansion.

## Project Structure

```
optionview/
  __init__.py      - Package entry point and version info
  models.py        - Pricing model implementations (BS, BT, MC, Heston, SABR)
  greeks.py        - Analytical Greeks calculations
  fetcher.py       - Market data retrieval from free APIs
  surface.py       - Implied volatility surface construction and smile analysis
  compare.py       - Model-vs-market comparison utilities
  portfolio.py     - Portfolio-level Greeks aggregation
```

## Pricing Models

### Black-Scholes
The classic closed-form solution for European options. Fast and accurate for vanilla contracts.

### Binomial Tree (Cox-Ross-Rubinstein)
A lattice-based approach that supports both European and American-style options. Configurable step count for precision tuning.

### Monte Carlo Simulation
Path-based simulation with antithetic variates for variance reduction. Best suited for exotic payoffs or when analytical solutions are unavailable.

### Heston Stochastic Volatility
A two-factor model where variance follows a mean-reverting CIR process correlated with the spot. Reproduces the implied vol smile and skew that flat-vol models cannot capture. Priced via Gil-Pelaez characteristic function inversion with the Albrecher et al. (2007) little-trap sign convention for numerical stability.

### SABR Stochastic Volatility
A two-factor model where the instantaneous vol follows a lognormal process correlated with the forward price. The Hagan et al. (2002) analytical approximation gives implied Black-Scholes vol without numerical integration, making it the industry standard for fast calibration to option smile slices. The beta parameter controls the CEV backbone: beta=1 is lognormal (equities/FX), beta=0 is normal (rates).

## Roadmap

- HTML dashboard for interactive comparison
- Volatility surface visualization (surface construction is implemented; charting is not)
- Historical backtesting of model accuracy

## License

MIT
