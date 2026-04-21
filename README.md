# optionview

A Python toolkit for comparing option pricing models against real-time market data. Built for traders and quant enthusiasts who want to evaluate how theoretical models stack up against live prices, all using freely available data sources.

## Features

- **Multiple Pricing Models**: Black-Scholes, Binomial Tree, Monte Carlo simulation, and Heston stochastic volatility
- **Live Market Data**: Fetches real-time option chains from Yahoo Finance (no API key required)
- **Model Comparison**: Side-by-side comparison of theoretical vs. market prices
- **Volatility Surface Construction**: Builds an implied volatility surface from multi-expiry option chains, with per-expiry smile analysis (OLS slope and IV range), ATM term structure extraction, log-moneyness normalized IVPoints, and piecewise-linear IV interpolation across strikes
- **Greeks Calculation**: Full analytical Greeks including Delta, Gamma, Theta, Vega, Rho, Epsilon (dividend sensitivity), Vanna, and Charm
- **Implied Volatility Solver**: Newton-Raphson IV solver with configurable tolerance
- **Portfolio Greeks Aggregation**: Aggregate Greeks across a collection of long and short option positions, with per-position and net portfolio risk
- **Scenario P&L**: Estimate portfolio P&L for a simultaneous move in spot, implied vol, and time using a second-order Taylor expansion (delta, gamma, vega, theta, vanna, charm) — useful for stress testing and what-if analysis without a full reprice
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
and calendar time (`dt_days`). It decomposes total P&L into six named components:

```
P&L = delta*ds + 0.5*gamma*ds^2 + vega*(dvol/0.01) + theta*dt_days
      + vanna*ds*dvol + charm*ds*dt_days
```

This is faster than a full reprice and gives a clear attribution across Greek factors.
Cross-sensitivity terms (vanna and charm) are small for modest moves but become
material when spot and vol shift together, which is common around macro events.

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
print(f"Total P&L:  {pnl.total_pnl:+.2f}")
```

For large moves (more than roughly 5% of spot) or horizons beyond 30 days, the
expansion accuracy degrades and a full reprice using the pricing models is more
reliable. The expansion does not capture vol-of-vol (volga) or higher-order spot
terms, so it works best for short-horizon stress scenarios.

## Project Structure

```
optionview/
  __init__.py      - Package entry point and version info
  models.py        - Pricing model implementations
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

## Roadmap

- HTML dashboard for interactive comparison
- Volatility surface visualization (surface construction is implemented; charting is not)
- Additional models (SABR)
- Historical backtesting of model accuracy

## License

MIT
