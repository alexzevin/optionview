# optionview

A Python toolkit for comparing option pricing models against real-time market data. Built for traders and quant enthusiasts who want to evaluate how theoretical models stack up against live prices, all using freely available data sources.

## Features

- **Multiple Pricing Models**: Black-Scholes, Binomial Tree, and Monte Carlo simulation
- **Live Market Data**: Fetches real-time option chains from Yahoo Finance (no API key required)
- **Model Comparison**: Side-by-side comparison of theoretical vs. market prices
- **Greeks Calculation**: Full analytical Greeks including Delta, Gamma, Theta, Vega, Rho, Epsilon (dividend sensitivity), Vanna, and Charm
- **Implied Volatility Solver**: Newton-Raphson IV solver with configurable tolerance
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
from optionview.models import black_scholes, binomial_tree, monte_carlo
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

# Compare all three models
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

## Project Structure

```
optionview/
  __init__.py      - Package entry point and version info
  models.py        - Pricing model implementations
  greeks.py        - Analytical Greeks calculations
  fetcher.py       - Market data retrieval from free APIs
  surface.py       - Implied volatility surface construction and smile analysis
  compare.py       - Model-vs-market comparison utilities
```

## Pricing Models

### Black-Scholes
The classic closed-form solution for European options. Fast and accurate for vanilla contracts.

### Binomial Tree (Cox-Ross-Rubinstein)
A lattice-based approach that supports both European and American-style options. Configurable step count for precision tuning.

### Monte Carlo Simulation
Path-based simulation with antithetic variates for variance reduction. Best suited for exotic payoffs or when analytical solutions are unavailable.

## Roadmap

- HTML dashboard for interactive comparison
- Volatility surface visualization (surface construction is implemented; charting is not)
- Additional models (Heston, SABR)
- Portfolio-level Greeks aggregation
- Historical backtesting of model accuracy

## License

MIT
