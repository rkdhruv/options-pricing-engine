import numpy as np
from scipy.optimize import brentq
from black_scholes import black_scholes # Layer 1
from greeks import greeks               # Layer 2 (reusing vega)

def _price(S, K, r, sigma, T, option):
    call, put = black_scholes(S, K, r, sigma, T)
    return call if option == "call" else put

def implied_vol(market_price, S, K, r, T, option="call", tol=1e-8, max_iter=100):
    """
    Back out the volatility implied by a market option price.
    Newton-Raphson (using vega as the derivative), whith Brent as a fallback
    """
    
    sigma = 0.2 # starting guess
    for _ in range(max_iter):
        diff = _price(S, K, r, sigma, T, option) - market_price
        if abs(diff) < tol:
            return sigma
        v = greeks(S, K, r, sigma, T)["vega"]       # the derivative Newton needs
        if v < 1e-8:        # vega too small: Newton is unsafe
            break
        sigma = sigma - diff / v
        if sigma <= 0:      # stepped into nonsense
            break
    
    # Robust fallback: bracketed root-find between 0 and 500% vol
    try:
        return brentq(lambda s: _price(S, K, r, s, T, option) - market_price, 1e-6, 5.0, xtol=tol)
    except ValueError:
        return np.nan       # Price lies outside no-arbitrage bounds.


if __name__ == "__main__":
    mkt, _ = black_scholes(100, 100, 0.05, 0.2, 1.0)    # price at vol=0.20
    print(implied_vol(mkt, 100, 100, 0.05, 1.0))        # recovers around 0.20

