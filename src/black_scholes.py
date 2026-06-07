import numpy as np
from scipy.stats import norm

def black_scholes(S, K, r, sigma, T):
    """
    Price a European call and put with the Black-Scholes model.
    
    S       :   spot price of the underlying
    K       :   strike price
    r       :   risk-free rate (annual, continuos compounding)
    sigma   :   volatility (annual)
    T       :   time to expiry in years
    
    Returns (call_price, put_price).
    """
    
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call, put


if __name__ == "__main__":
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    call, put = black_scholes(S, K, r, sigma, T)
    print(f"Call: {call:.4f}")
    print(f"Put: {put:.4f}")