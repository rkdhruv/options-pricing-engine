import numpy as np
from scipy.stats import norm

def greeks(S, K, r, sigma, T):
    """
    First-order Greeks (plus gamma) for European options, Black-Scholes.
    Returns a dict of raw per-unit sensitivities.
    """
    
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    
    # Delta = dV/dS
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1
    
    # Gamma = d2V/dS2 (same for call and put)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    
    # Vega = dV/dsigma (same for call and put)
    vega = S * pdf_d1 * np.sqrt(T)
    
    # Theta = dV/dt
    call_theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * disc * norm.cdf(d2)
    put_theta  = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * disc * norm.cdf(-d2)
    
    # Rho = dV/dr
    call_rho = K * T * disc * norm.cdf(d2)
    put_rho = -K * T * disc * norm.cdf(-d2)
    
    return {
        "call_delta": call_delta, "put_delta": put_delta,
        "gamma": gamma, "vega": vega,
        "call_theta": call_theta, "put_theta": put_theta,
        "call_rho": call_rho, "put_rho": put_rho,
    }

if __name__ == "__main__":
    g = greeks(100, 100, 0.05, 0.2, 1.0)
    for name, value in g.items():
        print(f"{name:12s}: {value: .4f}")