import numpy as np

def monte_carlo(S, K, r, sigma, T, n_paths=200_000, option="call", seed=42):
    """
    Price a European option by Monte Carlo under geometric Brownian motion.
    Returns (price, standard_error).
    """
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    
    S_T = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(S_T - K, 0.0) if option == "call" else np.maximum(K - S_T, 0.0)
    
    price = np.exp(-r * T) * payoff.mean()
    stderr = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr

if __name__ == "__main__":
    price, se = monte_carlo(100, 100, 0.05, 0.2, 1.0)
    print(f"{price:.4f}  (95% CI +/- {1.96 * se:.4f})")