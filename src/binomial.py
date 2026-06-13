import numpy as np

def binomial_tree(S, K, r, sigma, T, n=500, option="call", american=False):
    """
    Price an option with a Cox-Ross-Rubinstein binomial tree.
    n: number of time steps. american=True checks early exercise.
    """
    
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))     # up factor
    d = 1 / u                           # down factor
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral up-probability
    disc = np.exp(-r * dt)              # one-step discount
    
    # Stock prices at expiry: node j has had j up-moves, (n-j) down-moves
    j = np.arange(n + 1)
    S_T = S * u**j * d**(n-j)
    
    # Option value at expiry = payoff
    V = np.maximum(S_T - K, 0.0) if option == "call" else np.maximum(K - S_T, 0.0)
    
    # Backward induction: collapse the tree one layer at a time
    for i in range(n, 0, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            j = np.arange(i)
            S_i = S * u**j * d**(i - 1 - j)     # price at this earlier layer
            intrinsic = (S_i - K) if option == "call" else (K - S_i)
            V = np.maximum(V, intrinsic)        # exercise now if it beats holding
    return V[0]

if __name__ == "__main__":
    print(binomial_tree(100, 100, 0.05, 0.2, 1.0, n=500))
    print(binomial_tree(100, 100, 0.05, 0.2, 1.0, n=500, option="put", american=True))

