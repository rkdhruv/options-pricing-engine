import numpy as np
from black_scholes import black_scholes
from greeks import greeks
from binomial import binominal_tree
from monte_carlo import monte_carlo

def price(S, K, r, sigma, T, option="call"):
    call, put = black_scholes(S, K, r, sigma, T)
    return call if option == "call" else put

def check_parity(S, K, r, sigma, T, option="call"):
    call, put = black_scholes(S, K, r, sigma, T)
    lhs,rhs = call - put, S - K * np.exp(-r * T)
    print(f"[parity] diff = {abs(lhs - rhs):.2e}    "
          f"{'PASS' if abs(lhs - rhs) < 1e-9 else 'FAIL'}")

def check_greek(S, K, r, sigma, T, h=1e-4):
    g = greeks(S, K, r, sigma, T)
    fd = {
        "delta": (price(S+h,K,r,sigma,T) - price(S-h,K,r,sigma,T)) / (2*h),
        "gamma": (price(S+h,K,r,sigma,T) - 2*price(S,K,r,sigma,T)
                  + price(S-h,K,r,sigma,T)) / h**2,
        "vega":  (price(S,K,r,sigma+h,T) - price(S,K,r,sigma-h,T)) / (2*h),
        "theta": -(price(S,K,r,sigma,T+h) - price(S,K,r,sigma,T-h)) / (2*h),
        "rho":   (price(S,K,r+h,sigma,T) - price(S,K,r-h,sigma,T)) / (2*h),
    }
    analytic = {"delta": g["call_delta"], "gamma": g["gamma"], "vega": g["vega"],
                "theta": g["call_theta"], "rho": g["call_rho"]}
    for name in fd:
        diff = abs(analytic[name] - fd[name])
        print(f"[greek] {name:6} diff = {diff:.2e}  "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}")

def check_convergence(S, K, r, sigma, T):
    bs = price(S, K, r, sigma, T)
    for n in [10, 100, 1000]:
        print(f"[tree] n={n:5d} gap = {abs(binominal_tree(S, K, r, sigma, T, n) - bs):.4f}")
    for N in [1_000, 100_000, 1_000_000]:
        mc, _ = monte_carlo(S, K, r, sigma, T, N)
        print(f"[mc]    N={N:8d} gap = {abs(mc - bs):.4f}")

if __name__ == "__main__":
    args = (100, 100, 0.05, 0.2, 1.0)
    check_parity(*args)
    check_greek(*args)
    check_convergence(*args)

