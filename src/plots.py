import numpy as np
import matplotlib.pyplot as plt
from black_scholes import black_scholes
from greeks import greeks
from binomial import binomial_tree
from implied_vol import implied_vol

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def plot_smile(path):
    S, r, T = 100, 0.05, 1.0
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quoted  = [0.265, 0.245, 0.225, 0.21, 0.20, 0.205, 0.215, 0.235, 0.26]
    ivs = [implied_vol(black_scholes(S, K, r, q, T)[0], S, K, r, T)
           for K, q in zip(strikes, quoted)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(strikes, ivs, "o-", lw=2)
    ax.axhline(0.20, ls="--", color="gray", label="BS constant-vol assumption")
    ax.set(xlabel="Strike K", ylabel="Implied volatility", title="Volatility Smile")
    ax.legend(); fig.tight_layout(); fig.savefig(path)


def plot_convergence(path):
    r, T = 0.05, 1.0
    ns = np.arange(5, 401)
    prices = [binomial_tree(100, 100, r, 0.2, T, int(n)) for n in ns]
    bs = black_scholes(100, 100, r, 0.2, T)[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, prices, lw=1, label="Binomial price")
    ax.axhline(bs, ls="--", color="red", label=f"Black-Scholes = {bs:.4f}")
    ax.set(xlabel="Steps n", ylabel="Call price", title="Binomial Convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(path)


def plot_gamma(path):
    r = 0.05
    spots = np.linspace(60, 140, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    for Tx in [1.0, 0.5, 0.1, 0.02]:
        gammas = [greeks(s, 100, r, 0.2, Tx)["gamma"] for s in spots]
        ax.plot(spots, gammas, lw=2, label=f"T={Tx}yr")
    ax.axvline(100, ls=":", color="gray")
    ax.set(xlabel="Spot S", ylabel="Gamma", title="Gamma Near Expiry")
    ax.legend(); fig.tight_layout(); fig.savefig(path)


if __name__ == "__main__":
    import os
    os.makedirs("../figures", exist_ok=True)
    plot_smile("../figures/vol_smile.png")
    plot_convergence("../figures/convergence.png")
    plot_gamma("../figures/gamma_expiry.png")
    print("Saved 3 figures to figures/")