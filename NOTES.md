# Options Pricing Engine — Notes

## Layer 1: Black-Scholes

### Notation (used everywhere from here on)

- `S` — spot price of the underlying
- `K` — strike
- `r` — risk-free rate, annualized, continuous compounding
- `σ` — volatility, annualized
- `T` — time to expiry, in years
- `q` — continuous dividend yield. **Ignoring it for now** (`q = 0`, no dividends). Comes back in a later layer.
- `N(x)` — standard normal CDF → `scipy.stats.norm.cdf`
- `N'(x)` — standard normal PDF. Don't need it in this layer; it shows up when I do the Greeks.

### The call

```
Call = S·N(d1) − K·e^(−rT)·N(d2)
```

The way I read this whole thing: it's **benefit minus cost**, and each side is weighted by a probability.

- **`K·e^(−rT)·N(d2)` is the cost side.** It's the strike I have to pay, discounted back to today with `e^(−rT)`, because I pay `K` at expiry, not now, so it's worth less in today's dollars. Then it's weighted by `N(d2)`. **`N(d2)` is the risk-neutral probability that the call finishes in the money** (`S_T > K`),  basically the probability I ever pay that strike at all.
- **`S·N(d1)` is the benefit side.** It's the value of the stock I'd receive if I exercise. But `N(d1)` is *not* a clean probability the way `N(d2)` is. It's the **delta-adjusted** version. `N(d1)` is actually the delta of the call (∂Call/∂S, when `q = 0`), so this term is really "how much stock exposure am I effectively holding." **That's the whole reason d1 ≠ d2** — the two N(·) terms answer different questions: one is a probability, the other is an exposure.

### d1 and d2

```
d1 = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d2 = d1 − σ·√T
```

What's actually happening inside **d1**:
- `ln(S/K)` — how far in/out of the money I am, in log terms. Positive means spot is above the strike.
- `(r + σ²/2)·T` — the drift over the life of the option. The `r·T` part is just risk-free growth. The `σ²/2` part is the convexity correction that falls out of Itô — because the stock is **lognormal**, the expected log-return isn't simply `r`, you add back `σ²/2`.
- dividing by `σ·√T` standardizes the whole thing by the total volatility built up over the life of the option.

**d2** is basically d1 with that drift adjustment stripped back out. The way I think about it: **d2 is like a z-score.** It's how many standard deviations the (drift-adjusted) log-moneyness sits above the strike. That's exactly why `N(d2)` reads as a clean probability of finishing ITM — it's literally the area under the normal curve past that z-score.

The gap between them is `σ·√T` — the **total volatility over the life of the option**:
- short-dated or low-vol → small gap → d1 and d2 nearly equal → `N(d1)` and `N(d2)` close together.
- long-dated or high-vol → big gap → the two terms pull apart.

That gap is essentially where all the option's time/vol value lives.

### The put

```
Put = K·e^(−rT)·N(−d2) − S·N(−d1)
```

Same two pieces, just mirrored with the `−d` arguments (using `N(−x) = 1 − N(x)`). Here `N(−d2)` is the risk-neutral probability the **put** finishes ITM (`S_T < K`).

### Put-call parity

```
Call − Put = S − K·e^(−rT)
```

If my two prices don't satisfy this, something's broken. I could honestly just price the call and back the put out of parity, but for now I compute both directly so I can cross-check them against each other.
