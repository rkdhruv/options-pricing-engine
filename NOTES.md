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

## Layer 2: The Greeks

### The one idea that ties all of them together
 
The thing that made this click: **I don't have five new formulas to memorize. I have one function and five derivatives of it.** The price `V` is a function of `S, K, r, σ, T`, and a Greek is just `V` differentiated with respect to one of those inputs: "if I nudge this input a little, how much does `V` move?" That's a partial derivative. The Greek names are finance's labels for them.
 
Which means the engineering is the same as Layer 1, they're all closed-form, no new machinery. The actual value is being able to say what each one *means physically*. So that's what I'm pinning down here.
 
### The set, in plain English
 
- **Delta (`∂V/∂S`)** — how much the option's price moves per $1 move in the stock. A call delta of 0.64 means: stock goes up $1, the call gains ~$0.64. It's also the **hedge ratio** — short 0.64 shares per call and you're (momentarily) immune to small stock moves. Call delta lives in `[0, 1]`, put delta in `[−1, 0]`.
- **Gamma (`∂²V/∂S²`)** — how much delta *itself* changes as the stock moves. Second derivative, the curvature. This is why delta-hedging is hard: a 0.64-share hedge is only right for this instant; as the stock moves, delta drifts, and gamma says how fast. High gamma = hedge goes stale fast. The line to remember: **gamma is the risk that remains in a delta-hedged book.**
- **Vega (`∂V/∂σ`)** — how much the price moves when volatility changes. This is the one that connects forward: in Layer 4 I back out implied vol with Newton-Raphson, and **vega is exactly the derivative that method needs** — so I'm building it now and reusing it later.
- **Theta (`∂V/∂t`)** — how much value bleeds away as time passes. Almost always negative for a long option: every day that goes by with nothing happening, the option is worth a little less. This is **time decay**.
- **Rho (`∂V/∂r`)** — sensitivity to the interest rate. Least-watched of the five in practice, but it completes the set.

### Why gamma and vega are the same for calls and puts
 
This ties straight back to put-call parity from Layer 1. Parity says a call and put differ by `S − K·e^(−rT)` — a term that's **linear in `S`** and **has no `σ` in it**.
 
- Differentiate that difference twice by `S` (gamma) → it vanishes → **gamma is identical** for call and put.
- Differentiate it once by `σ` (vega) → it vanishes → **vega is identical** too.
- Delta, theta, and rho *do* differ between call and put, because that parity term survives their derivatives (it's linear in `S` so delta picks up a `1`; it carries `T` and `r` so theta and rho pick up terms).
Two-sentence version I want to be able to give on the spot: gamma and vega match because the call−put gap is linear in `S` and free of `σ`, so it dies under those derivatives. Delta, theta, and rho differ because that same gap survives differentiation by `S`, `t`, and `r`.
 