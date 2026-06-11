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

## Layer 3: Numerical methods (binomial tree + Monte Carlo)

### The one idea that unifies all three methods

Black-Scholes, the binomial tree, and Monte Carlo all compute the **same quantity** — the discounted expected payoff under the risk-neutral measure:

```
price = e^(−rT) · E[payoff]
```

They just evaluate that expectation three different ways:
- **Black-Scholes** does the integral analytically — clean formula, but only possible because the vanilla payoff is simple enough to integrate by hand.
- **Binomial tree** computes it by exhaustively summing over a discretized branching of all the prices the stock could take.
- **Monte Carlo** samples random futures and averages them.
Same target, three routes — and each route is good at something the others aren't. That's the whole reason this layer exists.

### Binomial tree (Cox-Ross-Rubinstein)

The picture: chop the life of the option into `n` little time steps. In each step the stock makes a coin flip — up by factor `u`, or down by `d = 1/u`. Do that `n` times and you get a tree of every price the stock could reach. CRR picks

```
u = e^(σ·√Δt)
d = 1/u
p = (e^(rΔt) − d) / (u − d)
```

calibrated so the discrete coin-flipping has the right volatility and drifts at the risk-free rate. `p` is the **risk-neutral** up-probability, not a real-world one.

How you actually price it — **backward induction**:
- At expiry the option value is dead simple: it's just the payoff, `max(S_T − K, 0)` for a call.
- Then walk *backward*. The value at any node is the discounted average of its two children: `e^(−rΔt)·[ p·V_up + (1−p)·V_down ]`.
- Repeat until the whole tree collapses to one number at today — that's the price.

**Why the tree earns its place: American options.** Because the tree knows the stock price at *every* intermediate node, at each one I can ask "would I be better off exercising right now?" and take `max(keep holding, exercise now)`. Closed-form Black-Scholes physically can't do this — it only knows expiry. My run priced the American put at **6.09** vs the European put's **5.57**. That **0.53 gap is the value of being allowed to exercise early**, and the tree is what captured it.

### Monte Carlo

Conceptually the simplest of the three: simulate the future a hundred-thousand-plus times and average what you get. Each simulated future is one draw of the terminal price from its risk-neutral distribution:

```
S_T = S · exp( (r − σ²/2)·T + σ·√T·Z ),   Z ~ N(0, 1)
```

That's geometric Brownian motion solved out to time `T`: a drift term `(r − σ²/2)·T` plus a random shock scaled by vol and a standard normal draw `Z`. Compute the payoff for each simulated `S_T`, average them, discount once. The law of large numbers guarantees that average converges to the true expected payoff.

**Why MC earns its place: flexibility.** The closed form is locked to vanilla European payoffs. MC doesn't care how weird the payoff is — Asian (average over the path), barrier (knocks out if a level is touched), lookback — I just swap out the payoff function. That generality is the whole point, and it's why exotic-option desks live on MC.

**The catch: convergence is slow, ~`1/√N`.** To halve the error I need *four times* the paths. That's why the function returns a standard error — a built-in error bar — and why my run reports a `±` band instead of an exact number. I also seed the RNG so runs are reproducible; without it MC gives a slightly different number every run, which is correct but annoying when checking work.

### Verify (the "done when")

All three land on ~10.45 from completely different machinery:

```
Black-Scholes call : 10.4506
Binomial (n=500)   : 10.4466            <- off by 0.004
Monte Carlo (200k) : 10.4634 ± 0.0649   <- BS sits inside the band
```

The tree is close and tightens as `n` grows; MC's confidence interval comfortably contains the true value. **That agreement from three independent directions is the deliverable for this layer.**

The convergence, the tree marching toward the Black-Scholes line as `n` grows:

```
n=10:   10.2534
n=50:   10.4107
n=100:  10.4306
n=500:  10.4466
n=2000: 10.4496
```

This is exactly the "binomial → Black-Scholes as n → ∞" demonstration. The actual plot is a later (README) item, but the numbers are already sitting here.

## Layer 4: Implied volatility — the conceptual flip

*(In progress — just the concept so far, code and the rest come later.)*

### The basic idea

Everything until now ran Black-Scholes *forward*: feed it σ, get a price. Here I flip it — instead of already knowing σ, I **find** σ for a given market price. That's how markets actually work: nobody can observe future volatility, but the option's price is right there on the screen, so I run the model in reverse to get the σ the price is implying.

The catch: there's no way to invert Black-Scholes for σ — you can't isolate it algebraically. So this becomes a **root-finding** problem instead.

It's a well-posed one, though, because the price is **strictly increasing in σ** (more vol always means a more valuable option — that's vega being positive everywhere). A strictly increasing function hits each output exactly once, so for any sensible market price there's a single, unique implied vol to find. I just have to hunt for it numerically.

### Newton-Raphson, and why vega is the engine

Newton-Raphson solves `f(x) = 0` by guessing and then repeatedly improving:

```
x_next = x − f(x) / f'(x)
```

In my case the function is `f(σ) = BlackScholes(σ) − market_price` — I want the σ that drives that difference to zero. Its derivative `f'(σ)` is `∂(price)/∂σ`, which is exactly **vega**. So the thing I built back in Layer 2 turns out to be the precise derivative this method needs. Each step:

```
σ_next = σ − (BS_price(σ) − market_price) / vega(σ)
```

That's the payoff I flagged two layers ago: vega wasn't just one of five Greeks to list — it's the gradient that lets me invert the model. Newton converges fast (roughly doubling the correct digits each step) when it behaves.

### Why I need a fallback

Newton is fast but not bulletproof. For options deep in- or out-of-the-money, vega shrinks toward zero — and dividing by a near-zero derivative makes Newton take wild, divergent jumps. A bad starting guess can send it off a cliff too. So the robust pattern is: **try Newton, and if it misbehaves, fall back to Brent's method** (`scipy.optimize.brentq`).

Brent is a bracketing method — from my understanding, you hand it an interval known to contain the root (say vol between 0 and 500%), and it's mathematically guaranteed to converge because it always keeps the root trapped inside a shrinking bracket. It trades a little speed for total reliability. Fast-but-fragile with a slow-but-safe backstop is the design choice here, and a clean thing to be able to explain.
