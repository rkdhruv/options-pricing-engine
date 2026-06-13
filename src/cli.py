import argparse
from black_scholes import black_scholes
from greeks import greeks
from binomial import binomial_tree
from monte_carlo import monte_carlo


def main():
    p = argparse.ArgumentParser(description="Option pricing engine with Greeks")
    p.add_argument("--S", type=float, required=True, help="spot price")
    p.add_argument("--K", type=float, required=True, help="strike price")
    p.add_argument("--sigma", type=float, required=True, help="volatility (annual)")
    p.add_argument("--T", type=float, required=True, help="years to expiry")
    p.add_argument("--r", type=float, default=0.05, help="risk-free rate")
    p.add_argument("--option", choices=["call", "put"], default="call")
    p.add_argument("--method", choices=["bs", "tree", "mc"], default="bs")
    args = p.parse_args()

    if args.method == "bs":
        call, put = black_scholes(args.S, args.K, args.r, args.sigma, args.T)
        price = call if args.option == "call" else put
    elif args.method == "tree":
        price = binomial_tree(args.S, args.K, args.r, args.sigma, args.T,
                              option=args.option)
    else:
        price, se = monte_carlo(args.S, args.K, args.r, args.sigma, args.T,
                                option=args.option)

    print(f"{args.option.capitalize()} price ({args.method}): {price:.4f}")

    if args.method == "bs":
        g = greeks(args.S, args.K, args.r, args.sigma, args.T)
        print("Greeks:")
        for name, value in g.items():
            print(f"  {name:12s}: {value: .4f}")


if __name__ == "__main__":
    main()