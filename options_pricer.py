

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#  1. Modèle Black-Scholes 

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S     : prix spot
    K     : strike
    T     : maturité (en années)
    r     : taux sans risque
    sigma : volatilité implicite
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


#  2. Les Grecques 


def greeks(S, K, T, r, sigma, option_type="call"):

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100          # pour 1% de vol

    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # pour 1bp de taux

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}

#  3. Stress Test (choc de volatilité) 


def stress_test_vol(S, K, T, r, sigma_base, option_type="call"):
    """Simule des chocs de ±10%, ±20%, ±30% sur la volatilité"""

    shocks = [-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30]
    base_price = black_scholes(S, K, T, r, sigma_base, option_type)

    print(f"{'Choc σ':>10} | {'Vol résult.':>12} | {'Prix':>8} | {'P&L':>8}")

    print("-" * 46)

    for shock in shocks:
        new_sigma = max(0.001, sigma_base + shock)
        price = black_scholes(S, K, T, r, new_sigma, option_type)
        pnl = price - base_price
        print(f"{shock*100:>+9.0f}% | {new_sigma*100:>11.1f}% | {price:>8.4f} | {pnl:>+8.4f}")


#  4. Graphe : Delta en fonction du Spot 

def plot_delta_vs_spot(K=100, T=1, r=0.05, sigma=0.20):

    spots = np.linspace(50, 150, 200)

    deltas_call = [greeks(s, K, T, r, sigma, "call")["delta"] for s in spots]

    deltas_put  = [greeks(s, K, T, r, sigma, "put")["delta"]  for s in spots]

    plt.figure(figsize=(9, 5))

    plt.plot(spots, deltas_call, label="Delta Call", color= "red", lw=2)

    plt.plot(spots, deltas_put,  label="Delta Put",  color="blue", lw=2)

    plt.axvline(K, ls="--", color="gray", alpha=0.5, label="Strike K")

    plt.xlabel("Spot (S)"); plt.ylabel("Delta")

    plt.title("Delta Call & Put en fonction du Spot")

    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

#  5. Main 

if __name__ == "__main__":

    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    price = black_scholes(S, K, T, r, sigma, "call")

    g     = greeks(S, K, T, r, sigma, "call")

    print(f"Prix Call : {price:.4f}")
    print(f"Delta : {g['delta']:.4f} | Gamma : {g['gamma']:.6f}")
    print(f"Vega  : {g['vega']:.4f} | Theta : {g['theta']:.4f} | Rho : {g['rho']:.4f}")
    print()
    stress_test_vol(S, K, T, r, sigma)
    plot_delta_vs_spot()