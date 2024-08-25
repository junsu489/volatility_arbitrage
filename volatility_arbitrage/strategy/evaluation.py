"""strategy evaluation tool"""

# pylint: disable=line-too-long
import matplotlib.pyplot as plt
import numpy as np

from volatility_arbitrage.pricing_model.interface import StrategyPnl


def plot_pnl_of_path(pnl: StrategyPnl, path: int) -> None:
    """plot P&L of one sample path"""
    fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    length = pnl["total_pnl"].shape[0]
    for i, (key, value) in enumerate(pnl.items()):
        ax[i // 2, i % 2].set_title(key)
        ax[i // 2, i % 2].plot(value[:, path].cumsum())  # type: ignore[index]
        ax[i // 2, i % 2].hlines(0, xmin=0, xmax=length, color="black")
    ax[3, 0].set_title("gamma + vanna + theta pnl")
    ax[3, 0].plot(
        (pnl["gamma_pnl"][:, path] + pnl["vanna_pnl"][:, path] + pnl["theta_pnl"][:, path]).cumsum()
    )
    ax[3, 0].hlines(0, xmin=0, xmax=length, color="black")
    ax[3, 1].set_title("var_vega_pnl + vega_hedge_pnl")
    ax[3, 1].plot((pnl["var_vega_pnl"][:, path] + pnl["vega_hedge_pnl"][:, path]).cumsum())
    ax[3, 1].hlines(0, xmin=0, xmax=length, color="black")
    fig.suptitle("P&L Decomposition")
    fig.tight_layout()
    plt.show()


def plot_vega_hedge_performance_of_path(pnl: StrategyPnl, path: int) -> None:
    """plot performance of vega hedging for one sample path"""
    var_vega_pnl = pnl["var_vega_pnl"][:, path]
    vega_hedge_pnl = pnl["vega_hedge_pnl"][:, path]
    hedged_pnl = vega_hedge_pnl + var_vega_pnl

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(vega_hedge_pnl, var_vega_pnl)
    ax[0].set_xlabel("Vega Hedge P&L")
    ax[0].set_ylabel("Variance Vega P&L")

    unhedged_pnl_var = (var_vega_pnl**2).mean()
    hedged_pnl_var = (hedged_pnl**2).mean()
    hedged_ratio = 1 - hedged_pnl_var / unhedged_pnl_var
    ax[0].hlines(
        y=0,
        xmin=vega_hedge_pnl.min(),
        xmax=vega_hedge_pnl.max(),
        color="black",
    )
    ax[0].vlines(
        x=0,
        ymin=var_vega_pnl.min(),
        ymax=var_vega_pnl.max(),
        color="black",
    )

    bins = int(np.ceil(np.sqrt(len(var_vega_pnl))))
    ax[1].hist(var_vega_pnl, label="Variance Vega P&L", density=True, alpha=0.5, bins=bins)
    ax[1].hist(hedged_pnl, label="Hedged P&L", density=True, alpha=0.5, bins=bins)
    ax[1].legend()

    fig.suptitle(f"{100*hedged_ratio:.2f}% P&L hedged")
    fig.tight_layout()
    plt.show()


def plot_pnl_distribution(pnl: StrategyPnl) -> None:
    """plot distribution of final P&L of each sample path"""
    fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    bins = int(np.ceil(np.sqrt(pnl["total_pnl"].shape[1])))
    for i, (key, value) in enumerate(pnl.items()):
        final_pnl = value.sum(axis=0)  # type: ignore[attr-defined]
        ax[i // 2, i % 2].set_title(
            f"{key}: mean = {final_pnl.mean():.2f}, std = {final_pnl.std():.2f}"
        )
        ax[i // 2, i % 2].hist(final_pnl, density=True, bins=bins)
    gamma_vanna_theta_pnl = (pnl["gamma_pnl"] + pnl["vanna_pnl"] + pnl["theta_pnl"]).sum(axis=0)
    ax[3, 0].set_title(
        f"gamma + vanna + theta pnl: mean = {gamma_vanna_theta_pnl.mean():.2f}, std = {gamma_vanna_theta_pnl.std():.2f}"
    )
    ax[3, 0].hist(gamma_vanna_theta_pnl, density=True, bins=bins)

    hedged_pnl = (pnl["var_vega_pnl"] + pnl["vega_hedge_pnl"]).sum(axis=0)
    ax[3, 1].set_title(
        f"var_vega_pnl + vega_hedge_pnl: mean = {hedged_pnl.mean():.2f}, std = {hedged_pnl.std():.2f}"
    )
    ax[3, 1].hist(hedged_pnl, density=True, bins=bins)
    fig.suptitle("P&L Distribution")
    fig.tight_layout()
    plt.show()
