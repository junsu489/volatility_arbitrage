import matplotlib.pyplot as plt

from volatility_arbitrage.pricing_model.interface import StrategyPnl


def plot_pnl_of_path(pnl: StrategyPnl, path: int) -> None:
    fig, ax = plt.subplots(4, 2, figsize=(10, 10))
    length = pnl["total_pnl"].shape[0]
    for i, key in enumerate(pnl.keys()):
        ax[i // 2, i % 2].set_title(key)
        ax[i // 2, i % 2].plot(pnl[key][:, path].cumsum())
        ax[i // 2, i % 2].hlines(0, xmin=0, xmax=length, color="black")
    ax[3, 0].set_title("gamma_pnl + theta_pnl")
    ax[3, 0].plot((pnl["gamma_pnl"][:, path] + pnl["theta_pnl"][:, path]).cumsum())
    ax[3, 0].hlines(0, xmin=0, xmax=length, color="black")
    ax[3, 1].set_title("var_vega_pnl + vega_hedge_pnl")
    ax[3, 1].plot((pnl["var_vega_pnl"][:, path] + pnl["vega_hedge_pnl"][:, path]).cumsum())
    ax[3, 1].hlines(0, xmin=0, xmax=length, color="black")
    fig.suptitle("P&L Decomposition")
    fig.tight_layout()
    plt.show()


def plot_vega_hedge_performance_of_path(pnl: StrategyPnl, path: int) -> None:
    var_vega_pnl = pnl["var_vega_pnl"][:, path]
    vega_hedge_pnl = pnl["vega_hedge_pnl"][:, path]

    plt.scatter(vega_hedge_pnl, var_vega_pnl)
    plt.xlabel("Vega Hedge P&L")
    plt.ylabel("Variance Vega P&L")

    unhedged_pnl_var = (var_vega_pnl**2).mean()
    hedged_pnl_var = ((var_vega_pnl + vega_hedge_pnl) ** 2).mean()
    hedged_ratio = 1 - hedged_pnl_var / unhedged_pnl_var
    plt.hlines(
        y=0,
        xmin=vega_hedge_pnl.min(),
        xmax=vega_hedge_pnl.max(),
        color="black",
    )
    plt.vlines(
        x=0,
        ymin=var_vega_pnl.min(),
        ymax=var_vega_pnl.max(),
        color="black",
    )
    plt.title(f"{100*hedged_ratio:.2f}% P&L hedged")
    plt.tight_layout()
    plt.show()


def plot_pnl_distribution(pnl: StrategyPnl) -> None:
    fig, ax = plt.subplots(4, 2, figsize=(10, 10))

    for i, key in enumerate(pnl.keys()):
        ax[i // 2, i % 2].set_title(key)
        ax[i // 2, i % 2].hist(pnl[key].sum(axis=0), density=True, bins=100)
    ax[3, 0].set_title("gamma_pnl + theta_pnl")
    ax[3, 0].hist((pnl["gamma_pnl"] + pnl["theta_pnl"]).sum(axis=0), density=True, bins=100)

    ax[3, 1].set_title("var_vega_pnl + vega_hedge_pnl")
    ax[3, 1].hist((pnl["var_vega_pnl"] + pnl["vega_hedge_pnl"]).sum(axis=0), density=True, bins=100)
    fig.suptitle("P&L Decomposition")
    fig.tight_layout()
    plt.show()
