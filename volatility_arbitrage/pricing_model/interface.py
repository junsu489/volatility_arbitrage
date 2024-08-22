"""interface of pricing model"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

ARRAY = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HestonParams:
    """Heston stochastic volatilty model parameters"""

    kappa: float
    mean_of_var: float
    vol_of_var: float
    rho: float

    def __post_init__(self) -> None:
        assert self.mean_of_var > 0
        assert self.kappa > 0
        assert self.vol_of_var > 0
        assert abs(self.rho) < 1
        # Check feller condition
        assert 2 * self.kappa * self.mean_of_var > self.vol_of_var**2


@dataclass(frozen=True)
class MarketModel:
    """Market model"""

    imp_model: HestonParams
    real_model: HestonParams
    rho_spot_imp_var: float
    rho_real_var_imp_var: float

    def __post_init__(self) -> None:
        assert abs(self.rho_spot_imp_var) < 1
        assert abs(self.rho_real_var_imp_var) < 1

    def cholesky(self) -> ARRAY:
        """Cholesky decomposition"""
        # Construct the correlation matrix
        corr_matrix = np.array(
            [
                [1.0, self.real_model.rho, self.rho_real_var_imp_var],
                [self.real_model.rho, 1.0, self.rho_spot_imp_var],
                [self.rho_real_var_imp_var, self.rho_spot_imp_var, 1.0],
            ]
        )

        try:
            return np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The correlation matrix is not positive definite.") from exc


class StrategyPnl(TypedDict):
    """strategy pnl dictionary"""

    total_pnl: ARRAY
    var_vega_pnl: ARRAY
    theta_pnl: ARRAY
    vanna_pnl: ARRAY
    gamma_pnl: ARRAY
    vega_hedge_pnl: ARRAY


@dataclass(frozen=True)
class StrategyPnlCalculator:
    """strategy pnl calculator"""

    total_pnl: ARRAY
    var_vega_pnl: ARRAY
    theta_pnl: ARRAY
    vanna_pnl: ARRAY
    gamma_pnl: ARRAY
    vega_hedge_pnl: ARRAY

    def get_strategy_pnl(self, position: npt.NDArray[np.float64]) -> StrategyPnl:
        """calculate pnl of a strategy"""
        return StrategyPnl(
            total_pnl=self.total_pnl * position,
            var_vega_pnl=self.var_vega_pnl * position,
            theta_pnl=self.theta_pnl * position,
            vanna_pnl=self.vanna_pnl * position,
            gamma_pnl=self.gamma_pnl * position,
            vega_hedge_pnl=self.vega_hedge_pnl * position,
        )
