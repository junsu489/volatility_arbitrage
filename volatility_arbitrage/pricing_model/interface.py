from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import numpy.typing as npt

ARRAY = npt.NDArray[np.float64]


@dataclass
class HestonParams:
    """Heston stochastic volatilty parameters"""

    kappa: float
    mean_of_var: float
    vol_of_var: float

    def __post_init__(self) -> None:
        assert self.mean_of_var > 0
        assert self.kappa > 0
        assert self.vol_of_var > 0
        # Check feller condition
        assert 2 * self.kappa * self.mean_of_var > self.vol_of_var**2


@dataclass
class Correlation:
    """Correlation among spot price, implied variance, and realized variance processes"""

    rho_spot_imp: float
    rho_spot_real: float
    rho_real_imp: float

    cholesky: ARRAY = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if 1 <= abs(value):
                raise ValueError(
                    f"Value of '{field_name}' must be between -1 and 1. Got {value} instead."
                )

        # Construct the correlation matrix
        corr_matrix = np.array(
            [
                [1.0, self.rho_spot_real, self.rho_real_imp],
                [self.rho_spot_real, 1.0, self.rho_spot_imp],
                [self.rho_real_imp, self.rho_spot_imp, 1.0],
            ]
        )

        try:
            self.cholesky = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The correlation matrix is not positive definite.") from exc


class StrategyPnl(TypedDict):
    total_pnl: ARRAY
    var_vega_pnl: ARRAY
    theta_pnl: ARRAY
    vanna_pnl: ARRAY
    gamma_pnl: ARRAY
    vega_hedge_pnl: ARRAY


class StrategyPnlCalculator:
    _total_pnl: ARRAY
    _var_vega_pnl: ARRAY
    _theta_pnl: ARRAY
    _vanna_pnl: ARRAY
    _gamma_pnl: ARRAY
    _vega_hedge_pnl: ARRAY

    def __init__(
        self, total_pnl, var_vega_pnl, theta_pnl, vanna_pnl, gamma_pnl, vega_hedge_pnl
    ) -> None:
        self._total_pnl = total_pnl
        self._var_vega_pnl = var_vega_pnl
        self._theta_pnl = theta_pnl
        self._vanna_pnl = vanna_pnl
        self._gamma_pnl = gamma_pnl
        self._vega_hedg_pnl = vega_hedge_pnl

    def get_strategy_pnl(self, position: npt.NDArray[np.float64]) -> StrategyPnl:
        return StrategyPnl(
            total_pnl=self._total_pnl * position,
            var_vega_pnl=self._var_vega_pnl * position,
            theta_pnl=self._theta_pnl * position,
            vanna_pnl=self._vanna_pnl * position,
            gamma_pnl=self._gamma_pnl * position,
            vega_hedge_pnl=self._vega_hedge_pnl * position,
        )
