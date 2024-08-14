"""Heston model"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=Union[npt.NDArray[np.float64], float])


@dataclass
class HestonModel:
    """Heston stochastic volatilty parameters"""

    kappa: float
    mean_of_var: float
    vol_of_var: float

    def __post_init__(self) -> None:
        assert self.mean_of_var > 0
        # Check feller condition
        assert 2 * self.kappa * self.mean_of_var > self.vol_of_var**2


@dataclass
class Correlation:
    """Correlation among spot price, implied variance, and realized variance processes"""

    rho_spot_imp: float
    rho_spot_real: float
    rho_real_imp: float

    cholesky: npt.NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if not (-1 < value < 1):
                raise ValueError(f"Value of '{field_name}' must be between -1 and 1. Got {value} instead.")

        # Construct the correlation matrix
        corr_matrix = np.array(
            [
                [1.0, self.rho_spot_imp, self.rho_spot_real],
                [self.rho_spot_imp, 1.0, self.rho_real_imp],
                [self.rho_spot_real, self.rho_real_imp, 1.0],
            ]
        )

        try:
            self.cholesky = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("The correlation matrix is not positive definite.")

    def simulate_multivariate_normal(size: tuple[int, int]) -> npt.NDArray[np.float64]:
        """
        Simulate multivariate normal path
        """
        normal = np.random.normal(size=(3,) + size)
        return self.cholesky @ normal


class WeightedVarianceSwap(ABC):
    def __init__(self, imp_var_param: HestonModel, real_var_param: HestonModel, corr: Correlation) -> None:
        self.imp_var_param = imp_var_param
        self.real_var_param = real_var_param
        self.corr = corr

    @abstractmethod
    def price(self, *, imp_var: T, tau: float) -> T:
        """
        Return price of weighted variance swap

        :param imp_var: instantaneous implied variance
        :param tau: time to expiry in years
        """

    @abstractmethod
    def var_vega(self, tau: float) -> float:
        """
        Return variance vega

        :param tau: time to expiry in years
        """

    def var_skew_stikiness_ratio(self, *, real_var: T, imp_var: T) -> T:
        """
        SSR with respect to implied instantaneous variance = d imp_var d log(F) / (d log(F))^2

        :param real_var: instantaneous realized variance
        :param imp_var: instantanoues implied variance
        """
        return self.corr.rho_spot_imp * self.imp_var_param.vol_of_var * np.sqrt(imp_var) / np.sqrt(real_var)

    def min_var_delta(self, *, real_var: T, imp_var: T, tau: float) -> T:
        """
        Return minimum variance delta

        :param real_var: instantaneous realized variance
        :param imp_var: instantanoues implied variance
        :param tau: time to expiry in years
        """
        return self.var_vega(tau) * self.var_skew_stikiness_ratio(real_var=real_var, imp_var=imp_var)

    @abstractmethod
    def total_pnl(
        self,
        *,
        f_0: T,
        f_t: T,
        imp_var_0: T,
        tau_0: float,
        imp_var_t: T,
        exp_imp_var_t: T,
        tau_t: float,
    ) -> T:
        """
        Return Total P&L

        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param tau_t: time to expiry in years at time t
        """

    @abstractmethod
    @staticmethod
    def gamma_pnl(*, f_0: T, f_t: T) -> T:
        """
        Return Gamma P&L

        :param f_0: forward price at time 0
        :param f_t: fowward price at time t
        """

    def theta_pnl(self, *, imp_var_0: T, tau_0: float, exp_imp_var_t: T, tau_t: float) -> T:
        """
        Return expected Theta P&L at time 0

        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        assert 0 <= tau_t <= tau_0
        return self.theta_pnl_from_initial_price(
            price_0=self.price(imp_var=imp_var_0, tau=tau_0),
            exp_imp_var_t=exp_imp_var_t,
            tau_t=tau_t,
        )

    def theta_pnl_from_initial_price(self, *, price_0: T, exp_imp_var_t: T, tau_t: float) -> T:
        """
        Return expected Theta P&L at time 0

        :param price_0: price at time 0
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        assert 0 <= tau_t
        return self.price(imp_var=exp_imp_var_t, tau=tau_t) - price_0

    def var_vega_pnl(
        self,
        *,
        imp_var_0: T,
        tau_0: float,
        imp_var_t: T,
        exp_imp_var_t: T,
        tau_t: float,
    ) -> T:
        """
        Return variance Vega P&L

        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        return self.var_vega_from_initial_price(
            price_0=self.price(imp_var=imp_var_0, tau=tau_0),
            imp_var_t=imp_var_t,
            exp_imp_var_t=exp_imp_var_t,
            tau_t=tau_t,
        )

    def var_vega_from_initial_price(self, price_0: T, imp_var_t: T, exp_imp_var_t: T, tau_t: float) -> T:
        """
        Return variance Vega P&L

        :param price_0: price at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        price_t = self.price(imp_var=imp_var_t, tau=tau_t)
        return (
            price_t
            - price_0
            - self.theta_pnl_from_initial_price(price_0=price_0, exp_imp_var_t=exp_imp_var_t, tau_t=tau_t)
        )

    def vega_hedge_pnl(self, *, f_0: T, f_t: T, real_var_0: T, imp_var_0: T, tau_0: float) -> T:
        """
        Return Vega hedge P&L

        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param real_var_0: instantaneous realized variance at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        """
        return -self.min_var_delta(real_var=real_var_0, imp_var=imp_var_0, tau=tau_0) * (f_t - f_0)


class VarianceSwap(WeightedVarianceSwap):
    def price(self, *, imp_var: T, tau: float) -> T:
        return (
            self.imp_var_param.mean_of_var * tau
            + (imp_var - self.imp_var_param.mean_of_var)
            * (1 - np.exp(-self.imp_var_param.kappa * tau))
            / self.imp_var_param.kappa
        )

    def var_vega(self, tau: float) -> float:
        assert tau > 0
        return (1 - np.exp(-self.imp_var_param.kappa * tau)) / self.imp_var_param.kappa

    def total_pnl(
        self,
        *,
        f_0: T,
        f_t: T,
        real_var_0: T,
        imp_var_0: T,
        tau_0: float,
        imp_var_t: T,
        tau_t: float,
    ) -> T:
        price_0 = self.price(imp_var=imp_var_0, tau=tau_0)
        price_t = self.price(imp_var=imp_var_t, tau=tau_t)
        gamma_pnl = self.gamma_pnl(f_0=f_0, f_t=f_t)
        vega_hedge_pnl = self.vega_hedge_pnl(f_0=f_0, f_t=f_t, real_var_0=real_var_0, imp_var_0=imp_var_0, tau_0=tau_0)
        return price_t - price_0 + gamma_pnl + vega_hedge_pnl

    @staticmethod
    def gamma_pnl(*, f_0: T, f_t: T) -> T:
        return 2 * (f_t / f_0 - 1 - np.log(f_t / f_0))
