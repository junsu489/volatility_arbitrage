"""Weighted variance swaps"""

# pylint: disable=line-too-long, too-many-arguments,too-many-locals

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from volatility_arbitrage.pricing_model.interface import (
    HestonParams,
    MarketModel,
    StrategyPnlCalculator,
)

ARRAY = npt.NDArray[np.float64]


class WeightedVarianceSwap(ABC):
    """
    Base class for weighted variance swap.
    Lee, R. (2010). Weighted variance swap. Encyclopedia of quantitative finance.
    """

    def __init__(
        self,
        market_model: MarketModel,
    ) -> None:
        self.market_model = market_model

    @staticmethod
    def price_var_swap(var: ARRAY, tau: ARRAY, mean_of_var: float, kappa: float) -> ARRAY:
        """
        :param var: instantaneous variance
        :param tau: time to expiry in years
        :param mean_of_var: mean of instantenous variance process
        :param kappa: mean reversion parameter
        :return: price of standard variance swap
        """
        return mean_of_var * tau + (var - mean_of_var) * (1 - np.exp(-kappa * tau)) / kappa

    @staticmethod
    def var_swap_var_vega(tau: ARRAY, kappa: float):
        """
        :param tau: time to expiry in years
        :param kappa: mena reversion paramter
        :return: variance vega of varince swap
        """
        return (1 - np.exp(-kappa * tau)) / kappa

    @abstractmethod
    def price(self, *, imp_var: ARRAY, tau: ARRAY) -> ARRAY:
        """
        :param imp_var: instantaneous implied variance
        :param tau: time to expiry in years
        :return: price of weighted variance swap
        """

    @abstractmethod
    def var_vega(self, tau: ARRAY) -> ARRAY:
        """
        :param tau: time to expiry in years
        :return: variance vega
        """

    def forward_var_vega(self, tau_front: ARRAY, tau_back: ARRAY) -> ARRAY:
        """
        :param tau_front: time to expiry in years
        :param tau_back: time to expiry in years
        :return: forward variance vega
        """
        return self.var_vega(tau_back) - self.var_vega(tau_front)

    def var_skew_stikiness_ratio(self, *, real_var: ARRAY, imp_var: ARRAY) -> ARRAY:
        """
        :param real_var: instantaneous realized variance
        :param imp_var: instantanoues implied variance
        :return: SSR with respect to implied instantaneous variance = d imp_var d log(F) / (d log(F))^2
        """
        return (
            self.market_model.rho_spot_imp_var
            * self.market_model.imp_model.vol_of_var
            * np.sqrt(imp_var)
            / np.sqrt(real_var)
        )

    def min_var_delta(
        self, *, real_var_0: ARRAY, imp_var_0: ARRAY, f_0: ARRAY, tau_0: ARRAY, tau_t: ARRAY
    ) -> ARRAY:
        """
        :param real_var_0: instantaneous realized variance at time 0
        :param imp_var_0: instantanoues implied variance at time 0
        :param f_0: forward price at time 0
        :param tau_0: time to expiry in years at time 0
        :param tau_t: time to expiry in years at time t
        :return: minimum variance delta
        """
        # forward_var_vega is used because at the next timestamp variance between time 0 and 1 is not a risk.
        forward_var_vega = self.forward_var_vega(tau_front=tau_0 - tau_t, tau_back=tau_0)
        ssr = self.var_skew_stikiness_ratio(real_var=real_var_0, imp_var=imp_var_0)
        return forward_var_vega * ssr / f_0

    def get_pnl_calculator(
        self,
        *,
        f_0: ARRAY,
        f_t: ARRAY,
        real_var_0: ARRAY,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        imp_var_t: ARRAY,
        tau_t: ARRAY,
    ) -> StrategyPnlCalculator:
        """
        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param real_var_0: instantaneous realize variacnce at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param tau_t: time to expiry in years at time t
        :return: StrategyPnlCalculator
        """
        var_vega_pnl = self.var_vega_pnl(
            imp_var_0=imp_var_0, tau_0=tau_0, imp_var_t=imp_var_t, tau_t=tau_t
        )
        theta_pnl = self.theta_pnl(imp_var_0=imp_var_0, tau_0=tau_0, tau_t=tau_t)
        vanna_pnl = self.vanna_pnl(
            imp_var_0=imp_var_0, tau_0=tau_0, imp_var_t=imp_var_t, tau_t=tau_t, f_0=f_0, f_t=f_t
        )
        gamma_pnl = self.gamma_pnl(f_0=f_0, f_t=f_t)
        vega_hedge_pnl = self.vega_hedge_pnl(
            f_0=f_0, f_t=f_t, real_var_0=real_var_0, imp_var_0=imp_var_0, tau_0=tau_0, tau_t=tau_t
        )
        total_pnl = var_vega_pnl + theta_pnl + vanna_pnl + gamma_pnl + vega_hedge_pnl
        return StrategyPnlCalculator(
            total_pnl=total_pnl,
            var_vega_pnl=var_vega_pnl,
            theta_pnl=theta_pnl,
            vanna_pnl=vanna_pnl,
            gamma_pnl=gamma_pnl,
            vega_hedge_pnl=vega_hedge_pnl,
        )

    @staticmethod
    @abstractmethod
    def gamma_pnl(*, f_0: ARRAY, f_t: ARRAY) -> ARRAY:
        """
        :param f_0: forward price at time 0
        :param f_t: fowward price at time t
        :return: Gamma P&L
        """

    def theta_pnl(
        self,
        *,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        tau_t: ARRAY,
    ) -> ARRAY:
        """
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param tau_t: time to expiry in years at time t
        :return: expected Theta P&L at time 0
        """
        return -self.price(imp_var=imp_var_0, tau=tau_0 - tau_t)

    def var_vega_pnl(
        self,
        *,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        imp_var_t: ARRAY,
        tau_t: ARRAY,
    ) -> ARRAY:
        """
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param tau_t: time to expiry in years at time t
        :return: variance Vega P&L
        """
        price_0 = self.price(imp_var=imp_var_0, tau=tau_0)
        price_t = self.price(imp_var=imp_var_t, tau=tau_t)
        theta_pnl = self.theta_pnl(imp_var_0=imp_var_0, tau_0=tau_0, tau_t=tau_t)
        return price_t - price_0 - theta_pnl

    @abstractmethod
    def vanna_pnl(
        self,
        *,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        imp_var_t: ARRAY,
        tau_t: ARRAY,
        f_0: ARRAY,
        f_t: ARRAY,
    ) -> ARRAY:
        """
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param tau_t: time to expiry in years at time t
        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :return: Vanna P&L
        """

    def vega_hedge_pnl(
        self,
        *,
        f_0: ARRAY,
        f_t: ARRAY,
        real_var_0: ARRAY,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        tau_t: ARRAY,
    ) -> ARRAY:
        """
        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param real_var_0: instantaneous realized variance at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param tau_t: time to expiry in years at time t
        :return: Vega hedge P&L
        """
        min_var_delta = self.min_var_delta(
            real_var_0=real_var_0, imp_var_0=imp_var_0, f_0=f_0, tau_0=tau_0, tau_t=tau_t
        )
        return -min_var_delta * (f_t - f_0)


class VarianceSwap(WeightedVarianceSwap):
    """
    Standard variance swap or log contract.
    Neuberger, A. (1994). The log contract. Journal of portfolio management, 20(2), 74.
    Fukasawa, M. (2014). Volatility derivatives and model-free implied leverage. International Journal of Theoretical and Applied Finance, 17(01), 1450002.
    """

    def price(self, *, imp_var: ARRAY, tau: ARRAY) -> ARRAY:
        return self.price_var_swap(
            var=imp_var,
            tau=tau,
            mean_of_var=self.market_model.imp_model.mean_of_var,
            kappa=self.market_model.imp_model.kappa,
        )

    def var_vega(self, tau: ARRAY) -> ARRAY:
        return self.var_swap_var_vega(tau=tau, kappa=self.market_model.imp_model.kappa)

    def vanna_pnl(
        self,
        *,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        imp_var_t: ARRAY,
        tau_t: ARRAY,
        f_0: ARRAY,
        f_t: ARRAY,
    ) -> ARRAY:
        return np.zeros_like(imp_var_0)

    @staticmethod
    def gamma_pnl(*, f_0: ARRAY, f_t: ARRAY) -> ARRAY:
        return 2 * (f_t / f_0 - 1 - np.log(f_t / f_0))


class GammaSwap(WeightedVarianceSwap):
    """
    Standard gamma swap or entropy contract.
    Fukasawa, M. (2014). Volatility derivatives and model-free implied leverage. International Journal of Theoretical and Applied Finance, 17(01), 1450002.

    """

    @staticmethod
    def _get_adjusted_kappa(model_params: HestonParams) -> float:
        return model_params.kappa - model_params.vol_of_var * model_params.rho

    @classmethod
    def _get_adjusted_mean_of_var(cls, model_params: HestonParams) -> float:
        kappa_adj = cls._get_adjusted_kappa(model_params)
        return model_params.kappa / kappa_adj * model_params.mean_of_var

    def price(self, *, imp_var: ARRAY, tau: ARRAY) -> ARRAY:
        kappa_adj = self._get_adjusted_kappa(self.market_model.imp_model)
        mean_of_var_adj = self._get_adjusted_mean_of_var(self.market_model.imp_model)
        return self.price_var_swap(
            var=imp_var, tau=tau, mean_of_var=mean_of_var_adj, kappa=kappa_adj
        )

    def var_vega(self, tau: ARRAY) -> ARRAY:
        kappa_adj = self._get_adjusted_kappa(self.market_model.imp_model)
        return self.var_swap_var_vega(tau=tau, kappa=kappa_adj)

    def vanna_pnl(
        self,
        *,
        imp_var_0: ARRAY,
        tau_0: ARRAY,
        imp_var_t: ARRAY,
        tau_t: ARRAY,
        f_0: ARRAY,
        f_t: ARRAY,
    ) -> ARRAY:
        return (
            (f_t - f_0)
            / f_0
            * (self.price(imp_var=imp_var_t, tau=tau_t) - self.price(imp_var=imp_var_0, tau=tau_0))
        )

    @staticmethod
    def gamma_pnl(*, f_0: ARRAY, f_t: ARRAY) -> ARRAY:
        return 2 * (f_t / f_0 * (np.log(f_t / f_0) - 1) + 1)
