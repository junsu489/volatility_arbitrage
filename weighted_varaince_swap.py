"""Weighted variance swaps"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from heston_model import Correlation, HestonParams

NP_ARRAY = npt.NDArray[np.float64]


class WeightedVarianceSwap(ABC):
    """
    Base class for weighted variance swap.
    Lee, R. (2010). Weighted variance swap. Encyclopedia of quantitative finance.
    """

    def __init__(
        self,
        imp_var_param: HestonParams,
        real_var_param: HestonParams,
        corr: Correlation,
    ) -> None:
        self.imp_var_param = imp_var_param
        self.real_var_param = real_var_param
        self.corr = corr

    @abstractmethod
    def price(self, *, imp_var: NP_ARRAY, tau: NP_ARRAY) -> NP_ARRAY:
        """
        Return price of weighted variance swap

        :param imp_var: instantaneous implied variance
        :param tau: time to expiry in years
        """

    @abstractmethod
    def var_vega(self, tau: NP_ARRAY) -> NP_ARRAY:
        """
        Return variance vega

        :param tau: time to expiry in years
        """

    def var_skew_stikiness_ratio(
        self, *, real_var: NP_ARRAY, imp_var: NP_ARRAY
    ) -> NP_ARRAY:
        """
        SSR with respect to implied instantaneous variance = d imp_var d log(F) / (d log(F))^2

        :param real_var: instantaneous realized variance
        :param imp_var: instantanoues implied variance
        """
        return (
            self.corr.rho_spot_imp
            * self.imp_var_param.vol_of_var
            * np.sqrt(imp_var)
            / np.sqrt(real_var)
        )

    def min_var_delta(
        self, *, real_var: NP_ARRAY, imp_var: NP_ARRAY, tau: NP_ARRAY
    ) -> NP_ARRAY:
        """
        Return minimum variance delta

        :param real_var: instantaneous realized variance
        :param imp_var: instantanoues implied variance
        :param tau: time to expiry in years
        """
        return self.var_vega(tau) * self.var_skew_stikiness_ratio(
            real_var=real_var, imp_var=imp_var
        )

    @abstractmethod
    def total_pnl(
        self,
        *,
        f_0: NP_ARRAY,
        f_t: NP_ARRAY,
        real_var_0: NP_ARRAY,
        imp_var_0: NP_ARRAY,
        tau_0: NP_ARRAY,
        imp_var_t: NP_ARRAY,
        tau_t: NP_ARRAY,
    ) -> NP_ARRAY:
        """
        Return Total P&L

        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param real_var_0: instantaneous realize variacnce at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param imp_var_t: instantaneous implied variance at time t
        :param tau_t: time to expiry in years at time t
        """

    @abstractmethod
    @staticmethod
    def gamma_pnl(*, f_0: NP_ARRAY, f_t: NP_ARRAY) -> NP_ARRAY:
        """
        Return Gamma P&L

        :param f_0: forward price at time 0
        :param f_t: fowward price at time t
        """

    def theta_pnl(
        self,
        *,
        imp_var_0: NP_ARRAY,
        tau_0: NP_ARRAY,
        exp_imp_var_t: NP_ARRAY,
        tau_t: NP_ARRAY,
    ) -> NP_ARRAY:
        """
        Return expected Theta P&L at time 0

        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        return self.theta_pnl_from_initial_price(
            price_0=self.price(imp_var=imp_var_0, tau=tau_0),
            exp_imp_var_t=exp_imp_var_t,
            tau_t=tau_t,
        )

    def theta_pnl_from_initial_price(
        self, *, price_0: NP_ARRAY, exp_imp_var_t: NP_ARRAY, tau_t: NP_ARRAY
    ) -> NP_ARRAY:
        """
        Return expected Theta P&L at time 0

        :param price_0: price at time 0
        :param exp_imp_var_t: E[imp_var_t|imp_var_0]
        :param tau_t: time to expiry in years at time t
        """
        return self.price(imp_var=exp_imp_var_t, tau=tau_t) - price_0

    def var_vega_pnl(
        self,
        *,
        imp_var_0: NP_ARRAY,
        tau_0: NP_ARRAY,
        imp_var_t: NP_ARRAY,
        exp_imp_var_t: NP_ARRAY,
        tau_t: NP_ARRAY,
    ) -> NP_ARRAY:
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

    def var_vega_from_initial_price(
        self,
        price_0: NP_ARRAY,
        imp_var_t: NP_ARRAY,
        exp_imp_var_t: NP_ARRAY,
        tau_t: NP_ARRAY,
    ) -> NP_ARRAY:
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
            - self.theta_pnl_from_initial_price(
                price_0=price_0, exp_imp_var_t=exp_imp_var_t, tau_t=tau_t
            )
        )

    def vega_hedge_pnl(
        self,
        *,
        f_0: NP_ARRAY,
        f_t: NP_ARRAY,
        real_var_0: NP_ARRAY,
        imp_var_0: NP_ARRAY,
        tau_0: NP_ARRAY,
    ) -> NP_ARRAY:
        """
        Return Vega hedge P&L

        :param f_0: forward price at time 0
        :param f_t: forward price at time t
        :param real_var_0: instantaneous realized variance at time 0
        :param imp_var_0: instantaneous implied variance at time 0
        :param tau_0: time to expiry in years at time 0
        """
        return -self.min_var_delta(
            real_var=real_var_0, imp_var=imp_var_0, tau=tau_0
        ) * (f_t - f_0)


class VarianceSwap(WeightedVarianceSwap):
    """
    Standard variance swap or log contract.
    Neuberger, A. (1994). The log contract. Journal of portfolio management, 20(2), 74.
    Fukasawa, M. (2014). Volatility derivatives and model-free implied leverage. International Journal of Theoretical and Applied Finance, 17(01), 1450002.
    """

    def price(self, *, imp_var: NP_ARRAY, tau: NP_ARRAY) -> NP_ARRAY:
        return (
            self.imp_var_param.mean_of_var * tau
            + (imp_var - self.imp_var_param.mean_of_var)
            * (1 - np.exp(-self.imp_var_param.kappa * tau))
            / self.imp_var_param.kappa
        )

    def var_vega(self, tau: NP_ARRAY) -> NP_ARRAY:
        return (1 - np.exp(-self.imp_var_param.kappa * tau)) / self.imp_var_param.kappa

    def total_pnl(
        self,
        *,
        f_0: NP_ARRAY,
        f_t: NP_ARRAY,
        real_var_0: NP_ARRAY,
        imp_var_0: NP_ARRAY,
        tau_0: NP_ARRAY,
        imp_var_t: NP_ARRAY,
        tau_t: NP_ARRAY,
    ) -> NP_ARRAY:
        price_0 = self.price(imp_var=imp_var_0, tau=tau_0)
        price_t = self.price(imp_var=imp_var_t, tau=tau_t)
        gamma_pnl = self.gamma_pnl(f_0=f_0, f_t=f_t)
        vega_hedge_pnl = self.vega_hedge_pnl(
            f_0=f_0, f_t=f_t, real_var_0=real_var_0, imp_var_0=imp_var_0, tau_0=tau_0
        )
        return price_t - price_0 + gamma_pnl + vega_hedge_pnl

    @staticmethod
    def gamma_pnl(*, f_0: NP_ARRAY, f_t: NP_ARRAY) -> NP_ARRAY:
        return 2 * (f_t / f_0 - 1 - np.log(f_t / f_0))
