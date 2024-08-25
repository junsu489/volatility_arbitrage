"""Heston model"""

# pylint: disable=line-too-long,too-many-arguments,too-many-locals

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from volatility_arbitrage.pricing_model.interface import HestonParams, MarketModel

ARRAY = npt.NDArray[np.float64]


def generate_initial_var(model_params: HestonParams, size: Union[int, Tuple[int, ...]]) -> ARRAY:
    """
    :param model_params: Heston parameters
    :param size: size
    :return: samples that follow the asymptotice distribution of CIR process
    """
    alpha = 2 * model_params.kappa * model_params.mean_of_var / model_params.vol_of_var**2
    beta = 2 * model_params.kappa / model_params.vol_of_var**2
    return np.random.gamma(shape=alpha, scale=1 / beta, size=size)


def generate_cir_processs(
    var_0: float,
    model_params: HestonParams,
    normal_var: ARRAY,
    num_path: int,
    length: int,
    time_delta: float,
) -> ARRAY:
    """
    Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513-536.

    :param var_0: initial variance
    :param model_params: Heston paremters
    :param normal_var: i.i.d. standard normal random
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: simulated CIR process
    """
    # initial_var = generate_initial_var(model_params=model_params, size=1)

    var = np.zeros(shape=(length + 1, num_path))
    var[0] = var_0

    drift = model_params.kappa * model_params.mean_of_var * time_delta
    mean_reversion_adj = 1 / (1 + model_params.kappa * time_delta)
    milstein_adj = 0.25 * model_params.vol_of_var**2 * (normal_var**2 - 1) * time_delta
    for i in range(length):
        diffusion = model_params.vol_of_var * np.sqrt(var[i]) * normal_var[i] * np.sqrt(time_delta)
        var[i + 1] = (
            np.maximum(var[i] + drift + diffusion + milstein_adj[i], 0) * mean_reversion_adj
        )

    return var


def generate_heston_processes(
    var_0: float,
    model_params: HestonParams,
    normal_var_1: ARRAY,
    normal_var_2: ARRAY,
    num_path: int,
    length: int,
    time_delta: float,
) -> Tuple[ARRAY, ARRAY]:
    """
    Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513-536.

    :param var_0: initial variance
    :param model_params: Heston paremters
    :param normal_var_1: i.i.d. standard normal random vector
    :param normal_var_2: i.i.d. standard normal random vector
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: log return and instantaneous variance
    """
    var = generate_cir_processs(var_0, model_params, normal_var_1, num_path, length, time_delta)
    vol = np.sqrt(var)
    lr = np.zeros((length + 1, num_path))

    avg_var = 0.5 * (var[:-1] + var[1:])
    avg_vol = 0.5 * (vol[:-1] + vol[1:])

    drift = -0.5 * avg_var * time_delta
    corr_diffusion = model_params.rho * vol[:-1] * normal_var_1 * np.sqrt(time_delta)
    uncorr_diffusion = (
        np.sqrt(1 - model_params.rho**2) * avg_vol * normal_var_2 * np.sqrt(time_delta)
    )
    milstein_correction = (
        0.5 * model_params.rho * model_params.vol_of_var * (normal_var_2**2 - 1) * time_delta
    )
    lr[1:] = drift + corr_diffusion + uncorr_diffusion + milstein_correction

    # First moment. Future price process should be martingale
    f = np.exp(lr.cumsum(axis=0))
    log_expectation = np.log(np.mean(f, axis=1))
    lr[1:] -= np.diff(log_expectation)[:, np.newaxis]
    return lr, var


def generate_inefficient_market(
    *,
    real_var_0: float,
    imp_var_0: float,
    market_model: MarketModel,
    num_path: int,
    length: int,
    time_delta: float,
) -> Tuple[ARRAY, ARRAY, ARRAY]:
    """
    :param real_var_0: initial realized variance
    :param imp_var_0: initial implied variance
    :param market_model: market_model
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: log return, realized instantaneous variance, implied instantaneous variance
    """
    normal_var = np.random.normal(size=(3, length, num_path))
    lr, real_var = generate_heston_processes(
        var_0=real_var_0,
        model_params=market_model.real_model,
        normal_var_1=normal_var[0],
        normal_var_2=normal_var[1],
        num_path=num_path,
        length=length,
        time_delta=time_delta,
    )
    corr_matrix = np.array(
        [
            [1.0, market_model.real_model.rho, market_model.rho_real_var_imp_var],
            [market_model.real_model.rho, 1.0, market_model.rho_spot_imp_var],
            [market_model.rho_real_var_imp_var, market_model.rho_spot_imp_var, 1.0],
        ]
    )

    try:
        cholesky = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The correlation matrix is not positive definite.") from exc

    correlated_normal = (
        cholesky[2][0] * normal_var[0]
        + cholesky[2][1] * normal_var[1]
        + cholesky[2][2] * normal_var[2]
    )
    imp_var = generate_cir_processs(
        imp_var_0, market_model.imp_model, correlated_normal, num_path, length, time_delta
    )
    return lr, real_var, imp_var


def predict_var(var: ARRAY, time_delta: ARRAY, model_params: HestonParams):
    """
    :param var: instantaneous variance
    :param time_delta: time delta in years
    :param model_params: Heston Parameters
    :return: expected instantaneous variance after time_delta
    """
    return model_params.mean_of_var + np.exp(-model_params.kappa * time_delta) * (
        var - model_params.mean_of_var
    )
