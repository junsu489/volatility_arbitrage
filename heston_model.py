"""Heston model"""

# pylint: disable=line-too-long,too-many-arguments,too-many-locals

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt

NP_ARRAY = npt.NDArray[np.float64]


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

    cholesky: NP_ARRAY = field(init=False, repr=False)

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
            raise ValueError(
                "The correlation matrix is not positive definite."
            ) from exc


def generate_initial_var(
    var_params: HestonParams, size: Union[int, tuple[int, ...]]
) -> NP_ARRAY:
    """
    :param var_params: Heston parameters
    :param size: size
    :return: samples that follow the asymptotice distribution of CIR process
    """
    alpha = 2 * var_params.kappa * var_params.mean_of_var / var_params.vol_of_var**2
    beta = 2 * var_params.kappa / var_params.vol_of_var**2
    return np.random.gamma(shape=alpha, scale=1 / beta, size=size)


def generate_cir_processs(
    var_params: HestonParams,
    normal_var: NP_ARRAY,
    num_path: int,
    length: int,
    time_delta: float,
) -> NP_ARRAY:
    """
    Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513-536.

    :param var_params: Heston paremters
    :param normal_var: i.i.d. standard normal random
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: Simulate CIR process
    """
    initial_var = generate_initial_var(var_params=var_params, size=1)

    var = np.zeros(shape=(length + 1, num_path))
    var[0] = initial_var

    drift = var_params.kappa * var_params.mean_of_var * time_delta
    mean_reversion_adj = 1 / (1 + var_params.kappa * time_delta)
    milstein_adj = 0.25 * var_params.vol_of_var**2 * (normal_var**2 - 1) * time_delta
    for i in range(length):
        diffusion = (
            var_params.vol_of_var
            * np.sqrt(var[i])
            * normal_var[i]
            * np.sqrt(time_delta)
        )
        var[i + 1] = (
            np.maximum(var[i] + drift + diffusion + milstein_adj[i], 0)
            * mean_reversion_adj
        )

    return var


def generate_heston_processes(
    var_params: HestonParams,
    normal_var_1: NP_ARRAY,
    normal_var_2: NP_ARRAY,
    rho: float,
    num_path: int,
    length: int,
    time_delta: float,
) -> tuple[NP_ARRAY, NP_ARRAY]:
    """
    Kahl, C., & Jäckel, P. (2006). Fast strong approximation Monte Carlo schemes for stochastic volatility models. Quantitative Finance, 6(6), 513-536.

    :param var_params: Heston paremters
    :param normal_var_1: i.i.d. standard normal random vector
    :param normal_var_2: i.i.d. standard normal random vector
    :param rho: corr bewteen normal var 1 and normal var 2
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: log return and instantaneous variance
    """
    var = generate_cir_processs(var_params, normal_var_1, num_path, length, time_delta)
    vol = np.sqrt(var)
    lr = np.zeros((length + 1, num_path))

    avg_var = 0.5 * (var[:-1] + var[1:])
    avg_vol = 0.5 * (vol[:-1] + vol[1:])

    drift = -0.5 * avg_var * time_delta
    corr_diffusion = rho * vol[:1] * normal_var_1 * np.sqrt(time_delta)
    uncorr_diffusion = avg_vol * normal_var_2 * np.sqrt(time_delta)
    milstein_correction = (
        0.5 * rho * var_params.vol_of_var * (normal_var_2**2 - 1) * time_delta
    )
    lr[1:] = drift + corr_diffusion + uncorr_diffusion + milstein_correction
    return lr, var


def generate_inefficient_market(
    *,
    real_var_params: HestonParams,
    imp_var_params: HestonParams,
    corr: Correlation,
    num_path: int,
    length: int,
    time_delta: float,
) -> tuple[NP_ARRAY, NP_ARRAY, NP_ARRAY]:
    """
    :param real_var_params: realzied Heston paremters
    :param imp_var_params: implied Heston paremters
    :param corr: correlation matrix
    :param num_path: number of paths
    :param length: length of a path
    :param time_delta: time delta in years
    :return: log return, realized instantaneous variance, implied instantaneous variance
    """
    normal_var = np.random.normal(size=(3, length, num_path))
    lr, real_var = generate_heston_processes(
        var_params=real_var_params,
        normal_var_1=normal_var[0],
        normal_var_2=normal_var[1],
        rho=corr.rho_spot_real,
        num_path=num_path,
        length=length,
        time_delta=time_delta,
    )

    correlated_normal = (
        corr.cholesky[2][0] * normal_var[0]
        + corr.cholesky[2][1] * normal_var[1]
        + corr.cholesky[2][2] * normal_var[2]
    )
    imp_var = generate_cir_processs(
        imp_var_params, correlated_normal, num_path, length, time_delta
    )
    return lr, real_var, imp_var
