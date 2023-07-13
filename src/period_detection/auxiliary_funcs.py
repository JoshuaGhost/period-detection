import functools
import multiprocessing
from typing import List, Tuple, Union

import numpy as np
from pydantic import PositiveFloat, PositiveInt
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import pandas as pd


def calculate_autocorrelation(
    x: List[float],
    i: PositiveInt,
    min_significance_demand: float,
    significant_corr_only: float,
) -> Tuple[float]:
    """
    This function calculates the autocorrelation function for a vector x and a shift given as index i.
    The parameters significant_corr_only and min_significance_demand decide if and which results are considered depending on their significance p.
    :param x: list
    :param i: positive integer
    :param min_significance_demand: float between 0 and 1
    :param significant_corr_only: Boolean or interger in {0,1}
    :return: float between -1 and 1, float between 0 and 1, float between -1 and 1, positive integer
    """
    if i == 0:
        return 1, 0, 1, 0
    else:
        pearson_corr, p_value = stats.pearsonr(x[i:], x[:-i])
        if np.isnan(pearson_corr):
            return 0, 0, 0, i
        if p_value > min_significance_demand and significant_corr_only:
            return pearson_corr, p_value, 0, i
        else:
            return pearson_corr, p_value, pearson_corr, i


def autocor(
    data: List[float],
    list_of_lags: List[int],
    min_significance_demand: float,
    significant_corr_only: Union[bool, int],
) -> Tuple[List[float], List[float], np.ndarray, List[int]]:
    """
    This function calculates the autocorrelation function for a time series given as vector x and the shifts given as indices in list_of_lags.
    The parameters significant_corr_only and min_significance_demand decide if and which results are considered depending on their significance p.
    It returns the correlation coefficients as r_list, their significance as p_list, the autocorrelation function as func and the shift indices as lag_list
    :param x: list
    :param list_of_lags: list of positive integers
    :param min_significance_demand: float between 0 and 1
    :param significant_corr_only: Boolean or interger in {0,1}
    :return: list of floats, list of floats, list of floats, list of positive integers
    """
    with multiprocessing.Pool() as pool:
        res = pool.map(
            functools.partial(
                calculate_autocorrelation,
                data,
                min_significance_demand=min_significance_demand,
                significant_corr_only=significant_corr_only,
            ),
            list_of_lags,
        )
        list_of_results = list(zip(*res))

    pearson_corrs = list(list_of_results[0])
    p_values = list(list_of_results[1])
    signif_pearson_corrs = np.array(list_of_results[2])
    lag_list = list(list_of_results[3])
    return pearson_corrs, p_values, signif_pearson_corrs, lag_list


def auto_l1_norm_on_offset(offset: int, data: List[float]) -> float:
    """
    This function calculates the L^1 norm of the difference of the original autocorrelation function and the shifted version.
    The values of the autocorrelation function results are given in corfunc and the shift index is given as i.
    :param i: positive integer
    :param corfunc: list of floats between -1 and  1
    :return: positive float
    """
    if offset == 0:
        return 0
    else:
        w1 = np.array(data)[offset:]
        w2 = np.array(data)[:-offset]
        return float(sum(abs(w1 - w2)) / w1.size)


def fit_model(data: pd.DataFrame) -> Tuple[List[float], RandomForestRegressor]:
    """
    This function uses the phase of a date in a suggested period as input in order to fit a model.
    The model will later also test how well besaid period fits the original time series.
    Disclaimer: Here the sklearn.ensemble.RandomForestRegressor() is used, but any model can be used instead.
    :param data: pd.DataFrame
    :return: list of 1-D lists of floats, RandomForestRegressor object
    """
    # The routine to fit a model based on the periodic information/phase of a date concerning the period
    # to the original time series to test the hypothesis how well the suggested period fits the original time series
    X = data["date_modulo"].to_numpy().reshape(-1, 1)
    y = data["value"].to_numpy().reshape(-1)

    model = RandomForestRegressor()

    model.fit(X, y)
    y_preds = model.predict(X).reshape(X.size, 1)

    return y_preds, model


def get_local_minima(diffs: List[float]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    This function detects the local minima in the differences diffs between the original autocorrelation function and its shifted versions.
    It returns a list of the peaks, their indices and a Boolean indicating whether further calculation can be aboarted do to a lack of local minima.
    :param diffs: list of positive floats
    :return: list of positive floats, list of positive integers, Boolean
    """
    stop_calculation = False
    peaks, _ = find_peaks(-np.array(diffs))
    if peaks.size <= 0:
        print(
            "No period detection possible!\n"
            "No local minima found in the auto-L1-distance of the autocorrelation vector!"
        )
        stop_calculation = True
    # find_peaks() ignores peaks located at both boundaries. When the shift=0, however, two vectors are identical and thus their L1 distance is 0.
    # Therefore, this boundary value needs to be inserted manually.
    peaks = np.insert(peaks, 0, 0)
    peaks = peaks.astype(int)
    return np.array(diffs)[peaks], peaks, stop_calculation


def get_normalized_sum_l1_norm_on_offset(offset: int, data: List[float]) -> float:
    """
    This function calculates the sum of the L^1 norm of the correlation function and its shifted version (given as corfunc) for a given shift i.
    This is used to see if their sum is already smaller than our criterion (for further detail see paper Algorithm 1 step 5))
    :param i: positive integer
    :param corfunc: list of floats between -1 and  1
    :return: positive float
    """
    if offset == 0:
        return 2 * sum(abs(np.array(data))) / np.array(data).size
    else:
        w1 = np.array(data[offset:])
        w2 = np.array(data[:-offset])
        return sum(abs(w1)) / w1.size + sum(abs(w2)) / w2.size
