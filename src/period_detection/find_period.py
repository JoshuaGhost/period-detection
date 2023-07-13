import traceback
from collections import namedtuple

import numpy as np
import pandas as pd
from pydantic import PositiveFloat, PositiveInt
from traitlets import Tuple

from .auxiliary_funcs import (
    autocor,
    fit_model,
    get_local_minima,
    auto_l1_norm_on_offset,
    get_normalized_sum_l1_norm_on_offset,
)
from .plot_funcs import plot_with_period, plot_without_period

DEFAULT_CRITERION_CLOSE_FIT = 1.5


def get_first_lag_len(df_data: pd.DataFrame) -> pd.Timedelta:
    # Test the datapoints for equidistance
    lag_len = 0
    delta_t = [y - x for x, y in zip(df_data[:-1]["date"], df_data[1:]["date"])]
    min_delta_t = min(delta_t)
    max_delta_t = max(delta_t)
    if max_delta_t == min_delta_t:
        lag_len = delta_t[0].total_seconds() / 60
        print(
            "Time-equidistant datapoints with a lag size of "
            + str(lag_len)
            + " minutes."
        )
    else:
        print("The datapoints are not time-equidistant!")

    if min_delta_t.total_seconds() <= 0:
        print("Warning: At least two records with the same timestamp!")
    return lag_len


def get_data_modulo(
    df_data: pd.DataFrame, relv_poses: np.array, reference_time: pd.Timestamp
) -> np.array:
    # Get the time difference between the shifts (Step 5 c) in Algorithm 1 in the paper)...
    relv_time_diff = (
        (df_data["date"].iloc[relv_poses] - df_data["date"].iloc[0])
        / pd.Timedelta("1 minutes")
    ).array()
    list_of_periods = np.diff(relv_time_diff)
    # ...and calculate their median as suggested period (Step 5 d) in Algorithm 1 in the paper)
    priod_in_min = np.median(list_of_periods)
    modulo = (
        (df_data["date"] - reference_time) / pd.Timedelta("1 minutes") % priod_in_min
    )
    return modulo.copy(), priod_in_min


def calculate_pred_error(
    value_preds: np.ndarray,
    df_data: pd.DataFrame,
    min_datapoint_num_for_corr: int,
    relv_poses: np.ndarray,
    corr: np.ndarray,
    plotting: bool,
    significant_corr_only: bool,
    min_significance_demand: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_data = df_data["value"].array().reshape(-1, 1)
    pred_errors = signal_data - value_preds
    df_errors = pd.DataFrame(data=pred_errors, columns=["value"])
    # If the output_flag=0 (no graphical output), then calculation time is saved by calculating the autocorrelation function only at the relevant shifts
    if plotting:
        (
            _,
            _,
            corfunc_diff,
            _,
        ) = autocor(
            df_errors["value"],
            range(df_errors["value"].size - min_datapoint_num_for_corr),
            min_significance_demand,
            significant_corr_only,
        )
        corr_at_pred_errors_minima = corfunc_diff[relv_poses]
    else:
        (
            _,
            _,
            cor_func_diff,
            _,
        ) = autocor(
            df_errors["value"],
            relv_poses,
            min_significance_demand,
            significant_corr_only,
        )
        corr_at_pred_errors_minima = cor_func_diff
    corr_reduction = (
        1 - abs(corr_at_pred_errors_minima[1:]).mean() / abs(corr[1:]).mean()
    )

    return signal_data, pred_errors, corr_reduction


def find_period(
    path,
    tol_norm_diff: float = 10 ** (-3),
    number_steps: PositiveInt = 1000,
    minimum_number_of_relevant_shifts: PositiveInt = 2,
    min_datapoint_num_for_corr: PositiveInt = 300,
    max_time_diff_for_l1_dist_pct: PositiveFloat = 0.3,
    significant_corr_only: bool = True,
    min_significance_demand: PositiveFloat = 0.01,
    plotting: bool = True,
    plot_tolerances: bool = True,
    reference_time: pd.Timestamp = pd.Timestamp("2017-01-01T12"),
):
    """
    This is the main period detection function.
    It reads your timeseries from a file given as path and calculates the difference between the original autocorrelation function and several possible shifts in order to find minima which indicate possible periods.
    Then a model is fitted for every suggested period. Afterwards each models performance is evaluated by taking the original time series and subtracting the model.
    If the model fits the time series well, the leftover should be noise and the autocorrelation function should deteriorate.
    It requires the data path,
    the tolerance for the norm difference between the unshifted and shifted autocorrelation function for a shift tol_norm_diff,
    the number of times iteratively we increase the tolerance number_steps,
    the minimum number of shifts required for calculation minimum_number_of_relevant_shifts,
    the minimum number of datapoints required for calculation minimum_number_of_datapoints_for_correlation_test,
    the minimum ratio of datapoints for which we calculate the autocorrelation of a shift minimum_ratio_of_datapoints_for_shift_autocorrelation,
    the flag declaring the usage only of correlations matching our criterion consider_only_significant_correlation,
    the minimum significance level for our correlation criterion level_of_significance_for_pearson,
    the output flag setting plotting to on/off plotting,
    the output flag allowing tolerances to be plotted plot_tolerances,
    a reference time for shift/phase calculation and relevant when fitting the model reference_time.
    The returns are the resulting period res_period, the fitted model res_model if a period was found and a performance criterion res_criteria

    :param path: string
    :param reference_time: pd.Timestamp
    :param tol_norm_diff: positive float
    :param number_steps: positive integer
    :param min_datapoint_num_for_corr: positive integer
    :param minimum_ratio_of_datapoints_for_shift_autocorrelation: positive float
    :param significant_corr_only: Boolean
    :param min_significance_demand: positive float
    :param plotting: Boolean
    :param plot_tolerances: Boolean
    :return: positive float, RandomForestRegressor (optional), positive float
    """

    # Load data
    df_data = pd.read_csv(path, parse_dates=["date"])

    Results = namedtuple("Results", "period model criteria")

    # Calculate the autocorrelation function and receive the correlation values r_list, the level of significance list p_list (Step 2 in Algorithm 1 in the paper)
    pearson_corrs, p_values, signif_pearson_corrs, lag_list = autocor(
        df_data["value"],
        list_of_lags=range(df_data["value"].size - min_datapoint_num_for_corr),
        min_significance_demand=min_significance_demand,
        significant_corr_only=significant_corr_only,
    )

    lag_len = get_first_lag_len(df_data)

    # Calculate the difference between the unshifted and shifted autocorrelation function for each shift
    # and determine which ones are relevant based on their local minima (Step 3 & 4 in Algorithm 1 in the paper)
    len_corrs = len(signif_pearson_corrs)
    max_time_diff_for_l1_dist = len_corrs - len_corrs * max_time_diff_for_l1_dist_pct
    diffs = [
        auto_l1_norm_on_offset(i, signif_pearson_corrs)
        for i in range(max_time_diff_for_l1_dist)
    ]
    minimal_diffs, local_minima, stop_calculation = get_local_minima(diffs)

    relv_poses = []
    size_relv_poses = 0

    suggested_periods = []
    criteria = []
    norms_signal_preds = []
    nomr_preds_errors = []
    preds_data = []
    list_tolerances = []
    models = []
    # stop calculation if no local minima (except the one at shift 0) are found
    if stop_calculation:
        res_period = -1
        res_criteria = 0
        if plotting:
            plot_without_period(
                df_data,
                diffs,
                lag_list,
                pearson_corrs,
                p_values,
                signif_pearson_corrs,
            )
        return Results(res_period, None, res_criteria)

    sum_of_shifted_correlation_function = [
        get_normalized_sum_l1_norm_on_offset(i, signif_pearson_corrs)
        for i in local_minima
    ]
    df_diffs_lag = pd.DataFrame(
        {
            "lags": local_minima,
            "diffs": minimal_diffs,
            "sum_of_norms": sum_of_shifted_correlation_function,
        }
    )

    # Step by step extend the set of considered shifts (Step 5 in Algorithm 1 in the paper)
    for tol_for_zero in np.linspace(0, 1, number_steps + 1):
        # Filter for shifts smaller or equal to our criterion tol_for_zero (Step 5 a) in the paper)
        vec_bool = (df_diffs_lag["diffs"] <= tol_for_zero) & (
            df_diffs_lag["sum_of_norms"] > tol_for_zero
        )
        relv_poses = df_diffs_lag["lags"][vec_bool].array()

        if len(relv_poses) < max(
            minimum_number_of_relevant_shifts, size_relv_poses + 1
        ):
            continue

        size_relv_poses = len(relv_poses)
        corr_at_current_minima = signif_pearson_corrs[relv_poses]
        # If we have no (further) relevant shifts, we can abort (Step 5 b) in Algorithm 1 in the paper)
        if np.any(corr_at_current_minima <= 0):
            print(
                "Relevant lag in autocorrelation function with non-positive correlation!"
            )
            break

        list_tolerances.append(tol_for_zero)

        # Fit a model based on the data and the phase inside the period, here calculated using modulo (Step 5 c, d, and e) in Algorithm 1 in the paper)
        df_data["date_modulo"], suggested_period = get_data_modulo(df_data, relv_poses)

        value_preds, model = fit_model(df_data)

        # Subtract the model data from the original and determine the autocorrelation function as a performance measure (Step 5 f & g) in Algorithm 1 in the paper)
        signal_data, pred_errors, corr_reduction = calculate_pred_error(
            value_preds=value_preds,
            df_data=df_data,
            corr=corr_at_current_minima,
            relv_poses=relv_poses,
            plotting=plotting,
            min_datapoint_num_for_corr=min_datapoint_num_for_corr,
            significant_corr_only=significant_corr_only,
            min_significance_demand=min_significance_demand,
        )

        norm_signal = sum(abs(signal_data))[0] / signal_data.size
        norm_preds = sum(abs(value_preds))[0] / value_preds.size

        norm_preds_error = sum(abs(pred_errors))[0] / pred_errors.size

        suggested_periods.append(suggested_period)
        criteria.append(corr_reduction)
        preds_data.append(value_preds)
        models.append(model)
        norms_signal_preds.append(norm_signal + norm_preds)
        nomr_preds_errors.append(norm_preds_error)

    if not len(suggested_periods):
        print("List of suggested periods is empty!")
        res_period = -1
        res_criteria = 0
        if plotting == 1:
            plot_without_period(
                df_data,
                diffs,
                lag_list,
                pearson_corrs,
                p_values,
                signif_pearson_corrs,
            )
        return Results(res_period, None, res_criteria)

    df_periods_criterion = pd.DataFrame(
        {
            "periods": suggested_periods,
            "criterion": criteria,
            "norm_diff": nomr_preds_errors,
            "sum_norms": norms_signal_preds,
            "model_data": preds_data,
            "models": models,
            "tolerances": list_tolerances,
        }
    )

    # Test if the fittet model is close to the data (almost perfect fit)
    res_period, res_criteria, res_model = test_results(
        tol_norm_diff,
        min_datapoint_num_for_corr,
        significant_corr_only,
        min_significance_demand,
        plotting,
        plot_tolerances,
        df_data,
        pearson_corrs,
        p_values,
        signif_pearson_corrs,
        lag_list,
        lag_len,
        diffs,
        norm_preds_error,
        df_periods_criterion,
    )

    return Results(res_period, res_model, res_criteria)


def test_results(
    tol_norm_diff,
    min_datapoint_num_for_corr,
    significant_corr_only,
    min_significance_demand,
    plotting,
    plot_tolerances,
    df_data,
    pearson_corrs,
    p_values,
    signif_pearson_corrs,
    lag_list,
    lag_len,
    diffs,
    norm_preds_error,
    df_periods_criterion,
):
    close_fit_mask = (df_periods_criterion["norm_diff"] <= tol_norm_diff) & (
        df_periods_criterion["sum_norms"] > tol_norm_diff
    )
    if close_fit_mask.any():
        period_very_close_fit = df_periods_criterion["periods"][close_fit_mask]
        model_very_close_fit = df_periods_criterion["models"][close_fit_mask]
        model_data_very_close_fit = df_periods_criterion["model_data"][close_fit_mask]
        best_tolerances_close_fit = df_periods_criterion["tolerances"][close_fit_mask]

        res_period = period_very_close_fit[0]
        res_model = model_very_close_fit[0]
        value_preds = model_data_very_close_fit[0]
        best_tolerance = best_tolerances_close_fit[0]
        res_criteria = DEFAULT_CRITERION_CLOSE_FIT

        other_tolerances = df_periods_criterion["tolerances"][
            df_periods_criterion["tolerances"] != best_tolerance
        ].to_list()
        print(
            f"Very small difference between data and model, difference smaller than {tol_norm_diff}"
        )
    else:
        # If the fit is not close but there are suggested periods, then find the period with the best criterion of correlation reduction (Step 6 in Algorithm 1 in the paper)
        index_min_criterion = df_periods_criterion["criterion"].idxmax()
        res_criteria = df_periods_criterion["criterion"].iloc[index_min_criterion]

        res_period = df_periods_criterion["periods"].iloc[index_min_criterion]
        res_model = df_periods_criterion["models"][index_min_criterion]
        value_preds = df_periods_criterion["model_data"][index_min_criterion]
        best_tolerance = df_periods_criterion["tolerances"][index_min_criterion]
        other_tolerances = df_periods_criterion["tolerances"][
            df_periods_criterion.index != index_min_criterion
        ].to_list()
        print(
            f"Reduction of correlation by model: {res_criteria} with sigma (tolerance) = {best_tolerance}"
        )

    print(
        f"The suggested period is {res_period} in minutes, {res_period / 60} in hours, {res_period / 60 / 24} in days.\n"
    )

    if lag_len:
        print(f"The lags is {res_period / lag_len}")

    if plotting:
        plot_with_period(
            df_data,
            diffs,
            other_tolerances,
            best_tolerance,
            lag_list,
            pearson_corrs,
            p_values,
            signif_pearson_corrs,
            value_preds,
            norm_preds_error,
            plot_tolerances,
            min_significance_demand,
            significant_corr_only,
            min_datapoint_num_for_corr,
        )

    return res_period, res_criteria, res_model
