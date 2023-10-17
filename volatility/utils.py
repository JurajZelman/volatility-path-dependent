from datetime import datetime

import numpy as np
import pandas as pd

# TODO: Remove in future from default arguments
train_start_date = pd.to_datetime("2000-01-01")
test_start_date = pd.to_datetime("2019-01-01")
test_end_date = pd.to_datetime("2022-05-15")
dt = 1 / 252


def negative_part(x: np.array):
    """
    Returns the negative part of x, i.e. max(-x, 0).

    Args:
        x: A numpy array of values to compute the negative part of.

    Returns:
        The negative part of x.
    """
    return np.maximum(-x, 0)


def power_to(p: float):
    """
    Returns a function that computes the power of the input.

    Args:
        p: The power to compute.

    Returns:
        A function that computes the power of the input.
    """

    def f(x):
        if p in (-1, -2):
            return negative_part(x) ** np.abs(p)
        else:
            if p == 0:
                return x**p
            if int(1 / p) % 2 == 1:
                return np.abs(x) ** p * np.sign(x)
            else:
                return x**p

    return f


def compute_kernel_weighted_sum(
    x: np.array,
    params: np.array,
    func_power_law: callable,
    transform: callable = power_to(1),
    result_transform: callable = power_to(1),
):
    """
    TODO: Computes the weighted averages of the transform(x) with weights

    :param x: np.array of shape (n_elements, n_timestamps). Default: returns
        ordered from the most recent to the oldest
    :param params: array_like of parameters of func_power_law
    :param func_power_law: callable apply the kernel on the timestamps
    :param transform: callable, applied to the values of x. Default: identity
        (f(x)=x)
    :param result_transform: callable, applied to the computed average.
        Default: identity (f(x)=x)
    :return: feature as the weighted averages of the transform(x) with weights
        kernel(ti)
    """
    timestamps = np.arange(x.shape[1]) * dt

    weights = func_power_law(timestamps, *params)
    x = transform(x)
    return result_transform(np.sum(x * weights, axis=1))


def shifted_power_law(t, alpha, delta):
    return (t + delta) ** (-alpha)


def data_between_dates(
    data: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Select the data between two dates.

    Args:
        data: Data frame to be split.
        start_date: Initial date of the split.
        end_date: Final date of the split.

    Returns:
        A data frame with the data between the two dates.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data.loc[start_date:end_date]


def split_data(
    data: pd.DataFrame,
    train_start_date: datetime,
    test_start_date: datetime,
    test_end_date: datetime,
) -> pd.DataFrame:
    """
    Split the data into train and test sets.

    Args:
        data: Historical timeseries with index market prices.
        train_start_date: Start date of the train set.
        test_start_date: Start date of the test set.
        test_end_date: End date of the test set.

    Returns:
        A tuple with the train and test sets.
    """
    train_data = data_between_dates(
        data, train_start_date, test_start_date - pd.offsets.Day(1)
    )
    test_data = data_between_dates(data, test_start_date, test_end_date)
    return train_data, test_data


def dataframe_of_returns(
    index: pd.Series, vol: pd.Series, max_delta: int = 1000
) -> pd.DataFrame:
    """
    Construct a dataframe where each row contains the past max_delta one-day
    returns from the timestamp corresponding to the index of the dataframe.

    Args:
        index: Historical timeseries with index market prices.
        vol: Historical timeseries with index volatility.
        max_delta: Number of past returns to use.

    Returns:
        A dataframe with the past returns.
    """
    df = pd.DataFrame.from_dict({"index": index, "vol": vol})
    df.dropna(subset=["index"], inplace=True)  # remove closed days
    df["return_1d"] = np.nan

    df.iloc[1:, df.columns.get_loc("return_1d")] = (
        np.diff(df["index"]) / df["index"].iloc[1:].values
    )

    lags = np.arange(0, max_delta)
    df = df.merge(
        pd.DataFrame({f"r_(t-{lag})": df.return_1d.shift(lag) for lag in lags}),
        left_index=True,
        right_index=True,
    )
    return df
