import numpy as np
import pandas as pd

# TODO: Remove in future from default arguments
train_start_date = pd.to_datetime("2000-01-01")
test_start_date = pd.to_datetime("2019-01-01")
test_end_date = pd.to_datetime("2022-05-15")
dt = 1 / 252


def negative_part(x):
    return np.clip(-x, 0, None)


def power_to(p):
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


squared = power_to(2)
sqrt = power_to(0.5)
identity = power_to(1)


def compute_kernel_weighted_sum(
    x, params, func_power_law, transform=identity, result_transform=identity
):
    """

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


def data_between_dates(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data.loc[start_date:end_date]


def split_data(
    data,
    train_start_date=train_start_date,
    test_start_date=test_start_date,
    test_end_date=test_end_date,
):
    train_data = data_between_dates(data, train_start_date, test_start_date)
    test_data = data_between_dates(data, test_start_date, test_end_date)
    return train_data, test_data


def dataframe_of_returns(
    index: pd.Series, vol: pd.Series, max_delta: int = 1000
) -> pd.DataFrame:
    """
    constructs a dataframe where each row contains the past max_delta one-day
        returns from the timestamp corresponding to the index of the dataframe.
    :param index: pd.Series of historical market prices of index
    :param vol: pd.Series of historical market prices of volatility index or
        realized vol
    :param max_delta: int number of past returns to use
    :param data: pd.DataFrame
    :return:pd.DataFrame
    """
    df = pd.DataFrame.from_dict({"index": index, "vol": vol})
    df.dropna(subset=["index"], inplace=True)  # remove closed days

    # df.loc[1:, "return_1d"] = np.diff(df["index"]) / df["index"].iloc[1:]
    # Add new empty column
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
